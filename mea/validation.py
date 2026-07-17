"""Cached small-sample scorers for Planner and Execution VQA artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from mea.execution_vqa import (
    ExecutionVQAError,
    ExecutionVQAQueryError,
    validate_execution_vqa_query,
    validate_execution_vqa_response,
)
from mea.protocol import validate_budget


class ValidationError(RuntimeError):
    """Raised when a validation suite or artifact is invalid."""


LABEL_SOURCES = {"human", "simulator_proxy", "synthetic_fixture"}


def _artifact_path(repo_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValidationError("artifact path must be a non-empty string")
    path = (repo_root / value).resolve()
    if not path.is_relative_to(repo_root.resolve()):
        raise ValidationError("artifact path escapes repo_root")
    if not path.is_file():
        raise ValidationError(f"artifact does not exist: {path}")
    return path


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"artifact must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field} must be an object")
    return value


def _unique_strings(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{field} must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValidationError(f"{field} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValidationError(f"{field} must not contain duplicates")
    return list(value)


def _planner_eligibility(kind: str, explicit: bool | None) -> tuple[bool, str | None]:
    if kind.startswith("deterministic_"):
        return False, "deterministic_planner"
    if explicit is False:
        return False, "suite_marks_non_model"
    if kind == "unknown" and explicit is not True:
        return False, "unknown_planner_kind"
    return True, None


def score_planner_case(repo_root: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(case["id"])
    gold = _mapping(case["gold"], field=f"planner case {case_id}.gold")
    prediction = _mapping(
        case["prediction"], field=f"planner case {case_id}.prediction"
    )
    expected = set(
        _unique_strings(
            gold["requested_template_ids"],
            field=f"planner case {case_id}.gold.requested_template_ids",
        )
    )
    acceptable_value = (
        gold["requested_template_ids"]
        if "acceptable_first_template_ids" not in gold
        else gold["acceptable_first_template_ids"]
    )
    acceptable_first = set(
        _unique_strings(
            acceptable_value,
            field=f"planner case {case_id}.gold.acceptable_first_template_ids",
        )
    )
    if not acceptable_first.issubset(expected):
        raise ValidationError(
            f"planner case {case_id}.gold.acceptable_first_template_ids "
            "must be a subset of requested_template_ids"
        )
    path = _artifact_path(repo_root, prediction.get("path"))
    artifact = _read_object(path)
    plan = artifact.get("plan") if isinstance(artifact.get("plan"), dict) else artifact
    planner = artifact.get("planner") or plan.get("planner") or {}
    if not isinstance(planner, Mapping):
        planner = {}
    kind = str(planner.get("kind") or prediction.get("planner_kind") or "unknown")
    explicit = prediction.get("model_generated")
    eligible, exclusion_reason = _planner_eligibility(kind, explicit)
    try:
        predicted = set(
            _unique_strings(
                plan.get("requested_template_ids"),
                field="prediction.requested_template_ids",
            )
        )
        rounds = plan.get("rounds")
        if not isinstance(rounds, list) or not rounds or not isinstance(rounds[0], dict):
            raise ValidationError("prediction plan has no first round")
        first = rounds[0].get("template_id")
        if not isinstance(first, str) or not first:
            raise ValidationError("prediction first template is invalid")
        if first not in predicted:
            raise ValidationError(
                "prediction first template must be in requested_template_ids"
            )
        schema_valid = True
        error = None
    except ValidationError as exc:
        predicted = set()
        first = None
        schema_valid = False
        error = str(exc)
    true_positive = len(expected & predicted)
    return {
        "id": case_id,
        "target": "planner",
        "artifact": str(path.relative_to(repo_root)),
        "artifact_sha256": _sha256(path),
        "planner_kind": kind,
        "eligible_for_model_metric": eligible,
        "exclusion_reason": exclusion_reason,
        "schema_valid": schema_valid,
        "error": error,
        "gold_templates": sorted(expected),
        "predicted_templates": sorted(predicted),
        "true_positive": true_positive,
        "false_positive": len(predicted - expected),
        "false_negative": len(expected - predicted),
        "exact_set_match": schema_valid and predicted == expected,
        "first_template_match": schema_valid and first in acceptable_first,
    }


def _failed_planner_case(case: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    gold = case["gold"]
    expected = set(gold["requested_template_ids"])
    prediction = case["prediction"]
    kind = str(prediction.get("planner_kind") or "unknown")
    eligible, exclusion_reason = _planner_eligibility(
        kind, prediction.get("model_generated")
    )
    return {
        "id": case["id"],
        "target": "planner",
        "artifact": prediction.get("path"),
        "artifact_sha256": None,
        "planner_kind": kind,
        "eligible_for_model_metric": eligible,
        "exclusion_reason": exclusion_reason,
        "schema_valid": False,
        "error": f"{type(error).__name__}: {error}",
        "gold_templates": sorted(expected),
        "predicted_templates": [],
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": len(expected),
        "exact_set_match": False,
        "first_template_match": False,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_planner_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [case for case in cases if case["eligible_for_model_metric"]]
    tp = sum(case["true_positive"] for case in eligible)
    fp = sum(case["false_positive"] for case in eligible)
    fn = sum(case["false_negative"] for case in eligible)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None:
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    return {
        "selected_case_count": len(cases),
        "case_count": len(eligible),
        "excluded_non_model_count": len(cases) - len(eligible),
        "artifact_failure_count": sum(bool(case["error"]) for case in cases),
        "schema_valid_rate": _ratio(
            sum(case["schema_valid"] for case in eligible), len(eligible)
        ),
        "template_micro_precision": precision,
        "template_micro_recall": recall,
        "template_micro_f1": f1,
        "template_exact_set_accuracy": _ratio(
            sum(case["exact_set_match"] for case in eligible), len(eligible)
        ),
        "first_template_accuracy": _ratio(
            sum(case["first_template_match"] for case in eligible), len(eligible)
        ),
    }


def score_vqa_case(repo_root: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(case["id"])
    phenomenon_id = str(case["phenomenon_id"])
    gold = _mapping(case["gold"], field=f"VQA case {case_id}.gold")
    prediction = _mapping(case["prediction"], field=f"VQA case {case_id}.prediction")
    path = _artifact_path(repo_root, prediction.get("path"))
    artifact = _read_object(path)
    if artifact.get("schema_version") != 1:
        raise ValidationError("Execution VQA artifact schema_version must be 1")
    try:
        query = validate_execution_vqa_query(artifact.get("query"))
    except ExecutionVQAQueryError as exc:
        raise ValidationError(f"invalid Execution VQA query: {exc}") from exc
    if phenomenon_id not in query["phenomenon_ids"]:
        raise ValidationError(
            f"suite phenomenon {phenomenon_id} is absent from artifact query"
        )
    selection = _mapping(artifact.get("selection"), field="artifact.selection")
    selected_frames = selection.get("selected_frames")
    if not isinstance(selected_frames, list) or not selected_frames:
        raise ValidationError("artifact.selection.selected_frames must be non-empty")
    allowed_frame_ids = [
        str(_mapping(item, field="selected frame").get("frame_id") or "")
        for item in selected_frames
    ]
    if any(not frame_id for frame_id in allowed_frame_ids):
        raise ValidationError("selected frame has an empty frame_id")
    observation = _mapping(artifact.get("observation"), field="artifact.observation")
    response_keys = {
        "phenomena",
        "confidence",
        "frame_ids",
        "numeric_consistency",
        "conflicts",
    }
    if set(observation) not in (
        response_keys,
        response_keys | {"evidence_conflict"},
    ):
        raise ValidationError("artifact.observation has invalid fields")
    raw_response = {key: observation.get(key) for key in response_keys}
    try:
        normalized = validate_execution_vqa_response(
            raw_response,
            allowed_frame_ids=allowed_frame_ids,
            expected_phenomenon_ids=query["phenomenon_ids"],
        )
    except ExecutionVQAError as exc:
        raise ValidationError(f"invalid Execution VQA response: {exc}") from exc
    phenomenon = next(
        item for item in normalized["phenomena"] if item["id"] == phenomenon_id
    )
    observed = phenomenon["observed"]
    confidence = float(phenomenon["confidence"])
    positive_score = (
        confidence
        if observed is True
        else 1.0 - confidence
        if observed is False
        else 0.5
    )
    return {
        "id": case_id,
        "target": "vqa",
        "artifact": str(path.relative_to(repo_root)),
        "artifact_sha256": _sha256(path),
        "phenomenon_id": phenomenon_id,
        "gold_observed": gold["observed"],
        "label_source": gold["label_source"],
        "schema_valid": True,
        "error": None,
        "predicted_observed": observed,
        "confidence": confidence,
        "positive_score": positive_score,
        "covered": isinstance(observed, bool),
        "correct_strict": observed == gold["observed"],
    }


def _failed_vqa_case(case: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "id": case["id"],
        "target": "vqa",
        "artifact": case["prediction"].get("path"),
        "artifact_sha256": None,
        "phenomenon_id": case["phenomenon_id"],
        "gold_observed": case["gold"]["observed"],
        "label_source": case["gold"]["label_source"],
        "schema_valid": False,
        "error": f"{type(error).__name__}: {error}",
        "predicted_observed": None,
        "confidence": 0.0,
        "positive_score": 0.5,
        "covered": False,
        "correct_strict": False,
    }


def binary_auroc(labels: list[bool], scores: list[float]) -> dict[str, Any]:
    if len(labels) != len(scores):
        raise ValidationError("AUROC labels and scores must have equal length")
    if any(not isinstance(label, bool) for label in labels):
        raise ValidationError("AUROC labels must be boolean")
    normalized_scores: list[float] = []
    for score in scores:
        if isinstance(score, bool):
            raise ValidationError("AUROC scores must be finite numbers")
        try:
            normalized = float(score)
        except (TypeError, ValueError) as exc:
            raise ValidationError("AUROC scores must be finite numbers") from exc
        if not math.isfinite(normalized):
            raise ValidationError("AUROC scores must be finite numbers")
        normalized_scores.append(normalized)
    positives = [score for label, score in zip(labels, normalized_scores) if label]
    negatives = [score for label, score in zip(labels, normalized_scores) if not label]
    if not labels:
        return {"value": None, "unavailable_reason": "no_cases"}
    if not positives or not negatives:
        return {"value": None, "unavailable_reason": "single_class"}
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return {
        "value": wins / (len(positives) * len(negatives)),
        "unavailable_reason": None,
    }


def _aggregate_vqa_core(cases: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [case["gold_observed"] for case in cases]
    scores = [case["positive_score"] for case in cases]
    true_positive = sum(
        case["gold_observed"] is True and case["predicted_observed"] is True
        for case in cases
    )
    predicted_positive = sum(case["predicted_observed"] is True for case in cases)
    precision = _ratio(true_positive, predicted_positive)
    return {
        "case_count": len(cases),
        "artifact_failure_count": sum(bool(case["error"]) for case in cases),
        "schema_valid_rate": _ratio(
            sum(case["schema_valid"] for case in cases), len(cases)
        ),
        "coverage": _ratio(sum(case["covered"] for case in cases), len(cases)),
        "accuracy_strict": _ratio(
            sum(case["correct_strict"] for case in cases), len(cases)
        ),
        "precision": {
            "value": precision,
            "unavailable_reason": None if precision is not None else "no_positive_prediction",
        },
        "auroc": binary_auroc(labels, scores),
    }


def aggregate_vqa_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result = _aggregate_vqa_core(cases)
    counts = Counter(case["label_source"] for case in cases)
    result["label_source_counts"] = dict(sorted(counts.items()))
    result["proxy_only"] = bool(cases) and set(counts) == {"simulator_proxy"}
    result["by_label_source"] = {
        source: _aggregate_vqa_core(
            [case for case in cases if case["label_source"] == source]
        )
        for source in sorted(counts)
    }
    return result


def validate_suite(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValidationError("validation suite schema_version must be 1")
    if not isinstance(value.get("suite_id"), str) or not value["suite_id"]:
        raise ValidationError("validation suite requires suite_id")
    seen_ids: set[str] = set()
    for field in ("planner_cases", "vqa_cases"):
        cases = value.get(field)
        if not isinstance(cases, list):
            raise ValidationError(f"{field} must be present and be a list")
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                raise ValidationError(f"{field}[{index}] must be an object")
            case_id = case.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ValidationError(f"{field}[{index}].id must be non-empty")
            if case_id in seen_ids:
                raise ValidationError(f"duplicate validation case id: {case_id}")
            seen_ids.add(case_id)
            gold = _mapping(case.get("gold"), field=f"{case_id}.gold")
            prediction = _mapping(case.get("prediction"), field=f"{case_id}.prediction")
            if not isinstance(prediction.get("path"), str) or not prediction["path"]:
                raise ValidationError(f"{case_id}.prediction.path must be non-empty")
            explicit = prediction.get("model_generated")
            if explicit is not None and not isinstance(explicit, bool):
                raise ValidationError(f"{case_id}.prediction.model_generated must be boolean")
            if field == "planner_cases":
                requested = _unique_strings(
                    gold.get("requested_template_ids"),
                    field=f"{case_id}.gold.requested_template_ids",
                )
                acceptable_value = (
                    requested
                    if "acceptable_first_template_ids" not in gold
                    else gold["acceptable_first_template_ids"]
                )
                acceptable = _unique_strings(
                    acceptable_value,
                    field=f"{case_id}.gold.acceptable_first_template_ids",
                )
                if not set(acceptable).issubset(requested):
                    raise ValidationError(
                        f"{case_id}.gold.acceptable_first_template_ids must be a "
                        "subset of requested_template_ids"
                    )
            else:
                phenomenon_id = case.get("phenomenon_id")
                if not isinstance(phenomenon_id, str) or not phenomenon_id:
                    raise ValidationError(f"{case_id}.phenomenon_id must be non-empty")
                if not isinstance(gold.get("observed"), bool):
                    raise ValidationError(f"{case_id}.gold.observed must be boolean")
                if gold.get("label_source") not in LABEL_SOURCES:
                    raise ValidationError(
                        f"{case_id}.gold.label_source must be one of {sorted(LABEL_SOURCES)}"
                    )
    return dict(value)


def score_cached_suite(
    repo_root: str | Path,
    suite: Mapping[str, Any],
    *,
    budget: int,
    target: str = "both",
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    normalized = validate_suite(suite)
    limit = validate_budget(budget, name="budget")
    if target not in {"planner", "vqa", "both"}:
        raise ValidationError("target must be planner, vqa, or both")
    for name in ("planner", "vqa"):
        if target in {name, "both"} and len(normalized[f"{name}_cases"]) < limit:
            raise ValidationError(
                f"{name}_cases has fewer than the requested budget={limit}"
            )
    planner_cases: list[dict[str, Any]] = []
    if target in {"planner", "both"}:
        for case in normalized["planner_cases"][:limit]:
            try:
                planner_cases.append(score_planner_case(root, case))
            except (OSError, ValidationError) as exc:
                planner_cases.append(_failed_planner_case(case, exc))
    vqa_cases: list[dict[str, Any]] = []
    if target in {"vqa", "both"}:
        for case in normalized["vqa_cases"][:limit]:
            try:
                vqa_cases.append(score_vqa_case(root, case))
            except (OSError, ValidationError) as exc:
                vqa_cases.append(_failed_vqa_case(case, exc))
    return {
        "schema_version": 1,
        "suite_id": normalized["suite_id"],
        "mode": "cached",
        "budget": limit,
        "target": target,
        "provider_called": False,
        "planner": {
            "metrics": aggregate_planner_cases(planner_cases),
            "cases": planner_cases,
        },
        "vqa": {
            "metrics": aggregate_vqa_cases(vqa_cases),
            "cases": vqa_cases,
        },
        "limitations": [
            "Budget 1/3/5 validates the scoring path; it is not the paper dataset.",
            "Cached mode never calls a model or reruns simulation.",
            "Human and simulator-proxy labels are reported separately.",
        ],
    }
