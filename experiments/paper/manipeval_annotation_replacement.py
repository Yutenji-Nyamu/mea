#!/usr/bin/env python3
"""Score frozen Plan/VQA predictions with one replaceable annotation file."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.independent_validity import majority_vote
from mea.query_planner_validation import aggregate_sub_aspect_predictions
from mea.validation import aggregate_vqa_cases


PACKET_PROTOCOL = "human_replaceable_annotation_packet_v1"
ANNOTATION_PROTOCOL = "human_replaceable_annotations_v1"
PREDICTION_PROTOCOL = "frozen_annotation_predictions_v1"
RESULT_PROTOCOL = "human_replaceable_annotation_score_v1"
PROXY_SOURCE = "codex_development_agent_proxy"
HUMAN_SOURCE = "independent_human_annotation"
LABEL_KINDS = {"plan_sub_aspect", "vqa_binary"}


class AnnotationReplacementError(ValueError):
    pass


def _items(value: Any, field: str) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(value, list) or not value:
        raise AnnotationReplacementError(f"{field} must be a non-empty list")
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise AnnotationReplacementError(f"{field}[{index}] must be an object")
        item_id = raw.get("id")
        if not isinstance(item_id, str) or not item_id or item_id in ids:
            raise AnnotationReplacementError(f"{field}[{index}] has invalid id")
        ids.append(item_id)
        rows.append(dict(raw))
    return rows, ids


def _header(value: Any, protocol: str, field: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != 1
        or value.get("protocol") != protocol
    ):
        raise AnnotationReplacementError(
            f"{field} must use schema_version=1 and protocol={protocol}"
        )
    return value


def validate_annotation_packet(value: Any) -> dict[str, Any]:
    value = _header(value, PACKET_PROTOCOL, "packet")
    packet_id = value.get("packet_id")
    label_kind = value.get("label_kind")
    if not isinstance(packet_id, str) or not packet_id:
        raise AnnotationReplacementError("packet_id must be a non-empty string")
    if label_kind not in LABEL_KINDS:
        raise AnnotationReplacementError("unsupported label_kind")
    if value.get("blinding") != {
        "prediction_values_hidden": True,
        "reference_labels_hidden": True,
    }:
        raise AnnotationReplacementError("packet blinding contract changed")
    if value.get("aggregation") != {
        "primary_human_rater_count": 4,
        "majority_rule": "strict_majority_3_of_4",
        "tie_break": "senior_robotics_annotator",
    }:
        raise AnnotationReplacementError("packet aggregation contract changed")
    slots = value.get("human_rater_slots")
    if not isinstance(slots, list) or len(slots) != 4:
        raise AnnotationReplacementError("packet requires four human rater slots")
    if any(
        not isinstance(slot, Mapping)
        or slot.get("kind") != "human"
        or slot.get("role") != "primary_annotator"
        or slot.get("labels") is not None
        for slot in slots
    ):
        raise AnnotationReplacementError(
            "blind packet human slots must remain empty"
        )
    senior = value.get("senior_tiebreaker_slot")
    if (
        not isinstance(senior, Mapping)
        or senior.get("kind") != "human"
        or senior.get("role") != "senior_tiebreaker"
        or senior.get("labels") is not None
    ):
        raise AnnotationReplacementError(
            "blind packet requires one empty senior tiebreaker slot"
        )
    items, _ = _items(value.get("items"), "packet.items")
    result = dict(value)
    result["items"] = items
    return result


def _label_value(
    value: Any,
    label_kind: str,
    field: str,
    *,
    allow_empty_plan: bool = False,
) -> Any:
    if label_kind == "vqa_binary":
        if not isinstance(value, bool):
            raise AnnotationReplacementError(f"{field} must be boolean")
        return value
    if (
        not isinstance(value, list)
        or (not value and not allow_empty_plan)
        or len(value) != len(set(value))
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise AnnotationReplacementError(
            f"{field} must be a unique string list"
        )
    return sorted(value)


def _rater_labels(
    rater: Mapping[str, Any],
    label_kind: str,
    expected_ids: list[str],
    field: str,
) -> dict[str, Any]:
    labels, ids = _items(rater.get("labels"), f"{field}.labels")
    if ids != expected_ids:
        raise AnnotationReplacementError(
            f"{field} label order/IDs differ from the frozen packet"
        )
    return {
        row["id"]: _label_value(
            row.get("value"), label_kind, f"{field}.{row['id']}"
        )
        for row in labels
    }


def _references(
    annotations: Any,
    packet_id: str,
    label_kind: str,
    ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _header(annotations, ANNOTATION_PROTOCOL, "annotations")
    if value.get("packet_id") != packet_id or value.get("label_kind") != label_kind:
        raise AnnotationReplacementError("annotations do not match packet")
    annotation_set_id = value.get("annotation_set_id")
    raters = value.get("raters")
    if not isinstance(annotation_set_id, str) or not annotation_set_id:
        raise AnnotationReplacementError("annotation_set_id is required")
    if (
        not isinstance(raters, list)
        or not raters
        or not all(isinstance(rater, Mapping) for rater in raters)
    ):
        raise AnnotationReplacementError("annotations.raters must contain objects")
    source = value.get("annotation_source")
    if source == PROXY_SOURCE:
        if (
            len(raters) != 1
            or raters[0].get("kind") != "development_agent_proxy"
            or raters[0].get("role") != "proxy"
        ):
            raise AnnotationReplacementError(
                "proxy annotations require one disclosed proxy rater"
            )
        return _rater_labels(raters[0], label_kind, ids, "proxy"), {
            "annotation_set_id": annotation_set_id,
            "annotation_source": source,
            "human_reviewer_count": 0,
            "reference_source": "codex_development_agent_proxy_not_human_gold",
            "unscorable_ids": [],
        }
    if source != HUMAN_SOURCE:
        raise AnnotationReplacementError("unsupported annotation_source")
    primary = [
        rater
        for rater in raters
        if rater.get("kind") == "human"
        and rater.get("role") == "primary_annotator"
    ]
    senior = [
        rater
        for rater in raters
        if rater.get("kind") == "human"
        and rater.get("role") == "senior_tiebreaker"
    ]
    if len(primary) != 4 or len(senior) > 1:
        raise AnnotationReplacementError(
            "human annotations require four primary raters and at most one senior"
        )
    primary_labels = [
        _rater_labels(rater, label_kind, ids, f"human[{index}]")
        for index, rater in enumerate(primary)
    ]
    senior_labels = (
        _rater_labels(senior[0], label_kind, ids, "senior") if senior else None
    )
    references: dict[str, Any] = {}
    unscorable: list[str] = []
    used_tiebreak = False
    for item_id in ids:
        if label_kind == "vqa_binary":
            reference = majority_vote(
                [labels[item_id] for labels in primary_labels]
            )
            if reference is None and senior_labels is not None:
                reference = senior_labels[item_id]
                used_tiebreak = True
        else:
            selected: list[str] = []
            tied: list[str] = []
            universe = {
                aspect
                for labels in primary_labels
                for aspect in labels[item_id]
            }
            for aspect in universe:
                vote = majority_vote(
                    [aspect in labels[item_id] for labels in primary_labels]
                )
                if vote is True:
                    selected.append(aspect)
                elif vote is None:
                    tied.append(aspect)
            if tied and senior_labels is None:
                reference = None
            else:
                if tied:
                    used_tiebreak = True
                    selected.extend(
                        aspect
                        for aspect in tied
                        if aspect in senior_labels[item_id]
                    )
                reference = sorted(selected)
        references[item_id] = reference
        if reference is None:
            unscorable.append(item_id)
    return references, {
        "annotation_set_id": annotation_set_id,
        "annotation_source": source,
        "human_reviewer_count": 4,
        "reference_source": (
            "four_human_majority_with_senior_tiebreak"
            if used_tiebreak
            else "four_human_strict_majority"
        ),
        "unscorable_ids": unscorable,
    }


def _predictions(
    value: Any, packet_id: str, label_kind: str, ids: list[str]
) -> tuple[dict[str, dict[str, Any]], str | None]:
    value = _header(value, PREDICTION_PROTOCOL, "predictions")
    if value.get("packet_id") != packet_id or value.get("label_kind") != label_kind:
        raise AnnotationReplacementError("predictions do not match packet")
    rows, prediction_ids = _items(value.get("items"), "predictions.items")
    if prediction_ids != ids:
        raise AnnotationReplacementError(
            "prediction item order/IDs differ from the frozen packet"
        )
    for row in rows:
        if label_kind == "plan_sub_aspect":
            row["predicted_sub_aspect_ids"] = _label_value(
                row.get("predicted_sub_aspect_ids"),
                label_kind,
                f"prediction.{row['id']}",
                allow_empty_plan=True,
            )
            if not isinstance(row.get("schema_valid"), bool):
                raise AnnotationReplacementError("schema_valid must be boolean")
        else:
            observed = row.get("predicted_observed")
            score = row.get("positive_score")
            if observed is not None and not isinstance(observed, bool):
                raise AnnotationReplacementError(
                    "predicted_observed must be boolean or null"
                )
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0 <= float(score) <= 1
            ):
                raise AnnotationReplacementError("positive_score must be in [0,1]")
            row["positive_score"] = float(score)
    return {row["id"]: row for row in rows}, value.get("prediction_source")


def _plan_metrics(
    packet: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for item in packet["items"]:
        item_id = item["id"]
        prediction = predictions[item_id]
        reference_value = references[item_id]
        if reference_value is None:
            rows.append(
                {
                    "id": item_id,
                    "paper_category": item.get("paper_category"),
                    "reference_sub_aspect_ids": None,
                    "predicted_sub_aspect_ids": prediction[
                        "predicted_sub_aspect_ids"
                    ],
                    "scorable": False,
                }
            )
            continue
        reference = set(reference_value)
        predicted = set(prediction["predicted_sub_aspect_ids"])
        rows.append(
            {
                "id": item_id,
                "paper_category": item.get("paper_category"),
                "reference_sub_aspect_ids": sorted(reference),
                "predicted_sub_aspect_ids": sorted(predicted),
                "schema_valid": prediction["schema_valid"],
                "scorable": True,
                "true_positive": len(reference & predicted),
                "false_positive": len(predicted - reference),
                "false_negative": len(reference - predicted),
                "exact_set_match": reference == predicted,
            }
        )
    scorable = [row for row in rows if row["scorable"]]
    aggregate = aggregate_sub_aspect_predictions(
        [
            {
                "paper_category": row["paper_category"],
                "true_positive": row["true_positive"],
                "false_positive": row["false_positive"],
                "false_negative": row["false_negative"],
                "aspect_exact_set_match": row["exact_set_match"],
            }
            for row in scorable
        ]
    )
    return {
        "case_count": len(rows),
        "scorable_case_count": len(scorable),
        "table6_sub_aspect_precision": aggregate["aspect_micro_precision"],
        "sub_aspect_micro_recall": aggregate["aspect_micro_recall"],
        "sub_aspect_micro_f1": aggregate["aspect_micro_f1"],
        "sub_aspect_exact_set_accuracy": aggregate[
            "aspect_exact_set_accuracy"
        ],
        "by_paper_category": aggregate["by_paper_category"],
    }, rows


def _vqa_metrics(
    packet: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for item in packet["items"]:
        item_id = item["id"]
        prediction = predictions[item_id]
        reference = references[item_id]
        rows.append(
            {
                "id": item_id,
                "condition": item["condition"],
                "reference_observed": reference,
                "predicted_observed": prediction["predicted_observed"],
                "positive_score": prediction["positive_score"],
                "covered": prediction["predicted_observed"] is not None,
                "scorable": reference is not None,
                "correct": (
                    None
                    if reference is None
                    else prediction["predicted_observed"] == reference
                ),
            }
        )
    scorable = [row for row in rows if row["scorable"]]

    def metrics(subset: list[dict[str, Any]]) -> dict[str, Any]:
        validation_rows = [
            {
                "gold_observed": row["reference_observed"],
                "predicted_observed": row["predicted_observed"],
                "positive_score": row["positive_score"],
                "error": None,
                "schema_valid": row["covered"],
                "covered": row["covered"],
                "correct_strict": row["correct"] is True,
                "label_source": "replaceable_annotation",
                "perturbation": row["condition"],
            }
            for row in subset
        ]
        aggregate = aggregate_vqa_cases(validation_rows)
        positives = sum(row["reference_observed"] is True for row in subset)
        negatives = len(subset) - positives
        auroc = aggregate["auroc"]
        return {
            "count": len(subset),
            "covered": sum(row["covered"] for row in subset),
            "accuracy": aggregate["accuracy_strict"],
            "auroc": {
                "value": auroc["value"],
                "positive_count": positives,
                "negative_count": negatives,
                "unavailable_reason": (
                    None
                    if auroc["unavailable_reason"] is None
                    else "requires_both_positive_and_negative_references"
                ),
            },
        }

    return {
        **metrics(scorable),
        "by_condition": {
            condition: metrics(
                [row for row in scorable if row["condition"] == condition]
            )
            for condition in sorted({row["condition"] for row in scorable})
        },
    }, rows


def score_annotation_replacement(
    packet_value: Any,
    prediction_value: Any,
    annotation_value: Any,
) -> dict[str, Any]:
    packet = validate_annotation_packet(packet_value)
    _, ids = _items(packet["items"], "packet.items")
    predictions, prediction_source = _predictions(
        prediction_value, packet["packet_id"], packet["label_kind"], ids
    )
    references, annotation_meta = _references(
        annotation_value, packet["packet_id"], packet["label_kind"], ids
    )
    scorer = (
        _plan_metrics if packet["label_kind"] == "plan_sub_aspect" else _vqa_metrics
    )
    metrics, rows = scorer(packet, predictions, references)
    proxy = annotation_meta["annotation_source"] == PROXY_SOURCE
    reasons = ["bounded_development_sample_does_not_reproduce_paper_scale"]
    if proxy:
        reasons.insert(0, "codex_development_agent_proxy_is_not_human_gold")
    if annotation_meta["human_reviewer_count"] != 4:
        reasons.append("requires_four_complete_independent_human_annotations")
    return {
        "schema_version": 1,
        "protocol": RESULT_PROTOCOL,
        "packet_id": packet["packet_id"],
        "label_kind": packet["label_kind"],
        **annotation_meta,
        "prediction_source": prediction_source,
        "sample_count": len(ids),
        "calls_started_by_scorer": {"provider": 0, "simulator": 0, "act": 0},
        "metrics": metrics,
        "rows": rows,
        "replacement_contract": {
            "frozen_packet_and_predictions": True,
            "replace_only_annotation_file_for_human_scoring": True,
        },
        "paper_eligible": False,
        "paper_ineligible_reasons": reasons,
        "limitations": [
            (
                "References are one Codex development-agent proxy, not independent human gold."
                if proxy
                else "Human identity and independence are not authenticated by this scorer."
            ),
            "Replacing annotations does not rerun frozen samples or predictions.",
            "This bounded dataset is not the paper-scale validity study.",
        ],
    }


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationReplacementError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnnotationReplacementError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_annotation_replacement(
            _read(args.packet),
            _read(args.predictions),
            _read(args.annotations),
        )
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except AnnotationReplacementError as exc:
        raise SystemExit(f"error: {exc}") from None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
