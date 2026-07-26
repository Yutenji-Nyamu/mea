"""Validation for bounded Planner/aspect development datasets."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


class QueryDatasetError(ValueError):
    pass


_DRAFT_ANNOTATION = {
    "source": "model_draft",
    "review_status": "unreviewed",
    "human_votes": [],
}
_PROXY_ANNOTATION = {
    "source": "development_agent_proxy",
    "review_status": "proxy_reviewed",
    "annotator_id": "codex_development_agent",
    "human_votes": [],
    "paper_eligible": False,
}
PAPER_CATEGORIES = {
    "generalization",
    "performance",
    "safety",
    "robustness",
    "multi_task",
}
DEFAULT_DATASET_CONTRACT = {
    "minimum_case_count": 20,
    "required_paper_categories": sorted(PAPER_CATEGORIES),
}


def _validate_annotation(value: Mapping[str, Any]) -> str:
    status = value.get("annotation_status")
    protocol = value.get("annotation_protocol")
    if status == "model_draft_unreviewed":
        if protocol is not None:
            raise QueryDatasetError("unreviewed draft must not claim a review protocol")
        return status
    if status != "development_agent_proxy_reviewed":
        raise QueryDatasetError("unsupported query annotation_status")
    expected_protocol = {
        "role": "development_agent_proxy",
        "tested_agent": "runtime_global_plan_agent",
        "human_reviewer_count": 0,
        "paper_eligible": False,
        "replacement_required": "independent_human_majority_annotation",
    }
    if protocol != expected_protocol:
        raise QueryDatasetError("development proxy protocol disclosure changed")
    return status


def _validate_dataset_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = value.get("dataset_contract", DEFAULT_DATASET_CONTRACT)
    if not isinstance(raw, Mapping) or set(raw) != {
        "minimum_case_count",
        "required_paper_categories",
    }:
        raise QueryDatasetError("invalid dataset_contract fields")
    minimum = raw.get("minimum_case_count")
    categories = raw.get("required_paper_categories")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise QueryDatasetError("minimum_case_count must be a positive integer")
    if (
        not isinstance(categories, list)
        or not categories
        or len(categories) != len(set(categories))
        or not set(categories).issubset(PAPER_CATEGORIES)
    ):
        raise QueryDatasetError("required_paper_categories are invalid")
    return {
        "minimum_case_count": minimum,
        "required_paper_categories": sorted(categories),
    }


def validate_query_dataset(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise QueryDatasetError("query dataset schema_version must be 1")
    status = _validate_annotation(value)
    contract = _validate_dataset_contract(value)
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) < contract["minimum_case_count"]:
        raise QueryDatasetError(
            "query dataset has fewer cases than minimum_case_count"
        )
    required = {
        "id",
        "query",
        "setting",
        "task_name",
        "task_profile",
        "gold_sub_aspect_ids",
        "acceptable_first_sub_aspect_ids",
        "capability_status",
        "annotation",
    }
    if status == "development_agent_proxy_reviewed":
        required.add("paper_category")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    expected_annotation = (
        _DRAFT_ANNOTATION if status == "model_draft_unreviewed" else _PROXY_ANNOTATION
    )
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or set(case) != required:
            raise QueryDatasetError(f"case {index} fields do not match the contract")
        case_id = str(case.get("id") or "")
        if not case_id or case_id in seen:
            raise QueryDatasetError(f"case id is missing or duplicate: {case_id!r}")
        seen.add(case_id)
        for text_field in ("query", "setting", "task_name", "task_profile"):
            if (
                not isinstance(case.get(text_field), str)
                or not case[text_field].strip()
            ):
                raise QueryDatasetError(f"{case_id} has invalid {text_field}")
        aspects = case.get("gold_sub_aspect_ids")
        acceptable = case.get("acceptable_first_sub_aspect_ids")
        if (
            not isinstance(aspects, list)
            or not aspects
            or len(aspects) != len(set(aspects))
            or not all(isinstance(item, str) and item for item in aspects)
        ):
            raise QueryDatasetError(f"{case_id} has invalid gold_sub_aspect_ids")
        if (
            not isinstance(acceptable, list)
            or not acceptable
            or not set(acceptable).issubset(aspects)
        ):
            raise QueryDatasetError(
                f"{case_id} acceptable first aspects must be a non-empty subset"
            )
        if case.get("capability_status") not in {"supported", "unsupported"}:
            raise QueryDatasetError(f"{case_id} has invalid capability_status")
        if (
            status == "development_agent_proxy_reviewed"
            and case.get("paper_category") not in PAPER_CATEGORIES
        ):
            raise QueryDatasetError(f"{case_id} has invalid paper_category")
        if case.get("annotation") != expected_annotation:
            message = (
                "must remain an unreviewed model draft"
                if status == "model_draft_unreviewed"
                else "must disclose development-agent proxy annotation"
            )
            raise QueryDatasetError(f"{case_id} {message}")
        normalized.append(dict(case))
    if status == "development_agent_proxy_reviewed":
        observed_categories = {
            case["paper_category"] for case in normalized
        }
        missing_categories = sorted(
            set(contract["required_paper_categories"]) - observed_categories
        )
        if missing_categories:
            raise QueryDatasetError(
                f"query dataset misses required paper categories: {missing_categories}"
            )
    result = dict(value)
    result["dataset_contract"] = contract
    result["cases"] = normalized
    return result


def summarize_query_dataset(value: Any) -> dict[str, Any]:
    dataset = validate_query_dataset(value)
    cases = dataset["cases"]
    status = Counter(case["capability_status"] for case in cases)
    aspects = Counter(
        aspect for case in cases for aspect in case["gold_sub_aspect_ids"]
    )
    categories = Counter(
        case["paper_category"] for case in cases if "paper_category" in case
    )
    annotation_status = dataset["annotation_status"]
    unavailable_reason = (
        "no_human_majority_annotation"
        if annotation_status == "model_draft_unreviewed"
        else "development_agent_proxy_is_not_human_gold"
    )
    return {
        "schema_version": 1,
        "dataset_id": dataset.get("dataset_id"),
        "annotation_status": annotation_status,
        "case_count": len(cases),
        "dataset_contract": dataset["dataset_contract"],
        "capability_status_counts": dict(sorted(status.items())),
        "aspect_counts": dict(sorted(aspects.items())),
        "paper_category_counts": dict(sorted(categories.items())),
        "human_agent_agreement": None,
        "paper_table_eligible": False,
        "unavailable_reason": unavailable_reason,
        "recommended_human_reviewers": 4,
    }


__all__ = [
    "DEFAULT_DATASET_CONTRACT",
    "QueryDatasetError",
    "summarize_query_dataset",
    "validate_query_dataset",
]
