"""Validation for the unreviewed 20-query Planner/aspect draft dataset."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


class QueryDatasetError(ValueError):
    pass


def validate_query_dataset(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise QueryDatasetError("query dataset schema_version must be 1")
    if value.get("annotation_status") != "model_draft_unreviewed":
        raise QueryDatasetError("draft dataset must remain model_draft_unreviewed")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 20:
        raise QueryDatasetError("query dataset must contain exactly 20 cases")
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
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or set(case) != required:
            raise QueryDatasetError(f"case {index} fields do not match the contract")
        case_id = str(case.get("id") or "")
        if not case_id or case_id in seen:
            raise QueryDatasetError(f"case id is missing or duplicate: {case_id!r}")
        seen.add(case_id)
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
        annotation = case.get("annotation")
        if annotation != {
            "source": "model_draft",
            "review_status": "unreviewed",
            "human_votes": [],
        }:
            raise QueryDatasetError(f"{case_id} must not masquerade as human gold")
        normalized.append(dict(case))
    result = dict(value)
    result["cases"] = normalized
    return result


def summarize_query_dataset(value: Any) -> dict[str, Any]:
    dataset = validate_query_dataset(value)
    cases = dataset["cases"]
    status = Counter(case["capability_status"] for case in cases)
    aspects = Counter(
        aspect for case in cases for aspect in case["gold_sub_aspect_ids"]
    )
    return {
        "schema_version": 1,
        "dataset_id": dataset.get("dataset_id"),
        "case_count": len(cases),
        "capability_status_counts": dict(sorted(status.items())),
        "aspect_counts": dict(sorted(aspects.items())),
        "human_agent_agreement": None,
        "paper_table_eligible": False,
        "unavailable_reason": "no_human_majority_annotation",
        "recommended_human_reviewers": 4,
    }
