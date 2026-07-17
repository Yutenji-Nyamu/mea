"""Metrics for live Global Plan Agent calls against development proxy labels."""

from __future__ import annotations

from typing import Any, Mapping

from .query_dataset import QueryDatasetError, validate_query_dataset


_AGILE_CASE_ORDER = (
    # Budgets 1/3/5 are nested and stratified across both tasks and one
    # task-qualified unsupported capability before the full 20-case run.
    "q001",
    "q004",
    "q006",
    "q013",
    "q020",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _unsupported_prediction(
    selection: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    """Read the strict task-qualified unsupported capability contract."""

    aspects: list[str] = []
    tasks: list[str] = []
    qualified = selection.get("unsupported_capabilities")
    if isinstance(qualified, list):
        for item in qualified:
            if not isinstance(item, Mapping):
                continue
            aspect = item.get("aspect_id")
            task = item.get("task_name")
            if isinstance(aspect, str) and aspect and aspect not in aspects:
                aspects.append(aspect)
            if isinstance(task, str) and task and task not in tasks:
                tasks.append(task)
    return aspects, tasks


def _qualified_contract_valid(selection: Mapping[str, Any], decision: str) -> bool:
    qualified = selection.get("unsupported_capabilities")
    if not isinstance(qualified, list):
        return False
    if decision == "route":
        return not qualified
    if decision != "unsupported" or not qualified:
        return False
    return all(
        isinstance(item, Mapping)
        and set(item) == {"task_name", "aspect_id"}
        and isinstance(item.get("task_name"), str)
        and bool(item["task_name"].strip())
        and isinstance(item.get("aspect_id"), str)
        and bool(item["aspect_id"].strip())
        for item in qualified
    )


def score_live_query_case(
    case: Mapping[str, Any],
    route_result: Mapping[str, Any] | None,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    expected_aspects = set(_strings(case.get("gold_sub_aspect_ids")))
    expected_status = str(case.get("capability_status"))
    selection = (
        route_result.get("selection")
        if isinstance(route_result, Mapping)
        and isinstance(route_result.get("selection"), Mapping)
        else None
    )
    schema_valid = selection is not None and error is None
    if selection is None:
        decision = "error"
        predicted_aspects: set[str] = set()
        predicted_task = None
        gap_tasks: list[str] = []
        first_aspect = None
    else:
        decision = str(selection.get("decision") or "invalid")
        schema_valid = schema_valid and _qualified_contract_valid(selection, decision)
        predicted_task = selection.get("task_name")
        first_aspect = selection.get("first_aspect_id")
        if decision == "route":
            predicted_aspects = set(_strings(selection.get("requested_aspect_ids")))
            gap_tasks = []
        else:
            gap_aspects, gap_tasks = _unsupported_prediction(selection)
            predicted_aspects = set(gap_aspects)
    expected_decision = "route" if expected_status == "supported" else "unsupported"
    decision_match = schema_valid and decision == expected_decision
    tp = len(expected_aspects & predicted_aspects)
    task_evaluable = schema_valid and (decision == "route" or bool(gap_tasks))
    task_match = None
    if task_evaluable:
        expected_tasks = set(str(case.get("task_name") or "").split("+"))
        predicted_tasks = (
            {str(predicted_task)} if decision == "route" else set(gap_tasks)
        )
        task_match = predicted_tasks == expected_tasks
    first_evaluable = expected_status == "supported"
    acceptable_first = set(_strings(case.get("acceptable_first_sub_aspect_ids")))
    return {
        "id": str(case.get("id")),
        "query": str(case.get("query")),
        "paper_category": case.get("paper_category"),
        "schema_valid": schema_valid,
        "error": error,
        "expected_decision": expected_decision,
        "predicted_decision": decision,
        "capability_decision_match": decision_match,
        "expected_task_name": str(case.get("task_name")),
        "predicted_task_name": predicted_task,
        "task_match": task_match,
        "task_match_evaluable": task_evaluable,
        "task_qualified_gap_available": decision == "unsupported" and bool(gap_tasks),
        "gold_aspects": sorted(expected_aspects),
        "predicted_aspects": sorted(predicted_aspects),
        "true_positive": tp,
        "false_positive": len(predicted_aspects - expected_aspects),
        "false_negative": len(expected_aspects - predicted_aspects),
        "aspect_exact_set_match": schema_valid
        and predicted_aspects == expected_aspects,
        "first_aspect_match": (
            schema_valid and first_aspect in acceptable_first
            if first_evaluable
            else None
        ),
        "first_aspect_evaluable": first_evaluable,
        "attempt_count": (
            route_result.get("attempt_count")
            if isinstance(route_result, Mapping)
            else None
        ),
    }


def aggregate_live_query_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(case["true_positive"] for case in cases)
    fp = sum(case["false_positive"] for case in cases)
    fn = sum(case["false_negative"] for case in cases)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None:
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    task_cases = [case for case in cases if case["task_match_evaluable"]]
    first_cases = [case for case in cases if case["first_aspect_evaluable"]]
    unsupported_cases = [
        case for case in cases if case["expected_decision"] == "unsupported"
    ]
    category_counts = {
        category: sum(case.get("paper_category") == category for case in cases)
        for category in sorted(
            {
                str(case["paper_category"])
                for case in cases
                if isinstance(case.get("paper_category"), str)
            }
        )
    }
    return {
        "case_count": len(cases),
        "paper_category_counts": category_counts,
        "schema_valid_rate": _ratio(
            sum(case["schema_valid"] for case in cases), len(cases)
        ),
        "capability_decision_accuracy": _ratio(
            sum(case["capability_decision_match"] for case in cases), len(cases)
        ),
        "task_accuracy": _ratio(
            sum(case["task_match"] is True for case in task_cases), len(task_cases)
        ),
        "task_evaluable_count": len(task_cases),
        "task_qualified_gap_coverage": _ratio(
            sum(case["task_qualified_gap_available"] for case in unsupported_cases),
            len(unsupported_cases),
        ),
        "aspect_micro_precision": precision,
        "aspect_micro_recall": recall,
        "aspect_micro_f1": f1,
        "aspect_exact_set_accuracy": _ratio(
            sum(case["aspect_exact_set_match"] for case in cases), len(cases)
        ),
        "first_aspect_accuracy": _ratio(
            sum(case["first_aspect_match"] is True for case in first_cases),
            len(first_cases),
        ),
        "provider_failure_count": sum(bool(case["error"]) for case in cases),
    }


def validate_live_query_budget(
    dataset: Mapping[str, Any], budget: int
) -> list[dict[str, Any]]:
    normalized = validate_query_dataset(dataset)
    if normalized["annotation_status"] != "development_agent_proxy_reviewed":
        raise QueryDatasetError(
            "live scoring requires reviewed development proxy labels"
        )
    if budget not in {1, 3, 5, 20}:
        raise QueryDatasetError("budget must be one of 1, 3, 5, or 20")
    by_id = {case["id"]: case for case in normalized["cases"]}
    order = list(_AGILE_CASE_ORDER) + [
        case["id"]
        for case in normalized["cases"]
        if case["id"] not in _AGILE_CASE_ORDER
    ]
    return [by_id[case_id] for case_id in order[:budget]]


def validate_capability_snapshot(
    cases: list[Mapping[str, Any]], catalog: Mapping[str, Any]
) -> None:
    """Fail before model calls if labels no longer match the trusted catalog."""

    available: dict[str, set[str]] = {}
    for task in catalog.get("tasks", []):
        if not isinstance(task, Mapping):
            continue
        task_name = task.get("task_name")
        aspects = task.get("aspects")
        if not isinstance(task_name, str) or not isinstance(aspects, list):
            continue
        available[task_name] = {
            str(item.get("aspect_id"))
            for item in aspects
            if isinstance(item, Mapping) and isinstance(item.get("aspect_id"), str)
        }
    stale: list[str] = []
    for case in cases:
        task_name = str(case.get("task_name") or "")
        aspects = set(_strings(case.get("gold_sub_aspect_ids")))
        derived = (
            "supported"
            if case.get("setting") == "single_task"
            and task_name in available
            and aspects.issubset(available[task_name])
            else "unsupported"
        )
        if case.get("capability_status") != derived:
            stale.append(f"{case.get('id')}:{case.get('capability_status')}->{derived}")
    if stale:
        raise QueryDatasetError(
            "dataset capability_status is stale for this catalog: " + ", ".join(stale)
        )


__all__ = [
    "aggregate_live_query_cases",
    "score_live_query_case",
    "validate_capability_snapshot",
    "validate_live_query_budget",
]
