"""Deterministic evidence policy for bounded MEA planning.

The language model may explain or order remaining user-requested sub-aspects,
but it cannot turn missing telemetry or a numeric/visual contradiction into a
confident conclusion.  This module converts the latest round artifacts into a
small, auditable control decision before the next Plan Agent call.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SEMANTIC_ABSENCE_REASONS = {
    "pickup_not_observed",
    "contact_not_observed_after_pickup",
}


def _base_template_id(round_plan: dict[str, Any]) -> str:
    return str(round_plan.get("verification_of") or round_plan["template_id"])


def _remaining_template_ids(current_plan: dict[str, Any]) -> list[str]:
    executed = {
        _base_template_id(round_plan)
        for round_plan in current_plan.get("rounds", [])
    }
    return [
        template_id
        for template_id in current_plan.get("requested_template_ids", [])
        if template_id not in executed
    ]


def _verification_attempts(
    current_plan: dict[str, Any], template_id: str
) -> int:
    return sum(
        1
        for round_plan in current_plan.get("rounds", [])
        if round_plan.get("verification_of") == template_id
    )


def _policy_cohort(
    aggregate: dict[str, Any], metric: str
) -> dict[str, Any] | None:
    for metric_result in aggregate.get("metrics", []):
        if metric_result.get("metric") != metric:
            continue
        for cohort in metric_result.get("cohorts", []):
            if cohort.get("role") == "policy_under_evaluation":
                return cohort
    return None


def _semantic_missing_count(
    planned_tool: dict[str, Any], metric: str
) -> tuple[int, list[str]]:
    count = 0
    reasons: list[str] = []
    route_metric = (
        planned_tool.get("route_decision", {}).get("metric")
        or planned_tool.get("reference_tool")
    )
    if route_metric and route_metric != metric:
        return 0, reasons
    for episode in planned_tool.get("episodes", []):
        if episode.get("role") != "policy_under_evaluation":
            continue
        if episode.get("value") is not None:
            continue
        reason = (episode.get("details") or {}).get("reason")
        if reason in SEMANTIC_ABSENCE_REASONS:
            count += 1
            reasons.append(str(reason))
    return count, sorted(reasons)


def _aggregate_quality(
    round_plan: dict[str, Any], round_summary: dict[str, Any]
) -> dict[str, Any]:
    observations = round_summary.get("observations") or {}
    aggregate = observations.get("aggregate") or {}
    planned_tool = observations.get("planned_tool") or {}
    expected = int(round_plan.get("execution", {}).get("num_episodes") or 0)
    metric = str(round_plan.get("tool_request", {}).get("metric") or "")
    result = {
        "metric": metric,
        "expected_policy_episodes": expected,
        "aggregate_status": aggregate.get("status"),
        "input_issue_count": len(aggregate.get("input_issues") or []),
        "valid": 0,
        "missing": 0,
        "invalid": 0,
        "semantic_missing": 0,
        "semantic_missing_reasons": [],
        "observed_policy_episodes": 0,
        "complete": False,
        "reasons": [],
    }
    if not str(aggregate.get("status", "")).startswith("passed"):
        result["reasons"].append("aggregate_not_passed")
    if result["input_issue_count"]:
        result["reasons"].append("aggregate_input_issues")
    cohort = _policy_cohort(aggregate, metric)
    if cohort is None:
        result["reasons"].append("policy_metric_cohort_missing")
        return result
    quality = (cohort.get("summary") or {}).get("quality") or {}
    for name in ("valid", "missing", "invalid"):
        value = quality.get(name, 0)
        if isinstance(value, dict):
            value = value.get("value", 0)
        result[name] = int(value or 0)
    semantic_missing, semantic_reasons = _semantic_missing_count(
        planned_tool, metric
    )
    result["semantic_missing"] = min(semantic_missing, result["missing"])
    result["semantic_missing_reasons"] = semantic_reasons
    result["observed_policy_episodes"] = (
        result["valid"] + result["semantic_missing"]
    )
    if result["invalid"]:
        result["reasons"].append("invalid_policy_results")
    unresolved_missing = result["missing"] - result["semantic_missing"]
    if unresolved_missing:
        result["reasons"].append("unexplained_missing_policy_results")
    if result["observed_policy_episodes"] < expected:
        result["reasons"].append("policy_episode_coverage_incomplete")
    result["complete"] = not result["reasons"]
    return result


def assess_evidence(
    current_plan: dict[str, Any],
    observation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the hard control action implied by current evaluation evidence."""

    if not current_plan.get("rounds") or not observation_history:
        raise ValueError("current plan and observation history must be non-empty")
    latest_plan = current_plan["rounds"][-1]
    latest = observation_history[-1]
    base_template = _base_template_id(latest_plan)
    remaining = _remaining_template_ids(current_plan)
    attempts = _verification_attempts(current_plan, base_template)
    budget_remaining = max(
        int(current_plan.get("max_rounds") or 0)
        - len(current_plan.get("rounds", [])),
        0,
    )
    quality = _aggregate_quality(latest_plan, latest)
    vqa = (latest.get("observations") or {}).get("execution_vqa") or {}
    conflict = bool(vqa.get("evidence_conflict"))
    pipeline_passed = bool(latest.get("pipeline_passed"))

    state = "sufficient"
    reasons: list[str] = []
    unresolved = False
    if not pipeline_passed:
        state = "pipeline_failure"
        reasons.append("latest_pipeline_failed")
    elif conflict:
        state = "evidence_conflict"
        reasons.append("execution_vqa_conflicts_with_numeric_evidence")
    elif not quality["complete"]:
        state = "aggregate_uncertain"
        reasons.extend(quality["reasons"])

    if state == "pipeline_failure":
        required_action = "stop"
    elif state in {"evidence_conflict", "aggregate_uncertain"}:
        if budget_remaining > 0 and attempts == 0:
            required_action = "verify"
        else:
            required_action = "stop"
            unresolved = True
            reasons.append(
                "verification_already_used"
                if attempts
                else "round_budget_exhausted"
            )
    elif remaining and budget_remaining > 0:
        required_action = "continue"
    else:
        required_action = "stop"
        if remaining:
            unresolved = True
            reasons.append("round_budget_exhausted_with_uncovered_templates")
        else:
            reasons.append("all_requested_sub_aspects_have_sufficient_evidence")

    return {
        "schema_version": 1,
        "state": state,
        "required_action": required_action,
        "reasons": reasons,
        "checks": deepcopy(quality),
        "latest_round_id": latest_plan.get("round_id"),
        "latest_template_id": latest_plan.get("template_id"),
        "verification_of": base_template,
        "verification_attempts_used": attempts,
        "round_budget_remaining": budget_remaining,
        "remaining_template_ids": remaining,
        "unresolved": unresolved,
    }


__all__ = [
    "SEMANTIC_ABSENCE_REASONS",
    "assess_evidence",
]
