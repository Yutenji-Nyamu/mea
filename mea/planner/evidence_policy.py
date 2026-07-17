"""Deterministic evidence policy for bounded MEA planning.

The language model may explain or order remaining user-requested sub-aspects,
but it cannot turn missing telemetry or a numeric/visual contradiction into a
confident conclusion.  This module converts the latest round artifacts into a
small, auditable control decision before the next Plan Agent call.
"""

from __future__ import annotations

from copy import deepcopy
import math
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


def assess_conditional_transition(
    current_plan: dict[str, Any],
    observation_history: list[dict[str, Any]],
    *,
    aspect_catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Map trusted evidence to one bounded aspect transition.

    This is the task-agnostic control contract used by adaptive planners.  A
    task adapter supplies only an ordered ``aspect_id -> template_ids``
    catalog.  The runtime, rather than the language model, then determines
    whether to stop, drill into the current aspect, or switch to an uncovered
    aspect.
    """

    if not current_plan.get("rounds") or not observation_history:
        raise ValueError("current plan and observation history must be non-empty")
    if len(current_plan["rounds"]) != len(observation_history):
        raise ValueError("each planned round needs exactly one observation")

    requested_aspects = current_plan.get("requested_aspect_ids")
    if not isinstance(requested_aspects, list) or not requested_aspects:
        raise ValueError("requested_aspect_ids must be a non-empty list")
    unknown = [item for item in requested_aspects if item not in aspect_catalog]
    if unknown:
        raise ValueError(f"unknown requested aspects: {unknown}")

    rounds = current_plan["rounds"]
    latest_round = rounds[-1]
    latest = observation_history[-1]
    executed_templates = {
        str(round_plan.get("template_id")) for round_plan in rounds
    }
    executed_aspects = {
        str(round_plan.get("aspect_id") or round_plan.get("sub_aspect"))
        for round_plan in rounds
    }
    current_aspect = str(
        latest_round.get("aspect_id") or latest_round.get("sub_aspect") or ""
    )
    if current_aspect not in requested_aspects:
        raise ValueError("latest round aspect is not requested")

    remaining_by_aspect: dict[str, list[str]] = {}
    for aspect_id in requested_aspects:
        template_ids = aspect_catalog[aspect_id].get("template_ids")
        if not isinstance(template_ids, list) or not template_ids:
            raise ValueError(f"aspect {aspect_id!r} has no trusted templates")
        remaining_by_aspect[aspect_id] = [
            str(template_id)
            for template_id in template_ids
            if str(template_id) not in executed_templates
        ]

    budget_remaining = max(
        int(current_plan.get("max_rounds") or 0) - len(rounds), 0
    )
    observations = latest.get("observations") or {}
    aggregate = observations.get("aggregate") or {}
    execution_vqa = observations.get("execution_vqa") or {}
    aggregate_status = str(aggregate.get("status") or "missing")
    evidence_conflict = bool(execution_vqa.get("evidence_conflict"))
    generic = assess_evidence(current_plan, observation_history)
    state = generic["state"]
    reasons = list(generic.get("reasons") or [])

    raw_policy_success = observations.get("policy_success")
    policy_success = None
    if (
        not isinstance(raw_policy_success, bool)
        and isinstance(raw_policy_success, (int, float))
        and math.isfinite(float(raw_policy_success))
        and 0.0 <= float(raw_policy_success) <= 1.0
    ):
        policy_success = float(raw_policy_success)
    elif state == "sufficient":
        state = "aggregate_uncertain"
        reasons.append("policy_success_missing_or_invalid")

    transitions: dict[str, list[str]] = {
        "drill_down": [],
        "switch_aspect": [],
    }
    unseen_aspects = [
        aspect_id
        for aspect_id in requested_aspects
        if aspect_id not in executed_aspects and remaining_by_aspect[aspect_id]
    ]
    other_remaining_aspects = [
        aspect_id
        for aspect_id in requested_aspects
        if aspect_id != current_aspect and remaining_by_aspect[aspect_id]
    ]
    required_action = "stop"
    required_transition = "stop"
    required_next_aspect = None
    # Preserve uncertainty already established by the generic evidence
    # contract. Navigation/budget logic may add unresolved coverage, but it
    # must never turn a final conflict into a resolved stop.
    unresolved = bool(generic.get("unresolved"))

    def require_continue(transition: str, aspect_id: str) -> None:
        nonlocal required_action, required_transition, required_next_aspect
        required_action = "continue"
        required_transition = transition
        required_next_aspect = aspect_id
        transitions[transition] = [aspect_id]

    if state == "pipeline_failure":
        reasons.append("pipeline_failure_forces_stop")
    elif budget_remaining <= 0:
        uncovered_variants = any(remaining_by_aspect.values())
        if state in {"evidence_conflict", "aggregate_uncertain"}:
            unresolved = True
            reasons.append("round_budget_exhausted_with_unresolved_evidence")
        if uncovered_variants:
            unresolved = True
            reasons.append("round_budget_exhausted_with_uncovered_variants")
    elif state in {"evidence_conflict", "aggregate_uncertain"}:
        if remaining_by_aspect[current_aspect]:
            require_continue("drill_down", current_aspect)
            reasons.append("uncertain_evidence_requires_same_aspect_counterfactual")
        else:
            unresolved = True
            reasons.append("uncertain_evidence_has_no_same_aspect_counterfactual")
    elif policy_success is not None and policy_success < 1.0:
        if remaining_by_aspect[current_aspect]:
            require_continue("drill_down", current_aspect)
            reasons.append("policy_failure_requires_same_aspect_counterfactual")
        elif unseen_aspects:
            require_continue("switch_aspect", unseen_aspects[0])
            reasons.append("failed_aspect_exhausted_switch_to_uncovered_aspect")
        elif other_remaining_aspects:
            require_continue("switch_aspect", other_remaining_aspects[0])
            reasons.append("failed_aspect_exhausted_switch_to_remaining_aspect")
    elif unseen_aspects:
        require_continue("switch_aspect", unseen_aspects[0])
        reasons.append("successful_sentinel_switches_to_uncovered_aspect")
    elif remaining_by_aspect[current_aspect]:
        require_continue("drill_down", current_aspect)
        reasons.append("all_aspects_seen_complete_current_counterfactual")
    elif other_remaining_aspects:
        require_continue("switch_aspect", other_remaining_aspects[0])
        reasons.append("current_aspect_complete_switch_to_remaining_aspect")
    else:
        reasons.append("all_requested_variants_exhausted")

    return {
        "schema_version": 1,
        "state": state,
        "pipeline_passed": bool(latest.get("pipeline_passed")),
        "latest_round_id": latest_round.get("round_id"),
        "latest_template_id": latest_round.get("template_id"),
        "current_aspect_id": current_aspect,
        "policy_success": policy_success,
        "aggregate_status": aggregate_status,
        "evidence_conflict": evidence_conflict,
        "aggregate_checks": generic.get("checks", {}),
        "reasons": reasons,
        "unresolved": unresolved,
        "round_budget_remaining": budget_remaining,
        "remaining_template_ids_by_aspect": remaining_by_aspect,
        "available_transitions": transitions,
        "required_action": required_action,
        "required_transition": required_transition,
        "required_next_aspect_id": required_next_aspect,
        "allowed_actions": [required_action],
    }


__all__ = [
    "SEMANTIC_ABSENCE_REASONS",
    "assess_conditional_transition",
    "assess_evidence",
]
