"""Deterministic evidence policy for bounded MEA planning.

The language model may explain or order remaining user-requested sub-aspects,
but it cannot turn missing telemetry or a numeric/visual contradiction into a
confident conclusion.  This module converts the latest round artifacts into a
small, auditable control decision before the next Plan Agent call.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping


SEMANTIC_ABSENCE_REASONS = {
    "pickup_not_observed",
    "contact_not_observed_after_pickup",
}


class EvidencePacketError(ValueError):
    """Raised when one typed evidence packet is incomplete or inconsistent."""


_EVIDENCE_PACKET_KEYS = {
    "schema_version",
    "round_id",
    "template_id",
    "pipeline",
    "policy",
    "rule",
    "vqa",
    "evidence_strength",
    "reason_codes",
}
_PIPELINE_KEYS = {"passed", "failure_stage"}
_POLICY_KEYS = {"success_rate", "reported"}
_RULE_KEYS = {
    "metric",
    "expected_policy_episodes",
    "aggregate_status",
    "input_issue_count",
    "valid",
    "missing",
    "invalid",
    "semantic_missing",
    "semantic_missing_reasons",
    "observed_policy_episodes",
    "complete",
    "reasons",
}
_VQA_KEYS = {"required", "status", "evidence_conflict"}
_VQA_STATUSES = {"passed", "failed", "skipped", "missing"}
_EVIDENCE_STRENGTHS = {
    "sufficient",
    "uncertain",
    "conflicting",
    "pipeline_invalid",
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
    expected_raw = round_plan.get("execution", {}).get("num_episodes", 0)
    if (
        isinstance(expected_raw, bool)
        or not isinstance(expected_raw, int)
        or expected_raw < 0
    ):
        raise EvidencePacketError(
            "round.execution.num_episodes must be a non-negative integer"
        )
    expected = expected_raw
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
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EvidencePacketError(
                f"aggregate Rule count {name!r} must be a non-negative integer"
            )
        result[name] = value
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


def _policy_success_rate(round_summary: Mapping[str, Any]) -> float | None:
    observations = round_summary.get("observations") or {}
    raw = observations.get("policy_success")
    if (
        not isinstance(raw, bool)
        and isinstance(raw, (int, float))
        and math.isfinite(float(raw))
        and 0.0 <= float(raw) <= 1.0
    ):
        return float(raw)
    return None


def _execution_vqa_required(round_plan: Mapping[str, Any]) -> bool:
    """Return whether this round contract explicitly asks for execution VQA.

    Older plan shapes did not request execution VQA.  They must remain usable,
    while modern rounds that list the observation or a visual phenomenon must
    not silently treat a missing/failed visual check as sufficient evidence.
    """

    requested = round_plan.get("observations")
    if isinstance(requested, list) and "execution_vqa" in requested:
        return True
    phenomenon_ids = round_plan.get("vqa_phenomenon_ids")
    if isinstance(phenomenon_ids, list) and bool(phenomenon_ids):
        return True
    tool_proposal = round_plan.get("tool_proposal")
    if isinstance(tool_proposal, Mapping):
        for field in ("vqa_phenomenon_ids", "vqa_question_specs"):
            values = tool_proposal.get(field)
            if isinstance(values, list) and bool(values):
                return True
    return False


def validate_evidence_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the compact Rule/VQA/policy evidence passed to Plan."""

    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_PACKET_KEYS:
        raise EvidencePacketError(
            "EvidencePacket fields must be exactly "
            f"{sorted(_EVIDENCE_PACKET_KEYS)}"
        )
    packet = deepcopy(dict(value))
    if packet.get("schema_version") != 1:
        raise EvidencePacketError("EvidencePacket.schema_version must be 1")
    for field in ("round_id", "template_id"):
        if not isinstance(packet.get(field), str) or not packet[field]:
            raise EvidencePacketError(f"EvidencePacket.{field} must be non-empty")

    pipeline = packet.get("pipeline")
    if not isinstance(pipeline, Mapping) or set(pipeline) != _PIPELINE_KEYS:
        raise EvidencePacketError("EvidencePacket.pipeline fields changed")
    if not isinstance(pipeline.get("passed"), bool):
        raise EvidencePacketError("EvidencePacket.pipeline.passed must be boolean")
    failure_stage = pipeline.get("failure_stage")
    if failure_stage is not None and (
        not isinstance(failure_stage, str) or not failure_stage
    ):
        raise EvidencePacketError(
            "EvidencePacket.pipeline.failure_stage must be null or non-empty"
        )

    policy = packet.get("policy")
    if not isinstance(policy, Mapping) or set(policy) != _POLICY_KEYS:
        raise EvidencePacketError("EvidencePacket.policy fields changed")
    if not isinstance(policy.get("reported"), bool):
        raise EvidencePacketError("EvidencePacket.policy.reported must be boolean")
    success_rate = policy.get("success_rate")
    if success_rate is not None and (
        isinstance(success_rate, bool)
        or not isinstance(success_rate, (int, float))
        or not math.isfinite(float(success_rate))
        or not 0.0 <= float(success_rate) <= 1.0
    ):
        raise EvidencePacketError(
            "EvidencePacket.policy.success_rate must be null or in [0, 1]"
        )
    if policy["reported"] != (success_rate is not None):
        raise EvidencePacketError(
            "EvidencePacket.policy.reported disagrees with success_rate"
        )

    rule = packet.get("rule")
    if not isinstance(rule, Mapping) or set(rule) != _RULE_KEYS:
        raise EvidencePacketError("EvidencePacket.rule fields changed")
    for field in (
        "expected_policy_episodes",
        "input_issue_count",
        "valid",
        "missing",
        "invalid",
        "semantic_missing",
        "observed_policy_episodes",
    ):
        item = rule.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise EvidencePacketError(
                f"EvidencePacket.rule.{field} must be non-negative"
            )
    if not isinstance(rule.get("complete"), bool):
        raise EvidencePacketError("EvidencePacket.rule.complete must be boolean")
    for field in ("semantic_missing_reasons", "reasons"):
        items = rule.get(field)
        if (
            not isinstance(items, list)
            or any(not isinstance(item, str) or not item for item in items)
        ):
            raise EvidencePacketError(
                f"EvidencePacket.rule.{field} must be a string list"
            )
        if field == "reasons" and len(items) != len(set(items)):
            raise EvidencePacketError(
                "EvidencePacket.rule.reasons must not repeat control reasons"
            )
    aggregate_status = rule.get("aggregate_status")
    if aggregate_status is not None and not isinstance(aggregate_status, str):
        raise EvidencePacketError(
            "EvidencePacket.rule.aggregate_status must be null or string"
        )
    if not isinstance(rule.get("metric"), str):
        raise EvidencePacketError("EvidencePacket.rule.metric must be a string")

    vqa = packet.get("vqa")
    if not isinstance(vqa, Mapping) or set(vqa) != _VQA_KEYS:
        raise EvidencePacketError("EvidencePacket.vqa fields changed")
    if not isinstance(vqa.get("required"), bool):
        raise EvidencePacketError("EvidencePacket.vqa.required must be boolean")
    if vqa.get("status") not in _VQA_STATUSES or not isinstance(
        vqa.get("evidence_conflict"), bool
    ):
        raise EvidencePacketError("EvidencePacket.vqa fields are invalid")

    strength = packet.get("evidence_strength")
    if strength not in _EVIDENCE_STRENGTHS:
        raise EvidencePacketError(
            f"unsupported EvidencePacket.evidence_strength: {strength!r}"
        )
    reasons = packet.get("reason_codes")
    if (
        not isinstance(reasons, list)
        or any(not isinstance(item, str) or not item for item in reasons)
        or len(reasons) != len(set(reasons))
    ):
        raise EvidencePacketError(
            "EvidencePacket.reason_codes must be a unique string list"
        )
    expected_strength = (
        "pipeline_invalid"
        if not pipeline["passed"]
        else "conflicting"
        if vqa["evidence_conflict"]
        else "uncertain"
        if (vqa["required"] and vqa["status"] != "passed")
        or not rule["complete"]
        else "sufficient"
    )
    if strength != expected_strength:
        raise EvidencePacketError(
            "EvidencePacket.evidence_strength disagrees with its typed evidence"
        )
    return packet


def build_evidence_packet(
    current_plan: Mapping[str, Any],
    observation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project raw round output into a small categorical evidence contract.

    ``evidence_strength`` describes whether the transport, Rule aggregate, and
    visual checks are usable.  Policy success is kept as a separate typed
    field because some generic metrics do not report it; adaptive navigation
    may still require it before choosing the next aspect.
    """

    rounds = current_plan.get("rounds") if isinstance(current_plan, Mapping) else None
    if not isinstance(rounds, list) or not rounds or not observation_history:
        raise EvidencePacketError("plan and observation history must be non-empty")
    if len(rounds) != len(observation_history):
        raise EvidencePacketError("each planned round needs exactly one observation")
    latest_plan = rounds[-1]
    latest = observation_history[-1]
    if not isinstance(latest_plan, Mapping) or not isinstance(latest, Mapping):
        raise EvidencePacketError("latest plan and observation must be objects")
    round_id = latest_plan.get("round_id")
    if latest.get("round_id") != round_id:
        raise EvidencePacketError("observation.round_id does not match plan")
    quality = _aggregate_quality(dict(latest_plan), dict(latest))
    observations = latest.get("observations") or {}
    if not isinstance(observations, Mapping):
        raise EvidencePacketError("observation.observations must be an object")
    raw_vqa = observations.get("execution_vqa")
    if raw_vqa is None:
        vqa: Mapping[str, Any] = {}
    elif not isinstance(raw_vqa, Mapping):
        raise EvidencePacketError("observation.execution_vqa must be an object")
    else:
        vqa = raw_vqa
    pipeline_passed = latest.get("pipeline_passed")
    if not isinstance(pipeline_passed, bool):
        raise EvidencePacketError("observation.pipeline_passed must be boolean")
    if vqa and not isinstance(vqa.get("evidence_conflict"), bool):
        raise EvidencePacketError(
            "observation.execution_vqa.evidence_conflict must be boolean"
        )
    conflict = vqa.get("evidence_conflict", False)
    vqa_required = _execution_vqa_required(latest_plan)
    vqa_status = vqa.get("status", "missing")
    if vqa_status not in _VQA_STATUSES:
        raise EvidencePacketError(
            "observation.execution_vqa.status must be passed, failed, skipped, "
            "or missing"
        )
    if not pipeline_passed:
        strength = "pipeline_invalid"
        reasons = ["latest_pipeline_failed"]
    elif conflict:
        strength = "conflicting"
        reasons = ["execution_vqa_conflicts_with_numeric_evidence"]
    elif vqa_required and vqa_status != "passed":
        strength = "uncertain"
        reasons = [f"execution_vqa_{vqa_status}"]
    elif not quality["complete"]:
        strength = "uncertain"
        reasons = list(quality["reasons"])
    else:
        strength = "sufficient"
        reasons = []
    success_rate = _policy_success_rate(latest)
    failure_stage = latest.get("failure_stage")
    if isinstance(failure_stage, str):
        failure_stage = failure_stage.strip() or None
    packet = {
        "schema_version": 1,
        "round_id": str(round_id or ""),
        "template_id": str(latest_plan.get("template_id") or ""),
        "pipeline": {
            "passed": pipeline_passed,
            "failure_stage": failure_stage,
        },
        "policy": {
            "success_rate": success_rate,
            "reported": success_rate is not None,
        },
        "rule": deepcopy(quality),
        "vqa": {
            "required": vqa_required,
            "status": vqa_status,
            "evidence_conflict": conflict,
        },
        "evidence_strength": strength,
        "reason_codes": reasons,
    }
    return validate_evidence_packet(packet)


def assess_evidence(
    current_plan: dict[str, Any],
    observation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the hard control action implied by current evaluation evidence."""

    if not current_plan.get("rounds") or not observation_history:
        raise ValueError("current plan and observation history must be non-empty")
    latest_plan = current_plan["rounds"][-1]
    base_template = _base_template_id(latest_plan)
    remaining = _remaining_template_ids(current_plan)
    attempts = _verification_attempts(current_plan, base_template)
    budget_remaining = max(
        int(current_plan.get("max_rounds") or 0)
        - len(current_plan.get("rounds", [])),
        0,
    )
    packet = build_evidence_packet(current_plan, observation_history)
    quality = packet["rule"]
    strength = packet["evidence_strength"]

    state = "sufficient"
    reasons: list[str] = list(packet["reason_codes"])
    unresolved = False
    if strength == "pipeline_invalid":
        state = "pipeline_failure"
    elif strength == "conflicting":
        state = "evidence_conflict"
    elif strength == "uncertain":
        state = "aggregate_uncertain"

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
        "evidence_packet": packet,
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
    generic = assess_evidence(current_plan, observation_history)
    packet = generic["evidence_packet"]
    aggregate_status = str(packet["rule"].get("aggregate_status") or "missing")
    evidence_conflict = bool(packet["vqa"]["evidence_conflict"])
    state = generic["state"]
    reasons = list(generic.get("reasons") or [])

    policy_success = packet["policy"]["success_rate"]
    if policy_success is None and state == "sufficient":
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

    def require_continue(transition: str, aspect_ids: list[str]) -> None:
        nonlocal required_action, required_transition, required_next_aspect
        candidates = list(dict.fromkeys(aspect_ids))
        if not candidates:
            raise ValueError("continue transition requires at least one aspect")
        required_action = "continue"
        required_transition = transition
        # The first item remains the deterministic fallback for callers that
        # do not ask a model to choose.  All items are legal bounded choices.
        required_next_aspect = candidates[0]
        transitions[transition] = candidates

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
            require_continue("drill_down", [current_aspect])
            reasons.append("uncertain_evidence_requires_same_aspect_counterfactual")
        else:
            unresolved = True
            reasons.append("uncertain_evidence_has_no_same_aspect_counterfactual")
    elif policy_success is not None and policy_success < 1.0:
        if remaining_by_aspect[current_aspect]:
            require_continue("drill_down", [current_aspect])
            reasons.append("policy_failure_requires_same_aspect_counterfactual")
        elif unseen_aspects:
            require_continue("switch_aspect", unseen_aspects)
            reasons.append("failed_aspect_exhausted_switch_to_uncovered_aspect")
        elif other_remaining_aspects:
            require_continue("switch_aspect", other_remaining_aspects)
            reasons.append("failed_aspect_exhausted_switch_to_remaining_aspect")
    elif unseen_aspects:
        require_continue("switch_aspect", unseen_aspects)
        reasons.append("successful_sentinel_switches_to_uncovered_aspect")
    elif remaining_by_aspect[current_aspect]:
        require_continue("drill_down", [current_aspect])
        reasons.append("all_aspects_seen_complete_current_counterfactual")
    elif other_remaining_aspects:
        require_continue("switch_aspect", other_remaining_aspects)
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
        "evidence_packet": packet,
    }


__all__ = [
    "EvidencePacketError",
    "SEMANTIC_ABSENCE_REASONS",
    "assess_conditional_transition",
    "assess_evidence",
    "build_evidence_packet",
    "validate_evidence_packet",
]
