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


_EVIDENCE_PACKET_V1_KEYS = {
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
_EVIDENCE_PACKET_V2_KEYS = _EVIDENCE_PACKET_V1_KEYS | {
    "valid_for_planning",
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
_VQA_STATUSES = {"passed", "abstained", "failed", "skipped", "missing"}
_EVIDENCE_STRENGTHS = {
    "sufficient",
    "uncertain",
    "conflicting",
    "pipeline_invalid",
}
_EVIDENCE_AGGREGATE_V1_KEYS = {
    "schema_version",
    "round_id",
    "execution_id",
    "pipeline",
    "policy",
    "rule",
    "tool",
    "vqa",
    "outcome_semantics",
    "coverage",
    "evidence_conflict",
    "evidence_strength",
    "reason_codes",
}
_EVIDENCE_AGGREGATE_V2_KEYS = _EVIDENCE_AGGREGATE_V1_KEYS | {
    "valid_for_planning",
}
_TOOL_EVIDENCE_KEYS = {
    "requested",
    "status",
    "metric",
    "route",
    "provider_called",
}
_OUTCOME_SEMANTICS_KEYS = {"status", "evidence_conflict"}
_COVERAGE_KEYS = {
    "expected_policy_episodes",
    "observed_policy_episodes",
    "rule_complete",
    "tool_required",
    "tool_complete",
    "vqa_required",
    "vqa_observed",
    "intent_required",
    "intent_complete",
    "complete",
}


def _external_uncertainty_from_reason_codes(reason_codes: list[str]) -> bool:
    """Return whether v1 reason codes carry newer Tool/intent uncertainty.

    EvidencePacket v1 has no typed Tool or original-intent coverage fields.
    EvidenceAggregate therefore preserves those two fail-closed conditions in
    the compatibility packet's reason codes.
    """

    return "original_intent_incomplete" in reason_codes or any(
        reason.startswith("planned_tool_") for reason in reason_codes
    )


def _valid_pre_policy_planning_observation(value: Any) -> bool:
    """Recognize the small typed N=0 boundary that Plan may learn from."""

    return bool(
        isinstance(value, Mapping)
        and value.get("kind")
        in {
            "candidate_unexecutable",
            "expert_oracle_unavailable",
            "taskgen_materialization_failed",
        }
        and value.get("policy_rollouts_started") == 0
        and value.get("policy_sample_count") == 0
    )


def _expected_evidence_strength(
    *,
    pipeline_passed: bool,
    valid_for_planning: bool | None = None,
    evidence_conflict: bool,
    rule_complete: bool,
    vqa_required: bool,
    vqa_status: str,
    additional_uncertainty: bool = False,
) -> str:
    # EvidencePacket v1 had no independent planning-validity bit, so its
    # historical contract treated every failed pipeline as invalid evidence.
    # Version 2 preserves that reader behavior while allowing a typed,
    # zero-rollout pre-policy failure to be useful to Plan without rewriting
    # the execution fact ``pipeline.passed``.
    planning_valid = (
        pipeline_passed
        if valid_for_planning is None
        else valid_for_planning
    )
    if not planning_valid:
        return "pipeline_invalid"
    if not pipeline_passed:
        return "uncertain"
    if evidence_conflict:
        return "conflicting"
    if (
        (vqa_required and vqa_status != "passed")
        or not rule_complete
        or additional_uncertainty
    ):
        return "uncertain"
    return "sufficient"


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
    if isinstance(requested, list) and any(
        item in requested for item in ("execution_vqa", "dynamic_vqa")
    ):
        return True
    semantic_needs = round_plan.get("semantic_need_execution")
    vqa_need = (
        semantic_needs.get("vqa_tool")
        if isinstance(semantic_needs, Mapping)
        else None
    )
    if isinstance(vqa_need, Mapping) and vqa_need.get("requested") is True:
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


def validate_evidence_aggregate(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the unified Rule/Tool/VQA evidence consumed by planning.

    ``EvidencePacket`` remains the compatibility projection used by older
    planners.  New round summaries persist this richer aggregate so every
    planner observes the same conflict and episode-coverage decision rather
    than independently re-reading Rule, Tool, and VQA fields.
    """

    if not isinstance(value, Mapping):
        raise EvidencePacketError(
            "EvidenceAggregate must be an object"
        )
    aggregate = deepcopy(dict(value))
    schema_version = aggregate.get("schema_version")
    expected_keys = (
        _EVIDENCE_AGGREGATE_V1_KEYS
        if schema_version == 1
        else _EVIDENCE_AGGREGATE_V2_KEYS
        if schema_version == 2
        else None
    )
    if expected_keys is None:
        raise EvidencePacketError(
            "EvidenceAggregate.schema_version must be 1 or 2"
        )
    if set(aggregate) != expected_keys:
        raise EvidencePacketError(
            "EvidenceAggregate fields must be exactly "
            f"{sorted(expected_keys)}"
        )
    valid_for_planning = (
        aggregate.get("valid_for_planning")
        if schema_version == 2
        else None
    )
    if schema_version == 2 and not isinstance(valid_for_planning, bool):
        raise EvidencePacketError(
            "EvidenceAggregate.valid_for_planning must be boolean"
        )
    for field in ("round_id", "execution_id"):
        if not isinstance(aggregate.get(field), str) or not aggregate[field]:
            raise EvidencePacketError(
                f"EvidenceAggregate.{field} must be non-empty"
            )

    pipeline = aggregate.get("pipeline")
    if not isinstance(pipeline, Mapping) or set(pipeline) != _PIPELINE_KEYS:
        raise EvidencePacketError("EvidenceAggregate.pipeline fields changed")
    if not isinstance(pipeline.get("passed"), bool):
        raise EvidencePacketError(
            "EvidenceAggregate.pipeline.passed must be boolean"
        )
    if (
        schema_version == 2
        and pipeline["passed"] is True
        and valid_for_planning is not True
    ):
        raise EvidencePacketError(
            "a passed pipeline must be valid for planning"
        )
    failure_stage = pipeline.get("failure_stage")
    if failure_stage is not None and (
        not isinstance(failure_stage, str) or not failure_stage
    ):
        raise EvidencePacketError(
            "EvidenceAggregate.pipeline.failure_stage must be null or non-empty"
        )

    policy = aggregate.get("policy")
    if not isinstance(policy, Mapping) or set(policy) != _POLICY_KEYS:
        raise EvidencePacketError("EvidenceAggregate.policy fields changed")
    if not isinstance(policy.get("reported"), bool):
        raise EvidencePacketError(
            "EvidenceAggregate.policy.reported must be boolean"
        )
    success_rate = policy.get("success_rate")
    if success_rate is not None and (
        isinstance(success_rate, bool)
        or not isinstance(success_rate, (int, float))
        or not math.isfinite(float(success_rate))
        or not 0.0 <= float(success_rate) <= 1.0
    ):
        raise EvidencePacketError(
            "EvidenceAggregate.policy.success_rate must be null or in [0, 1]"
        )
    if policy["reported"] != (success_rate is not None):
        raise EvidencePacketError(
            "EvidenceAggregate.policy.reported disagrees with success_rate"
        )

    # Reuse the mature EvidencePacket validator for the unchanged Rule shape.
    rule = aggregate.get("rule")
    vqa = aggregate.get("vqa")
    if not isinstance(vqa, Mapping) or set(vqa) != _VQA_KEYS:
        raise EvidencePacketError("EvidenceAggregate.vqa fields changed")
    if not isinstance(vqa.get("required"), bool):
        raise EvidencePacketError(
            "EvidenceAggregate.vqa.required must be boolean"
        )
    if vqa.get("status") not in _VQA_STATUSES or not isinstance(
        vqa.get("evidence_conflict"), bool
    ):
        raise EvidencePacketError("EvidenceAggregate.vqa fields are invalid")
    compatibility_packet = {
        "schema_version": schema_version,
        "round_id": aggregate["round_id"],
        "template_id": aggregate["execution_id"],
        "pipeline": deepcopy(dict(pipeline)),
        "policy": deepcopy(dict(policy)),
        "rule": deepcopy(dict(rule)) if isinstance(rule, Mapping) else rule,
        "vqa": deepcopy(dict(vqa)),
        "evidence_strength": _expected_evidence_strength(
            pipeline_passed=pipeline["passed"],
            valid_for_planning=valid_for_planning,
            evidence_conflict=vqa["evidence_conflict"],
            rule_complete=bool((rule or {}).get("complete")),
            vqa_required=vqa["required"],
            vqa_status=vqa["status"],
        ),
        "reason_codes": [],
    }
    if schema_version == 2:
        compatibility_packet["valid_for_planning"] = valid_for_planning
    validate_evidence_packet(compatibility_packet)

    tool = aggregate.get("tool")
    if not isinstance(tool, Mapping) or set(tool) != _TOOL_EVIDENCE_KEYS:
        raise EvidencePacketError("EvidenceAggregate.tool fields changed")
    if not isinstance(tool.get("requested"), bool):
        raise EvidencePacketError(
            "EvidenceAggregate.tool.requested must be boolean"
        )
    for field in ("status", "metric", "route"):
        item = tool.get(field)
        if item is not None and (not isinstance(item, str) or not item):
            raise EvidencePacketError(
                f"EvidenceAggregate.tool.{field} must be null or non-empty"
            )
    provider_called = tool.get("provider_called")
    if provider_called is not None and not isinstance(provider_called, bool):
        raise EvidencePacketError(
            "EvidenceAggregate.tool.provider_called must be boolean or null"
        )

    semantics = aggregate.get("outcome_semantics")
    if (
        not isinstance(semantics, Mapping)
        or set(semantics) != _OUTCOME_SEMANTICS_KEYS
    ):
        raise EvidencePacketError(
            "EvidenceAggregate.outcome_semantics fields changed"
        )
    if semantics.get("status") is not None and (
        not isinstance(semantics["status"], str) or not semantics["status"]
    ):
        raise EvidencePacketError(
            "EvidenceAggregate.outcome_semantics.status must be null or non-empty"
        )
    if not isinstance(semantics.get("evidence_conflict"), bool):
        raise EvidencePacketError(
            "EvidenceAggregate.outcome_semantics.evidence_conflict must be boolean"
        )

    coverage = aggregate.get("coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != _COVERAGE_KEYS:
        raise EvidencePacketError("EvidenceAggregate.coverage fields changed")
    for field in ("expected_policy_episodes", "observed_policy_episodes"):
        item = coverage.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise EvidencePacketError(
                f"EvidenceAggregate.coverage.{field} must be non-negative"
            )
    for field in (
        "rule_complete",
        "tool_required",
        "tool_complete",
        "vqa_required",
        "vqa_observed",
        "intent_required",
        "intent_complete",
        "complete",
    ):
        if not isinstance(coverage.get(field), bool):
            raise EvidencePacketError(
                f"EvidenceAggregate.coverage.{field} must be boolean"
            )
    if (
        not coverage["intent_required"]
        and coverage["intent_complete"]
    ):
        raise EvidencePacketError(
            "EvidenceAggregate intent cannot be complete when it is not required"
        )
    expected_coverage = {
        "expected_policy_episodes": rule["expected_policy_episodes"],
        "observed_policy_episodes": rule["observed_policy_episodes"],
        "rule_complete": rule["complete"],
        "tool_required": tool["requested"],
        "tool_complete": (
            not tool["requested"] or tool["status"] == "passed"
        ),
        "vqa_required": vqa["required"],
        "vqa_observed": vqa["status"] == "passed",
        "intent_required": coverage["intent_required"],
        "intent_complete": coverage["intent_complete"],
        "complete": bool(
            rule["complete"]
            and (
                not coverage["tool_required"]
                or coverage["tool_complete"]
            )
            and (not vqa["required"] or vqa["status"] == "passed")
            and (
                not coverage["intent_required"]
                or coverage["intent_complete"]
            )
        ),
    }
    if dict(coverage) != expected_coverage:
        raise EvidencePacketError(
            "EvidenceAggregate.coverage disagrees with Rule/VQA evidence"
        )

    conflict = aggregate.get("evidence_conflict")
    if not isinstance(conflict, bool):
        raise EvidencePacketError(
            "EvidenceAggregate.evidence_conflict must be boolean"
        )
    expected_conflict = bool(
        vqa["evidence_conflict"] or semantics["evidence_conflict"]
    )
    if conflict != expected_conflict:
        raise EvidencePacketError(
            "EvidenceAggregate conflict disagrees with VQA/outcome semantics"
        )
    expected_strength = _expected_evidence_strength(
        pipeline_passed=pipeline["passed"],
        valid_for_planning=valid_for_planning,
        evidence_conflict=conflict,
        rule_complete=rule["complete"],
        vqa_required=vqa["required"],
        vqa_status=vqa["status"],
        additional_uncertainty=not coverage["complete"],
    )
    if aggregate.get("evidence_strength") != expected_strength:
        raise EvidencePacketError(
            "EvidenceAggregate.evidence_strength disagrees with its evidence"
        )
    reasons = aggregate.get("reason_codes")
    if (
        not isinstance(reasons, list)
        or any(not isinstance(item, str) or not item for item in reasons)
        or len(reasons) != len(set(reasons))
    ):
        raise EvidencePacketError(
            "EvidenceAggregate.reason_codes must be a unique string list"
        )
    return aggregate


def build_evidence_aggregate(
    round_plan: Mapping[str, Any],
    round_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse Rule, planned Tool, VQA, conflicts, and coverage for Planner."""

    if not isinstance(round_plan, Mapping) or not isinstance(
        round_summary, Mapping
    ):
        raise EvidencePacketError(
            "EvidenceAggregate requires round plan and summary objects"
        )
    round_id = str(round_plan.get("round_id") or "")
    if round_summary.get("round_id") != round_id:
        raise EvidencePacketError(
            "EvidenceAggregate round plan and summary ids disagree"
        )
    execution_id = str(
        round_plan.get("candidate_id")
        or round_plan.get("template_id")
        or ""
    )
    if not execution_id:
        raise EvidencePacketError(
            "EvidenceAggregate requires candidate_id or template_id"
        )
    observations = round_summary.get("observations") or {}
    if not isinstance(observations, Mapping):
        raise EvidencePacketError(
            "EvidenceAggregate summary observations must be an object"
        )
    quality = _aggregate_quality(dict(round_plan), dict(round_summary))
    raw_vqa = observations.get("execution_vqa")
    vqa = raw_vqa if isinstance(raw_vqa, Mapping) else {}
    if vqa and not isinstance(vqa.get("evidence_conflict"), bool):
        raise EvidencePacketError(
            "EvidenceAggregate execution VQA conflict must be boolean"
        )
    vqa_required = _execution_vqa_required(round_plan)
    vqa_status = vqa.get("status", "missing")
    if vqa_status not in _VQA_STATUSES:
        raise EvidencePacketError(
            "EvidenceAggregate execution VQA status is invalid"
        )
    vqa_view = {
        "required": vqa_required,
        "status": vqa_status,
        "evidence_conflict": bool(vqa.get("evidence_conflict", False)),
    }

    planned_tool = observations.get("planned_tool")
    planned_tool = planned_tool if isinstance(planned_tool, Mapping) else {}
    route_decision = planned_tool.get("route_decision")
    route_decision = (
        route_decision if isinstance(route_decision, Mapping) else {}
    )
    semantic_needs = round_plan.get("semantic_need_execution")
    rule_need = (
        semantic_needs.get("rule_tool")
        if isinstance(semantic_needs, Mapping)
        else None
    )
    tool_requested = (
        rule_need.get("requested") is True
        if isinstance(rule_need, Mapping)
        else bool(
            "planned_tool" in (round_plan.get("observations") or [])
            or round_plan.get("open_tool_request_deferred")
            or round_plan.get("tool_request")
        )
    )
    tool_metric = (
        (
            route_decision.get("metric")
            or planned_tool.get("reference_tool")
            or (round_plan.get("tool_request") or {}).get("metric")
        )
        if tool_requested
        else None
    )
    tool_view = {
        "requested": tool_requested,
        "status": (
            (
                str(planned_tool["status"])
                if isinstance(planned_tool.get("status"), str)
                and planned_tool["status"]
                else "missing"
            )
            if tool_requested
            else None
        ),
        "metric": str(tool_metric) if tool_metric else None,
        "route": (
            (
                str(
                    planned_tool.get("route")
                    or route_decision.get("resolved_route")
                )
                if (
                    planned_tool.get("route")
                    or route_decision.get("resolved_route")
                )
                else None
            )
            if tool_requested
            else None
        ),
        "provider_called": (
            route_decision.get("provider_called")
            if tool_requested
            and isinstance(route_decision.get("provider_called"), bool)
            else None
        ),
    }

    raw_semantics = observations.get("outcome_semantics")
    raw_semantics = (
        raw_semantics if isinstance(raw_semantics, Mapping) else {}
    )
    semantics_view = {
        "status": (
            str(raw_semantics["status"])
            if isinstance(raw_semantics.get("status"), str)
            and raw_semantics["status"]
            else None
        ),
        "evidence_conflict": bool(
            raw_semantics.get("evidence_conflict")
            or raw_semantics.get("status") == "conflict"
        ),
    }
    pipeline_passed = round_summary.get("pipeline_passed")
    if not isinstance(pipeline_passed, bool):
        raise EvidencePacketError(
            "EvidenceAggregate summary pipeline_passed must be boolean"
        )
    planning_observation = observations.get("planning_observation")
    planning_evidence_valid = _valid_pre_policy_planning_observation(
        planning_observation
    )
    # ``round_summary.pipeline_passed`` is an execution fact and must never be
    # upgraded merely because the failure is useful to Plan.  Version 2 keeps
    # planning validity as an independent evidence property.
    valid_for_planning = pipeline_passed or planning_evidence_valid
    failure_stage = round_summary.get("failure_stage")
    if isinstance(failure_stage, str):
        failure_stage = failure_stage.strip() or None
    success_rate = _policy_success_rate(round_summary)
    implementation_trace = observations.get("implementation_trace")
    intent_required = isinstance(implementation_trace, Mapping)
    intent_complete = bool(
        intent_required
        and implementation_trace.get("coverage_status") == "complete"
    )
    coverage = {
        "expected_policy_episodes": quality["expected_policy_episodes"],
        "observed_policy_episodes": quality["observed_policy_episodes"],
        "rule_complete": quality["complete"],
        "tool_required": tool_requested,
        "tool_complete": (
            not tool_requested or tool_view["status"] == "passed"
        ),
        "vqa_required": vqa_required,
        "vqa_observed": vqa_status == "passed",
        "intent_required": intent_required,
        "intent_complete": intent_complete,
        "complete": bool(
            quality["complete"]
            and (not tool_requested or tool_view["status"] == "passed")
            and (not vqa_required or vqa_status == "passed")
            and (not intent_required or intent_complete)
        ),
    }
    conflict = bool(
        vqa_view["evidence_conflict"]
        or semantics_view["evidence_conflict"]
    )
    if not valid_for_planning:
        strength = "pipeline_invalid"
        reasons = ["latest_pipeline_failed"]
    elif not pipeline_passed:
        strength = "uncertain"
        reasons = [
            str(
                planning_observation.get("reason_code")
                or "pre_policy_planning_observation"
            )
        ]
    elif conflict:
        strength = "conflicting"
        reasons = [
            *(
                ["execution_vqa_conflicts_with_numeric_evidence"]
                if vqa_view["evidence_conflict"]
                else []
            ),
            *(
                ["outcome_semantics_conflict"]
                if semantics_view["evidence_conflict"]
                else []
            ),
        ]
    elif vqa_required and vqa_status != "passed":
        strength = "uncertain"
        reasons = [f"execution_vqa_{vqa_status}"]
    elif tool_requested and tool_view["status"] != "passed":
        strength = "uncertain"
        reasons = [f"planned_tool_{tool_view['status'] or 'missing'}"]
    elif intent_required and not intent_complete:
        strength = "uncertain"
        reasons = ["original_intent_incomplete"]
    elif not quality["complete"]:
        strength = "uncertain"
        reasons = list(quality["reasons"])
    else:
        strength = "sufficient"
        reasons = []
    return validate_evidence_aggregate(
        {
            "schema_version": 2,
            "round_id": round_id,
            "execution_id": execution_id,
            "pipeline": {
                "passed": pipeline_passed,
                "failure_stage": failure_stage,
            },
            "valid_for_planning": valid_for_planning,
            "policy": {
                "success_rate": success_rate,
                "reported": success_rate is not None,
            },
            "rule": quality,
            "tool": tool_view,
            "vqa": vqa_view,
            "outcome_semantics": semantics_view,
            "coverage": coverage,
            "evidence_conflict": conflict,
            "evidence_strength": strength,
            "reason_codes": list(dict.fromkeys(reasons)),
        }
    )


def validate_evidence_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the compact Rule/VQA/policy evidence passed to Plan."""

    if not isinstance(value, Mapping):
        raise EvidencePacketError(
            "EvidencePacket must be an object"
        )
    packet = deepcopy(dict(value))
    schema_version = packet.get("schema_version")
    expected_keys = (
        _EVIDENCE_PACKET_V1_KEYS
        if schema_version == 1
        else _EVIDENCE_PACKET_V2_KEYS
        if schema_version == 2
        else None
    )
    if expected_keys is None:
        raise EvidencePacketError(
            "EvidencePacket.schema_version must be 1 or 2"
        )
    if set(packet) != expected_keys:
        raise EvidencePacketError(
            "EvidencePacket fields must be exactly "
            f"{sorted(expected_keys)}"
        )
    valid_for_planning = (
        packet.get("valid_for_planning") if schema_version == 2 else None
    )
    if schema_version == 2 and not isinstance(valid_for_planning, bool):
        raise EvidencePacketError(
            "EvidencePacket.valid_for_planning must be boolean"
        )
    for field in ("round_id", "template_id"):
        if not isinstance(packet.get(field), str) or not packet[field]:
            raise EvidencePacketError(f"EvidencePacket.{field} must be non-empty")

    pipeline = packet.get("pipeline")
    if not isinstance(pipeline, Mapping) or set(pipeline) != _PIPELINE_KEYS:
        raise EvidencePacketError("EvidencePacket.pipeline fields changed")
    if not isinstance(pipeline.get("passed"), bool):
        raise EvidencePacketError("EvidencePacket.pipeline.passed must be boolean")
    if (
        schema_version == 2
        and pipeline["passed"] is True
        and valid_for_planning is not True
    ):
        raise EvidencePacketError(
            "a passed pipeline must be valid for planning"
        )
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
    expected_strength = _expected_evidence_strength(
        pipeline_passed=pipeline["passed"],
        valid_for_planning=valid_for_planning,
        evidence_conflict=vqa["evidence_conflict"],
        rule_complete=rule["complete"],
        vqa_required=vqa["required"],
        vqa_status=vqa["status"],
        additional_uncertainty=_external_uncertainty_from_reason_codes(
            reasons
        ),
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
    observations = latest.get("observations") or {}
    if not isinstance(observations, Mapping):
        raise EvidencePacketError("observation.observations must be an object")
    raw_evidence_aggregate = observations.get("evidence_aggregate")
    if raw_evidence_aggregate is not None:
        if not isinstance(raw_evidence_aggregate, Mapping):
            raise EvidencePacketError(
                "observation.evidence_aggregate must be an object"
            )
        unified = validate_evidence_aggregate(raw_evidence_aggregate)
        execution_id = str(
            latest_plan.get("candidate_id")
            or latest_plan.get("template_id")
            or ""
        )
        if unified["round_id"] != str(round_id or ""):
            raise EvidencePacketError(
                "EvidenceAggregate.round_id does not match the latest plan"
            )
        if unified["execution_id"] != execution_id:
            raise EvidencePacketError(
                "EvidenceAggregate.execution_id does not match the latest plan"
            )
        # Project the aggregate into the compact packet expected by existing
        # planner callers.  Version 1 remains readable; new aggregates retain
        # the independent planning-validity bit in a version 2 packet.
        compatibility_packet = {
            "schema_version": unified["schema_version"],
            "round_id": unified["round_id"],
            "template_id": unified["execution_id"],
            "pipeline": deepcopy(unified["pipeline"]),
            "policy": deepcopy(unified["policy"]),
            "rule": deepcopy(unified["rule"]),
            "vqa": {
                **deepcopy(unified["vqa"]),
                "evidence_conflict": unified["evidence_conflict"],
            },
            "evidence_strength": unified["evidence_strength"],
            "reason_codes": deepcopy(unified["reason_codes"]),
        }
        if unified["schema_version"] == 2:
            compatibility_packet["valid_for_planning"] = unified[
                "valid_for_planning"
            ]
        return validate_evidence_packet(compatibility_packet)

    quality = _aggregate_quality(dict(latest_plan), dict(latest))
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
            "observation.execution_vqa.status must be passed, abstained, "
            "failed, skipped, or missing"
        )
    planning_observation = observations.get("planning_observation")
    planning_evidence_valid = _valid_pre_policy_planning_observation(
        planning_observation
    )
    valid_for_planning = pipeline_passed or planning_evidence_valid
    if not valid_for_planning:
        strength = "pipeline_invalid"
        reasons = ["latest_pipeline_failed"]
    elif not pipeline_passed:
        strength = "uncertain"
        reasons = [
            str(
                planning_observation.get("reason_code")
                or "pre_policy_planning_observation"
            )
        ]
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
        "schema_version": 2,
        "round_id": str(round_id or ""),
        "template_id": str(
            latest_plan.get("candidate_id")
            or latest_plan.get("template_id")
            or ""
        ),
        "pipeline": {
            "passed": pipeline_passed,
            "failure_stage": failure_stage,
        },
        "valid_for_planning": valid_for_planning,
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
    "build_evidence_aggregate",
    "build_evidence_packet",
    "validate_evidence_aggregate",
    "validate_evidence_packet",
]
