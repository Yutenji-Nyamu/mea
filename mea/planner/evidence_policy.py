"""One compact typed evidence record for a completed production round.

The runtime records facts here; it does not decide what the Query means, rank
future experiments, or author a Plan action. The Plan Agent receives the same
policy, Rule, VQA, simulator, and typed planning facts persisted for the round.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping


class RoundEvidenceError(ValueError):
    """Raised when one production RoundEvidence record is malformed."""


_ROUND_KEYS = {
    "schema_version",
    "round_id",
    "candidate_id",
    "planning_observation",
    "policy",
    "rule",
    "vqa",
    "outcome_semantics",
    "scene_change",
}
_POLICY_KEYS = {
    "success_rate",
    "metric",
    "authority",
    "official_equivalent",
    "execution_scope",
    "seeds",
}
_RULE_KEYS = {
    "requested",
    "status",
    "metric",
    "route",
    "source",
    "results",
}
_RULE_RESULT_KEYS = {
    "policy_name",
    "seed",
    "role",
    "metric",
    "value",
    "unit",
    "passed",
    "evidence_steps",
    "details",
}
_VQA_KEYS = {"required", "status", "evidence_conflict", "observation"}
_VQA_STATUSES = {"passed", "abstained", "failed", "skipped", "missing"}
_OUTCOME_STATUSES = {
    "official_only",
    "equivalent_agreement",
    "expected_semantic_extension",
    "conflict",
    "non_comparable",
}
_OFFICIAL_POLICY_AUTHORITIES = {
    "official_check_success",
    "official_check_success_reused",
}
_GENERATED_POLICY_AUTHORITY = "llm_generated_python_ast_validated"


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RoundEvidenceError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _optional_rate(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise RoundEvidenceError(f"{field} must be null or in [0, 1]")
    return float(value)


def _execution_vqa_required(round_plan: Mapping[str, Any]) -> bool:
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
    if isinstance(vqa_need, Mapping):
        return vqa_need.get("requested") is True
    return bool(round_plan.get("vqa_phenomenon_ids"))


def _rule_requested(round_plan: Mapping[str, Any]) -> bool:
    semantic_needs = round_plan.get("semantic_need_execution")
    rule_need = (
        semantic_needs.get("rule_tool")
        if isinstance(semantic_needs, Mapping)
        else None
    )
    if isinstance(rule_need, Mapping):
        return rule_need.get("requested") is True
    requested = round_plan.get("observations")
    return bool(
        isinstance(requested, list)
        and "planned_tool" in requested
    )


def _rule_results(planned_tool: Mapping[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    episodes = planned_tool.get("episodes")
    if not isinstance(episodes, list):
        return results
    for raw in episodes:
        if not isinstance(raw, Mapping):
            continue
        results.append(
            {
                "policy_name": raw.get("policy_name"),
                "seed": raw.get("seed"),
                "role": raw.get("role"),
                "metric": raw.get("metric"),
                "value": deepcopy(raw.get("value")),
                "unit": raw.get("unit"),
                "passed": raw.get("passed"),
                "evidence_steps": deepcopy(raw.get("evidence_steps") or []),
                "details": deepcopy(raw.get("details") or {}),
            }
        )
    return results


def validate_round_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the single evidence schema consumed by the Plan runtime."""

    if not isinstance(value, Mapping) or set(value) != _ROUND_KEYS:
        raise RoundEvidenceError(
            f"RoundEvidence fields must be exactly {sorted(_ROUND_KEYS)}"
        )
    evidence = deepcopy(dict(value))
    if evidence.get("schema_version") != 1:
        raise RoundEvidenceError("RoundEvidence.schema_version must be 1")
    evidence["round_id"] = _text(evidence.get("round_id"), "round_id")
    evidence["candidate_id"] = _text(
        evidence.get("candidate_id"), "candidate_id"
    )

    planning = evidence.get("planning_observation")
    if planning is not None:
        if not isinstance(planning, Mapping):
            raise RoundEvidenceError(
                "planning_observation must be null or an object"
            )
        if (
            planning.get("policy_rollouts_started") != 0
            or planning.get("policy_sample_count") != 0
        ):
            raise RoundEvidenceError(
                "planning_observation must describe a typed N=0 boundary"
            )
        evidence["planning_observation"] = deepcopy(dict(planning))

    policy = evidence.get("policy")
    if not isinstance(policy, Mapping) or set(policy) != _POLICY_KEYS:
        raise RoundEvidenceError("RoundEvidence.policy fields changed")
    seeds = policy.get("seeds")
    if not isinstance(seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise RoundEvidenceError("RoundEvidence.policy.seeds must be int list")
    official_equivalent = policy.get("official_equivalent")
    if official_equivalent is not None and not isinstance(
        official_equivalent, bool
    ):
        raise RoundEvidenceError(
            "policy.official_equivalent must be bool or null"
        )
    normalized_policy = {
        "success_rate": _optional_rate(
            policy.get("success_rate"), "policy.success_rate"
        ),
        "metric": _optional_text(policy.get("metric"), "policy.metric"),
        "authority": _optional_text(
            policy.get("authority"), "policy.authority"
        ),
        "official_equivalent": official_equivalent,
        "execution_scope": _optional_text(
            policy.get("execution_scope"), "policy.execution_scope"
        ),
        "seeds": list(seeds),
    }
    success_rate = normalized_policy["success_rate"]
    metric = normalized_policy["metric"]
    authority = normalized_policy["authority"]
    execution_scope = normalized_policy["execution_scope"]
    if success_rate is not None and not seeds:
        raise RoundEvidenceError(
            "reported policy.success_rate requires at least one actual seed"
        )
    if metric is None:
        if authority is not None or official_equivalent is not None:
            raise RoundEvidenceError(
                "policy metric, authority, and official_equivalent must form "
                "one identity tuple"
            )
        if execution_scope not in {None, "not_executed"}:
            raise RoundEvidenceError(
                "policy without a metric must be unscoped or not_executed"
            )
        if success_rate is not None:
            raise RoundEvidenceError(
                "reported policy.success_rate requires a policy metric"
            )
    elif metric == "official_check_success":
        if (
            authority not in _OFFICIAL_POLICY_AUTHORITIES
            or official_equivalent is not True
        ):
            raise RoundEvidenceError(
                "official policy evidence requires official authority and "
                "official_equivalent=true"
            )
    elif metric == "generated_check_success":
        if (
            authority != _GENERATED_POLICY_AUTHORITY
            or not isinstance(official_equivalent, bool)
        ):
            raise RoundEvidenceError(
                "generated policy evidence requires its validated authority "
                "and a declared equivalence boolean"
            )
    else:
        raise RoundEvidenceError(f"unsupported policy metric: {metric!r}")
    if metric is not None and execution_scope in {None, "not_executed"}:
        raise RoundEvidenceError(
            "identified policy evidence requires an executed scope"
        )
    evidence["policy"] = normalized_policy

    rule = evidence.get("rule")
    if not isinstance(rule, Mapping) or set(rule) != _RULE_KEYS:
        raise RoundEvidenceError("RoundEvidence.rule fields changed")
    requested = rule.get("requested")
    if not isinstance(requested, bool):
        raise RoundEvidenceError("RoundEvidence.rule.requested must be bool")
    raw_results = rule.get("results")
    if not isinstance(raw_results, list):
        raise RoundEvidenceError("RoundEvidence.rule.results must be a list")
    results: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_results):
        if not isinstance(raw, Mapping) or set(raw) != _RULE_RESULT_KEYS:
            raise RoundEvidenceError(
                f"RoundEvidence.rule.results[{index}] fields changed"
            )
        seed = raw.get("seed")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise RoundEvidenceError(
                f"RoundEvidence.rule.results[{index}].seed is invalid"
            )
        passed = raw.get("passed")
        if passed is not None and not isinstance(passed, bool):
            raise RoundEvidenceError(
                f"RoundEvidence.rule.results[{index}].passed is invalid"
            )
        steps = raw.get("evidence_steps")
        details = raw.get("details")
        if not isinstance(steps, list) or not isinstance(details, Mapping):
            raise RoundEvidenceError(
                f"RoundEvidence.rule.results[{index}] evidence is invalid"
            )
        results.append(deepcopy(dict(raw)))
    normalized_rule = {
        "requested": requested,
        "status": _optional_text(rule.get("status"), "rule.status"),
        "metric": _optional_text(rule.get("metric"), "rule.metric"),
        "route": _optional_text(rule.get("route"), "rule.route"),
        "source": (
            deepcopy(dict(rule["source"]))
            if isinstance(rule.get("source"), Mapping)
            else None
        ),
        "results": results,
    }
    if requested:
        if normalized_rule["status"] is None:
            raise RoundEvidenceError("requested Rule evidence has no status")
    elif any(
        normalized_rule[field] is not None
        for field in ("status", "metric", "route", "source")
    ) or results:
        raise RoundEvidenceError(
            "an unrequested Rule must not carry execution evidence"
        )
    evidence["rule"] = normalized_rule

    vqa = evidence.get("vqa")
    if not isinstance(vqa, Mapping) or set(vqa) != _VQA_KEYS:
        raise RoundEvidenceError("RoundEvidence.vqa fields changed")
    if not isinstance(vqa.get("required"), bool):
        raise RoundEvidenceError("RoundEvidence.vqa.required must be bool")
    status = vqa.get("status")
    if status not in _VQA_STATUSES:
        raise RoundEvidenceError(
            f"unsupported RoundEvidence.vqa.status: {status!r}"
        )
    if not isinstance(vqa.get("evidence_conflict"), bool):
        raise RoundEvidenceError(
            "RoundEvidence.vqa.evidence_conflict must be bool"
        )
    evidence["vqa"] = deepcopy(dict(vqa))

    semantics = evidence.get("outcome_semantics")
    if not isinstance(semantics, Mapping):
        raise RoundEvidenceError("outcome_semantics must be an object")
    semantic_status = _text(
        semantics.get("status"), "outcome_semantics.status"
    )
    if semantic_status not in _OUTCOME_STATUSES:
        raise RoundEvidenceError(
            f"unsupported outcome_semantics.status: {semantic_status!r}"
        )
    if not isinstance(semantics.get("evidence_conflict"), bool):
        raise RoundEvidenceError(
            "outcome_semantics.evidence_conflict must be bool"
        )
    if semantics["evidence_conflict"] != (semantic_status == "conflict"):
        raise RoundEvidenceError(
            "outcome_semantics.evidence_conflict disagrees with status"
        )
    evidence["outcome_semantics"] = deepcopy(dict(semantics))
    evidence["outcome_semantics"]["status"] = semantic_status

    scene_change = evidence.get("scene_change")
    if scene_change is not None and not isinstance(scene_change, Mapping):
        raise RoundEvidenceError("scene_change must be null or an object")
    evidence["scene_change"] = (
        deepcopy(dict(scene_change))
        if isinstance(scene_change, Mapping)
        else None
    )
    return evidence


def build_round_evidence(
    round_plan: Mapping[str, Any],
    round_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build RoundEvidence once, after all runtime observations are attached."""

    if not isinstance(round_plan, Mapping) or not isinstance(
        round_summary, Mapping
    ):
        raise RoundEvidenceError("round plan and summary must be objects")
    round_id = _text(round_plan.get("round_id"), "round_plan.round_id")
    if round_summary.get("round_id") != round_id:
        raise RoundEvidenceError("round plan and summary ids disagree")
    candidate_id = _text(
        round_plan.get("candidate_id") or round_plan.get("template_id"),
        "round_plan.candidate_id",
    )
    observations = round_summary.get("observations")
    if not isinstance(observations, Mapping):
        raise RoundEvidenceError("round summary observations must be an object")
    policy_outcome = observations.get("policy_outcome")
    policy_outcome = (
        policy_outcome if isinstance(policy_outcome, Mapping) else {}
    )
    seeds = observations.get("actual_seeds")
    seeds = list(seeds) if isinstance(seeds, list) else []

    rule_requested = _rule_requested(round_plan)
    planned_tool = observations.get("planned_tool")
    planned_tool = (
        planned_tool if isinstance(planned_tool, Mapping) else {}
    )
    route_decision = planned_tool.get("route_decision")
    route_decision = (
        route_decision if isinstance(route_decision, Mapping) else {}
    )
    tool_request = round_plan.get("tool_request")
    tool_request = tool_request if isinstance(tool_request, Mapping) else {}
    rule_metric = (
        planned_tool.get("reference_tool")
        or route_decision.get("metric")
        or tool_request.get("metric")
    )
    rule_source = planned_tool.get("source")

    raw_vqa = observations.get("execution_vqa")
    raw_vqa = raw_vqa if isinstance(raw_vqa, Mapping) else {}
    semantics = observations.get("outcome_semantics")
    if not isinstance(semantics, Mapping):
        semantics = policy_outcome.get("outcome_semantics")
    if not isinstance(semantics, Mapping):
        raise RoundEvidenceError("round summary has no outcome_semantics")

    planning = observations.get("planning_observation")
    if planning is not None and not isinstance(planning, Mapping):
        raise RoundEvidenceError("planning_observation must be an object")
    scene_change = observations.get("scene_change")
    return validate_round_evidence(
        {
            "schema_version": 1,
            "round_id": round_id,
            "candidate_id": candidate_id,
            "planning_observation": (
                deepcopy(dict(planning))
                if isinstance(planning, Mapping)
                else None
            ),
            "policy": {
                "success_rate": observations.get("policy_success"),
                "metric": policy_outcome.get("metric"),
                "authority": policy_outcome.get("authority"),
                "official_equivalent": policy_outcome.get(
                    "official_equivalent"
                ),
                "execution_scope": policy_outcome.get("execution_scope"),
                "seeds": seeds,
            },
            "rule": {
                "requested": rule_requested,
                "status": planned_tool.get("status") if rule_requested else None,
                "metric": rule_metric if rule_requested else None,
                "route": (
                    planned_tool.get("route")
                    or route_decision.get("resolved_route")
                    or route_decision.get("route")
                )
                if rule_requested
                else None,
                "source": (
                    deepcopy(dict(rule_source))
                    if rule_requested and isinstance(rule_source, Mapping)
                    else None
                ),
                "results": (
                    _rule_results(planned_tool) if rule_requested else []
                ),
            },
            "vqa": {
                "required": _execution_vqa_required(round_plan),
                "status": raw_vqa.get("status", "missing"),
                "evidence_conflict": bool(
                    raw_vqa.get("evidence_conflict", False)
                ),
                "observation": deepcopy(raw_vqa.get("observation")),
            },
            "outcome_semantics": deepcopy(dict(semantics)),
            "scene_change": (
                deepcopy(dict(scene_change))
                if isinstance(scene_change, Mapping)
                else None
            ),
        }
    )


__all__ = [
    "RoundEvidenceError",
    "build_round_evidence",
    "validate_round_evidence",
]
