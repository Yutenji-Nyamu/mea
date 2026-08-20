"""Structured evidence scope for a completed Plan Agent answer."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


class AnswerScopeError(ValueError):
    """Raised when the structured answer scope is invalid."""


_SCOPE_KEYS = {
    "schema_version",
    "sample_count",
    "seeds",
    "tested_candidate_ids",
    "untested_candidate_ids",
    "evidence_conflict",
    "termination",
    "claim_verdict",
}
_TERMINATIONS = {
    "agent_stop",
    "budget_exhausted",
    "continue",
    "control_not_passed",
    "pipeline_invalid",
    "unknown",
}


def _dedupe(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _collect_seeds(evidence: Mapping[str, Any]) -> list[int]:
    raw: list[Any] = []
    for round_evidence in evidence.get("rounds", []):
        if not isinstance(round_evidence, Mapping):
            continue
        policy = round_evidence.get("policy")
        if isinstance(policy, Mapping) and isinstance(
            policy.get("seeds"), list
        ):
            raw.extend(policy["seeds"])
    seeds: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value not in seeds:
            seeds.append(value)
    return seeds


def _sample_count(evidence: Mapping[str, Any], seeds: list[int]) -> int | None:
    value = evidence.get("total_policy_episodes")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return len(seeds) if isinstance(evidence.get("rounds"), list) else None


def _execution_conflict(evidence: Mapping[str, Any]) -> bool:
    assessment = _query_assessment(evidence)
    if assessment is not None:
        candidates = assessment.get("conflict_candidate_ids")
        if isinstance(candidates, list) and bool(candidates):
            return True
    return any(
        isinstance(item, Mapping)
        and (
            (
                isinstance(item.get("vqa"), Mapping)
                and item["vqa"].get("evidence_conflict") is True
            )
            or (
                isinstance(item.get("outcome_semantics"), Mapping)
                and item["outcome_semantics"].get("evidence_conflict") is True
            )
        )
        for item in evidence.get("rounds", [])
    )


def _query_assessment(evidence: Mapping[str, Any]) -> Mapping[str, Any] | None:
    session = evidence.get("plan_agent_session")
    if isinstance(session, Mapping):
        value = session.get("assessment")
        if isinstance(value, Mapping):
            return value
    return None


def _tested_candidates(evidence: Mapping[str, Any]) -> list[str]:
    assessment = _query_assessment(evidence)
    if assessment is not None and isinstance(
        assessment.get("observed_candidate_ids"), list
    ):
        return _dedupe(
            [
                str(item)
                for item in assessment["observed_candidate_ids"]
                if isinstance(item, str) and item
            ]
        )
    result: list[str] = []
    for item in evidence.get("rounds", []):
        if not isinstance(item, Mapping):
            continue
        candidate = item.get("candidate_id")
        if candidate is not None and candidate not in result:
            result.append(str(candidate))
    return result


def _untested_candidates(evidence: Mapping[str, Any]) -> list[str]:
    assessment = _query_assessment(evidence)
    if assessment is not None and isinstance(
        assessment.get("untested_candidate_ids"), list
    ):
        return _dedupe(
            [
                str(item)
                for item in assessment["untested_candidate_ids"]
                if isinstance(item, str) and item
            ]
        )
    return []


def _termination(evidence: Mapping[str, Any]) -> tuple[str, str | None]:
    assessment = _query_assessment(evidence)
    if assessment is not None:
        reason = assessment.get("stop_reason")
        verdict = assessment.get("claim_verdict")
        sufficient = assessment.get("evidence_sufficient")
        should_stop = assessment.get("should_stop")
        valid_query_stop = (
            reason == "agent_stop"
            and should_stop is True
            and (
                (
                    sufficient is True
                    and verdict in {"supported", "refuted"}
                )
                or (
                    sufficient is False
                    and verdict == "inconclusive"
                )
            )
        ) or (
            reason == "budget_exhausted"
            and sufficient is False
            and should_stop is True
        ) or (
            reason == "continue"
            and sufficient is False
            and should_stop is False
        ) or (
            isinstance(reason, str)
            and reason.startswith("control_baseline_")
            and sufficient is False
            and should_stop is True
        )
        if valid_query_stop:
            termination = (
                "control_not_passed"
                if isinstance(reason, str)
                and reason.startswith("control_baseline_")
                else str(reason)
            )
            return termination, str(verdict) if verdict is not None else None
        raise AnswerScopeError(
            "query sufficiency assessment has an inconsistent stop verdict"
        )
    policy_rounds = [
        item
        for item in evidence.get("rounds", [])
        if isinstance(item, Mapping)
        and item.get("planning_observation") is None
    ]
    if policy_rounds and any(
        not isinstance(item.get("pipeline"), Mapping)
        or item["pipeline"].get("passed") is not True
        for item in policy_rounds
    ):
        return "pipeline_invalid", None
    plan = evidence.get("plan")
    if isinstance(plan, Mapping):
        budget_remaining = plan.get("round_budget_remaining")
        planning_state = plan.get("planning_state")
        if (
            budget_remaining == 0
            and isinstance(planning_state, str)
            and planning_state.startswith("stopped_after_round_")
        ):
            return "budget_exhausted", None
    return "unknown", None


def build_answer_scope(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Project the evidence facts that bound the final answer."""

    if not isinstance(evidence, Mapping):
        raise AnswerScopeError("evidence must be an object")
    seeds = _collect_seeds(evidence)
    sample_count = _sample_count(evidence, seeds)
    tested = _tested_candidates(evidence)
    untested = _untested_candidates(evidence)
    conflict = _execution_conflict(evidence)
    termination, claim_verdict = _termination(evidence)
    return validate_answer_scope(
        {
            "schema_version": 1,
            "sample_count": sample_count,
            "seeds": seeds,
            "tested_candidate_ids": tested,
            "untested_candidate_ids": untested,
            "evidence_conflict": conflict,
            "termination": termination,
            "claim_verdict": claim_verdict,
        }
    )


def validate_answer_scope(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _SCOPE_KEYS
    ):
        raise AnswerScopeError(
            "AnswerScope fields must match the current schema"
        )
    scope = deepcopy(dict(value))
    if scope.get("schema_version") != 1:
        raise AnswerScopeError("AnswerScope schema_version must be 1")
    count = scope.get("sample_count")
    if count is not None and (
        isinstance(count, bool) or not isinstance(count, int) or count < 0
    ):
        raise AnswerScopeError("sample_count must be a non-negative integer or null")
    seeds = scope.get("seeds")
    if (
        not isinstance(seeds, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise AnswerScopeError("seeds must be a unique integer list")
    for field in (
        "tested_candidate_ids",
        "untested_candidate_ids",
    ):
        values = scope.get(field)
        if (
            not isinstance(values, list)
            or any(not isinstance(item, str) or not item for item in values)
            or len(values) != len(set(values))
        ):
            raise AnswerScopeError(f"{field} must be a unique string list")
    if set(scope["tested_candidate_ids"]) & set(scope["untested_candidate_ids"]):
        raise AnswerScopeError("tested and untested candidates must be disjoint")
    if not isinstance(scope.get("evidence_conflict"), bool):
        raise AnswerScopeError("evidence_conflict must be boolean")
    if scope.get("termination") not in _TERMINATIONS:
        raise AnswerScopeError(
            f"termination must be one of {sorted(_TERMINATIONS)}"
        )
    verdict = scope.get("claim_verdict")
    if verdict is not None and (not isinstance(verdict, str) or not verdict):
        raise AnswerScopeError("claim_verdict must be a non-empty string or null")
    return scope


__all__ = [
    "AnswerScopeError",
    "build_answer_scope",
    "validate_answer_scope",
]
