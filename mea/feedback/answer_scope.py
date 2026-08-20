"""Deterministic answer scope and fail-closed limitation projection."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


class AnswerScopeError(ValueError):
    """Raised when final feedback omits an evidence-required limitation."""


_SCOPE_KEYS = {
    "schema_version",
    "sample_count",
    "seeds",
    "tested_candidate_ids",
    "untested_candidate_ids",
    "evidence_conflict",
    "termination",
    "claim_verdict",
    "required_limitations",
}
_LIMITATION_KEYS = {"code", "text"}
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


def _canonical_limitations(
    *,
    sample_count: int | None,
    seeds: list[int],
    untested: list[str],
    conflict: bool,
    termination: str,
) -> list[dict[str, str]]:
    limitations = [
        {
            "code": "sample_count_and_seeds",
            "text": (
                f"Evidence contains N={sample_count} policy episodes at seeds "
                f"{seeds}."
                if sample_count is not None and seeds
                else f"Evidence contains N={sample_count} policy episodes; seeds are unavailable."
                if sample_count is not None
                else f"Policy episode count is unavailable; observed seeds are {seeds}."
                if seeds
                else "Policy episode count and seeds are unavailable."
            ),
        }
    ]
    if untested:
        limitations.append(
            {
                "code": "untested_candidates",
                "text": f"Untested candidates remain: {untested}.",
            }
        )
    if conflict:
        limitations.append(
            {
                "code": "evidence_conflict",
                "text": "Execution VQA conflicts with another evidence source; the conflict remains unresolved.",
            }
        )
    termination_text = {
        "agent_stop": (
            "The Plan Agent stopped after reading the completed evidence. "
            "The conclusion remains limited to the recorded task, policy, "
            "variants, samples, and seeds."
        ),
        "budget_exhausted": (
            "The run stopped because its external round budget was exhausted "
            "before the Plan Agent produced a supported answer."
        ),
        "continue": (
            "The Plan Agent requested more evidence; the current answer is interim."
        ),
        "control_not_passed": (
            "The unchanged-scene control did not pass, so no property "
            "attribution or novel-variant conclusion is authorized."
        ),
        "pipeline_invalid": (
            "The evaluation pipeline is invalid, so it cannot support a policy "
            "performance conclusion."
        ),
        "unknown": (
            "No validated Agent-authored stop verdict is present in the evidence."
        ),
    }[termination]
    limitations.append(
        {
            "code": f"termination_{termination}",
            "text": termination_text,
        }
    )
    return limitations


def build_answer_scope(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Project facts that every final answer must expose as limitations."""

    if not isinstance(evidence, Mapping):
        raise AnswerScopeError("evidence must be an object")
    seeds = _collect_seeds(evidence)
    sample_count = _sample_count(evidence, seeds)
    tested = _tested_candidates(evidence)
    untested = _untested_candidates(evidence)
    conflict = _execution_conflict(evidence)
    termination, claim_verdict = _termination(evidence)
    limitations = _canonical_limitations(
        sample_count=sample_count,
        seeds=seeds,
        untested=untested,
        conflict=conflict,
        termination=termination,
    )
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
            "required_limitations": limitations,
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
    limitations = scope.get("required_limitations")
    if not isinstance(limitations, list) or not limitations:
        raise AnswerScopeError("required_limitations must be a non-empty list")
    codes = []
    for index, item in enumerate(limitations):
        if not isinstance(item, Mapping) or set(item) != _LIMITATION_KEYS:
            raise AnswerScopeError(
                f"required_limitations[{index}] must contain code and text"
            )
        code = item.get("code")
        text = item.get("text")
        if not isinstance(code, str) or not code:
            raise AnswerScopeError(f"required_limitations[{index}].code is invalid")
        if not isinstance(text, str) or not text:
            raise AnswerScopeError(f"required_limitations[{index}].text is invalid")
        codes.append(code)
    if len(codes) != len(set(codes)):
        raise AnswerScopeError("required limitation codes must be unique")
    expected_limitations = _canonical_limitations(
        sample_count=count,
        seeds=seeds,
        untested=scope["untested_candidate_ids"],
        conflict=scope["evidence_conflict"],
        termination=scope["termination"],
    )
    if limitations != expected_limitations:
        raise AnswerScopeError(
            "required_limitations do not match the structured answer scope"
        )
    return scope


def project_answer_scope(
    feedback: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Append canonical limitations and bind their codes to the final answer."""

    if not isinstance(feedback, Mapping):
        raise AnswerScopeError("feedback must be an object")
    normalized_scope = validate_answer_scope(scope)
    projected = deepcopy(dict(feedback))
    limitations = projected.get("limitations")
    if not isinstance(limitations, list) or any(
        not isinstance(item, str) or not item for item in limitations
    ):
        raise AnswerScopeError("feedback.limitations must be a string list")
    canonical = [
        item["text"] for item in normalized_scope["required_limitations"]
    ]
    projected["limitations"] = _dedupe([*limitations, *canonical])
    projected["answer_scope"] = normalized_scope
    projected["limitation_codes"] = [
        item["code"] for item in normalized_scope["required_limitations"]
    ]
    return validate_answer_scope_projection(projected, normalized_scope)


def validate_answer_scope_projection(
    feedback: Mapping[str, Any],
    expected_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed if structured scope or any canonical limitation is absent."""

    if not isinstance(feedback, Mapping):
        raise AnswerScopeError("feedback must be an object")
    embedded = feedback.get("answer_scope")
    if not isinstance(embedded, Mapping):
        raise AnswerScopeError("feedback is missing structured answer_scope")
    scope = validate_answer_scope(embedded)
    if expected_scope is not None and scope != validate_answer_scope(expected_scope):
        raise AnswerScopeError("feedback answer_scope differs from evidence")
    required_codes = [item["code"] for item in scope["required_limitations"]]
    codes = feedback.get("limitation_codes")
    if codes != required_codes:
        raise AnswerScopeError(
            "feedback limitation_codes do not exactly match answer_scope"
        )
    limitations = feedback.get("limitations")
    if not isinstance(limitations, list):
        raise AnswerScopeError("feedback.limitations must be a list")
    missing = [
        item["code"]
        for item in scope["required_limitations"]
        if item["text"] not in limitations
    ]
    if missing:
        raise AnswerScopeError(
            f"feedback omitted evidence-required limitations: {missing}"
        )
    return deepcopy(dict(feedback))


__all__ = [
    "AnswerScopeError",
    "build_answer_scope",
    "project_answer_scope",
    "validate_answer_scope",
    "validate_answer_scope_projection",
]
