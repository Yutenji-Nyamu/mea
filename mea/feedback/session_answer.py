"""Deterministic AnswerScope projection for a completed Plan Agent session."""

from __future__ import annotations

from typing import Any, Mapping

from .answer_scope import build_answer_scope, validate_answer_scope_projection
from .prototype import (
    PlanAgentFinalSummaryError,
    _deterministic_aggregate,
    _has_execution_vqa_conflict,
    _require_text,
    apply_deterministic_consistency_guard,
)


def build_scoped_plan_agent_answer(
    evidence: Mapping[str, Any],
    query_answer: Mapping[str, Any],
) -> dict[str, Any]:
    """Turn a session-owned Query answer into the shared final-answer schema."""

    if not isinstance(evidence, Mapping):
        raise PlanAgentFinalSummaryError("evidence must be an object")
    if not isinstance(query_answer, Mapping):
        raise PlanAgentFinalSummaryError("query_answer must be an object")

    scope = build_answer_scope(evidence)
    tested = list(query_answer.get("tested_candidate_ids") or [])
    untested = list(query_answer.get("untested_candidate_ids") or [])
    verdict = query_answer.get("claim_verdict") or scope.get("claim_verdict")
    findings: list[str] = []
    if verdict:
        findings.append(f"Plan Agent verdict: {verdict}.")
    if tested:
        findings.append("Tested candidates: " + ", ".join(map(str, tested)) + ".")
    outcomes = query_answer.get("evaluation_outcomes")
    if isinstance(outcomes, list) and outcomes:
        findings.append(
            f"The Plan Agent recorded {len(outcomes)} evaluated outcome(s)."
        )
    if not findings:
        findings.append(
            "The Plan Agent session recorded no supported candidate-level finding."
        )

    limitations = query_answer.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        limitations = [
            "The answer is limited to the evidence recorded in this evaluation."
        ]

    sample_count = scope.get("sample_count")
    seeds = scope.get("seeds") or []
    evaluation_scope = (
        f"N={sample_count if sample_count is not None else 'unknown'}; "
        f"seeds={seeds}; termination={scope['termination']}."
    )
    if untested:
        next_step = "Evaluate the remaining candidate(s): " + ", ".join(
            map(str, untested)
        )
        next_step += "."
    elif scope["evidence_conflict"]:
        next_step = "Resolve the recorded evidence conflict before extending the claim."
    elif scope["termination"] == "evidence_sufficient":
        next_step = (
            "Repeat the bounded evaluation on additional seeds before making a "
            "broader generalization claim."
        )
    else:
        next_step = "Collect the next executable evidence chosen by the Plan Agent."

    feedback = apply_deterministic_consistency_guard(
        {
            "answer": _require_text(
                query_answer.get("answer"), "query_answer.answer"
            ),
            "evaluation_scope": evaluation_scope,
            "findings": findings,
            "limitations": list(limitations),
            "recommended_next_step": next_step,
        },
        dict(evidence),
        attempts_used=0,
    )
    aggregate = _deterministic_aggregate(dict(evidence))
    feedback["evidence_policy"] = {
        "aggregate_source": (
            "deterministic_aggregate" if aggregate is not None else None
        ),
        "aggregate_status": (
            aggregate.get("status") if aggregate is not None else None
        ),
        "episode_math_by_plan_agent_summary": False,
        "numeric_simulator_tools_authoritative": True,
        "execution_vqa_is_visual_only": True,
        "evidence_conflict": _has_execution_vqa_conflict(dict(evidence)),
    }
    feedback["provider_metadata"] = {
        "called": False,
        "reason": "plan_agent_session_query_answer_projection",
    }
    validate_answer_scope_projection(feedback, scope)
    return feedback


__all__ = ["build_scoped_plan_agent_answer"]
