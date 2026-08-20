"""Concise RoundEvidence projection for the final Answer stage."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def _present(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {field: deepcopy(value[field]) for field in fields if field in value}


def project_answer_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the final Plan state and the single typed round protocol."""

    raw_rounds = evidence.get("rounds")
    rounds = (
        [
            deepcopy(dict(item))
            for item in raw_rounds
            if isinstance(item, Mapping)
        ]
        if isinstance(raw_rounds, list)
        else []
    )
    session = evidence.get("plan_agent_session")
    session = session if isinstance(session, Mapping) else {}
    assessment = session.get("assessment")
    query_answer = session.get("query_answer")
    plan = evidence.get("plan")
    plan = plan if isinstance(plan, Mapping) else {}
    return {
        "schema_version": 1,
        "evaluation_id": evidence.get("evaluation_id"),
        "query": evidence.get("query"),
        "execution_summary": {
            "executed_rounds": len(rounds),
            "total_policy_episodes": evidence.get(
                "total_policy_episodes"
            ),
            "planning_state": plan.get("planning_state"),
        },
        "rounds": rounds,
        "final_plan": {
            **_present(
                assessment,
                (
                    "should_stop",
                    "evidence_sufficient",
                    "stop_reason",
                    "claim_verdict",
                    "rationale",
                    "limitations",
                    "observed_candidate_ids",
                    "untested_candidate_ids",
                    "conflict_candidate_ids",
                ),
            ),
            "query_answer": _present(
                query_answer,
                (
                    "answered",
                    "answer_scope",
                    "official_benchmark_answered",
                    "stop_reason",
                    "claim_type",
                    "claim_verdict",
                    "answer",
                    "tested_candidate_ids",
                    "untested_candidate_ids",
                    "limitations",
                    "evidence_conflict",
                ),
            ),
        },
    }


__all__ = ["project_answer_evidence"]
