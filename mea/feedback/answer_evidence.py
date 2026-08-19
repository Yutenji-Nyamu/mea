"""Concise evidence projection for the provider-authored final answer."""

from __future__ import annotations

import json
from typing import Any


def _present(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {field: value[field] for field in fields if field in value}


def _aggregate(evidence: dict[str, Any]) -> dict[str, Any] | None:
    observations = evidence.get("observations")
    if isinstance(observations, dict) and isinstance(
        observations.get("aggregate"), dict
    ):
        aggregate = observations["aggregate"]
    else:
        aggregate = evidence.get("aggregate") or evidence.get("final_aggregate")
    if not isinstance(aggregate, dict):
        return None
    metrics = []
    for item in aggregate.get("metrics", []):
        if not isinstance(item, dict):
            continue
        metrics.append(
            {
                **_present(item, ("metric", "value_kind", "unit")),
                "cohorts": [
                    {
                        **_present(cohort, ("role", "policy_names")),
                        "statistics": (
                            cohort.get("summary", {}).get("statistics", {})
                            if isinstance(cohort.get("summary"), dict)
                            else {}
                        ),
                    }
                    for cohort in item.get("cohorts", [])
                    if isinstance(cohort, dict)
                ],
            }
        )
    return {
        **_present(
            aggregate,
            ("status", "source_count", "unique_episode_count"),
        ),
        "input_issue_count": len(aggregate.get("input_issues", []))
        if isinstance(aggregate.get("input_issues"), list)
        else None,
        "metrics": metrics,
    }


def _tool_measurements(round_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Project live values, not Tool source, registry, or validation metadata."""

    rows: list[dict[str, Any]] = []
    planned = round_evidence.get("tool_evaluation")
    if isinstance(planned, dict):
        for episode in planned.get("episodes", []):
            if isinstance(episode, dict):
                result = episode.get("result")
                result = result if isinstance(result, dict) else episode
                rows.append(
                    {
                        **_present(episode, ("policy_name", "role", "seed")),
                        "metric": result.get("tool") or result.get("metric"),
                        **_present(
                            result,
                            ("value", "unit", "passed", "evidence_steps"),
                        ),
                    }
                )
    trusted = round_evidence.get("trusted_tool_evaluation")
    if isinstance(trusted, dict):
        for episode in trusted.get("episodes", []):
            if not isinstance(episode, dict):
                continue
            identity = _present(episode, ("policy_name", "role", "seed"))
            for result in episode.get("results", []):
                if isinstance(result, dict):
                    rows.append(
                        {
                            **identity,
                            "metric": result.get("tool") or result.get("metric"),
                            **_present(
                                result,
                                ("value", "unit", "passed", "evidence_steps"),
                            ),
                        }
                    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        marker = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            unique.append(row)
    return unique


def _vqa(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        **_present(
            value,
            ("status", "reason", "evidence_conflict", "model_requested"),
        ),
        "observation": _present(
            value.get("observation"),
            (
                "answer",
                "confidence",
                "phenomenon",
                "phenomena",
                "numeric_consistency",
                "conflicts",
                "limitations",
            ),
        ),
    }


def _round(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    observations = value.get("observations")
    observations = observations if isinstance(observations, dict) else {}
    execution = value.get("execution")
    seeds = value.get("actual_seeds") or value.get("seeds")
    if not seeds and isinstance(execution, dict):
        seeds = execution.get("seeds")
    return {
        **_present(
            value,
            (
                "round_id",
                "candidate_id",
                "variant_id",
                "sub_aspect",
                "route",
            ),
        ),
        "seeds": seeds or [],
        "num_episodes": value.get(
            "num_episodes", observations.get("policy_sample_count")
        ),
        "execution": {
            **_present(
                observations,
                (
                    "execution_backend",
                    "pipeline_passed",
                    "policy_success",
                    "policy_sample_count",
                ),
            ),
            "policy_outcome": _present(
                observations.get("policy_outcome"),
                (
                    "metric",
                    "authority",
                    "binding",
                    "value",
                    "official_equivalent",
                    "execution_scope",
                ),
            ),
            "outcome_semantics": _present(
                observations.get("outcome_semantics"),
                ("status", "evidence_conflict", "official_equivalent", "reason_codes"),
            ),
        },
        "planning_observation": _present(
            observations.get("planning_observation"),
            (
                "kind",
                "candidate_id",
                "sub_aspect",
                "failure_stage",
                "reason_code",
                "diagnosis",
                "policy_rollouts_started",
                "policy_sample_count",
                "bounded_repair_evidence",
            ),
        ),
        "tool_measurements": _tool_measurements(value),
        "execution_vqa": _vqa(
            value.get("execution_vqa") or observations.get("execution_vqa")
        ),
    }


def project_answer_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return only evidence the final Answer Agent needs to reason.

    Raw evidence remains authoritative for AnswerScope and deterministic
    validators.  This projection excludes transport metadata, artifact paths,
    full Proposals, session records, registries, source code, and gate ledgers.
    """

    sources = evidence.get("rounds")
    sources = sources if isinstance(sources, list) else [evidence]
    rounds = [item for source in sources if (item := _round(source)) is not None]
    session = evidence.get("plan_agent_session")
    session = session if isinstance(session, dict) else {}
    assessment = session.get("assessment")
    query_answer = session.get("query_answer")
    query_answer = query_answer if isinstance(query_answer, dict) else {}
    plan = evidence.get("plan")
    plan = plan if isinstance(plan, dict) else {}
    history = evidence.get("history_retrieval")
    history = history if isinstance(history, dict) else {}
    return {
        "schema_version": 1,
        "evaluation_id": evidence.get("evaluation_id"),
        "query": evidence.get("user_request")
        or evidence.get("query")
        or evidence.get("original_query"),
        "execution_summary": {
            "executed_rounds": len(rounds),
            "total_policy_episodes": evidence.get("total_episodes"),
            "planning_state": plan.get("planning_state"),
        },
        "rounds": rounds,
        "final_aggregate": _aggregate(evidence),
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
        "history_retrieval": {
            "status": history.get("status"),
            "match_count": len(history.get("matches", []))
            if isinstance(history.get("matches"), list)
            else None,
        },
    }


__all__ = ["project_answer_evidence"]
