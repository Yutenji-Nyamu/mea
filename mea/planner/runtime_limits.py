"""Minimal runtime limits and evidence state for the Plan Agent.

The paper assigns the continue/stop judgement to the Plan Agent.  This module
therefore does not parse natural-language quantifiers, enumerate a candidate
universe, or prove a Query truth value.  It only keeps the external round cap,
the explicit control choice made during routing, and simulator-authoritative
evidence conflicts visible to the Agent and final answer.

The small schema below never recreates a finite-domain truth contract.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping


class PlanRuntimeError(ValueError):
    """Raised when runtime limits or evidence are malformed."""


OUTCOMES = frozenset({"pass", "fail", "mixed", "unknown", "conflict"})
_RUNTIME_KEYS = {"schema_version", "round_budget", "control_requirement"}
_CONTROL_REQUIREMENTS = frozenset({"required", "not_required"})


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PlanRuntimeError(f"{field} must be a positive integer")
    return value


def _control(value: Any) -> str:
    if value not in _CONTROL_REQUIREMENTS:
        raise PlanRuntimeError(
            "control_requirement must be required or not_required"
        )
    return str(value)


def infer_control_requirement(
    user_query: str,
    *,
    semantic_context: Mapping[str, Any] | None = None,
) -> str:
    """Return only an explicit official-only exemption.

    The Query interpreter/Plan Agent owns whether a comparison needs a control;
    this fallback deliberately avoids a lexical topic classifier.  Ambiguous
    requests keep the conservative execution default.
    """

    if not isinstance(user_query, str) or not user_query.strip():
        raise PlanRuntimeError("user_query must be a non-empty string")
    if semantic_context is not None and not isinstance(semantic_context, Mapping):
        raise PlanRuntimeError("semantic_context must be an object")
    return "required"


def build_plan_runtime_limits(
    user_query: str,
    *,
    round_budget: int,
    control_requirement: str = "required",
) -> dict[str, Any]:
    """Build the minimal production runtime limit object.

    The user Query is validated as text but is not parsed into a truth system.
    """

    if not isinstance(user_query, str) or not user_query.strip():
        raise PlanRuntimeError("user_query must be a non-empty string")
    return validate_plan_runtime_limits(
        {
            "schema_version": 4,
            "round_budget": round_budget,
            "control_requirement": control_requirement,
        }
    )


def validate_plan_runtime_limits(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Read runtime limits; collapse historical contracts to the same schema."""

    if not isinstance(value, Mapping):
        raise PlanRuntimeError("PlanAgentRuntimeLimits must be an object")
    budget = _positive_int(value.get("round_budget"), "round_budget")
    control = _control(value.get("control_requirement", "required"))
    return {
        "schema_version": 4,
        "round_budget": budget,
        "control_requirement": control,
    }


def _evidence_items(
    candidate_evidence: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, raw in enumerate(candidate_evidence):
        if not isinstance(raw, Mapping):
            raise PlanRuntimeError(
                f"candidate_evidence[{index}] must be an object"
            )
        outcome = raw.get("outcome")
        if outcome not in OUTCOMES:
            raise PlanRuntimeError(
                f"candidate_evidence[{index}].outcome must be one of "
                f"{sorted(OUTCOMES)}"
            )
        item = deepcopy(dict(raw))
        item["candidate_id"] = str(raw.get("candidate_id") or f"round_{index + 1}")
        items.append(item)
    return items


def summarize_plan_evidence(
    contract: Mapping[str, Any],
    candidate_evidence: Iterable[Mapping[str, Any]],
    *,
    completed_rounds: int | None = None,
) -> dict[str, Any]:
    """Summarize evidence without pre-authoring the Agent's stop/verdict."""

    limits = validate_plan_runtime_limits(contract)
    evidence = _evidence_items(candidate_evidence)
    rounds = len(evidence) if completed_rounds is None else completed_rounds
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
        raise PlanRuntimeError("completed_rounds must be non-negative")
    if rounds < len(evidence):
        raise PlanRuntimeError(
            "completed_rounds cannot be smaller than evidence count"
        )
    budget_remaining = max(limits["round_budget"] - rounds, 0)
    observed = list(dict.fromkeys(item["candidate_id"] for item in evidence))
    conflicts = list(
        dict.fromkeys(
            item["candidate_id"]
            for item in evidence
            if item["outcome"] == "conflict"
        )
    )
    # The Plan Agent, not this projection, judges semantic sufficiency.  The
    # external cap is the only hard runtime stop; simulator conflicts remain
    # explicit facts that may motivate a disambiguating experiment.
    stop_reason = "budget_exhausted" if budget_remaining == 0 else "continue"
    should_stop = budget_remaining == 0
    return {
        "schema_version": 2,
        "contract": limits,
        "should_stop": should_stop,
        "stop_reason": stop_reason,
        "claim_verdict": "inconclusive",
        "evidence_sufficient": False,
        "completed_rounds": rounds,
        "round_budget": limits["round_budget"],
        "budget_remaining": budget_remaining,
        "observed_candidate_ids": observed,
        "decisive_candidate_ids": [
            item["candidate_id"]
            for item in evidence
            if item["outcome"] in {"pass", "fail", "mixed"}
        ],
        "conflict_candidate_ids": conflicts,
        "unknown_candidate_ids": [
            item["candidate_id"]
            for item in evidence
            if item["outcome"] == "unknown"
        ],
        "untested_required_candidate_ids": [],
        "untested_candidate_ids": [],
        "recommended_candidate_ids": [],
        "rationale": (
            "The external round cap was reached."
            if budget_remaining == 0
            else (
                "Simulator-authoritative evidence conflicts remain unresolved; "
                "the Plan Agent may propose a disambiguating experiment or stop "
                "inconclusively."
            )
            if conflicts
            else "The Plan Agent must judge whether this evidence is sufficient."
        ),
        "statistics": {},
        "limitations": [
            "Semantic sufficiency is judged by the Plan Agent from completed evidence."
        ],
    }


def validate_agent_stop(
    assessment: Mapping[str, Any],
    *,
    rationale: str,
    answer: str | None,
    claim_verdict: str,
    evidence_sufficient: bool,
) -> dict[str, Any]:
    """Bind the Agent's stop judgement to observed runtime facts.

    This is deliberately not a Query truth formalizer.  It only prevents an
    answered stop with no decisive completed evidence or with a simulator
    conflict.  The Plan Agent owns the semantic conclusion.
    """

    if not isinstance(assessment, Mapping):
        raise PlanRuntimeError("assessment must be an object")
    if not isinstance(rationale, str) or not rationale.strip():
        raise PlanRuntimeError("Plan Agent stop requires a rationale")
    if not isinstance(evidence_sufficient, bool):
        raise PlanRuntimeError("evidence_sufficient must be bool")
    if claim_verdict not in {"supported", "refuted", "inconclusive"}:
        raise PlanRuntimeError(
            "claim_verdict must be supported, refuted, or inconclusive"
        )
    normalized_answer = answer.strip() if isinstance(answer, str) else None
    if evidence_sufficient:
        if assessment.get("conflict_candidate_ids"):
            raise PlanRuntimeError(
                "evidence with simulator conflicts cannot support an answered stop"
            )
        if not assessment.get("decisive_candidate_ids"):
            raise PlanRuntimeError(
                "an answered stop requires decisive completed evidence"
            )
        if claim_verdict not in {"supported", "refuted"}:
            raise PlanRuntimeError(
                "an answered stop requires a supported or refuted verdict"
            )
        if not normalized_answer:
            raise PlanRuntimeError("an answered stop requires an answer")
    elif claim_verdict != "inconclusive":
        raise PlanRuntimeError(
            "an insufficient stop must use the inconclusive verdict"
        )
    return {
        **deepcopy(dict(assessment)),
        "should_stop": True,
        "stop_reason": "agent_stop",
        "claim_verdict": claim_verdict,
        "evidence_sufficient": evidence_sufficient,
        "agent_answer": normalized_answer,
        "rationale": rationale.strip(),
    }


__all__ = [
    "OUTCOMES",
    "PlanRuntimeError",
    "build_plan_runtime_limits",
    "infer_control_requirement",
    "summarize_plan_evidence",
    "validate_agent_stop",
    "validate_plan_runtime_limits",
]
