"""Semantic coverage contract from an open Query to runtime artifacts.

The planner is allowed to refine an experiment after observing evidence, but
it must not silently replace the first Query-derived concern with an easier
nearby diagnostic before that concern is tested. ``EvaluationIntent`` freezes
that first candidate's semantics before runtime task binding. Query answer
sufficiency remains owned by the Plan Agent.

The contract is deliberately small.  It freezes Query-derived intent and
validates preservation as typed facts; runtime authority remains with the
structured Proposal and TaskGen/ToolGen evidence.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.taskgen.preservation_facts import (
    PreservationFactError,
    normalize_preservation_conditions,
)


class SemanticCoverageError(ValueError):
    """Raised when an EvaluationIntent is malformed."""


_INTENT_KEYS = {
    "schema_version",
    "intent_id",
    "source_query",
    "original_concern",
    "hypothesis",
    "requested_change",
    "preserved_conditions",
    "required_observation",
}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticCoverageError(f"{field} must be a non-empty string")
    return value.strip()


def build_evaluation_intent(
    *,
    source_query: str,
    original_concern: str,
    hypothesis: str,
    requested_change: str,
    required_observation: str,
    preserved_conditions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Freeze one Query-derived candidate before task/checkpoint binding."""

    try:
        preservation_facts = normalize_preservation_conditions(
            preserved_conditions
        )
    except PreservationFactError as exc:
        raise SemanticCoverageError(
            f"preserved_conditions: {exc}"
        ) from exc
    payload = {
        "source_query": _text(source_query, "source_query"),
        "original_concern": _text(original_concern, "original_concern"),
        "hypothesis": _text(hypothesis, "hypothesis"),
        "requested_change": _text(requested_change, "requested_change"),
        "preserved_conditions": preservation_facts,
        "required_observation": _text(
            required_observation, "required_observation"
        ),
    }
    return validate_evaluation_intent(
        {
            "schema_version": 1,
            # Runtime evidence carries the concrete round/candidate ids.  The
            # semantic intent needs only a readable local label, not a content
            # hash or an immutable provenance contract.
            "intent_id": "intent.query",
            **payload,
        }
    )


def evaluation_intent_from_query_interpretation(
    concern: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an intent from the provider-authored Query interpretation."""

    if not isinstance(concern, Mapping):
        raise SemanticCoverageError("query interpretation must be an object")
    source_query = _text(
        concern.get("source_query"),
        "query_interpretation.source_query",
    )
    requested_variation = _text(
        concern.get("requested_variation"),
        "query_interpretation.requested_variation",
    )
    return build_evaluation_intent(
        source_query=source_query,
        original_concern=concern.get("sub_aspect"),
        hypothesis=concern.get("hypothesis"),
        requested_change=requested_variation,
        required_observation=concern.get("measurement_need"),
        preserved_conditions=concern.get("preserved_conditions") or (),
    )


def validate_evaluation_intent(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _INTENT_KEYS:
        raise SemanticCoverageError(
            "EvaluationIntent fields must be exactly "
            f"{sorted(_INTENT_KEYS)}"
        )
    intent = deepcopy(dict(value))
    if intent.get("schema_version") != 1:
        raise SemanticCoverageError("EvaluationIntent schema_version must be 1")
    for field in (
        "intent_id",
        "source_query",
        "original_concern",
        "hypothesis",
        "requested_change",
        "required_observation",
    ):
        intent[field] = _text(intent.get(field), f"EvaluationIntent.{field}")
    try:
        intent["preserved_conditions"] = normalize_preservation_conditions(
            intent.get("preserved_conditions")
        )
    except PreservationFactError as exc:
        raise SemanticCoverageError(
            f"EvaluationIntent.preserved_conditions: {exc}"
        ) from exc
    return intent


evaluation_intent_from_free_concern = (
    evaluation_intent_from_query_interpretation
)


__all__ = [
    "SemanticCoverageError",
    "build_evaluation_intent",
    "evaluation_intent_from_query_interpretation",
    "evaluation_intent_from_free_concern",
    "validate_evaluation_intent",
]
