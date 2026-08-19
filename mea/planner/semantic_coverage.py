"""Semantic coverage contracts from an open Query to runtime artifacts.

The planner is allowed to refine an experiment after observing evidence, but
it must not silently replace the first Query-derived concern with an easier
nearby diagnostic before that concern is tested. ``EvaluationIntent`` freezes
that first candidate's semantics before runtime task binding. Query answer
sufficiency remains owned by the Plan Agent.

The contract is deliberately small.  It does not attempt to prove natural
language equivalence; it combines explicit planner declarations with
TaskGen/ToolGen validation facts and fails toward an explicit proxy label when
the candidate wording does not preserve the requested change and observation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.taskgen.preservation_facts import (
    PreservationFactError,
    normalize_preservation_conditions,
)


class SemanticCoverageError(ValueError):
    """Raised when an intent or alignment is malformed."""


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
_ALIGNMENT_KEYS = {
    "schema_version",
    "relationship",
    "rationale",
    "matched_intent_fields",
    "unmatched_intent_fields",
}
_INTENT_REQUIREMENT_FIELDS = (
    "requested_change",
    "preserved_conditions",
    "hypothesis",
    "required_observation",
)
_RELATIONSHIPS = {"direct", "diagnostic_proxy", "unsupported"}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticCoverageError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(
    value: Any,
    field: str,
    *,
    allow_empty: bool = True,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SemanticCoverageError(f"{field} must be a string list")
    result = [
        _text(item, f"{field}[]")
        for item in value
    ]
    if len(result) != len(set(result)):
        raise SemanticCoverageError(f"{field} must not contain duplicates")
    if not allow_empty and not result:
        raise SemanticCoverageError(f"{field} must not be empty")
    return result


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


def validate_intent_alignment(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ALIGNMENT_KEYS:
        raise SemanticCoverageError(
            "IntentAlignment fields must be exactly "
            f"{sorted(_ALIGNMENT_KEYS)}"
        )
    alignment = deepcopy(dict(value))
    if alignment.get("schema_version") != 1:
        raise SemanticCoverageError("IntentAlignment schema_version must be 1")
    if alignment.get("relationship") not in _RELATIONSHIPS:
        raise SemanticCoverageError(
            f"IntentAlignment.relationship must be one of "
            f"{sorted(_RELATIONSHIPS)}"
        )
    alignment["rationale"] = _text(
        alignment.get("rationale"), "IntentAlignment.rationale"
    )
    for field in ("matched_intent_fields", "unmatched_intent_fields"):
        alignment[field] = _string_list(
            alignment.get(field), f"IntentAlignment.{field}"
        )
        unknown = set(alignment[field]) - set(_INTENT_REQUIREMENT_FIELDS)
        if unknown:
            raise SemanticCoverageError(
                f"IntentAlignment.{field} contains unknown fields: "
                f"{sorted(unknown)}"
            )
    if set(alignment["matched_intent_fields"]) & set(
        alignment["unmatched_intent_fields"]
    ):
        raise SemanticCoverageError(
            "IntentAlignment matched/unmatched fields must be disjoint"
        )
    if set(alignment["matched_intent_fields"]) | set(
        alignment["unmatched_intent_fields"]
    ) != set(_INTENT_REQUIREMENT_FIELDS):
        raise SemanticCoverageError(
            "IntentAlignment must classify every candidate-contract field"
        )
    if (
        alignment["relationship"] == "direct"
        and alignment["unmatched_intent_fields"]
    ):
        raise SemanticCoverageError(
            "direct IntentAlignment cannot leave intent fields unmatched"
        )
    return alignment


def build_candidate_intent_alignment(
    intent: Mapping[str, Any],
    *,
    semantic_concern: str,
    scene_need: Mapping[str, Any] | None,
    checker_need: Mapping[str, Any] | None,
    rule_tool_need: Mapping[str, Any] | None = None,
    vqa_tool_need: Mapping[str, Any] | None = None,
    tool_need: Mapping[str, Any] | None = None,
    declared_relationship: str | None = None,
    declared_rationale: str | None = None,
) -> dict[str, Any]:
    """Classify a candidate conservatively against its frozen intent."""

    normalized = validate_evaluation_intent(intent)
    _text(semantic_concern, "semantic_concern")
    observation_owner_present = any(
        isinstance(need, Mapping)
        for need in (
            scene_need,
            checker_need,
            rule_tool_need,
            vqa_tool_need,
            tool_need,
        )
    )
    matched = []
    for field in _INTENT_REQUIREMENT_FIELDS:
        if field == "requested_change":
            # Whether a scene artifact is required is a typed Proposal choice.
            # Do not second-guess it by classifying the Query wording.
            matched_field = True
        elif field == "preserved_conditions":
            conditions = normalized["preserved_conditions"]
            # Preservation is simulator-authoritative.  Carry explicit
            # conditions into TaskGen rather than rejecting a Proposal because
            # its free-form wording has insufficient token overlap.
            matched_field = (
                not conditions
                or scene_need is None
                or isinstance(scene_need, Mapping)
            )
        elif field == "hypothesis":
            # The hypothesis is carried verbatim in EvaluationIntent.  Code
            # cannot establish semantic equivalence by token overlap.
            matched_field = True
        elif field == "required_observation":
            matched_field = observation_owner_present
        else:
            matched_field = False
        if matched_field:
            matched.append(field)
    unmatched = [
        field for field in _INTENT_REQUIREMENT_FIELDS if field not in matched
    ]
    inferred = "direct" if not unmatched else "diagnostic_proxy"
    relationship = declared_relationship or inferred
    if relationship not in _RELATIONSHIPS:
        raise SemanticCoverageError(
            f"declared relationship must be one of {sorted(_RELATIONSHIPS)}"
        )
    if relationship == "direct" and inferred != "direct":
        raise SemanticCoverageError(
            "candidate cannot declare direct coverage while contract "
            f"fields are unmatched: {unmatched}"
        )
    rationale = (
        _text(declared_rationale, "declared_rationale")
        if declared_rationale is not None
        else (
            "Candidate preserves the requested change, hypothesis, and "
            "observation semantics."
            if relationship == "direct"
            else
            "Candidate is a nearby diagnostic, not a direct implementation "
            f"of candidate-contract fields: {unmatched}."
        )
    )
    return validate_intent_alignment(
        {
            "schema_version": 1,
            "relationship": relationship,
            "rationale": rationale,
            "matched_intent_fields": matched,
            "unmatched_intent_fields": unmatched,
        }
    )


evaluation_intent_from_free_concern = (
    evaluation_intent_from_query_interpretation
)


__all__ = [
    "SemanticCoverageError",
    "build_candidate_intent_alignment",
    "build_evaluation_intent",
    "evaluation_intent_from_query_interpretation",
    "evaluation_intent_from_free_concern",
    "validate_evaluation_intent",
    "validate_intent_alignment",
]
