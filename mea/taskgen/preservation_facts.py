"""Strict typed preservation facts shared by Plan and TaskGen."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class PreservationFactError(ValueError):
    """Raised when a typed preservation fact is malformed."""


_FACT_KEYS = frozenset({"actor", "property", "axis", "relation"})
_PROPERTY_RULES = {
    "appearance": ({"preserve"}, {None}),
    "checker_semantics": ({"preserve"}, {None}),
    "contact_point": (
        {"preserve_local_offsets", "preserve_world_position"},
        {None},
    ),
    "geometry": ({"preserve"}, {None}),
    "model_identity": ({"preserve"}, {None}),
    "official_goal": ({"preserve", "required_conjunct"}, {None}),
    "orientation": ({"preserve"}, {None}),
    "position": ({"preserve"}, {"all", "x", "y", "z"}),
}
_TASK_WIDE_PROPERTIES = frozenset({"checker_semantics", "official_goal"})


def validate_preservation_fact(
    value: Mapping[str, Any],
) -> dict[str, str | None]:
    """Validate one ``{actor, property, axis, relation}`` fact."""

    if not isinstance(value, Mapping) or set(value) != _FACT_KEYS:
        raise PreservationFactError(
            "preservation fact fields must be exactly "
            f"{sorted(_FACT_KEYS)}"
        )
    actor = value.get("actor")
    if actor is not None and (
        not isinstance(actor, str) or not actor.strip()
    ):
        raise PreservationFactError(
            "preservation fact actor must be a non-empty string or null"
        )
    property_name = value.get("property")
    if property_name not in _PROPERTY_RULES:
        raise PreservationFactError(
            "preservation fact property must be one of "
            f"{sorted(_PROPERTY_RULES)}"
        )
    if property_name in _TASK_WIDE_PROPERTIES and actor is not None:
        raise PreservationFactError(
            f"preservation fact actor must be null for {property_name}"
        )
    relation = value.get("relation")
    axis = value.get("axis")
    allowed_relations, allowed_axes = _PROPERTY_RULES[str(property_name)]
    if relation not in allowed_relations:
        raise PreservationFactError(
            f"preservation fact relation for {property_name} must be one of "
            f"{sorted(allowed_relations)}"
        )
    if axis not in allowed_axes:
        rendered_axes = sorted(
            "null" if item is None else item for item in allowed_axes
        )
        raise PreservationFactError(
            f"preservation fact axis for {property_name} must be one of "
            f"{rendered_axes}"
        )
    return {
        "actor": actor.strip() if isinstance(actor, str) else None,
        "property": str(property_name),
        "axis": axis if isinstance(axis, str) else None,
        "relation": str(relation),
    }


def normalize_preservation_conditions(
    value: Sequence[Mapping[str, Any]],
) -> list[dict[str, str | None]]:
    """Validate a Proposal preservation list without parsing prose."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PreservationFactError("preservation conditions must be a list")
    result: list[dict[str, str | None]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise PreservationFactError(
                "each preservation condition must be a typed object"
            )
        result.append(validate_preservation_fact(item))
    return result


def describe_preservation_fact(value: Mapping[str, Any]) -> str:
    """Render a validated fact for a provider prompt or diagnosis."""

    fact = validate_preservation_fact(value)
    actor = f"{fact['actor']} " if fact["actor"] else ""
    axis = f" {fact['axis']}" if fact["axis"] else ""
    return f"{actor}{fact['property']}{axis} ({fact['relation']})"


__all__ = [
    "PreservationFactError",
    "describe_preservation_fact",
    "normalize_preservation_conditions",
    "validate_preservation_fact",
]
