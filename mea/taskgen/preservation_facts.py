"""Typed preservation facts shared by Plan and TaskGen.

New Proposals carry four explicit fields instead of preservation prose.  A
small legacy reader remains for frozen artifacts: it recognizes only the old
canonical vocabulary and maps everything else to ``unknown``.  Unknown text
is evidence that a condition was not verified, never a reason to reject an
otherwise executable candidate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class PreservationFactError(ValueError):
    """Raised when a newly authored typed preservation fact is malformed."""


_PROPERTIES = frozenset(
    {
        "appearance",
        "checker_semantics",
        "contact_point",
        "geometry",
        "model_identity",
        "official_goal",
        "orientation",
        "policy_checkpoint",
        "position",
        "task_identity",
        "unknown",
    }
)
_AXES = frozenset({None, "all", "x", "y", "z"})
_RELATIONS = frozenset(
    {
        "preserve",
        "preserve_local_offsets",
        "preserve_world_position",
        "required_conjunct",
        "unparsed",
    }
)
_FACT_KEYS = frozenset({"actor", "property", "axis", "relation"})


def _fact(
    *,
    actor: str | None,
    property_name: str,
    axis: str | None = None,
    relation: str = "preserve",
) -> dict[str, str | None]:
    return {
        "actor": actor,
        "property": property_name,
        "axis": axis,
        "relation": relation,
    }


def unknown_preservation_fact() -> dict[str, str | None]:
    return _fact(
        actor=None,
        property_name="unknown",
        relation="unparsed",
    )


def validate_preservation_fact(
    value: Mapping[str, Any],
) -> dict[str, str | None]:
    """Validate one newly authored ``{actor, property, axis, relation}`` fact."""

    if not isinstance(value, Mapping) or set(value) != _FACT_KEYS:
        raise PreservationFactError(
            "preservation fact fields must be exactly "
            f"{sorted(_FACT_KEYS)}"
        )
    actor = value.get("actor")
    property_name = value.get("property")
    axis = value.get("axis")
    relation = value.get("relation")
    if actor is not None and (
        not isinstance(actor, str) or not actor.strip()
    ):
        raise PreservationFactError(
            "preservation fact actor must be a non-empty string or null"
        )
    if property_name not in _PROPERTIES - {"unknown"}:
        raise PreservationFactError(
            "preservation fact property must be one of "
            f"{sorted(_PROPERTIES - {'unknown'})}"
        )
    if axis not in _AXES:
        raise PreservationFactError(
            "preservation fact axis must be x, y, z, all, or null"
        )
    if property_name != "position" and axis is not None:
        raise PreservationFactError(
            "preservation fact axis is valid only for property=position"
        )
    if relation not in _RELATIONS - {"unparsed"}:
        raise PreservationFactError(
            "preservation fact relation must be one of "
            f"{sorted(_RELATIONS - {'unparsed'})}"
        )
    return _fact(
        actor=actor.strip() if isinstance(actor, str) else None,
        property_name=str(property_name),
        axis=axis if isinstance(axis, str) else None,
        relation=str(relation),
    )


_GLOBAL_LEGACY_FACTS = {
    "task identity": _fact(actor=None, property_name="task_identity"),
    "base task": _fact(actor=None, property_name="task_identity"),
    "official task": _fact(actor=None, property_name="task_identity"),
    "policy checkpoint": _fact(actor=None, property_name="policy_checkpoint"),
    "checkpoint": _fact(actor=None, property_name="policy_checkpoint"),
    "policy weights": _fact(actor=None, property_name="policy_checkpoint"),
    "official core predicate": _fact(
        actor=None,
        property_name="official_goal",
        relation="required_conjunct",
    ),
    "official core predicate as a required conjunct": _fact(
        actor=None,
        property_name="official_goal",
        relation="required_conjunct",
    ),
    "official goal as a required conjunct": _fact(
        actor=None,
        property_name="official_goal",
        relation="required_conjunct",
    ),
    "official task goal as a required conjunct": _fact(
        actor=None,
        property_name="official_goal",
        relation="required_conjunct",
    ),
    "official success semantics": _fact(
        actor=None, property_name="checker_semantics"
    ),
    "success semantics": _fact(actor=None, property_name="checker_semantics"),
    "success predicate": _fact(actor=None, property_name="checker_semantics"),
    "goal semantics": _fact(actor=None, property_name="checker_semantics"),
    "checker semantics": _fact(actor=None, property_name="checker_semantics"),
    "height": _fact(actor=None, property_name="position", axis="z"),
}


def _legacy_fact(value: str) -> dict[str, str | None]:
    """Read the former canonical vocabulary without classifying free prose."""

    text = " ".join(
        value.casefold()
        .strip()
        .replace("contact-point", "contact point")
        .split()
    )
    if text in _GLOBAL_LEGACY_FACTS:
        return dict(_GLOBAL_LEGACY_FACTS[text])

    suffixes: tuple[tuple[str, str, str | None, str], ...] = (
        (
            "contact point world position",
            "contact_point",
            None,
            "preserve_world_position",
        ),
        (
            "contact point local offsets",
            "contact_point",
            None,
            "preserve_local_offsets",
        ),
        (
            "contact point references",
            "contact_point",
            None,
            "preserve_local_offsets",
        ),
        ("contact points", "contact_point", None, "preserve_local_offsets"),
        ("contact point", "contact_point", None, "preserve_local_offsets"),
        ("model identity", "model_identity", None, "preserve"),
        ("model instance", "model_identity", None, "preserve"),
        ("asset identity", "model_identity", None, "preserve"),
        ("orientation", "orientation", None, "preserve"),
        ("geometry", "geometry", None, "preserve"),
        ("shape", "geometry", None, "preserve"),
        ("size", "geometry", None, "preserve"),
        ("scale", "geometry", None, "preserve"),
        ("dimensions", "geometry", None, "preserve"),
        ("appearance", "appearance", None, "preserve"),
        ("color", "appearance", None, "preserve"),
        ("material", "appearance", None, "preserve"),
        ("texture", "appearance", None, "preserve"),
        ("layout", "appearance", None, "preserve"),
    )
    for suffix, property_name, axis, relation in suffixes:
        if text == suffix or text.endswith(" " + suffix):
            actor = text[: -len(suffix)].strip() or None
            return _fact(
                actor=actor,
                property_name=property_name,
                axis=axis,
                relation=relation,
            )

    tokens = text.split()
    if tokens and tokens[-1] in {"position", "coordinate"}:
        stem = tokens[:-1]
        axis = "all"
        if stem and stem[-1] in {"x", "y", "z", "vertical"}:
            raw_axis = stem.pop()
            axis = "z" if raw_axis == "vertical" else raw_axis
        modifiers = {"exact", "initial", "sampled", "center", "origin"}
        actor_tokens = [token for token in stem if token not in modifiers]
        actor = " ".join(actor_tokens) or None
        return _fact(
            actor=actor,
            property_name="position",
            axis=axis,
        )
    return unknown_preservation_fact()


def normalize_preservation_facts(
    value: str | Mapping[str, Any],
) -> list[dict[str, str | None]]:
    """Normalize a typed fact or read one frozen legacy condition.

    Mapping validation is strict for new writers.  At the TaskGen read boundary
    malformed mappings and unrecognized strings become one unknown fact so
    they remain visible as unverified evidence.
    """

    if isinstance(value, Mapping):
        try:
            return [validate_preservation_fact(value)]
        except PreservationFactError:
            return [unknown_preservation_fact()]
    text = str(value).strip()
    direct = _legacy_fact(text)
    if direct["property"] != "unknown":
        return [direct]
    pieces = [
        part.strip(" ,")
        for part in text.replace(",", " and ").split(" and ")
    ]
    pieces = [part for part in pieces if part]
    if len(pieces) > 1:
        facts = [_legacy_fact(piece) for piece in pieces]
        if all(fact["property"] != "unknown" for fact in facts):
            return facts
    return [direct]


def normalize_preservation_conditions(
    value: Sequence[str | Mapping[str, Any]],
    *,
    strict_mappings: bool = False,
) -> list[dict[str, str | None]]:
    """Normalize and flatten a Proposal preservation list."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PreservationFactError("preservation conditions must be a list")
    result: list[dict[str, str | None]] = []
    for item in value:
        if isinstance(item, Mapping) and strict_mappings:
            facts = [validate_preservation_fact(item)]
        elif isinstance(item, (str, Mapping)):
            facts = normalize_preservation_facts(item)
        else:
            raise PreservationFactError(
                "each preservation condition must be a typed fact or legacy string"
            )
        result.extend(facts)
    return result


def describe_preservation_fact(value: Mapping[str, Any]) -> str:
    """Render a typed fact for a human/provider prompt without re-parsing it."""

    try:
        fact = validate_preservation_fact(value)
    except PreservationFactError:
        return "unverified legacy preservation condition"
    actor = f"{fact['actor']} " if fact["actor"] else ""
    axis = f" {fact['axis']}" if fact["axis"] else ""
    return f"{actor}{fact['property']}{axis} ({fact['relation']})"


__all__ = [
    "PreservationFactError",
    "describe_preservation_fact",
    "normalize_preservation_conditions",
    "normalize_preservation_facts",
    "unknown_preservation_fact",
    "validate_preservation_fact",
]
