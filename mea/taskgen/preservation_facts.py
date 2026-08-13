"""Typed preservation facts at the TaskGen simulator boundary.

The public Proposal schema still carries ``preserved_conditions`` as strings.
This module is the migration bridge: new callers may pass the four-field fact
mapping directly, while legacy strings are accepted only when they use the
small canonical vocabulary emitted by the Plan Agent.  Unrecognized prose is
kept as an ``unknown`` fact and therefore cannot become a false hard reject.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_PROPERTIES = {
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
_AXES = {None, "all", "x", "y", "z"}
_RELATIONS = {
    "preserve",
    "preserve_local_offsets",
    "preserve_world_position",
    "required_conjunct",
    "unparsed",
}


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


def _structured_fact(value: Mapping[str, Any]) -> dict[str, str | None]:
    if set(value) != {"actor", "property", "axis", "relation"}:
        return _fact(
            actor=None,
            property_name="unknown",
            relation="unparsed",
        )
    actor = value.get("actor")
    property_name = value.get("property")
    axis = value.get("axis")
    relation = value.get("relation")
    if (
        actor is not None
        and (not isinstance(actor, str) or not actor.strip())
    ):
        actor = None
        property_name = "unknown"
    if property_name not in _PROPERTIES:
        property_name = "unknown"
    if axis not in _AXES:
        axis = None
        property_name = "unknown"
    if relation not in _RELATIONS:
        relation = "unparsed"
        property_name = "unknown"
    return _fact(
        actor=actor.strip() if isinstance(actor, str) else None,
        property_name=str(property_name),
        axis=axis if isinstance(axis, str) else None,
        relation=str(relation),
    )


def _canonical_string_fact(value: str) -> dict[str, str | None]:
    text = " ".join(value.casefold().strip().split())
    if text in {"task identity", "base task", "official task"}:
        return _fact(actor=None, property_name="task_identity")
    if text in {
        "policy checkpoint",
        "checkpoint",
        "policy weights",
    }:
        return _fact(actor=None, property_name="policy_checkpoint")
    if text in {
        "official core predicate",
        "official core predicate as a required conjunct",
        "official goal as a required conjunct",
        "official task goal as a required conjunct",
    }:
        return _fact(
            actor=None,
            property_name="official_goal",
            relation="required_conjunct",
        )
    if text in {
        "official success semantics",
        "success semantics",
        "success predicate",
        "goal semantics",
        "checker semantics",
    }:
        return _fact(actor=None, property_name="checker_semantics")

    contact = re.fullmatch(
        r"(?:(?P<actor>[a-z0-9_-]+) )?contact(?:-point| point)? "
        r"(?P<scope>local offsets|references|world position)",
        text,
    )
    if contact:
        world = contact.group("scope") == "world position"
        return _fact(
            actor=contact.group("actor"),
            property_name="contact_point",
            relation=(
                "preserve_world_position"
                if world
                else "preserve_local_offsets"
            ),
        )
    if text in {"contact point", "contact-point"}:
        return _fact(
            actor=None,
            property_name="contact_point",
            relation="preserve_local_offsets",
        )

    position = re.fullmatch(
        r"(?:exact )?(?:initial |sampled )?"
        r"(?:(?P<actor>[a-z0-9_-]+) )?"
        r"(?:(?:center|origin) )?"
        r"(?:(?P<axis>x|y|z|vertical)(?:-axis| axis)? )?"
        r"(?:position|coordinate)",
        text,
    )
    if not position:
        position = re.fullmatch(
            r"(?:center|origin) position along (?:the )?"
            r"(?P<axis>x|y|z|vertical)(?:-axis| axis)?",
            text,
        )
    if position:
        axis = position.groupdict().get("axis") or "all"
        return _fact(
            actor=position.groupdict().get("actor"),
            property_name="position",
            axis="z" if axis == "vertical" else axis,
        )
    if text == "height":
        return _fact(actor=None, property_name="position", axis="z")

    orientation = re.fullmatch(
        r"(?:(?P<actor>[a-z0-9_-]+) )?orientation", text
    )
    if orientation:
        return _fact(
            actor=orientation.group("actor"),
            property_name="orientation",
        )
    model = re.fullmatch(
        r"(?:(?P<actor>[a-z0-9_-]+) )?"
        r"(?:model identity|model instance|asset identity)",
        text,
    )
    if model:
        return _fact(
            actor=model.group("actor"),
            property_name="model_identity",
        )
    geometry = re.fullmatch(
        r"(?:(?P<actor>[a-z0-9_-]+) )?"
        r"(?:geometry|shape|size|scale|dimensions?)",
        text,
    )
    if geometry:
        return _fact(
            actor=geometry.group("actor"),
            property_name="geometry",
        )
    appearance = re.fullmatch(
        r"(?:(?P<actor>[a-z0-9_-]+) )?"
        r"(?:appearance|color|material|texture|layout)",
        text,
    )
    if appearance:
        return _fact(
            actor=appearance.group("actor"),
            property_name="appearance",
        )
    return _fact(
        actor=None,
        property_name="unknown",
        relation="unparsed",
    )


def normalize_preservation_facts(
    value: str | Mapping[str, Any],
) -> list[dict[str, str | None]]:
    """Normalize one Proposal condition into executable atomic facts.

    Canonical entries are atomic.  A very small compound adapter exists for
    historical Proposal strings; it is deliberately not a general natural-
    language classifier.
    """

    if isinstance(value, Mapping):
        return [_structured_fact(value)]
    text = str(value).strip()
    direct = _canonical_string_fact(text)
    if direct["property"] != "unknown":
        return [direct]
    pieces = [
        part.strip(" ,")
        for part in re.split(r"\s*(?:,|\band\b)\s*", text, flags=re.I)
        if part.strip(" ,")
    ]
    if len(pieces) > 1:
        facts = [_canonical_string_fact(piece) for piece in pieces]
        if all(fact["property"] != "unknown" for fact in facts):
            return facts
    return [direct]


__all__ = ["normalize_preservation_facts"]
