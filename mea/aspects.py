"""Canonical evaluation-aspect ontology used by trusted MEA adapters.

The model may use one of a small number of explicitly registered aliases, but
runtime artifacts always carry the existing canonical identifiers.  This is
deliberately not fuzzy matching: unknown text is either rejected or preserved
only when a caller explicitly needs to report an unsupported capability.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


class AspectError(ValueError):
    """Raised when an aspect identifier or semantic declaration is invalid."""


_ASPECT_ONTOLOGY: dict[str, dict[str, Any]] = {
    "object_appearance.color": {
        "semantic_scope": "object",
        "aliases": ["object.color", "appearance.color", "object_color"],
    },
    "object_appearance.material_gloss": {
        "semantic_scope": "object",
        "aliases": ["object.material_gloss", "appearance.material_gloss"],
    },
    "object_appearance.texture": {
        "semantic_scope": "object",
        "aliases": ["object.texture", "appearance.texture"],
    },
    "object_position": {
        "semantic_scope": "object",
        "aliases": ["object.position", "object_position_generalization"],
    },
    "object_instance": {
        "semantic_scope": "object",
        "aliases": ["object.instance", "object_instance_generalization"],
    },
    "object_physics.mass": {
        "semantic_scope": "physics",
        "aliases": ["object.mass", "physics.mass"],
    },
    "object_scale": {
        "semantic_scope": "object",
        "aliases": ["object.scale", "object_size"],
    },
    "camera_viewpoint": {
        "semantic_scope": "camera",
        "aliases": ["camera.viewpoint", "viewpoint"],
    },
    "occlusion.target_contact": {
        "semantic_scope": "scene",
        "aliases": ["target.occlusion", "occlusion.contact_target"],
    },
    "robustness.scene_clutter": {
        "semantic_scope": "scene",
        "aliases": [
            "scene.clutter",
            "scene_clutter",
            "robustness.clutter",
            "clutter.target_selection",
        ],
    },
    "robustness.distractor_avoidance": {
        "semantic_scope": "scene",
        "aliases": [
            "scene.distractor_avoidance",
            "distractor_avoidance",
            "robustness.lookalike_distractor",
        ],
    },
    "scene_background_texture": {
        "semantic_scope": "scene",
        "aliases": ["scene.background_texture", "background_texture"],
    },
    "scene_lighting": {
        "semantic_scope": "scene",
        "aliases": ["scene.lighting", "lighting"],
    },
    "performance.pickup_to_contact_timing": {
        "semantic_scope": "performance",
        "aliases": ["performance.pickup_to_contact", "pickup_to_contact_timing"],
    },
    "performance.completion_time_stability": {
        "semantic_scope": "performance",
        "aliases": ["performance.completion_time", "completion_time_stability"],
    },
    "performance.motion_smoothness": {
        "semantic_scope": "performance",
        "aliases": ["motion_smoothness"],
    },
    "performance.path_efficiency": {
        "semantic_scope": "performance",
        "aliases": ["path_efficiency"],
    },
    "language.paraphrase_consistency": {
        "semantic_scope": "language",
        "aliases": ["language.paraphrase", "instruction_paraphrase"],
    },
    "safety.boundary_clearance": {
        "semantic_scope": "safety",
        "aliases": ["boundary_clearance"],
    },
    "safety.unintended_contact": {
        "semantic_scope": "safety",
        "aliases": ["unintended_contact"],
    },
    "safety.hammer_left_camera_contact": {
        "semantic_scope": "safety",
        "aliases": ["hammer_left_camera_contact"],
    },
    "conclusion.multi_task_consistency": {
        "semantic_scope": "conclusion",
        "aliases": ["multi_task_consistency"],
    },
    "task_execution.official_baseline": {
        "semantic_scope": "execution",
        "aliases": ["task_execution.official", "official_baseline"],
    },
}


def _alias_index() -> dict[str, str]:
    result: dict[str, str] = {}
    for canonical, semantics in _ASPECT_ONTOLOGY.items():
        for identifier in (canonical, *semantics["aliases"]):
            key = identifier.casefold()
            previous = result.get(key)
            if previous is not None and previous != canonical:
                raise RuntimeError(
                    f"aspect alias {identifier!r} maps to both {previous!r} "
                    f"and {canonical!r}"
                )
            result[key] = canonical
    return result


_ALIASES = _alias_index()


def canonicalize_aspect_id(value: Any, *, allow_unknown: bool = False) -> str:
    """Return one allowlisted canonical id without performing fuzzy matching."""

    if not isinstance(value, str) or not value.strip():
        raise AspectError("aspect_id must be a non-empty string")
    normalized = value.strip()
    canonical = _ALIASES.get(normalized.casefold())
    if canonical is not None:
        return canonical
    if allow_unknown:
        return normalized
    raise AspectError(f"unknown aspect_id: {normalized!r}")


def canonicalize_aspect_ids(
    values: Iterable[Any], *, allow_unknown: bool = False
) -> list[str]:
    """Canonicalize a sequence and reject aliases that collapse to duplicates."""

    if isinstance(values, (str, bytes)):
        raise AspectError("aspect_ids must be an iterable of identifiers")
    try:
        normalized = [
            canonicalize_aspect_id(value, allow_unknown=allow_unknown)
            for value in values
        ]
    except TypeError as exc:
        raise AspectError("aspect_ids must be an iterable of identifiers") from exc
    if len(normalized) != len(set(normalized)):
        raise AspectError("aspect_ids contain duplicates after canonicalization")
    return normalized


def aspect_semantics(value: Any) -> dict[str, Any]:
    """Return a copy of the trusted semantic declaration for one aspect."""

    canonical = canonicalize_aspect_id(value)
    return {
        "aspect_id": canonical,
        **deepcopy(_ASPECT_ONTOLOGY[canonical]),
    }


def public_aspect_ontology() -> list[dict[str, Any]]:
    """Expose the deterministic ontology without granting mutation authority."""

    return [aspect_semantics(identifier) for identifier in _ASPECT_ONTOLOGY]


__all__ = [
    "AspectError",
    "aspect_semantics",
    "canonicalize_aspect_id",
    "canonicalize_aspect_ids",
    "public_aspect_ontology",
]
