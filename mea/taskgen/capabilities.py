"""Trusted TaskGen capabilities and the shared VariantSpec v2 envelope.

The catalog describes planning/generation authority.  Telemetry TaskSchemas
remain separate so adding a generated capability cannot invalidate Tool caches.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping


class CapabilityError(ValueError):
    """Raised when a task capability or VariantSpec envelope is invalid."""


EXPERIMENTAL_SUCCESS_PRESERVE_MARKER = "compiled_experimental_success_spec"


TASK_CAPABILITIES: dict[str, dict[str, dict[str, Any]]] = {
    "beat_block_hammer": {
        "object_appearance.color": {
            "controlled_axis": "object_appearance",
            "allowed_change_roots": ["block"],
            "generation_mode": "force_codegen",
            "allowed_generation_modes": ["force_codegen", "reuse"],
            "default_metric": "hammer_block_contact_ever",
            "preserve": [
                "official_position_sampling",
                "official_yaw_sampling",
                "play_once",
                "check_success",
                "checkpoint",
            ],
        },
        "object_scale.bounded": {
            "controlled_axis": "object_scale",
            "allowed_change_roots": ["block"],
            "generation_mode": "force_codegen",
            "allowed_generation_modes": ["force_codegen", "reuse"],
            "default_metric": "hammer_block_contact_ever",
            "scale_bounds": [0.75, 1.25],
            "preserve": [
                "official_position_sampling",
                "official_yaw_sampling",
                "official_block_color",
                "play_once",
                "check_success_semantics",
                "checkpoint",
            ],
        },
    },
    "click_bell": {
        "object_position.fixed_xy": {
            "controlled_axis": "object_position",
            "allowed_change_roots": ["bell"],
            "generation_mode": "bounded_variant_overlay",
            "allowed_generation_modes": ["bounded_variant_overlay"],
            "default_metric": "bell_active_tcp_min_xy_error",
            "preserve": [
                "official_pose_rng_consumption",
                "official_instance_rng_consumption",
                "official_bell_assets",
                "play_once",
                "check_success",
                "checkpoint",
            ],
        },
        "object_instance.official_id": {
            "controlled_axis": "object_instance",
            "allowed_change_roots": ["bell"],
            "generation_mode": "bounded_variant_overlay",
            "allowed_generation_modes": ["bounded_variant_overlay"],
            "default_metric": "official_check_success",
            "preserve": [
                "official_pose_rng_consumption",
                "official_instance_rng_consumption",
                "official_bell_assets",
                "play_once",
                "check_success",
                "checkpoint",
            ],
        },
        "robustness.scene_clutter": {
            "controlled_axis": "robustness.scene_clutter",
            "allowed_change_roots": ["domain_randomization"],
            "generation_mode": "bounded_variant_overlay",
            "allowed_generation_modes": ["bounded_variant_overlay"],
            "default_metric": "official_check_success",
            "preserve": [
                "official_pose_sampling",
                "official_instance_sampling",
                "official_bell_assets",
                "official_clutter_generator",
                "play_once",
                "check_success",
                "checkpoint",
            ],
        },
        "scene_background_texture": {
            "controlled_axis": "scene_background_texture",
            "allowed_change_roots": ["domain_randomization"],
            "generation_mode": "bounded_variant_overlay",
            "allowed_generation_modes": ["bounded_variant_overlay"],
            "default_metric": "official_check_success",
            "preserve": [
                "official_pose_sampling",
                "official_instance_sampling",
                "official_bell_assets",
                "official_background_texture_loader",
                "eval_mode_unseen_texture_split",
                "play_once",
                "check_success",
                "checkpoint",
            ],
        },
        "scene_lighting": {
            "controlled_axis": "scene_lighting",
            "allowed_change_roots": ["domain_randomization"],
            "generation_mode": "bounded_variant_overlay",
            "allowed_generation_modes": ["bounded_variant_overlay"],
            "default_metric": "official_check_success",
            "preserve": [
                "official_pose_sampling",
                "official_instance_sampling",
                "official_bell_assets",
                "official_light_randomizer",
                "static_per_episode_lighting",
                "play_once",
                "check_success",
                "checkpoint",
            ],
        },
    },
}


def get_capability(task_name: str, capability_id: str) -> dict[str, Any]:
    try:
        value = TASK_CAPABILITIES[str(task_name)][str(capability_id)]
    except KeyError as exc:
        raise CapabilityError(
            f"unknown TaskGen capability {task_name!r}/{capability_id!r}"
        ) from exc
    return deepcopy(value)


def capability_card(task_name: str) -> dict[str, Any]:
    capabilities = TASK_CAPABILITIES.get(str(task_name))
    if not capabilities:
        raise CapabilityError(f"task has no generated capabilities: {task_name!r}")
    return {
        "schema_version": 1,
        "task_name": str(task_name),
        "capabilities": [
            {"capability_id": capability_id, **deepcopy(value)}
            for capability_id, value in capabilities.items()
        ],
    }


def build_variant_spec(
    *,
    task_name: str,
    variant_id: str,
    capability_id: str,
    intent: str,
    changes: Mapping[str, Any],
    generation_mode: str | None = None,
    preserve_success_semantics: bool = True,
) -> dict[str, Any]:
    """Inject trusted capability fields around task-validated changes."""

    normalized_variant = str(variant_id).strip()
    normalized_intent = str(intent).strip()
    if not normalized_variant or not normalized_intent:
        raise CapabilityError("variant_id and intent must be non-empty")
    if not isinstance(changes, Mapping) or not changes:
        raise CapabilityError("changes must be a non-empty object")
    capability = get_capability(task_name, capability_id)
    change_roots = set(changes)
    allowed_change_roots = set(capability["allowed_change_roots"])
    if not change_roots <= allowed_change_roots:
        raise CapabilityError(
            f"changes for {capability_id!r} must stay within roots "
            f"{sorted(allowed_change_roots)}; got {sorted(change_roots)}"
        )
    if task_name == "beat_block_hammer" and capability_id in {
        "object_appearance.color",
        "object_scale.bounded",
    }:
        block = changes.get("block")
        if not isinstance(block, Mapping):
            raise CapabilityError(f"{capability_id} requires changes.block")
        color = block.get("color")
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            raise CapabilityError(f"{capability_id} block.color must have three channels")
        if any(
            isinstance(channel, bool) or not isinstance(channel, (int, float))
            for channel in color
        ):
            raise CapabilityError(f"{capability_id} block.color must be finite numeric")
        normalized_color = [float(channel) for channel in color]
        if not all(math.isfinite(channel) for channel in normalized_color):
            raise CapabilityError(f"{capability_id} block.color must be finite numeric")
        if any(channel < 0.0 or channel > 1.0 for channel in normalized_color):
            raise CapabilityError(f"{capability_id} block.color must be within [0, 1]")
    if task_name == "beat_block_hammer" and capability_id == "object_scale.bounded":
        block = changes.get("block")
        scale = block.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise CapabilityError("object_scale.bounded block.scale must be numeric")
        scale = float(scale)
        low, high = capability["scale_bounds"]
        if not math.isfinite(scale) or not low <= scale <= high:
            raise CapabilityError(
                f"object_scale.bounded scale must be within [{low}, {high}]"
            )
        if block.get("position_mode", "official_random") != "official_random":
            raise CapabilityError("object_scale.bounded preserves official position sampling")
        if block.get("yaw_mode", "official_random") != "official_random":
            raise CapabilityError("object_scale.bounded preserves official yaw sampling")
        if normalized_color != [1.0, 0.0, 0.0]:
            raise CapabilityError("object_scale.bounded preserves official block color")
    if task_name == "beat_block_hammer" and capability_id == "object_appearance.color":
        raw_scale = block.get("scale", 1.0)
        if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
            raise CapabilityError("object_appearance.color block.scale must be finite numeric")
        scale = float(raw_scale)
        if not math.isfinite(scale) or abs(scale - 1.0) > 1e-12:
            raise CapabilityError("object_appearance.color preserves official block scale")
    resolved_mode = str(generation_mode or capability["generation_mode"])
    if resolved_mode not in capability["allowed_generation_modes"]:
        raise CapabilityError(
            f"generation mode {resolved_mode!r} is not allowed by {capability_id!r}"
        )
    preserve = deepcopy(capability["preserve"])
    if not preserve_success_semantics:
        if not (
            task_name == "beat_block_hammer"
            and capability_id == "object_appearance.color"
            and resolved_mode == "force_codegen"
        ):
            raise CapabilityError(
                "replacement SuccessSpec VariantSpec is capability-gated to "
                "beat_block_hammer/object_appearance.color force_codegen"
            )
        if "check_success" not in preserve:
            raise CapabilityError(
                "experimental SuccessSpec capability lacks official check_success "
                "preservation marker"
            )
        preserve = [
            (
                EXPERIMENTAL_SUCCESS_PRESERVE_MARKER
                if item == "check_success"
                else item
            )
            for item in preserve
        ]
    return {
        "schema_version": 2,
        "task_name": str(task_name),
        "variant_id": normalized_variant,
        "capability_id": str(capability_id),
        "intent": normalized_intent,
        "controlled_axis": capability["controlled_axis"],
        "generation_mode": resolved_mode,
        "changes": deepcopy(dict(changes)),
        "preserve": preserve,
    }


def validate_variant_spec_envelope(spec: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "task_name",
        "variant_id",
        "capability_id",
        "intent",
        "controlled_axis",
        "generation_mode",
        "changes",
        "preserve",
    }
    if not isinstance(spec, Mapping) or set(spec) != required:
        raise CapabilityError(
            f"VariantSpec v2 fields must be exactly {sorted(required)}"
        )
    if spec.get("schema_version") != 2:
        raise CapabilityError("VariantSpec schema_version must be 2")
    raw_preserve = spec.get("preserve")
    experimental_success = (
        isinstance(raw_preserve, list)
        and EXPERIMENTAL_SUCCESS_PRESERVE_MARKER in raw_preserve
    )
    expected = build_variant_spec(
        task_name=str(spec.get("task_name")),
        variant_id=str(spec.get("variant_id") or ""),
        capability_id=str(spec.get("capability_id") or ""),
        intent=str(spec.get("intent") or ""),
        changes=spec.get("changes") if isinstance(spec.get("changes"), Mapping) else {},
        generation_mode=str(spec.get("generation_mode") or ""),
        preserve_success_semantics=not experimental_success,
    )
    if dict(spec) != expected:
        raise CapabilityError(
            "VariantSpec trusted axis, generation mode, or preserve contract changed"
        )
    return expected


def load_legacy_variant_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Read v2 directly or upgrade the two committed legacy families."""

    if spec.get("schema_version") == 2:
        return validate_variant_spec_envelope(spec)
    task_name = str(spec.get("task_name") or "")
    changes = spec.get("changes")
    if not isinstance(changes, Mapping):
        raise CapabilityError("legacy VariantSpec changes must be an object")
    if task_name == "beat_block_hammer":
        capability_id = "object_appearance.color"
        variant_id = str(spec.get("variant_id") or "object_appearance.color_custom")
    elif task_name == "click_bell":
        controlled_axis = str(spec.get("controlled_axis") or "")
        capability_id = {
            "object_position": "object_position.fixed_xy",
            "object_instance": "object_instance.official_id",
            "robustness.scene_clutter": "robustness.scene_clutter",
            "scene_background_texture": "scene_background_texture",
            "scene_lighting": "scene_lighting",
        }.get(controlled_axis, "")
        variant_id = str(spec.get("variant_id") or f"{controlled_axis}.legacy")
    else:
        raise CapabilityError(f"unsupported legacy VariantSpec task: {task_name!r}")
    return build_variant_spec(
        task_name=task_name,
        variant_id=variant_id,
        capability_id=capability_id,
        intent=str(spec.get("intent") or "legacy_generated_variant"),
        changes=changes,
    )
