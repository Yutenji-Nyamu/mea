"""Trusted TaskGen capabilities and the shared VariantSpec v2 envelope.

The catalog describes planning/generation authority.  Telemetry TaskSchemas
remain separate so adding a generated capability cannot invalidate Tool caches.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


class CapabilityError(ValueError):
    """Raised when a task capability or VariantSpec envelope is invalid."""


TASK_CAPABILITIES: dict[str, dict[str, dict[str, Any]]] = {
    "beat_block_hammer": {
        "object_appearance.color": {
            "controlled_axis": "object_appearance",
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
        }
    },
    "click_bell": {
        "object_position.fixed_xy": {
            "controlled_axis": "object_position",
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
) -> dict[str, Any]:
    """Inject trusted capability fields around task-validated changes."""

    normalized_variant = str(variant_id).strip()
    normalized_intent = str(intent).strip()
    if not normalized_variant or not normalized_intent:
        raise CapabilityError("variant_id and intent must be non-empty")
    if not isinstance(changes, Mapping) or not changes:
        raise CapabilityError("changes must be a non-empty object")
    capability = get_capability(task_name, capability_id)
    resolved_mode = str(generation_mode or capability["generation_mode"])
    if resolved_mode not in capability["allowed_generation_modes"]:
        raise CapabilityError(
            f"generation mode {resolved_mode!r} is not allowed by {capability_id!r}"
        )
    return {
        "schema_version": 2,
        "task_name": str(task_name),
        "variant_id": normalized_variant,
        "capability_id": str(capability_id),
        "intent": normalized_intent,
        "controlled_axis": capability["controlled_axis"],
        "generation_mode": resolved_mode,
        "changes": deepcopy(dict(changes)),
        "preserve": deepcopy(capability["preserve"]),
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
    expected = build_variant_spec(
        task_name=str(spec.get("task_name")),
        variant_id=str(spec.get("variant_id") or ""),
        capability_id=str(spec.get("capability_id") or ""),
        intent=str(spec.get("intent") or ""),
        changes=spec.get("changes") if isinstance(spec.get("changes"), Mapping) else {},
        generation_mode=str(spec.get("generation_mode") or ""),
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
