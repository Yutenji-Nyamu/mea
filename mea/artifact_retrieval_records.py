"""Immutable reviewed Task, Tool, and VQA record declarations.

Each trusted template resolves to one immutable-by-copy contract spanning the
Plan/TaskGen boundary and the later Tool, Execution VQA, and gate selection.
These declarations are the data behind the retrieval-only query API.
Membership here is never execution authorization and must not constrain the
concerns a production open-world Planner may propose.

The index contains identifiers and JSON-compatible values only: importing this
module never calls a provider, simulator, Tool, or planner.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .aspects import AspectError, aspect_semantics, canonicalize_aspect_id


OFFICIAL_CONTROL_TEMPLATE_ID = "task_execution.official_baseline"


class ArtifactRetrievalIndexError(ValueError):
    """Raised when a reviewed artifact record is malformed or changed."""


# Internal compatibility name while record validators are renamed gradually.
CapabilityAdapterError = ArtifactRetrievalIndexError


_CONTRACT_KEYS = {
    "schema_version",
    "task_name",
    "template_id",
    "aspect",
    "taskgen",
    "tool",
    "vqa",
    "required_gates",
}
_ASPECT_KEYS = {"aspect_id", "semantic_scope", "target_role"}
_TASKGEN_KEYS = {
    "operation",
    "capability_id",
    "task_variant_id",
    "controlled_axis",
    "change_scope",
    "generation_mode",
    "allowed_change_roots",
    "changes",
}
_TOOL_KEYS = {"request_factory_id", "metric"}
_VQA_KEYS = {"phenomenon_ids"}
_TASK_ADAPTER_KEYS = {
    "schema_version",
    "task_name",
    "control_template_id",
    "task_profile",
    "planner_kind",
    "max_rounds",
    "capability_contracts",
    "vqa_questions",
    "vqa_metric_rules",
}
_VQA_QUESTION_KEYS = {
    "question_type",
    "target_role",
    "question",
    "visual_scope",
    "numeric_authority",
}

_OPERATIONS = {
    "force_codegen",
    "provider_scene_checker_codegen",
    "bounded_variant_overlay",
    "reuse_variant",
    "official_passthrough",
}
_SEMANTIC_SCOPES = {"object", "scene", "performance", "execution", "safety"}
_TARGET_ROLES = {
    "object": {"target_object", "task_target"},
    "scene": {"scene"},
    "performance": {"execution"},
    "execution": {"execution", "task_target"},
    "safety": {"execution"},
}
_CHANGE_ROOT_SCOPES = {
    "block": "object",
    "bell": "object",
    "domain_randomization": "scene",
    "distractor": "scene",
}
_CONTROLLED_AXIS_SCOPES = {
    "object_appearance": "object",
    "object_position": "object",
    "object_instance": "object",
    "object_scale": "object",
    "robustness.scene_clutter": "scene",
    "robustness.distractor_avoidance": "scene",
    "scene_background_texture": "scene",
    "scene_lighting": "scene",
}

_GENERATED_GATES_BBH = [
    "variant_spec",
    "ast",
    "render",
    "rule",
    "scene_variant",
    "vision",
    "expert",
    "act",
    "toolkit",
    "planned_tool",
    "aggregate",
    "execution_vqa",
]
_REUSED_GATES_BBH = [
    "variant_spec",
    "render",
    "rule",
    "scene_variant",
    "vision",
    "expert",
    "act",
    "toolkit",
    "planned_tool",
    "aggregate",
    "execution_vqa",
]
_GENERATED_GATES_CLICK = [
    "variant_spec",
    "render",
    "rule",
    "scene_variant",
    "vision",
    "expert",
    "act",
    "toolkit",
    "planned_tool",
    "aggregate",
    "execution_vqa",
]
_OFFICIAL_ACT_GATES = [
    "render",
    "rule",
    "act",
    "toolkit",
    "planned_tool",
    "aggregate",
    "execution_vqa",
]

_BLUE_BLOCK = {
    "block": {
        "position_mode": "official_random",
        "yaw_mode": "official_random",
        "scale": 1.0,
        "color": [0.0, 0.2, 1.0],
    }
}
_SCALED_RED_BLOCK = {
    "block": {
        "position_mode": "official_random",
        "yaw_mode": "official_random",
        "scale": 1.2,
        "color": [1.0, 0.0, 0.0],
    }
}
_LOOKALIKE_DISTRACTOR = {
    "distractor": {
        "scene": {
            "target_name": "box",
            "distractor_name": "distractor_box",
            "target_color": [1.0, 0.0, 0.0],
            "distractor_color": [0.85, 0.05, 0.05],
            "half_size_m": [0.025, 0.025, 0.025],
            "distractor_offset_xy_m": [0.10, 0.0],
        },
        "success": {
            "target_alignment_thresholds_m": [0.025, 0.025],
            "require_target_contact": True,
            "forbid_distractor_contact": True,
            "latch_distractor_contact": True,
        },
    }
}
_LOOKALIKE_BELL_DISTRACTOR = {
    "distractor": {
        "scene": {
            "target_name": "050_bell",
            "distractor_name": "distractor_bell",
            "distractor_offset_xy_m": [0.0, 0.12],
            "instance_relation": "alternate_official_instance",
        },
        "success": {
            "target_xy_threshold_m": [0.025, 0.025],
            "target_z_threshold_m": 0.03,
            "require_correct_arm": True,
            "forbid_distractor_contact": True,
            "latch_distractor_contact": True,
        },
    }
}


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityAdapterError(f"{field} must be a non-empty string")
    return value.strip()


def _contract(
    *,
    task_name: str,
    template_id: str,
    aspect_id: str,
    target_role: str,
    operation: str,
    capability_id: str | None,
    task_variant_id: str | None,
    controlled_axis: str | None,
    change_scope: str | None,
    generation_mode: str | None,
    allowed_change_roots: list[str],
    changes: Mapping[str, Any],
    request_factory_id: str,
    metric: str,
    phenomenon_ids: list[str],
    required_gates: list[str],
) -> dict[str, Any]:
    semantics = aspect_semantics(aspect_id)
    return {
        "schema_version": 1,
        "task_name": task_name,
        "template_id": template_id,
        "aspect": {
            "aspect_id": semantics["aspect_id"],
            "semantic_scope": semantics["semantic_scope"],
            "target_role": target_role,
        },
        "taskgen": {
            "operation": operation,
            "capability_id": capability_id,
            "task_variant_id": task_variant_id,
            "controlled_axis": controlled_axis,
            "change_scope": change_scope,
            "generation_mode": generation_mode,
            "allowed_change_roots": list(allowed_change_roots),
            "changes": deepcopy(dict(changes)),
        },
        "tool": {
            "request_factory_id": request_factory_id,
            "metric": metric,
        },
        "vqa": {"phenomenon_ids": list(phenomenon_ids)},
        "required_gates": list(required_gates),
    }


def _bbh_contracts() -> list[dict[str, Any]]:
    shared = {
        "task_name": "beat_block_hammer",
        "capability_id": "object_appearance.color",
        "task_variant_id": "object_appearance.color_blue",
        "controlled_axis": "object_appearance",
        "change_scope": "object",
        "allowed_change_roots": ["block"],
        "changes": _BLUE_BLOCK,
    }
    return [
        _contract(
            task_name="beat_block_hammer",
            template_id="task_execution.official_baseline",
            aspect_id="task_execution.official_baseline",
            target_role="task_target",
            operation="official_passthrough",
            capability_id="task_execution.official_passthrough",
            task_variant_id=None,
            controlled_axis=None,
            change_scope=None,
            generation_mode=None,
            allowed_change_roots=[],
            changes={},
            request_factory_id="official_success_tool_request",
            metric="official_check_success",
            required_gates=_OFFICIAL_ACT_GATES,
            phenomenon_ids=[
                "hammer_visibly_lifted",
                "block_visibly_displaced",
            ],
        ),
        _contract(
            **shared,
            template_id="object_appearance.color_blue",
            aspect_id="object_appearance.color",
            target_role="target_object",
            operation="force_codegen",
            generation_mode="force_codegen",
            request_factory_id="contact_tool_request",
            metric="hammer_block_contact_ever",
            required_gates=_GENERATED_GATES_BBH,
            phenomenon_ids=[
                "block_color_blue",
                "hammer_visibly_lifted",
                "block_visibly_displaced",
            ],
        ),
        _contract(
            **shared,
            template_id="object_position.official_random",
            aspect_id="object_position",
            target_role="target_object",
            operation="reuse_variant",
            generation_mode="reuse",
            request_factory_id="contact_tool_request",
            metric="hammer_block_contact_ever",
            required_gates=_REUSED_GATES_BBH,
            phenomenon_ids=[
                "hammer_visibly_lifted",
                "block_visibly_displaced",
            ],
        ),
        _contract(
            **shared,
            template_id="performance.pickup_to_contact_timing",
            aspect_id="performance.pickup_to_contact_timing",
            target_role="execution",
            operation="reuse_variant",
            generation_mode="reuse",
            request_factory_id="pickup_to_contact_tool_request",
            metric="pickup_to_first_contact_time",
            required_gates=_REUSED_GATES_BBH,
            phenomenon_ids=[
                "hammer_visibly_lifted",
                "block_visibly_displaced",
            ],
        ),
        _contract(
            task_name="beat_block_hammer",
            template_id="object_scale.bounded_1_2",
            aspect_id="object_scale",
            target_role="target_object",
            operation="force_codegen",
            capability_id="object_scale.bounded",
            task_variant_id="object_scale.bounded_1_2",
            controlled_axis="object_scale",
            change_scope="object",
            generation_mode="force_codegen",
            allowed_change_roots=["block"],
            changes=_SCALED_RED_BLOCK,
            request_factory_id="contact_tool_request",
            metric="hammer_block_contact_ever",
            required_gates=_GENERATED_GATES_BBH,
            phenomenon_ids=[
                "hammer_visibly_lifted",
                "block_visibly_displaced",
            ],
        ),
        _contract(
            task_name="beat_block_hammer",
            template_id="safety.hammer_left_camera_contact.official",
            aspect_id="safety.hammer_left_camera_contact",
            target_role="execution",
            operation="official_passthrough",
            capability_id="task_execution.official_passthrough",
            task_variant_id=None,
            controlled_axis=None,
            change_scope=None,
            generation_mode=None,
            allowed_change_roots=[],
            changes={},
            request_factory_id="hammer_left_camera_contact_count_tool_request",
            metric="hammer_left_camera_contact_count",
            required_gates=_OFFICIAL_ACT_GATES,
            phenomenon_ids=["hammer_avoids_unintended_collision"],
        ),
        _contract(
            task_name="beat_block_hammer",
            template_id="robustness.distractor_avoidance.lookalike",
            aspect_id="robustness.distractor_avoidance",
            target_role="scene",
            operation="provider_scene_checker_codegen",
            capability_id="robustness.distractor_avoidance",
            task_variant_id="robustness.distractor_avoidance.lookalike",
            controlled_axis="robustness.distractor_avoidance",
            change_scope="scene",
            generation_mode="provider_scene_checker_codegen",
            allowed_change_roots=["distractor"],
            changes=_LOOKALIKE_DISTRACTOR,
            request_factory_id="bbh_distractor_success_tool_request",
            metric="bbh_target_without_distractor_success",
            required_gates=[
                "variant_spec",
                "ast",
                "render",
                "rule",
                "scene_variant",
                "expert",
                "act",
                "toolkit",
                "aggregate",
            ],
            phenomenon_ids=[
                "target_block_visible",
                "lookalike_distractor_visible",
                "distractor_not_struck",
            ],
        ),
    ]


def _click_generated_contract(
    *,
    template_id: str,
    aspect_id: str,
    target_role: str,
    capability_id: str,
    controlled_axis: str,
    change_scope: str,
    change_root: str,
    changes: Mapping[str, Any],
    request_factory_id: str,
    metric: str,
    phenomenon_ids: list[str],
) -> dict[str, Any]:
    return _contract(
        task_name="click_bell",
        template_id=template_id,
        aspect_id=aspect_id,
        target_role=target_role,
        operation="bounded_variant_overlay",
        capability_id=capability_id,
        task_variant_id=template_id,
        controlled_axis=controlled_axis,
        change_scope=change_scope,
        generation_mode="bounded_variant_overlay",
        allowed_change_roots=[change_root],
        changes=changes,
        request_factory_id=request_factory_id,
        metric=metric,
        phenomenon_ids=phenomenon_ids,
        required_gates=_GENERATED_GATES_CLICK,
    )


def _click_contracts() -> list[dict[str, Any]]:
    bell_pressed = ["bell_visibly_pressed"]
    result = [
        _contract(
            task_name="click_bell",
            template_id="robustness.distractor_avoidance.lookalike_bell",
            aspect_id="robustness.distractor_avoidance",
            target_role="scene",
            operation="provider_scene_checker_codegen",
            capability_id="robustness.distractor_avoidance",
            task_variant_id="robustness.distractor_avoidance.lookalike_bell",
            controlled_axis="robustness.distractor_avoidance",
            change_scope="scene",
            generation_mode="provider_scene_checker_codegen",
            allowed_change_roots=["distractor"],
            changes=_LOOKALIKE_BELL_DISTRACTOR,
            request_factory_id="click_bell_distractor_success_tool_request",
            metric="click_target_without_distractor_success",
            required_gates=[
                "variant_spec",
                "ast",
                "render",
                "rule",
                "scene_variant",
                "expert",
                "act",
                "toolkit",
                "aggregate",
            ],
            phenomenon_ids=[
                "bell_visibly_pressed",
                "lookalike_distractor_visible",
                "distractor_not_clicked",
            ],
        ),
        _click_generated_contract(
            template_id="object_position.left_fixed",
            aspect_id="object_position",
            target_role="task_target",
            capability_id="object_position.fixed_xy",
            controlled_axis="object_position",
            change_scope="object",
            change_root="bell",
            changes={"bell": {"position_mode": "fixed", "xy": [-0.20, -0.08]}},
            request_factory_id="bell_active_tcp_min_xy_error_tool_request",
            metric="bell_active_tcp_min_xy_error",
            phenomenon_ids=bell_pressed,
        ),
        _click_generated_contract(
            template_id="object_position.right_fixed",
            aspect_id="object_position",
            target_role="task_target",
            capability_id="object_position.fixed_xy",
            controlled_axis="object_position",
            change_scope="object",
            change_root="bell",
            changes={"bell": {"position_mode": "fixed", "xy": [0.20, -0.08]}},
            request_factory_id="bell_active_tcp_min_xy_error_tool_request",
            metric="bell_active_tcp_min_xy_error",
            phenomenon_ids=bell_pressed,
        ),
    ]
    for bell_id in (0, 1):
        result.append(
            _click_generated_contract(
                template_id=f"object_instance.base{bell_id}",
                aspect_id="object_instance",
                target_role="task_target",
                capability_id="object_instance.official_id",
                controlled_axis="object_instance",
                change_scope="object",
                change_root="bell",
                changes={
                    "bell": {
                        "position_mode": "official_random",
                        "instance_mode": "fixed",
                        "bell_id": bell_id,
                    }
                },
                request_factory_id="official_success_tool_request",
                metric="official_check_success",
                phenomenon_ids=bell_pressed,
            )
        )
    result.extend(
        [
            _click_generated_contract(
                template_id="robustness.scene_clutter.official_table",
                aspect_id="robustness.scene_clutter",
                target_role="scene",
                capability_id="robustness.scene_clutter",
                controlled_axis="robustness.scene_clutter",
                change_scope="scene",
                change_root="domain_randomization",
                changes={
                    "domain_randomization": {
                        "cluttered_table": True,
                        "clean_background_rate": 0.0,
                    }
                },
                request_factory_id="official_success_tool_request",
                metric="official_check_success",
                phenomenon_ids=[
                    "bell_visibly_pressed",
                    "bell_target_selected_among_clutter",
                ],
            ),
            _click_generated_contract(
                template_id="scene_background_texture.unseen",
                aspect_id="scene_background_texture",
                target_role="scene",
                capability_id="scene_background_texture",
                controlled_axis="scene_background_texture",
                change_scope="scene",
                change_root="domain_randomization",
                changes={
                    "domain_randomization": {
                        "random_background": True,
                        "clean_background_rate": 0.0,
                    }
                },
                request_factory_id="official_success_tool_request",
                metric="official_check_success",
                phenomenon_ids=[
                    "bell_visibly_pressed",
                    "bell_visible_with_unseen_background_texture",
                ],
            ),
            _click_generated_contract(
                template_id="scene_lighting.static_random",
                aspect_id="scene_lighting",
                target_role="scene",
                capability_id="scene_lighting",
                controlled_axis="scene_lighting",
                change_scope="scene",
                change_root="domain_randomization",
                changes={
                    "domain_randomization": {
                        "random_light": True,
                        "crazy_random_light_rate": 0.0,
                    }
                },
                request_factory_id="official_success_tool_request",
                metric="official_check_success",
                phenomenon_ids=[
                    "bell_visibly_pressed",
                    "bell_visible_under_random_lighting",
                ],
            ),
        ]
    )
    result.extend(
        [
            _contract(
                task_name="click_bell",
                template_id="performance.completion_time_stability.official",
                aspect_id="performance.completion_time_stability",
                target_role="execution",
                operation="official_passthrough",
                capability_id="task_execution.official_passthrough",
                task_variant_id=None,
                controlled_axis=None,
                change_scope=None,
                generation_mode=None,
                allowed_change_roots=[],
                changes={},
                request_factory_id="time_to_success_tool_request",
                metric="time_to_success",
                phenomenon_ids=bell_pressed,
                required_gates=_OFFICIAL_ACT_GATES,
            ),
            _contract(
                task_name="click_bell",
                template_id="task_execution.official_baseline",
                aspect_id="task_execution.official_baseline",
                target_role="task_target",
                operation="official_passthrough",
                capability_id="task_execution.official_passthrough",
                task_variant_id=None,
                controlled_axis=None,
                change_scope=None,
                generation_mode=None,
                allowed_change_roots=[],
                changes={},
                request_factory_id="official_success_tool_request",
                metric="official_check_success",
                phenomenon_ids=bell_pressed,
                required_gates=_OFFICIAL_ACT_GATES,
            ),
        ]
    )
    return result


def _generic_official_contracts() -> list[dict[str, Any]]:
    """Expose legacy retrieval hints for schema-backed ACT tasks.

    This compatibility inventory intentionally lists only the official
    baseline. It does not authorize or prohibit runtime Proposal generation;
    the Plan Agent and generic TaskGen backend own that decision.
    """

    phenomenon_by_task = {
        "adjust_bottle": ["bottle_visibly_repositioned"],
        "grab_roller": ["roller_visibly_lifted"],
        "place_phone_stand": ["phone_visibly_placed_on_stand"],
    }
    return [
        _contract(
            task_name=task_name,
            template_id="task_execution.official_baseline",
            aspect_id="task_execution.official_baseline",
            target_role="task_target",
            operation="official_passthrough",
            capability_id="task_execution.official_passthrough",
            task_variant_id=None,
            controlled_axis=None,
            change_scope=None,
            generation_mode=None,
            allowed_change_roots=[],
            changes={},
            request_factory_id="official_success_tool_request",
            metric="official_check_success",
            phenomenon_ids=phenomenon_ids,
            required_gates=_OFFICIAL_ACT_GATES,
        )
        for task_name, phenomenon_ids in phenomenon_by_task.items()
    ]


_CONTRACTS: dict[tuple[str, str], dict[str, Any]] = {}
for _item in [
    *_bbh_contracts(),
    *_click_contracts(),
    *_generic_official_contracts(),
]:
    _identity = (_item["task_name"], _item["template_id"])
    if _identity in _CONTRACTS:
        raise RuntimeError(f"duplicate capability adapter identity: {_identity!r}")
    _CONTRACTS[_identity] = _item


_TASK_ADAPTER_METADATA: dict[str, dict[str, Any]] = {
    "beat_block_hammer": {
        "control_template_id": "task_execution.official_baseline",
        "task_profile": "generated",
        "planner_kind": "bounded_bbh_v1",
        "max_rounds": 3,
        "vqa_questions": {},
        "vqa_metric_rules": {},
    },
    "click_bell": {
        "control_template_id": "task_execution.official_baseline",
        "task_profile": "adaptive_properties",
        "planner_kind": "model_click_bell_adaptive_v1",
        "max_rounds": 3,
        "vqa_questions": {
            "bell_visibly_pressed": {
                "question_type": "visible_state_change",
                "target_role": "task_target",
                "question": "Does the robot visibly press or actuate the target bell?",
                "visual_scope": "rollout_change",
                "numeric_authority": (
                    "official_core_predicate_is_authoritative_when_available_"
                    "else_official_check_success"
                ),
            },
        },
        "vqa_metric_rules": {
            "official_check_success": ["bell_visibly_pressed"],
        },
    },
    "adjust_bottle": {
        "control_template_id": "task_execution.official_baseline",
        "task_profile": "official",
        "planner_kind": "deterministic_official_task",
        "max_rounds": 1,
        "vqa_questions": {
            "bottle_visibly_repositioned": {
                "question_type": "visible_state_change",
                "target_role": "manipulated_object",
                "question": (
                    "Is the target bottle visibly moved from its initial resting "
                    "pose to the elevated side placement?"
                ),
                "visual_scope": "rollout_change",
                "numeric_authority": "official_check_success_is_authoritative",
            },
        },
        "vqa_metric_rules": {
            "official_check_success": ["bottle_visibly_repositioned"],
        },
    },
    "grab_roller": {
        "control_template_id": "task_execution.official_baseline",
        "task_profile": "official",
        "planner_kind": "deterministic_official_task",
        "max_rounds": 1,
        "vqa_questions": {
            "roller_visibly_lifted": {
                "question_type": "visible_state_change",
                "target_role": "manipulated_object",
                "question": "Is the target roller visibly lifted by both robot arms?",
                "visual_scope": "rollout_change",
                "numeric_authority": "official_check_success_is_authoritative",
            },
        },
        "vqa_metric_rules": {
            "official_check_success": ["roller_visibly_lifted"],
        },
    },
    "place_phone_stand": {
        "control_template_id": "task_execution.official_baseline",
        "task_profile": "official",
        "planner_kind": "deterministic_official_task",
        "max_rounds": 1,
        "vqa_questions": {
            "phone_visibly_placed_on_stand": {
                "question_type": "visible_state_change",
                "target_role": "task_target",
                "question": (
                    "Is the phone visibly placed and released on the phone stand?"
                ),
                "visual_scope": "rollout_change",
                "numeric_authority": "official_check_success_is_authoritative",
            },
        },
        "vqa_metric_rules": {
            "official_check_success": ["phone_visibly_placed_on_stand"],
        },
    },
}

_CONTRACT_TASK_NAMES = {task_name for task_name, _template_id in _CONTRACTS}
if _CONTRACT_TASK_NAMES != set(_TASK_ADAPTER_METADATA):
    raise RuntimeError(
        "task adapter metadata and capability task membership differ: "
        f"{sorted(_CONTRACT_TASK_NAMES ^ set(_TASK_ADAPTER_METADATA))}"
    )


def _raw_task_adapter(task_name: str) -> dict[str, Any]:
    """Assemble one task view from task metadata and the capability index."""

    metadata = _TASK_ADAPTER_METADATA[task_name]
    contracts = [
        deepcopy(contract)
        for (registered_task, _template), contract in sorted(_CONTRACTS.items())
        if registered_task == task_name
    ]
    return {
        "schema_version": 1,
        "task_name": task_name,
        "control_template_id": metadata["control_template_id"],
        "task_profile": metadata["task_profile"],
        "planner_kind": metadata["planner_kind"],
        "max_rounds": metadata["max_rounds"],
        "capability_contracts": contracts,
        "vqa_questions": deepcopy(metadata["vqa_questions"]),
        "vqa_metric_rules": deepcopy(metadata["vqa_metric_rules"]),
    }
