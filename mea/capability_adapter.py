"""Pure declarative task and capability adapters for RoboTwin MEA.

Each trusted template resolves to one immutable-by-copy contract spanning the
Plan/TaskGen boundary and the later Tool, Execution VQA, and gate selection.
The task-level registry is the single source for public task membership,
control templates, compatibility planner delegates, and official-task visual
contracts.  It contains identifiers and JSON-compatible values only: importing
this module never calls a provider, simulator, Tool, or planner.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .aspects import AspectError, aspect_semantics, canonicalize_aspect_id


class CapabilityAdapterError(ValueError):
    """Raised when a declarative adapter contract has been changed or misused."""


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
    """Expose unchanged official execution for schema-backed ACT tasks.

    These tasks do not yet claim generated variants.  Registering only the
    official baseline lets the public router and ClaimFirst control reuse the
    common TaskGen/ToolGen boundary without inventing unsupported aspects.
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
        "control_template_id": "safety.hammer_left_camera_contact.official",
        "task_profile": "generated",
        "planner_kind": "bounded_bbh_v1",
        "max_rounds": 3,
        "vqa_questions": {},
        "vqa_metric_rules": {},
    },
    "click_bell": {
        "control_template_id": "performance.completion_time_stability.official",
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


def _validate_change_roots(
    *,
    change_scope: Any,
    allowed_roots: Any,
    changes: Any,
) -> dict[str, Any]:
    if change_scope is None:
        if allowed_roots != [] or changes != {}:
            raise CapabilityAdapterError(
                "official passthrough must have no allowed roots or changes"
            )
        return {}
    if change_scope not in {"object", "scene"}:
        raise CapabilityAdapterError("taskgen.change_scope must be object, scene, or null")
    if (
        not isinstance(allowed_roots, list)
        or not allowed_roots
        or any(not isinstance(item, str) or not item for item in allowed_roots)
        or len(allowed_roots) != len(set(allowed_roots))
    ):
        raise CapabilityAdapterError(
            "taskgen.allowed_change_roots must be a non-empty unique string list"
        )
    unknown_roots = sorted(set(allowed_roots) - set(_CHANGE_ROOT_SCOPES))
    if unknown_roots:
        raise CapabilityAdapterError(f"unknown taskgen change roots: {unknown_roots}")
    wrong_scope = sorted(
        root for root in allowed_roots if _CHANGE_ROOT_SCOPES[root] != change_scope
    )
    if wrong_scope:
        raise CapabilityAdapterError(
            f"change roots do not belong to {change_scope!r}: {wrong_scope}"
        )
    if not isinstance(changes, Mapping) or not changes:
        raise CapabilityAdapterError("generated/reused task changes must be non-empty")
    extra = sorted(set(changes) - set(allowed_roots))
    if extra:
        raise CapabilityAdapterError(f"changes exceed allowed roots: {extra}")
    missing = sorted(set(allowed_roots) - set(changes))
    if missing:
        raise CapabilityAdapterError(f"changes omit required roots: {missing}")
    return deepcopy(dict(changes))


def _validate_structure(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CONTRACT_KEYS:
        raise CapabilityAdapterError(
            f"capability contract fields must be exactly {sorted(_CONTRACT_KEYS)}"
        )
    contract = deepcopy(dict(value))
    if contract.get("schema_version") != 1:
        raise CapabilityAdapterError("capability contract schema_version must be 1")
    task_name = _text(contract.get("task_name"), field="task_name")
    template_id = _text(contract.get("template_id"), field="template_id")

    aspect = contract.get("aspect")
    if not isinstance(aspect, dict) or set(aspect) != _ASPECT_KEYS:
        raise CapabilityAdapterError(
            f"aspect fields must be exactly {sorted(_ASPECT_KEYS)}"
        )
    try:
        canonical_aspect = canonicalize_aspect_id(aspect.get("aspect_id"))
        expected_semantics = aspect_semantics(canonical_aspect)
    except AspectError as exc:
        raise CapabilityAdapterError(str(exc)) from exc
    scope = aspect.get("semantic_scope")
    if scope not in _SEMANTIC_SCOPES or scope != expected_semantics["semantic_scope"]:
        raise CapabilityAdapterError("aspect semantic_scope does not match the ontology")
    role = aspect.get("target_role")
    if role not in _TARGET_ROLES[scope]:
        raise CapabilityAdapterError(
            f"target_role {role!r} is not valid for semantic_scope {scope!r}"
        )
    aspect["aspect_id"] = canonical_aspect

    taskgen = contract.get("taskgen")
    if not isinstance(taskgen, dict) or set(taskgen) != _TASKGEN_KEYS:
        raise CapabilityAdapterError(
            f"taskgen fields must be exactly {sorted(_TASKGEN_KEYS)}"
        )
    operation = taskgen.get("operation")
    if operation not in _OPERATIONS:
        raise CapabilityAdapterError(f"unsupported taskgen operation: {operation!r}")
    expected_generation_mode = {
        "force_codegen": "force_codegen",
        "provider_scene_checker_codegen": "provider_scene_checker_codegen",
        "bounded_variant_overlay": "bounded_variant_overlay",
        "reuse_variant": "reuse",
        "official_passthrough": None,
    }[operation]
    if taskgen.get("generation_mode") != expected_generation_mode:
        raise CapabilityAdapterError(
            "taskgen operation and generation_mode do not match"
        )
    if operation == "official_passthrough":
        if taskgen.get("capability_id") != "task_execution.official_passthrough":
            raise CapabilityAdapterError(
                "official passthrough must use its trusted capability id"
            )
        for field in (
            "task_variant_id",
            "controlled_axis",
            "change_scope",
            "generation_mode",
        ):
            if taskgen.get(field) is not None:
                raise CapabilityAdapterError(
                    f"official passthrough requires taskgen.{field}=null"
                )
    else:
        for field in (
            "capability_id",
            "task_variant_id",
            "controlled_axis",
            "generation_mode",
        ):
            _text(taskgen.get(field), field=f"taskgen.{field}")
        controlled_scope = _CONTROLLED_AXIS_SCOPES.get(taskgen.get("controlled_axis"))
        if controlled_scope is None:
            raise CapabilityAdapterError(
                f"unknown controlled_axis: {taskgen.get('controlled_axis')!r}"
            )
        if controlled_scope != taskgen.get("change_scope"):
            raise CapabilityAdapterError(
                "controlled_axis and taskgen.change_scope do not match"
            )
        if scope in {"object", "scene"} and scope != taskgen.get("change_scope"):
            raise CapabilityAdapterError(
                "evaluation object/scene scope and TaskGen change scope do not match"
            )
    taskgen["changes"] = _validate_change_roots(
        change_scope=taskgen.get("change_scope"),
        allowed_roots=taskgen.get("allowed_change_roots"),
        changes=taskgen.get("changes"),
    )

    tool = contract.get("tool")
    if not isinstance(tool, dict) or set(tool) != _TOOL_KEYS:
        raise CapabilityAdapterError(f"tool fields must be exactly {sorted(_TOOL_KEYS)}")
    _text(tool.get("request_factory_id"), field="tool.request_factory_id")
    _text(tool.get("metric"), field="tool.metric")

    vqa = contract.get("vqa")
    if not isinstance(vqa, dict) or set(vqa) != _VQA_KEYS:
        raise CapabilityAdapterError(f"vqa fields must be exactly {sorted(_VQA_KEYS)}")
    phenomenon_ids = vqa.get("phenomenon_ids")
    if (
        not isinstance(phenomenon_ids, list)
        or not phenomenon_ids
        or any(not isinstance(item, str) or not item for item in phenomenon_ids)
        or len(phenomenon_ids) != len(set(phenomenon_ids))
    ):
        raise CapabilityAdapterError(
            "vqa.phenomenon_ids must be a non-empty unique string list"
        )

    gates = contract.get("required_gates")
    if (
        not isinstance(gates, list)
        or not gates
        or any(not isinstance(item, str) or not item for item in gates)
        or len(gates) != len(set(gates))
    ):
        raise CapabilityAdapterError(
            "required_gates must be a non-empty unique string list"
        )
    if operation == "official_passthrough" and "variant_spec" in gates:
        raise CapabilityAdapterError("official passthrough cannot require variant_spec")
    if operation != "official_passthrough" and "variant_spec" not in gates:
        raise CapabilityAdapterError("generated/reused variants require variant_spec")

    contract.update(
        {
            "task_name": task_name,
            "template_id": template_id,
            "aspect": aspect,
            "taskgen": taskgen,
        }
    )
    return contract


def validate_capability_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate structure, semantic scope, and exact trusted registry identity."""

    contract = _validate_structure(value)
    identity = (contract["task_name"], contract["template_id"])
    expected = _CONTRACTS.get(identity)
    if expected is None:
        raise CapabilityAdapterError(f"unknown capability adapter: {identity!r}")
    if contract != expected:
        raise CapabilityAdapterError(
            f"capability adapter contract changed for {identity!r}"
        )
    return deepcopy(contract)


def resolve_capability_contract(task_name: Any, template_id: Any) -> dict[str, Any]:
    """Resolve one task/template identity to its complete trusted contract."""

    identity = (
        _text(task_name, field="task_name"),
        _text(template_id, field="template_id"),
    )
    try:
        contract = _CONTRACTS[identity]
    except KeyError as exc:
        raise CapabilityAdapterError(f"unknown capability adapter: {identity!r}") from exc
    return validate_capability_contract(contract)


def _validate_task_adapter_structure(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TASK_ADAPTER_KEYS:
        raise CapabilityAdapterError(
            f"task adapter fields must be exactly {sorted(_TASK_ADAPTER_KEYS)}"
        )
    adapter = deepcopy(dict(value))
    if adapter.get("schema_version") != 1:
        raise CapabilityAdapterError("task adapter schema_version must be 1")
    task_name = _text(adapter.get("task_name"), field="task_adapter.task_name")
    control_template_id = _text(
        adapter.get("control_template_id"),
        field="task_adapter.control_template_id",
    )
    _text(adapter.get("task_profile"), field="task_adapter.task_profile")
    _text(adapter.get("planner_kind"), field="task_adapter.planner_kind")
    max_rounds = adapter.get("max_rounds")
    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or max_rounds <= 0
    ):
        raise CapabilityAdapterError("task_adapter.max_rounds must be positive")

    raw_contracts = adapter.get("capability_contracts")
    if not isinstance(raw_contracts, list) or not raw_contracts:
        raise CapabilityAdapterError(
            "task_adapter.capability_contracts must be non-empty"
        )
    contracts = [validate_capability_contract(item) for item in raw_contracts]
    if any(contract["task_name"] != task_name for contract in contracts):
        raise CapabilityAdapterError(
            "task adapter cannot contain another task's capability contract"
        )
    template_ids = [contract["template_id"] for contract in contracts]
    if template_ids != sorted(set(template_ids)):
        raise CapabilityAdapterError(
            "task adapter capability contracts must be unique and sorted"
        )
    if control_template_id not in template_ids:
        raise CapabilityAdapterError(
            "task adapter control_template_id must name a registered capability"
        )
    if max_rounds > len(template_ids):
        raise CapabilityAdapterError(
            "task_adapter.max_rounds exceeds its capability count"
        )

    raw_questions = adapter.get("vqa_questions")
    if not isinstance(raw_questions, Mapping):
        raise CapabilityAdapterError("task_adapter.vqa_questions must be an object")
    questions: dict[str, dict[str, Any]] = {}
    for raw_id, raw_spec in raw_questions.items():
        phenomenon_id = _text(
            raw_id, field="task_adapter.vqa_questions.phenomenon_id"
        )
        if not isinstance(raw_spec, Mapping) or set(raw_spec) != _VQA_QUESTION_KEYS:
            raise CapabilityAdapterError(
                f"VQA question {phenomenon_id!r} fields must be exactly "
                f"{sorted(_VQA_QUESTION_KEYS)}"
            )
        spec = deepcopy(dict(raw_spec))
        for field in sorted(_VQA_QUESTION_KEYS):
            _text(
                spec.get(field),
                field=f"task_adapter.vqa_questions.{phenomenon_id}.{field}",
            )
        questions[phenomenon_id] = spec

    raw_metric_rules = adapter.get("vqa_metric_rules")
    if not isinstance(raw_metric_rules, Mapping):
        raise CapabilityAdapterError(
            "task_adapter.vqa_metric_rules must be an object"
        )
    metric_rules: dict[str, list[str]] = {}
    for raw_metric, raw_ids in raw_metric_rules.items():
        metric = _text(raw_metric, field="task_adapter.vqa_metric_rules.metric")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(item, str) or not item for item in raw_ids)
            or len(raw_ids) != len(set(raw_ids))
        ):
            raise CapabilityAdapterError(
                f"task adapter VQA metric rule {metric!r} must be a "
                "non-empty unique string list"
            )
        unknown = sorted(set(raw_ids) - set(questions))
        if unknown:
            raise CapabilityAdapterError(
                f"task adapter VQA metric rule {metric!r} lacks question specs: "
                f"{unknown}"
            )
        metric_rules[metric] = list(raw_ids)

    adapter.update(
        {
            "task_name": task_name,
            "control_template_id": control_template_id,
            "capability_contracts": contracts,
            "vqa_questions": questions,
            "vqa_metric_rules": metric_rules,
        }
    )
    return adapter


def validate_task_adapter(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one complete task-level adapter against the trusted registry."""

    adapter = _validate_task_adapter_structure(value)
    task_name = adapter["task_name"]
    if task_name not in _TASK_ADAPTER_METADATA:
        raise CapabilityAdapterError(f"unknown task adapter: {task_name!r}")
    expected = _raw_task_adapter(task_name)
    if adapter != expected:
        raise CapabilityAdapterError(f"task adapter changed for {task_name!r}")
    return deepcopy(adapter)


def resolve_task_adapter(task_name: Any) -> dict[str, Any]:
    """Resolve all trusted planning/evaluation capabilities for one task."""

    normalized = _text(task_name, field="task_name")
    if normalized not in _TASK_ADAPTER_METADATA:
        raise CapabilityAdapterError(f"unknown task adapter: {normalized!r}")
    return validate_task_adapter(_raw_task_adapter(normalized))


def registered_task_adapters() -> list[dict[str, Any]]:
    """Return every task adapter in deterministic registry order."""

    return [
        resolve_task_adapter(task_name)
        for task_name in _TASK_ADAPTER_METADATA
    ]


def registered_task_names() -> tuple[str, ...]:
    """Return the single public task membership list."""

    return tuple(adapter["task_name"] for adapter in registered_task_adapters())


def registered_task_vqa_questions() -> dict[str, dict[str, Any]]:
    """Return the union of task-owned audited VQA question definitions."""

    questions: dict[str, dict[str, Any]] = {}
    for adapter in registered_task_adapters():
        for phenomenon_id, spec in adapter["vqa_questions"].items():
            previous = questions.get(phenomenon_id)
            if previous is not None and previous != spec:
                raise CapabilityAdapterError(
                    f"conflicting task VQA question: {phenomenon_id!r}"
                )
            questions[phenomenon_id] = deepcopy(spec)
    return questions


def task_vqa_metric_phenomena(task_name: Any, metric: Any) -> list[str]:
    """Resolve task-scoped VQA phenomena for a trusted metric."""

    adapter = resolve_task_adapter(task_name)
    normalized_metric = _text(metric, field="metric")
    return list(adapter["vqa_metric_rules"].get(normalized_metric, []))


def validate_contract_changes(
    contract: Mapping[str, Any], changes: Mapping[str, Any]
) -> dict[str, Any]:
    """Enforce the contract's object/scene roots on candidate TaskGen changes.

    Task-specific validators remain responsible for numeric ranges and exact
    nested fields.  This function prevents a capability from crossing the
    top-level object/scene authority boundary before those validators run.
    """

    trusted = validate_capability_contract(contract)
    return _validate_change_roots(
        change_scope=trusted["taskgen"]["change_scope"],
        allowed_roots=trusted["taskgen"]["allowed_change_roots"],
        changes=changes,
    )


def build_contract_tool_request(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the trusted Tool request named by a capability contract.

    The registry remains declarative; executable factories are imported only
    when the runtime explicitly asks to materialize a request.
    """

    trusted = validate_capability_contract(contract)
    from .toolgen import (
        bell_active_tcp_min_xy_error_tool_request,
        contact_tool_request,
        bbh_distractor_success_tool_request,
        click_bell_distractor_success_tool_request,
        hammer_left_camera_contact_count_tool_request,
        official_success_tool_request,
        pickup_to_contact_tool_request,
        time_to_success_tool_request,
        validate_tool_request,
    )

    factory_id = trusted["tool"]["request_factory_id"]
    task_name = trusted["task_name"]
    if factory_id == "contact_tool_request":
        request = contact_tool_request()
    elif factory_id == "bbh_distractor_success_tool_request":
        request = bbh_distractor_success_tool_request()
    elif factory_id == "click_bell_distractor_success_tool_request":
        request = click_bell_distractor_success_tool_request()
    elif factory_id == "pickup_to_contact_tool_request":
        request = pickup_to_contact_tool_request()
    elif factory_id == "bell_active_tcp_min_xy_error_tool_request":
        request = bell_active_tcp_min_xy_error_tool_request()
    elif factory_id == "hammer_left_camera_contact_count_tool_request":
        request = hammer_left_camera_contact_count_tool_request()
    elif factory_id == "official_success_tool_request":
        request = official_success_tool_request(task_name)
    elif factory_id == "time_to_success_tool_request":
        request = time_to_success_tool_request(task_name)
    else:  # pragma: no cover - exact registry validation makes this defensive.
        raise CapabilityAdapterError(
            f"unknown Tool request factory: {factory_id!r}"
        )
    try:
        return validate_tool_request(
            request,
            expected_metric=trusted["tool"]["metric"],
        )
    except RuntimeError as exc:
        raise CapabilityAdapterError(
            f"Tool request does not match capability contract: {exc}"
        ) from exc


def taskgen_route(contract: Mapping[str, Any]) -> str:
    """Translate a declarative operation to the existing TaskGen CLI route."""

    operation = validate_capability_contract(contract)["taskgen"]["operation"]
    return {
        "force_codegen": "force_codegen",
        "provider_scene_checker_codegen": "provider_scene_checker_codegen",
        "bounded_variant_overlay": "reuse",
        "reuse_variant": "reuse",
        "official_passthrough": "official",
    }[operation]


def registered_capability_contracts(
    task_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return all contracts in deterministic task/template order."""

    normalized_task = None if task_name is None else _text(task_name, field="task_name")
    return [
        validate_capability_contract(contract)
        for (registered_task, _template), contract in sorted(_CONTRACTS.items())
        if normalized_task is None or registered_task == normalized_task
    ]


def registered_templates(task_name: str) -> list[str]:
    """Return every template covered by one task adapter."""

    return [
        contract["template_id"]
        for contract in registered_capability_contracts(task_name)
    ]


__all__ = [
    "CapabilityAdapterError",
    "build_contract_tool_request",
    "registered_capability_contracts",
    "registered_task_adapters",
    "registered_task_names",
    "registered_task_vqa_questions",
    "registered_templates",
    "resolve_capability_contract",
    "resolve_task_adapter",
    "task_vqa_metric_phenomena",
    "taskgen_route",
    "validate_capability_contract",
    "validate_contract_changes",
    "validate_task_adapter",
]
