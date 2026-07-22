"""Pure declarative capability adapters for the two generated MEA families.

Each trusted template resolves to one immutable-by-copy contract spanning the
Plan/TaskGen boundary and the later Tool, Execution VQA, and gate selection.
The registry contains identifiers and JSON-compatible values only: importing
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

_OPERATIONS = {
    "force_codegen",
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
}
_CONTROLLED_AXIS_SCOPES = {
    "object_appearance": "object",
    "object_position": "object",
    "object_instance": "object",
    "object_scale": "object",
    "robustness.scene_clutter": "scene",
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


_CONTRACTS: dict[tuple[str, str], dict[str, Any]] = {}
for _item in [*_bbh_contracts(), *_click_contracts()]:
    _identity = (_item["task_name"], _item["template_id"])
    if _identity in _CONTRACTS:
        raise RuntimeError(f"duplicate capability adapter identity: {_identity!r}")
    _CONTRACTS[_identity] = _item


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
    "registered_templates",
    "resolve_capability_contract",
    "taskgen_route",
    "validate_capability_contract",
    "validate_contract_changes",
]
