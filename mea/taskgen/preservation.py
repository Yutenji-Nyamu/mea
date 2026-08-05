"""Simulator-authoritative TaskGen preservation checks."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from typing import Any

_FROZEN_PRESERVATION_TERMS = (
    "task identity",
    "base task",
    "official task",
    "policy checkpoint",
    "checkpoint",
    "policy weights",
    "random seed",
    "seed",
    "action schema",
    "action interface",
    "robot configuration",
    "robot identity",
    "gripper configuration",
    "gripper identity",
    "task instruction",
    "language instruction",
    "robot state",
    "timing",
    "策略检查点",
    "随机种子",
    "动作接口",
    "机器人配置",
    "任务指令",
)
_VISUAL_PRESERVATION_TERMS = (
    "appearance",
    "color",
    "material",
    "texture",
    "lighting",
    "background",
    "surroundings",
    "environment",
    "camera",
    "visible",
    "layout",
    "table",
    "support",
    "floor",
    "wall",
    "interaction target",
    "target identity",
    "外观",
    "颜色",
    "材质",
    "纹理",
    "光照",
    "背景",
    "环境",
    "相机",
    "布局",
    "目标身份",
)
_SIMULATOR_STATE_PRESERVATION_TERMS = (
    "spatial",
    "contact point",
    "contact-point",
    "contact location",
    "center",
    "centre",
    "world position",
    "world-position",
    "pose",
    "position",
    "placement",
    "location",
    "orientation",
    "height",
    "z coordinate",
    "空间",
    "接触点",
    "接触位置",
    "中心",
    "世界位置",
    "位姿",
    "位置",
    "姿态",
    "高度",
)
_GEOMETRY_PRESERVATION_TERMS = (
    "geometry",
    "shape",
    "size",
    "scale",
    "dimension",
    "几何",
    "形状",
    "大小",
    "尺寸",
    "比例",
)
_CHECKER_PRESERVATION_TERMS = (
    "success semantics",
    "success criterion",
    "success criteria",
    "checker",
    "check_success",
    "task goal",
    "task objective",
    "task semantics",
    "goal semantics",
    "outcome semantics",
    "成功语义",
    "成功判据",
    "成功标准",
    "任务目标",
    "任务语义",
    "目标语义",
)
_OFFICIAL_CORE_CONJUNCT_TERMS = (
    "official core predicate",
    "official goal as a required conjunct",
    "official task goal as a required conjunct",
    "official goal as a necessary condition",
    "official task goal as a necessary condition",
)

_POSITION_ABS_TOLERANCE_M = 1e-5
_QUATERNION_COMPONENT_ABS_TOLERANCE = 5e-4


def _numeric_vectors_close(
    official: Any,
    generated: Any,
    *,
    absolute_tolerance: float,
    sign_invariant: bool = False,
) -> bool:
    """Compare finite simulator vectors with an explicit probe tolerance."""

    if (
        not isinstance(official, list)
        or not isinstance(generated, list)
        or len(official) != len(generated)
        or not official
    ):
        return False
    try:
        left = [float(item) for item in official]
        right = [float(item) for item in generated]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(item) for item in (*left, *right)):
        return False
    direct_error = max(abs(a - b) for a, b in zip(left, right))
    if not sign_invariant:
        return direct_error <= absolute_tolerance
    negated_error = max(abs(a + b) for a, b in zip(left, right))
    return min(direct_error, negated_error) <= absolute_tolerance


def _same_seed_tracked_actor_state(
    official_setup: Mapping[str, Any] | None,
    generated_setup: Mapping[str, Any] | None,
    condition: str,
) -> tuple[bool | None, str]:
    """Compare spatial facts using same-seed simulator state, never RGB."""

    if not isinstance(official_setup, Mapping) or not isinstance(
        generated_setup, Mapping
    ):
        return None, "no_same_seed_simulator_state_authority"
    official_seed = official_setup.get("seed")
    generated_seed = generated_setup.get("seed")
    if (
        official_seed is None
        or generated_seed is None
        or official_seed != generated_seed
    ):
        return None, "no_same_seed_simulator_state_authority"

    def actors_by_id(
        setup: Mapping[str, Any],
    ) -> dict[str, Mapping[str, Any]] | None:
        actors = setup.get("tracked_actors")
        if not isinstance(actors, list):
            return None
        result: dict[str, Mapping[str, Any]] = {}
        for actor in actors:
            if not isinstance(actor, Mapping):
                return None
            actor_id = actor.get("id")
            if not isinstance(actor_id, str) or not actor_id:
                return None
            if actor_id in result:
                return None
            result[actor_id] = actor
        return result

    official_actors = actors_by_id(official_setup)
    generated_actors = actors_by_id(generated_setup)
    if (
        official_actors is None
        or generated_actors is None
        or not official_actors
        or not set(official_actors).issubset(generated_actors)
    ):
        return None, "no_comparable_tracked_actor_state"

    lowered = condition.casefold()
    requires_contact = "contact" in lowered
    non_contact_spatial = re.sub(
        r"\bcontact(?:[\s-]+point)?(?:[\s-]+world)?"
        r"[\s-]+(?:position|location|coordinate)s?\b"
        r"|\bcontact[\s-]+point\b",
        "",
        lowered,
    )
    requires_vertical_axis = any(
        marker in lowered
        for marker in (
            "vertical axis",
            "vertical coordinate",
            "z-axis",
            "z axis",
            "z-coordinate",
            "垂直轴",
            "竖直轴",
            "z轴",
        )
    )
    requires_actor_position = (
        any(
            term in (
                non_contact_spatial
                if requires_contact
                else lowered
            )
            for term in (
                "position",
                "location",
                "coordinate",
                "placement",
                "spatial",
            )
        )
    ) or any(
        term in non_contact_spatial
        for term in ("center", "centre", "origin", "pose")
    )
    requires_orientation = (
        "orientation" in lowered or "pose" in lowered
    )
    requires_height = (
        "height" in lowered
        or "z coordinate" in lowered
        or "高度" in lowered
        or requires_vertical_axis
    )
    if requires_vertical_axis:
        # "Position along the vertical axis" constrains only z.  Treating it
        # as full xyz preservation would reject the horizontal perturbation
        # that the same Proposal explicitly requests.
        requires_actor_position = False
    component_results: list[bool | None] = []
    components: list[str] = []

    if requires_contact:
        comparable = [
            (
                official_actors[actor_id].get("contact_points"),
                generated_actors[actor_id].get("contact_points"),
            )
            for actor_id in official_actors
        ]
        if not any(
            isinstance(official, Mapping) and bool(official)
            for official, _generated in comparable
        ):
            component_results.append(None)
            components.append("contact_points")
        else:
            component_results.append(
                all(
                    official == generated
                    for official, generated in comparable
                )
            )
            components.append("contact_points")

    fields: list[str] = []
    if requires_actor_position:
        fields.append("position")
    if requires_orientation:
        fields.append("quaternion")
    for field in fields:
        components.append(field)
        if any(
            field not in actor
            for actor in (
                *official_actors.values(),
                *(
                    generated_actors[actor_id]
                    for actor_id in official_actors
                ),
            )
        ):
            component_results.append(None)
            continue
        component_results.append(
            all(
                _numeric_vectors_close(
                    official_actors[actor_id][field],
                    generated_actors[actor_id][field],
                    absolute_tolerance=(
                        _QUATERNION_COMPONENT_ABS_TOLERANCE
                        if field == "quaternion"
                        else _POSITION_ABS_TOLERANCE_M
                    ),
                    sign_invariant=(field == "quaternion"),
                )
                for actor_id in official_actors
            )
        )
    if requires_height and not requires_actor_position:
        components.append("position_z")
        positions = [
            (
                official_actors[actor_id].get("position"),
                generated_actors[actor_id].get("position"),
            )
            for actor_id in official_actors
        ]
        if any(
            not isinstance(official, list)
            or not isinstance(generated, list)
            or len(official) < 3
            or len(generated) < 3
            for official, generated in positions
        ):
            component_results.append(None)
        else:
            component_results.append(
                all(
                    _numeric_vectors_close(
                        [official[2]],
                        [generated[2]],
                        absolute_tolerance=_POSITION_ABS_TOLERANCE_M,
                    )
                    for official, generated in positions
                )
            )

    if not component_results:
        return None, "no_comparable_tracked_actor_state"
    verified: bool | None
    if any(result is False for result in component_results):
        verified = False
    elif any(result is None for result in component_results):
        verified = None
    else:
        verified = True
    return (
        verified,
        "same_seed_simulator_state:tracked_actors."
        + "+".join(components),
    )


def _same_seed_tracked_actor_geometry(
    official_setup: Mapping[str, Any] | None,
    generated_setup: Mapping[str, Any] | None,
) -> tuple[bool | None, str]:
    """Compare collision dimensions/scales from simulator probes, never RGB."""

    if not isinstance(official_setup, Mapping) or not isinstance(
        generated_setup, Mapping
    ):
        return None, "no_same_seed_simulator_geometry_authority"
    if (
        official_setup.get("seed") is None
        or official_setup.get("seed") != generated_setup.get("seed")
    ):
        return None, "no_same_seed_simulator_geometry_authority"

    def geometry_by_id(
        setup: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        actors = setup.get("tracked_actors")
        if not isinstance(actors, list):
            return None
        result: dict[str, Any] = {}
        for actor in actors:
            if not isinstance(actor, Mapping):
                return None
            actor_id = actor.get("id")
            geometry = actor.get("collision_geometry")
            if (
                not isinstance(actor_id, str)
                or not actor_id
                or not isinstance(geometry, list)
            ):
                return None
            if actor_id in result:
                return None
            result[actor_id] = geometry
        return result

    official_geometry = geometry_by_id(official_setup)
    generated_geometry = geometry_by_id(generated_setup)
    if (
        not official_geometry
        or not generated_geometry
        or not set(official_geometry).issubset(generated_geometry)
        or any(not value for value in official_geometry.values())
        or any(
            not generated_geometry[actor_id]
            for actor_id in official_geometry
        )
    ):
        return None, "no_comparable_simulator_collision_geometry"
    return (
        all(
            official_geometry[actor_id]
            == generated_geometry[actor_id]
            for actor_id in official_geometry
        ),
        "same_seed_simulator_state:tracked_actors.collision_geometry",
    )


def build_preservation_report(
    conditions: list[str],
    *,
    scene_generated: bool,
    checker_generated: bool,
    checker_references_official_core: bool | None = None,
    visual_self_check_enabled: bool,
    visual: Mapping[str, Any],
    official_setup: Mapping[str, Any] | None = None,
    generated_setup: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record only the preservation facts supported by their actual authority."""

    checks: list[dict[str, Any]] = []
    exact_task_reuse = not scene_generated and not checker_generated
    for raw_condition in conditions:
        condition = str(raw_condition).strip()
        lowered = condition.casefold()
        if exact_task_reuse:
            kind = "exact_task_method_reuse"
            verified: bool | None = True
            authority = "exact_official_scene_and_checker_method_reuse"
        else:
            has_official_core_conjunct_term = any(
                term in lowered for term in _OFFICIAL_CORE_CONJUNCT_TERMS
            )
            has_checker_term = any(
                term in lowered for term in _CHECKER_PRESERVATION_TERMS
            ) and not has_official_core_conjunct_term
            has_visual_term = any(
                term in lowered for term in _VISUAL_PRESERVATION_TERMS
            )
            has_simulator_state_term = any(
                term in lowered
                for term in _SIMULATOR_STATE_PRESERVATION_TERMS
            )
            has_geometry_term = any(
                term in lowered for term in _GEOMETRY_PRESERVATION_TERMS
            )
            has_frozen_term = any(
                term in lowered for term in _FROZEN_PRESERVATION_TERMS
            )
            component_results: list[bool | None] = []
            authorities: list[str] = []
            kinds: list[str] = []
            if has_official_core_conjunct_term:
                kinds.append("official_core_conjunct")
                component_results.append(
                    True
                    if not checker_generated
                    else checker_references_official_core
                )
                authorities.append(
                    "exact_official_check_success_reuse"
                    if not checker_generated
                    else (
                        "generated_checker_direct_official_core_reference"
                        if checker_references_official_core is True
                        else "generated_checker_missing_official_core_reference"
                        if checker_references_official_core is False
                        else "official_core_reference_not_inspected"
                    )
                )
            if has_checker_term:
                kinds.append("checker_semantics")
                component_results.append(
                    True if not checker_generated else None
                )
                authorities.append(
                    "exact_official_check_success_reuse"
                    if not checker_generated
                    else "no_equivalence_authority_for_generated_checker"
                )
            if has_simulator_state_term:
                kinds.append("simulator_state")
                simulator_verified, simulator_authority = (
                    _same_seed_tracked_actor_state(
                        official_setup,
                        generated_setup,
                        condition,
                    )
                )
                component_results.append(simulator_verified)
                authorities.append(simulator_authority)
            if has_geometry_term:
                kinds.append("geometry")
                if not scene_generated:
                    component_results.append(True)
                    authorities.append("exact_official_load_actors_reuse")
                else:
                    geometry_verified, geometry_authority = (
                        _same_seed_tracked_actor_geometry(
                            official_setup,
                            generated_setup,
                        )
                    )
                    component_results.append(geometry_verified)
                    authorities.append(geometry_authority)
            if has_visual_term:
                kinds.append("visual")
                if not scene_generated:
                    component_results.append(True)
                    authorities.append("exact_official_load_actors_reuse")
                elif not visual_self_check_enabled:
                    component_results.append(None)
                    authorities.append("visual_self_check_disabled")
                else:
                    unexpected = visual.get("unexpected_changes")
                    component_results.append(
                        bool(
                            visual.get("passed") is True
                            and isinstance(unexpected, list)
                            and not unexpected
                        )
                    )
                    authorities.append("same_seed_visual_diagnosis")
            if has_frozen_term:
                kinds.append("frozen_runtime_binding")
                component_results.append(True)
                authorities.append(
                    "frozen_task_policy_seed_and_action_binding"
                )
            if component_results:
                kind = "+".join(kinds)
                if any(item is False for item in component_results):
                    verified = False
                elif all(item is True for item in component_results):
                    verified = True
                else:
                    verified = None
                authority = "+".join(authorities)
            else:
                kind = "unverified"
                verified = None
                authority = "no_available_preservation_authority"
        checks.append(
            {
                "condition": condition,
                "kind": kind,
                "verified": verified,
                "authority": authority,
            }
        )
    if not checks:
        verified_all: bool | None = True
        status = "not_required"
    elif any(item["verified"] is False for item in checks):
        verified_all = False
        status = "failed"
    elif all(item["verified"] is True for item in checks):
        verified_all = True
        status = "verified"
    else:
        verified_all = None
        status = "partially_unverified"
    return {
        "schema_version": 1,
        "status": status,
        "verified": verified_all,
        "checks": checks,
    }


def _checker_references_official_core(module_source: str) -> bool:
    """Detect the explicit untouched-official predicate call in a checker.

    This proves only a direct code reference, not semantic equivalence.  The
    simulator fixtures remain responsible for executable positive/negative
    evidence, and outcome reporting keeps the generated checker separate.
    """

    module = ast.parse(module_source)
    checker = next(
        (
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef)
            and node.name == "check_success"
        ),
        None,
    )
    if checker is None:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mea_official_check_success"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and not node.args
        and not node.keywords
        for node in ast.walk(checker)
    )


__all__ = ["build_preservation_report"]

