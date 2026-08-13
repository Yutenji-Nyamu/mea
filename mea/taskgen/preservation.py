"""Simulator-authoritative TaskGen preservation checks."""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping
from typing import Any

from .preservation_facts import normalize_preservation_facts

_POSITION_ABS_TOLERANCE_M = 1e-5
_CONTACT_LOCAL_POSITION_ABS_TOLERANCE_M = 5e-4
_QUATERNION_COMPONENT_ABS_TOLERANCE = 5e-4


def _finite_numeric_vector(
    value: Any,
    *,
    minimum_length: int,
) -> list[float] | None:
    if not isinstance(value, list) or len(value) < minimum_length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _contact_points_preserved(
    official_actor: Mapping[str, Any],
    generated_actor: Mapping[str, Any],
    *,
    compare_world_positions: bool,
) -> bool | None:
    """Compare contact identity/local geometry unless world position is explicit."""

    official_points = official_actor.get("contact_points")
    generated_points = generated_actor.get("contact_points")
    if not isinstance(official_points, Mapping) or not official_points:
        return None
    if not isinstance(generated_points, Mapping):
        return False
    if set(official_points) != set(generated_points):
        return False
    official_actor_position = _finite_numeric_vector(
        official_actor.get("position"), minimum_length=3
    )
    generated_actor_position = _finite_numeric_vector(
        generated_actor.get("position"), minimum_length=3
    )
    if not compare_world_positions:
        official_quaternion = _finite_numeric_vector(
            official_actor.get("quaternion"), minimum_length=4
        )
        generated_quaternion = _finite_numeric_vector(
            generated_actor.get("quaternion"), minimum_length=4
        )
        if (
            official_quaternion is not None
            and generated_quaternion is not None
            and not _numeric_vectors_close(
                official_quaternion[:4],
                generated_quaternion[:4],
                absolute_tolerance=_QUATERNION_COMPONENT_ABS_TOLERANCE,
                sign_invariant=True,
            )
        ):
            # World contact positions can only be converted to actor-relative
            # offsets by subtraction while orientation is preserved.  A
            # rotated actor needs a body-frame transform that the probe does
            # not currently publish, so report unknown rather than a false
            # preservation verdict.
            return None
    for point_id in official_points:
        official_point = official_points[point_id]
        generated_point = generated_points[point_id]
        if not isinstance(official_point, Mapping) or not isinstance(
            generated_point, Mapping
        ):
            return False
        official_position = _finite_numeric_vector(
            official_point.get("position"), minimum_length=3
        )
        generated_position = _finite_numeric_vector(
            generated_point.get("position"), minimum_length=3
        )
        if official_position is None or generated_position is None:
            return False
        if compare_world_positions:
            left = official_position[:3]
            right = generated_position[:3]
            tolerance = _POSITION_ABS_TOLERANCE_M
        elif official_actor_position is None or generated_actor_position is None:
            # The declared IDs and finite contact values still establish the
            # reference contract when an actor center is not published.
            continue
        else:
            left = [
                official_position[index] - official_actor_position[index]
                for index in range(3)
            ]
            right = [
                generated_position[index] - generated_actor_position[index]
                for index in range(3)
            ]
            tolerance = _CONTACT_LOCAL_POSITION_ABS_TOLERANCE_M
        if not _numeric_vectors_close(
            left,
            right,
            absolute_tolerance=tolerance,
        ):
            return False
    return True


def _actor_model_identity(actor: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    geometry = actor.get("collision_geometry")
    if not isinstance(geometry, list):
        return None
    identity: list[dict[str, Any]] = []
    for item in geometry:
        if not isinstance(item, Mapping):
            continue
        projected = {
            field: item[field]
            for field in ("modelname", "model_id", "collision_asset")
            if field in item
        }
        if projected:
            identity.append(projected)
    return identity or None


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
    *,
    fact: Mapping[str, Any] | None = None,
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

    if fact is None:
        normalized = normalize_preservation_facts(condition)
        if len(normalized) != 1:
            return None, "non_atomic_preservation_fact"
        fact = normalized[0]
    property_name = fact.get("property")
    actor = fact.get("actor")
    axis = fact.get("axis") if property_name == "position" else None
    requires_contact = property_name == "contact_point"
    contact_world_position = fact.get("relation") == "preserve_world_position"
    position_axes = (
        [(str(axis), {"x": 0, "y": 1, "z": 2}[str(axis)])]
        if axis in {"x", "y", "z"}
        else []
    )
    requires_model_identity = property_name == "model_identity"
    if isinstance(actor, str):
        scoped_actor_ids = [actor] if actor in official_actors else []
    else:
        scoped_actor_ids = list(official_actors)
    if not scoped_actor_ids:
        return None, "preservation_actor_not_in_simulator_state"
    requires_actor_position = property_name == "position" and axis in {
        None,
        "all",
    }
    requires_orientation = property_name == "orientation"
    requires_height = property_name == "position" and axis == "z"
    if position_axes:
        # An explicitly named coordinate constrains only that axis. Treating
        # "sampled y position" as full xyz preservation would reject the x
        # perturbation requested by the same Proposal.
        requires_actor_position = False
    component_results: list[bool | None] = []
    components: list[str] = []

    if requires_contact:
        contact_results = [
            _contact_points_preserved(
                official_actors[actor_id],
                generated_actors[actor_id],
                compare_world_positions=contact_world_position,
            )
            for actor_id in scoped_actor_ids
        ]
        defined_contact_results = [
            result for result in contact_results if result is not None
        ]
        if not defined_contact_results:
            component_results.append(None)
        elif any(result is False for result in defined_contact_results):
            component_results.append(False)
        else:
            component_results.append(True)
        components.append(
            "contact_point_world_positions"
            if contact_world_position
            else "contact_point_references"
        )

    if requires_model_identity:
        identities = [
            (
                _actor_model_identity(official_actors[actor_id]),
                _actor_model_identity(generated_actors[actor_id]),
            )
            for actor_id in scoped_actor_ids
        ]
        comparable_identities = [
            (official, generated)
            for official, generated in identities
            if official is not None
        ]
        if not comparable_identities:
            component_results.append(None)
        else:
            component_results.append(
                all(
                    generated is not None and official == generated
                    for official, generated in comparable_identities
                )
            )
        components.append("model_identity")

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
                *(official_actors[actor_id] for actor_id in scoped_actor_ids),
                *(
                    generated_actors[actor_id]
                    for actor_id in scoped_actor_ids
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
                for actor_id in scoped_actor_ids
            )
        )
    for axis, index in position_axes:
        components.append(f"position_{axis}")
        positions = [
            (
                official_actors[actor_id].get("position"),
                generated_actors[actor_id].get("position"),
            )
            for actor_id in scoped_actor_ids
        ]
        if any(
            not isinstance(official, list)
            or not isinstance(generated, list)
            or len(official) <= index
            or len(generated) <= index
            for official, generated in positions
        ):
            component_results.append(None)
        else:
            component_results.append(
                all(
                    _numeric_vectors_close(
                        [official[index]],
                        [generated[index]],
                        absolute_tolerance=_POSITION_ABS_TOLERANCE_M,
                    )
                    for official, generated in positions
                )
            )
    if (
        requires_height
        and not requires_actor_position
        and not any(axis == "z" for axis, _index in position_axes)
    ):
        components.append("position_z")
        positions = [
            (
                official_actors[actor_id].get("position"),
                generated_actors[actor_id].get("position"),
            )
            for actor_id in scoped_actor_ids
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
    conditions: list[str | Mapping[str, Any]],
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
        condition = (
            str(raw_condition).strip()
            if not isinstance(raw_condition, Mapping)
            else json.dumps(
                dict(raw_condition),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for fact in normalize_preservation_facts(raw_condition):
            if exact_task_reuse:
                kind = "exact_task_method_reuse"
                verified: bool | None = True
                authority = "exact_official_scene_and_checker_method_reuse"
                checks.append(
                    {
                        "condition": condition,
                        "fact": fact,
                        "kind": kind,
                        "verified": verified,
                        "authority": authority,
                    }
                )
                continue
            else:
                property_name = fact["property"]
                has_official_core_conjunct_term = (
                    property_name == "official_goal"
                )
                has_checker_term = property_name == "checker_semantics"
                has_visual_term = property_name == "appearance"
                has_simulator_state_term = property_name in {
                    "contact_point",
                    "model_identity",
                    "orientation",
                    "position",
                }
                has_geometry_term = property_name == "geometry"
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
                        fact=fact,
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
                    "fact": fact,
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
        "schema_version": 2,
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
