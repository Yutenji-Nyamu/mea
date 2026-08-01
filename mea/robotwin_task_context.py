"""Retrieve-first RoboTwin TaskContext for generic TaskGen.

The repository TaskSchema is a reviewed cache, not the definition of which
official tasks ManipEvalAgent may evaluate.  When that cache is absent, a
fresh simulator reset may supply the minimal actor/telemetry authority needed
by generic TaskGen.  Static source inspection alone never invents actors,
semantic roles, success thresholds, or simulator state.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mea.toolkit.schema import (
    TaskSchemaError,
    actor_access_path,
    actor_access_path_key,
    load_task_schema,
    task_schema_path,
    validate_task_schema,
)


class RoboTwinTaskContextError(RuntimeError):
    """Raised when TaskContext lacks verifiable simulator authority."""


_TASK_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_PROBE_KEYS = {
    "schema_version",
    "task_name",
    "official_source",
    "official_source_sha256",
    "setup_success",
    "official_check_success_callable",
    "physics_timestep_seconds",
    "action_dimension",
    "actors",
}
_ACTOR_KEY_SETS = {
    frozenset({"task_attribute", "scene_name"}),
    frozenset({"access_path", "scene_name"}),
    frozenset({"task_attribute", "access_path", "scene_name"}),
}
_OBSERVABLE_KEYS = {
    "simulation_clock",
    "policy_action",
    "robot_tcp",
    "contact_events",
}


@dataclass(frozen=True)
class RoboTwinTaskContext:
    """Source identity plus the strongest available execution schema."""

    task_name: str
    official_source: str
    official_class: str
    official_source_sha256: str
    declared_methods: tuple[str, ...]
    source_task_attributes: tuple[str, ...]
    task_schema: Mapping[str, Any] | None
    schema_origin: str
    runtime_probe: Mapping[str, Any] | None
    telemetry_observables: Mapping[str, Any]

    @property
    def taskgen_ready(self) -> bool:
        return self.task_schema is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_name": self.task_name,
            "official_source": self.official_source,
            "official_class": self.official_class,
            "official_source_sha256": self.official_source_sha256,
            "declared_methods": list(self.declared_methods),
            "source_task_attributes": list(self.source_task_attributes),
            "taskgen_ready": self.taskgen_ready,
            "schema_origin": self.schema_origin,
            "task_schema": (
                deepcopy(dict(self.task_schema))
                if self.task_schema is not None
                else None
            ),
            "runtime_probe": (
                deepcopy(dict(self.runtime_probe))
                if self.runtime_probe is not None
                else None
            ),
            "telemetry_observables": deepcopy(
                dict(self.telemetry_observables)
            ),
            "authority": {
                "official_source": "repository_source_sha256",
                "actor_telemetry": (
                    "reviewed_task_schema"
                    if self.schema_origin == "reviewed_task_schema"
                    else (
                        "fresh_simulator_reset_probe"
                        if self.schema_origin == "runtime_probe"
                        else "unavailable"
                    )
                ),
                "success": (
                    "reviewed_task_schema"
                    if self.schema_origin == "reviewed_task_schema"
                    else (
                        "official_check_success_runtime_callable"
                        if self.schema_origin == "runtime_probe"
                        else "unavailable"
                    )
                ),
            },
        }


def _source_facts(
    repo_root: Path,
    task_name: str,
) -> tuple[str, Path, str, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(task_name, str) or _TASK_NAME.fullmatch(task_name) is None:
        raise RoboTwinTaskContextError(
            "task_name is not a RoboTwin identifier"
        )
    relative_source = f"envs/{task_name}.py"
    source = (repo_root / relative_source).resolve()
    try:
        source.relative_to(repo_root)
    except ValueError as exc:
        raise RoboTwinTaskContextError(
            "official source escapes the repository"
        ) from exc
    try:
        source_bytes = source.read_bytes()
        source_text = source_bytes.decode("utf-8")
        tree = ast.parse(source_text, filename=str(source))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise RoboTwinTaskContextError(
            f"official task source is invalid: {relative_source}"
        ) from exc
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == task_name
    ]
    if len(classes) != 1:
        raise RoboTwinTaskContextError(
            f"official source must declare one class {task_name!r}"
        )
    methods = tuple(
        sorted(
            node.name
            for node in classes[0].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    )
    attributes = tuple(
        sorted(
            {
                node.attr
                for node in ast.walk(classes[0])
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr.isidentifier()
                and not node.attr.startswith("_")
            }
        )
    )
    return (
        relative_source,
        source,
        hashlib.sha256(source_bytes).hexdigest(),
        methods,
        attributes,
    )


def _actor_id_from_access_path(path: list[dict[str, Any]]) -> str:
    """Derive a stable telemetry id from a validated structured path."""

    key = actor_access_path_key({"access_path": path})
    root = str(path[0]["attribute"])
    if len(path) == 1 and _TASK_NAME.fullmatch(root):
        return root
    parts = [root] if _TASK_NAME.fullmatch(root) else ["actor"]
    for segment in path[1:]:
        operation, operand = next(iter(segment.items()))
        if operation == "index":
            parts.extend(("index", str(operand)))
        elif isinstance(operand, int):
            encoded = (
                str(operand)
                if operand >= 0
                else f"negative_{abs(operand)}"
            )
            parts.extend(("key", "int", encoded))
        elif _TASK_NAME.fullmatch(operand):
            parts.extend(("key", "str", operand))
        else:
            digest = hashlib.sha256(operand.encode("utf-8")).hexdigest()[:12]
            parts.extend(("key", "str", digest))
    candidate = "_".join(parts)
    if _TASK_NAME.fullmatch(candidate):
        return (
            candidate
            + "_path_"
            + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        )
    return "actor_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _container_actor_paths(
    value: Any,
    *,
    path: list[dict[str, Any]],
    scene_identity: set[int],
    ancestors: frozenset[int] = frozenset(),
) -> list[tuple[int, list[dict[str, Any]], Any]]:
    """Find scene actors through builtin containers without arbitrary reads."""

    simulator_actor = getattr(value, "actor", value)
    simulator_identity = id(simulator_actor)
    if simulator_identity in scene_identity:
        get_name = getattr(value, "get_name", None)
        get_pose = getattr(value, "get_pose", None)
        if callable(get_name) and callable(get_pose):
            return [(simulator_identity, path, value)]
        return []
    if type(value) not in {list, tuple, dict}:
        return []
    container_identity = id(value)
    if container_identity in ancestors:
        return []
    nested_ancestors = ancestors | {container_identity}
    found: list[tuple[int, list[dict[str, Any]], Any]] = []
    if type(value) in {list, tuple}:
        for index, item in enumerate(value):
            found.extend(
                _container_actor_paths(
                    item,
                    path=[*path, {"index": index}],
                    scene_identity=scene_identity,
                    ancestors=nested_ancestors,
                )
            )
        return found
    keys = [
        key
        for key in value
        if not isinstance(key, bool) and isinstance(key, (str, int))
    ]
    keys.sort(
        key=lambda key: (
            0 if isinstance(key, int) else 1,
            key if isinstance(key, int) else str(key),
        )
    )
    for key in keys:
        found.extend(
            _container_actor_paths(
                value[key],
                path=[*path, {"key": key}],
                scene_identity=scene_identity,
                ancestors=nested_ancestors,
            )
        )
    return found


def _schema_telemetry_observables(
    schema: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Describe only streams that the recorder can actually emit.

    This is an execution-capability contract, not a task-semantic map.  Actor
    roles and success thresholds remain outside it.
    """

    if schema is None:
        return {
            "schema_version": 1,
            "authority": "unavailable_until_simulator_reset",
            "simulation_clock": {
                "available": False,
                "signals": [],
            },
            "policy_action": {
                "available": False,
                "dimension": None,
                "signals": [],
            },
            "robot_tcp": {
                "available_sides": [],
                "signals": [],
            },
            "contact_events": {
                "available": False,
                "scope": "unavailable",
                "signals": [],
            },
            "actor_pose_signals": [],
        }
    semantic_fields = schema.get("semantic_fields")
    fields = (
        [dict(item) for item in semantic_fields if isinstance(item, Mapping)]
        if isinstance(semantic_fields, list)
        else []
    )
    tcp_sides = sorted(
        {
            str(item["side"])
            for item in fields
            if item.get("source") == "robot_tcp_position"
            and item.get("side") in {"left", "right"}
        }
    )
    actor_pose_signals = [
        str(item["name"])
        for item in fields
        if item.get("source")
        in {
            "actor_position",
            "actor_functional_position",
            "actor_contact_position",
        }
        and isinstance(item.get("name"), str)
    ]
    action_dimension = schema.get("action_dimension")
    action_available = (
        isinstance(action_dimension, int)
        and not isinstance(action_dimension, bool)
        and action_dimension > 0
    )
    contact_focus = schema.get("contact_focus_actor_ids")
    contact_available = bool(
        isinstance(contact_focus, list) and contact_focus
    )
    return {
        "schema_version": 1,
        "authority": (
            "validated_task_schema_and_recorder_contract"
        ),
        "simulation_clock": {
            "available": True,
            "signals": [
                "physics_step",
                "policy_step",
                "simulation_time_seconds",
            ],
        },
        "policy_action": {
            "available": action_available,
            "dimension": (
                int(action_dimension) if action_available else None
            ),
            "signals": (
                [f"action.{index}" for index in range(int(action_dimension))]
                if action_available
                else []
            ),
        },
        "robot_tcp": {
            "available_sides": tcp_sides,
            "signals": [
                str(item["name"])
                for item in fields
                if item.get("source") == "robot_tcp_position"
                and isinstance(item.get("name"), str)
            ],
        },
        "contact_events": {
            "available": contact_available,
            "scope": (
                "declared_contact_focus_actors"
                if contact_available
                else "unavailable"
            ),
            "signals": (
                [
                    "contact_pair",
                    "physical_contact",
                    "start_simulation_time_seconds",
                    "end_simulation_time_seconds",
                ]
                if contact_available
                else []
            ),
        },
        "actor_pose_signals": actor_pose_signals,
    }


def _runtime_schema(
    probe: Mapping[str, Any],
    *,
    task_name: str,
    relative_source: str,
    source_sha256: str,
    source_attributes: tuple[str, ...],
) -> dict[str, Any]:
    probe_keys = set(probe) if isinstance(probe, Mapping) else set()
    if probe_keys not in (
        _PROBE_KEYS,
        _PROBE_KEYS | {"observables"},
    ):
        raise RoboTwinTaskContextError(
            "runtime TaskContext probe has an invalid schema"
        )
    if probe.get("schema_version") != 1:
        raise RoboTwinTaskContextError(
            "runtime TaskContext probe schema_version must be 1"
        )
    if probe.get("task_name") != task_name:
        raise RoboTwinTaskContextError(
            "runtime TaskContext probe task_name differs"
        )
    if probe.get("official_source") != relative_source:
        raise RoboTwinTaskContextError(
            "runtime TaskContext probe official_source differs"
        )
    if probe.get("official_source_sha256") != source_sha256:
        raise RoboTwinTaskContextError(
            "runtime TaskContext probe source hash differs"
        )
    if probe.get("setup_success") is not True:
        raise RoboTwinTaskContextError(
            "runtime TaskContext probe did not complete simulator reset"
        )
    if probe.get("official_check_success_callable") is not True:
        raise RoboTwinTaskContextError(
            "runtime TaskContext probe lacks official check_success authority"
        )
    physics_dt = probe.get("physics_timestep_seconds")
    if (
        isinstance(physics_dt, bool)
        or not isinstance(physics_dt, (int, float))
        or float(physics_dt) <= 0
    ):
        raise RoboTwinTaskContextError(
            "runtime TaskContext probe physics timestep is invalid"
        )
    action_dimension = probe.get("action_dimension")
    if (
        isinstance(action_dimension, bool)
        or not isinstance(action_dimension, int)
        or action_dimension < 0
    ):
        raise RoboTwinTaskContextError(
            "runtime TaskContext probe action dimension is invalid"
        )
    actors = probe.get("actors")
    if not isinstance(actors, list) or not actors:
        raise RoboTwinTaskContextError(
            "runtime TaskContext probe found no source-bound actors"
        )
    observables = probe.get("observables")
    current_observable_probe = observables is not None
    if observables is None:
        # Reader compatibility for already-published TaskContext artifacts.
        # Old probes did not claim TCP/contact capability, so retain the
        # conservative old schema rather than inventing those signals.
        observables = {
            "simulation_clock": True,
            "policy_action": action_dimension > 0,
            "robot_tcp": {"left": False, "right": False},
            "contact_events": False,
        }
    elif (
        not isinstance(observables, Mapping)
        or set(observables) != _OBSERVABLE_KEYS
        or observables.get("simulation_clock") is not True
    ):
        raise RoboTwinTaskContextError(
            "runtime TaskContext observable probe is invalid"
        )
    policy_action = observables.get("policy_action")
    if not isinstance(policy_action, bool) or policy_action is not (
        action_dimension > 0
    ):
        raise RoboTwinTaskContextError(
            "runtime TaskContext policy-action authority differs"
        )
    robot_tcp = observables.get("robot_tcp")
    if (
        not isinstance(robot_tcp, Mapping)
        or set(robot_tcp) != {"left", "right"}
        or any(not isinstance(robot_tcp[side], bool) for side in robot_tcp)
    ):
        raise RoboTwinTaskContextError(
            "runtime TaskContext TCP authority is invalid"
        )
    contact_events = observables.get("contact_events")
    if not isinstance(contact_events, bool):
        raise RoboTwinTaskContextError(
            "runtime TaskContext contact authority is invalid"
        )
    tracked_actors: list[dict[str, Any]] = []
    semantic_fields: list[dict[str, Any]] = []
    seen_access_paths: set[str] = set()
    seen_actor_ids: set[str] = set()
    for index, raw_actor in enumerate(actors):
        if (
            not isinstance(raw_actor, Mapping)
            or frozenset(raw_actor) not in _ACTOR_KEY_SETS
        ):
            raise RoboTwinTaskContextError(
                f"runtime TaskContext actor {index} has an invalid schema"
            )
        try:
            access_path = actor_access_path(raw_actor)
            access_key = actor_access_path_key(raw_actor)
        except TaskSchemaError as exc:
            raise RoboTwinTaskContextError(
                f"runtime TaskContext actor {index} access_path is invalid"
            ) from exc
        root_attribute = access_path[0]["attribute"]
        scene_name = raw_actor.get("scene_name")
        if root_attribute not in source_attributes:
            raise RoboTwinTaskContextError(
                "runtime actor access root lacks source authority: "
                f"{root_attribute!r}"
            )
        if not isinstance(scene_name, str) or not scene_name.strip():
            raise RoboTwinTaskContextError(
                f"runtime actor {access_path!r} has no simulator name"
            )
        if access_key in seen_access_paths:
            raise RoboTwinTaskContextError(
                "runtime TaskContext actors must have unique access paths"
            )
        actor_id = _actor_id_from_access_path(access_path)
        if actor_id in seen_actor_ids:
            raise RoboTwinTaskContextError(
                "runtime actor paths produced duplicate stable ids: "
                f"{actor_id!r}"
            )
        seen_access_paths.add(access_key)
        seen_actor_ids.add(actor_id)
        actor_spec: dict[str, Any] = {"id": actor_id}
        if len(access_path) == 1:
            # Preserve the established direct-attribute TaskSchema shape.
            actor_spec["task_attribute"] = root_attribute
        else:
            actor_spec["access_path"] = access_path
        actor_spec.update(
            {
                "scene_name": scene_name.strip(),
                "functional_points": [],
                "contact_points": [],
            }
        )
        tracked_actors.append(actor_spec)
        semantic_fields.append(
            {
                "name": f"{actor_id}_position",
                "source": "actor_position",
                "actor_id": actor_id,
            }
        )
    for side in ("left", "right"):
        if robot_tcp[side]:
            semantic_fields.append(
                {
                    "name": f"{side}_tcp_position",
                    "source": "robot_tcp_position",
                    "side": side,
                }
            )
    contact_focus_actor_ids = (
        [actor["id"] for actor in tracked_actors]
        if contact_events
        else []
    )
    schema = {
        "schema_version": 1,
        "task_name": task_name,
        "task_family": "robotwin_runtime_discovered",
        "trusted_tool_profile": "runtime_actor_positions",
        "physics_timestep_seconds": float(physics_dt),
        "action_dimension": action_dimension,
        "probe_task_attributes": [],
        "tracked_actors": tracked_actors,
        # Record every contact involving a source-bound actor.  This is only a
        # scope declaration; it does not guess which actor is the target.
        "contact_focus_actor_ids": contact_focus_actor_ids,
        "semantic_fields": semantic_fields,
        # Role labels and task-specific thresholds cannot be inferred from an
        # actor name or image.  Query-derived ToolGen may consume the raw pose
        # fields, while unsupported semantic requests remain unsupported.
        "semantic_roles": {},
        "success_contract": {
            "type": "official_check_success",
            "authority": "official_check_success_runtime_callable",
            "official_source_sha256": source_sha256,
            "semantic_telemetry_available": True,
        },
    }
    if current_observable_probe:
        schema["telemetry_observables"] = _schema_telemetry_observables(
            schema
        )
    try:
        return validate_task_schema(
            schema,
            expected_task_name=task_name,
        )
    except TaskSchemaError as exc:
        raise RoboTwinTaskContextError(
            f"runtime TaskContext produced an invalid telemetry schema: {exc}"
        ) from exc


def resolve_robotwin_task_context(
    repo_root: str | Path,
    task_name: str,
    *,
    runtime_probe: Mapping[str, Any] | None = None,
) -> RoboTwinTaskContext:
    """Retrieve a reviewed schema or validate one fresh runtime probe.

    A source-only result is intentionally not TaskGen-ready.  It remains useful
    for official rollout routing and for requesting the reset probe, but never
    authorizes generated scene/checker execution by itself.
    """

    root = Path(repo_root).expanduser().resolve()
    (
        relative_source,
        _source,
        source_sha256,
        methods,
        attributes,
    ) = _source_facts(root, task_name)
    schema_path = task_schema_path(root, task_name)
    if schema_path.is_file():
        try:
            schema: Mapping[str, Any] | None = load_task_schema(root, task_name)
        except TaskSchemaError as exc:
            raise RoboTwinTaskContextError(
                f"reviewed TaskSchema is invalid: {schema_path}"
            ) from exc
        origin = "reviewed_task_schema"
        accepted_probe: Mapping[str, Any] | None = None
    elif runtime_probe is not None:
        schema = _runtime_schema(
            runtime_probe,
            task_name=task_name,
            relative_source=relative_source,
            source_sha256=source_sha256,
            source_attributes=attributes,
        )
        origin = "runtime_probe"
        accepted_probe = deepcopy(dict(runtime_probe))
    else:
        schema = None
        origin = "source_only"
        accepted_probe = None
    return RoboTwinTaskContext(
        task_name=task_name,
        official_source=relative_source,
        official_class=task_name,
        official_source_sha256=source_sha256,
        declared_methods=methods,
        source_task_attributes=attributes,
        task_schema=deepcopy(schema),
        schema_origin=origin,
        runtime_probe=accepted_probe,
        telemetry_observables=_schema_telemetry_observables(schema),
    )


def build_runtime_task_context_probe(
    task: Any,
    *,
    repo_root: str | Path,
    task_name: str,
    action_dimension: int = 0,
) -> dict[str, Any]:
    """Describe actors proved by both official source and a live reset.

    The function compares object identity against ``scene.get_all_actors()``.
    It does not guess target roles, functional points, contact points, or
    success thresholds.
    """

    context = resolve_robotwin_task_context(repo_root, task_name)
    if type(task).__name__ != task_name:
        raise RoboTwinTaskContextError(
            "runtime task class differs from official task identity"
        )
    scene = getattr(task, "scene", None)
    get_all_actors = getattr(scene, "get_all_actors", None)
    if not callable(get_all_actors):
        raise RoboTwinTaskContextError(
            "runtime task exposes no simulator actor inventory"
        )
    scene_objects = list(get_all_actors() or [])
    get_all_articulations = getattr(scene, "get_all_articulations", None)
    if callable(get_all_articulations):
        scene_objects.extend(list(get_all_articulations() or []))
    actor_identity = {id(actor) for actor in scene_objects}
    paths_by_identity: dict[
        int, list[tuple[list[dict[str, Any]], Any]]
    ] = {}
    for attribute in context.source_task_attributes:
        try:
            value = getattr(task, attribute)
        except AttributeError:
            continue
        for identity, path, actor in _container_actor_paths(
            value,
            path=[{"attribute": attribute}],
            scene_identity=actor_identity,
        ):
            paths_by_identity.setdefault(identity, []).append((path, actor))
    canonical_actors: list[tuple[list[dict[str, Any]], Any]] = []
    for paths in paths_by_identity.values():
        canonical_actors.append(
            min(
                paths,
                key=lambda item: (
                    len(item[0]),
                    actor_access_path_key({"access_path": item[0]}),
                ),
            )
        )
    canonical_actors.sort(
        key=lambda item: actor_access_path_key({"access_path": item[0]})
    )
    candidates: list[dict[str, Any]] = []
    for access_path, actor in canonical_actors:
        scene_name = actor.get_name()
        if not isinstance(scene_name, str) or not scene_name.strip():
            continue
        candidate: dict[str, Any] = {"scene_name": scene_name.strip()}
        if len(access_path) == 1:
            # Preserve the old direct probe shape exactly.
            candidate["task_attribute"] = access_path[0]["attribute"]
        else:
            candidate["access_path"] = access_path
        candidates.append(candidate)
    get_timestep = getattr(scene, "get_timestep", None)
    physics_dt = get_timestep() if callable(get_timestep) else None
    if (
        isinstance(physics_dt, bool)
        or not isinstance(physics_dt, (int, float))
        or float(physics_dt) <= 0
    ):
        raise RoboTwinTaskContextError(
            "runtime scene exposes no positive physics timestep"
        )
    robot = getattr(task, "robot", None)
    robot_tcp = {
        side: callable(getattr(robot, f"get_{side}_tcp_pose", None))
        for side in ("left", "right")
    }
    probe = {
        "schema_version": 1,
        "task_name": task_name,
        "official_source": context.official_source,
        "official_source_sha256": context.official_source_sha256,
        "setup_success": True,
        "official_check_success_callable": callable(
            getattr(task, "check_success", None)
        ),
        "physics_timestep_seconds": float(physics_dt),
        "action_dimension": action_dimension,
        "actors": candidates,
        "observables": {
            "simulation_clock": True,
            "policy_action": action_dimension > 0,
            "robot_tcp": robot_tcp,
            "contact_events": callable(getattr(scene, "get_contacts", None)),
        },
    }
    # Validate before exposing the probe artifact.
    _runtime_schema(
        probe,
        task_name=task_name,
        relative_source=context.official_source,
        source_sha256=context.official_source_sha256,
        source_attributes=context.source_task_attributes,
    )
    return probe


def probe_official_robotwin_task_context(
    *,
    repo_root: str | Path,
    task_name: str,
    seed: int,
    action_dimension: int,
) -> dict[str, Any]:
    """Perform one policy-free official reset and return a validated probe.

    This is the production fallback for a schema-less Tool-only Proposal.  It
    does not execute a policy, generate a scene/checker, or infer task roles.
    """

    root = Path(repo_root).expanduser().resolve()
    context = resolve_robotwin_task_context(root, task_name)
    module = importlib.import_module(f"envs.{task_name}")
    module_file = getattr(module, "__file__", None)
    expected_source = (root / context.official_source).resolve()
    if (
        not isinstance(module_file, str)
        or Path(module_file).resolve() != expected_source
    ):
        raise RoboTwinTaskContextError(
            "official reset imported a different RoboTwin task source"
        )
    task_class = getattr(module, task_name, None)
    if not isinstance(task_class, type):
        raise RoboTwinTaskContextError(
            "official task module does not expose its declared class"
        )

    import yaml
    from envs import CONFIGS_PATH

    def read_yaml(path: Path) -> dict[str, Any]:
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise RoboTwinTaskContextError(
                f"official reset config is invalid: {path}"
            ) from exc
        if not isinstance(value, Mapping):
            raise RoboTwinTaskContextError(
                f"official reset config is not an object: {path}"
            )
        return dict(value)

    config_root = Path(CONFIGS_PATH)
    args = read_yaml(config_root / "demo_clean.yml")
    embodiments = read_yaml(config_root / "_embodiment_config.yml")
    embodiment = args.get("embodiment", ["aloha-agilex"])
    if not isinstance(embodiment, list) or len(embodiment) != 1:
        raise RoboTwinTaskContextError(
            "official reset requires one dual-arm embodiment"
        )
    try:
        robot_file = Path(embodiments[embodiment[0]]["file_path"])
    except (KeyError, TypeError) as exc:
        raise RoboTwinTaskContextError(
            "official reset embodiment config is incomplete"
        ) from exc
    robot_config = read_yaml(robot_file / "config.yml")
    cameras = read_yaml(config_root / "_camera_config.yml")
    try:
        head_type = args["camera"]["head_camera_type"]
        head_camera = cameras[head_type]
        args.update(
            left_robot_file=str(robot_file),
            right_robot_file=str(robot_file),
            dual_arm_embodied=True,
            left_embodiment_config=robot_config,
            right_embodiment_config=robot_config,
            head_camera_h=head_camera["h"],
            head_camera_w=head_camera["w"],
            task_name=task_name,
            task_config="demo_clean",
            render_freq=0,
            eval_mode=True,
            save_data=False,
            eval_video_save_dir=None,
        )
    except (KeyError, TypeError) as exc:
        raise RoboTwinTaskContextError(
            "official reset camera config is incomplete"
        ) from exc

    task = task_class()
    try:
        task.setup_demo(
            now_ep_num=0,
            seed=int(seed),
            is_test=True,
            **args,
        )
        return build_runtime_task_context_probe(
            task,
            repo_root=root,
            task_name=task_name,
            action_dimension=action_dimension,
        )
    except RoboTwinTaskContextError:
        raise
    except Exception as exc:
        raise RoboTwinTaskContextError(
            "official reset could not establish TaskContext authority: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        close_env = getattr(task, "close_env", None)
        if callable(close_env):
            close_env(clear_cache=True)


__all__ = [
    "RoboTwinTaskContext",
    "RoboTwinTaskContextError",
    "build_runtime_task_context_probe",
    "probe_official_robotwin_task_context",
    "resolve_robotwin_task_context",
]
