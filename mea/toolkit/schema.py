"""TaskSchema discovery, validation, and telemetry signal contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class TaskSchemaError(RuntimeError):
    """Raised when telemetry cannot interpret a task schema."""


SEMANTIC_FIELD_SOURCES = frozenset(
    {
        "actor_position",
        "actor_functional_position",
        "actor_contact_position",
        "robot_tcp_position",
    }
)
COMMON_TRACE_KEYS = frozenset(
    {"physics_step", "policy_step", "simulation_time_seconds", "success"}
)
_TASK_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def task_schema_path(repo_root: str | Path, task_name: str) -> Path:
    """Return the canonical schema path without silently falling back."""

    if not isinstance(task_name, str) or not _TASK_NAME.fullmatch(task_name):
        raise TaskSchemaError(f"非法 task_name: {task_name!r}")
    root = Path(repo_root).expanduser().resolve()
    return root / "mea/toolkit/schemas" / f"{task_name}.json"


def _nonnegative_int_list(value: Any, *, field: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in value
    ):
        raise TaskSchemaError(f"{field} 必须是非负整数 list")
    if len(value) != len(set(value)):
        raise TaskSchemaError(f"{field} 不能重复")
    return value


def validate_task_schema(
    value: Any,
    *,
    expected_task_name: str | None = None,
) -> dict[str, Any]:
    """Validate one schema and return it unchanged for deterministic snapshots."""

    if not isinstance(value, dict):
        raise TaskSchemaError("TaskSchema 必须是 JSON object")
    task_name = value.get("task_name")
    if not isinstance(task_name, str) or not _TASK_NAME.fullmatch(task_name):
        raise TaskSchemaError("TaskSchema.task_name 非法")
    if expected_task_name is not None and task_name != expected_task_name:
        raise TaskSchemaError(
            f"TaskSchema.task_name 不匹配: {task_name!r} != {expected_task_name!r}"
        )
    version = value.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise TaskSchemaError("TaskSchema.schema_version 必须是正整数")
    physics_dt = value.get("physics_timestep_seconds")
    if not isinstance(physics_dt, (int, float)) or physics_dt <= 0:
        raise TaskSchemaError("physics_timestep_seconds 必须为正数")
    action_dimension = value.get("action_dimension")
    if (
        not isinstance(action_dimension, int)
        or isinstance(action_dimension, bool)
        or action_dimension < 0
    ):
        raise TaskSchemaError("action_dimension 必须是非负整数")

    actors = value.get("tracked_actors")
    if not isinstance(actors, list) or not actors:
        raise TaskSchemaError("TaskSchema.tracked_actors 必须是非空 list")
    actor_map: dict[str, dict[str, Any]] = {}
    attributes: set[str] = set()
    for index, actor in enumerate(actors):
        if not isinstance(actor, dict):
            raise TaskSchemaError(f"tracked_actors[{index}] 必须是 object")
        actor_id = actor.get("id")
        attribute = actor.get("task_attribute")
        scene_name = actor.get("scene_name")
        if not isinstance(actor_id, str) or not _TASK_NAME.fullmatch(actor_id):
            raise TaskSchemaError(f"tracked_actors[{index}].id 非法")
        if actor_id in actor_map:
            raise TaskSchemaError(f"tracked actor id 重复: {actor_id}")
        if not isinstance(attribute, str) or not attribute:
            raise TaskSchemaError(
                f"tracked_actors[{index}].task_attribute 必须是非空字符串"
            )
        if attribute in attributes:
            raise TaskSchemaError(f"task_attribute 重复: {attribute}")
        if not isinstance(scene_name, str) or not scene_name:
            raise TaskSchemaError(
                f"tracked_actors[{index}].scene_name 必须是非空字符串"
            )
        _nonnegative_int_list(
            actor.get("functional_points", []),
            field=f"tracked_actors[{index}].functional_points",
        )
        _nonnegative_int_list(
            actor.get("contact_points", []),
            field=f"tracked_actors[{index}].contact_points",
        )
        actor_map[actor_id] = actor
        attributes.add(attribute)

    focus_ids = value.get("contact_focus_actor_ids", [])
    if not isinstance(focus_ids, list) or any(
        not isinstance(item, str) for item in focus_ids
    ):
        raise TaskSchemaError("contact_focus_actor_ids 必须是字符串 list")
    if len(focus_ids) != len(set(focus_ids)):
        raise TaskSchemaError("contact_focus_actor_ids 不能重复")
    unknown_focus = sorted(set(focus_ids) - set(actor_map))
    if unknown_focus:
        raise TaskSchemaError(f"contact focus 引用了未知 actor: {unknown_focus}")

    fields = value.get("semantic_fields")
    if not isinstance(fields, list) or not fields:
        raise TaskSchemaError("TaskSchema.semantic_fields 必须是非空 list")
    field_names: set[str] = set()
    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            raise TaskSchemaError(f"semantic_fields[{index}] 必须是 object")
        name = field.get("name")
        source = field.get("source")
        if not isinstance(name, str) or not name or name in COMMON_TRACE_KEYS:
            raise TaskSchemaError(f"semantic_fields[{index}].name 非法或占用保留字段")
        if name in field_names:
            raise TaskSchemaError(f"semantic field name 重复: {name}")
        if source not in SEMANTIC_FIELD_SOURCES:
            raise TaskSchemaError(
                f"semantic_fields[{index}].source 不受支持: {source!r}"
            )
        if source == "robot_tcp_position":
            if field.get("side") not in {"left", "right"}:
                raise TaskSchemaError(
                    f"semantic_fields[{index}].side 必须是 left 或 right"
                )
        else:
            actor_id = field.get("actor_id")
            if actor_id not in actor_map:
                raise TaskSchemaError(
                    f"semantic_fields[{index}] 引用了未知 actor: {actor_id!r}"
                )
            if source in {
                "actor_functional_position",
                "actor_contact_position",
            }:
                point_id = field.get("point_id")
                point_key = (
                    "functional_points"
                    if source == "actor_functional_position"
                    else "contact_points"
                )
                if point_id not in actor_map[actor_id].get(point_key, []):
                    raise TaskSchemaError(
                        f"semantic_fields[{index}].point_id 未在 actor {point_key} 声明"
                    )
        field_names.add(name)

    roles = value.get("semantic_roles", {})
    if not isinstance(roles, dict) or any(
        not isinstance(key, str)
        or not isinstance(field_name, str)
        or field_name not in field_names
        for key, field_name in roles.items()
    ):
        raise TaskSchemaError("semantic_roles 必须将 role 映射到已声明 semantic field")
    return value


def load_task_schema(
    repo_root: str | Path,
    task_name: str,
) -> dict[str, Any]:
    path = task_schema_path(repo_root, task_name)
    if not path.is_file():
        raise TaskSchemaError(f"TaskSchema 不存在: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskSchemaError(f"无法读取 TaskSchema: {path}: {exc}") from exc
    return validate_task_schema(value, expected_task_name=task_name)


def list_task_schemas(repo_root: str | Path) -> list[dict[str, Any]]:
    """Discover valid schemas; invalid files fail loudly instead of disappearing."""

    root = Path(repo_root).expanduser().resolve()
    directory = root / "mea/toolkit/schemas"
    summaries: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        schema = load_task_schema(root, path.stem)
        summaries.append(
            {
                "task_name": schema["task_name"],
                "schema_version": schema["schema_version"],
                "task_family": schema.get("task_family"),
                "tracked_actor_ids": [
                    actor["id"] for actor in schema["tracked_actors"]
                ],
                "semantic_field_names": [
                    field["name"] for field in schema["semantic_fields"]
                ],
                "trusted_tool_profile": schema.get("trusted_tool_profile"),
                "path": str(path.relative_to(root)).replace("\\", "/"),
            }
        )
    return summaries


def required_trace_keys(schema: dict[str, Any]) -> set[str]:
    """Return the exact common + schema-declared semantic signal contract."""

    validate_task_schema(schema, expected_task_name=schema.get("task_name"))
    return set(COMMON_TRACE_KEYS) | {
        field["name"] for field in schema["semantic_fields"]
    }
