"""TaskSchema loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TaskSchemaError(RuntimeError):
    """Raised when telemetry cannot interpret a task schema."""


def load_task_schema(
    repo_root: str | Path,
    task_name: str,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    path = root / "mea/toolkit/schemas" / f"{task_name}.json"
    if not path.is_file():
        raise TaskSchemaError(f"TaskSchema 不存在: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("task_name") != task_name:
        raise TaskSchemaError("TaskSchema.task_name 不匹配")
    actors = value.get("tracked_actors")
    if not isinstance(actors, list) or not actors:
        raise TaskSchemaError("TaskSchema.tracked_actors 必须是非空 list")
    ids = [item.get("id") for item in actors]
    if any(not isinstance(item, str) or not item for item in ids):
        raise TaskSchemaError("tracked actor id 必须是非空字符串")
    if len(ids) != len(set(ids)):
        raise TaskSchemaError("tracked actor id 不能重复")
    return value
