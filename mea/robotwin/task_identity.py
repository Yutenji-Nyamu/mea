"""Data-driven RoboTwin task identity, independent of policy and TaskGen.

An official rollout needs only an importable task class.  A generated
scene/checker or telemetry Tool additionally needs a repository TaskSchema.
Keeping those capabilities separate lets a multi-task policy attempt every
official task without pretending that every task already has deep MEA hooks.
"""

from __future__ import annotations

import ast
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mea.toolkit.schema import (
    TaskSchemaError,
    load_task_schema,
    task_schema_path,
)


class RoboTwinTaskIdentityError(ValueError):
    """Raised when an official RoboTwin task identity cannot be established."""


_TASK_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class RoboTwinTaskIdentity:
    """Policy-neutral identity and the capabilities actually available."""

    task_name: str
    official_source: str
    official_class: str
    description: str
    task_family: str
    task_schema: Mapping[str, Any] | None

    @property
    def task_schema_available(self) -> bool:
        return self.task_schema is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_name": self.task_name,
            "official_source": self.official_source,
            "official_class": self.official_class,
            "description": self.description,
            "task_family": self.task_family,
            "task_schema_available": self.task_schema_available,
            "task_schema": (
                deepcopy(dict(self.task_schema))
                if self.task_schema is not None
                else None
            ),
        }


def _task_name(value: str) -> str:
    if not isinstance(value, str) or _TASK_NAME.fullmatch(value) is None:
        raise RoboTwinTaskIdentityError(
            "task_name is not a RoboTwin identifier"
        )
    return value


def _official_class(source: Path, task_name: str) -> None:
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise RoboTwinTaskIdentityError(
            f"official task source is invalid: {source}"
        ) from exc
    if not any(
        isinstance(node, ast.ClassDef) and node.name == task_name
        for node in tree.body
    ):
        raise RoboTwinTaskIdentityError(
            f"official task source does not declare class {task_name!r}"
        )


def _description(root: Path, task_name: str) -> str:
    path = root / "description" / "task_instruction" / f"{task_name}.json"
    if not path.is_file():
        return f"RoboTwin task {task_name.replace('_', ' ')}."
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RoboTwinTaskIdentityError(
            f"official task instruction is invalid: {path}"
        ) from exc
    description = value.get("full_description") if isinstance(value, Mapping) else None
    if not isinstance(description, str) or not description.strip():
        raise RoboTwinTaskIdentityError(
            f"official task instruction has no full_description: {path}"
        )
    return description.strip()


def discover_robotwin_task_identity(
    repo_root: str | Path,
    task_name: str,
) -> RoboTwinTaskIdentity:
    """Discover one task from repository source; TaskSchema is optional."""

    name = _task_name(task_name)
    root = Path(repo_root).expanduser().resolve()
    relative_source = f"envs/{name}.py"
    source = root / relative_source
    try:
        valid_source = source.is_file() and source.stat().st_size > 0
    except OSError:
        valid_source = False
    if not valid_source:
        raise RoboTwinTaskIdentityError(
            f"official task source is missing or empty: {source}"
        )
    _official_class(source, name)
    schema_path = task_schema_path(root, name)
    if schema_path.is_file():
        try:
            schema: Mapping[str, Any] | None = load_task_schema(root, name)
        except TaskSchemaError as exc:
            raise RoboTwinTaskIdentityError(
                f"task toolkit schema is invalid: {schema_path}"
            ) from exc
    else:
        schema = None
    family = (
        str(schema["task_family"]).strip()
        if schema is not None
        else "robotwin_official_task"
    )
    return RoboTwinTaskIdentity(
        task_name=name,
        official_source=relative_source,
        official_class=name,
        description=_description(root, name),
        task_family=family,
        task_schema=deepcopy(schema),
    )


def discover_robotwin_official_tasks(
    repo_root: str | Path,
) -> tuple[RoboTwinTaskIdentity, ...]:
    """Discover the official task library without a task-name allowlist."""

    root = Path(repo_root).expanduser().resolve()
    env_root = root / "envs"
    instruction_root = root / "description" / "task_instruction"
    if not env_root.is_dir() or not instruction_root.is_dir():
        raise RoboTwinTaskIdentityError(
            "RoboTwin envs and task_instruction directories are required"
        )
    tasks = [
        discover_robotwin_task_identity(root, path.stem)
        for path in sorted(env_root.glob("*.py"))
        if not path.stem.startswith("_")
        and (instruction_root / f"{path.stem}.json").is_file()
    ]
    if not tasks:
        raise RoboTwinTaskIdentityError(
            "no official RoboTwin tasks were discovered"
        )
    return tuple(tasks)


__all__ = [
    "RoboTwinTaskIdentity",
    "RoboTwinTaskIdentityError",
    "discover_robotwin_official_tasks",
    "discover_robotwin_task_identity",
]
