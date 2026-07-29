"""Catalog-independent RoboTwin ACT execution bindings.

This module discovers only the physical/runtime authority needed to execute a
policy on one base task: the official task source, its validated TaskSchema,
and the two ACT checkpoint artifacts.  CapabilityAdapter entries, planner
aspects, templates, metrics, and experiment menus are deliberately absent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mea.taskgen.generic_backend import (
    GenericTaskGenError,
    discover_generic_robotwin_task_identity,
)

from .open_world_session import (
    OpenWorldSessionError,
    validate_open_world_evaluation_target,
)
from .policy_task_binding import (
    PolicyTaskBindingError,
    build_policy_task_binding,
)


class RuntimeTaskBindingError(ValueError):
    """Raised when a source/schema/checkpoint runtime boundary is incomplete."""


_CHECKPOINT_SETTING = re.compile(r"^[a-z][a-z0-9_]*$")


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeTaskBindingError(f"{field} must be a positive integer")
    return value


def _checkpoint_setting(value: Any) -> str:
    if (
        not isinstance(value, str)
        or _CHECKPOINT_SETTING.fullmatch(value) is None
    ):
        raise RuntimeTaskBindingError(
            "checkpoint_setting must be a lowercase identifier"
        )
    return value


def _require_nonempty_file(path: Path, *, label: str) -> None:
    try:
        valid = path.is_file() and path.stat().st_size > 0
    except OSError:
        valid = False
    if not valid:
        raise RuntimeTaskBindingError(f"{label} is missing or empty: {path}")


def build_runtime_policy_task_binding(
    repo_root: str | Path,
    task_name: str,
    *,
    checkpoint_setting: str = "demo_clean",
    expert_data_num: int = 50,
    language_conditioned: bool = False,
) -> dict[str, Any]:
    """Build a PolicyTaskBinding without catalog or adapter membership.

    The ACT checkpoint follows RoboTwin's canonical
    ``act-<task>/<setting>-<expert_data_num>`` layout.  This is an execution
    binding only; Query-derived concerns and generated artifacts remain outside
    it.
    """

    root = Path(repo_root).expanduser().resolve()
    setting = _checkpoint_setting(checkpoint_setting)
    expert_count = _positive_int(
        expert_data_num,
        field="expert_data_num",
    )
    if not isinstance(language_conditioned, bool):
        raise RuntimeTaskBindingError("language_conditioned must be bool")

    try:
        identity = discover_generic_robotwin_task_identity(root, task_name)
    except GenericTaskGenError as exc:
        raise RuntimeTaskBindingError(str(exc)) from exc
    schema = identity["task_schema"]
    task_family = schema.get("task_family")
    if not isinstance(task_family, str) or not task_family.strip():
        raise RuntimeTaskBindingError(
            "TaskSchema.task_family must be a non-empty string"
        )

    checkpoint_leaf = f"{setting}-{expert_count}"
    checkpoint_id = f"act-{task_name}/{checkpoint_leaf}"
    checkpoint_dir = (
        root
        / "policy"
        / "ACT"
        / "act_ckpt"
        / f"act-{task_name}"
        / checkpoint_leaf
    )
    _require_nonempty_file(
        checkpoint_dir / "dataset_stats.pkl",
        label="ACT dataset statistics",
    )
    _require_nonempty_file(
        checkpoint_dir / "policy_last.ckpt",
        label="ACT policy weights",
    )

    try:
        return build_policy_task_binding(
            task_name=task_name,
            task_family=task_family.strip(),
            task_module=f"envs.{task_name}",
            policy={
                "name": "ACT",
                "checkpoint_setting": setting,
                "expert_data_num": expert_count,
                "language_conditioned": language_conditioned,
            },
            checkpoint={
                "policy_name": "ACT",
                "checkpoint_setting": setting,
                "expert_data_num": expert_count,
                "checkpoint_id": checkpoint_id,
                "ready": True,
            },
        )
    except PolicyTaskBindingError as exc:
        raise RuntimeTaskBindingError(str(exc)) from exc


def build_runtime_open_world_evaluation_target(
    repo_root: str | Path,
    task_name: str,
    *,
    max_rounds: int,
    checkpoint_setting: str = "demo_clean",
    expert_data_num: int = 50,
    language_conditioned: bool = False,
) -> dict[str, Any]:
    """Build and validate a catalog-free OpenWorldEvaluationTarget."""

    budget = _positive_int(max_rounds, field="max_rounds")
    binding = build_runtime_policy_task_binding(
        repo_root,
        task_name,
        checkpoint_setting=checkpoint_setting,
        expert_data_num=expert_data_num,
        language_conditioned=language_conditioned,
    )
    try:
        return validate_open_world_evaluation_target(
            {
                "schema_version": 3,
                "binding_mode": "single_task_single_checkpoint_open_world",
                "policy_task_binding": binding,
                "max_rounds": budget,
            }
        )
    except OpenWorldSessionError as exc:
        raise RuntimeTaskBindingError(str(exc)) from exc


__all__ = [
    "RuntimeTaskBindingError",
    "build_runtime_open_world_evaluation_target",
    "build_runtime_policy_task_binding",
]
