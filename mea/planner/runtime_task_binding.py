"""Catalog-independent RoboTwin policy/task execution bindings.

This module discovers only the physical/runtime authority needed to execute a
policy on one base task.  Official rollout identity is source-driven; a
TaskSchema is an additional capability required by TaskGen and telemetry, not
a prerequisite for trying a multi-task policy on an official task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from mea.robotwin.task_identity import (
    RoboTwinTaskIdentityError,
    discover_robotwin_official_tasks,
    discover_robotwin_task_identity,
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
_BACKEND_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class RuntimePolicySpec:
    """Policy checkpoint and rollout contract, independent of one task."""

    backend: str
    policy_name: str
    checkpoint_id: str
    checkpoint_dir: Path
    task_scope: str
    rollout_kind: str
    rollout_entrypoint: str
    required_artifacts: tuple[str, ...]
    language_conditioned: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.backend, str)
            or _BACKEND_NAME.fullmatch(self.backend) is None
        ):
            raise RuntimeTaskBindingError(
                "policy backend must be a lowercase identifier"
            )
        for field_name in (
            "policy_name",
            "checkpoint_id",
            "task_scope",
            "rollout_kind",
            "rollout_entrypoint",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeTaskBindingError(
                    f"policy {field_name} must be a non-empty string"
                )
        if not isinstance(self.language_conditioned, bool):
            raise RuntimeTaskBindingError(
                "policy language_conditioned must be bool"
            )
        if not self.required_artifacts or any(
            not isinstance(item, str) or not item.strip()
            for item in self.required_artifacts
        ):
            raise RuntimeTaskBindingError(
                "policy required_artifacts must be non-empty paths"
            )
        object.__setattr__(
            self,
            "checkpoint_dir",
            Path(self.checkpoint_dir).expanduser().resolve(),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


def build_smolvla_policy_spec(
    checkpoint_dir: str | Path,
    *,
    checkpoint_id: str = "lerobot/smolvla_robotwin",
) -> RuntimePolicySpec:
    """Describe the shared SmolVLA RoboTwin checkpoint."""

    return RuntimePolicySpec(
        backend="smolvla",
        policy_name="SmolVLA",
        checkpoint_id=checkpoint_id,
        checkpoint_dir=Path(checkpoint_dir),
        task_scope="robotwin_official_tasks",
        rollout_kind="smolvla_robotwin",
        rollout_entrypoint="mea.robotwin.smolvla_rollout",
        required_artifacts=("config.json", "model.safetensors"),
        language_conditioned=True,
        metadata={
            "language_input": "task_name_words",
            "action_chunk_size": 50,
        },
    )


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


def _build_act_policy_spec(
    root: Path,
    task_name: str,
    *,
    checkpoint_setting: str,
    expert_data_num: int,
    language_conditioned: bool,
) -> RuntimePolicySpec:
    setting = _checkpoint_setting(checkpoint_setting)
    expert_count = _positive_int(
        expert_data_num,
        field="expert_data_num",
    )
    if not isinstance(language_conditioned, bool):
        raise RuntimeTaskBindingError("language_conditioned must be bool")
    checkpoint_leaf = f"{setting}-{expert_count}"
    return RuntimePolicySpec(
        backend="act",
        policy_name="ACT",
        checkpoint_id=f"act-{task_name}/{checkpoint_leaf}",
        checkpoint_dir=(
            root
            / "policy"
            / "ACT"
            / "act_ckpt"
            / f"act-{task_name}"
            / checkpoint_leaf
        ),
        task_scope=task_name,
        rollout_kind="act_eval_mea",
        rollout_entrypoint="policy/ACT/eval_mea.sh",
        required_artifacts=("dataset_stats.pkl", "policy_last.ckpt"),
        language_conditioned=language_conditioned,
        metadata={
            "checkpoint_setting": setting,
            "expert_data_num": expert_count,
        },
    )


def build_runtime_policy_task_binding(
    repo_root: str | Path,
    task_name: str,
    *,
    checkpoint_setting: str = "demo_clean",
    expert_data_num: int = 50,
    language_conditioned: bool = False,
    policy_spec: RuntimePolicySpec | None = None,
) -> dict[str, Any]:
    """Build a PolicyTaskBinding without catalog or adapter membership.

    Omitting ``policy_spec`` preserves the legacy ACT reader.  A shared
    multi-task policy supplies one explicit spec and can bind any discovered
    official task without a task-named checkpoint directory.
    """

    root = Path(repo_root).expanduser().resolve()
    try:
        identity = discover_robotwin_task_identity(root, task_name)
    except RoboTwinTaskIdentityError as exc:
        raise RuntimeTaskBindingError(str(exc)) from exc
    spec = policy_spec or _build_act_policy_spec(
        root,
        identity.task_name,
        checkpoint_setting=checkpoint_setting,
        expert_data_num=expert_data_num,
        language_conditioned=language_conditioned,
    )
    if spec.task_scope not in {
        identity.task_name,
        "robotwin_official_tasks",
    }:
        raise RuntimeTaskBindingError(
            "policy task_scope does not cover the requested task"
        )
    if spec.backend == "act" and not identity.task_schema_available:
        raise RuntimeTaskBindingError(
            "ACT runtime binding requires the task toolkit schema"
        )
    for relative_path in spec.required_artifacts:
        label = {
            ("act", "dataset_stats.pkl"): "ACT dataset statistics",
            ("act", "policy_last.ckpt"): "ACT policy weights",
        }.get(
            (spec.backend, relative_path),
            f"{spec.policy_name} checkpoint artifact",
        )
        _require_nonempty_file(
            spec.checkpoint_dir / relative_path,
            label=label,
        )

    try:
        policy_metadata = dict(spec.metadata)
        if spec.language_conditioned:
            language_input = str(
                policy_metadata.get("language_input") or "official_description"
            )
            policy_metadata["task_instruction"] = (
                identity.task_name.replace("_", " ")
                if language_input == "task_name_words"
                else identity.description
            )
        return build_policy_task_binding(
            task_name=identity.task_name,
            task_family=identity.task_family,
            task_module=f"envs.{identity.task_name}",
            task_schema_available=identity.task_schema_available,
            policy={
                "name": spec.policy_name,
                "backend": spec.backend,
                "language_conditioned": spec.language_conditioned,
                "task_schema_available": identity.task_schema_available,
                **policy_metadata,
            },
            checkpoint={
                "policy_name": spec.policy_name,
                "checkpoint_id": spec.checkpoint_id,
                "checkpoint_path": str(spec.checkpoint_dir),
                "task_scope": spec.task_scope,
                "ready": True,
                **dict(spec.metadata),
            },
            rollout={
                "kind": spec.rollout_kind,
                "entrypoint": spec.rollout_entrypoint,
                "task_name": identity.task_name,
            },
        )
    except PolicyTaskBindingError as exc:
        raise RuntimeTaskBindingError(str(exc)) from exc


def build_runtime_policy_task_manifest(
    repo_root: str | Path,
    policy_spec: RuntimePolicySpec,
) -> dict[str, Any]:
    """Bind a policy to its data-driven official RoboTwin task scope."""

    root = Path(repo_root).expanduser().resolve()
    try:
        if policy_spec.task_scope == "robotwin_official_tasks":
            identities = discover_robotwin_official_tasks(root)
        else:
            identities = (
                discover_robotwin_task_identity(
                    root,
                    policy_spec.task_scope,
                ),
            )
    except RoboTwinTaskIdentityError as exc:
        raise RuntimeTaskBindingError(str(exc)) from exc
    bindings = [
        build_runtime_policy_task_binding(
            root,
            identity.task_name,
            policy_spec=policy_spec,
        )
        for identity in identities
    ]
    return {
        "schema_version": 1,
        "policy_backend": policy_spec.backend,
        "policy_name": policy_spec.policy_name,
        "checkpoint_id": policy_spec.checkpoint_id,
        "task_scope": policy_spec.task_scope,
        "task_count": len(bindings),
        "tasks": [
            {
                "task_name": binding["task_name"],
                "task_module": binding["task_module"],
                "task_schema_available": bool(
                    binding["policy"]["task_schema_available"]
                ),
                "official_rollout_ready": True,
                "policy_task_binding": binding,
            }
            for binding in bindings
        ],
    }


def build_runtime_open_world_evaluation_target(
    repo_root: str | Path,
    task_name: str,
    *,
    max_rounds: int,
    checkpoint_setting: str = "demo_clean",
    expert_data_num: int = 50,
    language_conditioned: bool = False,
    policy_spec: RuntimePolicySpec | None = None,
) -> dict[str, Any]:
    """Build and validate a catalog-free OpenWorldEvaluationTarget."""

    budget = _positive_int(max_rounds, field="max_rounds")
    binding = build_runtime_policy_task_binding(
        repo_root,
        task_name,
        checkpoint_setting=checkpoint_setting,
        expert_data_num=expert_data_num,
        language_conditioned=language_conditioned,
        policy_spec=policy_spec,
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
    "RuntimePolicySpec",
    "RuntimeTaskBindingError",
    "build_smolvla_policy_spec",
    "build_runtime_open_world_evaluation_target",
    "build_runtime_policy_task_manifest",
    "build_runtime_policy_task_binding",
]
