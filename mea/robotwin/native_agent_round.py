"""Native RoboTwin policy-round entry points for the production Plan Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mea.robotwin.act_rollout import ACTRobotwinRolloutRunner
from mea.robotwin.hyvla_rollout import HyVLARobotwinRolloutRunner
from mea.robotwin.smolvla_rollout import SmolVLARobotwinRolloutRunner

from .native_round_contracts import NativeAgentRoundError
from .native_round_execution import (
    execute_robotwin_method_round as _execute_robotwin_method_round,
)
from .runtime import AcceptedTaskGenMaterializer

GeneratedTaskMaterializer = AcceptedTaskGenMaterializer


def execute_smolvla_method_round(
    *,
    repo_root: str | Path,
    evaluation_dir: str | Path,
    evaluation_id: str,
    round_plan: Mapping[str, Any],
    runtime_target: Mapping[str, Any],
    telemetry_profile: str,
    policy_server_port: int,
    gpu: int = 0,
    provider: Any = None,
    text_model: str = "",
    vision_model: str = "",
    max_reflections: int = 1,
    generated_task_materializer: (
        GeneratedTaskMaterializer | None
    ) = None,
) -> dict[str, Any]:
    """Construct a SmolVLA runner and execute the shared native round."""

    del gpu
    return _execute_robotwin_method_round(
        policy_backend="smolvla",
        policy_name="SmolVLA",
        rollout_runner=SmolVLARobotwinRolloutRunner(
            port=policy_server_port,
            repo_root=repo_root,
            telemetry_profile=telemetry_profile,
        ),
        repo_root=repo_root,
        evaluation_dir=evaluation_dir,
        evaluation_id=evaluation_id,
        round_plan=round_plan,
        runtime_target=runtime_target,
        telemetry_profile=telemetry_profile,
        provider=provider,
        text_model=text_model,
        vision_model=vision_model,
        max_reflections=max_reflections,
        generated_task_materializer=generated_task_materializer,
        execution_vqa_connected=True,
        rollout_output_subdir="evaluation",
    )


def execute_hyvla_method_round(
    *,
    repo_root: str | Path,
    evaluation_dir: str | Path,
    evaluation_id: str,
    round_plan: Mapping[str, Any],
    runtime_target: Mapping[str, Any],
    telemetry_profile: str,
    policy_server_port: int,
    gpu: int = 0,
    provider: Any = None,
    text_model: str = "",
    vision_model: str = "",
    max_reflections: int = 1,
    generated_task_materializer: (
        GeneratedTaskMaterializer | None
    ) = None,
) -> dict[str, Any]:
    """Execute a round against an explicitly started Hy-VLA server."""

    del gpu
    return _execute_robotwin_method_round(
        policy_backend="hyvla",
        policy_name="Hy-VLA",
        rollout_runner=HyVLARobotwinRolloutRunner(
            port=policy_server_port,
            repo_root=repo_root,
            telemetry_profile=telemetry_profile,
        ),
        repo_root=repo_root,
        evaluation_dir=evaluation_dir,
        evaluation_id=evaluation_id,
        round_plan=round_plan,
        runtime_target=runtime_target,
        telemetry_profile=telemetry_profile,
        provider=provider,
        text_model=text_model,
        vision_model=vision_model,
        max_reflections=max_reflections,
        generated_task_materializer=generated_task_materializer,
        execution_vqa_connected=True,
        rollout_output_subdir="evaluation",
    )


def execute_act_method_round(
    *,
    repo_root: str | Path,
    evaluation_dir: str | Path,
    evaluation_id: str,
    round_plan: Mapping[str, Any],
    runtime_target: Mapping[str, Any],
    telemetry_profile: str,
    policy_server_port: int,
    gpu: int = 0,
    provider: Any = None,
    text_model: str = "",
    vision_model: str = "",
    max_reflections: int = 1,
    generated_task_materializer: (
        GeneratedTaskMaterializer | None
    ) = None,
) -> dict[str, Any]:
    """Construct an ACT runner and execute the shared native round.

    ``policy_server_port`` belongs to the common native-backend call contract;
    ACT runs in-process and intentionally ignores it.
    """

    del policy_server_port
    return _execute_robotwin_method_round(
        policy_backend="act",
        policy_name="ACT",
        rollout_runner=ACTRobotwinRolloutRunner(
            repo_root=repo_root,
            gpu=gpu,
            telemetry_profile=telemetry_profile,
        ),
        repo_root=repo_root,
        evaluation_dir=evaluation_dir,
        evaluation_id=evaluation_id,
        round_plan=round_plan,
        runtime_target=runtime_target,
        telemetry_profile=telemetry_profile,
        provider=provider,
        text_model=text_model,
        vision_model=vision_model,
        max_reflections=max_reflections,
        generated_task_materializer=generated_task_materializer,
        execution_vqa_connected=True,
        rollout_output_subdir=None,
    )


__all__ = [
    "GeneratedTaskMaterializer",
    "NativeAgentRoundError",
    "execute_act_method_round",
    "execute_hyvla_method_round",
    "execute_smolvla_method_round",
]
