"""One native RoboTwin method round over the shared MethodRuntime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .native_round_assembly import assemble_robotwin_method_round
from .native_round_contracts import NativeRoundPreparation
from .native_round_evaluation import evaluate_robotwin_method_round
from .native_round_materialization import prepare_robotwin_method_round
from .native_round_policy import invoke_robotwin_policy
from .runtime import AcceptedTaskGenMaterializer, RoboTwinRolloutRunner


def execute_robotwin_method_round(
    *,
    policy_backend: str,
    policy_name: str,
    rollout_runner: RoboTwinRolloutRunner,
    repo_root: str | Path,
    evaluation_dir: str | Path,
    evaluation_id: str,
    round_plan: Mapping[str, Any],
    runtime_target: Mapping[str, Any],
    telemetry_profile: str,
    provider: Any = None,
    text_model: str = "",
    vision_model: str = "",
    max_reflections: int = 1,
    generated_task_materializer: AcceptedTaskGenMaterializer | None = None,
    execution_vqa_connected: bool = True,
    rollout_output_subdir: str | None = "evaluation",
) -> dict[str, Any]:
    prepared = prepare_robotwin_method_round(
        policy_backend=policy_backend,
        policy_name=policy_name,
        rollout_runner=rollout_runner,
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
        execution_vqa_connected=execution_vqa_connected,
        rollout_output_subdir=rollout_output_subdir,
    )
    if not isinstance(prepared, NativeRoundPreparation):
        return prepared
    rollouts = invoke_robotwin_policy(
        prepared,
        round_plan=round_plan,
        evaluation_id=evaluation_id,
        policy_backend=policy_backend,
    )
    evaluated = evaluate_robotwin_method_round(
        prepared,
        rollouts,
        round_plan=round_plan,
        policy_backend=policy_backend,
        policy_name=policy_name,
    )
    return assemble_robotwin_method_round(
        prepared,
        evaluated,
        round_plan=round_plan,
        policy_backend=policy_backend,
        policy_name=policy_name,
    )
