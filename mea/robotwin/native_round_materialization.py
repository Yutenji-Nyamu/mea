"""Task binding and Proposal materialization for a native RoboTwin round."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import BackendBindingRequest, CandidateRequest, MethodRuntime
from mea.planner.experiment_candidate import validate_experiment_candidate
from mea.planner.policy_task_binding import policy_task_binding_from_target
from mea.taskgen.attempts import CandidateUnexecutableError
from mea.taskgen.generic_backend import GenericTaskGenError

from .native_round_contracts import (
    NativeAgentRoundError,
    NativeRoundPreparation,
    build_native_run_id as _build_native_run_id,
)
from .native_round_failures import (
    _candidate_unexecutable_round,
    _taskgen_materialization_failure_round,
    _unsupported_round,
)
from .runtime import (
    AcceptedTaskGenMaterializer,
    RoboTwinMethodBackend,
    RoboTwinRolloutRunner,
)


def prepare_robotwin_method_round(
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
) -> NativeRoundPreparation | dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    evaluation_root = Path(evaluation_dir).expanduser().resolve()
    contract = policy_task_binding_from_target(runtime_target)
    if contract["policy"].get("backend") != policy_backend:
        raise NativeAgentRoundError(
            f"native {policy_name} round requires a {policy_backend} "
            "PolicyTaskBinding"
        )
    if contract["task_name"] != round_plan.get("task_name"):
        raise NativeAgentRoundError(
            f"round task differs from the bound {policy_name} task"
        )
    execution = round_plan.get("execution")
    seeds = execution.get("seeds") if isinstance(execution, Mapping) else None
    if (
        not isinstance(seeds, list)
        or len(seeds) != 1
        or isinstance(seeds[0], bool)
        or not isinstance(seeds[0], int)
    ):
        raise NativeAgentRoundError(
            f"native {policy_name} production rounds require exactly one seed"
        )
    seed = int(seeds[0])
    proposal_value = round_plan.get("proposal") or round_plan.get(
        "experiment_candidate"
    )
    proposal = (
        validate_experiment_candidate(proposal_value)
        if isinstance(proposal_value, Mapping)
        else None
    )
    generated_task_required = proposal is not None and (
        proposal["scene_need"] is not None
        or proposal["checker_need"] is not None
    )
    if (
        generated_task_required
        and generated_task_materializer is None
    ):
        return _unsupported_round(
            root=root,
            evaluation_root=evaluation_root,
            evaluation_id=evaluation_id,
            round_plan=round_plan,
            task_name=contract["task_name"],
            proposal=proposal,
            policy_backend=policy_backend,
            policy_name=policy_name,
            reason_code=f"{policy_backend}_taskgen_not_connected",
            reason=(
                f"The native {policy_name} MethodRuntime has no injected "
                "generic TaskGen materializer for this scene/checker Proposal."
            ),
        )
    if (
        proposal is not None
        and proposal["vqa_tool_need"] is not None
        and not execution_vqa_connected
    ):
        return _unsupported_round(
            root=root,
            evaluation_root=evaluation_root,
            evaluation_id=evaluation_id,
            round_plan=round_plan,
            task_name=contract["task_name"],
            proposal=proposal,
            policy_backend=policy_backend,
            policy_name=policy_name,
            reason_code=f"{policy_backend}_vqa_not_connected",
            reason=f"The native {policy_name} VQA bridge is not connected.",
        )
    if proposal is None and round_plan.get("route") != "official":
        raise NativeAgentRoundError(
            f"candidate-free {policy_name} execution requires an official round"
        )
    if (
        isinstance(max_reflections, bool)
        or not isinstance(max_reflections, int)
        or max_reflections < 0
    ):
        raise NativeAgentRoundError(
            "max_reflections must be a non-negative integer"
        )

    run_id = _build_native_run_id(
        evaluation_id,
        str(round_plan["round_id"]),
        policy_backend,
    )
    child_dir = root / "mea" / "generated_tasks" / run_id
    query = str(round_plan["task_instruction"])
    if generated_task_required:
        assert generated_task_materializer is not None
        if (
            provider is None
            or not isinstance(text_model, str)
            or not text_model.strip()
            or not isinstance(vision_model, str)
            or not vision_model.strip()
        ):
            raise NativeAgentRoundError(
                "generated TaskGen execution requires provider, text model, "
                "and vision model"
            )
    else:
        child_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = (
        child_dir / rollout_output_subdir
        if rollout_output_subdir is not None
        else child_dir
    )
    backend = RoboTwinMethodBackend(
        repo_root=root,
        rollout_runner=rollout_runner,
        accepted_taskgen_materializer=generated_task_materializer,
        taskgen_provider=provider,
        taskgen_text_model=text_model,
        taskgen_vision_model=vision_model,
        taskgen_telemetry_profile=telemetry_profile,
    )
    runtime = MethodRuntime(backend)
    binding = runtime.bind_task(
        BackendBindingRequest(
            task_reference={
                "task_name": contract["task_name"],
                "binding_id": (
                    f"{contract['task_name']}/{contract['policy']['name']}"
                ),
                "policy": deepcopy(contract["policy"]),
            },
            artifacts={
                "checkpoint": str(
                    contract["checkpoint"]["checkpoint_path"]
                )
            },
            metadata={
                "checkpoint_id": contract["checkpoint"]["checkpoint_id"],
            },
        )
    )
    if proposal is None:
        candidate = backend.official_candidate(
            binding,
            source_query=query,
            seed=seed,
            candidate_id=str(
                round_plan.get("candidate_id")
                or round_plan.get("template_id")
                or "official_control"
            ),
        )
    else:
        try:
            candidate = runtime.materialize_candidate(
                binding,
                CandidateRequest(
                    candidate_id=proposal["candidate_id"],
                    source_query=proposal["source_query"],
                    proposal_bundle=proposal,
                    output_dir=child_dir,
                    seed=seed,
                    context={
                        "taskgen_run_id": run_id,
                        "requested_max_reflections": max_reflections,
                    },
                ),
            )
        except CandidateUnexecutableError:
            return _candidate_unexecutable_round(
                evaluation_root=evaluation_root,
                round_plan=round_plan,
                child_dir=child_dir,
                proposal=proposal,
                policy_backend=policy_backend,
                policy_name=policy_name,
            )
        except GenericTaskGenError:
            return _taskgen_materialization_failure_round(
                evaluation_root=evaluation_root,
                round_plan=round_plan,
                child_dir=child_dir,
                proposal=proposal,
                policy_backend=policy_backend,
                policy_name=policy_name,
            )
    taskgen_manifest: dict[str, Any] | None = None
    if generated_task_required:
        manifest_path = Path(candidate.artifacts["manifest"])
        try:
            materialized_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise NativeAgentRoundError(
                "materialized TaskGen manifest is unavailable"
            ) from exc
        if not isinstance(materialized_manifest, Mapping):
            raise NativeAgentRoundError(
                "materialized TaskGen manifest must be an object"
            )
        taskgen_manifest = deepcopy(dict(materialized_manifest))
    return NativeRoundPreparation(
        root=root,
        evaluation_root=evaluation_root,
        contract=contract,
        seed=seed,
        proposal=proposal,
        generated_task_required=generated_task_required,
        run_id=run_id,
        child_dir=child_dir,
        rollout_dir=rollout_dir,
        query=query,
        runtime=runtime,
        binding=binding,
        candidate=candidate,
        taskgen_manifest=taskgen_manifest,
    )
