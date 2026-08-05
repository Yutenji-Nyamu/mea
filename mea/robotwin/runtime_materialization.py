"""Proposal materialization for the RoboTwin MethodRuntime backend."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import (
    BackendTaskBinding,
    CandidateRequest,
    MaterializedCandidate,
)
from mea.planner.experiment_candidate import (
    ExperimentCandidateError,
    validate_experiment_candidate,
)
from mea.robotwin_task_context import (
    RoboTwinTaskContextError,
    resolve_robotwin_task_context,
)
from mea.taskgen.attempts import CandidateUnexecutableError
from mea.taskgen.generic_backend import GenericRoboTwinTaskAdapter, GenericTaskGenError
from mea.taskgen.runtime import record_generic_taskgen_generation_failure
from mea.visual_capture import visual_capture_profile_for_proposal

from .runtime_contracts import _RoboTwinNativeCandidate, _write_json
from .runtime_taskgen_manifest import (
    _validate_taskgen_checker_artifacts,
    bind_validated_taskgen_candidate,
    taskgen_rollout_manifest,
)
from .task_identity import RoboTwinTaskIdentity


def materialize_candidate(
    backend: Any,
    binding: BackendTaskBinding,
    request: CandidateRequest,
) -> MaterializedCandidate:
    adapter = binding.native_task
    if not isinstance(
        adapter,
        (GenericRoboTwinTaskAdapter, RoboTwinTaskIdentity),
    ):
        raise TypeError(
            "RoboTwin binding native_task has the wrong runtime type"
        )
    try:
        candidate = validate_experiment_candidate(
            request.proposal_bundle
        )
    except ExperimentCandidateError as exc:
        raise ValueError(f"invalid Proposal: {exc}") from exc
    if candidate["candidate_id"] != request.candidate_id:
        raise ValueError(
            "CandidateRequest.candidate_id differs from "
            "Proposal.candidate_id"
        )
    if candidate["source_query"] != request.source_query:
        raise ValueError(
            "CandidateRequest.source_query differs from "
            "Proposal.source_query"
        )
    if candidate["base_task"] != adapter.task_name:
        raise ValueError(
            "Proposal.base_task differs from the bound task"
        )

    taskgen_required = bool(
        candidate["scene_need"] is not None
        or candidate["checker_need"] is not None
    )
    task_context_value = binding.task_contract.get("task_context")
    execution_schema = binding.task_contract.get("task_schema")
    task_context_artifact: Path | None = None
    tool_observation_required = bool(
        candidate["rule_tool_need"] is not None
        or candidate["vqa_tool_need"] is not None
    )
    if (
        not taskgen_required
        and tool_observation_required
        and not isinstance(execution_schema, Mapping)
    ):
        supplied_probe = request.context.get("runtime_task_context_probe")
        if supplied_probe is not None and not isinstance(
            supplied_probe, Mapping
        ):
            raise ValueError(
                "runtime_task_context_probe must be an object"
            )
        policy = binding.task_contract.get("policy")
        action_dimension = (
            policy.get("action_dimension", 0)
            if isinstance(policy, Mapping)
            else 0
        )
        if (
            isinstance(action_dimension, bool)
            or not isinstance(action_dimension, int)
            or action_dimension < 1
        ):
            raise ValueError(
                "schema-less Tool Proposal requires the bound policy "
                "action_dimension"
            )
        try:
            runtime_probe = (
                deepcopy(dict(supplied_probe))
                if isinstance(supplied_probe, Mapping)
                else deepcopy(
                    dict(
                        backend.task_context_probe_runner(
                            repo_root=backend.repo_root,
                            task_name=adapter.task_name,
                            seed=request.seed,
                            action_dimension=action_dimension,
                        )
                    )
                )
            )
            task_context = resolve_robotwin_task_context(
                backend.repo_root,
                adapter.task_name,
                runtime_probe=runtime_probe,
            )
        except (RoboTwinTaskContextError, TypeError, ValueError) as exc:
            raise ValueError(
                "Tool Proposal could not establish pre-rollout "
                f"TaskContext authority: {exc}"
            ) from exc
        if task_context.task_schema is None:
            raise ValueError(
                "Tool Proposal pre-rollout TaskContext has no telemetry "
                "schema"
            )
        task_context_value = task_context.to_dict()
        execution_schema = deepcopy(dict(task_context.task_schema))
        task_context_artifact = (
            request.output_dir.resolve()
            / "validation"
            / "task_context.json"
        )
        _write_json(task_context_artifact, task_context_value)
    if taskgen_required:
        if backend.accepted_taskgen_materializer is not None:
            if (
                backend.taskgen_provider is None
                or not backend.taskgen_text_model
                or not backend.taskgen_vision_model
            ):
                raise ValueError(
                    "accepted TaskGen materialization requires provider, "
                    "text model, and vision model"
                )
            run_id = str(
                request.context.get("taskgen_run_id")
                or request.output_dir.name
            )
            policy = binding.task_contract.get("policy")
            action_dimension = (
                policy.get("action_dimension", 0)
                if isinstance(policy, Mapping)
                else 0
            )
            try:
                accepted_manifest = backend.accepted_taskgen_materializer(
                    backend.repo_root,
                    user_request=request.source_query,
                    provider=backend.taskgen_provider,
                    model=backend.taskgen_text_model,
                    vision_model=backend.taskgen_vision_model,
                    experiment_candidate=candidate,
                    run_id=run_id,
                    seed=request.seed,
                    telemetry_profile=backend.taskgen_telemetry_profile,
                    action_dimension=int(action_dimension or 0),
                )
            except CandidateUnexecutableError as exc:
                record_generic_taskgen_generation_failure(
                    backend.repo_root,
                    run_id=run_id,
                    user_request=request.source_query,
                    experiment_candidate=candidate,
                    model=backend.taskgen_text_model,
                    telemetry_profile=backend.taskgen_telemetry_profile,
                    error=exc,
                )
                raise
            except GenericTaskGenError as exc:
                record_generic_taskgen_generation_failure(
                    backend.repo_root,
                    run_id=run_id,
                    user_request=request.source_query,
                    experiment_candidate=candidate,
                    model=backend.taskgen_text_model,
                    telemetry_profile=backend.taskgen_telemetry_profile,
                    error=exc,
                )
                raise
            if not isinstance(accepted_manifest, Mapping):
                raise ValueError(
                    "accepted_taskgen_materializer must return a "
                    "TaskGen manifest"
                )
            return bind_validated_taskgen_candidate(
                backend,
                binding,
                request,
                accepted_manifest,
            )
        if not isinstance(adapter, GenericRoboTwinTaskAdapter):
            raise ValueError(
                "generated scene/checker requires a validated TaskSchema"
            )
        if backend.taskgen_backend is None:
            raise ValueError(
                "generated scene/checker requires a TaskGen backend"
            )
        run_id = str(
            request.context.get("taskgen_run_id")
            or request.output_dir.name
        )
        resolution = backend.taskgen_backend.materialize(
            candidate,
            adapter,
            run_id=run_id,
            max_regenerations=int(
                request.context.get("max_regenerations", 1)
            ),
        )
        manifest, artifacts = taskgen_rollout_manifest(backend, resolution)
        _validate_taskgen_checker_artifacts(
            candidate=candidate,
            manifest=manifest,
            candidate_manifest_path=Path(
                artifacts["candidate_manifest"]
            ),
            task_source_path=Path(artifacts["task_source"]),
        )
        validation = {
            "route": resolution["route"],
            "status": resolution["status"],
            "provider_call_count": int(
                resolution.get("provider_call_count") or 0
            ),
            "taskgen": deepcopy(
                dict(resolution.get("validation") or {})
            ),
        }
        run_dir = Path(artifacts["task_source"]).parent
    else:
        resolution = {
            "schema_version": 1,
            "status": "bypassed",
            "route": "official_task_tool_only",
            "candidate": candidate,
            "provider_required": False,
            "provider_call_count": 0,
        }
        manifest = {
            "schema_version": 1,
            "status": "official",
            "task_name": adapter.task_name,
            "task_module": f"envs.{adapter.task_name}",
            "generation_kind": "official_passthrough",
            "proposal": candidate,
            **(
                {
                    "task_context": {
                        "path": str(task_context_artifact),
                        "schema_origin": "runtime_probe",
                    }
                }
                if task_context_artifact is not None
                else {}
            ),
        }
        artifacts = {
            "official_source": adapter.official_source,
            "task_module": manifest["task_module"],
            **(
                {"task_context": str(task_context_artifact)}
                if task_context_artifact is not None
                else {}
            ),
        }
        validation = {
            "route": resolution["route"],
            "status": resolution["status"],
            "provider_call_count": 0,
        }
        run_dir = request.output_dir.resolve()

    run_dir.mkdir(parents=True, exist_ok=True)
    overlay = run_dir / "overlay.yml"
    if not overlay.is_file():
        overlay.write_text("{}\n", encoding="utf-8")
    manifest = {
        **manifest,
        "overlay": str(overlay),
    }
    artifacts.update(
        {
            "run_dir": str(run_dir),
            "overlay": str(overlay),
        }
    )

    native = _RoboTwinNativeCandidate(
        adapter=adapter,
        experiment_candidate=candidate,
        taskgen_resolution=resolution,
        rollout_manifest=manifest,
    )
    return MaterializedCandidate(
        benchmark=backend.benchmark,
        candidate_id=request.candidate_id,
        binding_id=binding.binding_id,
        source_query=request.source_query,
        task_contract={
            **dict(binding.task_contract),
            "candidate_id": request.candidate_id,
            "semantic_concern": candidate["semantic_concern"],
            "task_module": manifest["task_module"],
            "task_schema": (
                deepcopy(dict(execution_schema))
                if isinstance(execution_schema, Mapping)
                else None
            ),
            "task_schema_available": isinstance(
                execution_schema,
                Mapping,
            ),
            "task_context": (
                deepcopy(dict(task_context_value))
                if isinstance(task_context_value, Mapping)
                else None
            ),
            "visual_capture_profile_id": (
                visual_capture_profile_for_proposal(candidate)
            ),
        },
        native_task=native,
        artifacts={**binding.artifacts, **artifacts},
        validation=validation,
        metadata={
            "official_control": False,
            "official_task_reused": not taskgen_required,
            "taskgen_route": resolution["route"],
            "task_context_bound_before_rollout": isinstance(
                execution_schema,
                Mapping,
            ),
        },
    )
