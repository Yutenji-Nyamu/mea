"""Native RoboTwin policy rounds for the production Plan Agent.

The module owns only the benchmark-specific MethodRuntime boundary.  Planning,
ToolGen, Aggregate, VQA, and answer construction remain in the existing Agent
loop.  Policy wrappers only construct their rollout runner; the shared
``_execute_robotwin_method_round`` owns bind/materialize/rollout/evidence.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import (
    BackendBindingRequest,
    CandidateRequest,
    EvidenceRequest,
    MethodRuntime,
    RolloutRequest,
)
from mea.planner.experiment_candidate import validate_experiment_candidate
from mea.planner.policy_task_binding import policy_task_binding_from_target
from mea.robotwin.act_rollout import ACTRobotwinRolloutRunner
from mea.robotwin.hyvla_rollout import HyVLARobotwinRolloutRunner
from mea.robotwin.runtime import (
    AcceptedTaskGenMaterializer,
    RoboTwinMethodBackend,
    RoboTwinRolloutRunner,
)
from mea.robotwin.smolvla_rollout import SmolVLARobotwinRolloutRunner
from mea.taskgen.attempts import CandidateUnexecutableError
from mea.taskgen.rollout_evidence import (
    evaluate_generic_task_rollout_telemetry,
)


class NativeAgentRoundError(RuntimeError):
    """Raised when a native policy round exceeds its validated capabilities."""


# Compatibility name retained for wrapper callers; the backend owns the type.
GeneratedTaskMaterializer = AcceptedTaskGenMaterializer


def _build_native_run_id(
    evaluation_id: str,
    round_id: str,
    policy_backend: str,
) -> str:
    """Return one stable, importable generated-task package identifier."""

    digest = hashlib.sha256(
        f"{evaluation_id}:{round_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"run_native_{policy_backend}_{digest}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _unsupported_round(
    *,
    root: Path,
    evaluation_root: Path,
    evaluation_id: str,
    round_plan: Mapping[str, Any],
    task_name: str,
    proposal: Mapping[str, Any] | None,
    policy_backend: str,
    policy_name: str,
    reason_code: str,
    reason: str,
) -> dict[str, Any]:
    """Persist an unsupported capability as evidence, not a process crash."""

    run_id = _build_native_run_id(
        evaluation_id,
        str(round_plan["round_id"]),
        policy_backend,
    )
    child_dir = root / "mea" / "generated_tasks" / run_id
    candidate_id = str(
        (proposal or {}).get("candidate_id")
        or round_plan.get("candidate_id")
        or round_plan.get("template_id")
        or "unsupported_candidate"
    )
    method_runtime = {
        "schema_version": 1,
        "status": "unsupported",
        "reason_code": reason_code,
        "reason": reason,
        "candidate_id": candidate_id,
        "proposal": deepcopy(dict(proposal)) if proposal is not None else None,
    }
    child_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "unsupported",
        "task_name": task_name,
        "task_module": f"envs.{task_name}",
        "generation_kind": "unsupported",
        "policy_backend": policy_backend,
        "unsupported_capability": {
            "reason_code": reason_code,
            "reason": reason,
        },
        "scene_validation": {
            "render_success": False,
            "rule_check": {
                "passed": False,
                "authority": "runtime_capability_boundary",
            },
        },
        "act_evaluation": {
            "passed": False,
            "actual_seeds": [],
            "policy_name": policy_name,
        },
        "task_artifact_summary": {
            "success_official_equivalent": None,
            "success_execution_scope": "not_executed",
        },
        "trusted_tool_evaluation": {
            "schema_version": 1,
            "status": "skipped",
            "outcome_metric": None,
            "outcome_authority": None,
            "episode_count": 0,
            "episodes": [],
        },
        "method_runtime": method_runtime,
    }
    manifest_path = child_dir / "manifest.json"
    method_runtime_path = (
        evaluation_root
        / "execution"
        / str(round_plan["round_id"])
        / "method_runtime.json"
    )
    _write_json(manifest_path, child_manifest)
    _write_json(method_runtime_path, method_runtime)
    return {
        "child_manifest": child_manifest,
        "child_dir": child_dir,
        "manifest_path": manifest_path,
        "method_runtime_path": method_runtime_path,
        "semantic_telemetry_ready": False,
        "candidate_id": candidate_id,
        "evidence_outcome": "unsupported",
        "unsupported": True,
    }


def _candidate_unexecutable_round(
    *,
    evaluation_root: Path,
    round_plan: Mapping[str, Any],
    child_dir: Path,
    proposal: Mapping[str, Any],
    policy_backend: str,
    policy_name: str,
) -> dict[str, Any]:
    """Return one rejected TaskGen candidate as pre-policy planning evidence."""

    manifest_path = child_dir / "manifest.json"
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeAgentRoundError(
            "candidate-unexecutable TaskGen manifest is unavailable"
        ) from exc
    if not isinstance(raw_manifest, Mapping):
        raise NativeAgentRoundError(
            "candidate-unexecutable TaskGen manifest must be an object"
        )
    child_manifest = deepcopy(dict(raw_manifest))
    failure = child_manifest.get("failure")
    policy_execution = child_manifest.get("policy_execution")
    if not (
        child_manifest.get("status") == "candidate_unexecutable"
        and isinstance(failure, Mapping)
        and failure.get("failure_kind") == "candidate_unexecutable"
        and isinstance(policy_execution, Mapping)
        and policy_execution.get("rollouts_started") == 0
        and policy_execution.get("sample_count") == 0
    ):
        raise NativeAgentRoundError(
            "candidate-unexecutable TaskGen manifest changed its typed boundary"
        )
    candidate_id = str(proposal["candidate_id"])
    planning_observation = {
        "schema_version": 1,
        "kind": "candidate_unexecutable",
        "candidate_id": candidate_id,
        "sub_aspect": str(round_plan["sub_aspect"]),
        "reason_code": "taskgen_expert_gate_candidate_unexecutable",
        "diagnosis": str(failure.get("diagnosis") or failure.get("message")),
        "policy_rollouts_started": 0,
        "policy_sample_count": 0,
        "taskgen_attempt_summary": child_manifest.get(
            "task_generation_attempts"
        ),
    }
    method_runtime = {
        "schema_version": 1,
        "status": "candidate_unexecutable",
        "candidate_id": candidate_id,
        "proposal": deepcopy(dict(proposal)),
        "planning_observation": planning_observation,
    }
    child_manifest.update(
        {
            "policy_backend": policy_backend,
            "candidate_unexecutable": planning_observation,
            "act_evaluation": {
                "passed": False,
                "actual_seeds": [],
                "policy_name": policy_name,
                "outcome_metric": None,
                "outcome_value": None,
            },
            "task_artifact_summary": {
                "success_official_equivalent": None,
                "success_execution_scope": "not_executed",
            },
            "trusted_tool_evaluation": {
                "schema_version": 1,
                "status": "skipped",
                "outcome_metric": None,
                "outcome_authority": None,
                "episode_count": 0,
                "episodes": [],
            },
            "method_runtime": method_runtime,
        }
    )
    method_runtime_path = (
        evaluation_root
        / "execution"
        / str(round_plan["round_id"])
        / "method_runtime.json"
    )
    _write_json(manifest_path, child_manifest)
    _write_json(method_runtime_path, method_runtime)
    return {
        "child_manifest": child_manifest,
        "child_dir": child_dir,
        "manifest_path": manifest_path,
        "method_runtime_path": method_runtime_path,
        "semantic_telemetry_ready": False,
        "candidate_id": candidate_id,
        "evidence_outcome": "candidate_unexecutable",
        "candidate_unexecutable": True,
    }


def _artifact_exists(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return Path(value).expanduser().is_file()
    except OSError:
        return False


def _trusted_checker_result(
    evaluation: Mapping[str, Any],
    *,
    expected_metric: str,
) -> dict[str, Any]:
    """Return the one policy ToolResult that owns the round outcome."""

    if evaluation.get("status") != "passed":
        raise NativeAgentRoundError(
            "trusted checker evaluation did not pass"
        )
    if evaluation.get("outcome_metric") != expected_metric:
        raise NativeAgentRoundError(
            "trusted checker metric differs from the executed checker"
        )
    policy_episodes = [
        episode
        for episode in evaluation.get("episodes", [])
        if isinstance(episode, Mapping)
        and episode.get("role") == "policy_under_evaluation"
    ]
    if len(policy_episodes) != 1:
        raise NativeAgentRoundError(
            "trusted checker evaluation requires exactly one policy episode"
        )
    results = [
        result
        for result in policy_episodes[0].get("tool_results", [])
        if isinstance(result, Mapping)
        and result.get("tool") == expected_metric
    ]
    if len(results) != 1:
        raise NativeAgentRoundError(
            "trusted checker evaluation requires exactly one bound ToolResult"
        )
    result = deepcopy(dict(results[0]))
    if not isinstance(result.get("value"), bool):
        raise NativeAgentRoundError(
            "trusted checker ToolResult requires a boolean value"
        )
    if result.get("passed") is not result["value"]:
        raise NativeAgentRoundError(
            "trusted checker ToolResult passed/value semantics disagree"
        )
    return result


def _project_trusted_checker_outcome(
    rollout: Any,
    evaluation: Mapping[str, Any],
    *,
    expected_metric: str,
    policy_backend: str,
) -> tuple[Any, dict[str, Any]]:
    """Project MethodRuntime evidence onto the same result Aggregate consumes."""

    result = _trusted_checker_result(
        evaluation,
        expected_metric=expected_metric,
    )
    episode = rollout.episode
    if policy_backend in {"smolvla", "hyvla"}:
        if episode.get("active_checker_metric") != expected_metric:
            raise NativeAgentRoundError(
                f"{policy_backend} active checker differs from trusted ToolResult"
            )
        if not isinstance(
            episode.get("episode_latched_success"),
            bool,
        ):
            raise NativeAgentRoundError(
                f"{policy_backend} result lacks an explicit episode latch"
            )
        if expected_metric == "generated_check_success":
            generated = episode.get("generated_checker_success")
            official_core = episode.get(
                "official_core_predicate_satisfied"
            )
            details = result.get("details")
            if (
                not isinstance(generated, bool)
                or not isinstance(official_core, bool)
                or not isinstance(details, Mapping)
                or details.get("generated_checker_success") is not generated
                or details.get("official_core_predicate_satisfied")
                is not official_core
            ):
                raise NativeAgentRoundError(
                    f"{policy_backend} generated/official checker channels disagree "
                    "with the trusted ToolResult"
                )
        elif episode.get("official_check_success") is not result["value"]:
            raise NativeAgentRoundError(
                f"{policy_backend} official checker differs from trusted ToolResult"
            )
    projected = replace(
        rollout,
        success=result["value"],
        metadata={
            **dict(rollout.metadata),
            "trusted_checker": {
                "metric": expected_metric,
                "authority": evaluation.get("outcome_authority"),
                "value": result["value"],
            },
        },
    )
    return projected, result


def _execute_robotwin_method_round(
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
    generated_task_materializer: GeneratedTaskMaterializer | None = None,
    execution_vqa_connected: bool = True,
    rollout_output_subdir: str | None = "evaluation",
) -> dict[str, Any]:
    """Run one policy candidate through the shared RoboTwin MethodRuntime.

    Scene/checker generation is configured on ``RoboTwinMethodBackend``.
    ``MethodRuntime.materialize_candidate`` is the single production
    materialization entry; this function only binds the task, requests the
    candidate, executes the rollout, and projects evidence.
    """

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
    rollout = runtime.rollout(
        candidate,
        RolloutRequest(
            round_id=str(round_plan["round_id"]),
            seed=seed,
            output_dir=rollout_dir,
            provenance={
                "evaluation_id": evaluation_id,
                "policy_backend": policy_backend,
            },
        ),
    )
    generated_checker = bool(
        proposal is not None and proposal["checker_need"] is not None
    )
    executed_schema_available = bool(
        candidate.task_contract.get("task_schema_available")
    )
    executed_task_context = candidate.task_contract.get("task_context")
    executed_schema_origin = (
        executed_task_context.get("schema_origin")
        if isinstance(executed_task_context, Mapping)
        else None
    )
    execution_scope = (
        "generated_check_success"
        if generated_checker
        else "official_check_success"
    )
    limitations = ("N=1",)
    if generated_checker:
        limitations += (
            "The generated checker is experimental, not certified "
            "as official-equivalent.",
        )
    elif not executed_schema_available:
        limitations += (
            "No reviewed TaskSchema; the Task context is limited to official "
            "source identity and executed telemetry.",
        )
    elif executed_schema_origin == "runtime_probe":
        limitations += (
            "The TaskContext was derived from a fresh official reset rather "
            "than a reviewed task-specific schema; semantic roles and "
            "thresholds remain unavailable unless directly observed.",
        )
    semantic_ready = bool(
        rollout.metadata.get("semantic_telemetry_ready")
    )
    trusted_tool_evaluation = (
        {
            "schema_version": 1,
            "status": "passed",
            **evaluate_generic_task_rollout_telemetry(
                root,
                child_dir,
                taskgen_manifest,
            ),
        }
        if taskgen_manifest is not None and semantic_ready
        else {
            "schema_version": 1,
            "status": "pending" if semantic_ready else "skipped",
            "outcome_metric": execution_scope,
            "outcome_authority": (
                "llm_generated_python_ast_validated"
                if generated_checker
                else "official_check_success"
            ),
            "episode_count": 0,
            "episodes": [],
        }
    )
    authoritative_rollout = rollout
    checker_result: dict[str, Any] | None = None
    if taskgen_manifest is not None and semantic_ready:
        authoritative_rollout, checker_result = (
            _project_trusted_checker_outcome(
                rollout,
                trusted_tool_evaluation,
                expected_metric=execution_scope,
                policy_backend=policy_backend,
            )
        )
    evidence = runtime.evidence(
        authoritative_rollout,
        EvidenceRequest(
            sub_aspect=str(round_plan["sub_aspect"]),
            hypothesis=query,
            perturbation=(
                str(proposal["semantic_concern"])
                if proposal is not None
                else "unchanged official-scene control"
            ),
            summary=(
                f"{policy_name} completed one RoboTwin rollout; "
                f"{execution_scope}={authoritative_rollout.success}."
            ),
            limitations=limitations,
            metadata={
                "policy_backend": policy_backend,
                "semantic_telemetry_ready": semantic_ready,
                "trusted_checker_result": checker_result,
            },
        ),
    )
    result_path = child_dir / "evaluation" / "_result.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        f"{1.0 if authoritative_rollout.success else 0.0}\n",
        encoding="utf-8",
    )
    scene_validation = (
        deepcopy(taskgen_manifest["scene_validation"])
        if taskgen_manifest is not None
        else {
            "render_success": (
                _artifact_exists(rollout.artifacts.get("initial_frame"))
                or _artifact_exists(rollout.artifacts.get("video"))
            ),
            "rule_check": {
                "passed": True,
                "authority": "official_task_setup_completed",
            },
        }
    )
    task_artifact_summary = (
        deepcopy(taskgen_manifest["task_artifact_summary"])
        if taskgen_manifest is not None
        else {
            "success_official_equivalent": True,
            "success_execution_scope": execution_scope,
        }
    )
    child_manifest = {
        **(deepcopy(taskgen_manifest) if taskgen_manifest is not None else {}),
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "task_name": contract["task_name"],
        "task_module": str(
            candidate.task_contract.get("task_module")
            or contract["task_module"]
        ),
        "generation_kind": (
            str(taskgen_manifest["generation_kind"])
            if taskgen_manifest is not None
            else "official_passthrough"
        ),
        "policy_backend": policy_backend,
        "scene_validation": scene_validation,
        "act_evaluation": {
            "passed": True,
            "actual_seeds": [seed],
            "policy_name": policy_name,
            "outcome_metric": execution_scope,
            "outcome_value": authoritative_rollout.success,
            "episode_latched_success": (
                authoritative_rollout.episode.get(
                    "episode_latched_success"
                )
            ),
            "official_core_predicate_satisfied": (
                authoritative_rollout.episode.get(
                    "official_core_predicate_satisfied"
                )
            ),
        },
        "task_artifact_summary": task_artifact_summary,
        "trusted_tool_evaluation": trusted_tool_evaluation,
        "taskgen_runtime_binding": (
            {
                "validation": deepcopy(dict(candidate.validation)),
                "artifacts": deepcopy(dict(candidate.artifacts)),
            }
            if taskgen_manifest is not None
            else None
        ),
        "runtime_task_context": (
            deepcopy(dict(candidate.task_contract["task_context"]))
            if isinstance(
                candidate.task_contract.get("task_context"), Mapping
            )
            else None
        ),
        "method_runtime": {
            "binding": binding.to_dict(),
            "candidate": candidate.to_dict(),
            "rollout": authoritative_rollout.to_dict(),
            "evidence": evidence.to_dict(),
        },
    }
    manifest_path = child_dir / "manifest.json"
    _write_json(manifest_path, child_manifest)
    method_runtime_path = (
        evaluation_root
        / "execution"
        / str(round_plan["round_id"])
        / "method_runtime.json"
    )
    _write_json(method_runtime_path, child_manifest["method_runtime"])
    return {
        "child_manifest": child_manifest,
        "child_dir": child_dir,
        "manifest_path": manifest_path,
        "method_runtime_path": method_runtime_path,
        "semantic_telemetry_ready": semantic_ready,
        "candidate_id": candidate.candidate_id,
        "evidence_outcome": evidence.outcome,
        "unsupported": False,
    }


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
