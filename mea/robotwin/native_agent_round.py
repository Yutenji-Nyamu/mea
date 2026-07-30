"""Native SmolVLA rollout adapter for the production Plan Agent.

The module owns only the benchmark-specific MethodRuntime boundary.  Planning,
ToolGen, Aggregate, VQA, and answer construction remain in the existing Agent
loop.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
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
from mea.robotwin.runtime import RoboTwinMethodBackend
from mea.robotwin.smolvla_rollout import SmolVLARobotwinRolloutRunner


class NativeAgentRoundError(RuntimeError):
    """Raised when a native policy round exceeds its validated capabilities."""


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
    reason_code: str,
    reason: str,
) -> dict[str, Any]:
    """Persist an unsupported capability as evidence, not a process crash."""

    digest = hashlib.sha256(
        f"{evaluation_id}:{round_plan['round_id']}".encode("utf-8")
    ).hexdigest()[:12]
    run_id = f"native_smolvla_{digest}"
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
        "policy_backend": "smolvla",
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
            "policy_name": "SmolVLA",
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


def execute_smolvla_method_round(
    *,
    repo_root: str | Path,
    evaluation_dir: str | Path,
    evaluation_id: str,
    round_plan: Mapping[str, Any],
    runtime_target: Mapping[str, Any],
    telemetry_profile: str,
    policy_server_port: int,
) -> dict[str, Any]:
    """Run one SmolVLA candidate through the shared MethodRuntime.

    A schema-less task is limited to the unchanged official control.  A
    schema-backed task records semantic telemetry automatically; the caller
    may then pass that episode through the existing Tool/Aggregate stages.
    """

    root = Path(repo_root).expanduser().resolve()
    evaluation_root = Path(evaluation_dir).expanduser().resolve()
    contract = policy_task_binding_from_target(runtime_target)
    if contract["policy"].get("backend") != "smolvla":
        raise NativeAgentRoundError(
            "native SmolVLA round requires a smolvla PolicyTaskBinding"
        )
    if contract["task_name"] != round_plan.get("task_name"):
        raise NativeAgentRoundError(
            "round task differs from the bound SmolVLA task"
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
            "native SmolVLA production rounds currently require exactly one seed"
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
    schema_available = bool(
        contract["task_schema"].get("available", True)
    )
    if proposal is not None and (
        proposal["scene_need"] is not None
        or proposal["checker_need"] is not None
    ):
        return _unsupported_round(
            root=root,
            evaluation_root=evaluation_root,
            evaluation_id=evaluation_id,
            round_plan=round_plan,
            task_name=contract["task_name"],
            proposal=proposal,
            reason_code="smolvla_taskgen_not_connected",
            reason=(
                "The SmolVLA MethodRuntime does not yet connect the shared "
                "generic TaskGen scene/checker backend."
            ),
        )
    if proposal is not None and proposal["vqa_tool_need"] is not None:
        return _unsupported_round(
            root=root,
            evaluation_root=evaluation_root,
            evaluation_id=evaluation_id,
            round_plan=round_plan,
            task_name=contract["task_name"],
            proposal=proposal,
            reason_code="smolvla_vqa_not_connected",
            reason="The SmolVLA MethodRuntime VQA bridge is not connected.",
        )
    if proposal is not None and not schema_available:
        return _unsupported_round(
            root=root,
            evaluation_root=evaluation_root,
            evaluation_id=evaluation_id,
            round_plan=round_plan,
            task_name=contract["task_name"],
            proposal=proposal,
            reason_code="task_schema_unavailable",
            reason=(
                "This task has no TaskSchema, so only the unchanged official "
                "SmolVLA control is executable."
            ),
        )
    if proposal is None and round_plan.get("route") != "official":
        raise NativeAgentRoundError(
            "schema-less SmolVLA execution is restricted to an official round"
        )

    digest = hashlib.sha256(
        f"{evaluation_id}:{round_plan['round_id']}".encode("utf-8")
    ).hexdigest()[:12]
    run_id = f"native_smolvla_{digest}"
    child_dir = root / "mea" / "generated_tasks" / run_id
    rollout_dir = child_dir / "evaluation"
    child_dir.mkdir(parents=True, exist_ok=True)
    backend = RoboTwinMethodBackend(
        repo_root=root,
        rollout_runner=SmolVLARobotwinRolloutRunner(
            port=policy_server_port,
            repo_root=root,
            telemetry_profile=telemetry_profile,
        ),
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
    query = str(round_plan["task_instruction"])
    candidate = (
        backend.official_candidate(
            binding,
            source_query=query,
            candidate_id=str(
                round_plan.get("candidate_id")
                or round_plan.get("template_id")
                or "official_control"
            ),
        )
        if proposal is None
        else runtime.materialize_candidate(
            binding,
            CandidateRequest(
                candidate_id=proposal["candidate_id"],
                source_query=proposal["source_query"],
                proposal_bundle=proposal,
                output_dir=child_dir,
                seed=seed,
            ),
        )
    )
    rollout = runtime.rollout(
        candidate,
        RolloutRequest(
            round_id=str(round_plan["round_id"]),
            seed=seed,
            output_dir=rollout_dir,
            provenance={
                "evaluation_id": evaluation_id,
                "policy_backend": "smolvla",
            },
        ),
    )
    evidence = runtime.evidence(
        rollout,
        EvidenceRequest(
            sub_aspect=str(round_plan["sub_aspect"]),
            hypothesis=query,
            perturbation=(
                str(proposal["semantic_concern"])
                if proposal is not None
                else "unchanged official-scene control"
            ),
            summary=(
                "SmolVLA completed one RoboTwin rollout; official "
                f"check_success={rollout.success}."
            ),
            limitations=(
                ("N=1",)
                if schema_available
                else (
                    "N=1",
                    "No TaskSchema; Rule Tool and Aggregate were not run.",
                )
            ),
            metadata={
                "policy_backend": "smolvla",
                "semantic_telemetry_ready": bool(
                    rollout.metadata.get("semantic_telemetry_ready")
                ),
            },
        ),
    )
    semantic_ready = bool(
        rollout.metadata.get("semantic_telemetry_ready")
    )
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "_result.txt").write_text(
        f"{1.0 if rollout.success else 0.0}\n",
        encoding="utf-8",
    )
    child_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "task_name": contract["task_name"],
        "task_module": contract["task_module"],
        "generation_kind": "official_passthrough",
        "policy_backend": "smolvla",
        "scene_validation": {
            "render_success": bool(
                rollout.artifacts.get("initial_frame")
                and Path(rollout.artifacts["initial_frame"]).is_file()
            ),
            "rule_check": {
                "passed": True,
                "authority": "official_task_setup_completed",
            },
        },
        "act_evaluation": {
            "passed": True,
            "actual_seeds": [seed],
            "policy_name": "SmolVLA",
        },
        "task_artifact_summary": {
            "success_official_equivalent": True,
            "success_execution_scope": "official_check_success",
        },
        "trusted_tool_evaluation": {
            "schema_version": 1,
            "status": "pending" if semantic_ready else "skipped",
            "outcome_metric": "official_check_success",
            "outcome_authority": "official_check_success",
            "episode_count": 0,
            "episodes": [],
        },
        "method_runtime": {
            "binding": binding.to_dict(),
            "candidate": candidate.to_dict(),
            "rollout": rollout.to_dict(),
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


__all__ = [
    "NativeAgentRoundError",
    "execute_smolvla_method_round",
]
