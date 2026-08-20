"""Completed-round manifest and compact MethodRuntime evidence assembly."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .native_round_contracts import (
    NativeRoundEvaluation,
    NativeRoundPreparation,
    write_json,
)
from .native_round_evaluation import _artifact_exists


def assemble_robotwin_method_round(
    prepared: NativeRoundPreparation,
    evaluated: NativeRoundEvaluation,
    *,
    round_plan: Mapping[str, Any],
    policy_backend: str,
    policy_name: str,
) -> dict[str, Any]:
    evaluation_root = prepared.evaluation_root
    contract = prepared.contract
    seeds = prepared.seeds
    run_id = prepared.run_id
    child_dir = prepared.child_dir
    candidate = prepared.candidate
    taskgen_manifest = prepared.taskgen_manifest
    authoritative_rollouts = evaluated.authoritative_rollouts
    trusted_tool_evaluation = evaluated.trusted_tool_evaluation
    evidence = evaluated.evidence
    execution_scope = evaluated.execution_scope
    semantic_ready = evaluated.semantic_ready
    result_path = child_dir / "evaluation" / "_result.txt"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        f"{evaluated.success_rate}\n",
        encoding="utf-8",
    )
    scene_validation = (
        deepcopy(taskgen_manifest["scene_validation"])
        if taskgen_manifest is not None
        else {
            "render_success": (
                all(
                    _artifact_exists(rollout.artifacts.get("initial_frame"))
                    or _artifact_exists(rollout.artifacts.get("video"))
                    for rollout in authoritative_rollouts
                )
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
        "materialization_anchor_seed": prepared.materialization_anchor_seed,
        "policy_backend": policy_backend,
        "scene_validation": scene_validation,
        "act_evaluation": {
            "passed": True,
            "actual_seeds": list(seeds),
            "policy_name": policy_name,
            "outcome_metric": execution_scope,
            "outcome_value": evaluated.success_rate,
            "success_count": sum(
                1 for rollout in authoritative_rollouts if rollout.success
            ),
            "trial_count": len(authoritative_rollouts),
            "episode_results": [
                {
                    "seed": rollout.seed,
                    "outcome_value": rollout.success,
                    "episode_latched_success": rollout.episode.get(
                        "episode_latched_success"
                    ),
                    "official_core_predicate_satisfied": rollout.episode.get(
                        "official_core_predicate_satisfied"
                    ),
                }
                for rollout in authoritative_rollouts
            ],
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
            "binding": prepared.binding.to_dict(),
            "candidate": candidate.to_dict(),
            "rollouts": [
                rollout.to_dict() for rollout in authoritative_rollouts
            ],
            "evidence": evidence.to_dict(),
        },
    }
    manifest_path = child_dir / "manifest.json"
    write_json(manifest_path, child_manifest)
    method_runtime_path = (
        evaluation_root
        / "execution"
        / str(round_plan["round_id"])
        / "method_runtime.json"
    )
    write_json(method_runtime_path, child_manifest["method_runtime"])
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
