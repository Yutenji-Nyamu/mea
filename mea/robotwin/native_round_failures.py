"""Typed zero-rollout and unsupported round evidence assembly."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.taskgen.attempts import TERMINAL, task_generation_recovery_action

from .native_round_contracts import (
    NativeAgentRoundError,
    build_native_run_id as _build_native_run_id,
    canonical_sha256 as _canonical_sha256,
    is_zero_rollout_count as _is_zero_rollout_count,
    write_json as _write_json,
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
        and _is_zero_rollout_count(policy_execution.get("rollouts_started"))
        and _is_zero_rollout_count(policy_execution.get("sample_count"))
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


def _taskgen_materialization_failure_round(
    *,
    evaluation_root: Path,
    round_plan: Mapping[str, Any],
    child_dir: Path,
    proposal: Mapping[str, Any],
    policy_backend: str,
    policy_name: str,
) -> dict[str, Any]:
    """Turn a bounded pre-policy TaskGen failure into planning evidence."""

    manifest_path = child_dir / "manifest.json"
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeAgentRoundError(
            "failed TaskGen manifest is unavailable"
        ) from exc
    if not isinstance(raw_manifest, Mapping):
        raise NativeAgentRoundError("failed TaskGen manifest must be an object")
    child_manifest = deepcopy(dict(raw_manifest))
    failure = child_manifest.get("failure")
    policy_execution = child_manifest.get("policy_execution")
    if not (
        child_manifest.get("status") == "failed"
        and isinstance(failure, Mapping)
        and failure.get("type") == "GenericTaskGenError"
        and isinstance(policy_execution, Mapping)
        and policy_execution.get("started") is False
        and _is_zero_rollout_count(policy_execution.get("rollouts_started"))
        and _is_zero_rollout_count(policy_execution.get("sample_count"))
        and isinstance(child_manifest.get("task_generation_attempts"), str)
        and bool(str(child_manifest["task_generation_attempts"]).strip())
    ):
        raise NativeAgentRoundError(
            "TaskGen failure cannot be projected as pre-policy evidence"
        )

    attempt_summary_path = (
        child_dir / "validation/task_generation_attempt_summary.json"
    )
    attempt_summary: dict[str, Any] | None = None
    try:
        candidate_summary = json.loads(
            attempt_summary_path.read_text(encoding="utf-8")
        )
        if isinstance(candidate_summary, Mapping):
            attempt_summary = dict(candidate_summary)
    except (OSError, json.JSONDecodeError):
        pass
    if (
        attempt_summary is None
        or attempt_summary.get("recovery_scope")
        != "task_generation_before_policy"
        or attempt_summary.get("status") != "failed"
        or attempt_summary.get("proposal_identity_sha256")
        != _canonical_sha256(proposal)
    ):
        raise NativeAgentRoundError(
            "TaskGen failure lacks a bounded failed attempt summary"
        )
    attempts = attempt_summary.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise NativeAgentRoundError("TaskGen attempt summary has no attempts")
    for attempt in attempts:
        attempt_runtime = (
            attempt.get("runtime") if isinstance(attempt, Mapping) else None
        )
        if (
            not isinstance(attempt, Mapping)
            or not str(attempt.get("status") or "").startswith("failed_")
            or not isinstance(attempt_runtime, Mapping)
            or not _is_zero_rollout_count(
                attempt_runtime.get("act_rollouts_started")
            )
        ):
            raise NativeAgentRoundError(
                "TaskGen attempt is not a failed zero-rollout record"
            )
    last_attempt = attempts[-1]
    terminal_failure = (
        last_attempt.get("failure")
        if isinstance(last_attempt, Mapping)
        else None
    )
    runtime = attempt_summary.get("runtime")
    if (
        not isinstance(terminal_failure, Mapping)
        or last_attempt.get("status") != "failed_terminal"
        or terminal_failure.get("type") != "TaskGenerationStageError"
        or not isinstance(runtime, Mapping)
        or not _is_zero_rollout_count(runtime.get("act_rollouts_started"))
    ):
        raise NativeAgentRoundError(
            "TaskGen attempt summary is not a typed pre-policy failure"
        )
    failure_stage = str(terminal_failure.get("stage") or "")
    failure_kind = str(terminal_failure.get("failure_kind") or "")
    planning_failures = {
        ("scene_codegen", "invalid_candidate"),
        ("success_spec", "invalid_spec"),
    }
    if (
        (failure_stage, failure_kind) not in planning_failures
        or task_generation_recovery_action(failure_stage, failure_kind)
        == TERMINAL
    ):
        raise NativeAgentRoundError(
            "terminal TaskGen system failure cannot become planning evidence"
        )
    candidate_id = str(proposal["candidate_id"])
    planning_observation = {
        "schema_version": 1,
        "kind": "taskgen_materialization_failed",
        "candidate_id": candidate_id,
        "sub_aspect": str(
            round_plan.get("sub_aspect")
            or proposal.get("semantic_concern")
            or "taskgen_materialization"
        ),
        "failure_stage": failure_stage,
        "reason_code": f"taskgen_{failure_stage}_{failure_kind}",
        "diagnosis": str(
            terminal_failure.get("message")
            or failure.get("message")
            or "bounded TaskGen validation failed before policy execution"
        ),
        "policy_rollouts_started": 0,
        "policy_sample_count": 0,
        "taskgen_attempt_summary": child_manifest.get(
            "task_generation_attempts"
        ),
    }
    bounded_repair_evidence: list[dict[str, str]] = []
    for attempt in attempts[:-1]:
        attempt_failure = (
            attempt.get("failure") if isinstance(attempt, Mapping) else None
        )
        if not isinstance(attempt_failure, Mapping):
            continue
        message = str(attempt_failure.get("message") or "").strip()
        if not message:
            continue
        bounded_repair_evidence.append(
            {
                "stage": str(attempt_failure.get("stage") or "unknown"),
                "failure_kind": str(
                    attempt_failure.get("failure_kind") or "unknown"
                ),
                "message": message[:2400],
            }
        )
    if bounded_repair_evidence:
        planning_observation["bounded_repair_evidence"] = (
            bounded_repair_evidence[-1:]
        )
    method_runtime = {
        "schema_version": 1,
        "status": "taskgen_materialization_failed",
        "candidate_id": candidate_id,
        "proposal": deepcopy(dict(proposal)),
        "planning_observation": planning_observation,
    }
    child_manifest.update(
        {
            "status": "taskgen_materialization_failed",
            "taskgen_materialization_failed": planning_observation,
            "policy_backend": policy_backend,
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
        "evidence_outcome": "taskgen_materialization_failed",
        "taskgen_materialization_failed": True,
        "planning_observation": planning_observation,
    }
