"""Resume a Plan Agent boundary from immutable round artifacts.

The decision-only path starts no simulator, TaskGen runtime, or policy backend.
The explicit pending-round path executes exactly one already-persisted Proposal
and never replays completed rounds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, cast

from mea.history import EvaluationHistoryDB
from mea.plan_agent_application import PlanAgentApplication, update_manifest
from mea.planner import PlanAgent, PlanAgentSession
from mea.round_executor import RoundExecutor


class PlanAgentDecisionResumeError(RuntimeError):
    """Raised when cached artifacts are not at a decision-only boundary."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlanAgentDecisionResumeError(
            f"cannot read {label}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise PlanAgentDecisionResumeError(f"{label} must be a JSON object")
    return value


def _optional_json(path: Path) -> dict[str, Any] | None:
    return _read_json(path, label=path.name) if path.is_file() else None


def _repo_artifact(repo_root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise PlanAgentDecisionResumeError(f"{label} path is missing")
    path = (repo_root / relative).resolve()
    if not path.is_relative_to(repo_root):
        raise PlanAgentDecisionResumeError(f"{label} leaves the repository")
    if not path.is_file():
        raise PlanAgentDecisionResumeError(f"{label} does not exist: {path}")
    return path


def _load_tool_evaluation(
    execution_dir: Path,
    child_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = (
        execution_dir / "planned_tool/tool_execution.json",
        execution_dir / "planned_tool_skipped.json",
    )
    for path in candidates:
        if path.is_file():
            return _read_json(path, label="round Tool evaluation")
    embedded = child_manifest.get("planned_tool_evaluation")
    if isinstance(embedded, Mapping):
        return dict(embedded)
    raise PlanAgentDecisionResumeError(
        f"completed round has no persisted Tool evaluation: {execution_dir}"
    )


def _load_immutable_round_runs(
    repo_root: Path,
    evaluation_dir: Path,
    plan: Mapping[str, Any],
    *,
    completed_rounds: int,
) -> list[dict[str, Any]]:
    rounds = plan.get("rounds")
    if not isinstance(rounds, list) or len(rounds) < completed_rounds:
        raise PlanAgentDecisionResumeError(
            "evaluation plan contains fewer rounds than completed evidence"
        )
    result: list[dict[str, Any]] = []
    for index, raw_plan in enumerate(rounds[:completed_rounds], start=1):
        if not isinstance(raw_plan, Mapping):
            raise PlanAgentDecisionResumeError(
                f"plan round {index} must be an object"
            )
        round_plan = dict(raw_plan)
        round_id = round_plan.get("round_id")
        if not isinstance(round_id, str) or not round_id.strip():
            raise PlanAgentDecisionResumeError(
                f"plan round {index} has no round_id"
            )
        summary = _read_json(
            evaluation_dir / "summary" / f"{round_id}.json",
            label=f"{round_id} summary",
        )
        if summary.get("round_id") != round_id:
            raise PlanAgentDecisionResumeError(
                f"{round_id} summary has a different round_id"
            )
        execution_dir = evaluation_dir / "execution" / round_id
        child_run = _read_json(
            execution_dir / "child_run.json",
            label=f"{round_id} child run",
        )
        child_manifest_path = _repo_artifact(
            repo_root,
            child_run.get("manifest_path"),
            label=f"{round_id} child manifest",
        )
        child_manifest = _read_json(
            child_manifest_path,
            label=f"{round_id} child manifest",
        )
        if child_manifest.get("run_id") != child_run.get("run_id"):
            raise PlanAgentDecisionResumeError(
                f"{round_id} child run id differs from its manifest"
            )
        returncode = child_run.get("returncode")
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise PlanAgentDecisionResumeError(
                f"{round_id} child returncode must be an integer"
            )
        result.append(
            {
                "round_plan": round_plan,
                "child_manifest": child_manifest,
                "child_dir": child_manifest_path.parent,
                "round_summary": summary,
                "tool_evaluation": _load_tool_evaluation(
                    execution_dir,
                    child_manifest,
                ),
                "returncode": returncode,
            }
        )
    return result


def _history_database(
    repo_root: Path,
    manifest: Mapping[str, Any],
) -> EvaluationHistoryDB | None:
    if manifest.get("history_retrieval_status") == "disabled":
        return None
    relative = manifest.get("history_database")
    if not isinstance(relative, str) or not relative.strip():
        return None
    return EvaluationHistoryDB(repo_root / relative, repo_root=repo_root)


def _registry_path(repo_root: Path, value: Any) -> Path | None:
    return repo_root / value if isinstance(value, str) and value.strip() else None


def _latest_runtime_capabilities(
    evaluation_dir: Path,
    *,
    completed_rounds: int,
) -> dict[str, Any]:
    """Restore the exact capability card persisted at this boundary."""

    path = (
        evaluation_dir
        / "plan/plan_agent_steps"
        / f"after_round_{completed_rounds:02d}"
        / "runtime_capabilities.json"
    )
    return _read_json(
        path,
        label=f"round {completed_rounds} runtime capabilities",
    )


def _rebuild_application(
    *,
    root: Path,
    evaluation_dir: Path,
    evaluation_id: str,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    completed_rounds: int,
    provider: Any,
    models: Mapping[str, str],
    round_executor: RoundExecutor,
    gpu: int,
    max_reflections: int,
    policy_server_port: int,
) -> tuple[PlanAgentApplication, list[dict[str, Any]]]:
    """Rebuild the production application without executing prior rounds."""

    snapshot = _read_json(
        evaluation_dir / "plan/bound_task_session.json",
        label="Plan Agent session snapshot",
    )
    capabilities = _latest_runtime_capabilities(
        evaluation_dir,
        completed_rounds=completed_rounds,
    )
    user_request = manifest.get("user_request")
    if not isinstance(user_request, str) or not user_request.strip():
        raise PlanAgentDecisionResumeError("manifest user_request is missing")
    target = snapshot.get("target")
    contract = plan.get("query_contract") or snapshot.get("query_contract")
    if not isinstance(target, Mapping) or not isinstance(contract, Mapping):
        raise PlanAgentDecisionResumeError(
            "Plan Agent target or QueryContract is missing"
        )
    control_required = contract.get("control_requirement") == "required"
    control_round = snapshot.get("control_round") if control_required else None
    session = PlanAgentSession(
        user_request,
        target,
        query_contract=contract,
        require_control_anchor=control_required,
        control_round=(
            control_round if isinstance(control_round, Mapping) else None
        ),
    )
    normalized_plan = session.normalize_plan(plan)
    round_runs = _load_immutable_round_runs(
        root,
        evaluation_dir,
        normalized_plan,
        completed_rounds=completed_rounds,
    )
    planner_model = models.get("planner")
    feedback_model = models.get("feedback") or models.get("answer")
    if not planner_model or not feedback_model:
        raise PlanAgentDecisionResumeError(
            "models must provide planner and feedback"
        )
    history_retrieval = _optional_json(
        evaluation_dir / "plan/history_retrieval.json"
    ) or {}
    history_candidates = history_retrieval.get("matches")
    if not isinstance(history_candidates, list):
        history_candidates = history_retrieval.get("candidates")
    app = PlanAgentApplication(
        repo_root=root,
        evaluation_dir=evaluation_dir,
        evaluation_id=evaluation_id,
        user_request=user_request,
        plan=normalized_plan,
        session=session,
        agent=PlanAgent(provider, model=planner_model, repo_root=root),
        capabilities=capabilities,
        provider=provider,
        round_executor=round_executor,
        models={**dict(models), "feedback": feedback_model},
        gpu=gpu,
        max_reflections=max_reflections,
        telemetry_profile=str(
            manifest.get("telemetry_profile") or "balanced_v1"
        ),
        policy_backend=str(manifest.get("policy_backend") or "smolvla"),
        runtime_target=target,
        policy_server_port=policy_server_port,
        reviewed_tool_registry=_registry_path(
            root, manifest.get("reviewed_tool_registry")
        ),
        reviewed_vqa_registry=_registry_path(
            root, manifest.get("reviewed_vqa_registry")
        ),
        max_agent_rounds=manifest.get("max_agent_rounds"),
        global_route_result=_optional_json(
            evaluation_dir / "plan/global_query_route.json"
        ),
        free_concern_bundle=_optional_json(
            evaluation_dir / "plan/query_interpretation.json"
        ),
        open_task_resolution=_optional_json(
            evaluation_dir / "plan/open_task_resolution.json"
        ),
        concern_candidate_resolution=_optional_json(
            evaluation_dir / "plan/concern_candidate_resolution.json"
        ),
        history_database=_history_database(root, manifest),
        history_retrieval=history_retrieval,
        history_context_count=(
            len(history_candidates)
            if isinstance(history_candidates, list)
            else 0
        ),
        history_disabled=(
            manifest.get("history_retrieval_status") == "disabled"
        ),
        cli_candidate_hint_used=(
            manifest.get("bound_requested_aspect_ids") is not None
            or manifest.get("cli_candidate_hint_used") is True
        ),
    )
    return app, round_runs


def resume_plan_agent_decision(
    repo_root: str | Path,
    evaluation_id: str,
    *,
    provider: Any,
    models: Mapping[str, str],
) -> dict[str, Any]:
    """Retry only a transient post-round Plan Agent provider decision.

    ``models`` must provide ``planner`` and ``feedback``.  Existing round
    artifacts are consumed as immutable evidence.  A continuing Proposal is
    persisted but deliberately not executed.
    """

    root = Path(repo_root).expanduser().resolve()
    evaluation_dir = root / "mea" / "evaluation_runs" / evaluation_id
    manifest = _read_json(evaluation_dir / "manifest.json", label="manifest")
    if manifest.get("evaluation_id") != evaluation_id:
        raise PlanAgentDecisionResumeError("manifest evaluation_id mismatch")
    if manifest.get("lifecycle_status") == "completed":
        raise PlanAgentDecisionResumeError("evaluation is already completed")
    match = re.fullmatch(
        r"plan_agent_decision_after_round_(\d+)",
        str(manifest.get("failure_stage") or ""),
    )
    if match is None:
        raise PlanAgentDecisionResumeError(
            "evaluation did not fail at a Plan Agent decision boundary"
        )
    completed_rounds = int(match.group(1))
    if manifest.get("completed_rounds") not in {None, completed_rounds}:
        raise PlanAgentDecisionResumeError(
            "manifest completed_rounds conflicts with failure_stage"
        )
    if (
        evaluation_dir
        / "plan"
        / f"decision_after_round_{completed_rounds}.json"
    ).exists():
        raise PlanAgentDecisionResumeError(
            "the failed boundary already has a persisted Plan Agent decision"
        )

    step_dir = (
        evaluation_dir
        / "plan/plan_agent_steps"
        / f"after_round_{completed_rounds:02d}"
    )
    if not (step_dir / "runtime_capabilities.json").is_file():
        raise PlanAgentDecisionResumeError(
            "decision retry requires the capability card persisted before "
            "the failed provider call"
        )
    if (step_dir / "semantic_proposal_bundle.json").exists() or (
        step_dir / "bound_semantic_step.json"
    ).exists():
        raise PlanAgentDecisionResumeError(
            "decision retry is only valid before a semantic Proposal was "
            "persisted"
        )
    retry = manifest.get("plan_agent_cached_retry")
    retry_attempt_count = (
        retry.get("attempt_count") if isinstance(retry, Mapping) else None
    )
    if (
        isinstance(retry, Mapping)
        and retry.get("after_round") == completed_rounds
        and isinstance(retry_attempt_count, int)
        and not isinstance(retry_attempt_count, bool)
        and retry_attempt_count >= 1
    ):
        raise PlanAgentDecisionResumeError(
            "the one bounded cached decision retry was already used at this "
            "round boundary"
        )

    plan = _read_json(
        evaluation_dir / "plan/evaluation_plan.json",
        label="evaluation plan",
    )
    if not isinstance(plan.get("rounds"), list) or len(
        plan["rounds"]
    ) != completed_rounds:
        raise PlanAgentDecisionResumeError(
            "decision-only resume requires no pending execution round"
        )
    app, round_runs = _rebuild_application(
        root=root,
        evaluation_dir=evaluation_dir,
        evaluation_id=evaluation_id,
        manifest=manifest,
        plan=plan,
        completed_rounds=completed_rounds,
        provider=provider,
        round_executor=cast(RoundExecutor, None),
        models=models,
        gpu=0,
        max_reflections=1,
        policy_server_port=0,
    )
    original_failure = manifest.get("failure")
    update_manifest(
        evaluation_dir,
        status=f"resuming_plan_agent_decision_after_round_{completed_rounds}",
        lifecycle_status="resuming_plan_agent_decision",
        plan_agent_decision_resume={
            "status": "running",
            "after_round": completed_rounds,
            "rollouts_executed": 0,
            "source_round_artifacts": [
                f"summary/{item['round_plan']['round_id']}.json"
                for item in round_runs
            ],
            "original_failure": original_failure,
        },
        plan_agent_cached_retry={
            "after_round": completed_rounds,
            "attempt_count": 1,
            "status": "running",
            "rollouts_executed": 0,
            "original_failure": original_failure,
        },
    )
    try:
        result = app.resume_decision(round_runs=round_runs)
    except Exception:
        update_manifest(
            evaluation_dir,
            plan_agent_cached_retry={
                "after_round": completed_rounds,
                "attempt_count": 1,
                "status": "failed",
                "rollouts_executed": 0,
                "original_failure": original_failure,
            },
        )
        raise
    update_manifest(
        evaluation_dir,
        plan_agent_cached_retry={
            "after_round": completed_rounds,
            "attempt_count": 1,
            "status": "completed",
            "rollouts_executed": 0,
            "original_failure": original_failure,
        },
    )
    return result


def continue_pending_plan_agent_round(
    repo_root: str | Path,
    evaluation_id: str,
    *,
    provider: Any,
    models: Mapping[str, str],
    policy_server_port: int,
    gpu: int = 0,
    max_reflections: int = 1,
) -> dict[str, Any]:
    """Execute exactly one saved pending Proposal, then resume planning."""

    from mea.robotwin.production_round_executor import (
        build_production_round_executor,
    )

    root = Path(repo_root).expanduser().resolve()
    evaluation_dir = root / "mea" / "evaluation_runs" / evaluation_id
    manifest = _read_json(evaluation_dir / "manifest.json", label="manifest")
    if manifest.get("evaluation_id") != evaluation_id:
        raise PlanAgentDecisionResumeError("manifest evaluation_id mismatch")
    continuation = manifest.get("plan_agent_decision_resume")
    if not (
        manifest.get("lifecycle_status")
        == "awaiting_explicit_round_execution"
        and isinstance(continuation, Mapping)
        and continuation.get("status") == "next_proposal_persisted"
        and continuation.get("action") == "continue"
    ):
        raise PlanAgentDecisionResumeError(
            "evaluation has no explicitly pending Plan Agent Proposal"
        )
    completed_rounds = manifest.get("completed_rounds")
    if (
        isinstance(completed_rounds, bool)
        or not isinstance(completed_rounds, int)
        or completed_rounds < 1
    ):
        raise PlanAgentDecisionResumeError(
            "manifest has no valid completed-round count"
        )
    if (
        isinstance(policy_server_port, bool)
        or not isinstance(policy_server_port, int)
        or not 1 <= policy_server_port <= 65535
    ):
        raise PlanAgentDecisionResumeError(
            "policy_server_port must be in [1, 65535]"
        )
    plan = _read_json(
        evaluation_dir / "plan/evaluation_plan.json",
        label="evaluation plan",
    )
    rounds = plan.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != completed_rounds + 1:
        raise PlanAgentDecisionResumeError(
            "explicit continuation requires exactly one saved pending round"
        )
    if not (
        evaluation_dir
        / "plan"
        / f"decision_after_round_{completed_rounds}.json"
    ).is_file():
        raise PlanAgentDecisionResumeError(
            "pending Proposal has no preceding Plan Agent decision artifact"
        )
    pending = rounds[-1]
    if not isinstance(pending, Mapping) or not isinstance(
        pending.get("round_id"), str
    ):
        raise PlanAgentDecisionResumeError("pending round is invalid")
    pending_round_id = pending["round_id"]
    if (
        (evaluation_dir / "summary" / f"{pending_round_id}.json").exists()
        or (
            evaluation_dir / "execution" / pending_round_id / "child_run.json"
        ).exists()
    ):
        raise PlanAgentDecisionResumeError(
            "pending round already has execution evidence"
        )

    app, round_runs = _rebuild_application(
        root=root,
        evaluation_dir=evaluation_dir,
        evaluation_id=evaluation_id,
        manifest=manifest,
        plan=plan,
        completed_rounds=completed_rounds,
        provider=provider,
        models=models,
        round_executor=build_production_round_executor(),
        gpu=gpu,
        max_reflections=max_reflections,
        policy_server_port=policy_server_port,
    )
    update_manifest(
        evaluation_dir,
        status=f"executing_explicit_{pending_round_id}",
        lifecycle_status="executing_pending_plan_agent_round",
        pending_round_continuation={
            "status": "running",
            "pending_round_id": pending_round_id,
            "prior_rounds_replayed": 0,
        },
    )
    return app.execute_pending_round(round_runs=round_runs)


__all__ = [
    "PlanAgentDecisionResumeError",
    "continue_pending_plan_agent_round",
    "resume_plan_agent_decision",
]
