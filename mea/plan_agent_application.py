"""Production Plan Agent application lifecycle.

This module owns the paper-method loop after route and runtime binding:

``round execution -> evidence -> Plan Agent step -> stop/continue -> answer``.

The command-line entry point is responsible only for resolving CLI compatibility,
binding the executable policy/task, and constructing this application.  Legacy
fixed/catalog protocols deliberately remain outside this production object.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from mea.history import EvaluationHistoryDB
from mea.planner import (
    PlanAgent,
    PlanAgentSession,
)
from mea.providers import OpenAICompatibleProvider
from mea.round_executor import (
    RoundExecutionRequest,
    RoundExecutionResult,
    RoundExecutor,
)

from mea.plan_agent_finalization import PlanAgentFinalizationMixin
from mea.plan_agent_persistence import (
    apply_external_hard_round_cap,
    refresh_plan_agent_capabilities_from_runtime_context,
    update_manifest,
)
from mea.plan_agent_runtime_decisions import PlanAgentRuntimeDecisionMixin


@dataclass
class PlanAgentApplication(
    PlanAgentRuntimeDecisionMixin,
    PlanAgentFinalizationMixin,
):
    """Execute one bound production Plan Agent evaluation."""

    repo_root: Path
    evaluation_dir: Path
    evaluation_id: str
    user_request: str
    plan: dict[str, Any]
    session: PlanAgentSession
    agent: PlanAgent
    capabilities: dict[str, Any]
    provider: OpenAICompatibleProvider
    round_executor: RoundExecutor
    models: Mapping[str, str]
    gpu: int
    max_reflections: int
    telemetry_profile: str
    policy_backend: str
    runtime_target: Mapping[str, Any] | None
    policy_server_port: int
    reviewed_tool_registry: Path | None = None
    reviewed_vqa_registry: Path | None = None
    max_agent_rounds: int | None = None
    global_route_result: Mapping[str, Any] | None = None
    free_concern_bundle: Mapping[str, Any] | None = None
    open_task_resolution: Mapping[str, Any] | None = None
    concern_candidate_resolution: Mapping[str, Any] | None = None
    history_database: EvaluationHistoryDB | None = None
    history_retrieval: Mapping[str, Any] | None = None
    history_context_count: int = 0
    history_disabled: bool = False
    cli_candidate_hint_used: bool = False

    def _execute_round_plan(
        self,
        round_plan: dict[str, Any],
    ) -> RoundExecutionResult:
        """Execute one already-materialized Proposal through the runtime."""

        return self.round_executor.execute(
            RoundExecutionRequest(
                repo_root=self.repo_root,
                evaluation_dir=self.evaluation_dir,
                evaluation_id=self.evaluation_id,
                round_plan=round_plan,
                text_model=self.models["taskgen"],
                vision_model=self.models["vision"],
                gpu=self.gpu,
                max_reflections=self.max_reflections,
                provider=self.provider,
                toolgen_model=self.models["toolgen"],
                telemetry_profile=self.telemetry_profile,
                reviewed_tool_registry=self.reviewed_tool_registry,
                reviewed_vqa_registry=self.reviewed_vqa_registry,
                policy_backend=self.policy_backend,
                runtime_target=self.runtime_target,
                policy_server_port=self.policy_server_port,
            )
        )

    def execute_pending_round(
        self,
        *,
        round_runs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute exactly one persisted pending Proposal, then decide.

        Completed rounds are supplied from immutable artifacts and are never
        replayed.  If the post-round Plan Agent continues, the next Proposal is
        persisted for another explicit command rather than executed here.
        """

        completed_rounds = len(round_runs)
        if len(self.plan["rounds"]) != completed_rounds + 1:
            raise RuntimeError(
                "explicit continuation requires exactly one pending round"
            )
        pending_plan = self.plan["rounds"][-1]
        pending_round_id = pending_plan["round_id"]
        try:
            round_result = self._execute_round_plan(pending_plan)
        except Exception as exc:
            update_manifest(
                self.evaluation_dir,
                status="failed",
                lifecycle_status="failed",
                failure_stage=f"{pending_round_id}_execution",
                completed_rounds=completed_rounds,
                active_child_run_id=None,
                execution_finished_at=datetime.now().astimezone().isoformat(),
                failure={"type": type(exc).__name__, "message": str(exc)},
                pending_round_continuation={
                    "status": "failed",
                    "pending_round_id": pending_round_id,
                    "prior_rounds_replayed": 0,
                },
            )
            raise
        round_runs.append(
            {
                "round_plan": pending_plan,
                "child_manifest": round_result.child_manifest,
                "child_dir": round_result.child_dir,
                "round_summary": round_result.round_summary,
                "tool_evaluation": round_result.tool_evaluation,
                "returncode": round_result.returncode,
            }
        )
        result = self.resume_decision(round_runs=round_runs)
        continuation = {
            "status": "completed",
            "executed_round_id": pending_round_id,
            "rounds_executed": 1,
            "prior_rounds_replayed": 0,
            "automatic_followup_round_execution": False,
        }
        update_manifest(
            self.evaluation_dir,
            pending_round_continuation=continuation,
        )
        return {**result, "pending_round_continuation": continuation}

    def resume_decision(
        self,
        *,
        round_runs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Resume only the Plan Agent decision after immutable evidence.

        This boundary deliberately does not call ``RoundExecutor``.  A stop
        decision completes the normal QueryContract/Answer finalization; a
        continue decision persists its next Proposal and returns control to a
        later explicit execution command.
        """

        executed_rounds = len(round_runs)
        if executed_rounds < 1 or executed_rounds != len(self.plan["rounds"]):
            raise RuntimeError(
                "Plan Agent decision resume requires one immutable artifact "
                "set for every currently planned round"
            )
        active_failure_stage = (
            f"plan_agent_decision_after_round_{executed_rounds}"
        )
        try:
            runtime_state = self._observe(
                round_runs,
                executed_rounds=executed_rounds,
            )
            round_plan = round_runs[-1]["round_plan"]
            query_answer: dict[str, Any] | None = None
            self.plan, decision, query_answer = self._decide_next_step(
                plan=self.plan,
                round_plan=round_plan,
                round_runs=round_runs,
                runtime_state=runtime_state,
                executed_rounds=executed_rounds,
            )

            if decision["action"] == "stop":
                active_failure_stage = "evaluation_aggregation"
                result = self._finalize(
                    plan=self.plan,
                    round_runs=round_runs,
                    runtime_state=runtime_state,
                    query_answer=query_answer,
                    executed_rounds=executed_rounds,
                )
                update_manifest(
                    self.evaluation_dir,
                    failure=None,
                    failure_stage=None,
                    plan_agent_decision_resume={
                        "status": "completed",
                        "after_round": executed_rounds,
                        "action": "stop",
                        "rollouts_executed": 0,
                    },
                )
                return {
                    **result,
                    "decision_resume": {
                        "action": "stop",
                        "rollouts_executed": 0,
                    },
                }

            update_manifest(
                self.evaluation_dir,
                status=self.plan["planning_state"],
                lifecycle_status="awaiting_explicit_round_execution",
                failure=None,
                failure_stage=None,
                completed_rounds=executed_rounds,
                plan_agent_decision_resume={
                    "status": "next_proposal_persisted",
                    "after_round": executed_rounds,
                    "action": "continue",
                    "rollouts_executed": 0,
                    "automatic_round_execution": False,
                },
            )
            return {
                "evaluation_id": self.evaluation_id,
                "status": self.plan["planning_state"],
                "lifecycle_status": "awaiting_explicit_round_execution",
                "decision_resume": {
                    "action": "continue",
                    "rollouts_executed": 0,
                    "automatic_round_execution": False,
                    "next_round": decision.get("next_round"),
                },
            }
        except Exception as exc:
            update_manifest(
                self.evaluation_dir,
                status="failed",
                lifecycle_status="failed",
                failure_stage=active_failure_stage,
                completed_rounds=executed_rounds,
                active_child_run_id=None,
                execution_finished_at=datetime.now().astimezone().isoformat(),
                failure={"type": type(exc).__name__, "message": str(exc)},
                plan_agent_decision_resume={
                    "status": "failed",
                    "after_round": executed_rounds,
                    "rollouts_executed": 0,
                },
            )
            raise

    def run(self) -> dict[str, Any]:
        """Run the production evidence-conditioned method application."""

        round_runs: list[dict[str, Any]] = []
        runtime_state: dict[str, Any] | None = None
        query_answer: dict[str, Any] | None = None
        active_failure_stage = "round_execution"
        try:
            executed_rounds = 0
            while executed_rounds < len(self.plan["rounds"]):
                active_failure_stage = (
                    f"round_{executed_rounds + 1}_execution"
                )
                round_plan = self.plan["rounds"][executed_rounds]
                round_result = self._execute_round_plan(round_plan)
                round_runs.append(
                    {
                        "round_plan": round_plan,
                        "child_manifest": round_result.child_manifest,
                        "child_dir": round_result.child_dir,
                        "round_summary": round_result.round_summary,
                        "tool_evaluation": round_result.tool_evaluation,
                        "returncode": round_result.returncode,
                    }
                )
                executed_rounds += 1
                active_failure_stage = (
                    f"plan_agent_evidence_after_round_{executed_rounds}"
                )
                runtime_state = self._observe(
                    round_runs,
                    executed_rounds=executed_rounds,
                )
                active_failure_stage = (
                    f"plan_agent_decision_after_round_{executed_rounds}"
                )
                self.plan, decision, step_answer = self._decide_next_step(
                    plan=self.plan,
                    round_plan=round_plan,
                    round_runs=round_runs,
                    runtime_state=runtime_state,
                    executed_rounds=executed_rounds,
                )
                if step_answer is not None:
                    query_answer = step_answer
                if decision["action"] == "stop":
                    break

            if runtime_state is None:
                raise RuntimeError(
                    "Plan Agent application completed no evidence round"
                )
            active_failure_stage = "evaluation_aggregation"
            return self._finalize(
                plan=self.plan,
                round_runs=round_runs,
                runtime_state=runtime_state,
                query_answer=query_answer,
                executed_rounds=executed_rounds,
            )
        except Exception as exc:
            update_manifest(
                self.evaluation_dir,
                status="failed",
                lifecycle_status="failed",
                failure_stage=active_failure_stage,
                completed_rounds=len(round_runs),
                active_child_run_id=None,
                execution_finished_at=datetime.now().astimezone().isoformat(),
                failure={"type": type(exc).__name__, "message": str(exc)},
            )
            raise


__all__ = [
    "PlanAgentApplication",
    "apply_external_hard_round_cap",
    "refresh_plan_agent_capabilities_from_runtime_context",
    "update_manifest",
]
