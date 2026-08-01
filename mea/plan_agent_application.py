"""Production Plan Agent application lifecycle.

This module owns the paper-method loop after route and runtime binding:

``round execution -> evidence -> Plan Agent step -> stop/continue -> answer``.

The command-line entry point is responsible only for resolving CLI compatibility,
binding the executable policy/task, and constructing this application.  Legacy
fixed/catalog protocols deliberately remain outside this production object.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from mea.agent_acceptance import build_compact_flagship_acceptance
from mea.agent_evidence import (
    build_evidence_bundle,
    compact_aggregate_result,
)
from mea.feedback import (
    PlanAgentFinalSummary,
    render_evaluation_report,
    write_evidence_report,
)
from mea.history import EvaluationHistoryDB
from mea.plan_artifacts import (
    PLAN_AGENT_SESSION,
    PLAN_AGENT_STEPS,
    PROPOSAL_MATERIALIZATION,
)
from mea.planner import (
    PlanAgent,
    PlanAgentSession,
    render_query_answer,
    validate_open_query_capabilities,
)
from mea.providers import OpenAICompatibleProvider
from mea.round_evidence import aggregate_evaluation_results
from mea.round_executor import RoundExecutionRequest, RoundExecutor


MaterializeRound = Callable[..., tuple[dict[str, Any], dict[str, Any]]]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_manifest(evaluation_dir: Path, **updates: Any) -> dict[str, Any]:
    """Update the evaluation manifest owned by the application lifecycle."""

    path = evaluation_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    _write_json(path, manifest)
    return manifest


def refresh_plan_agent_capabilities_from_runtime_context(
    capabilities: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote one backend-validated TaskContext into the next Plan step.

    Shared policies may start with source-only task identity.  The unchanged
    control establishes actor and telemetry authority before inference; expose
    that newly observed execution surface to the next evidence-conditioned
    decision without adding a task-specific capability menu.
    """

    current = deepcopy(dict(capabilities))
    raw_context = child_manifest.get("runtime_task_context")
    if raw_context is None:
        return current
    if not isinstance(raw_context, Mapping):
        raise ValueError("runtime_task_context must be an object")
    context = deepcopy(dict(raw_context))
    if (
        context.get("schema_version") != 1
        or context.get("taskgen_ready") is not True
        or context.get("schema_origin")
        not in {"runtime_probe", "reviewed_task_schema"}
    ):
        raise ValueError(
            "runtime_task_context is not a validated execution context"
        )
    task_schema = context.get("task_schema")
    if not isinstance(task_schema, Mapping):
        raise ValueError("runtime_task_context has no task schema")
    simulator = current.get("simulator_card")
    if not isinstance(simulator, Mapping):
        raise ValueError("Plan Agent capabilities have no simulator card")
    task_name = context.get("task_name")
    if (
        not isinstance(task_name, str)
        or task_name != simulator.get("task_name")
        or task_schema.get("task_name") != task_name
    ):
        raise ValueError(
            "runtime TaskContext task differs from the Plan Agent binding"
        )

    refreshed_simulator = deepcopy(dict(simulator))
    for field in (
        "physics_timestep_seconds",
        "action_dimension",
        "tracked_actors",
        "semantic_fields",
        "semantic_roles",
        "success_contract",
        "telemetry_observables",
    ):
        if field in task_schema:
            refreshed_simulator[field] = deepcopy(task_schema[field])
    refreshed_simulator["task_context_authority"] = {
        "schema_origin": context["schema_origin"],
        "official_source_sha256": context.get("official_source_sha256"),
        "authority": deepcopy(dict(context.get("authority") or {})),
    }
    current["simulator_card"] = refreshed_simulator

    generation = current.get("generation_card")
    primitives = (
        generation.get("backend_primitives")
        if isinstance(generation, Mapping)
        else None
    )
    if not isinstance(primitives, Mapping):
        raise ValueError(
            "runtime TaskContext promotion requires backend primitives"
        )
    refreshed_primitives = deepcopy(dict(primitives))
    refreshed_primitives["telemetry"] = True
    current["generation_card"] = {
        "backend_primitives": refreshed_primitives,
    }

    policy = current.get("policy_card")
    if isinstance(policy, Mapping):
        refreshed_policy = deepcopy(dict(policy))
        unknown = refreshed_policy.get("unknown_metadata")
        if isinstance(unknown, list):
            refreshed_policy["unknown_metadata"] = [
                item for item in unknown if item != "semantic_actor_schema"
            ]
        current["policy_card"] = refreshed_policy
    return validate_open_query_capabilities(current)


def apply_external_hard_round_cap(
    *,
    evaluation_dir: Path,
    plan: dict[str, Any],
    round_runs: list[dict[str, Any]],
    executed_rounds: int,
    max_agent_rounds: int,
    user_request: str,
    bound_plan_session: Any = None,
    plan_agent_proposal: Mapping[str, Any] | None = None,
    plan_agent_artifact_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Persist a hard-cap stop after any evidence-backed Agent decision."""

    completed = [
        item["round_plan"].get("candidate_id")
        or item["round_plan"].get("template_id")
        for item in round_runs
    ]
    requested = [
        *plan.get("requested_candidate_ids", []),
        *plan.get("requested_template_ids", []),
    ]
    remaining = [
        candidate_id
        for candidate_id in dict.fromkeys(requested)
        if candidate_id not in completed
    ]
    agent_decision = None
    if plan_agent_proposal is not None:
        agent_decision = {
            "action": plan_agent_proposal.get("action"),
            "sub_aspect": plan_agent_proposal.get("sub_aspect"),
            "authored_from_completed_evidence": True,
            "artifact_path": plan_agent_artifact_path,
        }
    assessment = {
        "schema_version": 2,
        "state": "external_hard_round_cap_reached",
        "required_action": "stop",
        "completed_rounds": executed_rounds,
        "max_agent_rounds": max_agent_rounds,
        "remaining_candidate_ids": remaining,
        "policy_outcome_not_inferred": True,
        "plan_agent_decision_before_cap": agent_decision,
    }
    decision = {
        "schema_version": 3,
        "action": "stop",
        "transition": "stop",
        "observation_summary": (
            f"Completed {executed_rounds} round(s); the task-agnostic hard "
            "execution cap rejected the Agent's request for another round."
            if agent_decision is not None
            else (
                f"Completed {executed_rounds} round(s); the task-agnostic "
                "hard execution cap is now exhausted."
            )
        ),
        "decision_reason": "external_max_agent_rounds_budget",
        "next_aspect_id": None,
        "next_template_id": None,
        "remaining_candidate_ids_before_decision": remaining,
        "round_budget_before_decision": 0,
        "evidence_assessment": assessment,
        "plan_agent_decision_before_cap": agent_decision,
        "next_round": None,
    }
    plan.setdefault("round_decisions", []).append(decision)
    plan["planning_state"] = (
        f"stopped_after_round_{executed_rounds}_by_hard_cap"
    )
    _write_json(
        evaluation_dir / f"plan/evidence_after_round_{executed_rounds}.json",
        assessment,
    )
    _write_json(
        evaluation_dir / f"plan/decision_after_round_{executed_rounds}.json",
        decision,
    )
    _write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
    if bound_plan_session is not None:
        _write_json(
            evaluation_dir / "plan/bound_task_session.json",
            bound_plan_session.snapshot(
                user_request,
                plan,
                [item["round_summary"] for item in round_runs],
            ),
        )
    update_manifest(
        evaluation_dir,
        status=plan["planning_state"],
        plan=plan,
        hard_round_cap_stop={
            "max_agent_rounds": max_agent_rounds,
            "executed_rounds": executed_rounds,
            "decision_path": (
                f"plan/decision_after_round_{executed_rounds}.json"
            ),
            "plan_agent_action_before_cap": (
                agent_decision["action"]
                if agent_decision is not None
                else None
            ),
            "plan_agent_artifact_path": plan_agent_artifact_path,
        },
    )
    return plan, decision, assessment


@dataclass
class PlanAgentApplication:
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
    base_url: str
    gpu: int
    max_reflections: int
    telemetry_profile: str
    policy_backend: str
    runtime_target: Mapping[str, Any] | None
    smolvla_port: int
    materialize_round: MaterializeRound
    reviewed_task_registry: Path | None = None
    reviewed_tool_registry: Path | None = None
    reviewed_vqa_registry: Path | None = None
    registration_identity: Mapping[str, Any] | None = None
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

    def _persist_session(
        self,
        plan: Mapping[str, Any],
        round_runs: list[dict[str, Any]],
    ) -> None:
        _write_json(
            self.evaluation_dir / "plan/bound_task_session.json",
            self.session.snapshot(
                self.user_request,
                plan,
                [item["round_summary"] for item in round_runs],
            ),
        )

    def _observe(
        self,
        round_runs: list[dict[str, Any]],
        *,
        executed_rounds: int,
    ) -> dict[str, Any]:
        state = self.session.observe(
            [item["round_plan"] for item in round_runs],
            [item["round_summary"] for item in round_runs],
        )
        contract_candidate_ids = {
            str(item)
            for item in state["query_contract"].get("candidate_universe", [])
        }
        records_by_round = {
            str(record["round_id"]): record
            for record in state["records"]
        }
        if len(records_by_round) != len(round_runs):
            raise RuntimeError(
                "Plan Agent records are not one-to-one with completed runtime "
                "rounds"
            )
        for completed_run in round_runs:
            round_id = str(completed_run["round_plan"]["round_id"])
            record = records_by_round.get(round_id)
            if record is None:
                raise RuntimeError(
                    "Plan Agent record is missing for completed round "
                    f"{round_id!r}"
                )
            if record["candidate_id"] in contract_candidate_ids:
                completed_run["round_summary"]["candidate_evidence"] = deepcopy(
                    record["candidate_evidence"]
                )
        _write_json(
            self.evaluation_dir
            / PLAN_AGENT_SESSION
            / f"evidence_after_round_{executed_rounds:02d}.json",
            state,
        )
        return state

    def _persist_contract_stop(
        self,
        *,
        plan: dict[str, Any],
        round_plan: Mapping[str, Any],
        round_runs: list[dict[str, Any]],
        runtime_state: Mapping[str, Any],
        executed_rounds: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assessment = runtime_state["assessment"]
        query_answer = deepcopy(dict(runtime_state["query_answer"]))
        decision = {
            "schema_version": 3,
            "action": "stop",
            "transition": "stop",
            "next_aspect_id": None,
            "next_template_id": None,
            "observation_summary": assessment["rationale"],
            "decision_reason": "external_query_contract_stop",
            "answered_query": bool(query_answer["answered"]),
            "plan_step_source": "external_query_contract_stop",
            "round_budget_before_decision": assessment["budget_remaining"],
            "evidence_assessment": assessment,
            "semantic_stop_step": None,
            "next_round": None,
        }
        plan.setdefault("round_decisions", []).append(decision)
        plan["planning_state"] = (
            f"stopped_after_round_{executed_rounds}_"
            f"{assessment['stop_reason']}"
        )
        _write_json(
            self.evaluation_dir
            / f"plan/decision_after_{round_plan['round_id']}.json",
            decision,
        )
        _write_json(
            self.evaluation_dir / PLAN_AGENT_SESSION / "query_answer.json",
            query_answer,
        )
        _write_json(
            self.evaluation_dir / "plan/evaluation_plan.json",
            plan,
        )
        self._persist_session(plan, round_runs)
        update_manifest(
            self.evaluation_dir,
            status=plan["planning_state"],
            plan=plan,
            plan_agent_stop={
                "stop_reason": assessment["stop_reason"],
                "evidence_sufficient": assessment["evidence_sufficient"],
                "answered_query": query_answer["answered"],
                "answer_path": (
                    (PLAN_AGENT_SESSION / "query_answer.json").as_posix()
                ),
                "plan_agent_stop_proposed": False,
            },
        )
        return decision, query_answer

    def _decide_next_step(
        self,
        *,
        plan: dict[str, Any],
        round_plan: Mapping[str, Any],
        round_runs: list[dict[str, Any]],
        runtime_state: Mapping[str, Any],
        executed_rounds: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        observation_history = [
            item["round_summary"] for item in round_runs
        ]
        executed_candidate_ids = [
            str(
                item["round_plan"].get("candidate_id")
                or item["round_plan"].get("template_id")
            )
            for item in round_runs
        ]
        step_dir = (
            self.evaluation_dir
            / PLAN_AGENT_STEPS
            / f"after_round_{executed_rounds:02d}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)
        self.capabilities = (
            refresh_plan_agent_capabilities_from_runtime_context(
                self.capabilities,
                round_runs[-1]["child_manifest"],
            )
        )
        _write_json(
            step_dir / "runtime_capabilities.json",
            self.capabilities,
        )
        semantic_bundle = self.session.propose_semantic_step(
            self.agent,
            runtime_state,
            capabilities=self.capabilities,
            evaluation_intent=None,
        )
        (step_dir / "prompt.md").write_text(
            self.agent.last_prompt or "",
            encoding="utf-8",
        )
        for index, response in enumerate(
            self.agent.last_responses,
            start=1,
        ):
            (step_dir / f"response_{index}.txt").write_text(
                response + "\n",
                encoding="utf-8",
            )
        _write_json(
            step_dir / "semantic_proposal_bundle.json",
            semantic_bundle,
        )
        raw_proposal = semantic_bundle.get("proposal")
        if not isinstance(raw_proposal, Mapping):
            raise RuntimeError(
                "Plan Agent decision artifact has no semantic proposal"
            )

        if (
            raw_proposal.get("action") != "stop"
            and self.max_agent_rounds is not None
            and executed_rounds >= self.max_agent_rounds
        ):
            cap_artifact = {
                "schema_version": 1,
                "status": "continue_rejected_by_external_hard_cap",
                "semantic_proposal_bundle": deepcopy(semantic_bundle),
                "planning_lineage": deepcopy(
                    semantic_bundle.get("planning_lineage")
                ),
                "query_contract": deepcopy(self.session.query_contract),
                "plan_step": {
                    "action": "continue",
                    "sub_aspect": raw_proposal.get("sub_aspect"),
                    "next_round": None,
                },
            }
            _write_json(step_dir / "bound_semantic_step.json", cap_artifact)
            artifact_path = (
                f"{PLAN_AGENT_STEPS.as_posix()}/"
                f"after_round_{executed_rounds:02d}/bound_semantic_step.json"
            )
            update_manifest(
                self.evaluation_dir,
                last_plan_agent_step={
                    "status": "continue_rejected_by_external_hard_cap",
                    "after_round": executed_rounds,
                    "action": "continue",
                    "answered_query": False,
                    "semantic_sub_aspect": raw_proposal.get("sub_aspect"),
                    "resolved_template_id": None,
                    "resolved_candidate_id": None,
                    "evidence_conditioned": bool(
                        semantic_bundle.get("planning_lineage", {}).get(
                            "evidence_conditioned"
                        )
                    ),
                    "planning_lineage": deepcopy(
                        semantic_bundle.get("planning_lineage")
                    ),
                    "artifact_path": artifact_path,
                },
            )
            capped_plan, decision, _ = apply_external_hard_round_cap(
                evaluation_dir=self.evaluation_dir,
                plan=plan,
                round_runs=round_runs,
                executed_rounds=executed_rounds,
                max_agent_rounds=self.max_agent_rounds,
                user_request=self.user_request,
                bound_plan_session=self.session,
                plan_agent_proposal=raw_proposal,
                plan_agent_artifact_path=artifact_path,
            )
            return capped_plan, decision, None

        bound_step = self.session.bind_evidence_conditioned_semantic_step(
            semantic_bundle,
            runtime_state,
            capabilities=self.capabilities,
            executed_candidate_ids=executed_candidate_ids,
            evaluation_intent=None,
        )
        _write_json(step_dir / "bound_semantic_step.json", bound_step)
        plan_step = bound_step["plan_step"]
        query_answer: dict[str, Any] | None = None
        materialized_round = None
        if plan_step["action"] == "stop":
            raw_query_answer = bound_step.get("query_answer")
            if not isinstance(raw_query_answer, Mapping):
                raise RuntimeError(
                    "validated Plan Agent stop has no Query answer"
                )
            query_answer = deepcopy(dict(raw_query_answer))
            _write_json(
                self.evaluation_dir
                / PLAN_AGENT_SESSION
                / "query_answer.json",
                query_answer,
            )
        else:
            dynamic_candidate = (
                plan_step.get("proposal")
                or plan_step.get("experiment_candidate")
            )
            if not isinstance(dynamic_candidate, Mapping):
                raise RuntimeError(
                    "Plan Agent must bind every continue decision to a typed "
                    "Proposal before execution"
                )
            next_round_number = len(plan["rounds"]) + 1
            materialized_round, open_tool_bundle = self.materialize_round(
                self.repo_root,
                evaluation_dir=self.evaluation_dir,
                round_number=next_round_number,
                candidate=dynamic_candidate,
                control_execution=plan["rounds"][0]["execution"],
                policy_backend=self.policy_backend,
            )
            bound_step["execution_binding"] = {
                "schema_version": 2,
                "candidate_id": dynamic_candidate["candidate_id"],
                "materialization_path": (
                    f"{PROPOSAL_MATERIALIZATION.as_posix()}/"
                    f"round_{next_round_number:02d}"
                ),
                "taskgen_route": materialized_round["route"],
                "toolgen_route": open_tool_bundle["source"],
                "catalog_template_used": False,
                "retrieval_template_hint": bound_step["resolution"].get(
                    "retrieval_template_id"
                ),
            }
            _write_json(step_dir / "bound_semantic_step.json", bound_step)

        next_plan, decision, runtime_directive = self.session.apply_plan_step(
            plan,
            observation_history,
            plan_step,
            materialized_round=materialized_round,
            source=str(
                semantic_bundle.get("source")
                or "provider_plan_agent_open_query"
            ),
            query_contract=bound_step.get("query_contract"),
        )
        decision["semantic_proposal"] = deepcopy(semantic_bundle["proposal"])
        decision["semantic_resolution"] = deepcopy(bound_step["resolution"])
        _write_json(
            self.evaluation_dir
            / f"plan/runtime_directive_after_{round_plan['round_id']}.json",
            {
                "schema_version": 1,
                "owner": type(self.session).__name__,
                "adapter_role": (
                    "plan_agent_stop_validated_by_query_contract"
                    if plan_step["action"] == "stop"
                    else "plan_agent_retrieve_or_generate_and_adjudicate"
                ),
                **runtime_directive,
            },
        )
        artifact_path = (
            f"{PLAN_AGENT_STEPS.as_posix()}/"
            f"after_round_{executed_rounds:02d}/bound_semantic_step.json"
        )
        update_manifest(
            self.evaluation_dir,
            last_plan_agent_step={
                "status": "transition_applied",
                "after_round": executed_rounds,
                "action": plan_step["action"],
                "answered_query": bool(
                    plan_step.get("answered_query", False)
                ),
                "semantic_sub_aspect": semantic_bundle["proposal"].get(
                    "sub_aspect"
                ),
                "resolved_template_id": plan_step.get("template_id"),
                "resolved_candidate_id": plan_step.get("candidate_id"),
                "evidence_conditioned": bool(
                    bound_step.get("planning_lineage", {}).get(
                        "evidence_conditioned"
                    )
                ),
                "planning_lineage": deepcopy(
                    bound_step.get("planning_lineage")
                ),
                "artifact_path": artifact_path,
            },
        )
        if plan_step["action"] == "stop":
            update_manifest(
                self.evaluation_dir,
                plan_agent_stop={
                    "stop_reason": plan_step.get("stop_reason"),
                    "evidence_sufficient": True,
                    "answered_query": True,
                    "answer_path": (
                        (PLAN_AGENT_SESSION / "query_answer.json").as_posix()
                    ),
                    "plan_agent_stop_proposed": True,
                    "artifact_path": artifact_path,
                },
            )
        next_plan = self.session.normalize_plan(next_plan)
        if decision.get("next_round") is not None:
            decision["next_round"] = next_plan["rounds"][-1]
        _write_json(
            self.evaluation_dir / "plan/evaluation_plan.json",
            next_plan,
        )
        _write_json(
            self.evaluation_dir
            / f"plan/decision_after_{round_plan['round_id']}.json",
            decision,
        )
        update_manifest(
            self.evaluation_dir,
            status=next_plan.get("planning_state"),
            plan=next_plan,
        )
        self._persist_session(next_plan, round_runs)
        return next_plan, decision, query_answer

    def _finalize(
        self,
        *,
        plan: dict[str, Any],
        round_runs: list[dict[str, Any]],
        runtime_state: Mapping[str, Any],
        query_answer: Mapping[str, Any] | None,
        executed_rounds: int,
    ) -> dict[str, Any]:
        evaluation_aggregate = aggregate_evaluation_results(
            round_runs,
            self.evaluation_dir / "summary/aggregate_result.json",
        )
        summary = {
            "schema_version": 2,
            "evaluation_id": self.evaluation_id,
            "status": (
                "completed"
                if round_runs
                and all(
                    item["round_summary"]["pipeline_passed"]
                    for item in round_runs
                )
                else "completed_with_pipeline_failure"
            ),
            "rounds": [item["round_summary"] for item in round_runs],
            "aggregate": compact_aggregate_result(evaluation_aggregate),
        }
        evidence = build_evidence_bundle(
            self.repo_root,
            self.evaluation_id,
            self.user_request,
            plan,
            round_runs,
            evaluation_aggregate,
        )
        final_query_answer = (
            deepcopy(dict(query_answer))
            if query_answer is not None
            else None
        )
        if final_query_answer is None:
            externally_stopped_assessment = {
                **runtime_state["assessment"],
                "should_stop": True,
                "stop_reason": "external_hard_round_cap",
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": (
                    "An external execution cap stopped the run before the "
                    "query-sufficiency contract was satisfied."
                ),
            }
            final_query_answer = render_query_answer(
                self.user_request,
                externally_stopped_assessment,
                runtime_state["records"],
                baseline_valid=bool(runtime_state["control_passed"]),
            )
            _write_json(
                self.evaluation_dir
                / PLAN_AGENT_SESSION
                / "query_answer.json",
                final_query_answer,
            )
        evidence["plan_agent_session"] = {
            "schema_version": 1,
            "query_contract": runtime_state["query_contract"],
            "assessment": runtime_state["assessment"],
            "query_answer": final_query_answer,
            "records": runtime_state["records"],
            "artifacts": {
                "query_answer": (
                    f"mea/evaluation_runs/{self.evaluation_id}/plan/"
                    "plan_agent_session/query_answer.json"
                ),
                "latest_evidence": (
                    f"mea/evaluation_runs/{self.evaluation_id}/plan/"
                    "plan_agent_session/"
                    f"evidence_after_round_{executed_rounds:02d}.json"
                ),
            },
        }
        flagship_acceptance = build_compact_flagship_acceptance(
            round_runs,
            global_route_result=self.global_route_result,
            claim_first_runtime_state=runtime_state,
            claim_first_query_answer=final_query_answer,
            free_concern_bundle=self.free_concern_bundle,
            open_task_resolution=self.open_task_resolution,
            concern_candidate_resolution=self.concern_candidate_resolution,
            history_disabled=self.history_disabled,
            cli_candidate_hint_used=self.cli_candidate_hint_used,
        )
        if flagship_acceptance is not None:
            summary["flagship_acceptance"] = flagship_acceptance
            evidence["flagship_acceptance"] = flagship_acceptance
        _write_json(self.evaluation_dir / "summary/summary.json", summary)
        _write_json(
            self.evaluation_dir / "summary/evidence_bundle.json",
            evidence,
        )
        update_manifest(
            self.evaluation_dir,
            status="generating_answer",
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            summary=summary,
        )
        feedback = PlanAgentFinalSummary(
            self.repo_root,
            self.provider,
            model=self.models["feedback"],
        ).generate(
            evidence,
            output_dir=self.evaluation_dir / "answer",
        )
        report_path = self.evaluation_dir / "evaluation_report.md"
        report_path.write_text(
            render_evaluation_report(evidence, feedback),
            encoding="utf-8",
        )
        update_manifest(
            self.evaluation_dir,
            status=summary["status"],
            lifecycle_status="completed",
            execution_finished_at=datetime.now().astimezone().isoformat(),
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            answer_path="answer/answer.json",
            report_path="evaluation_report.md",
            child_run_ids=[
                item["child_manifest"].get("run_id")
                for item in round_runs
            ],
            summary=summary,
            answer=feedback,
            flagship_acceptance=flagship_acceptance,
        )
        compact_evidence_report = write_evidence_report(
            self.repo_root,
            self.evaluation_dir,
            destination=self.evaluation_dir / "evidence_report.md",
        )
        update_manifest(
            self.evaluation_dir,
            evidence_report_path="evidence_report.md",
            evidence_report_bundle=compact_evidence_report,
        )
        history_index: dict[str, Any] = {
            "status": (
                "disabled" if self.history_disabled else "not_available"
            )
        }
        if self.history_database is not None:
            try:
                history_index = {
                    "status": "passed",
                    **self.history_database.index_evaluation_dir(
                        self.evaluation_dir
                    ),
                }
            except Exception as exc:
                history_index = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        update_manifest(self.evaluation_dir, history_index=history_index)
        retrieval = self.history_retrieval or {}
        return {
            "evaluation_id": self.evaluation_id,
            "child_run_ids": [
                item["child_manifest"].get("run_id")
                for item in round_runs
            ],
            "summary": summary,
            "answer": feedback,
            "history_retrieval": {
                "status": retrieval.get("status"),
                "selected_count": self.history_context_count,
            },
            "history_index": history_index,
            "report_path": str(report_path.relative_to(self.repo_root)),
            "evidence_report_path": str(
                (self.evaluation_dir / "evidence_report.md").relative_to(
                    self.repo_root
                )
            ),
        }

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
                round_result = self.round_executor.execute(
                    RoundExecutionRequest(
                        repo_root=self.repo_root,
                        evaluation_dir=self.evaluation_dir,
                        evaluation_id=self.evaluation_id,
                        round_plan=round_plan,
                        text_model=self.models["taskgen"],
                        vision_model=self.models["vision"],
                        base_url=self.base_url,
                        gpu=self.gpu,
                        max_reflections=self.max_reflections,
                        provider=self.provider,
                        toolgen_model=self.models["toolgen"],
                        telemetry_profile=self.telemetry_profile,
                        reviewed_task_registry=self.reviewed_task_registry,
                        reviewed_tool_registry=self.reviewed_tool_registry,
                        reviewed_vqa_registry=self.reviewed_vqa_registry,
                        registration_identity=self.registration_identity,
                        policy_backend=self.policy_backend,
                        runtime_target=self.runtime_target,
                        smolvla_port=self.smolvla_port,
                    )
                )
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
                assessment = runtime_state["assessment"]
                if (
                    assessment["should_stop"]
                    and assessment.get("evidence_sufficient") is not True
                ):
                    _, query_answer = self._persist_contract_stop(
                        plan=self.plan,
                        round_plan=round_plan,
                        round_runs=round_runs,
                        runtime_state=runtime_state,
                        executed_rounds=executed_rounds,
                    )
                    break

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
