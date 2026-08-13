"""Compatibility runner for legacy and catalog ManipEvalAgent sessions.

The production Plan Agent owns its loop in :mod:`mea.plan_agent_application`.
This module retains the historical bounded-task/catalog lifecycle without
keeping a second application loop in the production CLI entrypoint.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from experiments.paper.compat_bounded_proposals import (
    adjudicate_bounded_transition,
    apply_bounded_round_proposal,
    persist_adaptive_step_selection,
)

from mea.agent_evidence import build_evidence_bundle, compact_aggregate_result
from mea.feedback import (
    PlanAgentFinalSummary,
    render_evaluation_report,
    write_evidence_report,
)
from mea.plan_agent_application import (
    apply_external_hard_round_cap,
    update_manifest,
)
from experiments.paper.compat_round_executor import (
    LegacySubprocessRoundExecutor,
    LegacySubprocessServices,
)
from mea.robotwin.native_agent_round import (
    execute_act_method_round,
    execute_hyvla_method_round,
    execute_smolvla_method_round,
)
from mea.robotwin.production_round_executor import (
    build_production_round_executor,
)
from mea.round_evidence import aggregate_evaluation_results
from mea.round_executor import (
    RoundExecutionRequest,
    RoundExecutor,
)
from mea.taskgen import round_materialization as taskgen_round_materialization
from mea.taskgen.runtime import create_generic_provider_taskgen_run


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> int:
    """Run the historical TaskGen subprocess while preserving its log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def _build_round_executor(
    *,
    native_act: bool,
    reviewed_task_registry: Path | None = None,
    registration_identity: dict[str, Any] | None = None,
) -> RoundExecutor:
    """Build the compatibility executor used by the historical wrapper API."""

    native_policy_rounds = {
        "hyvla": partial(
            execute_hyvla_method_round,
            generated_task_materializer=create_generic_provider_taskgen_run,
        ),
        "smolvla": partial(
            execute_smolvla_method_round,
            generated_task_materializer=create_generic_provider_taskgen_run,
        ),
    }
    if native_act:
        native_policy_rounds["act"] = partial(
            execute_act_method_round,
            generated_task_materializer=create_generic_provider_taskgen_run,
        )

    return LegacySubprocessRoundExecutor(
        LegacySubprocessServices(
            update_manifest=update_manifest,
            build_taskgen_command=taskgen_round_materialization.build_taskgen_command,
            run_logged=run_logged,
            native_policy_rounds=native_policy_rounds,
        ),
        reviewed_task_registry=reviewed_task_registry,
        registration_identity=registration_identity,
    )


def execute_round(
    repo_root: Path,
    evaluation_dir: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    *,
    text_model: str,
    vision_model: str,
    base_url: str | None,
    gpu: int,
    max_reflections: int,
    provider: Any,
    toolgen_model: str,
    telemetry_profile: str = "balanced_v1",
    reviewed_task_registry: Path | None = None,
    reviewed_tool_registry: Path | None = None,
    reviewed_vqa_registry: Path | None = None,
    registration_identity: dict[str, Any] | None = None,
    policy_backend: str = "act",
    runtime_target: Mapping[str, Any] | None = None,
    smolvla_port: int = 18771,
    hyvla_port: int = 18781,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any], int]:
    """Compatibility wrapper for callers that predate ``RoundExecutor``."""

    request = RoundExecutionRequest(
        repo_root=repo_root,
        evaluation_dir=evaluation_dir,
        evaluation_id=evaluation_id,
        round_plan=round_plan,
        text_model=text_model,
        vision_model=vision_model,
        gpu=gpu,
        max_reflections=max_reflections,
        provider=provider,
        toolgen_model=toolgen_model,
        telemetry_profile=telemetry_profile,
        reviewed_tool_registry=reviewed_tool_registry,
        reviewed_vqa_registry=reviewed_vqa_registry,
        policy_backend=policy_backend,
        runtime_target=runtime_target,
        policy_server_port=(
            hyvla_port if policy_backend == "hyvla" else smolvla_port
        ),
    )
    executor = (
        build_production_round_executor()
        if policy_backend != "act" or runtime_target is not None
        else _build_round_executor(
            native_act=False,
            reviewed_task_registry=reviewed_task_registry,
            registration_identity=registration_identity,
        )
    )
    result = executor.execute(request)
    return (
        result.child_manifest,
        result.child_dir,
        result.round_summary,
        result.tool_evaluation,
        result.returncode,
    )


def run_legacy_catalog_agent(
    *,
    args: Any,
    repo_root: Path,
    evaluation_dir: Path,
    evaluation_id: str,
    plan: dict[str, Any],
    planner: Any,
    plan_session: Any,
    adaptive_step_agent: Any,
    planning_context: dict[str, Any] | None,
    registered_execution: dict[str, Any] | None,
    proposal_agent: Any,
    fixed_click_bell: bool,
    legacy_click_bell: bool,
    provider: Any,
    models: Mapping[str, str],
    reviewed_tool_registry: Path | None,
    reviewed_vqa_registry: Path | None,
    registration_identity: dict[str, Any] | None,
    runtime_target: Mapping[str, Any] | None,
    history_database: Any,
    history_retrieval: Mapping[str, Any],
    history_context_count: int,
) -> dict[str, Any]:
    """Execute the retained legacy/catalog loop after explicit CLI dispatch."""

    round_runs: list[dict[str, Any]] = []
    round_executor = build_production_round_executor()
    active_failure_stage = "round_execution"
    try:
        executed_rounds = 0
        while executed_rounds < len(plan["rounds"]):
            active_failure_stage = f"round_{executed_rounds + 1}_execution"
            round_plan = plan["rounds"][executed_rounds]
            round_result = round_executor.execute(
                RoundExecutionRequest(
                    repo_root=repo_root,
                    evaluation_dir=evaluation_dir,
                    evaluation_id=evaluation_id,
                    round_plan=round_plan,
                    text_model=models["taskgen"],
                    vision_model=models["vision"],
                    gpu=args.gpu,
                    max_reflections=args.max_reflections,
                    provider=provider,
                    toolgen_model=models["toolgen"],
                    telemetry_profile=args.telemetry_profile,
                    reviewed_tool_registry=reviewed_tool_registry,
                    reviewed_vqa_registry=reviewed_vqa_registry,
                    policy_backend=args.policy_backend,
                    runtime_target=runtime_target,
                    policy_server_port=(
                        args.hyvla_port
                        if args.policy_backend == "hyvla"
                        else args.smolvla_port
                    ),
                )
            )
            child_manifest = round_result.child_manifest
            child_dir = round_result.child_dir
            round_summary = round_result.round_summary
            tool_evaluation = round_result.tool_evaluation
            returncode = round_result.returncode
            round_runs.append(
                {
                    "round_plan": round_plan,
                    "child_manifest": child_manifest,
                    "child_dir": child_dir,
                    "round_summary": round_summary,
                    "tool_evaluation": tool_evaluation,
                    "returncode": returncode,
                }
            )
            executed_rounds += 1

            plan_before_decision = plan
            observation_history = [
                item["round_summary"] for item in round_runs
            ]
            dynamic_step_session = (
                plan_session is not None
                and adaptive_step_agent is not None
                and planning_context is not None
            )
            if (
                args.max_agent_rounds is not None
                and executed_rounds >= args.max_agent_rounds
            ):
                plan, decision, _cap_assessment = apply_external_hard_round_cap(
                    evaluation_dir=evaluation_dir,
                    plan=plan,
                    round_runs=round_runs,
                    executed_rounds=executed_rounds,
                    max_agent_rounds=args.max_agent_rounds,
                    user_request=args.request,
                    bound_plan_session=plan_session,
                )
                break
            if dynamic_step_session:
                active_failure_stage = (
                    f"adaptive_decision_after_round_{executed_rounds}"
                )
                navigation_options = plan_session.navigation_options(
                    plan_before_decision,
                    observation_history,
                    allowed_template_ids=(
                        registered_execution["expected_candidate_suite"]
                        if registered_execution is not None
                        else None
                    ),
                )
                step_bundle = adaptive_step_agent.propose(
                    args.request,
                    navigation_options=navigation_options,
                    planning_context=planning_context,
                )
                plan_step = step_bundle["proposal"]
                step_path = persist_adaptive_step_selection(
                    evaluation_dir,
                    after_round=executed_rounds,
                    prompt=adaptive_step_agent.last_prompt,
                    responses=adaptive_step_agent.last_responses,
                    step_bundle=step_bundle,
                    navigation_options=navigation_options,
                )
                update_manifest(
                    evaluation_dir,
                    last_adaptive_step={
                        "status": "selected_pending_materialization",
                        "after_round": executed_rounds,
                        "action": plan_step["action"],
                        "artifact_path": f"{step_path}/plan_step_bundle.json",
                    },
                )
                materialized_round = None
                if plan_step["action"] != "stop":
                    active_failure_stage = (
                        f"template_materialization_after_round_{executed_rounds}"
                    )
                    materialize = getattr(planner, "materialize_plan_step", None)
                    if not callable(materialize):
                        raise RuntimeError(
                            "bound task planner cannot materialize PlanStepProposal"
                        )
                    materialized_round = materialize(
                        plan_step["template_id"],
                        len(plan_before_decision["rounds"]) + 1,
                        args.request,
                    )
                    if args.proposal_mode == "bounded_each_round":
                        active_failure_stage = (
                            f"bounded_proposal_after_round_{executed_rounds}"
                        )
                        if proposal_agent is None:
                            raise RuntimeError(
                                "bounded_each_round proposal state was not initialized"
                            )
                        next_round_number = len(plan_before_decision["rounds"]) + 1
                        materialized_round, _proposal_artifact = (
                            apply_bounded_round_proposal(
                                proposal_agent=proposal_agent,
                                user_query=args.request,
                                target=plan_session.target,
                                planning_context=planning_context,
                                round_plan=materialized_round,
                                evaluation_dir=evaluation_dir,
                                round_number=next_round_number,
                            )
                        )
                active_failure_stage = (
                    f"plan_transition_after_round_{executed_rounds}"
                )
                plan, decision, runtime_directive = plan_session.apply_plan_step(
                    plan_before_decision,
                    observation_history,
                    plan_step,
                    materialized_round=materialized_round,
                    source=step_bundle["source"],
                )
                write_json(
                    evaluation_dir
                    / f"plan/runtime_directive_after_{round_plan['round_id']}.json",
                    {
                        "schema_version": 1,
                        "owner": "BoundTaskPlanSession",
                        "adapter_role": "discover_materialize_and_adjudicate",
                        **runtime_directive,
                    },
                )
            else:
                candidate_plan, candidate_decision = planner.decide_next_round(
                    evaluation_id=evaluation_id,
                    user_request=args.request,
                    current_plan=plan_before_decision,
                    observation_history=observation_history,
                )
                common_adaptive_session = (
                    plan_session is not None
                    and not fixed_click_bell
                    and not legacy_click_bell
                    and candidate_decision.get("action") in {"continue", "stop"}
                )
                if common_adaptive_session:
                    plan, decision, runtime_directive = adjudicate_bounded_transition(
                        plan_session=plan_session,
                        user_query=args.request,
                        observation_history=observation_history,
                        current_plan=plan_before_decision,
                        candidate_plan=candidate_plan,
                        candidate_decision=candidate_decision,
                        proposal_mode=args.proposal_mode,
                        proposal_agent=proposal_agent,
                        planning_context=planning_context,
                        evaluation_dir=evaluation_dir,
                    )
                    write_json(
                        evaluation_dir
                        / f"plan/runtime_directive_after_{round_plan['round_id']}.json",
                        {
                            "schema_version": 1,
                            "owner": "BoundTaskPlanSession",
                            "adapter_role": "materialize_and_explain",
                            **runtime_directive,
                        },
                    )
                else:
                    plan, decision = candidate_plan, candidate_decision
            if plan_session is not None:
                plan = plan_session.normalize_plan(plan)
                if decision.get("next_round") is not None:
                    decision["next_round"] = plan["rounds"][-1]
                write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
                write_json(
                    evaluation_dir
                    / f"plan/decision_after_{round_plan['round_id']}.json",
                    decision,
                )
                update_manifest(
                    evaluation_dir,
                    status=plan.get("planning_state"),
                    plan=plan,
                    last_adaptive_step=(
                        {
                            "status": "transition_applied",
                            "after_round": executed_rounds,
                            "action": decision.get("action"),
                            "artifact_path": (
                                "plan/adaptive_steps/"
                                f"after_round_{executed_rounds:02d}"
                                "/plan_step_bundle.json"
                            ),
                            "decision_path": (
                                f"plan/decision_after_{round_plan['round_id']}.json"
                            ),
                        }
                        if dynamic_step_session
                        else None
                    ),
                )
            if decision["action"] == "stop":
                if plan_session is not None:
                    write_json(
                        evaluation_dir / "plan/bound_task_session.json",
                        plan_session.snapshot(
                            args.request, plan, observation_history
                        ),
                    )
                break
            if plan_session is not None:
                write_json(
                    evaluation_dir / "plan/bound_task_session.json",
                    plan_session.snapshot(args.request, plan, observation_history),
                )

        active_failure_stage = "evaluation_aggregation"
        evaluation_aggregate = aggregate_evaluation_results(
            round_runs,
            evaluation_dir / "summary/aggregate_result.json",
        )
        summary = {
            "schema_version": 2,
            "evaluation_id": evaluation_id,
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
            repo_root,
            evaluation_id,
            args.request,
            plan,
            round_runs,
            evaluation_aggregate,
        )
        flagship_acceptance = None
        write_json(evaluation_dir / "summary/summary.json", summary)
        write_json(evaluation_dir / "summary/evidence_bundle.json", evidence)
        update_manifest(
            evaluation_dir,
            status="generating_answer",
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            summary=summary,
        )
        active_failure_stage = "final_answer"
        feedback = PlanAgentFinalSummary(
            repo_root,
            provider,
            model=models["feedback"],
        ).generate(
            evidence,
            output_dir=evaluation_dir / "answer",
        )
        report_path = evaluation_dir / "evaluation_report.md"
        report_path.write_text(
            render_evaluation_report(evidence, feedback),
            encoding="utf-8",
        )
        update_manifest(
            evaluation_dir,
            status=summary["status"],
            lifecycle_status="completed",
            execution_finished_at=datetime.now().astimezone().isoformat(),
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            answer_path="answer/answer.json",
            report_path="evaluation_report.md",
            child_run_ids=[
                item["child_manifest"].get("run_id") for item in round_runs
            ],
            summary=summary,
            answer=feedback,
            flagship_acceptance=flagship_acceptance,
        )
        compact_evidence_report = write_evidence_report(
            repo_root,
            evaluation_dir,
            destination=evaluation_dir / "evidence_report.md",
        )
        update_manifest(
            evaluation_dir,
            evidence_report_path="evidence_report.md",
            evidence_report_bundle=compact_evidence_report,
        )
        history_index = {
            "status": "disabled" if args.no_history else "not_available"
        }
        if history_database is not None:
            try:
                history_index = {
                    "status": "passed",
                    **history_database.index_evaluation_dir(evaluation_dir),
                }
            except Exception as exc:
                history_index = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        update_manifest(evaluation_dir, history_index=history_index)
        result = {
            "evaluation_id": evaluation_id,
            "child_run_ids": [
                item["child_manifest"].get("run_id") for item in round_runs
            ],
            "summary": summary,
            "answer": feedback,
            "history_retrieval": {
                "status": history_retrieval.get("status"),
                "selected_count": history_context_count,
            },
            "history_index": history_index,
            "report_path": str(report_path.relative_to(repo_root)),
            "evidence_report_path": str(
                (evaluation_dir / "evidence_report.md").relative_to(repo_root)
            ),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except Exception as exc:
        update_manifest(
            evaluation_dir,
            status="failed",
            lifecycle_status="failed",
            failure_stage=active_failure_stage,
            completed_rounds=len(round_runs),
            active_child_run_id=None,
            execution_finished_at=datetime.now().astimezone().isoformat(),
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise


__all__ = ["execute_round", "run_legacy_catalog_agent", "run_logged"]
