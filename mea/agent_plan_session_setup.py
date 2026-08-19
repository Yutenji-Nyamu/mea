"""Production Plan Agent session construction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Mapping

from mea.agent_run_context import AgentRunContext
from mea.plan_artifacts import PLAN_AGENT_CAPABILITIES
from mea.plan_agent_application import update_manifest
from mea.plan_agent_bootstrap import persist_runtime_limits
from mea.planner import PlanAgent, PlanAgentSession, project_open_query_capabilities


def prepare_plan_session(
    context: AgentRunContext,
    *,
    write_json: Callable[[Any, Any], None],
) -> None:
    """Normalize the initial plan and create its evidence-aware session."""

    args = context.args
    if context.plan is None or context.manifest is None:
        raise RuntimeError("initial plan must exist before session setup")
    if context.evaluation_dir is None or context.round_budget is None:
        raise RuntimeError("Plan Agent runtime was not initialized")
    if context.initial_target is None:
        raise RuntimeError("Plan Agent runtime target was not initialized")

    plan = context.plan
    manifest = context.manifest
    initial_failure_stage = "initial_plan_session_validation"
    try:
        plan["max_rounds"] = context.round_budget
        session = PlanAgentSession(
            args.request,
            context.initial_target,
            runtime_limits=context.plan_runtime_limits,
            require_control_anchor=context.control_required,
            control_round=(
                plan["rounds"][0] if context.control_required else None
            ),
        )
        if context.frozen_first_candidate is not None:
            context.frozen_first_candidate = session.register_frozen_candidate(
                context.frozen_first_candidate
            )
        plan = session.normalize_plan(plan)
        planning_context = session.planning_context(context.repo_root)
        write_json(
            context.evaluation_dir / "plan/planning_context.json",
            planning_context,
        )
        capabilities = project_open_query_capabilities(planning_context)
        if (
            isinstance(context.query_interpretation_bundle, Mapping)
            and isinstance(
                context.query_interpretation_bundle.get("concern"), Mapping
            )
        ):
            if not isinstance(context.concern_candidate_resolution, Mapping):
                raise RuntimeError(
                    "online Query-interpretation candidate domain was not "
                    "resolved before planning"
                )
            write_json(
                context.evaluation_dir
                / "plan/concern_candidate_resolution.json",
                {
                    **context.concern_candidate_resolution,
                    "planner_domain_role": "routing_and_retrieval_hint_only",
                    "planner_domain_restricted": False,
                },
            )
        agent = None
        if not args.plan_only:
            if context.provider is None:
                raise RuntimeError("Plan Agent provider was not initialized")
            agent = PlanAgent(
                context.provider,
                model=context.models["planner"],
                repo_root=context.repo_root,
            )
        write_json(
            context.evaluation_dir / PLAN_AGENT_CAPABILITIES,
            capabilities,
        )
        session.runtime_limits = persist_runtime_limits(
            context.evaluation_dir,
            plan,
            session.runtime_limits,
        )
        manifest.setdefault("planner", {}).update(
            {
                "public_planner": "PlanAgent",
                "control_anchor_owned_by_runtime": (
                    session.require_control_anchor
                ),
                "catalog_navigation_was_model_visible": False,
                "global_router_scope": "task_and_checkpoint_only",
                "aspect_selection_owner": "PlanAgent",
                "candidate_domain_source": (
                    "runtime_capabilities_plus_open_generation"
                ),
                "pre_control_concern_restricts_planner_domain": False,
                "concern_candidate_resolution_path": (
                    "plan/concern_candidate_resolution.json"
                    if context.concern_candidate_resolution is not None
                    else None
                ),
            }
        )
        manifest["plan"] = plan
        session_snapshot = session.snapshot(args.request, plan)
    except (ValueError, RuntimeError) as exc:
        if (context.evaluation_dir / "manifest.json").is_file():
            update_manifest(
                context.evaluation_dir,
                status="failed",
                lifecycle_status="failed",
                failure_stage=initial_failure_stage,
                completed_rounds=0,
                active_child_run_id=None,
                execution_finished_at=datetime.now().astimezone().isoformat(),
                failure={"type": type(exc).__name__, "message": str(exc)},
            )
        raise RuntimeError(
            f"PlanAgentSession validation failed: {exc}"
        ) from exc

    session_path = "plan/plan_agent_session.json"
    write_json(context.evaluation_dir / "plan/evaluation_plan.json", plan)
    write_json(context.evaluation_dir / session_path, session_snapshot)
    update_manifest(
        context.evaluation_dir,
        plan=plan,
        planner=manifest.get("planner"),
        planning_context_path="plan/planning_context.json",
    )
    context.plan = plan
    context.manifest = manifest
    context.plan_session = session
    context.plan_session_path = session_path
    context.evaluation_target = session_snapshot["target"]
    context.planning_context = planning_context
    context.plan_agent = agent
    context.plan_agent_capabilities = capabilities


__all__ = ["prepare_plan_session"]
