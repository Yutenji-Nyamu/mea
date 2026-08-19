"""Production Plan Agent manifest finalization and dispatch."""

from __future__ import annotations

import json
from typing import Any, Callable

from mea.agent_run_context import AgentRunContext
from mea.plan_artifacts import (
    INITIAL_SUB_ASPECT_PROPOSAL,
    PLAN_AGENT_CAPABILITIES,
    PROPOSAL_FILENAME,
)
from mea.plan_agent_application import update_manifest
from mea.plan_agent_bootstrap import run_plan_agent_application
from mea.planner import PlanAgentSession, policy_task_binding_from_target


def finalize_manifest_and_dispatch(
    context: AgentRunContext,
    *,
    write_json: Callable[[Any, Any], None],
    write_global_route_trace: Callable[..., None],
    bound_target_task_name: Callable[[Any], str],
) -> dict[str, Any] | None:
    """Freeze run metadata and enter the sole production Application."""

    args = context.args
    if (
        context.plan is None
        or context.manifest is None
        or context.evaluation_dir is None
        or context.evaluation_id is None
    ):
        raise RuntimeError("Agent plan/session setup is incomplete")
    plan = context.plan
    evaluation_dir = context.evaluation_dir
    if context.global_route_result is not None:
        write_global_route_trace(
            evaluation_dir,
            route_result=context.global_route_result,
            history_retrieval=context.global_history_retrieval,
        )

    update_manifest(
        evaluation_dir,
        auto_route=args.auto_route,
        global_query_route_path=(
            "plan/global_query_route.json" if args.auto_route else None
        ),
        global_route_selection=(
            context.global_route_result["selection"]
            if context.global_route_result is not None
            else None
        ),
        model_profile=args.model_profile,
        resolved_models={
            "planner": context.models["planner"],
            "taskgen": context.models["taskgen"],
            "toolgen": context.models["toolgen"],
            "vision": context.models["vision"],
            "answer": context.models["feedback"],
        },
        history_database=(
            str(context.history_path.relative_to(context.repo_root))
            if context.history_path.is_relative_to(context.repo_root)
            else str(context.history_path)
        ),
        history_retrieval_status=context.history_retrieval.get("status"),
        task_name=args.task_name,
        task_module=(
            policy_task_binding_from_target(context.initial_target)[
                "task_module"
            ]
            if context.initial_target is not None
            else args.task_module
        ),
        telemetry_profile=args.telemetry_profile,
        execution_backend=context.execution_backend,
        policy_backend=args.policy_backend,
        policy_checkpoint_id=(
            context.runtime_policy_spec.checkpoint_id
            if context.runtime_policy_spec is not None
            else None
        ),
        open_query_planner="plan_agent_v1",
        plan_runtime_limits_path="plan/runtime_limits.json",
        plan_agent_capabilities_path=PLAN_AGENT_CAPABILITIES.as_posix(),
        max_agent_rounds=args.max_agent_rounds,
        bound_task_name=(
            bound_target_task_name(context.evaluation_target)
            if context.evaluation_target is not None
            else None
        ),
        evaluation_target=context.evaluation_target,
        plan_session_path=context.plan_session_path,
    )

    if (
        context.initial_semantic_bundle is not None
        and context.frozen_first_candidate is not None
    ):
        frozen_dir = evaluation_dir / INITIAL_SUB_ASPECT_PROPOSAL
        write_json(
            frozen_dir / "semantic_proposal_bundle.json",
            context.initial_semantic_bundle,
        )
        write_json(
            frozen_dir / PROPOSAL_FILENAME,
            context.frozen_first_candidate,
        )
        update_manifest(
            evaluation_dir,
            initial_candidate_source="provider_plan_agent_direct_materialization",
            frozen_first_candidate_path=(
                (INITIAL_SUB_ASPECT_PROPOSAL / PROPOSAL_FILENAME).as_posix()
            ),
        )

    if args.plan_only:
        update_manifest(evaluation_dir, status="planned_only")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    if context.provider is None:
        raise RuntimeError("execution provider was not initialized")
    if not isinstance(context.plan_session, PlanAgentSession):
        raise RuntimeError("production runtime requires PlanAgentSession")
    if context.plan_agent is None or context.plan_agent_capabilities is None:
        raise RuntimeError("production Plan Agent application was not initialized")
    result = run_plan_agent_application(
        args=args,
        repo_root=context.repo_root,
        evaluation_dir=evaluation_dir,
        evaluation_id=context.evaluation_id,
        plan=plan,
        session=context.plan_session,
        agent=context.plan_agent,
        capabilities=context.plan_agent_capabilities,
        provider=context.provider,
        models=context.models,
        runtime_target=context.initial_target,
        global_route_result=context.global_route_result,
        concern_bundle=context.query_interpretation_bundle,
        open_task_resolution=context.open_task_resolution,
        concern_candidate_resolution=context.concern_candidate_resolution,
        history_database=context.history_database,
        history_retrieval=context.history_retrieval,
        history_context_count=len(context.history_context),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


__all__ = ["finalize_manifest_and_dispatch"]
