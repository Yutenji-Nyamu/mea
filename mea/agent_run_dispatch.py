"""Manifest finalization and production/compat Agent dispatch."""

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
from mea.plan_agent_bootstrap import canonical_sha256, run_plan_agent_application
from mea.planner import PlanAgentSession, policy_task_binding_from_target


def finalize_manifest_and_dispatch(
    context: AgentRunContext,
    *,
    write_json: Callable[[Any, Any], None],
    write_global_route_trace: Callable[..., None],
    bound_target_task_name: Callable[[Any], str],
) -> dict[str, Any] | None:
    """Freeze run identity, then dispatch the production or paper runner."""

    args = context.args
    if (
        context.plan is None
        or context.manifest is None
        or context.evaluation_dir is None
        or context.evaluation_id is None
    ):
        raise RuntimeError("Agent plan/session setup is incomplete")
    plan = context.plan
    manifest = context.manifest
    evaluation_dir = context.evaluation_dir

    candidate_suite = list(plan.get("requested_template_ids") or [])
    planning_policy = (
        "fixed_predeclared_v1"
        if context.fixed_click_bell
        else "dynamic_evidence_v1"
        if (
            context.plan_agent_mode
            or context.adaptive_click_bell
            or args.task_name == "beat_block_hammer"
        )
        else None
    )
    registration_identity = None
    if context.registered_execution is not None:
        registration_identity = dict(
            context.registered_execution["registration_identity"]
        )
        if planning_policy != args.registered_strategy:
            raise RuntimeError(
                "registered strategy does not match resolved planner policy"
            )
        if (
            candidate_suite
            != context.registered_execution["expected_candidate_suite"]
        ):
            update_manifest(
                evaluation_dir,
                status="registration_failed",
                registration_identity=registration_identity,
                registration_failure=(
                    "planner candidate suite differs from preregistration"
                ),
            )
            raise RuntimeError(
                "planner candidate suite differs from preregistered route"
            )
        write_json(
            evaluation_dir / "plan/registered_route.json",
            context.registered_execution["route"],
        )

    if (
        context.global_catalog is not None
        and context.global_route_result is not None
        and context.global_router is not None
    ):
        write_global_route_trace(
            evaluation_dir,
            catalog=context.global_catalog,
            route_result=context.global_route_result,
            router=context.global_router,
            history_retrieval=context.global_history_retrieval,
        )

    update_manifest(
        evaluation_dir,
        auto_route=args.auto_route,
        global_query_route_path=(
            "plan/global_query_route.json" if args.auto_route else None
        ),
        global_act_catalog_path=(
            "plan/global_act_catalog.json" if args.auto_route else None
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
        task_profile=context.routed_task_profile or args.task_profile,
        generated_rounds=(
            args.generated_rounds if context.bounded_click_bell else None
        ),
        telemetry_profile=args.telemetry_profile,
        execution_backend=context.execution_backend,
        policy_backend=args.policy_backend,
        policy_checkpoint_id=(
            context.runtime_policy_spec.checkpoint_id
            if context.runtime_policy_spec is not None
            else None
        ),
        planning_policy=planning_policy,
        open_query_planner=args.open_query_planner,
        query_sufficiency_contract_path=(
            "plan/query_sufficiency_contract.json"
            if context.plan_agent_mode
            else None
        ),
        plan_agent_capabilities_path=(
            PLAN_AGENT_CAPABILITIES.as_posix()
            if context.plan_agent_mode
            else None
        ),
        candidate_suite_sha256=(
            canonical_sha256(candidate_suite) if candidate_suite else None
        ),
        reviewed_tool_registry=_display_path(
            context.reviewed_tool_registry,
            repo_root=context.repo_root,
        ),
        reviewed_task_registry=_display_path(
            context.reviewed_task_registry,
            repo_root=context.repo_root,
        ),
        reviewed_vqa_registry=_display_path(
            context.reviewed_vqa_registry,
            repo_root=context.repo_root,
        ),
        registration_identity=registration_identity,
        evidence_manifest=(
            str(args.evidence_manifest)
            if registration_identity is not None
            else None
        ),
        command_plan=(
            str(args.command_plan)
            if registration_identity is not None
            else None
        ),
        registered_route=(
            str(args.registered_route)
            if registration_identity is not None
            else None
        ),
        max_agent_rounds=args.max_agent_rounds,
        bound_task_name=(
            bound_target_task_name(context.evaluation_target)
            if context.evaluation_target is not None
            else None
        ),
        bound_requested_aspect_ids=(
            list(args.bound_requested_aspect_ids)
            if args.bound_requested_aspect_ids is not None
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
            initial_candidate_source=(
                "provider_plan_agent_direct_materialization"
            ),
            frozen_first_candidate_path=(
                (INITIAL_SUB_ASPECT_PROPOSAL / PROPOSAL_FILENAME).as_posix()
            ),
        )

    context.registration_identity = registration_identity
    if args.plan_only:
        update_manifest(evaluation_dir, status="planned_only")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan

    if context.provider is None:
        raise RuntimeError("execution provider was not initialized")
    if isinstance(context.plan_session, PlanAgentSession):
        if (
            context.plan_agent is None
            or context.plan_agent_capabilities is None
        ):
            raise RuntimeError(
                "production Plan Agent application was not initialized"
            )
        application_result = run_plan_agent_application(
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
            reviewed_tool_registry=context.reviewed_tool_registry,
            reviewed_vqa_registry=context.reviewed_vqa_registry,
            global_route_result=context.global_route_result,
            concern_bundle=context.query_interpretation_bundle,
            open_task_resolution=context.open_task_resolution,
            concern_candidate_resolution=context.concern_candidate_resolution,
            history_database=context.history_database,
            history_retrieval=context.history_retrieval,
            history_context_count=len(context.history_context),
        )
        print(json.dumps(application_result, ensure_ascii=False, indent=2))
        return application_result

    from experiments.paper.compat_agent_runner import run_legacy_catalog_agent

    if context.compat_bounded_proposals is None:
        raise RuntimeError("paper compatibility Proposal services are unavailable")
    return run_legacy_catalog_agent(
        args=args,
        repo_root=context.repo_root,
        evaluation_dir=evaluation_dir,
        evaluation_id=context.evaluation_id,
        plan=plan,
        planner=context.planner,
        plan_session=context.plan_session,
        adaptive_step_agent=context.adaptive_step_agent,
        planning_context=context.planning_context,
        registered_execution=context.registered_execution,
        proposal_agent=context.proposal_agent,
        fixed_click_bell=context.fixed_click_bell,
        legacy_click_bell=context.legacy_click_bell,
        provider=context.provider,
        models=context.models,
        reviewed_task_registry=context.reviewed_task_registry,
        reviewed_tool_registry=context.reviewed_tool_registry,
        reviewed_vqa_registry=context.reviewed_vqa_registry,
        registration_identity=registration_identity,
        runtime_target=context.initial_target,
        history_database=context.history_database,
        history_retrieval=context.history_retrieval,
        history_context_count=len(context.history_context),
    )


def _display_path(path: Any, *, repo_root: Any) -> str | None:
    if path is None:
        return None
    return (
        str(path.relative_to(repo_root))
        if path.is_relative_to(repo_root)
        else str(path)
    )


__all__ = ["finalize_manifest_and_dispatch"]
