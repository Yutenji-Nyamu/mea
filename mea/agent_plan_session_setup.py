"""Plan Agent session construction and first Proposal validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Mapping

from mea.agent_cli import resolve_plan_agent_allowed_aspects
from mea.agent_run_context import AgentRunContext
from mea.plan_artifacts import PLAN_AGENT_CAPABILITIES
from mea.plan_agent_application import update_manifest
from mea.plan_agent_bootstrap import persist_query_contract
from mea.planner import PlanAgent, PlanAgentSession, project_open_query_capabilities


def should_enable_adaptive_plan_step(
    *,
    fixed_click_bell: bool,
    legacy_click_bell: bool,
    registered_strategy: str | None,
) -> bool:
    return (
        not fixed_click_bell
        and not legacy_click_bell
        and registered_strategy != "fixed_predeclared_v1"
    )


def prepare_plan_session(
    context: AgentRunContext,
    *,
    write_json: Callable[[Any, Any], None],
) -> None:
    """Normalize the initial plan and create its evidence-aware session."""

    args = context.args
    if context.plan is None or context.manifest is None:
        raise RuntimeError("initial plan must exist before session setup")
    if context.evaluation_dir is None:
        raise RuntimeError("evaluation directory was not initialized")

    plan = context.plan
    manifest = context.manifest
    plan_session = None
    plan_session_path = None
    evaluation_target = None
    planning_context = None
    proposal_agent = None
    adaptive_step_agent = None
    plan_agent = None
    plan_agent_capabilities = None

    if context.global_catalog is not None:
        initial_failure_stage = "initial_plan_session_validation"
        try:
            raw_round_budget = plan.get("max_rounds")
            if (
                isinstance(raw_round_budget, bool)
                or not isinstance(raw_round_budget, int)
                or raw_round_budget < 1
            ):
                raise ValueError("planner max_rounds must be a positive integer")
            if context.plan_agent_mode:
                if context.round_budget is None:
                    raise RuntimeError(
                        "Plan Agent open-world budget was not initialized"
                    )
                effective_round_budget = context.round_budget
                plan["max_rounds"] = effective_round_budget
                if context.initial_target is None:
                    raise RuntimeError(
                        "Plan Agent runtime target was not initialized"
                    )
                explicit_candidate_aspect_ids = (
                    resolve_plan_agent_allowed_aspects(
                        args.bound_requested_aspect_ids
                    )
                )
                plan_session = PlanAgentSession(
                    args.request,
                    context.initial_target,
                    query_contract=context.query_sufficiency_contract,
                    candidate_aspect_ids=explicit_candidate_aspect_ids,
                    require_control_anchor=context.control_required,
                    control_round=(
                        plan["rounds"][0]
                        if context.control_required
                        else None
                    ),
                )
                if context.frozen_first_candidate is not None:
                    context.frozen_first_candidate = (
                        plan_session.register_frozen_candidate(
                            context.frozen_first_candidate
                        )
                    )
            else:
                assert context.compat_bounded_proposals is not None
                effective_round_budget = raw_round_budget
                if args.max_agent_rounds is not None:
                    effective_round_budget = min(
                        effective_round_budget,
                        int(args.max_agent_rounds),
                    )
                    plan["max_rounds"] = effective_round_budget
                plan_session = (
                    context.compat_bounded_proposals.create_bound_task_plan_session(
                        context.global_catalog,
                        args.task_name,
                        max_rounds=effective_round_budget,
                    )
                )
            plan = plan_session.normalize_plan(plan)
            planning_context = plan_session.planning_context(context.repo_root)
            write_json(
                context.evaluation_dir / "plan/planning_context.json",
                planning_context,
            )
            if context.plan_agent_mode:
                assert isinstance(plan_session, PlanAgentSession)
                plan_agent_capabilities = project_open_query_capabilities(
                    planning_context,
                    allowed_aspect_ids=explicit_candidate_aspect_ids,
                )
                if (
                    isinstance(context.query_interpretation_bundle, Mapping)
                    and isinstance(
                        context.query_interpretation_bundle.get("concern"),
                        Mapping,
                    )
                ):
                    if not isinstance(
                        context.concern_candidate_resolution,
                        Mapping,
                    ):
                        raise RuntimeError(
                            "online Query-interpretation candidate domain was "
                            "not resolved before planning"
                        )
                    write_json(
                        context.evaluation_dir
                        / "plan/concern_candidate_resolution.json",
                        {
                            **context.concern_candidate_resolution,
                            "planner_domain_role": (
                                "routing_and_retrieval_hint_only"
                            ),
                            "planner_domain_restricted": False,
                        },
                    )
                if not args.plan_only:
                    if context.provider is None:
                        raise RuntimeError(
                            "Plan Agent provider was not initialized"
                        )
                    plan_agent = PlanAgent(
                        context.provider,
                        model=context.models["planner"],
                    )
                write_json(
                    context.evaluation_dir / PLAN_AGENT_CAPABILITIES,
                    plan_agent_capabilities,
                )
                plan_session.query_contract = persist_query_contract(
                    context.evaluation_dir,
                    plan,
                    plan_session.query_contract,
                )
                manifest.setdefault("planner", {}).update(
                    {
                        "public_planner": "PlanAgent",
                        "control_anchor_owned_by_runtime": (
                            plan_session.require_control_anchor
                        ),
                        "control_template_id": plan_session.control_template,
                        "catalog_navigation_was_model_visible": False,
                        "global_router_scope": "task_and_checkpoint_only",
                        "aspect_selection_owner": "PlanAgent",
                        "candidate_domain_source": (
                            "explicit_user_binding"
                            if explicit_candidate_aspect_ids is not None
                            else (
                                "full_retrieval_inventory_plus_open_generation"
                            )
                        ),
                        "pre_control_concern_restricts_planner_domain": False,
                        "concern_candidate_resolution_path": (
                            "plan/concern_candidate_resolution.json"
                            if context.concern_candidate_resolution is not None
                            else None
                        ),
                    }
                )
            elif should_enable_adaptive_plan_step(
                fixed_click_bell=context.fixed_click_bell,
                legacy_click_bell=context.legacy_click_bell,
                registered_strategy=args.registered_strategy,
            ):
                if context.provider is None:
                    raise RuntimeError("compat provider was not initialized")
                assert context.compat_bounded_proposals is not None
                adaptive_step_agent = (
                    context.compat_bounded_proposals.create_adaptive_plan_step_agent(
                        context.provider,
                        model=context.models["planner"],
                    )
                )
            if args.proposal_mode != "catalog":
                if context.provider is None:
                    raise RuntimeError("compat provider was not initialized")
                assert context.compat_bounded_proposals is not None
                proposal_agent = (
                    context.compat_bounded_proposals.create_bounded_proposal_agent(
                        context.provider,
                        model=context.models["taskgen"],
                    )
                )
                first_round = plan["rounds"][0]
                first_aspect = str(first_round["task_proposal"]["aspect_id"])
                if args.proposal_mode == "novel_first_round" and (
                    args.task_name != "click_bell"
                    or first_aspect != "object_position"
                ):
                    raise ValueError(
                        "novel_first_round currently supports the bounded "
                        "click_bell object_position capability only"
                    )
                initial_failure_stage = "initial_bounded_proposal"
                plan["rounds"][0], proposal_bundle = (
                    context.compat_bounded_proposals.apply_bounded_round_proposal(
                        proposal_agent=proposal_agent,
                        user_query=args.request,
                        target=plan_session.target,
                        planning_context=planning_context,
                        round_plan=first_round,
                        evaluation_dir=context.evaluation_dir,
                        round_number=1,
                    )
                )
                plan = plan_session.normalize_plan(plan)
                manifest.setdefault("planner", {}).update(
                    {
                        "round_1_task_tool_proposal_source": "bounded_model",
                        "round_1_proposal_mode": args.proposal_mode,
                        "round_1_proposal_path": (
                            "plan/bounded_proposal/proposal_bundle.json"
                        ),
                        "round_1_proposal_capability_mode": proposal_bundle[
                            "proposal_capability_mode"
                        ],
                    }
                )
            manifest["plan"] = plan
            session_snapshot = plan_session.snapshot(args.request, plan)
        except context.compat_plan_errors as exc:
            manifest_path = context.evaluation_dir / "manifest.json"
            if manifest_path.is_file():
                update_manifest(
                    context.evaluation_dir,
                    status="failed",
                    lifecycle_status="failed",
                    failure_stage=initial_failure_stage,
                    completed_rounds=0,
                    active_child_run_id=None,
                    execution_finished_at=(
                        datetime.now().astimezone().isoformat()
                    ),
                    failure={"type": type(exc).__name__, "message": str(exc)},
                )
            raise RuntimeError(
                f"bound PlanSession validation failed: {exc}"
            ) from exc
        plan_session_path = "plan/bound_task_session.json"
        evaluation_target = session_snapshot["target"]
        write_json(context.evaluation_dir / "plan/evaluation_plan.json", plan)
        write_json(
            context.evaluation_dir / plan_session_path,
            session_snapshot,
        )
        update_manifest(
            context.evaluation_dir,
            plan=plan,
            planner=manifest.get("planner"),
            proposal_mode=args.proposal_mode,
            planning_context_path="plan/planning_context.json",
        )

    context.plan = plan
    context.manifest = manifest
    context.plan_session = plan_session
    context.plan_session_path = plan_session_path
    context.evaluation_target = evaluation_target
    context.planning_context = planning_context
    context.proposal_agent = proposal_agent
    context.adaptive_step_agent = adaptive_step_agent
    context.plan_agent = plan_agent
    context.plan_agent_capabilities = plan_agent_capabilities


__all__ = ["prepare_plan_session", "should_enable_adaptive_plan_step"]
