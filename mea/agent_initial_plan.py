"""Initial Plan Agent proposal, runtime limits, and plan materialization."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from mea.agent_cli import resolve_plan_agent_control_required
from mea.agent_run_context import AgentRunContext
from mea.plan_artifacts import QUERY_INTERPRETATION
from mea.plan_agent_application import update_manifest
from mea.plan_agent_bootstrap import write_open_task_resolution_trace
from mea.planner import (
    PlanAgentInitialPlanBuilder,
    build_dynamic_experiment_candidate,
    build_initial_semantic_proposal_bundle,
    evaluation_intent_from_query_interpretation,
)
from mea.planner.experiment_candidate import build_experiment_candidate
from mea.planner.runtime_limits import build_plan_runtime_limits
from mea.planner.runtime_task_binding import (
    build_runtime_open_world_evaluation_target,
)
from mea.taskgen import round_materialization as taskgen_round_materialization


def prepare_initial_plan(
    context: AgentRunContext,
    *,
    write_json: Callable[[Any, Any], None],
    write_global_route_trace: Callable[..., None],
) -> None:
    """Build the first executable plan without a catalog-authored concern."""

    args = context.args
    runtime_limits = context.plan_runtime_limits
    semantic_bundle = None
    frozen_candidate = None
    initial_candidate = None
    evaluation_intent = None
    direct_single_candidate_query = False
    round_budget = None
    control_required = True
    initial_target = None

    if not context.runtime_plan_agent_targets:
        raise RuntimeError(
            "Plan Agent runtime requires source/checkpoint-ready bindings"
        )
    semantic_context = (
        context.query_interpretation_bundle.get("concern")
        if isinstance(context.query_interpretation_bundle, Mapping)
        and isinstance(
            context.query_interpretation_bundle.get("concern"), Mapping
        )
        else None
    )
    if semantic_context is not None:
        evaluation_intent = evaluation_intent_from_query_interpretation(
            semantic_context
        )
        raw_experiment_needs = (
            context.query_interpretation_bundle.get("experiment_needs")
            if isinstance(context.query_interpretation_bundle, Mapping)
            else None
        )
        if isinstance(raw_experiment_needs, Mapping):
            semantic_bundle = build_initial_semantic_proposal_bundle(
                user_query=args.request,
                concern=semantic_context,
                experiment_needs=raw_experiment_needs,
                evaluation_intent=evaluation_intent,
                provider_record=context.query_interpretation_bundle.get(
                    "provider"
                ),
            )
    direct_single_candidate_query = bool(
        isinstance(context.concern_candidate_resolution, Mapping)
        and context.concern_candidate_resolution.get("resolution")
        == "official_execution_from_typed_needs"
        and context.concern_candidate_resolution.get("execution_authorized")
        is True
    )
    control_required = resolve_plan_agent_control_required(
        candidate_resolution=context.concern_candidate_resolution,
    )
    if not control_required and semantic_bundle is not None:
        frozen_candidate = build_dynamic_experiment_candidate(
            user_query=args.request,
            task_name=args.task_name,
            proposal=semantic_bundle["proposal"],
            evaluation_intent=evaluation_intent,
            candidate_id=f"dynamic.{args.task_name}.round_1",
            official_success_reuse=bool(
                isinstance(context.concern_candidate_resolution, Mapping)
                and context.concern_candidate_resolution.get(
                    "official_success_reuse"
                )
                is True
            ),
        )
    round_budget = (
        int(args.max_agent_rounds)
        if args.max_agent_rounds is not None
        else max(1 + int(control_required), int(args.generated_rounds))
    )
    minimum_rounds = 1 + int(control_required)
    if round_budget < minimum_rounds:
        raise SystemExit(
            "Plan Agent round budget is smaller than the runtime "
            "control plus candidate requirement"
        )
    if runtime_limits is None:
        runtime_limits = build_plan_runtime_limits(
            args.request,
            round_budget=round_budget - int(control_required),
            control_requirement=(
                "required" if control_required else "not_required"
            ),
        )
    if not control_required:
        if semantic_context is None:
            raise SystemExit(
                "a no-control Plan Agent run requires online Query "
                "interpretation"
            )
        if semantic_bundle is not None:
            assert frozen_candidate is not None
            initial_candidate = frozen_candidate
        else:
            initial_candidate = build_experiment_candidate(
                source_query=args.request,
                base_task=args.task_name,
                semantic_concern=(
                    f"{semantic_context['sub_aspect']}: "
                    f"{semantic_context['hypothesis']}"
                ),
                scene_need=None,
                checker_need=None,
                tool_need={
                    "kind": "measure",
                    "description": semantic_context["measurement_need"],
                    "reuse_first": True,
                },
                candidate_id=f"dynamic.{args.task_name}.round_1",
                evaluation_intent=evaluation_intent,
            )
    initial_target = build_runtime_open_world_evaluation_target(
        context.repo_root,
        args.task_name,
        max_rounds=round_budget,
        task_module=args.task_module,
        policy_spec=context.runtime_policy_spec,
    )

    if initial_target is None:
        raise RuntimeError("Plan Agent runtime target was not initialized")
    initial_builder = PlanAgentInitialPlanBuilder(
        context.repo_root,
        target=initial_target,
        max_rounds=int(round_budget),
        start_seed=(
            args.start_seed if args.start_seed is not None else 100000
        ),
        num_episodes=args.num_episodes,
        execution_backend=context.execution_backend,
        task_module=args.task_module,
        telemetry_profile=args.telemetry_profile,
    )
    manifest = initial_builder.plan(
        args.request,
        evaluation_id=str(args.evaluation_id),
        control_required=control_required,
        runtime_limits=runtime_limits,
        history_context=context.history_context,
        history_metadata=context.planner_kwargs["history_metadata"],
    )

    evaluation_id = manifest["evaluation_id"]
    evaluation_dir = (
        context.repo_root / "mea/evaluation_runs" / evaluation_id
    )
    if context.global_route_result is not None:
        write_global_route_trace(
            evaluation_dir,
            route_result=context.global_route_result,
            history_retrieval=context.global_history_retrieval,
        )
        route_manifest = {
            "global_query_route_path": "plan/global_query_route.json",
            "global_route_selection": context.global_route_result["selection"],
            "task_resolution_scope": context.global_route_result[
                "task_resolution_scope"
            ],
        }
        update_manifest(evaluation_dir, **route_manifest)
        if (
            context.query_interpreter is not None
            and context.query_interpretation_bundle is not None
            and context.open_task_inventory is not None
            and context.open_task_resolution is not None
        ):
            write_open_task_resolution_trace(
                evaluation_dir,
                concern_bundle=context.query_interpretation_bundle,
                task_inventory=context.open_task_inventory,
                task_resolution=context.open_task_resolution,
                concern_agent=context.query_interpreter,
            )
            update_manifest(
                evaluation_dir,
                query_interpretation_path=QUERY_INTERPRETATION.as_posix(),
                open_task_resolution_path="plan/open_task_resolution.json",
                robotwin_task_inventory_path=(
                    "plan/robotwin_task_inventory.json"
                ),
            )

    plan = manifest["plan"]
    if initial_candidate is not None:
        if (
            not isinstance(plan.get("rounds"), list)
            or not isinstance(
                manifest.get("initial_execution_binding"), Mapping
            )
        ):
            raise RuntimeError(
                "Plan Agent materializer did not provide an execution binding"
            )
        initial_round, initial_tool_bundle = (
            taskgen_round_materialization.materialize_open_world_round(
                context.repo_root,
                evaluation_dir=evaluation_dir,
                round_number=1,
                candidate=initial_candidate,
                control_execution=manifest["initial_execution_binding"],
                policy_backend=args.policy_backend,
            )
        )
        plan["rounds"] = [initial_round]
        plan["runtime_limits"] = deepcopy(runtime_limits)
        plan["planning_state"] = (
            "initial_query_derived_candidate_materialized"
        )
        write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
        update_manifest(
            evaluation_dir,
            status=plan["planning_state"],
            plan=plan,
            initial_candidate_source="online_query_interpretation_no_control",
            initial_toolgen_route=initial_tool_bundle["source"],
        )

    context.plan_runtime_limits = runtime_limits
    context.evaluation_intent = evaluation_intent
    context.initial_semantic_bundle = semantic_bundle
    context.frozen_first_candidate = frozen_candidate
    context.initial_open_candidate = initial_candidate
    context.direct_single_candidate_query = direct_single_candidate_query
    context.round_budget = round_budget
    context.control_required = control_required
    context.initial_target = initial_target
    context.manifest = manifest
    context.evaluation_id = evaluation_id
    context.evaluation_dir = evaluation_dir
    context.plan = plan


__all__ = ["prepare_initial_plan"]
