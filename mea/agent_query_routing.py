"""Query interpretation and runtime task binding for the Agent command."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from mea.agent_cli import resolve_plan_agent_candidate_budget
from mea.agent_run_context import AgentRunContext
from mea.history import EvaluationHistoryDB
from mea.plan_agent_bootstrap import (
    bind_ready_task_after_query_interpretation,
    build_bound_plan_agent_handoff,
    build_pending_task_binding_policy_card,
    concern_candidate_domain_is_executable,
    discover_ready_plan_agent_targets,
    finish_unsupported_open_task_resolution,
)
from mea.planner import (
    GlobalQueryRouter,
    PlanAgentQueryInterpreter,
    build_act_catalog,
    build_planning_context,
    resolve_concern_candidate_domain,
    resolve_open_task,
    route_to_planner_proposal,
)
from mea.planner.open_task_resolver import (
    discover_robotwin_runtime_task_inventory,
)
from mea.providers import OpenAICompatibleProvider


def route_agent_query(
    context: AgentRunContext,
    *,
    finish_unsupported_global_route: Callable[..., dict[str, Any]],
) -> dict[str, Any] | None:
    """Interpret the open Query and bind only an executable policy task.

    Returns a completed unsupported result when no execution can be bound;
    otherwise mutates ``context`` with routing artifacts and returns ``None``.
    """

    args = context.args
    global_catalog = context.global_catalog
    provider = context.provider
    open_task_inventory = context.open_task_inventory
    open_task_resolution = context.open_task_resolution
    query_interpreter = context.query_interpreter
    query_interpretation_bundle = context.query_interpretation_bundle
    concern_candidate_resolution = context.concern_candidate_resolution
    runtime_targets = context.runtime_plan_agent_targets
    runtime_binding_excluded = context.runtime_binding_excluded
    routed_task_profile = context.routed_task_profile

    if context.bound_plan_only:
        global_catalog = build_act_catalog(context.repo_root)
        open_task_inventory = discover_robotwin_runtime_task_inventory(
            context.repo_root,
            capability_catalog=global_catalog,
            schema_backed_only=(args.policy_backend == "act"),
        )
        runtime_discovery = discover_ready_plan_agent_targets(
            context.repo_root,
            open_task_inventory,
            max_rounds=(
                int(args.max_agent_rounds)
                if args.max_agent_rounds is not None
                else max(2, int(args.generated_rounds))
            ),
            policy_spec=context.runtime_policy_spec,
        )
        runtime_targets = runtime_discovery["targets"]
        runtime_binding_excluded = runtime_discovery["excluded"]
        ready_tasks = sorted(runtime_targets)
        assert args.bound_task_name is not None
        if args.bound_task_name not in ready_tasks:
            raise SystemExit(
                "bound task has no source/checkpoint runtime binding for "
                f"{args.policy_backend}: {args.bound_task_name!r}"
            )
        args.task_name = args.bound_task_name
        args.task_profile = "official"
        routed_task_profile = args.task_profile

    global_route_result = None
    global_router = None
    global_history_retrieval = context.global_history_retrieval
    validated_proposal = context.validated_proposal

    if args.auto_route:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=context.models["planner"],
            vision_model=context.models["vision"],
            timeout=180.0,
        )
        global_catalog = build_act_catalog(context.repo_root)
        if context.plan_agent_mode:
            open_task_inventory = discover_robotwin_runtime_task_inventory(
                context.repo_root,
                capability_catalog=global_catalog,
                schema_backed_only=(args.policy_backend == "act"),
            )
            runtime_discovery = discover_ready_plan_agent_targets(
                context.repo_root,
                open_task_inventory,
                max_rounds=(
                    int(args.max_agent_rounds)
                    if args.max_agent_rounds is not None
                    else max(2, int(args.generated_rounds))
                ),
                policy_spec=context.runtime_policy_spec,
            )
            runtime_targets = runtime_discovery["targets"]
            runtime_binding_excluded = runtime_discovery["excluded"]
            ready_tasks = sorted(runtime_targets)
        else:
            ready_tasks = [
                task["task_name"] for task in global_catalog.get("tasks", [])
            ]
        if not ready_tasks:
            raise SystemExit(
                "no source/checkpoint-ready task is available for "
                f"{args.policy_backend}"
            )
        if args.bound_task_name is not None and args.bound_task_name not in ready_tasks:
            raise SystemExit(
                f"bound task is not {args.policy_backend}-ready: "
                f"{args.bound_task_name!r}"
            )
        if context.plan_agent_mode:
            global_planning_contexts = {
                task_name: build_planning_context(
                    context.repo_root,
                    runtime_targets[task_name],
                )
                for task_name in ready_tasks
            }
        else:
            assert context.compat_bounded_proposals is not None
            global_planning_contexts = {
                task_name: context.compat_bounded_proposals.create_bound_task_plan_session(
                    global_catalog,
                    task_name,
                ).planning_context(context.repo_root)
                for task_name in ready_tasks
            }

        if context.plan_agent_mode:
            provider.max_retries = 0
            initially_bound_task = args.bound_task_name
            concern_policy_card = (
                global_planning_contexts[initially_bound_task]["policy_card"]
                if initially_bound_task is not None
                else build_pending_task_binding_policy_card(
                    context.runtime_policy_spec
                )
            )
            query_interpreter = PlanAgentQueryInterpreter(
                provider,
                model=context.models["planner"],
                max_attempts=2,
            )
            query_interpretation_bundle = query_interpreter.propose(
                args.request,
                policy_card=concern_policy_card,
            )
            assert open_task_inventory is not None
            checkpoint_binding = None
            if initially_bound_task is None:
                checkpoint_binding = bind_ready_task_after_query_interpretation(
                    query_interpretation_bundle["concern"],
                    inventory=open_task_inventory,
                    ready_task_names=[str(item) for item in ready_tasks],
                    default_task_name=str(args.task_name),
                )
                if checkpoint_binding["selected_task_name"] is None:
                    unresolved_task = {
                        "schema_version": 1,
                        "decision": "unsupported",
                        "resolved_task_name": None,
                        "reason_code": checkpoint_binding["reason_code"],
                        "checkpoint_binding": checkpoint_binding,
                    }
                    return finish_unsupported_open_task_resolution(
                        context.repo_root,
                        evaluation_id=args.evaluation_id,
                        user_request=args.request,
                        catalog=global_catalog,
                        concern_bundle=query_interpretation_bundle,
                        task_inventory=open_task_inventory,
                        task_resolution=unresolved_task,
                        concern_agent=query_interpreter,
                    )
                args.bound_task_name = checkpoint_binding["selected_task_name"]
            assert args.bound_task_name is not None
            bound_policy_card = global_planning_contexts[
                args.bound_task_name
            ]["policy_card"]
            resolution_inventory = (
                [
                    item
                    for item in open_task_inventory
                    if item["task_name"] in ready_tasks
                ]
                if checkpoint_binding is not None
                else open_task_inventory
            )
            open_task_resolution = resolve_open_task(
                query_interpretation_bundle["concern"],
                policy_card=bound_policy_card,
                inventory=resolution_inventory,
                can_generate_new_task=False,
            )
            if (
                checkpoint_binding is not None
                and checkpoint_binding["fallback_used"]
                and open_task_resolution["reason_code"]
                == "no_semantic_task_match"
            ):
                selected_inventory = next(
                    item
                    for item in open_task_inventory
                    if item["task_name"] == args.bound_task_name
                )
                open_task_resolution["decision"] = "retrieve_and_adapt"
                open_task_resolution["reason_code"] = checkpoint_binding[
                    "reason_code"
                ]
                open_task_resolution["selected_base_task"] = {
                    "task_name": selected_inventory["task_name"],
                    "score": 0.0,
                    "execution_status": selected_inventory["execution_status"],
                    "capability_aspects": list(
                        selected_inventory["capability_aspects"]
                    ),
                }
                open_task_resolution["resolution_contract"][
                    "preserve_base_task_semantics"
                ] = True
                open_task_resolution["resolution_contract"][
                    "task_underspecified_fallback"
                ] = True
            open_task_resolution["checkpoint_binding"] = (
                checkpoint_binding
                if checkpoint_binding is not None
                else {
                    "schema_version": 1,
                    "selected_task_name": args.bound_task_name,
                    "reason_code": "explicit_bound_task",
                    "fallback_used": False,
                    "catalog_visible_to_concern_model": False,
                    "retrieval_field": "explicit_policy_binding",
                    "semantic_threshold": 0.2,
                    "ranked_ready_tasks": [],
                }
            )
            if open_task_resolution["decision"] != "retrieve_and_adapt":
                return finish_unsupported_open_task_resolution(
                    context.repo_root,
                    evaluation_id=args.evaluation_id,
                    user_request=args.request,
                    catalog=global_catalog,
                    concern_bundle=query_interpretation_bundle,
                    task_inventory=open_task_inventory,
                    task_resolution=open_task_resolution,
                    concern_agent=query_interpreter,
                )
            concern_candidate_resolution = resolve_concern_candidate_domain(
                query_interpretation_bundle["concern"],
                target=runtime_targets[args.bound_task_name],
                experiment_needs=query_interpretation_bundle.get(
                    "experiment_needs"
                ),
            )
            semantic_context_for_budget = (
                query_interpretation_bundle.get("concern")
                if isinstance(query_interpretation_bundle, Mapping)
                and isinstance(
                    query_interpretation_bundle.get("concern"), Mapping
                )
                else None
            )
            candidate_budget = resolve_plan_agent_candidate_budget(
                args.max_agent_rounds,
                user_request=args.request,
                query_contract=context.query_sufficiency_contract,
                semantic_context=semantic_context_for_budget,
                candidate_resolution=concern_candidate_resolution,
            )
            if not concern_candidate_domain_is_executable(
                concern_candidate_resolution,
                candidate_budget=candidate_budget,
            ):
                return finish_unsupported_open_task_resolution(
                    context.repo_root,
                    evaluation_id=args.evaluation_id,
                    user_request=args.request,
                    catalog=global_catalog,
                    concern_bundle=query_interpretation_bundle,
                    task_inventory=open_task_inventory,
                    task_resolution=open_task_resolution,
                    concern_agent=query_interpreter,
                    candidate_resolution=concern_candidate_resolution,
                )

        global_history_context: list[dict[str, Any]] = []
        if not args.no_history:
            try:
                global_history_db = EvaluationHistoryDB(
                    context.history_path,
                    repo_root=context.repo_root,
                )
                global_history_retrieval = (
                    global_history_db.retrieve_similar_global(
                        args.request,
                        allowed_task_names=ready_tasks,
                        policy_name=(
                            context.runtime_policy_spec.policy_name
                            if context.runtime_policy_spec is not None
                            else "ACT"
                        ),
                        checkpoint_setting=(
                            str(
                                context.runtime_policy_spec.metadata.get(
                                    "checkpoint_setting"
                                )
                            )
                            if context.runtime_policy_spec is not None
                            else "demo_clean"
                        ),
                        limit=args.history_limit,
                        exclude_evaluation_id=args.evaluation_id,
                    )
                )
                global_history_retrieval["status"] = "passed"
                global_history_context = list(
                    global_history_retrieval.get("candidates", [])
                )
            except Exception as exc:
                global_history_retrieval = {
                    "schema_version": 1,
                    "status": "failed",
                    "candidates": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }

        if open_task_resolution is not None:
            assert args.bound_task_name is not None
            global_route_result, routed = build_bound_plan_agent_handoff(
                None,
                task_name=args.bound_task_name,
                user_request=args.request,
                runtime_target=runtime_targets[args.bound_task_name],
            )
            global_route_result["task_resolution_scope"] = {
                "mode": (
                    "query_first_bound_policy_task"
                    if open_task_resolution["checkpoint_binding"]["reason_code"]
                    == "explicit_bound_task"
                    else "query_first_then_checkpoint_binding"
                ),
                "artifact": "plan/open_task_resolution.json",
            }
            global_route_result["runtime_binding_scope"] = {
                "authority": (
                    "official_source_policy_checkpoint_with_optional_schema"
                ),
                "policy_backend": args.policy_backend,
                "catalog_membership_required": False,
                "ready_task_names": sorted(runtime_targets),
                "excluded_task_names": sorted(
                    item["task_name"] for item in runtime_binding_excluded
                ),
            }
        else:
            global_router = GlobalQueryRouter(
                provider,
                model=context.models["planner"],
                catalog=global_catalog,
                planning_contexts=global_planning_contexts,
            )
            global_route_result = global_router.route(
                args.request,
                history_context=global_history_context,
            )
            global_route_result["task_resolution_scope"] = {
                "mode": "checkpoint_portfolio_selection",
                "paper_claim": (
                    "selects among task-specific ACT checkpoints; it is not "
                    "open-task execution by one policy"
                ),
            }
            selection = global_route_result["selection"]
            if selection["decision"] == "unsupported":
                return finish_unsupported_global_route(
                    context.repo_root,
                    evaluation_id=args.evaluation_id,
                    user_request=args.request,
                    catalog=global_catalog,
                    route_result=global_route_result,
                    router=global_router,
                    history_retrieval=global_history_retrieval,
                )
            routed = route_to_planner_proposal(selection, global_catalog)
        args.task_name = routed["task_name"]
        routed_task_profile = routed["task_profile"]
        args.task_profile = (
            "official"
            if context.plan_agent_mode
            else (
                (
                    "fixed_suite"
                    if args.planning_policy == "fixed_predeclared_v1"
                    else "adaptive_properties"
                )
                if args.task_name == "click_bell"
                else "official"
            )
        )
        validated_proposal = routed["proposal"]
        if (
            context.plan_agent_mode
            and args.task_name not in runtime_targets
        ):
            raise SystemExit(
                "the Plan Agent requires source/schema/checkpoint runtime "
                f"authority; {args.task_name!r} is unavailable"
            )

    context.provider = provider
    context.global_catalog = global_catalog
    context.runtime_plan_agent_targets = runtime_targets
    context.runtime_binding_excluded = runtime_binding_excluded
    context.global_route_result = global_route_result
    context.global_history_retrieval = global_history_retrieval
    context.global_router = global_router
    context.query_interpreter = query_interpreter
    context.query_interpretation_bundle = query_interpretation_bundle
    context.concern_candidate_resolution = concern_candidate_resolution
    context.open_task_inventory = open_task_inventory
    context.open_task_resolution = open_task_resolution
    context.validated_proposal = validated_proposal
    context.routed_task_profile = routed_task_profile
    return None


__all__ = ["route_agent_query"]
