"""Query interpretation and executable policy-task binding."""

from __future__ import annotations

from typing import Any, Mapping

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
from mea.planner.context import build_planning_context
from mea.planner.open_task_resolver import (
    PlanAgentQueryInterpreter,
    discover_robotwin_runtime_task_inventory,
    resolve_open_task,
)
from mea.planner.query_interpretation import resolve_concern_candidate_domain
from mea.providers import OpenAICompatibleProvider


def _runtime_discovery(context: AgentRunContext) -> dict[str, Any]:
    args = context.args
    inventory = discover_robotwin_runtime_task_inventory(
        context.repo_root,
        capability_catalog=None,
        schema_backed_only=(args.policy_backend == "act"),
    )
    discovery = discover_ready_plan_agent_targets(
        context.repo_root,
        inventory,
        max_rounds=(
            int(args.max_agent_rounds)
            if args.max_agent_rounds is not None
            else max(2, int(args.generated_rounds))
        ),
        policy_spec=context.runtime_policy_spec,
    )
    return {**discovery, "inventory": inventory}


def _global_history(
    context: AgentRunContext,
    *,
    ready_tasks: list[str],
) -> dict[str, Any]:
    args = context.args
    if args.no_history:
        return {"schema_version": 1, "status": "disabled", "candidates": []}
    try:
        database = EvaluationHistoryDB(
            context.history_path,
            repo_root=context.repo_root,
        )
        result = database.retrieve_similar_global(
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
        result["status"] = "passed"
        return result
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "failed",
            "candidates": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def route_agent_query(context: AgentRunContext) -> dict[str, Any] | None:
    """Interpret the Query, then bind one executable task/checkpoint.

    An unsupported Query returns a completed no-execution result.  The task
    inventory is used only after semantic interpretation and never supplies an
    aspect/template menu to the Plan Agent.
    """

    args = context.args
    discovery = _runtime_discovery(context)
    inventory = discovery["inventory"]
    runtime_targets = discovery["targets"]
    excluded = discovery["excluded"]
    ready_tasks = sorted(runtime_targets)
    if not ready_tasks:
        raise SystemExit(
            "no source/checkpoint-ready task is available for "
            f"{args.policy_backend}"
        )

    if context.bound_plan_only:
        assert args.bound_task_name is not None
        if args.bound_task_name not in ready_tasks:
            raise SystemExit(
                "bound task has no source/checkpoint runtime binding for "
                f"{args.policy_backend}: {args.bound_task_name!r}"
            )
        args.task_name = args.bound_task_name
        context.open_task_inventory = inventory
        context.runtime_plan_agent_targets = runtime_targets
        context.runtime_binding_excluded = excluded
        return None

    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=context.models["planner"],
        vision_model=context.models["vision"],
        timeout=180.0,
    )
    provider.max_retries = 0
    initially_bound_task = args.bound_task_name
    if initially_bound_task is not None and initially_bound_task not in ready_tasks:
        raise SystemExit(
            f"bound task is not {args.policy_backend}-ready: "
            f"{initially_bound_task!r}"
        )

    planning_contexts = {
        task_name: build_planning_context(
            context.repo_root,
            runtime_targets[task_name],
        )
        for task_name in ready_tasks
    }
    concern_policy_card = (
        planning_contexts[initially_bound_task]["policy_card"]
        if initially_bound_task is not None
        else build_pending_task_binding_policy_card(context.runtime_policy_spec)
    )
    interpreter = PlanAgentQueryInterpreter(
        provider,
        model=context.models["planner"],
        max_attempts=2,
    )
    bundle = interpreter.propose(
        args.request,
        policy_card=concern_policy_card,
    )

    checkpoint_binding = None
    if initially_bound_task is None:
        checkpoint_binding = bind_ready_task_after_query_interpretation(
            bundle["concern"],
            inventory=inventory,
            ready_task_names=ready_tasks,
            default_task_name=str(args.task_name),
        )
        args.bound_task_name = checkpoint_binding["selected_task_name"]
        if args.bound_task_name is None:
            unresolved = {
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
                catalog=None,
                concern_bundle=bundle,
                task_inventory=inventory,
                task_resolution=unresolved,
                concern_agent=interpreter,
            )

    assert args.bound_task_name is not None
    resolution_inventory = (
        [item for item in inventory if item["task_name"] in ready_tasks]
        if checkpoint_binding is not None
        else inventory
    )
    task_resolution = resolve_open_task(
        bundle["concern"],
        policy_card=planning_contexts[args.bound_task_name]["policy_card"],
        inventory=resolution_inventory,
        can_generate_new_task=False,
    )
    resolved_base = task_resolution.get("selected_base_task")
    if (
        initially_bound_task is not None
        and isinstance(resolved_base, Mapping)
        and resolved_base.get("task_name") not in {None, args.bound_task_name}
    ):
        raise SystemExit(
            "Query interpretation selected a different task than the explicit "
            f"policy binding: {resolved_base.get('task_name')!r} != "
            f"{args.bound_task_name!r}"
        )
    if (
        checkpoint_binding is not None
        and checkpoint_binding["fallback_used"]
        and task_resolution["reason_code"] == "no_semantic_task_match"
    ):
        selected = next(
            item for item in inventory
            if item["task_name"] == args.bound_task_name
        )
        task_resolution.update(
            {
                "decision": "retrieve_and_adapt",
                "reason_code": checkpoint_binding["reason_code"],
                "selected_base_task": {
                    "task_name": selected["task_name"],
                    "score": 0.0,
                    "execution_status": selected["execution_status"],
                    "capability_aspects": list(selected["capability_aspects"]),
                },
            }
        )
        task_resolution["resolution_contract"].update(
            {
                "preserve_base_task_semantics": True,
                "task_underspecified_fallback": True,
            }
        )
    task_resolution["checkpoint_binding"] = checkpoint_binding or {
        "schema_version": 1,
        "selected_task_name": args.bound_task_name,
        "reason_code": "explicit_bound_task",
        "fallback_used": False,
        "catalog_visible_to_concern_model": False,
        "retrieval_field": "explicit_policy_binding",
        "semantic_threshold": 0.2,
        "ranked_ready_tasks": [],
    }
    if task_resolution["decision"] != "retrieve_and_adapt":
        return finish_unsupported_open_task_resolution(
            context.repo_root,
            evaluation_id=args.evaluation_id,
            user_request=args.request,
            catalog=None,
            concern_bundle=bundle,
            task_inventory=inventory,
            task_resolution=task_resolution,
            concern_agent=interpreter,
        )

    candidate_resolution = resolve_concern_candidate_domain(
        bundle["concern"],
        experiment_needs=bundle.get("experiment_needs"),
    )
    candidate_budget = resolve_plan_agent_candidate_budget(
        args.max_agent_rounds,
        candidate_resolution=candidate_resolution,
    )
    if not concern_candidate_domain_is_executable(
        candidate_resolution,
        candidate_budget=candidate_budget,
    ):
        return finish_unsupported_open_task_resolution(
            context.repo_root,
            evaluation_id=args.evaluation_id,
            user_request=args.request,
            catalog=None,
            concern_bundle=bundle,
            task_inventory=inventory,
            task_resolution=task_resolution,
            concern_agent=interpreter,
            candidate_resolution=candidate_resolution,
        )

    route_result, routed = build_bound_plan_agent_handoff(
        None,
        task_name=args.bound_task_name,
        user_request=args.request,
        runtime_target=runtime_targets[args.bound_task_name],
    )
    route_result["task_resolution_scope"] = {
        "mode": (
            "query_first_bound_policy_task"
            if task_resolution["checkpoint_binding"]["reason_code"]
            == "explicit_bound_task"
            else "query_first_then_checkpoint_binding"
        ),
        "artifact": "plan/open_task_resolution.json",
    }
    route_result["runtime_binding_scope"] = {
        "authority": "official_source_policy_checkpoint_with_optional_schema",
        "policy_backend": args.policy_backend,
        "catalog_membership_required": False,
        "ready_task_names": ready_tasks,
        "excluded_task_names": sorted(item["task_name"] for item in excluded),
    }
    args.task_name = routed["task_name"]
    context.provider = provider
    context.runtime_plan_agent_targets = runtime_targets
    context.runtime_binding_excluded = excluded
    context.global_route_result = route_result
    context.global_history_retrieval = _global_history(
        context,
        ready_tasks=ready_tasks,
    )
    context.query_interpreter = interpreter
    context.query_interpretation_bundle = bundle
    context.concern_candidate_resolution = candidate_resolution
    context.open_task_inventory = inventory
    context.open_task_resolution = task_resolution
    return None


__all__ = ["route_agent_query"]
