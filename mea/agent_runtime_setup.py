"""Startup, provider, registry, and history setup for the Agent CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mea.agent_cli import load_query_sufficiency_contract
from mea.agent_run_context import AgentRunContext
from mea.history import EvaluationHistoryDB
from mea.planner import make_evaluation_id
from mea.planner.runtime_task_binding import (
    build_hyvla_policy_spec,
    build_smolvla_policy_spec,
)
from mea.providers import OpenAICompatibleProvider, resolve_model_profile


def create_agent_run_context(
    args: Any,
    *,
    plan_agent_mode: bool,
    compat_bounded_proposals: Any | None,
    compat_plan_errors: tuple[type[Exception], ...],
    bound_plan_only: bool,
    observed_argv: list[str],
    initialize_registered_runtime: Callable[..., tuple[Any, Any]],
) -> AgentRunContext:
    """Resolve immutable command configuration before routing the Query."""

    repo_root = args.repo_root.expanduser().resolve()
    if args.policy_backend == "smolvla":
        runtime_policy_spec = build_smolvla_policy_spec(
            args.smolvla_checkpoint.expanduser().resolve()
        )
    elif args.policy_backend == "hyvla":
        runtime_policy_spec = build_hyvla_policy_spec(
            args.hyvla_checkpoint.expanduser().resolve(),
            source_dir=args.hyvla_source.expanduser().resolve(),
            python_env=args.hyvla_python_env.expanduser().resolve(),
        )
    else:
        runtime_policy_spec = None

    query_sufficiency_contract = None
    if args.query_sufficiency_contract is not None:
        query_sufficiency_contract = load_query_sufficiency_contract(
            args.query_sufficiency_contract
        )
    args.evaluation_id = args.evaluation_id or make_evaluation_id()

    registered_execution = None
    if args.registered_strategy is not None:
        from experiments.paper.registered_execution_adapter import (
            RegisteredExecutionAdapterError,
            load_registered_execution_for_cli,
        )

        try:
            registered_execution = load_registered_execution_for_cli(
                repo_root,
                evidence_manifest_path=str(args.evidence_manifest),
                command_plan_path=str(args.command_plan),
                registered_route_path=str(args.registered_route),
                strategy=str(args.registered_strategy),
                evaluation_id=str(args.evaluation_id),
                observed_argv=observed_argv,
            )
        except RegisteredExecutionAdapterError as exc:
            raise SystemExit(
                f"registered execution preflight failed: {exc}"
            ) from exc

    models = resolve_model_profile(
        args.model_profile,
        {
            "planner": args.planner_model,
            "taskgen": args.taskgen_model,
            "toolgen": args.toolgen_model,
            "vision": args.vision_model,
            "feedback": args.feedback_model,
        },
    )
    history_path = (
        args.history_database.expanduser().resolve()
        if args.history_database
        else repo_root / "mea/evaluation_runs/history.sqlite3"
    )
    reviewed_task_registry = (
        args.reviewed_task_registry.expanduser().resolve()
        if args.reviewed_task_registry is not None
        else None
    )
    reviewed_tool_registry = (
        args.reviewed_tool_registry.expanduser().resolve()
        if args.reviewed_tool_registry is not None
        else None
    )
    reviewed_vqa_registry = (
        args.reviewed_vqa_registry.expanduser().resolve()
        if args.reviewed_vqa_registry is not None
        else None
    )

    global_catalog = None
    provider = None
    if registered_execution is not None:
        global_catalog, provider = initialize_registered_runtime(
            repo_root,
            global_catalog,
            provider,
            registered_strategy=args.registered_strategy,
            base_url=args.base_url,
            text_model=models["planner"],
            vision_model=models["vision"],
        )

    return AgentRunContext(
        args=args,
        plan_agent_mode=plan_agent_mode,
        compat_bounded_proposals=compat_bounded_proposals,
        compat_plan_errors=compat_plan_errors,
        bound_plan_only=bound_plan_only,
        repo_root=repo_root,
        runtime_policy_spec=runtime_policy_spec,
        query_sufficiency_contract=query_sufficiency_contract,
        registered_execution=registered_execution,
        models=models,
        history_path=history_path,
        reviewed_task_registry=reviewed_task_registry,
        reviewed_tool_registry=reviewed_tool_registry,
        reviewed_vqa_registry=reviewed_vqa_registry,
        provider=provider,
        global_catalog=global_catalog,
        global_history_retrieval={
            "schema_version": 1,
            "status": "disabled" if args.no_history else "empty",
            "candidates": [],
        },
        validated_proposal=(
            registered_execution["validated_proposal"]
            if registered_execution is not None
            else None
        ),
        routed_task_profile=(
            "adaptive_properties" if registered_execution is not None else None
        ),
    )


def prepare_task_runtime_and_history(context: AgentRunContext) -> None:
    """Bind the task runtime, provider, planner, and task-local history."""

    args = context.args
    provider = context.provider

    if (
        not context.plan_agent_mode
        and (
            args.task_profile != "official"
            or args.task_name == "beat_block_hammer"
        )
    ):
        from experiments.paper.compat_agent_profile import (
            CompatAgentProfileError,
            resolve_task_specific_runtime_profile,
        )

        try:
            task_runtime_profile = resolve_task_specific_runtime_profile(
                args,
                claim_first_mode=context.plan_agent_mode,
            )
        except CompatAgentProfileError as exc:
            raise SystemExit(str(exc)) from exc
        context.legacy_click_bell = task_runtime_profile["legacy_click_bell"]
        context.adaptive_click_bell = task_runtime_profile[
            "adaptive_click_bell"
        ]
        context.fixed_click_bell = task_runtime_profile["fixed_click_bell"]
        context.bounded_click_bell = task_runtime_profile[
            "bounded_click_bell"
        ]
        context.execution_backend = task_runtime_profile["execution_backend"]
    else:
        context.execution_backend = (
            "act"
            if context.plan_agent_mode
            else (args.execution_backend or "expert")
        )

    if provider is None and (
        args.task_name == "beat_block_hammer"
        or context.adaptive_click_bell
        or context.fixed_click_bell
        or not args.plan_only
    ) and not context.bound_plan_only:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=context.models["planner"],
            vision_model=context.models["vision"],
            timeout=180.0,
        )
    context.provider = provider

    if not context.plan_agent_mode:
        from experiments.paper.legacy_planner_factory import build_legacy_planner

        context.planner = build_legacy_planner(
            context.repo_root,
            task_name=args.task_name,
            task_profile=args.task_profile,
            provider=provider,
            model=context.models["planner"],
            task_module=args.task_module,
            start_seed=args.start_seed,
            num_episodes=args.num_episodes,
            telemetry_profile=args.telemetry_profile,
            max_rounds=args.generated_rounds,
            execution_backend=context.execution_backend,
        )

    history_retrieval: dict[str, Any] = {
        "schema_version": 1,
        "status": "disabled" if args.no_history else "empty",
        "candidates": [],
    }
    history_database = None
    history_context: list[dict[str, Any]] = []
    if not args.no_history:
        try:
            history_database = EvaluationHistoryDB(
                context.history_path,
                repo_root=context.repo_root,
            )
            history_retrieval = history_database.retrieve_similar(
                args.request,
                task_name=args.task_name,
                policy_name=(
                    context.runtime_policy_spec.policy_name
                    if context.runtime_policy_spec is not None
                    else (
                        "ACT"
                        if context.execution_backend in {"act", "both"}
                        else "expert"
                    )
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
                requested_aspect_ids=(
                    context.validated_proposal.get("requested_aspect_ids")
                    if context.validated_proposal is not None
                    else None
                ),
                limit=args.history_limit,
                exclude_evaluation_id=args.evaluation_id,
            )
            history_retrieval["status"] = "passed"
            history_context = list(history_retrieval.get("candidates", []))
        except Exception as exc:
            history_retrieval = {
                "schema_version": 1,
                "status": "failed",
                "candidates": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    planner_kwargs: dict[str, Any] = {
        "evaluation_id": args.evaluation_id,
        "history_context": history_context,
        "history_metadata": {
            key: value
            for key, value in history_retrieval.items()
            if key != "candidates"
        },
    }
    if context.validated_proposal is not None:
        planner_kwargs["validated_proposal"] = context.validated_proposal
    context.history_database = history_database
    context.history_context = history_context
    context.history_retrieval = history_retrieval
    context.planner_kwargs = planner_kwargs


__all__ = ["create_agent_run_context", "prepare_task_runtime_and_history"]
