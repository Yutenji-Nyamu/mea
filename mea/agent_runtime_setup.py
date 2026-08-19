"""Startup, provider, registry, and history setup for the Agent CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mea.agent_run_context import AgentRunContext
from mea.evaluation_identity import make_evaluation_id
from mea.history import EvaluationHistoryDB
from mea.planner.runtime_task_binding import (
    build_hyvla_policy_spec,
    build_smolvla_policy_spec,
)
from mea.providers import OpenAICompatibleProvider, resolve_model_profile


def create_agent_run_context(
    args: Any,
    *,
    bound_plan_only: bool,
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

    args.evaluation_id = args.evaluation_id or make_evaluation_id()

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
    return AgentRunContext(
        args=args,
        bound_plan_only=bound_plan_only,
        repo_root=repo_root,
        runtime_policy_spec=runtime_policy_spec,
        plan_runtime_limits=None,
        models=models,
        history_path=history_path,
        provider=None,
        global_history_retrieval={
            "schema_version": 1,
            "status": "disabled" if args.no_history else "empty",
            "candidates": [],
        },
    )


def prepare_task_runtime_and_history(context: AgentRunContext) -> None:
    """Bind the task runtime, provider, planner, and task-local history."""

    args = context.args
    provider = context.provider

    context.execution_backend = "act"

    if provider is None and not args.plan_only and not context.bound_plan_only:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=context.models["planner"],
            vision_model=context.models["vision"],
            timeout=180.0,
        )
    context.provider = provider

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
                requested_aspect_ids=None,
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
    context.history_database = history_database
    context.history_context = history_context
    context.history_retrieval = history_retrieval
    context.planner_kwargs = planner_kwargs


__all__ = ["create_agent_run_context", "prepare_task_runtime_and_history"]
