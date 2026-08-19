"""CLI dispatcher for the paper-aligned ManipEvalAgent method."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from mea.agent_cli import (
    parse_args,
    resolve_plan_agent_candidate_budget,
    resolve_plan_agent_control_required,
    validate_and_normalize_agent_args,
)
from mea.agent_initial_plan import prepare_initial_plan
from mea.agent_plan_session_setup import (
    prepare_plan_session,
)
from mea.agent_query_routing import route_agent_query
from mea.agent_run_dispatch import finalize_manifest_and_dispatch
from mea.agent_runtime_setup import (
    create_agent_run_context,
    prepare_task_runtime_and_history,
)
from mea.plan_agent_application import update_manifest
from mea.plan_agent_bootstrap import (
    bind_ready_task_after_query_interpretation,
    build_bound_plan_agent_handoff,
    build_pending_task_binding_policy_card,
    concern_candidate_domain_is_executable,
    discover_ready_plan_agent_targets,
    finish_unsupported_open_task_resolution,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def bound_target_task_name(target: Mapping[str, Any]) -> str:
    """Read a legacy target or the production PolicyTaskBinding identity."""

    task_name = target.get("task_name")
    if not isinstance(task_name, str) or not task_name.strip():
        binding = target.get("policy_task_binding")
        task_name = (
            binding.get("task_name") if isinstance(binding, Mapping) else None
        )
    if not isinstance(task_name, str) or not task_name.strip():
        raise RuntimeError("evaluation target has no bound task identity")
    return task_name.strip()


def write_global_route_trace(
    evaluation_dir: Path,
    *,
    route_result: dict[str, Any],
    history_retrieval: dict[str, Any],
) -> None:
    """Persist the Query-first task/checkpoint binding trace."""

    write_json(
        evaluation_dir / "plan/global_query_route.json",
        {**route_result, "history_retrieval": history_retrieval},
    )
def main(args: Any | None = None) -> None:
    """Parse, validate, and dispatch one Agent evaluation."""

    args = args or parse_args()
    if args.benchmark == "libero":
        from mea.libero.chain import run_libero_agent_cli

        run_libero_agent_cli(args)
        return

    bound_plan_only = validate_and_normalize_agent_args(args)
    context = create_agent_run_context(
        args,
        bound_plan_only=bound_plan_only,
    )
    unsupported = route_agent_query(context)
    if unsupported is not None:
        print(json.dumps(unsupported, ensure_ascii=False, indent=2))
        return
    prepare_task_runtime_and_history(context)
    prepare_initial_plan(
        context,
        write_json=write_json,
        write_global_route_trace=write_global_route_trace,
    )
    prepare_plan_session(context, write_json=write_json)
    finalize_manifest_and_dispatch(
        context,
        write_json=write_json,
        write_global_route_trace=write_global_route_trace,
        bound_target_task_name=bound_target_task_name,
    )


if __name__ == "__main__":
    main()
