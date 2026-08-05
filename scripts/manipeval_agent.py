"""CLI dispatcher for the paper-aligned ManipEvalAgent method."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from mea.agent_cli import (
    load_query_sufficiency_contract,
    parse_args,
    paper_compat_profile_requested,
    resolve_plan_agent_allowed_aspects,
    resolve_plan_agent_candidate_budget,
    resolve_plan_agent_control_required,
    resolve_default_open_query_planner,
    validate_and_normalize_agent_args,
)
from mea.agent_initial_plan import prepare_initial_plan
from mea.agent_plan_session_setup import (
    prepare_plan_session,
    should_enable_adaptive_plan_step,
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
    persist_query_contract,
)
from mea.planner import GlobalQueryRouter, build_act_catalog, make_evaluation_id
from mea.providers import OpenAICompatibleProvider


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# Historical import spellings remain readers only; new artifacts use Plan Agent
# terminology.
discover_ready_claim_first_targets = discover_ready_plan_agent_targets
build_bound_claim_first_handoff = build_bound_plan_agent_handoff
bind_ready_task_after_free_concern = (
    bind_ready_task_after_query_interpretation
)


def initialize_registered_dynamic_runtime(
    repo_root: Path,
    existing_catalog: dict[str, Any] | None,
    existing_provider: OpenAICompatibleProvider | None,
    *,
    registered_strategy: str | None,
    base_url: str | None,
    text_model: str,
    vision_model: str,
) -> tuple[dict[str, Any] | None, OpenAICompatibleProvider | None]:
    """Initialize catalog/provider skipped by a registered dynamic route."""

    if registered_strategy != "dynamic_evidence_v1":
        return existing_catalog, existing_provider
    catalog = existing_catalog or build_act_catalog(repo_root)
    provider = existing_provider or OpenAICompatibleProvider(
        base_url=base_url,
        text_model=text_model,
        vision_model=vision_model,
        timeout=180.0,
    )
    return catalog, provider


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
    catalog: dict[str, Any],
    route_result: dict[str, Any],
    router: GlobalQueryRouter | None,
    history_retrieval: dict[str, Any],
) -> None:
    """Persist the bounded global route without leaking credentials."""

    write_json(evaluation_dir / "plan/global_act_catalog.json", catalog)
    write_json(
        evaluation_dir / "plan/global_query_route.json",
        {**route_result, "history_retrieval": history_retrieval},
    )
    if router is not None and router.last_prompt is not None:
        (evaluation_dir / "plan/global_query_prompt.md").write_text(
            router.last_prompt,
            encoding="utf-8",
        )
    for index, response in enumerate(
        router.last_responses if router is not None else [],
        start=1,
    ):
        (evaluation_dir / f"plan/global_query_response_{index}.txt").write_text(
            response + "\n",
            encoding="utf-8",
        )


def finish_unsupported_global_route(
    repo_root: Path,
    *,
    evaluation_id: str | None,
    user_request: str,
    catalog: dict[str, Any],
    route_result: dict[str, Any],
    router: GlobalQueryRouter,
    history_retrieval: dict[str, Any],
) -> dict[str, Any]:
    """Create an auditable no-execution result for an unsupported Query."""

    resolved_id = evaluation_id or make_evaluation_id()
    if not re.fullmatch(r"eval_[A-Za-z0-9_]+", resolved_id):
        raise ValueError("evaluation_id must match eval_[A-Za-z0-9_]+")
    evaluation_dir = repo_root / "mea/evaluation_runs" / resolved_id
    if evaluation_dir.exists():
        raise RuntimeError(f"evaluation directory already exists: {evaluation_dir}")
    for child in ("plan", "execution", "summary"):
        (evaluation_dir / child).mkdir(parents=True, exist_ok=False)
    write_json(evaluation_dir / "request.json", {"user_request": user_request})
    write_global_route_trace(
        evaluation_dir,
        catalog=catalog,
        route_result=route_result,
        router=router,
        history_retrieval=history_retrieval,
    )
    manifest = {
        "schema_version": 1,
        "evaluation_id": resolved_id,
        "status": "unsupported",
        "lifecycle_status": "completed_without_execution",
        "created_at": datetime.now().astimezone().isoformat(),
        "execution_finished_at": datetime.now().astimezone().isoformat(),
        "user_request": user_request,
        "auto_route": True,
        "global_query_route_path": "plan/global_query_route.json",
        "global_act_catalog_path": "plan/global_act_catalog.json",
        "route": route_result["selection"],
        "limitations": [
            "query requires an aspect outside the trusted ACT catalog"
        ],
    }
    write_json(evaluation_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    """Parse, validate, and dispatch one Agent evaluation."""

    args = parse_args()
    if args.benchmark == "libero":
        from mea.libero.chain import run_libero_agent_cli

        run_libero_agent_cli(args)
        return

    requested_planner = args.open_query_planner
    args.open_query_planner = resolve_default_open_query_planner(args)
    compat_proposals = None
    compat_errors: tuple[type[Exception], ...] = (ValueError,)
    if paper_compat_profile_requested(
        args,
        requested_open_query_planner=requested_planner,
    ):
        from experiments.paper.compat_agent_profile import (
            CompatAgentProfileError,
            resolve_compat_agent_services,
        )

        try:
            compat_proposals, compat_errors = resolve_compat_agent_services(
                args,
                requested_open_query_planner=requested_planner,
            )
        except CompatAgentProfileError as exc:
            raise SystemExit(str(exc)) from exc
    plan_agent_mode = args.open_query_planner == "plan_agent_v1"
    bound_plan_only = validate_and_normalize_agent_args(
        args,
        plan_agent_mode=plan_agent_mode,
    )
    context = create_agent_run_context(
        args,
        plan_agent_mode=plan_agent_mode,
        compat_bounded_proposals=compat_proposals,
        compat_plan_errors=compat_errors,
        bound_plan_only=bound_plan_only,
        observed_argv=list(sys.argv),
        initialize_registered_runtime=initialize_registered_dynamic_runtime,
    )
    unsupported = route_agent_query(
        context,
        finish_unsupported_global_route=finish_unsupported_global_route,
    )
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
