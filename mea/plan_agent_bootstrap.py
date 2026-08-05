"""Production bootstrap for the paper-aligned Plan Agent runtime.

This module owns the executable-policy discovery, post-interpretation task
binding, auditable unsupported exits, and final :class:`PlanAgentApplication`
wiring.  The command-line script parses arguments and preserves the explicit
paper compatibility dispatch; it does not own these production lifecycle
operations.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from mea.evaluation_identity import make_evaluation_id
from mea.history import EvaluationHistoryDB
from mea.plan_artifacts import (
    QUERY_INTERPRETATION,
    QUERY_INTERPRETATION_PROMPT,
    QUERY_INTERPRETATION_RESPONSE_PREFIX,
)
from mea.plan_agent_application import PlanAgentApplication
from mea.planner import (
    PlanAgent,
    PlanAgentQueryInterpreter,
    PlanAgentSession,
    policy_task_binding_from_target,
)
from mea.planner.open_task_resolver import rank_official_tasks
from mea.planner.query_contract import validate_query_sufficiency_contract
from mea.planner.runtime_task_binding import (
    RuntimePolicySpec,
    RuntimeTaskBindingError,
    build_runtime_open_world_evaluation_target,
)
from mea.providers import OpenAICompatibleProvider
from mea.robotwin.production_round_executor import build_production_round_executor


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def persist_query_contract(
    evaluation_dir: Path,
    plan: dict[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep the public contract artifact aligned with runtime discoveries."""

    normalized = validate_query_sufficiency_contract(contract)
    plan["query_contract"] = deepcopy(normalized)
    _write_json(
        evaluation_dir / "plan/query_sufficiency_contract.json",
        normalized,
    )
    return normalized


def discover_ready_plan_agent_targets(
    repo_root: Path,
    task_inventory: list[dict[str, Any]],
    *,
    max_rounds: int,
    policy_spec: RuntimePolicySpec | None = None,
) -> dict[str, Any]:
    """Bind every source/checkpoint-ready task without a task menu."""

    targets: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, str]] = []
    for item in sorted(
        task_inventory,
        key=lambda candidate: str(candidate.get("task_name", "")),
    ):
        task_name = str(item.get("task_name", "")).strip()
        if not task_name:
            raise RuntimeError("runtime task inventory contains no task_name")
        try:
            targets[task_name] = build_runtime_open_world_evaluation_target(
                repo_root,
                task_name,
                max_rounds=max_rounds,
                policy_spec=policy_spec,
            )
        except RuntimeTaskBindingError as exc:
            excluded.append(
                {
                    "task_name": task_name,
                    "reason": str(exc),
                }
            )
    return {
        "schema_version": 1,
        "targets": targets,
        "excluded": excluded,
    }


def build_bound_plan_agent_handoff(
    catalog: dict[str, Any] | None,
    *,
    task_name: str,
    user_request: str,
    runtime_target: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a resolved task/checkpoint without choosing an aspect."""

    target = next(
        (
            deepcopy(item)
            for item in (catalog or {}).get("tasks", [])
            if item.get("task_name") == task_name
        ),
        None,
    )
    runtime_binding = (
        policy_task_binding_from_target(runtime_target)
        if runtime_target is not None
        else None
    )
    if runtime_binding is not None:
        if runtime_binding["task_name"] != task_name:
            raise RuntimeError(
                "runtime target task differs from the resolved task"
            )
        target = {
            "task_name": runtime_binding["task_name"],
            "task_family": runtime_binding["task_schema"]["task_family"],
            "task_profile": "official",
            "planner_kind": "plan_agent_v1",
            "checkpoint": deepcopy(runtime_binding["checkpoint"]),
        }
    if target is None:
        raise RuntimeError(f"bound task is not checkpoint-ready: {task_name!r}")
    request = str(user_request or "").strip()
    if not request:
        raise RuntimeError("user_request must be non-empty")

    selection = {
        "schema_version": 3,
        "decision": "route",
        "task_name": target["task_name"],
        "task_profile": target["task_profile"],
        "evaluation_goal": f"answer_open_query_with_evidence: {request}",
        "requested_aspect_ids": [],
        "first_aspect_id": None,
        "unsupported_capabilities": [],
        "binding_only": True,
    }
    routed = {
        "task_name": target["task_name"],
        "task_profile": target["task_profile"],
        "proposal": None,
    }
    route_result = {
        "schema_version": 2,
        "selection": selection,
        "resolved": {
            "task_name": target["task_name"],
            "task_family": target["task_family"],
            "task_profile": target["task_profile"],
            "planner_kind": target["planner_kind"],
            "checkpoint": deepcopy(target["checkpoint"]),
            "aspects": [],
        },
        "catalog_sha256": (
            catalog.get("catalog_sha256")
            if isinstance(catalog, Mapping)
            else None
        ),
        "runtime_binding_sha256": (
            canonical_sha256(runtime_target)
            if runtime_target is not None
            else None
        ),
        "provider_called": False,
        "attempt_count": 0,
        "validation_errors": [],
        "provider_metadata": {},
        "route_source": "runtime_task_checkpoint_binding",
        "global_router_provider_calls": 0,
    }
    return route_result, routed


def build_pending_task_binding_policy_card(
    policy_spec: RuntimePolicySpec | None = None,
) -> dict[str, Any]:
    """Describe an unbound policy portfolio without exposing its task menu."""

    if policy_spec is None:
        return {
            "policy_name": "ACT task-specific checkpoint portfolio",
            "checkpoint_id": "selected_after_query_interpretation",
            "single_task_checkpoint": False,
            "training_tasks": ["withheld_until_semantic_task_retrieval"],
            "language_conditioned": False,
            "checkpoint_ready": True,
            "supports_unseen_tasks": False,
        }
    return {
        "policy_name": policy_spec.policy_name,
        "checkpoint_id": policy_spec.checkpoint_id,
        "single_task_checkpoint": (
            policy_spec.task_scope != "robotwin_official_tasks"
        ),
        "training_tasks": [policy_spec.task_scope],
        "language_conditioned": policy_spec.language_conditioned,
        "checkpoint_ready": True,
        "supports_unseen_tasks": False,
        "official_task_portfolio": (
            policy_spec.task_scope == "robotwin_official_tasks"
        ),
    }


def bind_ready_task_after_query_interpretation(
    concern: Mapping[str, Any],
    *,
    inventory: list[dict[str, Any]],
    ready_task_names: list[str],
    default_task_name: str,
    semantic_threshold: float = 0.2,
) -> dict[str, Any]:
    """Bind a checkpoint only after inventory-free Query interpretation."""

    if not 0.0 < float(semantic_threshold) <= 1.0:
        raise ValueError("semantic_threshold must be in (0, 1]")
    ranked = rank_official_tasks(
        concern,
        inventory,
        top_k=len(inventory),
    )
    ready = set(ready_task_names)
    ranked_ready = [
        item for item in ranked if str(item["task_name"]) in ready
    ]
    if not ranked_ready:
        raise RuntimeError(
            "no checkpoint-ready task remains after semantic task retrieval"
        )
    best = ranked_ready[0]
    del default_task_name  # retained for historical Python-call compatibility
    if float(best["score"]) < float(semantic_threshold):
        return {
            "schema_version": 1,
            "binding_status": "ambiguous",
            "selected_task_name": None,
            "reason_code": "task_underspecified_no_checkpoint_binding",
            "fallback_used": False,
            "catalog_visible_to_concern_model": False,
            "retrieval_field": "QueryInterpretation.task_intent",
            "semantic_threshold": float(semantic_threshold),
            "ranked_ready_tasks": ranked_ready,
        }
    return {
        "schema_version": 1,
        "binding_status": "bound",
        "selected_task_name": str(best["task_name"]),
        "reason_code": "semantic_task_intent_retrieval",
        "fallback_used": False,
        "catalog_visible_to_concern_model": False,
        "retrieval_field": "QueryInterpretation.task_intent",
        "semantic_threshold": float(semantic_threshold),
        "ranked_ready_tasks": ranked_ready,
    }


def concern_candidate_domain_is_executable(
    resolution: Mapping[str, Any],
    *,
    candidate_budget: int | None,
) -> bool:
    """Admit a semantic domain without requiring a preselected template."""

    if (
        resolution.get("resolution")
        == "official_execution_from_typed_needs"
        and resolution.get("execution_authorized") is True
    ):
        return candidate_budget is None or candidate_budget >= 0
    if candidate_budget is not None and candidate_budget < 1:
        return False
    decision = resolution.get("decision")
    if decision == "bind_single_aspect":
        templates = resolution.get("selected_template_ids")
        return isinstance(templates, list) and bool(templates)
    if decision == "discover_candidates":
        aspects = resolution.get("candidate_aspect_ids")
        return isinstance(aspects, list) and bool(aspects)
    if decision == "catalog_external":
        return True
    return False


def write_open_task_resolution_trace(
    evaluation_dir: Path,
    *,
    concern_bundle: dict[str, Any],
    task_inventory: list[dict[str, Any]],
    task_resolution: dict[str, Any],
    concern_agent: PlanAgentQueryInterpreter,
) -> None:
    """Persist Query interpretation and later executable-task resolution."""

    _write_json(evaluation_dir / QUERY_INTERPRETATION, concern_bundle)
    _write_json(
        evaluation_dir / "plan/robotwin_task_inventory.json",
        task_inventory,
    )
    _write_json(
        evaluation_dir / "plan/open_task_resolution.json",
        task_resolution,
    )
    if concern_agent.last_prompt is not None:
        (evaluation_dir / QUERY_INTERPRETATION_PROMPT).write_text(
            concern_agent.last_prompt,
            encoding="utf-8",
        )
    for index, response in enumerate(concern_agent.last_responses, start=1):
        (
            evaluation_dir
            / "plan"
            / f"{QUERY_INTERPRETATION_RESPONSE_PREFIX}{index}.txt"
        ).write_text(response + "\n", encoding="utf-8")


def finish_unsupported_open_task_resolution(
    repo_root: Path,
    *,
    evaluation_id: str | None,
    user_request: str,
    catalog: dict[str, Any] | None,
    concern_bundle: dict[str, Any],
    task_inventory: list[dict[str, Any]],
    task_resolution: dict[str, Any],
    concern_agent: PlanAgentQueryInterpreter,
    candidate_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a no-execution bundle for a policy/task mismatch."""

    resolved_id = evaluation_id or make_evaluation_id()
    if not re.fullmatch(r"eval_[A-Za-z0-9_]+", resolved_id):
        raise ValueError("evaluation_id must match eval_[A-Za-z0-9_]+")
    evaluation_dir = repo_root / "mea/evaluation_runs" / resolved_id
    if evaluation_dir.exists():
        raise RuntimeError(f"evaluation directory already exists: {evaluation_dir}")
    for child in ("plan", "execution", "summary"):
        (evaluation_dir / child).mkdir(parents=True, exist_ok=False)
    _write_json(
        evaluation_dir / "request.json",
        {"user_request": user_request},
    )
    if catalog is not None:
        _write_json(evaluation_dir / "plan/global_act_catalog.json", catalog)
    write_open_task_resolution_trace(
        evaluation_dir,
        concern_bundle=concern_bundle,
        task_inventory=task_inventory,
        task_resolution=task_resolution,
        concern_agent=concern_agent,
    )
    if candidate_resolution is not None:
        _write_json(
            evaluation_dir / "plan/concern_candidate_resolution.json",
            candidate_resolution,
        )
    now = datetime.now().astimezone().isoformat()
    manifest = {
        "schema_version": 1,
        "evaluation_id": resolved_id,
        "status": "unsupported",
        "lifecycle_status": "completed_without_execution",
        "created_at": now,
        "execution_finished_at": now,
        "user_request": user_request,
        "auto_route": True,
        "query_interpretation_path": QUERY_INTERPRETATION.as_posix(),
        "open_task_resolution_path": "plan/open_task_resolution.json",
        "route": task_resolution,
        "limitations": [
            (
                "the open Query does not uniquely authorize one executable "
                "candidate domain within the bounded rollout budget"
                if candidate_resolution is not None
                else "the evaluated policy checkpoint cannot execute the resolved task"
            )
        ],
    }
    if catalog is not None:
        manifest["global_act_catalog_path"] = "plan/global_act_catalog.json"
    if candidate_resolution is not None:
        manifest.update(
            {
                "status": "unsupported_candidate_domain",
                "concern_candidate_resolution_path": (
                    "plan/concern_candidate_resolution.json"
                ),
                "rollouts_executed": 0,
            }
        )
    _write_json(evaluation_dir / "manifest.json", manifest)
    return manifest


def run_plan_agent_application(
    *,
    args: Any,
    repo_root: Path,
    evaluation_dir: Path,
    evaluation_id: str,
    plan: dict[str, Any],
    session: PlanAgentSession,
    agent: PlanAgent,
    capabilities: dict[str, Any],
    provider: OpenAICompatibleProvider,
    models: Mapping[str, str],
    runtime_target: Mapping[str, Any],
    reviewed_tool_registry: Path | None,
    reviewed_vqa_registry: Path | None,
    global_route_result: dict[str, Any] | None,
    concern_bundle: dict[str, Any] | None,
    open_task_resolution: dict[str, Any] | None,
    concern_candidate_resolution: dict[str, Any] | None,
    history_database: EvaluationHistoryDB | None,
    history_retrieval: dict[str, Any],
    history_context_count: int,
) -> dict[str, Any]:
    """Construct and run the production application with backend-neutral wiring."""

    policy_server_port = (
        args.hyvla_port
        if args.policy_backend == "hyvla"
        else args.smolvla_port
    )
    return PlanAgentApplication(
        repo_root=repo_root,
        evaluation_dir=evaluation_dir,
        evaluation_id=evaluation_id,
        user_request=args.request,
        plan=plan,
        session=session,
        agent=agent,
        capabilities=capabilities,
        provider=provider,
        round_executor=build_production_round_executor(),
        models=models,
        base_url=args.base_url,
        gpu=args.gpu,
        max_reflections=args.max_reflections,
        telemetry_profile=args.telemetry_profile,
        policy_backend=args.policy_backend,
        runtime_target=runtime_target,
        policy_server_port=policy_server_port,
        reviewed_tool_registry=reviewed_tool_registry,
        reviewed_vqa_registry=reviewed_vqa_registry,
        max_agent_rounds=args.max_agent_rounds,
        global_route_result=global_route_result,
        free_concern_bundle=concern_bundle,
        open_task_resolution=open_task_resolution,
        concern_candidate_resolution=concern_candidate_resolution,
        history_database=history_database,
        history_retrieval=history_retrieval,
        history_context_count=history_context_count,
        history_disabled=bool(args.no_history),
        cli_candidate_hint_used=(
            args.bound_requested_aspect_ids is not None
        ),
    ).run()


__all__ = [
    "bind_ready_task_after_query_interpretation",
    "build_bound_plan_agent_handoff",
    "build_pending_task_binding_policy_card",
    "canonical_sha256",
    "concern_candidate_domain_is_executable",
    "discover_ready_plan_agent_targets",
    "finish_unsupported_open_task_resolution",
    "persist_query_contract",
    "run_plan_agent_application",
    "write_open_task_resolution_trace",
]
