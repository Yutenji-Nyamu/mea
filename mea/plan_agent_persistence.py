"""Manifest and runtime-decision persistence for Plan Agent evaluations."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.planner.plan_agent_schema import validate_open_query_capabilities

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_manifest(evaluation_dir: Path, **updates: Any) -> dict[str, Any]:
    """Update the evaluation manifest owned by the application lifecycle."""

    path = evaluation_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    _write_json(path, manifest)
    return manifest


def refresh_plan_agent_capabilities_from_runtime_context(
    capabilities: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote one backend-validated TaskContext into the next Plan step.

    Shared policies may start with source-only task identity.  The unchanged
    control establishes actor and telemetry authority before inference; expose
    that newly observed execution surface to the next evidence-conditioned
    decision without adding a task-specific capability menu.
    """

    current = deepcopy(dict(capabilities))
    raw_context = child_manifest.get("runtime_task_context")
    if raw_context is None:
        return current
    if not isinstance(raw_context, Mapping):
        raise ValueError("runtime_task_context must be an object")
    context = deepcopy(dict(raw_context))
    if (
        context.get("schema_version") != 1
        or context.get("taskgen_ready") is not True
        or context.get("schema_origin")
        not in {"runtime_probe", "task_schema"}
    ):
        raise ValueError(
            "runtime_task_context is not a validated execution context"
        )
    task_schema = context.get("task_schema")
    if not isinstance(task_schema, Mapping):
        raise ValueError("runtime_task_context has no task schema")
    task_name = context.get("task_name")
    if (
        child_manifest.get("generation_kind") != "official_passthrough"
        or child_manifest.get("task_module") != f"envs.{task_name}"
    ):
        # Current TaskGen contexts establish authority from the official base
        # reset.  They do not yet describe provider-added actors in the
        # generated module, so retaining the last validated capability card is
        # more accurate than promoting the base schema as generated-scene
        # authority.
        return current
    simulator = current.get("simulator_card")
    if not isinstance(simulator, Mapping):
        raise ValueError("Plan Agent capabilities have no simulator card")
    if (
        not isinstance(task_name, str)
        or task_name != simulator.get("task_name")
        or task_schema.get("task_name") != task_name
    ):
        raise ValueError(
            "runtime TaskContext task differs from the Plan Agent binding"
        )

    refreshed_simulator = deepcopy(dict(simulator))
    for field in (
        "physics_timestep_seconds",
        "action_dimension",
        "tracked_actors",
        "semantic_fields",
        "semantic_roles",
        "success_contract",
        "telemetry_observables",
    ):
        if field in task_schema:
            refreshed_simulator[field] = deepcopy(task_schema[field])
    refreshed_simulator["task_context_authority"] = {
        "schema_origin": context["schema_origin"],
        "official_source": context.get("official_source"),
        "official_class": context.get("official_class"),
        "authority": deepcopy(dict(context.get("authority") or {})),
    }
    current["simulator_card"] = refreshed_simulator

    generation = current.get("generation_card")
    primitives = (
        generation.get("backend_primitives")
        if isinstance(generation, Mapping)
        else None
    )
    if not isinstance(primitives, Mapping):
        raise ValueError(
            "runtime TaskContext promotion requires backend primitives"
        )
    refreshed_primitives = deepcopy(dict(primitives))
    refreshed_primitives["telemetry"] = True
    current["generation_card"] = {
        "backend_primitives": refreshed_primitives,
    }

    policy = current.get("policy_card")
    if isinstance(policy, Mapping):
        refreshed_policy = deepcopy(dict(policy))
        unknown = refreshed_policy.get("unknown_metadata")
        if isinstance(unknown, list):
            refreshed_policy["unknown_metadata"] = [
                item for item in unknown if item != "semantic_actor_schema"
            ]
        current["policy_card"] = refreshed_policy
    return validate_open_query_capabilities(current)


def apply_external_hard_round_cap(
    *,
    evaluation_dir: Path,
    plan: dict[str, Any],
    round_runs: list[dict[str, Any]],
    executed_rounds: int,
    max_agent_rounds: int,
    user_request: str,
    bound_plan_session: Any = None,
    plan_agent_proposal: Mapping[str, Any] | None = None,
    plan_agent_artifact_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Persist a hard-cap stop after any evidence-backed Agent decision."""

    completed = [
        item["round_plan"].get("candidate_id")
        or item["round_plan"].get("template_id")
        for item in round_runs
    ]
    requested = [
        *plan.get("requested_candidate_ids", []),
        *plan.get("requested_template_ids", []),
    ]
    remaining = [
        candidate_id
        for candidate_id in dict.fromkeys(requested)
        if candidate_id not in completed
    ]
    agent_decision = None
    if plan_agent_proposal is not None:
        agent_decision = {
            "action": plan_agent_proposal.get("action"),
            "sub_aspect": plan_agent_proposal.get("sub_aspect"),
            "authored_from_completed_evidence": True,
            "artifact_path": plan_agent_artifact_path,
        }
    assessment = {
        "schema_version": 2,
        "state": "external_hard_round_cap_reached",
        "required_action": "stop",
        "completed_rounds": executed_rounds,
        "max_agent_rounds": max_agent_rounds,
        "remaining_candidate_ids": remaining,
        "policy_outcome_not_inferred": True,
        "plan_agent_decision_before_cap": agent_decision,
    }
    decision = {
        "schema_version": 3,
        "action": "stop",
        "transition": "stop",
        "observation_summary": (
            f"Completed {executed_rounds} round(s); the task-agnostic hard "
            "execution cap rejected the Agent's request for another round."
            if agent_decision is not None
            else (
                f"Completed {executed_rounds} round(s); the task-agnostic "
                "hard execution cap is now exhausted."
            )
        ),
        "decision_reason": "external_max_agent_rounds_budget",
        "next_aspect_id": None,
        "next_template_id": None,
        "remaining_candidate_ids_before_decision": remaining,
        "round_budget_before_decision": 0,
        "evidence_assessment": assessment,
        "plan_agent_decision_before_cap": agent_decision,
        "next_round": None,
    }
    plan.setdefault("round_decisions", []).append(decision)
    plan["planning_state"] = (
        f"stopped_after_round_{executed_rounds}_by_hard_cap"
    )
    _write_json(
        evaluation_dir / f"plan/evidence_after_round_{executed_rounds}.json",
        assessment,
    )
    _write_json(
        evaluation_dir / f"plan/decision_after_round_{executed_rounds}.json",
        decision,
    )
    _write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
    if bound_plan_session is not None:
        _write_json(
            evaluation_dir / "plan/bound_task_session.json",
            bound_plan_session.snapshot(
                user_request,
                plan,
                [item["round_summary"] for item in round_runs],
            ),
        )
    update_manifest(
        evaluation_dir,
        status=plan["planning_state"],
        plan=plan,
        hard_round_cap_stop={
            "max_agent_rounds": max_agent_rounds,
            "executed_rounds": executed_rounds,
            "decision_path": (
                f"plan/decision_after_round_{executed_rounds}.json"
            ),
            "plan_agent_action_before_cap": (
                agent_decision["action"]
                if agent_decision is not None
                else None
            ),
            "plan_agent_artifact_path": plan_agent_artifact_path,
        },
    )
    return plan, decision, assessment


__all__ = [
    "apply_external_hard_round_cap",
    "refresh_plan_agent_capabilities_from_runtime_context",
    "update_manifest",
]
