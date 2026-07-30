"""Plan and execute a bounded, evidence-driven multi-round MEA evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.execution_vqa import (
    build_execution_vqa_query,
    is_run_local_phenomenon_id,
    run_execution_vqa,
)
from mea.agent_cli import (
    parse_args,
    resolve_plan_agent_allowed_aspects,
    resolve_plan_agent_candidate_budget,
    resolve_plan_agent_control_required,
    resolve_default_open_query_planner,
)
from mea.agent_acceptance import (
    _episode_tool_results,
    build_compact_flagship_acceptance,
)
from mea.agent_evidence import (
    _round_evidence,
    build_evidence_bundle,
    compact_aggregate_result,
    compact_trusted_tools,
    round_execution_backend,
)
from mea.capability_adapter import (
    CapabilityAdapterError,
    build_contract_tool_request,
    taskgen_route,
    validate_capability_contract,
    validate_contract_changes,
)
from mea.feedback import (
    PlanAgentFinalSummary,
    render_evaluation_report,
    write_evidence_report,
)
from mea.history import EvaluationHistoryDB
from mea.plan_artifacts import (
    INITIAL_SUB_ASPECT_PROPOSAL,
    PLAN_AGENT_CAPABILITIES,
    PLAN_AGENT_SESSION,
    PLAN_AGENT_STEPS,
    PROPOSAL_FILENAME,
    PROPOSAL_MATERIALIZATION,
    QUERY_INTERPRETATION,
    QUERY_INTERPRETATION_PROMPT,
    QUERY_INTERPRETATION_RESPONSE_PREFIX,
)
from mea.planner import (
    AdaptivePlanStepAgent,
    advance_implementation_trace_with_tool,
    BoundTaskPlanSession,
    PlanAgentExecutionSession,
    PlanAgent,
    PlanAgentInitialPlanBuilder,
    PlanAgentQueryInterpreter,
    PlanAgentSession,
    GlobalQueryRouter,
    build_evidence_aggregate,
    build_implementation_trace,
    build_dynamic_experiment_candidate,
    build_initial_semantic_proposal_bundle,
    build_act_catalog,
    build_open_world_evaluation_target,
    evaluation_intent_from_query_interpretation,
    make_evaluation_id,
    policy_task_binding_from_target,
    project_open_query_capabilities,
    render_query_answer,
    resolve_concern_candidate_domain,
    resolve_open_task,
    route_to_planner_proposal,
    validate_open_query_plan_proposal,
)
from mea.planner.experiment_candidate import (
    build_experiment_candidate,
    validate_experiment_candidate,
)
from mea.planner.open_task_resolver import (
    discover_robotwin_runtime_task_inventory,
    rank_official_tasks,
)
from mea.planner.query_contract import (
    build_query_sufficiency_contract,
    extend_query_candidate_universe,
    infer_claim_type,
    validate_query_sufficiency_contract,
)
from mea.planner.runtime_task_binding import (
    RuntimeTaskBindingError,
    build_runtime_open_world_evaluation_target,
)
from mea.proposals import (
    ProposalError,
    materialize_round_proposals,
    tool_request_from_proposal,
    validate_task_proposal,
    validate_tool_proposal,
)
from mea.proposal_agent import (
    BoundedProposalAgent,
    ProposalAgentError,
    proposal_capability_mode,
)
from mea.providers import (
    OpenAICompatibleProvider,
    resolve_model_profile,
)
from mea.robotwin import project_executed_round_through_method_runtime
from mea.toolgen import (
    OpenToolRequestAgent,
    compatible_reviewed_tool_requests,
    compatible_run_local_tool_requests,
    execute_tool_request,
    route_tool_request,
)
from mea.toolkit import aggregate_tool_executions
def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def persist_query_contract(
    evaluation_dir: Path,
    plan: dict[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep the public contract artifact aligned with runtime discoveries."""

    normalized = validate_query_sufficiency_contract(contract)
    plan["query_contract"] = deepcopy(normalized)
    write_json(
        evaluation_dir / "plan/query_sufficiency_contract.json",
        normalized,
    )
    return normalized


def should_enable_adaptive_plan_step(
    *,
    fixed_click_bell: bool,
    legacy_click_bell: bool,
    registered_strategy: str | None,
) -> bool:
    """Select the evidence-driven planner for normal and registered dynamic runs.

    Registration freezes candidate identity, not the planning method. Fixed and
    legacy baselines retain their pre-existing deterministic planners.
    """

    return (
        not fixed_click_bell
        and not legacy_click_bell
        and registered_strategy != "fixed_predeclared_v1"
    )


def discover_ready_plan_agent_targets(
    repo_root: Path,
    task_inventory: list[dict[str, Any]],
    *,
    max_rounds: int,
) -> dict[str, Any]:
    """Bind every source/schema/checkpoint-ready task without a task menu.

    Task discovery is source/schema driven; checkpoint availability is the
    only policy-specific execution filter.  CapabilityAdapter and the legacy
    catalog may later enrich retrieval, but neither authorizes membership in
    this production runtime map.
    """

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
    """Bind an already resolved task/checkpoint without choosing an aspect.

    Query interpretation and the policy compatibility gate have already
    selected the executable task.  QueryContract and the direct Plan Agent
    initial builder
    decide whether an unchanged control is required later; this handoff must
    not manufacture an aspect/template proposal.  Production passes only the
    runtime target; ``catalog`` remains an optional legacy trace input.
    """

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


def build_pending_task_binding_policy_card() -> dict[str, Any]:
    """Describe an unbound checkpoint portfolio without exposing its menu.

    Query interpretation must happen before the official task inventory is
    retrieved.  This neutral card makes that ordering explicit: it describes
    the evaluation surface, but contains no executable task or aspect name.
    """

    return {
        "policy_name": "ACT task-specific checkpoint portfolio",
        "checkpoint_id": "selected_after_query_interpretation",
        "single_task_checkpoint": False,
        "training_tasks": ["withheld_until_semantic_task_retrieval"],
        "language_conditioned": False,
        "checkpoint_ready": True,
        "supports_unseen_tasks": False,
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
    del default_task_name  # retained only for historical Python-call compatibility
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
    selected = best
    return {
        "schema_version": 1,
        "binding_status": "bound",
        "selected_task_name": str(selected["task_name"]),
        "reason_code": "semantic_task_intent_retrieval",
        "fallback_used": False,
        "catalog_visible_to_concern_model": False,
        "retrieval_field": "QueryInterpretation.task_intent",
        "semantic_threshold": float(semantic_threshold),
        "ranked_ready_tasks": ranked_ready,
    }


# Compatibility names for paper protocols and historical imports.
discover_ready_claim_first_targets = discover_ready_plan_agent_targets
build_bound_claim_first_handoff = build_bound_plan_agent_handoff
bind_ready_task_after_free_concern = (
    bind_ready_task_after_query_interpretation
)


def concern_candidate_domain_is_executable(
    resolution: Mapping[str, Any],
    *,
    candidate_budget: int | None,
) -> bool:
    """Admit a semantic domain without requiring a preselected template."""

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
    """Initialize the catalog/provider pair skipped by a registered route."""

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


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bound_target_task_name(target: Mapping[str, Any]) -> str:
    """Read a legacy target or the production PolicyTaskBinding identity."""

    task_name = target.get("task_name")
    if not isinstance(task_name, str) or not task_name.strip():
        binding = target.get("policy_task_binding")
        task_name = (
            binding.get("task_name")
            if isinstance(binding, Mapping)
            else None
        )
    if not isinstance(task_name, str) or not task_name.strip():
        raise RuntimeError("evaluation target has no bound task identity")
    return task_name.strip()


def apply_bounded_round_proposal(
    *,
    proposal_agent: BoundedProposalAgent,
    user_query: str,
    target: dict[str, Any],
    planning_context: dict[str, Any],
    round_plan: dict[str, Any],
    evaluation_dir: Path,
    round_number: int,
    semantic_proposal: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Author and persist one bounded Task/Tool Proposal for a materialized round."""

    round_plan = deepcopy(round_plan)
    bound_task_name = bound_target_task_name(target)
    supplied_task_name = round_plan.get("task_name")
    if supplied_task_name not in {None, bound_task_name}:
        raise RuntimeError("round proposal cannot change the bound task")
    # The legacy BBH materializer predates explicit per-round task identity.
    # Inject it only from the already frozen EvaluationTarget.
    round_plan["task_name"] = bound_task_name
    aspect_id = str(
        (round_plan.get("task_proposal") or {}).get("aspect_id")
        or round_plan.get("aspect_id")
        or round_plan.get("sub_aspect")
        or ""
    )
    template_id = str(round_plan.get("template_id") or "")
    task_name = bound_task_name
    if (
        isinstance(semantic_proposal, Mapping)
        and semantic_proposal
        and set(semantic_proposal).issubset({"task_need", "tool_need"})
    ):
        # Paper-ablation compatibility only.  The old bounded adapter carried
        # a read-only two-need projection, not a complete Plan Agent Proposal.
        # Translate it at this legacy boundary without weakening the typed
        # production Proposal validator.
        legacy_task_need = deepcopy(semantic_proposal.get("task_need"))
        legacy_tool_need = deepcopy(semantic_proposal.get("tool_need"))
        semantic_proposal = {
            "scene_need": legacy_task_need,
            "checker_need": deepcopy(legacy_task_need),
            "rule_tool_need": legacy_tool_need,
            "vqa_tool_need": None,
            "legacy_need_projection": True,
        }
    else:
        semantic_proposal = (
            validate_open_query_plan_proposal(
                semantic_proposal,
                has_evidence=True,
            )
            if isinstance(semantic_proposal, Mapping)
            else None
        )
    scene_need = (
        semantic_proposal.get("scene_need")
        if semantic_proposal is not None
        else None
    )
    checker_need = (
        semantic_proposal.get("checker_need")
        if semantic_proposal is not None
        else None
    )
    rule_tool_need = (
        semantic_proposal.get("rule_tool_need")
        if semantic_proposal is not None
        else None
    )
    vqa_tool_need = (
        semantic_proposal.get("vqa_tool_need")
        if semantic_proposal is not None
        else None
    )
    taskgen_requested = bool(
        any(
            isinstance(need, Mapping) and need.get("required") is True
            for need in (scene_need, checker_need)
        )
    )
    capability_contract = round_plan.get("capability_contract")
    selected_taskgen_route = (
        taskgen_route(capability_contract)
        if isinstance(capability_contract, Mapping)
        else None
    )
    provider_scene_checker_requested = bool(
        taskgen_requested
        and selected_taskgen_route == "provider_scene_checker_codegen"
    )
    generated_success_requested = bool(
        (
            isinstance(checker_need, Mapping)
            and checker_need.get("required") is True
        )
        or (
            taskgen_requested
            and task_name == "beat_block_hammer"
            and aspect_id == "object_appearance.color"
        )
    )
    semantic_rule_tool_requested = bool(
        isinstance(rule_tool_need, Mapping)
        and rule_tool_need.get("required") is True
    )
    semantic_vqa_tool_requested = bool(
        isinstance(vqa_tool_need, Mapping)
        and vqa_tool_need.get("required") is True
    )
    mode = proposal_capability_mode(
        task_name,
        aspect_id,
        experimental_success=bool(
            taskgen_requested
            and task_name == "beat_block_hammer"
            and aspect_id == "object_appearance.color"
        ),
    )
    tool_satisfied_by_task_checker = bool(
        semantic_rule_tool_requested
        and selected_taskgen_route == "provider_scene_checker_codegen"
    )
    new_tool_requested = bool(
        semantic_rule_tool_requested
        and not tool_satisfied_by_task_checker
    )
    proposal_context = deepcopy(planning_context)
    if semantic_proposal is not None:
        proposal_context["upstream_semantic_plan_proposal"] = deepcopy(
            semantic_proposal
        )
        proposal_context["semantic_need_binding"] = {
            "taskgen_required": taskgen_requested,
            "generated_success_requested": generated_success_requested,
            "toolgen_required": new_tool_requested,
            "vqa_tool_required": semantic_vqa_tool_requested,
            "taskgen_capability_mode": mode,
            "selected_taskgen_route": selected_taskgen_route,
            "toolgen_requires_new_typed_metric": new_tool_requested,
            "tool_satisfied_by_task_checker": (
                tool_satisfied_by_task_checker
            ),
        }
    proposal_dir = (
        evaluation_dir / "plan/bounded_proposal" / f"round_{round_number:02d}"
    )
    proposal_dir.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] | None = None
    proposal_source = "BoundedProposalAgent"
    prompt_text = ""
    try:
        if (
            provider_scene_checker_requested
            and semantic_proposal is not None
        ):
            # The public Plan Agent already authored the semantic
            # Proposal. Re-asking a second model to copy the registered
            # scene/checker envelope adds no paper-level decision and can
            # invent incompatible VQA ids. Bind that semantic need directly
            # to the trusted capability contract; TaskGen still performs the
            # provider-authored scene/checker code generation downstream.
            task_proposal = deepcopy(round_plan["task_proposal"])
            tool_proposal = deepcopy(round_plan["tool_proposal"])
            bundle = {
                "schema_version": 1,
                "task_proposal": task_proposal,
                "tool_proposal": tool_proposal,
                "tool_route_preview": route_tool_request(
                    tool_request_from_proposal(tool_proposal)
                )["route_decision"],
            }
            proposal_source = (
                "runtime_bound_plan_agent_semantics_to_registered_"
                "scene_checker"
            )
            prompt_text = (
                "No second Proposal-model call. The provider-authored "
                "Plan Agent semantic need is bound to the registered "
                "provider scene+checker capability; executable code remains "
                "provider-generated and validated by TaskGen.\n"
            )
        else:
            bundle = proposal_agent.propose(
                user_query,
                target=target,
                aspect_id=aspect_id,
                base_template_id=template_id,
                capability_mode=mode,
                planning_context=proposal_context,
                require_novel_changes=(mode == "novel_bounded"),
                require_new_tool=new_tool_requested,
            )
            prompt_text = proposal_agent.last_prompt or ""
        write_json(proposal_dir / "proposal_candidate_bundle.json", bundle)
        materialized = materialize_round_proposals(
            round_plan,
            bundle["task_proposal"],
            bundle["tool_proposal"],
        )
    except Exception as exc:
        write_json(
            proposal_dir / "proposal_failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "proposal_capability_mode": mode,
                "base_template_id": template_id,
                "round_number": round_number,
                "provider_or_validation_errors": deepcopy(
                    getattr(proposal_agent, "last_errors", [])
                ),
                "bounded_repairs": deepcopy(
                    getattr(proposal_agent, "last_repairs", [])
                ),
                "candidate_bundle_path": (
                    "proposal_candidate_bundle.json" if bundle is not None else None
                ),
            },
        )
        raise
    finally:
        if proposal_source == "BoundedProposalAgent":
            prompt_text = proposal_agent.last_prompt or prompt_text
        (proposal_dir / "prompt.md").write_text(
            prompt_text, encoding="utf-8"
        )
        for index, response in enumerate(
            (
                proposal_agent.last_responses
                if proposal_source == "BoundedProposalAgent"
                else []
            ),
            start=1,
        ):
            (proposal_dir / f"response_{index}.txt").write_text(
                response + "\n", encoding="utf-8"
            )
    assert bundle is not None
    artifact = {
        **bundle,
        "proposal_capability_mode": mode,
        "proposal_source": proposal_source,
        "base_template_id": template_id,
        "round_number": round_number,
        "attempt_count": (
            len(proposal_agent.last_responses)
            if proposal_source == "BoundedProposalAgent"
            else 0
        ),
        "provider_or_validation_errors": deepcopy(
            getattr(proposal_agent, "last_errors", [])
        ),
        "bounded_repairs": deepcopy(
            getattr(proposal_agent, "last_repairs", [])
        ),
        "semantic_plan_proposal": deepcopy(semantic_proposal),
        "semantic_need_binding": {
            "taskgen_required": taskgen_requested,
            "generated_success_requested": generated_success_requested,
            "toolgen_required": new_tool_requested,
            "taskgen_capability_mode": mode,
            "selected_taskgen_route": selected_taskgen_route,
            "toolgen_requires_new_typed_metric": new_tool_requested,
            "tool_satisfied_by_task_checker": (
                tool_satisfied_by_task_checker
            ),
        },
    }
    write_json(proposal_dir / "proposal_bundle.json", artifact)
    # Preserve the batch-12 first-round artifact path for existing readers.
    if round_number == 1:
        compatibility_dir = evaluation_dir / "plan/bounded_proposal"
        (compatibility_dir / "prompt.md").write_text(
            prompt_text, encoding="utf-8"
        )
        write_json(compatibility_dir / "proposal_bundle.json", artifact)
    return materialized, artifact


def adjudicate_bounded_transition(
    *,
    plan_session: BoundTaskPlanSession,
    user_query: str,
    observation_history: list[dict[str, Any]],
    current_plan: dict[str, Any],
    candidate_plan: dict[str, Any],
    candidate_decision: dict[str, Any],
    proposal_mode: str,
    proposal_agent: BoundedProposalAgent | None,
    planning_context: dict[str, Any] | None,
    evaluation_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Use one task-agnostic boundary for adaptive BBH/click transitions.

    Task-specific planners still explain and materialize their candidate next
    round.  This common entry point optionally authors that round's bounded
    Task/Tool Proposals, then lets ``BoundTaskPlanSession`` enforce the
    evidence-selected transition and frozen task/checkpoint/round budget.
    """

    candidate = deepcopy(candidate_plan)
    decision = deepcopy(candidate_decision)
    action = decision.get("action")
    if action not in {"continue", "stop"}:
        raise RuntimeError(
            "common bounded transition accepts continue or stop; task-specific "
            "verification remains a compatibility path"
        )
    if proposal_mode == "bounded_each_round" and action == "continue":
        if proposal_agent is None or planning_context is None:
            raise RuntimeError(
                "bounded_each_round proposal state was not initialized"
            )
        next_round_number = len(current_plan["rounds"]) + 1
        proposed_round, _proposal_artifact = apply_bounded_round_proposal(
            proposal_agent=proposal_agent,
            user_query=user_query,
            target=plan_session.target,
            planning_context=planning_context,
            round_plan=candidate["rounds"][-1],
            evaluation_dir=evaluation_dir,
            round_number=next_round_number,
        )
        candidate["rounds"][-1] = proposed_round
        decision["next_round"] = proposed_round
        candidate["round_decisions"][-1] = decision
    updated, canonical = plan_session.adjudicate(
        current_plan,
        observation_history,
        candidate_plan=candidate,
        candidate_decision=decision,
    )
    directive = plan_session.directive(
        current_plan,
        observation_history,
        candidate_decision=decision,
    )
    return updated, canonical, directive


def update_manifest(evaluation_dir: Path, **updates: Any) -> dict[str, Any]:
    path = evaluation_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    write_json(path, manifest)
    return manifest


def persist_adaptive_step_selection(
    evaluation_dir: Path,
    *,
    after_round: int,
    prompt: str | None,
    responses: list[str],
    step_bundle: dict[str, Any],
    navigation_options: dict[str, Any],
) -> str:
    """Persist a selected PlanStep before materializing its next task proposal.

    The planner decision is evidence in its own right.  Writing it before the
    fallible TaskGen/Proposal stage keeps an interrupted evaluation auditable
    and makes the boundary between decision and execution explicit.
    """

    step_dir = (
        evaluation_dir
        / "plan"
        / "adaptive_steps"
        / f"after_round_{after_round:02d}"
    )
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "prompt.md").write_text(prompt or "", encoding="utf-8")
    for index, response in enumerate(responses, start=1):
        (step_dir / f"response_{index}.txt").write_text(
            response + "\n", encoding="utf-8"
        )
    write_json(step_dir / "plan_step_bundle.json", step_bundle)
    write_json(
        evaluation_dir / f"plan/evidence_after_round_{after_round}.json",
        navigation_options,
    )
    return str(step_dir.relative_to(evaluation_dir)).replace("\\", "/")


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
        {
            **route_result,
            "history_retrieval": history_retrieval,
        },
    )
    if router is not None and router.last_prompt is not None:
        (evaluation_dir / "plan/global_query_prompt.md").write_text(
            router.last_prompt, encoding="utf-8"
        )
    for index, response in enumerate(
        router.last_responses if router is not None else [], start=1
    ):
        (evaluation_dir / f"plan/global_query_response_{index}.txt").write_text(
            response + "\n", encoding="utf-8"
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
    """Create an auditable no-execution result for an unsupported query."""

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
        "limitations": ["query requires an aspect outside the trusted ACT catalog"],
    }
    write_json(evaluation_dir / "manifest.json", manifest)
    return manifest


def write_open_task_resolution_trace(
    evaluation_dir: Path,
    *,
    concern_bundle: dict[str, Any],
    task_inventory: list[dict[str, Any]],
    task_resolution: dict[str, Any],
    concern_agent: PlanAgentQueryInterpreter,
) -> None:
    """Persist Plan Agent Query interpretation and later task resolution."""

    write_json(evaluation_dir / QUERY_INTERPRETATION, concern_bundle)
    write_json(evaluation_dir / "plan/robotwin_task_inventory.json", task_inventory)
    write_json(evaluation_dir / "plan/open_task_resolution.json", task_resolution)
    if concern_agent.last_prompt is not None:
        (evaluation_dir / QUERY_INTERPRETATION_PROMPT).write_text(
            concern_agent.last_prompt, encoding="utf-8"
        )
    for index, response in enumerate(concern_agent.last_responses, start=1):
        (
            evaluation_dir
            / "plan"
            / f"{QUERY_INTERPRETATION_RESPONSE_PREFIX}{index}.txt"
        ).write_text(
            response + "\n", encoding="utf-8"
        )


def finish_unsupported_open_task_resolution(
    repo_root: Path,
    *,
    evaluation_id: str | None,
    user_request: str,
    catalog: dict[str, Any],
    concern_bundle: dict[str, Any],
    task_inventory: list[dict[str, Any]],
    task_resolution: dict[str, Any],
    concern_agent: PlanAgentQueryInterpreter,
    candidate_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the normal no-execution bundle for a policy/task mismatch."""

    resolved_id = evaluation_id or make_evaluation_id()
    if not re.fullmatch(r"eval_[A-Za-z0-9_]+", resolved_id):
        raise ValueError("evaluation_id must match eval_[A-Za-z0-9_]+")
    evaluation_dir = repo_root / "mea/evaluation_runs" / resolved_id
    if evaluation_dir.exists():
        raise RuntimeError(f"evaluation directory already exists: {evaluation_dir}")
    for child in ("plan", "execution", "summary"):
        (evaluation_dir / child).mkdir(parents=True, exist_ok=False)
    write_json(evaluation_dir / "request.json", {"user_request": user_request})
    write_json(evaluation_dir / "plan/global_act_catalog.json", catalog)
    write_open_task_resolution_trace(
        evaluation_dir,
        concern_bundle=concern_bundle,
        task_inventory=task_inventory,
        task_resolution=task_resolution,
        concern_agent=concern_agent,
    )
    if candidate_resolution is not None:
        write_json(
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
        "global_act_catalog_path": "plan/global_act_catalog.json",
        "route": task_resolution,
        "limitations": [
            (
                "the open Query does not uniquely authorize one executable "
                "candidate domain within the bounded rollout budget"
                if candidate_resolution is not None
                else
                "the evaluated policy checkpoint cannot execute the resolved task"
            )
        ],
    }
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
    write_json(evaluation_dir / "manifest.json", manifest)
    return manifest


def child_run_id(evaluation_id: str, round_id: str) -> str:
    return f"run_{evaluation_id.removeprefix('eval_')}_{round_id}"


def validate_round_capability_contract(
    round_plan: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind every duplicated runtime field to one trusted adapter contract."""

    raw = round_plan.get("capability_contract")
    if raw is None:
        return None
    try:
        contract = validate_capability_contract(raw)
        registered_tool = build_contract_tool_request(contract)
    except (CapabilityAdapterError, ValueError) as exc:
        raise ValueError(f"invalid round capability contract: {exc}") from exc
    taskgen = contract["taskgen"]
    task_proposal = round_plan.get("task_proposal")
    tool_proposal = round_plan.get("tool_proposal")
    if task_proposal is not None or tool_proposal is not None:
        if task_proposal is None or tool_proposal is None:
            raise ValueError("round must provide TaskProposal and ToolProposal together")
        try:
            task_proposal = validate_task_proposal(
                task_proposal, expected_task_name=contract["task_name"]
            )
            tool_proposal = validate_tool_proposal(
                tool_proposal,
                expected_task_name=contract["task_name"],
                expected_aspect_id=contract["aspect"]["aspect_id"],
            )
            proposal_changes = validate_contract_changes(
                contract, task_proposal["changes"]
            )
        except (ProposalError, CapabilityAdapterError) as exc:
            raise ValueError(f"invalid round proposal: {exc}") from exc
        if task_proposal["capability_id"] != taskgen["capability_id"]:
            raise ValueError("TaskProposal capability differs from capability envelope")
        proposed_tool_request = tool_request_from_proposal(tool_proposal)
        proposed_tool_route = route_tool_request(proposed_tool_request)[
            "route_decision"
        ]
        typed_metric = (
            tool_proposal["schema_version"] == 3
            and proposed_tool_route["resolved_route"]
            == "typed_metric_spec_compile"
        )
        if (
            tool_proposal["metric"] != contract["tool"]["metric"]
            and not typed_metric
        ):
            raise ValueError("ToolProposal metric differs from capability envelope")
        catalog_phenomena = {
            item
            for item in tool_proposal["vqa_phenomenon_ids"]
            if not is_run_local_phenomenon_id(item)
        }
        if not catalog_phenomena <= set(
            contract["vqa"]["phenomenon_ids"]
        ):
            raise ValueError("ToolProposal VQA assignment exceeds capability envelope")
        expected_variant = (
            task_proposal["proposal_id"]
            if taskgen["task_variant_id"] is not None
            else None
        )
        expected_changes = proposal_changes
        expected_tool = tool_request_from_proposal(tool_proposal)
        expected_vqa = tool_proposal["vqa_phenomenon_ids"]
    else:
        expected_variant = taskgen["task_variant_id"]
        expected_changes = taskgen["changes"]
        expected_tool = registered_tool
        expected_vqa = contract["vqa"]["phenomenon_ids"]
    expected = {
        "task_name": contract["task_name"],
        "template_id": contract["template_id"],
        "capability_id": taskgen["capability_id"],
        "task_variant_id": expected_variant,
        "sub_aspect": contract["aspect"]["aspect_id"],
        "route": taskgen_route(contract),
        "variant_hint": expected_changes,
        "tool_request": expected_tool,
        "vqa_phenomenon_ids": expected_vqa,
        "required_gates": contract["required_gates"],
    }
    raw_task_name = round_plan.get("task_name")
    if not isinstance(raw_task_name, str) or not raw_task_name.strip():
        raise ValueError("round task_name must be explicit")
    observed = {
        "task_name": raw_task_name.strip(),
        "template_id": round_plan.get("template_id"),
        "capability_id": round_plan.get("capability_id"),
        "task_variant_id": round_plan.get("task_variant_id"),
        "sub_aspect": round_plan.get("sub_aspect"),
        "route": round_plan.get("route"),
        "variant_hint": round_plan.get("variant_hint") or {},
        "tool_request": round_plan.get("tool_request"),
        "vqa_phenomenon_ids": round_plan.get("vqa_phenomenon_ids"),
        "required_gates": (round_plan.get("execution") or {}).get("gates"),
    }
    mismatches = sorted(key for key in expected if observed[key] != expected[key])
    if mismatches:
        raise ValueError(
            "round fields differ from capability contract: " + ", ".join(mismatches)
        )
    return contract


def build_taskgen_command(
    repo_root: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    *,
    text_model: str,
    vision_model: str,
    base_url: str | None,
    gpu: int,
    max_reflections: int,
    telemetry_profile: str = "balanced_v1",
    reviewed_task_registry: Path | None = None,
    registration_identity: dict[str, Any] | None = None,
    run_id_suffix: str = "",
) -> tuple[list[str], str]:
    capability_contract = validate_round_capability_contract(round_plan)
    if run_id_suffix and re.fullmatch(r"_[A-Za-z0-9_]+", run_id_suffix) is None:
        raise ValueError("run_id_suffix must be empty or a safe underscore suffix")
    run_id = child_run_id(evaluation_id, round_plan["round_id"]) + run_id_suffix
    execution = round_plan["execution"]
    seed = execution["seeds"][0]
    raw_task_name = round_plan.get("task_name")
    if not isinstance(raw_task_name, str) or not raw_task_name.strip():
        raise ValueError("round task_name must be explicit")
    task_name = (
        capability_contract["task_name"]
        if capability_contract is not None
        else raw_task_name.strip()
    )
    task_proposal = round_plan.get("task_proposal")
    variant_hint = round_plan.get("variant_hint") or {}
    if task_proposal is not None:
        try:
            normalized_task_proposal = validate_task_proposal(
                task_proposal, expected_task_name=task_name
            )
        except ProposalError as exc:
            raise ValueError(f"invalid TaskProposal before TaskGen: {exc}") from exc
        variant_hint = normalized_task_proposal["changes"]
    task_module = round_plan.get("task_module")
    route = (
        taskgen_route(capability_contract)
        if capability_contract is not None
        else str(round_plan["route"])
    )
    execution_backend = round_execution_backend(round_plan)
    command = [
        sys.executable,
        str(repo_root / "scripts/manipeval_taskgen.py"),
        "--repo-root",
        str(repo_root),
        "--request",
        round_plan["task_instruction"],
        "--run-id",
        run_id,
        "--task-name",
        task_name,
        "--mode",
        route,
        "--text-model",
        text_model,
        "--vision-model",
        vision_model,
        "--seed",
        str(seed),
        "--num-episodes",
        str(execution["num_episodes"]),
        "--gpu",
        str(gpu),
        "--telemetry-profile",
        telemetry_profile,
        "--probe",
        "--max-reflections",
        str(max_reflections),
    ]
    if task_module:
        command.extend(["--task-module", str(task_module)])
    if variant_hint:
        command.extend(
            [
                "--variant-hint-json",
                json.dumps(
                    variant_hint,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    experiment_candidate = (
        round_plan.get("proposal")
        or round_plan.get("experiment_candidate")
    )
    if (
        experiment_candidate is not None
        and route == "generic_provider_scene_checker_codegen"
    ):
        command.extend(
            [
                "--proposal-json",
                json.dumps(
                    experiment_candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    task_variant_id = round_plan.get("task_variant_id")
    if task_variant_id:
        command.extend(["--variant-id", str(task_variant_id)])
    elif (
        round_plan.get("template_id")
        and round_plan.get("capability_contract") is None
    ):
        # Compatibility for hand-authored legacy plans that predate the
        # capability adapter's template/task-variant identity split.
        command.extend(["--variant-id", str(round_plan["template_id"])])
    if round_plan.get("capability_contract") is not None:
        command.extend(
            [
                "--capability-contract-json",
                json.dumps(
                    round_plan["capability_contract"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    if task_proposal is not None:
        command.extend(
            [
                "--task-proposal-json",
                json.dumps(
                    normalized_task_proposal,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    if route == "official":
        if execution_backend in {"expert", "both"}:
            command.append("--expert")
        if execution_backend in {"act", "both"}:
            command.append("--run-act")
    elif route == "generic_provider_scene_checker_codegen":
        # Generic TaskGen already performs live initial-negative, render, and
        # official-expert-positive preflight inside its single repair loop.
        command.append("--run-act")
    elif route == "provider_scene_checker_codegen":
        # The proposal-derived visual contract checks that both intended
        # blocks are visible before the expert and ACT gates.
        command.extend(["--expert", "--vision-check", "--run-act"])
    else:
        # The bounded generated-task prototype keeps its original expert
        # solvability gate before the ACT policy rollout.
        command.extend(["--expert", "--vision-check", "--run-act"])
    if base_url:
        command.extend(["--base-url", base_url])
    if reviewed_task_registry is not None:
        command.extend(["--reviewed-task-registry", str(reviewed_task_registry)])
    if registration_identity is not None:
        command.extend(
            [
                "--registration-identity-json",
                json.dumps(
                    registration_identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    return command, run_id


def materialize_open_world_round(
    repo_root: Path,
    evaluation_dir: Path,
    *,
    round_number: int,
    candidate: Mapping[str, Any],
    control_execution: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize only the TaskGen and ToolGen stages requested by a candidate."""

    normalized = validate_experiment_candidate(candidate)
    scene_need = normalized["scene_need"]
    checker_need = normalized["checker_need"]
    rule_tool_need = normalized["rule_tool_need"]
    vqa_tool_need = normalized["vqa_tool_need"]
    taskgen_requested = scene_need is not None or checker_need is not None
    toolgen_requested = rule_tool_need is not None
    vqa_tool_requested = vqa_tool_need is not None
    route = (
        "generic_provider_scene_checker_codegen"
        if taskgen_requested
        else "official"
    )
    outcome_metric = (
        "generated_check_success"
        if checker_need is not None
        else "official_check_success"
    )
    deferred_tool_request = {
        "schema_version": 1,
        "task_name": str(normalized["base_task"]),
        "metric": outcome_metric,
        "question": "Fallback only: did the task success predicate pass?",
    }
    tool_bundle = {
        "schema_version": 1,
        "source": (
            "deferred_until_executed_telemetry_schema"
            if toolgen_requested
            else "vqa_only_no_rule_tool_requested"
            if vqa_tool_requested
            else "task_checker_evidence_no_new_tool_requested"
        ),
        "tool_request": deferred_tool_request,
    }
    artifact_dir = (
        evaluation_dir
        / PROPOSAL_MATERIALIZATION
        / f"round_{round_number:02d}"
    )
    write_json(artifact_dir / PROPOSAL_FILENAME, normalized)
    write_json(artifact_dir / "tool_request_bundle.json", tool_bundle)
    execution = deepcopy(dict(control_execution))
    execution["backend"] = "act"
    execution["gates"] = (
        ["ast", "render", "visual_diagnosis", "expert", "act", "toolkit"]
        if taskgen_requested
        else ["render", "act", "toolkit"]
    )
    if toolgen_requested:
        execution["gates"].append("planned_tool")
    if vqa_tool_requested:
        execution["gates"].append("dynamic_vqa")
    execution["gates"].append("aggregate")
    candidate_id = str(normalized["candidate_id"])
    sub_aspect = str(normalized["semantic_concern"]).split(":", 1)[0].strip()

    def need_description(need: Mapping[str, Any] | None) -> str:
        return (
            str(need["description"])
            if need is not None
            else "reuse the official implementation"
        )

    round_plan = {
        "round_id": f"round_{round_number}",
        "template_id": None,
        "candidate_id": candidate_id,
        "proposal": normalized,
        "sub_aspect": sub_aspect,
        "rationale": (
            "Materialize only the Query-derived Task or Tool needs; no catalog "
            "template authorizes this round."
        ),
        "task_instruction": (
            f"{normalized['source_query']}\nScene need: "
            f"{need_description(scene_need)}\nChecker need: "
            f"{need_description(checker_need)}"
        ),
        "task_name": str(normalized["base_task"]),
        "task_module": (
            None
            if taskgen_requested
            else f"envs.{normalized['base_task']}"
        ),
        "telemetry_profile": "balanced_v1",
        "route": route,
        "variant_hint": {},
        "execution": execution,
        "observations": (
            ["scene_alignment", "expert_solvable", "trusted_tools"]
            + (["planned_tool"] if toolgen_requested else [])
            + (["dynamic_vqa"] if vqa_tool_requested else [])
            + ["aggregate"]
        ),
        "tool_request": deepcopy(deferred_tool_request),
        "open_tool_request_deferred": toolgen_requested,
        "vqa_phenomenon_ids": [],
        "semantic_need_execution": {
            "schema_version": 2,
            "candidate_id": candidate_id,
            "task": {
                "requested": scene_need is not None,
                "description": (
                    str(scene_need["description"])
                    if scene_need is not None
                    else None
                ),
                "route": (
                    "generic_provider_scene_checker_codegen"
                    if scene_need is not None
                    else "official_scene_reuse"
                ),
                "status": (
                    "selected" if scene_need is not None else "not_requested"
                ),
            },
            "checker": {
                "requested": checker_need is not None,
                "description": (
                    str(checker_need["description"])
                    if checker_need is not None
                    else None
                ),
                "route": (
                    "provider_written_python"
                    if checker_need is not None
                    else "official_checker_reuse"
                ),
                "status": (
                    "selected" if checker_need is not None else "not_requested"
                ),
            },
            "rule_tool": {
                "requested": rule_tool_need is not None,
                "description": (
                    str(rule_tool_need["description"])
                    if rule_tool_need is not None
                    else None
                ),
                "route": (
                    "after_executed_telemetry_schema"
                    if rule_tool_need is not None
                    else "task_checker_evidence"
                ),
                "status": (
                    "pending"
                    if rule_tool_need is not None
                    else "not_requested"
                ),
            },
            "vqa_tool": {
                "requested": vqa_tool_need is not None,
                "description": (
                    str(vqa_tool_need["description"])
                    if vqa_tool_need is not None
                    else None
                ),
                "route": (
                    "task_owned_or_generated_question"
                    if vqa_tool_need is not None
                    else "not_requested"
                ),
                "status": (
                    "pending"
                    if vqa_tool_need is not None
                    else "not_requested"
                ),
            },
        },
    }
    return round_plan, tool_bundle


def _executed_runtime_task_schema(
    child_dir: Path,
    *,
    task_name: str,
) -> dict[str, Any]:
    schema_paths = sorted(
        (child_dir / "evaluation/telemetry").glob(
            "act/episode_*/schema.json"
        )
    )
    if not schema_paths:
        raise RuntimeError(
            "open ToolGen requires an executed ACT telemetry schema"
        )
    schemas = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in schema_paths
    ]
    canonical = {
        json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for schema in schemas
    }
    if len(canonical) != 1:
        raise RuntimeError(
            "executed ACT episodes expose inconsistent telemetry schemas"
        )
    schema = schemas[0]
    if schema.get("task_name") != task_name:
        raise RuntimeError(
            "executed telemetry schema changed the bound task"
        )
    return schema


def materialize_open_world_tool_request(
    repo_root: Path,
    execution_dir: Path,
    *,
    round_plan: Mapping[str, Any],
    child_dir: Path,
    provider: Any,
    toolgen_model: str,
    reviewed_tool_registry: Path | None = None,
) -> dict[str, Any]:
    """Run ToolGen after TaskGen/ACT using the schema actually recorded."""

    candidate = round_plan.get("proposal") or round_plan.get(
        "experiment_candidate"
    )
    if not isinstance(candidate, Mapping):
        raise RuntimeError(
            "deferred open ToolGen requires a typed Proposal"
        )
    candidate = validate_experiment_candidate(candidate)
    rule_tool_need = candidate["rule_tool_need"]
    if rule_tool_need is None:
        raise RuntimeError(
            "deferred open Rule ToolGen requires rule_tool_need"
        )
    runtime_schema = _executed_runtime_task_schema(
        child_dir,
        task_name=str(candidate["base_task"]),
    )
    episode_dirs = [
        path.parent
        for path in sorted(
            (child_dir / "evaluation/telemetry").glob(
                "act/episode_*/schema.json"
            )
        )
    ]
    run_local_registry = execution_dir.parent.parent / "tool_registry"
    reusable_tool_requests = compatible_run_local_tool_requests(
        run_local_registry,
        task_name=str(candidate["base_task"]),
        episode_dirs=episode_dirs,
    )
    if reviewed_tool_registry is not None:
        reusable_tool_requests.extend(
            compatible_reviewed_tool_requests(
                reviewed_tool_registry,
                task_name=str(candidate["base_task"]),
                episode_dirs=episode_dirs,
            )
        )
    child_manifest_path = child_dir / "manifest.json"
    child_manifest = json.loads(
        child_manifest_path.read_text(encoding="utf-8")
    )
    trusted = child_manifest.get("trusted_tool_evaluation") or {}
    already_measured_metrics = {
        str(result["tool"])
        for episode in trusted.get("episodes", [])
        if isinstance(episode, Mapping)
        for result in (
            episode.get("tool_results")
            if isinstance(episode.get("tool_results"), list)
            else [episode.get("result")]
        )
        if isinstance(result, Mapping)
        and isinstance(result.get("tool"), str)
        and str(result["tool"]).strip()
    }
    outcome_metric = trusted.get("outcome_metric")
    if isinstance(outcome_metric, str) and outcome_metric.strip():
        already_measured_metrics.add(outcome_metric.strip())
    tool_agent = OpenToolRequestAgent(
        repo_root,
        provider,
        model=toolgen_model,
    )
    bundle = tool_agent.propose(
        source_query=str(candidate["source_query"]),
        semantic_concern=str(candidate["semantic_concern"]),
        tool_need=str(rule_tool_need["description"]),
        task_name=str(candidate["base_task"]),
        generated_checker_semantics=bool(
            candidate["checker_need"] is not None
        ),
        runtime_schema=runtime_schema,
        reusable_tool_requests=reusable_tool_requests,
        forbidden_metric_ids=already_measured_metrics,
    )
    artifact_dir = execution_dir / "open_tool_request"
    write_json(artifact_dir / "runtime_schema.json", runtime_schema)
    write_json(artifact_dir / "tool_request_bundle.json", bundle)
    if tool_agent.last_prompt is not None:
        (artifact_dir / "prompt.md").write_text(
            tool_agent.last_prompt,
            encoding="utf-8",
        )
    for index, response in enumerate(tool_agent.last_responses, start=1):
        (artifact_dir / f"response_{index}.txt").write_text(
            response + "\n",
            encoding="utf-8",
        )
    return bundle


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def read_policy_success(result_path: Path) -> float | None:
    if not result_path.is_file():
        return None
    for line in reversed(result_path.read_text(encoding="utf-8").splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    return None


def reuse_bound_child_checker_tool(
    repo_root: Path,
    child_manifest: dict[str, Any],
    output_dir: Path,
    tool_request: dict[str, Any],
) -> dict[str, Any] | None:
    """Expose an already-bound provider checker as the planned Tool evidence.

    A provider-written task checker is executed by the simulator and projected
    into ``trusted_tool_evaluation`` before the parent round starts ToolGen.
    When the route-free ToolProposal asks for that exact same metric, generating
    or routing a second Tool would duplicate one policy episode and can fail on
    an intentionally run-bound metric.  Typed MetricSpec requests remain on the
    normal ToolGen path because equal names alone do not prove equal semantics.
    """

    trusted = child_manifest.get("trusted_tool_evaluation")
    if (
        child_manifest.get("generation_kind")
        not in {
            "provider_scene_checker_codegen",
            "generic_provider_scene_checker_codegen",
        }
        or not isinstance(trusted, Mapping)
        or tool_request.get("schema_version") != 1
        or "metric_spec" in tool_request
        or trusted.get("outcome_metric") != tool_request.get("metric")
        or trusted.get("outcome_authority")
        != "llm_generated_python_ast_validated"
        or not isinstance(trusted.get("tool_retrieval"), Mapping)
        or trusted["tool_retrieval"].get("route")
        != "bound_llm_generated_checker"
    ):
        return None

    metric = str(tool_request["metric"])
    episodes = trusted.get("episodes")
    binding = trusted.get("outcome_binding")
    source_artifact = trusted.get("artifact")
    if (
        tool_request.get("task_name") != child_manifest.get("task_name")
        or not isinstance(binding, Mapping)
        or binding.get("metric") != metric
        or binding.get("authority") != "llm_generated_python_ast_validated"
        or binding.get("task_module") != child_manifest.get("task_module")
        or binding.get("module_sha256")
        != child_manifest.get("candidate_module_sha256")
        or not isinstance(source_artifact, str)
        or not source_artifact.strip()
        or not isinstance(episodes, list)
        or not episodes
        or trusted.get("episode_count") != len(episodes)
    ):
        raise RuntimeError(
            "provider checker metric matched the ToolProposal but its trusted "
            "execution binding is incomplete"
        )
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise RuntimeError("provider checker Tool episode must be an object")
        results = _episode_tool_results(episode)
        if len(results) != 1:
            raise RuntimeError(
                "provider checker Tool episode must contain exactly one result"
            )
        result = results[0]
        details = result.get("details")
        if (
            result.get("tool") != metric
            or not isinstance(result.get("value"), bool)
            or not isinstance(result.get("passed"), bool)
            or result.get("passed") != result.get("value")
            or episode.get("role") != "policy_under_evaluation"
            or not isinstance(details, Mapping)
            or details.get("authority") != "llm_generated_python_ast_validated"
            or details.get("task_module") != child_manifest.get("task_module")
            or details.get("module_sha256") != binding.get("module_sha256")
        ):
            raise RuntimeError(
                "provider checker ToolResult does not match its task/module authority"
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    tool_execution_path = output_dir / "tool_execution.json"
    request_path = output_dir / "tool_request.json"
    route_path = output_dir / "route_decision.json"
    route_decision = {
        "schema_version": 1,
        "status": "resolved",
        "matching_policy": "exact_bound_child_metric",
        "requested_route": "auto",
        "resolved_route": "bound_child_trusted_checker",
        "task_name": tool_request.get("task_name"),
        "metric": metric,
        "exact_match": True,
        "matched_registry": "child_task_checker",
        "reference_tool": metric,
        "provider_required": False,
        "provider_called": False,
        "reason": (
            "the executed provider-written task checker already produced this "
            "exact metric for the same bound ACT episode"
        ),
    }
    def relative(path: Path) -> str:
        return path.relative_to(repo_root).as_posix()

    evaluation = {
        "schema_version": 1,
        "status": "passed",
        "requested_route": "auto",
        "route": "bound_child_trusted_checker",
        "reference_tool": metric,
        "tool_request": deepcopy(tool_request),
        "route_decision": route_decision,
        "source": {
            "scope": "bound_child_task_checker",
            "artifact": source_artifact,
            "aggregate_artifact": trusted.get("aggregate_artifact"),
            "authority": trusted.get("outcome_authority"),
        },
        "episodes": deepcopy(episodes),
        "validation": {
            "status": "passed",
            "provider_called": False,
            "exact_metric_match": True,
            "episode_count": len(episodes),
            "authority": trusted.get("outcome_authority"),
        },
        "artifacts": {
            "tool_request": relative(request_path),
            "route_decision": relative(route_path),
            "tool_execution": relative(tool_execution_path),
            "source_execution": trusted.get("artifact"),
        },
    }
    write_json(request_path, tool_request)
    write_json(route_path, route_decision)
    write_json(tool_execution_path, evaluation)
    return evaluation


def compact_tool_evaluation(
    tool_evaluation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep planned Tool evidence compact while preserving ACT/expert roles."""

    if not tool_evaluation:
        return None
    compact_episodes: list[dict[str, Any]] = []
    for item in tool_evaluation.get("episodes", []):
        if not isinstance(item, Mapping):
            continue
        for result in _episode_tool_results(item):
            compact_episodes.append(
                {
                    "policy_name": item.get("policy_name"),
                    "seed": item.get("seed"),
                    "role": item.get("role"),
                    "metric": (
                        result.get("tool")
                        or tool_evaluation.get("reference_tool")
                    ),
                    "value": result.get("value"),
                    "unit": result.get("unit"),
                    "passed": result.get("passed"),
                    "evidence_steps": result.get("evidence_steps", []),
                    "details": result.get("details", {}),
                }
            )
    return {
        "status": tool_evaluation.get("status"),
        "requested_route": tool_evaluation.get("requested_route"),
        "route": tool_evaluation.get("route"),
        "reference_tool": tool_evaluation.get("reference_tool"),
        "route_decision": tool_evaluation.get("route_decision", {}),
        "source": tool_evaluation.get("source", {}),
        "episodes": compact_episodes,
        "validation": tool_evaluation.get("validation", {}),
    }


def _aggregate_sources(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    tool_evaluation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build one de-duplicated set of episode ToolResult sources."""

    context = {
        "round_id": round_plan["round_id"],
        "variant": round_plan.get("template_id")
        or round_plan.get("sub_aspect")
        or round_plan.get("route"),
    }
    sources: list[dict[str, Any]] = []
    trusted = child_manifest.get("trusted_tool_evaluation") or {}
    trusted_tools = {
        result.get("tool")
        for episode in trusted.get("episodes", [])
        if isinstance(episode, Mapping)
        for result in _episode_tool_results(episode)
        if result.get("tool")
    }
    if trusted.get("episodes"):
        sources.append(
            {
                **trusted,
                "context": {
                    **context,
                    "source_artifact": trusted.get("artifact"),
                },
            }
        )
    if tool_evaluation and tool_evaluation.get("episodes"):
        request = tool_evaluation.get("tool_request") or tool_evaluation.get(
            "tool_spec", {}
        )
        metric = request.get("metric") if isinstance(request, dict) else None
        if metric not in trusted_tools:
            sources.append(
                {
                    "tool_execution": tool_evaluation,
                    "context": {
                        **context,
                        "source_artifact": tool_evaluation.get("artifacts", {}).get(
                            "tool_execution"
                        ),
                    },
                }
            )
    return sources


def aggregate_round_results(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    tool_evaluation: dict[str, Any] | None,
    output_path: Path,
) -> dict[str, Any]:
    sources = _aggregate_sources(round_plan, child_manifest, tool_evaluation)
    if not sources:
        result = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "no episode ToolResult rows were available",
            "metrics": [],
        }
        write_json(output_path, result)
        return result
    return aggregate_tool_executions(sources, output_path=output_path)


def aggregate_evaluation_results(
    round_runs: list[dict[str, Any]], output_path: Path
) -> dict[str, Any]:
    sources = [
        source
        for item in round_runs
        for source in _aggregate_sources(
            item["round_plan"], item["child_manifest"], item["tool_evaluation"]
        )
    ]
    if not sources:
        result = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "no completed round ToolResult rows were available",
            "metrics": [],
        }
        write_json(output_path, result)
        return result
    return aggregate_tool_executions(sources, output_path=output_path)


def _execution_vqa_video_contract(
    episode_dir: Path,
    *,
    execution_backend: str,
) -> tuple[bool, dict[str, Any], str]:
    """Validate backend-specific video evidence before it reaches VQA."""

    metadata_path = episode_dir / "episode.json"
    metadata: dict[str, Any] = {}
    metadata_error: str | None = None
    if metadata_path.is_file():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
            else:
                metadata_error = "episode.json is not a JSON object"
        except (OSError, json.JSONDecodeError) as exc:
            metadata_error = f"episode.json is unreadable: {type(exc).__name__}"
    else:
        metadata_error = "episode.json is missing"

    if not (episode_dir / "video.mp4").is_file():
        return False, metadata, "is missing video.mp4"
    if (episode_dir / "video.mp4").stat().st_size <= 0:
        return False, metadata, "has an empty video.mp4"
    if metadata_error:
        return False, metadata, metadata_error
    if (metadata.get("artifacts") or {}).get("video") != "video.mp4":
        return False, metadata, "does not declare artifacts.video=video.mp4"
    if execution_backend != "expert":
        return True, metadata, ""

    visual_capture = metadata.get("visual_capture") or {}
    if visual_capture.get("status") != "completed":
        return False, metadata, "does not declare a completed visual_capture"
    return True, metadata, ""


def _policy_episode_for_execution_vqa(
    child_manifest: dict[str, Any],
    child_dir: Path,
    *,
    execution_backend: str,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]] | None:
    """Select evidence from the backend that this round actually evaluated."""

    desired_policy = "expert" if execution_backend == "expert" else "act"
    trusted = child_manifest.get("trusted_tool_evaluation") or {}
    candidates = sorted(
        (
            episode
            for episode in trusted.get("episodes", [])
            if str(episode.get("policy_name", "")).casefold() == desired_policy
        ),
        key=lambda episode: (
            not _execution_vqa_video_contract(
                child_dir
                / "evaluation/telemetry"
                / str(episode.get("episode_dir") or ""),
                execution_backend=execution_backend,
            )[0],
            int(episode.get("seed") or 0),
            str(episode.get("episode_dir") or ""),
        ),
    )
    if not candidates:
        return None
    episode = candidates[0]
    episode_dir = child_dir / "evaluation/telemetry" / episode["episode_dir"]
    return episode_dir, episode, list(episode.get("tool_results", []))


def _same_telemetry_episode(
    candidate: dict[str, Any], representative: dict[str, Any]
) -> bool:
    """Match generated and Trusted Tool rows to one physical rollout."""

    candidate_dir = candidate.get("episode_dir")
    representative_dir = representative.get("episode_dir")
    if candidate_dir and representative_dir:
        return str(candidate_dir) == str(representative_dir)
    return (
        candidate.get("seed") == representative.get("seed")
        and str(candidate.get("policy_name", "")).casefold()
        == str(representative.get("policy_name", "")).casefold()
    )


def run_round_execution_vqa(
    *,
    repo_root: Path,
    child_manifest: dict[str, Any],
    child_dir: Path,
    tool_evaluation: dict[str, Any] | None,
    execution_dir: Path,
    provider: Any,
    model: str,
    round_plan: dict[str, Any] | None = None,
    reviewed_vqa_registry: Path | None = None,
) -> dict[str, Any]:
    """Run VQA on official-expert or ACT evidence without mixing their roles."""

    semantic_needs = (round_plan or {}).get("semantic_need_execution")
    vqa_need = (
        semantic_needs.get("vqa_tool")
        if isinstance(semantic_needs, Mapping)
        else None
    )
    generated_vqa_specs = None
    generated_vqa_ids = None
    if (
        isinstance(vqa_need, Mapping)
        and vqa_need.get("requested") is True
        and isinstance(vqa_need.get("description"), str)
        and vqa_need["description"].strip()
    ):
        description = " ".join(vqa_need["description"].split())
        question = (
            "Does the rollout visibly show whether "
            + description.rstrip("?.。？！")
            + "?"
        )[:240]
        if not question.endswith("?"):
            question = question[:239].rstrip("?.。？！") + "?"
        phenomenon_id = (
            "run_local.query_"
            + hashlib.sha256(description.encode("utf-8")).hexdigest()[:12]
        )
        generated_vqa_specs = [
            {
                "id": phenomenon_id,
                "question_type": "visible_state_change",
                "target_role": "manipulated_object",
                "question": question,
                "visual_scope": "rollout_change",
                "numeric_authority": "no_numeric_oracle",
            }
        ]
        generated_vqa_ids = [phenomenon_id]
    proposal = ((round_plan or {}).get("tool_proposal") or {})
    proposal_vqa_explicit = bool(
        proposal.get("vqa_phenomenon_ids")
        or proposal.get("vqa_question_specs")
    )
    query = build_execution_vqa_query(
        task_name=(
            str((round_plan or {}).get("task_name") or child_manifest.get("task_name"))
            if (round_plan or {}).get("task_name") or child_manifest.get("task_name")
            else None
        ),
        template_id=(round_plan or {}).get("template_id"),
        sub_aspect=(round_plan or {}).get("sub_aspect"),
        tool_contract=(round_plan or {}).get("tool_request"),
        proposed_phenomenon_ids=(
            proposal.get("vqa_phenomenon_ids")
            if proposal_vqa_explicit
            else generated_vqa_ids
        ),
        proposed_question_specs=(
            proposal.get("vqa_question_specs")
            if proposal_vqa_explicit
            else generated_vqa_specs
        ),
        reviewed_registry_dir=reviewed_vqa_registry,
    )
    write_json(execution_dir / "execution_vqa_query.json", query)
    route = (round_plan or {}).get("route")
    execution_backend = round_execution_backend(round_plan or {"route": route})
    evidence_backend = "expert" if execution_backend == "expert" else "act"
    selected = _policy_episode_for_execution_vqa(
        child_manifest,
        child_dir,
        execution_backend=evidence_backend,
    )
    if selected is None:
        backend = "expert" if evidence_backend == "expert" else "ACT"
        result = {
            "schema_version": 1,
            "status": "skipped" if evidence_backend == "expert" else "failed",
            "reason": f"no completed {backend} telemetry episode was available",
            "evidence_conflict": False,
            "query": query,
        }
        write_json(
            execution_dir
            / (
                "execution_vqa_skipped.json"
                if evidence_backend == "expert"
                else "execution_vqa_error.json"
            ),
            result,
        )
        return result
    episode_dir, representative, numeric_results = selected
    representative_path = str(episode_dir.relative_to(repo_root))
    video_ready, metadata, video_reason = _execution_vqa_video_contract(
        episode_dir,
        execution_backend=evidence_backend,
    )
    if not video_ready:
        backend = "expert" if evidence_backend == "expert" else "ACT"
        result = {
            "schema_version": 1,
            "status": "skipped" if evidence_backend == "expert" else "failed",
            "reason": f"completed {backend} telemetry episode {video_reason}",
            "representative_episode": representative_path,
            "evidence_conflict": False,
            "query": query,
            "visual_capture": metadata.get("visual_capture"),
        }
        write_json(
            execution_dir
            / (
                "execution_vqa_skipped.json"
                if evidence_backend == "expert"
                else "execution_vqa_error.json"
            ),
            result,
        )
        return result
    known_tools = {item.get("tool") for item in numeric_results}
    desired_role = (
        "expert_validation"
        if evidence_backend == "expert"
        else "policy_under_evaluation"
    )
    for episode in (tool_evaluation or {}).get("episodes", []):
        if episode.get("role") != desired_role:
            continue
        if not _same_telemetry_episode(episode, representative):
            continue
        result = episode.get("result", {})
        if result.get("tool") not in known_tools:
            numeric_results.append(result)
            known_tools.add(result.get("tool"))
    try:
        scene_seed = (child_manifest.get("scene_validation") or {}).get("seed")
        representative_seed = representative.get("seed")
        reference_scene = child_dir / "evidence/initial_head.png"
        if (
            scene_seed is not None
            and representative_seed is not None
            and int(scene_seed) != int(representative_seed)
        ):
            # Never label an image from a skipped seed as the rollout's
            # reference scene. The rollout video remains valid evidence.
            reference_scene = None
        result = run_execution_vqa(
            provider=provider,
            model=model,
            video_path=episode_dir / "video.mp4",
            output_dir=execution_dir / "execution_vqa",
            numeric_tool_results=numeric_results,
            events_path=episode_dir / "events.jsonl",
            semantic_trace_path=episode_dir / "semantic_trace.npz",
            reference_scene=reference_scene,
            query=query,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "representative_episode": representative_path,
            "evidence_conflict": False,
            "query": query,
        }
        write_json(execution_dir / "execution_vqa_error.json", result)
        return result
    result["status"] = "passed"
    result["representative_episode"] = representative_path
    result["artifacts"] = {
        key: (
            str(Path(value).resolve().relative_to(repo_root))
            if isinstance(value, str)
            and Path(value).is_absolute()
            and Path(value).resolve().is_relative_to(repo_root)
            else value
        )
        for key, value in result.get("artifacts", {}).items()
    }
    write_json(execution_dir / "execution_vqa/execution_vqa.json", result)
    return result


def compact_execution_vqa(
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not result:
        return None
    return {
        "status": result.get("status"),
        "model_requested": result.get("model_requested"),
        "representative_episode": result.get("representative_episode"),
        "evidence_conflict": bool(result.get("evidence_conflict")),
        "observation": result.get("observation"),
        "selected_frames": result.get("selection", {}).get("selected_frames", []),
        "artifacts": result.get("artifacts", {}),
        "reason": result.get("reason"),
        "query": result.get("query"),
    }


def taskgen_ast_gate_passed(static_validation: Mapping[str, Any]) -> bool:
    """Accept either legacy TaskGen AST output or provider scene+checker AST."""

    legacy = static_validation.get("load_actors_ast") or {}
    if isinstance(legacy, Mapping) and legacy.get("valid") is True:
        return True
    provider = static_validation.get("provider_scene_checker") or {}
    return bool(
        isinstance(provider, Mapping)
        and provider.get("valid") is True
        and provider.get("model_written_python") is True
        and provider.get("restricted_success_spec_compiler_used") is False
        and isinstance(provider.get("ast_policy"), str)
        and provider["ast_policy"].strip()
    )


def normalize_outcome_semantics(
    trusted_tool_evaluation: Mapping[str, Any],
    task_artifact_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare generated and official predicates without conflating semantics."""

    raw_official_equivalent = task_artifact_summary.get(
        "success_official_equivalent"
    )
    outcome_authority = trusted_tool_evaluation.get("outcome_authority")
    official_authority = outcome_authority in {
        "official_check_success",
        "official_check_success_reused",
    }
    official_equivalent = (
        raw_official_equivalent
        if isinstance(raw_official_equivalent, bool)
        else (
            True
            if official_authority
            else None
        )
    )
    episodes: list[dict[str, Any]] = []
    for episode in trusted_tool_evaluation.get("episodes", []):
        if not isinstance(episode, Mapping):
            continue
        for result in _episode_tool_results(episode):
            details = result.get("details")
            if not isinstance(details, Mapping):
                continue
            raw_generated = details.get("generated_checker_success")
            raw_official = details.get("official_success")
            raw_official_core = details.get("official_core_predicate_satisfied")
            generated = raw_generated if isinstance(raw_generated, bool) else None
            official = raw_official if isinstance(raw_official, bool) else None
            official_core = (
                raw_official_core if isinstance(raw_official_core, bool) else None
            )
            if generated is None and official is None and official_core is None:
                continue

            if (
                generated is not None
                and outcome_authority
                != "llm_generated_python_ast_validated"
            ):
                status = "non_comparable"
                reason = "generated_result_has_untrusted_outcome_authority"
            elif generated is not None and official_equivalent is None:
                status = "non_comparable"
                reason = "generated_checker_equivalence_not_declared"
            elif (
                generated is not None
                and official is not None
                and official_equivalent is True
            ):
                status = (
                    "equivalent_agreement"
                    if generated == official
                    else "conflict"
                )
                reason = (
                    "generated_and_official_equivalent_predicates_agree"
                    if generated == official
                    else "generated_and_official_equivalent_predicates_disagree"
                )
            elif generated is not None and official_equivalent is False:
                if generated is True and official_core is False:
                    status = "conflict"
                    reason = "generated_success_without_official_core_predicate"
                elif official_core is not None:
                    status = "expected_semantic_extension"
                    reason = (
                        "generated_checker_adds_constraints_beyond_official_core"
                    )
                else:
                    status = "non_comparable"
                    reason = "non_equivalent_checker_has_no_official_core_projection"
            elif generated is None and official is not None:
                status = "official_only"
                reason = "no_generated_checker_result"
            else:
                status = "non_comparable"
                reason = "insufficient_dual_predicate_results"

            episodes.append(
                {
                    "seed": episode.get("seed"),
                    "generated_checker_success": generated,
                    "official_success": official,
                    "official_core_predicate_satisfied": official_core,
                    "official_equivalent": official_equivalent,
                    "status": status,
                    "reason": reason,
                }
            )

    statuses = {item["status"] for item in episodes}
    if (
        not episodes
        and official_authority
        and official_equivalent is True
    ):
        status = "official_only"
        reason_codes = ["official_outcome_has_no_generated_checker"]
    elif "conflict" in statuses:
        status = "conflict"
        reason_codes = list(
            dict.fromkeys(item["reason"] for item in episodes)
        )
    elif "expected_semantic_extension" in statuses:
        status = "expected_semantic_extension"
        reason_codes = list(
            dict.fromkeys(item["reason"] for item in episodes)
        )
    elif statuses == {"equivalent_agreement"}:
        status = "equivalent_agreement"
        reason_codes = list(
            dict.fromkeys(item["reason"] for item in episodes)
        )
    elif statuses == {"official_only"}:
        status = "official_only"
        reason_codes = list(
            dict.fromkeys(item["reason"] for item in episodes)
        )
    else:
        status = "non_comparable"
        reason_codes = list(
            dict.fromkeys(item["reason"] for item in episodes)
        )
    return {
        "schema_version": 1,
        "status": status,
        "evidence_conflict": status == "conflict",
        "official_equivalent": official_equivalent,
        "outcome_authority": outcome_authority,
        "episodes": episodes,
        "reason_codes": reason_codes,
    }


def summarize_round(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    child_dir: Path,
    tool_evaluation: dict[str, Any] | None = None,
    aggregate_result: dict[str, Any] | None = None,
    execution_vqa: dict[str, Any] | None = None,
    taskgen_returncode: int = 0,
) -> dict[str, Any]:
    capability_contract = validate_round_capability_contract(round_plan)
    scene = child_manifest.get("scene_validation", {})
    vision = child_manifest.get("vision_validation", {})
    act = child_manifest.get("act_evaluation", {})
    expert = scene.get("expert", {})
    positions = child_manifest.get("position_samples", {})
    position_metrics = positions.get("metrics", {})
    variant_samples = positions.get("samples", [])
    observed_bell_ids = sorted(
        {
            int(item["bell_id"])
            for item in variant_samples
            if isinstance(item, dict)
            and not isinstance(item.get("bell_id"), bool)
            and isinstance(item.get("bell_id"), int)
        }
    )
    clutter_counts = [
        int(item["clutter_count"])
        for item in variant_samples
        if isinstance(item, dict)
        and not isinstance(item.get("clutter_count"), bool)
        and isinstance(item.get("clutter_count"), int)
    ]
    policy_success = read_policy_success(child_dir / "evaluation/_result.txt")
    trusted_tools = compact_trusted_tools(child_manifest)
    trusted_tool_evaluation = child_manifest.get("trusted_tool_evaluation") or {}
    task_artifact_summary = child_manifest.get("task_artifact_summary") or {}
    is_official = round_plan.get("route") == "official"
    is_generic_provider = (
        round_plan.get("route")
        == "generic_provider_scene_checker_codegen"
    )
    execution_backend = round_execution_backend(round_plan)
    uses_act = execution_backend in {"act", "both"}
    uses_expert = execution_backend in {"expert", "both"}
    outcome_semantics = normalize_outcome_semantics(
        trusted_tool_evaluation,
        task_artifact_summary,
    )
    static = child_manifest.get("static_validation") or {}
    policy_outcome = {
        "metric": trusted_tool_evaluation.get("outcome_metric"),
        "authority": trusted_tool_evaluation.get("outcome_authority"),
        "binding": deepcopy(trusted_tool_evaluation.get("outcome_binding")),
        "value": policy_success if uses_act else None,
        "official_equivalent": bool(
            task_artifact_summary.get("success_official_equivalent", True)
        ),
        "execution_scope": task_artifact_summary.get(
            "success_execution_scope", "official_equivalent"
        ),
        "outcome_semantics": deepcopy(outcome_semantics),
    }
    if uses_act:
        actual_seeds = [int(value) for value in act.get("actual_seeds", [])]
    else:
        actual_seeds = [
            int(item["seed"])
            for item in scene.get("expert_batch", {}).get("episodes", [])
            if item.get("seed") is not None
        ]
    gate_status = {
        "variant_spec": (
            (child_manifest.get("capability_contract_validation") or {}).get(
                "status"
            )
            == "passed"
        ),
        "ast": taskgen_ast_gate_passed(static),
        "render": bool(scene.get("render_success")),
        "rule": bool((scene.get("rule_check") or {}).get("passed")),
        "scene_variant": bool(positions.get("passed")),
        "vision": bool(vision.get("passed")),
        "expert": bool((scene.get("expert_batch") or expert).get("passed")),
        "act": bool((not uses_act and is_official) or act.get("passed")),
        "toolkit": bool(
            (child_manifest.get("trusted_tool_evaluation") or {}).get(
                "episode_count"
            )
        ),
        "planned_tool": bool(
            tool_evaluation and tool_evaluation.get("status") == "passed"
        ),
        "aggregate": bool(
            aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
        ),
        "execution_vqa": bool(
            execution_vqa
            and (
                execution_vqa.get("status") == "passed"
                or (
                    not uses_act
                    and execution_vqa.get("status") == "skipped"
                )
            )
        ),
    }
    required_gates = (
        list(capability_contract["required_gates"])
        if capability_contract is not None
        else []
    )
    required_gate_status = {
        "required": required_gates,
        "by_gate": {gate: bool(gate_status.get(gate, False)) for gate in required_gates},
    }
    required_gate_status["passed"] = all(
        required_gate_status["by_gate"].values()
    )
    if is_official:
        expert_batch = scene.get("expert_batch") or expert
        pipeline_passed = bool(
            child_manifest.get("status")
            == ("completed" if uses_act else "completed_without_act")
            and taskgen_returncode == 0
            and scene.get("render_success")
            and scene.get("rule_check", {}).get("passed")
            and (not uses_expert or expert_batch.get("passed"))
            and (not uses_act or act.get("passed"))
            and child_manifest.get("trusted_tool_evaluation", {}).get("episode_count")
            and tool_evaluation
            and tool_evaluation.get("status") == "passed"
            and aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
            and execution_vqa
            and execution_vqa.get("status") in {"passed", "skipped"}
        )
    elif is_generic_provider:
        preflight = scene.get("generic_preflight") or {}
        fixtures = preflight.get("checker_fixtures") or []
        acceptance = child_manifest.get("task_generation_acceptance") or {}
        visual_required = acceptance.get(
            "visual_self_check_required", True
        )
        pipeline_passed = bool(
            child_manifest.get("status") == "completed"
            and taskgen_returncode == 0
            and taskgen_ast_gate_passed(static)
            and scene.get("render_success")
            and scene.get("rule_check", {}).get("passed")
            and expert.get("passed")
            and preflight.get("render_passed") is True
            and preflight.get("expert_passed") is True
            and preflight.get("scene_change_passed") is True
            and (
                not visual_required
                or (
                    vision.get("status") == "passed"
                    and vision.get("passed") is True
                )
            )
            and fixtures
            and all(item.get("passed") is True for item in fixtures)
            and positions.get("passed")
            and act.get("passed")
            and tool_evaluation
            and tool_evaluation.get("status") == "passed"
            and aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
            and execution_vqa
            and execution_vqa.get("status") in {"passed", "skipped"}
        )
    else:
        # Generated rounds keep their expert, visual, and task-specific
        # position gates while ACT remains the policy under evaluation.
        pipeline_passed = bool(
            child_manifest.get("status") == "completed"
            and taskgen_returncode == 0
            and scene.get("rule_check", {}).get("passed")
            and vision.get("passed")
            and expert.get("passed")
            and positions.get("passed")
            and act.get("passed")
            and tool_evaluation
            and tool_evaluation.get("status") == "passed"
            and aggregate_result
            and str(aggregate_result.get("status", "")).startswith("passed")
            and execution_vqa
            and execution_vqa.get("status") in {"passed", "skipped"}
        )
    if capability_contract is not None:
        pipeline_passed = bool(pipeline_passed and required_gate_status["passed"])
    implementation_trace = child_manifest.get("implementation_trace")
    if (
        not isinstance(implementation_trace, Mapping)
        and isinstance(
            round_plan.get("proposal")
            or round_plan.get("experiment_candidate"),
            Mapping,
        )
    ):
        implementation_trace = build_implementation_trace(
            round_plan.get("proposal")
            or round_plan["experiment_candidate"]
        )
    if isinstance(implementation_trace, Mapping):
        semantic_needs = round_plan.get("semantic_need_execution")
        rule_need = (
            semantic_needs.get("rule_tool")
            if isinstance(semantic_needs, Mapping)
            else None
        )
        checker_need = (
            semantic_needs.get("checker")
            if isinstance(semantic_needs, Mapping)
            else None
        )
        vqa_need = (
            semantic_needs.get("vqa_tool")
            if isinstance(semantic_needs, Mapping)
            else None
        )
        rule_requested = bool(
            isinstance(rule_need, Mapping)
            and rule_need.get("requested") is True
        )
        checker_requested = bool(
            isinstance(checker_need, Mapping)
            and checker_need.get("requested") is True
        )
        vqa_requested = bool(
            isinstance(vqa_need, Mapping)
            and vqa_need.get("requested") is True
        )
        implementation_trace = advance_implementation_trace_with_tool(
            implementation_trace,
            tool_evaluation,
            # The default task checker remains the Rule observation for
            # scene-only rounds.  VQA-only rounds must not depend on it.
            rule_required=bool(
                not isinstance(semantic_needs, Mapping)
                or rule_requested
                or checker_requested
                or not vqa_requested
            ),
            vqa_evaluation=execution_vqa,
            vqa_required=vqa_requested,
        )
    summary = {
        "round_id": round_plan["round_id"],
        "variant_id": (
            round_plan.get("task_variant_id") or round_plan.get("template_id")
        ),
        "template_id": round_plan.get("template_id"),
        "capability_id": round_plan.get("capability_id"),
        "capability_contract": round_plan.get("capability_contract"),
        "semantic_need_execution": deepcopy(
            round_plan.get("semantic_need_execution")
        ),
        "required_gate_status": required_gate_status,
        "sub_aspect": round_plan["sub_aspect"],
        "task_instruction": round_plan["task_instruction"],
        "route": round_plan["route"],
        "taskgen_run_id": child_manifest.get("run_id"),
        "taskgen_returncode": taskgen_returncode,
        "execution": round_plan["execution"],
        "observations": {
            "execution_backend": {
                "expert": "expert",
                "act": "ACT",
                "both": "ACT+expert",
            }[execution_backend],
            "requested_seeds": [
                int(value) for value in round_plan["execution"].get("seeds", [])
            ],
            "actual_seeds": actual_seeds,
            "scene_alignment": bool(scene.get("rule_check", {}).get("passed")),
            "observed_color": vision.get("observed_color"),
            "bell_visible": vision.get("bell_visible"),
            "position_authority": vision.get("position_authority"),
            "expert_solvable": (
                bool((scene.get("expert_batch") or expert).get("passed"))
                if uses_expert or not is_official
                else None
            ),
            "act_pipeline_status": bool(act.get("passed")) if uses_act else None,
            "policy_success": policy_success if uses_act else None,
            "policy_outcome": policy_outcome,
            "outcome_semantics": outcome_semantics,
            "semantic_need_execution": deepcopy(
                round_plan.get("semantic_need_execution")
            ),
            "position_samples": positions.get("samples", []),
            "position_metrics": position_metrics,
            "controlled_axis": positions.get("controlled_axis"),
            "variant_samples": variant_samples,
            "variant_metrics": position_metrics,
            "observed_bell_ids": observed_bell_ids,
            "bell_instance_id": (
                observed_bell_ids[0] if len(observed_bell_ids) == 1 else None
            ),
            "scene_clutter": {
                "expected": bool(position_metrics.get("expected_clutter")),
                "counts": clutter_counts,
                "all_matched": position_metrics.get("all_clutter_matched"),
                "authority": (
                    "simulator_task_info:cluttered_table_info"
                    if clutter_counts
                    else None
                ),
            },
            "trusted_tools": trusted_tools,
            "planned_tool": compact_tool_evaluation(tool_evaluation),
            "aggregate": compact_aggregate_result(aggregate_result),
            "execution_vqa": compact_execution_vqa(execution_vqa),
            "implementation_trace": implementation_trace,
            "required_gate_status": required_gate_status,
        },
        "pipeline_passed": pipeline_passed,
        "interpretation": (
            "任务路由与执行后端分别记录；ACT 策略结果和流水线状态分开报告，" "策略失败不会被误记为 pipeline failure。"
        ),
    }
    summary["observations"]["evidence_aggregate"] = (
        build_evidence_aggregate(round_plan, summary)
    )
    return summary


def execute_round(
    repo_root: Path,
    evaluation_dir: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    *,
    text_model: str,
    vision_model: str,
    base_url: str | None,
    gpu: int,
    max_reflections: int,
    provider: Any,
    toolgen_model: str,
    telemetry_profile: str = "balanced_v1",
    reviewed_task_registry: Path | None = None,
    reviewed_tool_registry: Path | None = None,
    reviewed_vqa_registry: Path | None = None,
    registration_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any], int,]:
    round_id = round_plan["round_id"]
    command, run_id = build_taskgen_command(
        repo_root,
        evaluation_id,
        round_plan,
        text_model=text_model,
        vision_model=vision_model,
        base_url=base_url,
        gpu=gpu,
        max_reflections=max_reflections,
        telemetry_profile=telemetry_profile,
        reviewed_task_registry=reviewed_task_registry,
        registration_identity=registration_identity,
        run_id_suffix="",
    )
    execution_dir = evaluation_dir / "execution" / round_id
    write_json(
        execution_dir / "taskgen_command.json",
        {"command": command, "child_run_id": run_id},
    )
    update_manifest(
        evaluation_dir,
        status=f"executing_{round_id}",
        active_child_run_id=run_id,
    )
    returncode = run_logged(
        command,
        cwd=repo_root,
        log_path=execution_dir / "taskgen.log",
    )
    child_dir = repo_root / "mea/generated_tasks" / run_id
    child_manifest_path = child_dir / "manifest.json"
    if not child_manifest_path.is_file():
        raise RuntimeError(f"child TaskGen manifest 不存在: {child_manifest_path}")
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    if registration_identity is not None and child_manifest.get(
        "registration_identity"
    ) != registration_identity:
        raise RuntimeError(
            f"child registration identity mismatch: {run_id}"
        )
    write_json(
        execution_dir / "child_run.json",
        {
            "run_id": run_id,
            "returncode": returncode,
            "manifest_path": str(child_manifest_path.relative_to(repo_root)),
            "status": child_manifest.get("status"),
        },
    )
    if (
        child_manifest.get("status")
        in {
            "completed",
            "completed_without_act",
        }
        and returncode == 0
    ):
        tool_kwargs: dict[str, Any] = {
            "provider": provider,
            "model": toolgen_model,
        }
        if reviewed_tool_registry is not None:
            tool_kwargs["reviewed_registry_dir"] = reviewed_tool_registry
        if round_plan.get("open_tool_request_deferred") is True:
            tool_bundle = materialize_open_world_tool_request(
                repo_root,
                execution_dir,
                round_plan=round_plan,
                child_dir=child_dir,
                provider=provider,
                toolgen_model=toolgen_model,
                reviewed_tool_registry=reviewed_tool_registry,
            )
            round_plan["tool_request"] = deepcopy(
                tool_bundle["tool_request"]
            )
            round_plan["open_tool_request_deferred"] = False
            semantic_execution = round_plan.get("semantic_need_execution")
            if isinstance(semantic_execution, dict):
                tool_execution = semantic_execution.get("rule_tool")
                if isinstance(tool_execution, dict):
                    tool_execution.update(
                        {
                            "route": route_tool_request(
                                tool_bundle["tool_request"]
                            )["route_decision"]["resolved_route"],
                            "status": "selected",
                            "request_artifact": str(
                                (
                                    execution_dir
                                    / "open_tool_request/"
                                    "tool_request_bundle.json"
                                ).relative_to(repo_root)
                            ).replace("\\", "/"),
                        }
                    )
        proposed_request = (
            tool_request_from_proposal(round_plan["tool_proposal"])
            if round_plan.get("tool_proposal") is not None
            else round_plan["tool_request"]
        )
        if round_plan.get("task_proposal") is not None:
            tool_kwargs["task_proposal"] = round_plan["task_proposal"]
        planned_tool_dir = execution_dir / "planned_tool"
        tool_evaluation = reuse_bound_child_checker_tool(
            repo_root,
            child_manifest,
            planned_tool_dir,
            proposed_request,
        )
        if tool_evaluation is None:
            tool_evaluation = execute_tool_request(
                repo_root,
                child_dir,
                planned_tool_dir,
                proposed_request,
                **tool_kwargs,
            )
    else:
        skip_reason = (
            f"child TaskGen exited with code {returncode}"
            if returncode != 0
            else "child TaskGen pipeline did not complete"
        )
        tool_evaluation = {
            "schema_version": 1,
            "status": "skipped",
            "requested_route": "auto",
            "route": None,
            "reference_tool": None,
            "tool_request": (
                tool_request_from_proposal(round_plan["tool_proposal"])
                if round_plan.get("tool_proposal") is not None
                else round_plan["tool_request"]
            ),
            "route_decision": {
                "status": "skipped",
                "requested_route": "auto",
                "resolved_route": None,
                "reason": skip_reason,
                "provider_required": None,
                "provider_called": False,
            },
            "source": {},
            "episodes": [],
            "validation": {"reason": skip_reason},
            "artifacts": {},
        }
        write_json(execution_dir / "planned_tool_skipped.json", tool_evaluation)
    semantic_execution = round_plan.get("semantic_need_execution")
    if isinstance(semantic_execution, dict):
        rule_execution = semantic_execution.get("rule_tool")
        if (
            isinstance(rule_execution, dict)
            and rule_execution.get("requested") is True
        ):
            route_decision = tool_evaluation.get("route_decision")
            route_decision = (
                route_decision
                if isinstance(route_decision, Mapping)
                else {}
            )
            rule_execution.update(
                {
                    "status": str(
                        tool_evaluation.get("status") or "missing"
                    ),
                    "route": (
                        tool_evaluation.get("route")
                        or route_decision.get("resolved_route")
                    ),
                }
            )
    aggregate_result = aggregate_round_results(
        round_plan,
        child_manifest,
        tool_evaluation,
        execution_dir / "aggregate_result.json",
    )
    execution_vqa = run_round_execution_vqa(
        repo_root=repo_root,
        child_manifest=child_manifest,
        child_dir=child_dir,
        tool_evaluation=tool_evaluation,
        execution_dir=execution_dir,
        provider=provider,
        model=vision_model,
        round_plan=round_plan,
        reviewed_vqa_registry=reviewed_vqa_registry,
    )
    if isinstance(semantic_execution, dict):
        vqa_execution = semantic_execution.get("vqa_tool")
        if (
            isinstance(vqa_execution, dict)
            and vqa_execution.get("requested") is True
        ):
            vqa_execution.update(
                {
                    "status": str(
                        execution_vqa.get("status") or "missing"
                    ),
                    "route": "run_local_query_vqa",
                }
            )
    round_summary = summarize_round(
        round_plan,
        child_manifest,
        child_dir,
        tool_evaluation,
        aggregate_result,
        execution_vqa,
        returncode,
    )
    round_summary["round_attempt_index"] = 1
    round_summary["execution_artifact_dir"] = str(
        execution_dir.relative_to(repo_root)
    ).replace("\\", "/")
    if isinstance(
        round_plan.get("proposal")
        or round_plan.get("experiment_candidate"),
        Mapping,
    ):
        method_runtime_path = (
            execution_dir / "method_runtime_projection.json"
        )
        method_runtime_projection = (
            project_executed_round_through_method_runtime(
                task_name=str(round_plan["task_name"]),
                round_plan=round_plan,
                child_manifest=child_manifest,
                round_summary=round_summary,
                artifacts={
                    "child_manifest": str(
                        child_manifest_path.relative_to(repo_root)
                    ).replace("\\", "/"),
                    "taskgen_command": str(
                        (
                            execution_dir / "taskgen_command.json"
                        ).relative_to(repo_root)
                    ).replace("\\", "/"),
                    "aggregate": str(
                        (
                            execution_dir / "aggregate_result.json"
                        ).relative_to(repo_root)
                    ).replace("\\", "/"),
                },
            )
        )
        write_json(method_runtime_path, method_runtime_projection)
        round_summary["observations"]["method_runtime"] = {
            "status": "validated",
            "runtime": method_runtime_projection["runtime"],
            "backend": method_runtime_projection["backend"],
            "execution_reused": method_runtime_projection[
                "execution_reused"
            ],
            "taskgen_reinvoked": method_runtime_projection[
                "taskgen_reinvoked"
            ],
            "policy_rollout_reinvoked": method_runtime_projection[
                "policy_rollout_reinvoked"
            ],
            "candidate_id": method_runtime_projection["candidate"][
                "candidate_id"
            ],
            "outcome": method_runtime_projection["evidence"]["outcome"],
            "artifact": str(
                method_runtime_path.relative_to(repo_root)
            ).replace("\\", "/"),
        }
    write_json(
        execution_dir / "evidence_aggregate.json",
        round_summary["observations"]["evidence_aggregate"],
    )
    write_json(evaluation_dir / "summary" / f"{round_id}.json", round_summary)
    return child_manifest, child_dir, round_summary, tool_evaluation, returncode


def main() -> None:
    args = parse_args()
    if args.benchmark == "libero":
        from mea.libero.chain import run_libero_agent_cli

        run_libero_agent_cli(args)
        return
    requested_open_query_planner = args.open_query_planner
    args.open_query_planner = resolve_default_open_query_planner(args)
    compat_profile_requested = bool(
        requested_open_query_planner == "catalog_step_v1"
        or args.task_profile != "official"
        or args.planning_policy != "dynamic_evidence_v1"
        or args.proposal_mode != "catalog"
        or any(
            value is not None
            for value in (
                args.evidence_manifest,
                args.command_plan,
                args.registered_route,
                args.registered_strategy,
            )
        )
    )
    if compat_profile_requested:
        from experiments.paper.compat_agent_profile import (
            CompatAgentProfileError,
            resolve_compat_agent_profile,
        )

        try:
            compat_profile = resolve_compat_agent_profile(
                args,
                requested_open_query_planner=requested_open_query_planner,
            )
        except CompatAgentProfileError as exc:
            raise SystemExit(str(exc)) from exc
        args.open_query_planner = compat_profile["open_query_planner"]
    claim_first_mode = args.open_query_planner == "plan_agent_v1"
    claim_first_bound_plan_only = bool(
        claim_first_mode
        and args.plan_only
        and args.bound_task_name is not None
        and not args.auto_route
    )
    if claim_first_mode and not claim_first_bound_plan_only:
        # Query-first routing is the production default.  ``--auto-route`` is
        # retained as an explicit spelling for existing commands.
        args.auto_route = True
    if args.num_episodes <= 0:
        raise SystemExit("--num-episodes must be positive")
    if args.auto_route and args.task_module is not None:
        raise SystemExit(
            "--auto-route resolves a trusted task module; do not pass --task-module"
        )
    if (
        args.bound_task_name is not None
        and not args.auto_route
        and not claim_first_bound_plan_only
    ):
        raise SystemExit("--bound-task-name requires --auto-route")
    if args.bound_requested_aspect_ids is not None and args.bound_task_name is None:
        raise SystemExit(
            "--bound-requested-aspect-id requires --bound-task-name"
        )
    if claim_first_mode and not (
        args.auto_route or claim_first_bound_plan_only
    ):
        raise SystemExit(
            "plan_agent_v1 requires --auto-route, or --plan-only with "
            "--bound-task-name"
        )
    if claim_first_bound_plan_only and args.bound_requested_aspect_ids is not None:
        raise SystemExit(
            "providerless Plan Agent plan-only owns the control anchor; "
            "do not predeclare aspect ids"
        )
    if args.query_sufficiency_contract is not None and not claim_first_mode:
        raise SystemExit(
            "--query-sufficiency-contract requires the production Plan Agent"
        )
    repo_root = args.repo_root.expanduser().resolve()
    query_sufficiency_contract: dict[str, Any] | None = None
    if args.query_sufficiency_contract is not None:
        contract_path = args.query_sufficiency_contract.expanduser().resolve()
        try:
            loaded_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"cannot read --query-sufficiency-contract: {exc}"
            ) from exc
        if not isinstance(loaded_contract, dict):
            raise SystemExit("--query-sufficiency-contract must contain a JSON object")
        try:
            query_sufficiency_contract = (
                validate_query_sufficiency_contract(loaded_contract)
            )
        except ValueError as exc:
            raise SystemExit(
                f"invalid --query-sufficiency-contract: {exc}"
            ) from exc
    args.evaluation_id = args.evaluation_id or make_evaluation_id()
    registered_execution: dict[str, Any] | None = None
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
                observed_argv=list(sys.argv),
            )
        except RegisteredExecutionAdapterError as exc:
            raise SystemExit(f"registered execution preflight failed: {exc}") from exc
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
    provider = None
    global_catalog: dict[str, Any] | None = None
    runtime_claim_first_targets: dict[str, dict[str, Any]] = {}
    runtime_binding_excluded: list[dict[str, str]] = []
    # A registered dynamic run already carries a validated route, so it skips
    # --auto-route.  It still needs the trusted catalog to construct the bound
    # PlanSession that owns evidence-conditioned PlanStep proposals.  Keep the
    # registered fixed baseline on its existing planner path.
    if registered_execution is not None:
        global_catalog, provider = initialize_registered_dynamic_runtime(
            repo_root,
            global_catalog,
            provider,
            registered_strategy=args.registered_strategy,
            base_url=args.base_url,
            text_model=models["planner"],
            vision_model=models["vision"],
        )
    global_route_result: dict[str, Any] | None = None
    global_history_retrieval: dict[str, Any] = {
        "schema_version": 1,
        "status": "disabled" if args.no_history else "empty",
        "candidates": [],
    }
    global_router: GlobalQueryRouter | None = None
    free_concern_agent: PlanAgentQueryInterpreter | None = None
    free_concern_bundle: dict[str, Any] | None = None
    concern_candidate_resolution: dict[str, Any] | None = None
    open_task_inventory: list[dict[str, Any]] | None = None
    open_task_resolution: dict[str, Any] | None = None
    validated_proposal: dict[str, Any] | None = (
        registered_execution["validated_proposal"]
        if registered_execution is not None
        else None
    )
    routed_task_profile: str | None = (
        "adaptive_properties" if registered_execution is not None else None
    )

    if claim_first_bound_plan_only:
        global_catalog = build_act_catalog(repo_root)
        open_task_inventory = discover_robotwin_runtime_task_inventory(
            repo_root,
            capability_catalog=global_catalog,
        )
        runtime_discovery = discover_ready_plan_agent_targets(
            repo_root,
            open_task_inventory,
            max_rounds=(
                int(args.max_agent_rounds)
                if args.max_agent_rounds is not None
                else max(2, int(args.generated_rounds))
            ),
        )
        runtime_claim_first_targets = runtime_discovery["targets"]
        runtime_binding_excluded = runtime_discovery["excluded"]
        ready_tasks = sorted(runtime_claim_first_targets)
        assert args.bound_task_name is not None
        if args.bound_task_name not in ready_tasks:
            raise SystemExit(
                "bound task has no source/schema/checkpoint runtime binding: "
                f"{args.bound_task_name!r}"
            )
        args.task_name = args.bound_task_name
        args.task_profile = "official"
        routed_task_profile = args.task_profile

    if args.auto_route:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=models["planner"],
            vision_model=models["vision"],
            timeout=180.0,
        )
        global_catalog = build_act_catalog(repo_root)
        if claim_first_mode:
            open_task_inventory = discover_robotwin_runtime_task_inventory(
                repo_root,
                capability_catalog=global_catalog,
            )
            runtime_discovery = discover_ready_plan_agent_targets(
                repo_root,
                open_task_inventory,
                max_rounds=(
                    int(args.max_agent_rounds)
                    if args.max_agent_rounds is not None
                    else max(2, int(args.generated_rounds))
                ),
            )
            runtime_claim_first_targets = runtime_discovery["targets"]
            runtime_binding_excluded = runtime_discovery["excluded"]
            ready_tasks = sorted(runtime_claim_first_targets)
        else:
            ready_tasks = [
                task["task_name"]
                for task in global_catalog.get("tasks", [])
            ]
        if not ready_tasks:
            raise SystemExit(
                "no source/schema/checkpoint-ready ACT task is available"
            )
        if args.bound_task_name is not None and args.bound_task_name not in ready_tasks:
            raise SystemExit(
                f"bound task is not ACT-ready: {args.bound_task_name!r}"
            )
        global_planning_contexts = (
            {
                task_name: PlanAgentExecutionSession.from_target(
                    runtime_claim_first_targets[task_name]
                ).planning_context(repo_root)
                for task_name in ready_tasks
            }
            if claim_first_mode
            else {
                task_name: BoundTaskPlanSession.from_catalog(
                    global_catalog, task_name
                ).planning_context(repo_root)
                for task_name in ready_tasks
            }
        )
        if claim_first_mode:
            # The query-first acceptance path is intentionally fail-fast:
            # one Query-interpretation call before any task/aspect inventory reaches the
            # model, followed by deterministic semantic task retrieval.
            provider.max_retries = 0
            initially_bound_task = args.bound_task_name
            concern_policy_card = (
                global_planning_contexts[initially_bound_task]["policy_card"]
                if initially_bound_task is not None
                else build_pending_task_binding_policy_card()
            )
            free_concern_agent = PlanAgentQueryInterpreter(
                provider,
                model=models["planner"],
                # One strict response plus one schema-guided repair.  This is
                # bounded provider regeneration, never cached replay or an
                # alternate prewritten concern.
                max_attempts=2,
            )
            free_concern_bundle = free_concern_agent.propose(
                args.request,
                policy_card=concern_policy_card,
            )
            assert open_task_inventory is not None
            checkpoint_binding: dict[str, Any] | None = None
            if initially_bound_task is None:
                checkpoint_binding = bind_ready_task_after_query_interpretation(
                    free_concern_bundle["concern"],
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
                    unsupported = finish_unsupported_open_task_resolution(
                        repo_root,
                        evaluation_id=args.evaluation_id,
                        user_request=args.request,
                        catalog=global_catalog,
                        concern_bundle=free_concern_bundle,
                        task_inventory=open_task_inventory,
                        task_resolution=unresolved_task,
                        concern_agent=free_concern_agent,
                    )
                    print(json.dumps(unsupported, ensure_ascii=False, indent=2))
                    return
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
                free_concern_bundle["concern"],
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
                unsupported = finish_unsupported_open_task_resolution(
                    repo_root,
                    evaluation_id=args.evaluation_id,
                    user_request=args.request,
                    catalog=global_catalog,
                    concern_bundle=free_concern_bundle,
                    task_inventory=open_task_inventory,
                    task_resolution=open_task_resolution,
                    concern_agent=free_concern_agent,
                )
                print(json.dumps(unsupported, ensure_ascii=False, indent=2))
                return
            concern_candidate_resolution = resolve_concern_candidate_domain(
                free_concern_bundle["concern"],
                target=runtime_claim_first_targets[
                    args.bound_task_name
                ],
            )
            semantic_context_for_budget = (
                free_concern_bundle.get("concern")
                if isinstance(free_concern_bundle, Mapping)
                and isinstance(free_concern_bundle.get("concern"), Mapping)
                else None
            )
            candidate_budget = resolve_plan_agent_candidate_budget(
                args.max_agent_rounds,
                user_request=args.request,
                query_contract=query_sufficiency_contract,
                semantic_context=semantic_context_for_budget,
            )
            candidate_domain_supported = (
                concern_candidate_domain_is_executable(
                    concern_candidate_resolution,
                    candidate_budget=candidate_budget,
                )
            )
            if not candidate_domain_supported:
                unsupported = finish_unsupported_open_task_resolution(
                    repo_root,
                    evaluation_id=args.evaluation_id,
                    user_request=args.request,
                    catalog=global_catalog,
                    concern_bundle=free_concern_bundle,
                    task_inventory=open_task_inventory,
                    task_resolution=open_task_resolution,
                    concern_agent=free_concern_agent,
                    candidate_resolution=concern_candidate_resolution,
                )
                print(json.dumps(unsupported, ensure_ascii=False, indent=2))
                return
        global_history_context: list[dict[str, Any]] = []
        if not args.no_history:
            try:
                global_history_db = EvaluationHistoryDB(
                    history_path,
                    repo_root=repo_root,
                )
                global_history_retrieval = global_history_db.retrieve_similar_global(
                    args.request,
                    allowed_task_names=ready_tasks,
                    policy_name="ACT",
                    checkpoint_setting="demo_clean",
                    limit=args.history_limit,
                    exclude_evaluation_id=args.evaluation_id,
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
                runtime_target=runtime_claim_first_targets[
                    args.bound_task_name
                ],
            )
            global_route_result["task_resolution_scope"] = {
                "mode": (
                    "query_first_bound_policy_task"
                    if open_task_resolution["checkpoint_binding"][
                        "reason_code"
                    ]
                    == "explicit_bound_task"
                    else "query_first_then_checkpoint_binding"
                ),
                "artifact": "plan/open_task_resolution.json",
            }
            global_route_result["runtime_binding_scope"] = {
                "authority": "official_source_task_schema_act_checkpoint",
                "catalog_membership_required": False,
                "ready_task_names": sorted(runtime_claim_first_targets),
                "excluded_task_names": sorted(
                    item["task_name"] for item in runtime_binding_excluded
                ),
            }
        else:
            global_router = GlobalQueryRouter(
                provider,
                model=models["planner"],
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
                unsupported = finish_unsupported_global_route(
                    repo_root,
                    evaluation_id=args.evaluation_id,
                    user_request=args.request,
                    catalog=global_catalog,
                    route_result=global_route_result,
                    router=global_router,
                    history_retrieval=global_history_retrieval,
                )
                print(json.dumps(unsupported, ensure_ascii=False, indent=2))
                return
            routed = route_to_planner_proposal(selection, global_catalog)
        args.task_name = routed["task_name"]
        routed_task_profile = routed["task_profile"]
        args.task_profile = (
            "official"
            if claim_first_mode
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
            claim_first_mode
            and args.task_name not in runtime_claim_first_targets
        ):
            raise SystemExit(
                "the Plan Agent requires source/schema/checkpoint runtime "
                f"authority; {args.task_name!r} is unavailable"
            )

    if (
        not claim_first_mode
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
                claim_first_mode=claim_first_mode,
            )
        except CompatAgentProfileError as exc:
            raise SystemExit(str(exc)) from exc
        legacy_click_bell = task_runtime_profile["legacy_click_bell"]
        adaptive_click_bell = task_runtime_profile["adaptive_click_bell"]
        fixed_click_bell = task_runtime_profile["fixed_click_bell"]
        bounded_click_bell = task_runtime_profile["bounded_click_bell"]
        execution_backend = task_runtime_profile["execution_backend"]
    else:
        legacy_click_bell = False
        adaptive_click_bell = False
        fixed_click_bell = False
        bounded_click_bell = False
        execution_backend = (
            "act" if claim_first_mode else (args.execution_backend or "expert")
        )
    # The deterministic official planner can materialize --plan-only without
    # any provider credential. Full execution still creates the provider for
    # final Feedback (and for VQA when an ACT video exists).
    if provider is None and (
        args.task_name == "beat_block_hammer"
        or adaptive_click_bell
        or fixed_click_bell
        or not args.plan_only
    ) and not claim_first_bound_plan_only:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=models["planner"],
            vision_model=models["vision"],
            timeout=180.0,
        )
    planner = None
    if claim_first_mode:
        # Plan Agent owns initial planning directly below. Legacy planners
        # remain available only for explicit compatibility/experiment modes.
        pass
    else:
        # Compatibility and paper-ablation planners live outside the
        # production method path.  Import them lazily only when the caller
        # explicitly selects a historical protocol.
        from experiments.paper.legacy_planner_factory import (
            build_legacy_planner,
        )

        planner = build_legacy_planner(
            repo_root,
            task_name=args.task_name,
            task_profile=args.task_profile,
            provider=provider,
            model=models["planner"],
            task_module=args.task_module,
            start_seed=args.start_seed,
            num_episodes=args.num_episodes,
            telemetry_profile=args.telemetry_profile,
            max_rounds=args.generated_rounds,
            execution_backend=execution_backend,
        )
    history_database = None
    history_context: list[dict[str, Any]] = []
    history_retrieval: dict[str, Any] = {
        "schema_version": 1,
        "status": "disabled" if args.no_history else "empty",
        "candidates": [],
    }
    history_path = (
        args.history_database.expanduser().resolve()
        if args.history_database
        else repo_root / "mea/evaluation_runs/history.sqlite3"
    )
    if not args.no_history:
        try:
            history_database = EvaluationHistoryDB(
                history_path,
                repo_root=repo_root,
            )
            history_retrieval = history_database.retrieve_similar(
                args.request,
                task_name=args.task_name,
                policy_name=(
                    "ACT" if execution_backend in {"act", "both"} else "expert"
                ),
                checkpoint_setting="demo_clean",
                requested_aspect_ids=(
                    validated_proposal.get("requested_aspect_ids")
                    if validated_proposal is not None
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
    if validated_proposal is not None:
        planner_kwargs["validated_proposal"] = validated_proposal
    claim_first_initial_target: dict[str, Any] | None = None
    claim_first_round_budget: int | None = None
    claim_first_control_required = True
    claim_first_evaluation_intent: dict[str, Any] | None = None
    initial_free_concern_semantic_bundle: dict[str, Any] | None = None
    frozen_first_open_candidate: dict[str, Any] | None = None
    initial_open_candidate: dict[str, Any] | None = None
    if claim_first_mode:
        if global_catalog is None:
            raise RuntimeError(
                "Plan Agent runtime requires a trusted policy-task binding index"
            )
        semantic_context = (
            free_concern_bundle.get("concern")
            if isinstance(free_concern_bundle, Mapping)
            and isinstance(free_concern_bundle.get("concern"), Mapping)
            else None
        )
        if semantic_context is not None:
            claim_first_evaluation_intent = (
                evaluation_intent_from_query_interpretation(semantic_context)
            )
            raw_experiment_needs = (
                free_concern_bundle.get("experiment_needs")
                if isinstance(free_concern_bundle, Mapping)
                else None
            )
            if isinstance(raw_experiment_needs, Mapping):
                initial_free_concern_semantic_bundle = (
                    build_initial_semantic_proposal_bundle(
                        user_query=args.request,
                        concern=semantic_context,
                        experiment_needs=raw_experiment_needs,
                        evaluation_intent=claim_first_evaluation_intent,
                        provider_record=free_concern_bundle.get("provider"),
                    )
                )
        claim_first_control_required = resolve_plan_agent_control_required(
            args.request,
            query_contract=query_sufficiency_contract,
            semantic_context=semantic_context,
        )
        if (
            not claim_first_control_required
            and initial_free_concern_semantic_bundle is not None
        ):
            # A control-free Query has no earlier rollout evidence, so its first
            # candidate is legitimately Query-conditioned.  Control-required
            # runs intentionally do not freeze this candidate: the Planner
            # chooses the first tested sub-aspect only after observing control.
            frozen_first_open_candidate = (
                build_dynamic_experiment_candidate(
                    user_query=args.request,
                    task_name=args.task_name,
                    proposal=initial_free_concern_semantic_bundle["proposal"],
                    evaluation_intent=claim_first_evaluation_intent,
                )
            )
        claim_first_round_budget = (
            int(args.max_agent_rounds)
            if args.max_agent_rounds is not None
            else max(
                1 + int(claim_first_control_required),
                int(args.generated_rounds),
            )
        )
        minimum_rounds = 1 + int(claim_first_control_required)
        if claim_first_round_budget < minimum_rounds:
            raise SystemExit(
                "Plan Agent round budget is smaller than the QueryContract "
                "control plus candidate requirement"
            )
        if query_sufficiency_contract is None:
            inferred_claim_type = infer_claim_type(args.request)
            if inferred_claim_type == "comparative":
                raise SystemExit(
                    "comparative Query requires an explicit preregistered "
                    "--query-sufficiency-contract"
                )
            initial_candidate_ids = (
                [frozen_first_open_candidate["candidate_id"]]
                if frozen_first_open_candidate is not None
                else []
            )
            query_sufficiency_contract = (
                build_query_sufficiency_contract(
                    args.request,
                    candidate_universe=initial_candidate_ids,
                    round_budget=(
                        claim_first_round_budget
                        - int(claim_first_control_required)
                    ),
                    claim_type=inferred_claim_type,
                    candidate_universe_closed=False,
                    control_requirement=(
                        "required"
                        if claim_first_control_required
                        else "not_required"
                    ),
                )
            )
        if not claim_first_control_required:
            if semantic_context is None:
                raise SystemExit(
                    "a no-control Plan Agent run requires online Query "
                    "interpretation"
                )
            if initial_free_concern_semantic_bundle is not None:
                assert frozen_first_open_candidate is not None
                initial_open_candidate = frozen_first_open_candidate
            else:
                # Backward compatibility for cached Query-interpretation
                # artifacts that predate independent typed needs.
                initial_open_candidate = build_experiment_candidate(
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
                    evaluation_intent=claim_first_evaluation_intent,
                )
            if query_sufficiency_contract is None:
                query_sufficiency_contract = (
                    build_query_sufficiency_contract(
                        args.request,
                        candidate_universe=[
                            initial_open_candidate["candidate_id"]
                        ],
                        round_budget=claim_first_round_budget,
                        claim_type=infer_claim_type(args.request),
                        candidate_universe_closed=False,
                        control_requirement="not_required",
                    )
                )
            elif (
                initial_open_candidate["candidate_id"]
                not in query_sufficiency_contract["candidate_universe"]
            ):
                if query_sufficiency_contract["candidate_universe_closed"]:
                    raise SystemExit(
                        "closed no-control QueryContract does not contain the "
                        "runtime Query-derived Proposal"
                    )
                query_sufficiency_contract = extend_query_candidate_universe(
                    query_sufficiency_contract,
                    [initial_open_candidate["candidate_id"]],
                    candidate_universe_closed=False,
                )
        if args.task_module is not None:
            # Retained only for the explicit providerless compatibility
            # spelling. Production auto-route always binds the official
            # ``envs.<task>`` source through runtime authority below.
            claim_first_initial_target = build_open_world_evaluation_target(
                global_catalog,
                args.task_name,
                max_rounds=claim_first_round_budget,
                task_module=args.task_module,
            )
        else:
            claim_first_initial_target = (
                build_runtime_open_world_evaluation_target(
                    repo_root,
                    args.task_name,
                    max_rounds=claim_first_round_budget,
                )
            )
    if claim_first_mode:
        if claim_first_initial_target is None:
            raise RuntimeError("Plan Agent runtime target was not initialized")
        initial_builder = PlanAgentInitialPlanBuilder(
            repo_root,
            target=claim_first_initial_target,
            max_rounds=int(claim_first_round_budget),
            start_seed=(
                args.start_seed if args.start_seed is not None else 100000
            ),
            num_episodes=args.num_episodes,
            execution_backend=execution_backend,
            task_module=args.task_module,
            telemetry_profile=args.telemetry_profile,
        )
        manifest = initial_builder.plan(
            args.request,
            evaluation_id=str(args.evaluation_id),
            control_required=claim_first_control_required,
            query_contract=query_sufficiency_contract,
            history_context=history_context,
            history_metadata=planner_kwargs["history_metadata"],
        )
    else:
        if planner is None:
            raise RuntimeError("legacy planner was not initialized")
        manifest = planner.plan(args.request, **planner_kwargs)
    evaluation_id = manifest["evaluation_id"]
    evaluation_dir = repo_root / "mea/evaluation_runs" / evaluation_id
    if (
        global_catalog is not None
        and global_route_result is not None
    ):
        write_global_route_trace(
            evaluation_dir,
            catalog=global_catalog,
            route_result=global_route_result,
            router=global_router,
            history_retrieval=global_history_retrieval,
        )
        update_manifest(
            evaluation_dir,
            global_query_route_path="plan/global_query_route.json",
            global_act_catalog_path="plan/global_act_catalog.json",
            global_route_selection=global_route_result["selection"],
            task_resolution_scope=global_route_result["task_resolution_scope"],
        )
        if (
            free_concern_agent is not None
            and free_concern_bundle is not None
            and open_task_inventory is not None
            and open_task_resolution is not None
        ):
            write_open_task_resolution_trace(
                evaluation_dir,
                concern_bundle=free_concern_bundle,
                task_inventory=open_task_inventory,
                task_resolution=open_task_resolution,
                concern_agent=free_concern_agent,
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
    if initial_open_candidate is not None:
        if (
            not isinstance(plan.get("rounds"), list)
            or not isinstance(
                manifest.get("initial_execution_binding"), Mapping
            )
        ):
            raise RuntimeError(
                "Plan Agent materializer did not provide an execution binding"
            )
        initial_round, initial_tool_bundle = materialize_open_world_round(
            repo_root,
            evaluation_dir=evaluation_dir,
            round_number=1,
            candidate=initial_open_candidate,
            control_execution=manifest["initial_execution_binding"],
        )
        plan["rounds"] = [initial_round]
        plan["query_contract"] = deepcopy(query_sufficiency_contract)
        plan["planning_state"] = "initial_query_derived_candidate_materialized"
        write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
        update_manifest(
            evaluation_dir,
            status=plan["planning_state"],
            plan=plan,
            initial_candidate_source="online_query_interpretation_no_control",
            initial_toolgen_route=initial_tool_bundle["source"],
        )
    bound_plan_session: BoundTaskPlanSession | PlanAgentExecutionSession | None = None
    bound_plan_session_path: str | None = None
    evaluation_target: dict[str, Any] | None = None
    planning_context: dict[str, Any] | None = None
    proposal_agent: BoundedProposalAgent | None = None
    adaptive_step_agent: AdaptivePlanStepAgent | None = None
    claim_first_controller: PlanAgentSession | None = None
    claim_first_agent: PlanAgent | None = None
    claim_first_capabilities: dict[str, Any] | None = None
    if global_catalog is not None:
        initial_failure_stage = "initial_plan_session_validation"
        try:
            raw_round_budget = plan.get("max_rounds")
            if (
                isinstance(raw_round_budget, bool)
                or not isinstance(raw_round_budget, int)
                or raw_round_budget < 1
            ):
                raise ValueError("planner max_rounds must be a positive integer")
            if claim_first_mode:
                if claim_first_round_budget is None:
                    raise RuntimeError(
                        "Plan Agent open-world budget was not initialized"
                    )
                effective_round_budget = claim_first_round_budget
                plan["max_rounds"] = effective_round_budget
                if claim_first_initial_target is None:
                    raise RuntimeError(
                        "Plan Agent runtime target was not initialized"
                    )
                bound_plan_session = PlanAgentExecutionSession.from_target(
                    claim_first_initial_target,
                    control_round=(
                        plan["rounds"][0]
                        if claim_first_control_required
                        else None
                    ),
                    query_contract=(
                        query_sufficiency_contract
                        if isinstance(query_sufficiency_contract, Mapping)
                        else None
                    ),
                )
            else:
                effective_round_budget = raw_round_budget
                if args.max_agent_rounds is not None:
                    effective_round_budget = min(
                        effective_round_budget, int(args.max_agent_rounds)
                    )
                    plan["max_rounds"] = effective_round_budget
                bound_plan_session = BoundTaskPlanSession.from_catalog(
                    global_catalog,
                    args.task_name,
                    max_rounds=effective_round_budget,
                )
            plan = bound_plan_session.normalize_plan(plan)
            planning_context = bound_plan_session.planning_context(repo_root)
            write_json(evaluation_dir / "plan/planning_context.json", planning_context)
            if claim_first_mode:
                explicit_candidate_aspect_ids = (
                    resolve_plan_agent_allowed_aspects(
                        args.bound_requested_aspect_ids
                    )
                )
                claim_first_capabilities = project_open_query_capabilities(
                    planning_context,
                    allowed_aspect_ids=explicit_candidate_aspect_ids,
                )
                # Global routing selects only the executable task/checkpoint.
                # Query interpretation is authored before inventory lookup and remains
                # useful routing evidence, but it must not freeze the semantic
                # domain later seen by the evidence-conditioned Planner. Only
                # an explicit caller binding narrows reusable capabilities;
                # generation outside that inventory remains available.
                resolved_candidate_aspect_ids = (
                    explicit_candidate_aspect_ids
                )
                if (
                    isinstance(free_concern_bundle, Mapping)
                    and isinstance(
                        free_concern_bundle.get("concern"), Mapping
                    )
                ):
                    if not isinstance(
                        concern_candidate_resolution, Mapping
                    ):
                        raise RuntimeError(
                            "online Query-interpretation candidate domain was not "
                            "resolved before planning"
                        )
                    write_json(
                        evaluation_dir
                        / "plan/concern_candidate_resolution.json",
                        {
                            **concern_candidate_resolution,
                            "planner_domain_role": (
                                "routing_and_retrieval_hint_only"
                            ),
                            "planner_domain_restricted": False,
                        },
                    )
                claim_first_controller = PlanAgentSession(
                    args.request,
                    bound_plan_session.target,
                    query_contract=query_sufficiency_contract,
                    candidate_aspect_ids=resolved_candidate_aspect_ids,
                    require_control_anchor=claim_first_control_required,
                    retrieval_aspects=bound_plan_session.retrieval_aspects,
                )
                if frozen_first_open_candidate is not None:
                    frozen_first_open_candidate = (
                        claim_first_controller.register_frozen_candidate(
                            frozen_first_open_candidate
                        )
                    )
                if not args.plan_only:
                    assert provider is not None
                    claim_first_agent = PlanAgent(
                    provider,
                    model=models["planner"],
                )
                write_json(
                    evaluation_dir / PLAN_AGENT_CAPABILITIES,
                    claim_first_capabilities,
                )
                claim_first_controller.query_contract = persist_query_contract(
                    evaluation_dir,
                    plan,
                    claim_first_controller.query_contract,
                )
                manifest.setdefault("planner", {}).update(
                    {
                        "public_planner": "PlanAgent",
                        "control_anchor_owned_by_runtime": (
                            claim_first_controller.require_control_anchor
                        ),
                        "control_template_id": (
                            claim_first_controller.control_template
                        ),
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
                            if concern_candidate_resolution is not None
                            else None
                        ),
                    }
                )
            elif should_enable_adaptive_plan_step(
                fixed_click_bell=fixed_click_bell,
                legacy_click_bell=legacy_click_bell,
                registered_strategy=args.registered_strategy,
            ):
                assert provider is not None
                adaptive_step_agent = AdaptivePlanStepAgent(
                    provider, model=models["planner"]
                )
            if args.proposal_mode != "catalog":
                assert provider is not None
                proposal_agent = BoundedProposalAgent(
                    provider, model=models["taskgen"]
                )
            if args.proposal_mode != "catalog":
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
                plan["rounds"][0], proposal_bundle = apply_bounded_round_proposal(
                    proposal_agent=proposal_agent,
                    user_query=args.request,
                    target=bound_plan_session.target,
                    planning_context=planning_context,
                    round_plan=first_round,
                    evaluation_dir=evaluation_dir,
                    round_number=1,
                )
                plan = bound_plan_session.normalize_plan(plan)
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
            session_snapshot = bound_plan_session.snapshot(args.request, plan)
        except (ValueError, ProposalError, ProposalAgentError) as exc:
            manifest_path = evaluation_dir / "manifest.json"
            if manifest_path.is_file():
                update_manifest(
                    evaluation_dir,
                    status="failed",
                    lifecycle_status="failed",
                    failure_stage=initial_failure_stage,
                    completed_rounds=0,
                    active_child_run_id=None,
                    execution_finished_at=datetime.now().astimezone().isoformat(),
                    failure={"type": type(exc).__name__, "message": str(exc)},
                )
            raise RuntimeError(f"bound PlanSession validation failed: {exc}") from exc
        bound_plan_session_path = "plan/bound_task_session.json"
        evaluation_target = session_snapshot["target"]
        write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
        write_json(evaluation_dir / bound_plan_session_path, session_snapshot)
        update_manifest(
            evaluation_dir,
            plan=plan,
            planner=manifest.get("planner"),
            proposal_mode=args.proposal_mode,
            planning_context_path="plan/planning_context.json",
        )
    candidate_suite = list(plan.get("requested_template_ids") or [])
    planning_policy = (
        "fixed_predeclared_v1"
        if fixed_click_bell
        else "dynamic_evidence_v1"
        if (
            claim_first_mode
            or adaptive_click_bell
            or args.task_name == "beat_block_hammer"
        )
        else None
    )
    registration_identity: dict[str, Any] | None = None
    if registered_execution is not None:
        registration_identity = dict(
            registered_execution["registration_identity"]
        )
        if planning_policy != args.registered_strategy:
            raise RuntimeError(
                "registered strategy does not match resolved planner policy"
            )
        if candidate_suite != registered_execution["expected_candidate_suite"]:
            update_manifest(
                evaluation_dir,
                status="registration_failed",
                registration_identity=registration_identity,
                registration_failure="planner candidate suite differs from preregistration",
            )
            raise RuntimeError(
                "planner candidate suite differs from preregistered route"
            )
        write_json(
            evaluation_dir / "plan/registered_route.json",
            registered_execution["route"],
        )
    if (
        global_catalog is not None
        and global_route_result is not None
        and global_router is not None
    ):
        write_global_route_trace(
            evaluation_dir,
            catalog=global_catalog,
            route_result=global_route_result,
            router=global_router,
            history_retrieval=global_history_retrieval,
        )
    update_manifest(
        evaluation_dir,
        auto_route=args.auto_route,
        global_query_route_path=(
            "plan/global_query_route.json" if args.auto_route else None
        ),
        global_act_catalog_path=(
            "plan/global_act_catalog.json" if args.auto_route else None
        ),
        global_route_selection=(
            global_route_result["selection"]
            if global_route_result is not None
            else None
        ),
        model_profile=args.model_profile,
        resolved_models={
            "planner": models["planner"],
            "taskgen": models["taskgen"],
            "toolgen": models["toolgen"],
            "vision": models["vision"],
            "answer": models["feedback"],
        },
        history_database=(
            str(history_path.relative_to(repo_root))
            if history_path.is_relative_to(repo_root)
            else str(history_path)
        ),
        history_retrieval_status=history_retrieval.get("status"),
        task_name=args.task_name,
        task_module=(
            policy_task_binding_from_target(claim_first_initial_target)[
                "task_module"
            ]
            if claim_first_initial_target is not None
            else args.task_module
        ),
        task_profile=routed_task_profile or args.task_profile,
        generated_rounds=(args.generated_rounds if bounded_click_bell else None),
        telemetry_profile=args.telemetry_profile,
        execution_backend=execution_backend,
        planning_policy=planning_policy,
        open_query_planner=args.open_query_planner,
        query_sufficiency_contract_path=(
            "plan/query_sufficiency_contract.json" if claim_first_mode else None
        ),
        plan_agent_capabilities_path=(
            PLAN_AGENT_CAPABILITIES.as_posix() if claim_first_mode else None
        ),
        candidate_suite_sha256=(
            canonical_sha256(candidate_suite) if candidate_suite else None
        ),
        reviewed_tool_registry=(
            str(reviewed_tool_registry.relative_to(repo_root))
            if reviewed_tool_registry is not None
            and reviewed_tool_registry.is_relative_to(repo_root)
            else str(reviewed_tool_registry)
            if reviewed_tool_registry is not None
            else None
        ),
        reviewed_task_registry=(
            str(reviewed_task_registry.relative_to(repo_root))
            if reviewed_task_registry is not None
            and reviewed_task_registry.is_relative_to(repo_root)
            else str(reviewed_task_registry)
            if reviewed_task_registry is not None
            else None
        ),
        reviewed_vqa_registry=(
            str(reviewed_vqa_registry.relative_to(repo_root))
            if reviewed_vqa_registry is not None
            and reviewed_vqa_registry.is_relative_to(repo_root)
            else str(reviewed_vqa_registry)
            if reviewed_vqa_registry is not None
            else None
        ),
        registration_identity=registration_identity,
        evidence_manifest=(
            str(args.evidence_manifest) if registration_identity is not None else None
        ),
        command_plan=(
            str(args.command_plan) if registration_identity is not None else None
        ),
        registered_route=(
            str(args.registered_route) if registration_identity is not None else None
        ),
        max_agent_rounds=args.max_agent_rounds,
        bound_task_name=(
            bound_target_task_name(evaluation_target)
            if evaluation_target is not None
            else None
        ),
        bound_requested_aspect_ids=(
            list(args.bound_requested_aspect_ids)
            if args.bound_requested_aspect_ids is not None
            else None
        ),
        evaluation_target=evaluation_target,
        plan_session_path=bound_plan_session_path,
    )

    frozen_first_candidate_path: str | None = None
    if (
        initial_free_concern_semantic_bundle is not None
        and frozen_first_open_candidate is not None
    ):
        frozen_dir = evaluation_dir / INITIAL_SUB_ASPECT_PROPOSAL
        write_json(
            frozen_dir / "semantic_proposal_bundle.json",
            initial_free_concern_semantic_bundle,
        )
        write_json(
            frozen_dir / PROPOSAL_FILENAME,
            frozen_first_open_candidate,
        )
        frozen_first_candidate_path = (
            (INITIAL_SUB_ASPECT_PROPOSAL / PROPOSAL_FILENAME).as_posix()
        )
        update_manifest(
            evaluation_dir,
            initial_candidate_source=(
                "provider_plan_agent_direct_materialization"
            ),
            frozen_first_candidate_path=frozen_first_candidate_path,
        )

    if args.plan_only:
        update_manifest(evaluation_dir, status="planned_only")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    assert provider is not None

    round_runs: list[dict[str, Any]] = []
    claim_first_runtime_state: dict[str, Any] | None = None
    claim_first_query_answer: dict[str, Any] | None = None
    active_failure_stage = "round_execution"
    try:
        executed_rounds = 0
        while executed_rounds < len(plan["rounds"]):
            active_failure_stage = f"round_{executed_rounds + 1}_execution"
            round_plan = plan["rounds"][executed_rounds]
            (
                child_manifest,
                child_dir,
                round_summary,
                tool_evaluation,
                returncode,
            ) = execute_round(
                repo_root,
                evaluation_dir,
                evaluation_id,
                round_plan,
                text_model=models["taskgen"],
                vision_model=models["vision"],
                base_url=args.base_url,
                gpu=args.gpu,
                max_reflections=args.max_reflections,
                provider=provider,
                toolgen_model=models["toolgen"],
                telemetry_profile=args.telemetry_profile,
                reviewed_task_registry=reviewed_task_registry,
                reviewed_tool_registry=reviewed_tool_registry,
                reviewed_vqa_registry=reviewed_vqa_registry,
                registration_identity=registration_identity,
            )
            round_runs.append(
                {
                    "round_plan": round_plan,
                    "child_manifest": child_manifest,
                    "child_dir": child_dir,
                    "round_summary": round_summary,
                    "tool_evaluation": tool_evaluation,
                    "returncode": returncode,
                }
            )
            executed_rounds += 1

            if claim_first_controller is not None:
                active_failure_stage = (
                    f"plan_agent_evidence_after_round_{executed_rounds}"
                )
                claim_first_runtime_state = claim_first_controller.observe(
                    [item["round_plan"] for item in round_runs],
                    [item["round_summary"] for item in round_runs],
                )
                # PlanAgentExecutionSession is only the execution transport
                # for the Plan Agent. Give it the same normalized candidate
                # evidence that the authoritative Plan Agent controller just
                # derived, while leaving the control round outside the Query
                # candidate domain.
                contract_candidate_ids = {
                    str(item)
                    for item in claim_first_runtime_state["query_contract"].get(
                        "candidate_universe", []
                    )
                }
                records_by_round = {
                    str(record["round_id"]): record
                    for record in claim_first_runtime_state["records"]
                }
                if len(records_by_round) != len(round_runs):
                    raise RuntimeError(
                        "Plan Agent records are not one-to-one with completed "
                        "runtime rounds"
                    )
                for completed_run in round_runs:
                    round_id = str(completed_run["round_plan"]["round_id"])
                    record = records_by_round.get(round_id)
                    if record is None:
                        raise RuntimeError(
                            "Plan Agent record is missing for completed round "
                            f"{round_id!r}"
                        )
                    if record["candidate_id"] in contract_candidate_ids:
                        completed_run["round_summary"][
                            "candidate_evidence"
                        ] = deepcopy(record["candidate_evidence"])
                claim_first_dir = evaluation_dir / PLAN_AGENT_SESSION
                write_json(
                    claim_first_dir
                    / f"evidence_after_round_{executed_rounds:02d}.json",
                    claim_first_runtime_state,
                )
                assessment = claim_first_runtime_state["assessment"]
                if assessment["should_stop"]:
                    claim_first_query_answer = claim_first_runtime_state[
                        "query_answer"
                    ]
                    decision = {
                        "schema_version": 3,
                        "action": "stop",
                        "transition": "stop",
                        "next_aspect_id": None,
                        "next_template_id": None,
                        "observation_summary": assessment["rationale"],
                        "decision_reason": (
                            "plan_agent_evidence_sufficiency"
                        ),
                        "answered_query": bool(
                            claim_first_query_answer["answered"]
                        ),
                        "plan_step_source": (
                            "deterministic_query_sufficiency_contract"
                        ),
                        "round_budget_before_decision": assessment[
                            "budget_remaining"
                        ],
                        "evidence_assessment": assessment,
                        "next_round": None,
                    }
                    plan.setdefault("round_decisions", []).append(decision)
                    plan["planning_state"] = (
                        f"stopped_after_round_{executed_rounds}_"
                        f"{assessment['stop_reason']}"
                    )
                    write_json(
                        evaluation_dir
                        / f"plan/decision_after_{round_plan['round_id']}.json",
                        decision,
                    )
                    write_json(
                        claim_first_dir / "query_answer.json",
                        claim_first_query_answer,
                    )
                    write_json(
                        evaluation_dir / "plan/evaluation_plan.json",
                        plan,
                    )
                    if bound_plan_session is not None:
                        write_json(
                            evaluation_dir / "plan/bound_task_session.json",
                            bound_plan_session.snapshot(
                                args.request,
                                plan,
                                [item["round_summary"] for item in round_runs],
                            ),
                        )
                    update_manifest(
                        evaluation_dir,
                        status=plan["planning_state"],
                        plan=plan,
                        plan_agent_stop={
                            "stop_reason": assessment["stop_reason"],
                            "evidence_sufficient": assessment[
                                "evidence_sufficient"
                            ],
                            "answered_query": claim_first_query_answer["answered"],
                            "answer_path": (
                                (PLAN_AGENT_SESSION / "query_answer.json").as_posix()
                            ),
                        },
                    )
                    break

            if (
                args.max_agent_rounds is not None
                and executed_rounds >= args.max_agent_rounds
            ):
                completed = [
                    item["round_plan"].get("template_id") for item in round_runs
                ]
                remaining = [
                    template
                    for template in plan.get("requested_template_ids", [])
                    if template not in completed
                ]
                assessment = {
                    "schema_version": 1,
                    "state": "external_hard_round_cap_reached",
                    "required_action": "stop",
                    "completed_rounds": executed_rounds,
                    "max_agent_rounds": args.max_agent_rounds,
                    "remaining_template_ids": remaining,
                    "policy_outcome_not_inferred": True,
                }
                decision = {
                    "schema_version": 2,
                    "action": "stop",
                    "observation_summary": (
                        f"Completed {executed_rounds} round(s); the task-agnostic "
                        "hard execution cap is now exhausted."
                    ),
                    "decision_reason": "external_max_agent_rounds_budget",
                    "next_template_id": None,
                    "remaining_template_ids_before_decision": remaining,
                    "round_budget_before_decision": 0,
                    "evidence_assessment": assessment,
                    "next_round": None,
                }
                plan.setdefault("round_decisions", []).append(decision)
                plan["planning_state"] = (
                    f"stopped_after_round_{executed_rounds}_by_hard_cap"
                )
                write_json(
                    evaluation_dir
                    / f"plan/evidence_after_round_{executed_rounds}.json",
                    assessment,
                )
                write_json(
                    evaluation_dir
                    / f"plan/decision_after_round_{executed_rounds}.json",
                    decision,
                )
                write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
                if bound_plan_session is not None:
                    write_json(
                        evaluation_dir / "plan/bound_task_session.json",
                        bound_plan_session.snapshot(
                            args.request,
                            plan,
                            [item["round_summary"] for item in round_runs],
                        ),
                    )
                update_manifest(
                    evaluation_dir,
                    status=plan["planning_state"],
                    plan=plan,
                    hard_round_cap_stop={
                        "max_agent_rounds": args.max_agent_rounds,
                        "executed_rounds": executed_rounds,
                        "decision_path": (
                            f"plan/decision_after_round_{executed_rounds}.json"
                        ),
                    },
                )
                break

            plan_before_decision = plan
            observation_history = [
                item["round_summary"] for item in round_runs
            ]
            dynamic_step_session = (
                bound_plan_session is not None
                and adaptive_step_agent is not None
                and planning_context is not None
            )
            claim_first_step_session = (
                bound_plan_session is not None
                and claim_first_controller is not None
                and claim_first_agent is not None
                and claim_first_capabilities is not None
                and claim_first_runtime_state is not None
            )
            if claim_first_step_session:
                active_failure_stage = (
                    f"plan_agent_decision_after_round_{executed_rounds}"
                )
                executed_candidate_ids = [
                    str(
                        item["round_plan"].get("candidate_id")
                        or item["round_plan"].get("template_id")
                    )
                    for item in round_runs
                ]
                # This is the temporal boundary required by Fig. 5: validate the
                # latest Aggregate/Evidence first, then ask the Plan Agent which
                # semantic sub-aspect should be tested next.  A pre-control
                # Query interpretation is routing context, not a frozen experiment.
                bound_semantic_step = (
                    claim_first_controller.propose_and_bind_semantic_step(
                        claim_first_agent,
                        claim_first_runtime_state,
                        capabilities=claim_first_capabilities,
                        executed_candidate_ids=executed_candidate_ids,
                        evaluation_intent=None,
                    )
                )
                semantic_bundle = bound_semantic_step[
                    "semantic_proposal_bundle"
                ]
                step_prompt = claim_first_agent.last_prompt
                step_responses = list(claim_first_agent.last_responses)
                step_dir = (
                    evaluation_dir
                    / PLAN_AGENT_STEPS
                    / f"after_round_{executed_rounds:02d}"
                )
                step_dir.mkdir(parents=True, exist_ok=True)
                (step_dir / "prompt.md").write_text(
                    step_prompt or "",
                    encoding="utf-8",
                )
                for index, response in enumerate(
                    step_responses,
                    start=1,
                ):
                    (step_dir / f"response_{index}.txt").write_text(
                        response + "\n",
                        encoding="utf-8",
                    )
                write_json(step_dir / "semantic_proposal_bundle.json", semantic_bundle)
                write_json(step_dir / "bound_semantic_step.json", bound_semantic_step)
                plan_step = bound_semantic_step["plan_step"]
                dynamic_candidate = (
                    plan_step.get("proposal")
                    or plan_step.get("experiment_candidate")
                )
                if not isinstance(dynamic_candidate, Mapping):
                    raise RuntimeError(
                        "Plan Agent must bind every continue decision to a "
                        "typed Proposal before execution"
                    )
                next_round_number = len(plan_before_decision["rounds"]) + 1
                (
                    materialized_round,
                    open_tool_bundle,
                ) = materialize_open_world_round(
                    repo_root,
                    evaluation_dir=evaluation_dir,
                    round_number=next_round_number,
                    candidate=dynamic_candidate,
                    control_execution=plan_before_decision["rounds"][0][
                        "execution"
                    ],
                )
                bound_semantic_step["execution_binding"] = {
                    "schema_version": 2,
                    "candidate_id": dynamic_candidate["candidate_id"],
                    "materialization_path": (
                        f"{PROPOSAL_MATERIALIZATION.as_posix()}/"
                        f"round_{next_round_number:02d}"
                    ),
                    "taskgen_route": materialized_round["route"],
                    "toolgen_route": open_tool_bundle["source"],
                    "catalog_template_used": False,
                    "retrieval_template_hint": bound_semantic_step[
                        "resolution"
                    ].get("retrieval_template_id"),
                }
                write_json(
                    step_dir / "bound_semantic_step.json",
                    bound_semantic_step,
                )
                apply_kwargs: dict[str, Any] = {}
                apply_kwargs["query_contract"] = bound_semantic_step.get(
                    "query_contract"
                )
                plan, decision, runtime_directive = (
                    bound_plan_session.apply_plan_step(
                        plan_before_decision,
                        observation_history,
                        plan_step,
                        materialized_round=materialized_round,
                        source=str(
                            semantic_bundle.get("source")
                            or "provider_plan_agent_open_query"
                        ),
                        **apply_kwargs,
                    )
                )
                decision["semantic_proposal"] = deepcopy(
                    semantic_bundle["proposal"]
                )
                decision["semantic_resolution"] = deepcopy(
                    bound_semantic_step["resolution"]
                )
                write_json(
                    evaluation_dir
                    / f"plan/runtime_directive_after_{round_plan['round_id']}.json",
                    {
                        "schema_version": 1,
                        "owner": type(bound_plan_session).__name__,
                        "adapter_role": (
                            "plan_agent_retrieve_or_generate_and_adjudicate"
                        ),
                        **runtime_directive,
                    },
                )
                update_manifest(
                    evaluation_dir,
                    last_plan_agent_step={
                        "status": "transition_applied",
                        "after_round": executed_rounds,
                        "action": plan_step["action"],
                        "semantic_sub_aspect": (
                            semantic_bundle["proposal"]["sub_aspect"]
                        ),
                        "resolved_template_id": plan_step.get("template_id"),
                        "resolved_candidate_id": plan_step.get("candidate_id"),
                        "evidence_conditioned": bool(
                            bound_semantic_step.get(
                                "planning_lineage", {}
                            ).get("evidence_conditioned")
                        ),
                        "planning_lineage": deepcopy(
                            bound_semantic_step.get("planning_lineage")
                        ),
                        "artifact_path": (
                            f"{PLAN_AGENT_STEPS.as_posix()}/"
                            f"after_round_{executed_rounds:02d}/"
                            "bound_semantic_step.json"
                        ),
                    },
                )
            elif dynamic_step_session:
                active_failure_stage = (
                    f"adaptive_decision_after_round_{executed_rounds}"
                )
                navigation_options = bound_plan_session.navigation_options(
                    plan_before_decision,
                    observation_history,
                    allowed_template_ids=(
                        registered_execution["expected_candidate_suite"]
                        if registered_execution is not None
                        else None
                    ),
                )
                step_bundle = adaptive_step_agent.propose(
                    args.request,
                    navigation_options=navigation_options,
                    planning_context=planning_context,
                )
                plan_step = step_bundle["proposal"]
                step_path = persist_adaptive_step_selection(
                    evaluation_dir,
                    after_round=executed_rounds,
                    prompt=adaptive_step_agent.last_prompt,
                    responses=adaptive_step_agent.last_responses,
                    step_bundle=step_bundle,
                    navigation_options=navigation_options,
                )
                update_manifest(
                    evaluation_dir,
                    last_adaptive_step={
                        "status": "selected_pending_materialization",
                        "after_round": executed_rounds,
                        "action": plan_step["action"],
                        "artifact_path": f"{step_path}/plan_step_bundle.json",
                    },
                )
                materialized_round = None
                if plan_step["action"] != "stop":
                    active_failure_stage = (
                        f"template_materialization_after_round_{executed_rounds}"
                    )
                    materialize = getattr(planner, "materialize_plan_step", None)
                    if not callable(materialize):
                        raise RuntimeError(
                            "bound task planner cannot materialize PlanStepProposal"
                        )
                    materialized_round = materialize(
                        plan_step["template_id"],
                        len(plan_before_decision["rounds"]) + 1,
                        args.request,
                    )
                    if args.proposal_mode == "bounded_each_round":
                        active_failure_stage = (
                            f"bounded_proposal_after_round_{executed_rounds}"
                        )
                        if proposal_agent is None:
                            raise RuntimeError(
                                "bounded_each_round proposal state was not initialized"
                            )
                        next_round_number = len(plan_before_decision["rounds"]) + 1
                        materialized_round, _proposal_artifact = (
                            apply_bounded_round_proposal(
                                proposal_agent=proposal_agent,
                                user_query=args.request,
                                target=bound_plan_session.target,
                                planning_context=planning_context,
                                round_plan=materialized_round,
                                evaluation_dir=evaluation_dir,
                                round_number=next_round_number,
                            )
                        )
                active_failure_stage = (
                    f"plan_transition_after_round_{executed_rounds}"
                )
                plan, decision, runtime_directive = bound_plan_session.apply_plan_step(
                    plan_before_decision,
                    observation_history,
                    plan_step,
                    materialized_round=materialized_round,
                    source=step_bundle["source"],
                )
                write_json(
                    evaluation_dir
                    / f"plan/runtime_directive_after_{round_plan['round_id']}.json",
                    {
                        "schema_version": 1,
                        "owner": "BoundTaskPlanSession",
                        "adapter_role": "discover_materialize_and_adjudicate",
                        **runtime_directive,
                    },
                )
            else:
                candidate_plan, candidate_decision = planner.decide_next_round(
                    evaluation_id=evaluation_id,
                    user_request=args.request,
                    current_plan=plan_before_decision,
                    observation_history=observation_history,
                )
                common_adaptive_session = (
                    bound_plan_session is not None
                    and not fixed_click_bell
                    and not legacy_click_bell
                    and candidate_decision.get("action") in {"continue", "stop"}
                )
                if common_adaptive_session:
                    plan, decision, runtime_directive = adjudicate_bounded_transition(
                        plan_session=bound_plan_session,
                        user_query=args.request,
                        observation_history=observation_history,
                        current_plan=plan_before_decision,
                        candidate_plan=candidate_plan,
                        candidate_decision=candidate_decision,
                        proposal_mode=args.proposal_mode,
                        proposal_agent=proposal_agent,
                        planning_context=planning_context,
                        evaluation_dir=evaluation_dir,
                    )
                    write_json(
                        evaluation_dir
                        / f"plan/runtime_directive_after_{round_plan['round_id']}.json",
                        {
                            "schema_version": 1,
                            "owner": "BoundTaskPlanSession",
                            "adapter_role": "materialize_and_explain",
                            **runtime_directive,
                        },
                    )
                else:
                    plan, decision = candidate_plan, candidate_decision
            if bound_plan_session is not None:
                # Persist and execute the exact normalized proposal-bearing plan;
                # snapshot() alone normalizes only a deep copy for reporting.
                plan = bound_plan_session.normalize_plan(plan)
                if decision.get("next_round") is not None:
                    decision["next_round"] = plan["rounds"][-1]
                write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
                write_json(
                    evaluation_dir
                    / f"plan/decision_after_{round_plan['round_id']}.json",
                    decision,
                )
                update_manifest(
                    evaluation_dir,
                    status=plan.get("planning_state"),
                    plan=plan,
                    last_adaptive_step=(
                        {
                            "status": "transition_applied",
                            "after_round": executed_rounds,
                            "action": decision.get("action"),
                            "artifact_path": (
                                f"plan/adaptive_steps/after_round_{executed_rounds:02d}"
                                "/plan_step_bundle.json"
                            ),
                            "decision_path": (
                                f"plan/decision_after_{round_plan['round_id']}.json"
                            ),
                        }
                        if dynamic_step_session
                        else None
                    ),
                )
            if decision["action"] == "stop":
                if bound_plan_session is not None:
                    write_json(
                        evaluation_dir / "plan/bound_task_session.json",
                        bound_plan_session.snapshot(
                            args.request, plan, observation_history
                        ),
                    )
                break
            if bound_plan_session is not None:
                write_json(
                    evaluation_dir / "plan/bound_task_session.json",
                    bound_plan_session.snapshot(
                        args.request, plan, observation_history
                    ),
                )

        active_failure_stage = "evaluation_aggregation"
        evaluation_aggregate = aggregate_evaluation_results(
            round_runs,
            evaluation_dir / "summary/aggregate_result.json",
        )
        summary = {
            "schema_version": 2,
            "evaluation_id": evaluation_id,
            "status": (
                "completed"
                if round_runs
                and all(item["round_summary"]["pipeline_passed"] for item in round_runs)
                else "completed_with_pipeline_failure"
            ),
            "rounds": [item["round_summary"] for item in round_runs],
            "aggregate": compact_aggregate_result(evaluation_aggregate),
        }
        evidence = build_evidence_bundle(
            repo_root,
            evaluation_id,
            args.request,
            plan,
            round_runs,
            evaluation_aggregate,
        )
        if claim_first_runtime_state is not None:
            if claim_first_query_answer is None:
                externally_stopped_assessment = {
                    **claim_first_runtime_state["assessment"],
                    "should_stop": True,
                    "stop_reason": "external_hard_round_cap",
                    "evidence_sufficient": False,
                    "claim_verdict": "inconclusive",
                    "rationale": (
                        "An external execution cap stopped the run before the "
                        "query-sufficiency contract was satisfied."
                    ),
                }
                claim_first_query_answer = render_query_answer(
                    args.request,
                    externally_stopped_assessment,
                    claim_first_runtime_state["records"],
                    baseline_valid=bool(
                        claim_first_runtime_state["control_passed"]
                    ),
                )
                write_json(
                    evaluation_dir
                    / PLAN_AGENT_SESSION
                    / "query_answer.json",
                    claim_first_query_answer,
                )
            evidence["plan_agent_session"] = {
                "schema_version": 1,
                "query_contract": claim_first_runtime_state["query_contract"],
                "assessment": claim_first_runtime_state["assessment"],
                "query_answer": claim_first_query_answer,
                "records": claim_first_runtime_state["records"],
                "artifacts": {
                    "query_answer": (
                        f"mea/evaluation_runs/{evaluation_id}/plan/"
                        "plan_agent_session/query_answer.json"
                    ),
                    "latest_evidence": (
                        f"mea/evaluation_runs/{evaluation_id}/plan/"
                        "plan_agent_session/"
                        f"evidence_after_round_{executed_rounds:02d}.json"
                    ),
                },
            }
        flagship_acceptance = (
            build_compact_flagship_acceptance(
                round_runs,
                global_route_result=global_route_result,
                claim_first_runtime_state=claim_first_runtime_state,
                claim_first_query_answer=claim_first_query_answer,
                free_concern_bundle=free_concern_bundle,
                open_task_resolution=open_task_resolution,
                concern_candidate_resolution=concern_candidate_resolution,
                history_disabled=bool(args.no_history),
                cli_candidate_hint_used=(
                    args.bound_requested_aspect_ids is not None
                ),
            )
            if claim_first_controller is not None
            else None
        )
        if flagship_acceptance is not None:
            summary["flagship_acceptance"] = flagship_acceptance
            evidence["flagship_acceptance"] = flagship_acceptance
        write_json(evaluation_dir / "summary/summary.json", summary)
        write_json(evaluation_dir / "summary/evidence_bundle.json", evidence)
        update_manifest(
            evaluation_dir,
            status="generating_answer",
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            summary=summary,
        )
        active_failure_stage = "final_answer"
        feedback = PlanAgentFinalSummary(
            repo_root,
            provider,
            model=models["feedback"],
        ).generate(
            evidence,
            output_dir=evaluation_dir / "answer",
        )
        report_path = evaluation_dir / "evaluation_report.md"
        report_path.write_text(
            render_evaluation_report(evidence, feedback),
            encoding="utf-8",
        )
        update_manifest(
            evaluation_dir,
            status=summary["status"],
            lifecycle_status="completed",
            execution_finished_at=datetime.now().astimezone().isoformat(),
            summary_path="summary/summary.json",
            aggregate_path="summary/aggregate_result.json",
            evidence_path="summary/evidence_bundle.json",
            answer_path="answer/answer.json",
            report_path="evaluation_report.md",
            child_run_ids=[item["child_manifest"].get("run_id") for item in round_runs],
            summary=summary,
            answer=feedback,
            flagship_acceptance=flagship_acceptance,
        )
        compact_evidence_report = write_evidence_report(
            repo_root,
            evaluation_dir,
            destination=evaluation_dir / "evidence_report.md",
        )
        update_manifest(
            evaluation_dir,
            evidence_report_path="evidence_report.md",
            evidence_report_bundle=compact_evidence_report,
        )
        history_index = {"status": "disabled" if args.no_history else "not_available"}
        if history_database is not None:
            try:
                history_index = {
                    "status": "passed",
                    **history_database.index_evaluation_dir(evaluation_dir),
                }
            except Exception as exc:
                history_index = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        update_manifest(evaluation_dir, history_index=history_index)
        print(
            json.dumps(
                {
                    "evaluation_id": evaluation_id,
                    "child_run_ids": [
                        item["child_manifest"].get("run_id") for item in round_runs
                    ],
                    "summary": summary,
                    "answer": feedback,
                    "history_retrieval": {
                        "status": history_retrieval.get("status"),
                        "selected_count": len(history_context),
                    },
                    "history_index": history_index,
                    "report_path": str(report_path.relative_to(repo_root)),
                    "evidence_report_path": str(
                        (evaluation_dir / "evidence_report.md").relative_to(repo_root)
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        update_manifest(
            evaluation_dir,
            status="failed",
            lifecycle_status="failed",
            failure_stage=active_failure_stage,
            completed_rounds=len(round_runs),
            active_child_run_id=None,
            execution_finished_at=datetime.now().astimezone().isoformat(),
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
