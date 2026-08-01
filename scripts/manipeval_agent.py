"""Plan and execute a bounded, evidence-driven multi-round MEA evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from functools import partial
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
from mea.agent_evidence import (
    _round_evidence,
    build_evidence_bundle,
    compact_aggregate_result,
)
from mea.capability_adapter import taskgen_route
from mea.feedback import (
    PlanAgentFinalSummary,
    render_evaluation_report,
    write_evidence_report,
)
from mea.history import EvaluationHistoryDB
from mea.plan_artifacts import (
    INITIAL_SUB_ASPECT_PROPOSAL,
    PLAN_AGENT_CAPABILITIES,
    PROPOSAL_FILENAME,
    QUERY_INTERPRETATION,
    QUERY_INTERPRETATION_PROMPT,
    QUERY_INTERPRETATION_RESPONSE_PREFIX,
)
from mea.plan_agent_application import (
    PlanAgentApplication,
    apply_external_hard_round_cap,
    update_manifest,
)
from mea.planner import (
    AdaptivePlanStepAgent,
    BoundTaskPlanSession,
    PlanAgent,
    PlanAgentInitialPlanBuilder,
    PlanAgentQueryInterpreter,
    PlanAgentSession,
    GlobalQueryRouter,
    build_planning_context,
    build_dynamic_experiment_candidate,
    build_initial_semantic_proposal_bundle,
    build_act_catalog,
    build_open_world_evaluation_target,
    evaluation_intent_from_query_interpretation,
    make_evaluation_id,
    policy_task_binding_from_target,
    project_open_query_capabilities,
    resolve_concern_candidate_domain,
    resolve_open_task,
    route_to_planner_proposal,
    validate_open_query_plan_proposal,
)
from mea.planner.experiment_candidate import (
    build_experiment_candidate,
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
    RuntimePolicySpec,
    RuntimeTaskBindingError,
    build_hyvla_policy_spec,
    build_runtime_open_world_evaluation_target,
    build_smolvla_policy_spec,
)
from mea.proposals import (
    ProposalError,
    materialize_round_proposals,
    tool_request_from_proposal,
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
from mea.round_executor import (
    RoundExecutionRequest,
    RoundExecutionServices,
    RoundExecutor,
)
from mea.round_evidence import aggregate_evaluation_results
from mea.robotwin.native_agent_round import (
    execute_act_method_round,
    execute_hyvla_method_round,
    execute_smolvla_method_round,
)
from mea.taskgen import round_materialization as taskgen_round_materialization
from mea.taskgen.runtime import create_generic_provider_taskgen_run
from mea.toolgen import route_tool_request


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
    policy_spec: RuntimePolicySpec | None = None,
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


def build_pending_task_binding_policy_card(
    policy_spec: RuntimePolicySpec | None = None,
) -> dict[str, Any]:
    """Describe an unbound checkpoint portfolio without exposing its menu.

    Query interpretation must happen before the official task inventory is
    retrieved.  This neutral card makes that ordering explicit: it describes
    the evaluation surface, but contains no executable task or aspect name.
    """

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

    if (
        resolution.get("resolution")
        == "official_execution_from_typed_needs"
        and resolution.get("execution_authorized") is True
    ):
        # The required official control is itself the requested experiment;
        # it does not need an additional generated-candidate round.
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


def _build_round_executor(*, native_act: bool) -> RoundExecutor:
    native_policy_rounds = {
        "hyvla": partial(
            execute_hyvla_method_round,
            generated_task_materializer=(
                create_generic_provider_taskgen_run
            ),
        ),
        "smolvla": partial(
            execute_smolvla_method_round,
            generated_task_materializer=(
                create_generic_provider_taskgen_run
            ),
        ),
    }
    if native_act:
        native_policy_rounds["act"] = partial(
            execute_act_method_round,
            generated_task_materializer=(
                create_generic_provider_taskgen_run
            ),
        )

    return RoundExecutor(
        RoundExecutionServices(
            update_manifest=update_manifest,
            build_taskgen_command=(
                taskgen_round_materialization.build_taskgen_command
            ),
            run_logged=run_logged,
            native_policy_rounds=native_policy_rounds,
        )
    )


def build_production_round_executor() -> RoundExecutor:
    """Assemble the production lifecycle with native ACT and SmolVLA."""

    return _build_round_executor(native_act=True)


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
    policy_backend: str = "act",
    runtime_target: Mapping[str, Any] | None = None,
    smolvla_port: int = 18771,
    hyvla_port: int = 18781,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any], int]:
    """Compatibility import for callers migrating to :class:`RoundExecutor`."""

    request = RoundExecutionRequest(
        repo_root=repo_root,
        evaluation_dir=evaluation_dir,
        evaluation_id=evaluation_id,
        round_plan=round_plan,
        text_model=text_model,
        vision_model=vision_model,
        base_url=base_url,
        gpu=gpu,
        max_reflections=max_reflections,
        provider=provider,
        toolgen_model=toolgen_model,
        telemetry_profile=telemetry_profile,
        reviewed_task_registry=reviewed_task_registry,
        reviewed_tool_registry=reviewed_tool_registry,
        reviewed_vqa_registry=reviewed_vqa_registry,
        registration_identity=registration_identity,
        policy_backend=policy_backend,
        runtime_target=runtime_target,
        policy_server_port=(
            hyvla_port if policy_backend == "hyvla" else smolvla_port
        ),
    )
    executor = (
        build_production_round_executor()
        if policy_backend != "act" or runtime_target is not None
        else _build_round_executor(native_act=False)
    )
    result = executor.execute(request)
    return (
        result.child_manifest,
        result.child_dir,
        result.round_summary,
        result.tool_evaluation,
        result.returncode,
    )


def main() -> None:
    args = parse_args()
    if args.benchmark == "libero":
        from mea.libero.chain import run_libero_agent_cli

        run_libero_agent_cli(args)
        return
    requested_open_query_planner = args.open_query_planner
    args.open_query_planner = resolve_default_open_query_planner(args)
    compat_profile_requested = paper_compat_profile_requested(
        args,
        requested_open_query_planner=requested_open_query_planner,
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
    claim_first_bound_plan_only = validate_and_normalize_agent_args(
        args,
        plan_agent_mode=claim_first_mode,
    )
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
    query_sufficiency_contract: dict[str, Any] | None = None
    if args.query_sufficiency_contract is not None:
        query_sufficiency_contract = load_query_sufficiency_contract(
            args.query_sufficiency_contract
        )
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
            schema_backed_only=(args.policy_backend == "act"),
        )
        runtime_discovery = discover_ready_plan_agent_targets(
            repo_root,
            open_task_inventory,
            max_rounds=(
                int(args.max_agent_rounds)
                if args.max_agent_rounds is not None
                else max(2, int(args.generated_rounds))
            ),
            policy_spec=runtime_policy_spec,
        )
        runtime_claim_first_targets = runtime_discovery["targets"]
        runtime_binding_excluded = runtime_discovery["excluded"]
        ready_tasks = sorted(runtime_claim_first_targets)
        assert args.bound_task_name is not None
        if args.bound_task_name not in ready_tasks:
            raise SystemExit(
                "bound task has no source/checkpoint runtime binding for "
                f"{args.policy_backend}: "
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
                schema_backed_only=(args.policy_backend == "act"),
            )
            runtime_discovery = discover_ready_plan_agent_targets(
                repo_root,
                open_task_inventory,
                max_rounds=(
                    int(args.max_agent_rounds)
                    if args.max_agent_rounds is not None
                    else max(2, int(args.generated_rounds))
                ),
                policy_spec=runtime_policy_spec,
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
                "no source/checkpoint-ready task is available for "
                f"{args.policy_backend}"
            )
        if args.bound_task_name is not None and args.bound_task_name not in ready_tasks:
            raise SystemExit(
                f"bound task is not {args.policy_backend}-ready: "
                f"{args.bound_task_name!r}"
            )
        global_planning_contexts = (
            {
                task_name: build_planning_context(
                    repo_root,
                    runtime_claim_first_targets[task_name],
                )
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
                else build_pending_task_binding_policy_card(
                    runtime_policy_spec,
                )
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
                experiment_needs=free_concern_bundle.get(
                    "experiment_needs"
                ),
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
                candidate_resolution=concern_candidate_resolution,
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
                    policy_name=(
                        runtime_policy_spec.policy_name
                        if runtime_policy_spec is not None
                        else "ACT"
                    ),
                    checkpoint_setting=(
                        str(
                            runtime_policy_spec.metadata.get(
                                "checkpoint_setting"
                            )
                        )
                        if runtime_policy_spec is not None
                        else "demo_clean"
                    ),
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
                "authority": (
                    "official_source_policy_checkpoint_with_optional_schema"
                ),
                "policy_backend": args.policy_backend,
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
                    runtime_policy_spec.policy_name
                    if runtime_policy_spec is not None
                    else (
                        "ACT"
                        if execution_backend in {"act", "both"}
                        else "expert"
                    )
                ),
                checkpoint_setting=(
                    str(
                        runtime_policy_spec.metadata.get(
                            "checkpoint_setting"
                        )
                    )
                    if runtime_policy_spec is not None
                    else "demo_clean"
                ),
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
    direct_single_candidate_query = False
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
        direct_single_candidate_query = bool(
            isinstance(concern_candidate_resolution, Mapping)
            and concern_candidate_resolution.get("resolution")
            == "official_execution_from_typed_needs"
            and concern_candidate_resolution.get("execution_authorized") is True
            and infer_claim_type(args.request) == "diagnostic"
        )
        claim_first_control_required = resolve_plan_agent_control_required(
            args.request,
            query_contract=query_sufficiency_contract,
            semantic_context=semantic_context,
            candidate_resolution=concern_candidate_resolution,
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
                    official_success_reuse=bool(
                        isinstance(concern_candidate_resolution, Mapping)
                        and concern_candidate_resolution.get(
                            "official_success_reuse"
                        )
                        is True
                    ),
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
                    candidate_universe_closed=direct_single_candidate_query,
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
        if args.task_module is not None and runtime_policy_spec is None:
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
                    policy_spec=runtime_policy_spec,
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
        initial_round, initial_tool_bundle = (
            taskgen_round_materialization.materialize_open_world_round(
                repo_root,
                evaluation_dir=evaluation_dir,
                round_number=1,
                candidate=initial_open_candidate,
                control_execution=manifest["initial_execution_binding"],
                policy_backend=args.policy_backend,
            )
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
    plan_session: BoundTaskPlanSession | PlanAgentSession | None = None
    plan_session_path: str | None = None
    evaluation_target: dict[str, Any] | None = None
    planning_context: dict[str, Any] | None = None
    proposal_agent: BoundedProposalAgent | None = None
    adaptive_step_agent: AdaptivePlanStepAgent | None = None
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
                explicit_candidate_aspect_ids = (
                    resolve_plan_agent_allowed_aspects(
                        args.bound_requested_aspect_ids
                    )
                )
                plan_session = PlanAgentSession(
                    args.request,
                    claim_first_initial_target,
                    query_contract=query_sufficiency_contract,
                    candidate_aspect_ids=explicit_candidate_aspect_ids,
                    require_control_anchor=claim_first_control_required,
                    control_round=(
                        plan["rounds"][0]
                        if claim_first_control_required
                        else None
                    ),
                )
                if frozen_first_open_candidate is not None:
                    frozen_first_open_candidate = (
                        plan_session.register_frozen_candidate(
                            frozen_first_open_candidate
                        )
                    )
            else:
                effective_round_budget = raw_round_budget
                if args.max_agent_rounds is not None:
                    effective_round_budget = min(
                        effective_round_budget, int(args.max_agent_rounds)
                    )
                    plan["max_rounds"] = effective_round_budget
                plan_session = BoundTaskPlanSession.from_catalog(
                    global_catalog,
                    args.task_name,
                    max_rounds=effective_round_budget,
                )
            plan = plan_session.normalize_plan(plan)
            planning_context = plan_session.planning_context(repo_root)
            write_json(evaluation_dir / "plan/planning_context.json", planning_context)
            if claim_first_mode:
                assert isinstance(plan_session, PlanAgentSession)
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
                plan_session.query_contract = persist_query_contract(
                    evaluation_dir,
                    plan,
                    plan_session.query_contract,
                )
                manifest.setdefault("planner", {}).update(
                    {
                        "public_planner": "PlanAgent",
                        "control_anchor_owned_by_runtime": (
                            plan_session.require_control_anchor
                        ),
                        "control_template_id": (
                            plan_session.control_template
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
                    target=plan_session.target,
                    planning_context=planning_context,
                    round_plan=first_round,
                    evaluation_dir=evaluation_dir,
                    round_number=1,
                )
                plan = plan_session.normalize_plan(plan)
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
            session_snapshot = plan_session.snapshot(args.request, plan)
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
        plan_session_path = "plan/bound_task_session.json"
        evaluation_target = session_snapshot["target"]
        write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
        write_json(evaluation_dir / plan_session_path, session_snapshot)
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
        policy_backend=args.policy_backend,
        policy_checkpoint_id=(
            runtime_policy_spec.checkpoint_id
            if runtime_policy_spec is not None
            else None
        ),
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
        plan_session_path=plan_session_path,
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
    if isinstance(plan_session, PlanAgentSession):
        if claim_first_agent is None or claim_first_capabilities is None:
            raise RuntimeError(
                "production Plan Agent application was not initialized"
            )
        application_result = PlanAgentApplication(
            repo_root=repo_root,
            evaluation_dir=evaluation_dir,
            evaluation_id=evaluation_id,
            user_request=args.request,
            plan=plan,
            session=plan_session,
            agent=claim_first_agent,
            capabilities=claim_first_capabilities,
            provider=provider,
            round_executor=build_production_round_executor(),
            models=models,
            base_url=args.base_url,
            gpu=args.gpu,
            max_reflections=args.max_reflections,
            telemetry_profile=args.telemetry_profile,
            policy_backend=args.policy_backend,
            runtime_target=claim_first_initial_target,
            policy_server_port=(
                args.hyvla_port
                if args.policy_backend == "hyvla"
                else args.smolvla_port
            ),
            materialize_round=(
                taskgen_round_materialization.materialize_open_world_round
            ),
            reviewed_task_registry=reviewed_task_registry,
            reviewed_tool_registry=reviewed_tool_registry,
            reviewed_vqa_registry=reviewed_vqa_registry,
            registration_identity=registration_identity,
            max_agent_rounds=args.max_agent_rounds,
            global_route_result=global_route_result,
            free_concern_bundle=free_concern_bundle,
            open_task_resolution=open_task_resolution,
            concern_candidate_resolution=concern_candidate_resolution,
            history_database=history_database,
            history_retrieval=history_retrieval,
            history_context_count=len(history_context),
            history_disabled=bool(args.no_history),
            cli_candidate_hint_used=(
                args.bound_requested_aspect_ids is not None
            ),
        ).run()
        print(
            json.dumps(
                application_result,
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    round_runs: list[dict[str, Any]] = []
    round_executor = build_production_round_executor()
    active_failure_stage = "round_execution"
    try:
        executed_rounds = 0
        while executed_rounds < len(plan["rounds"]):
            active_failure_stage = f"round_{executed_rounds + 1}_execution"
            round_plan = plan["rounds"][executed_rounds]
            round_result = round_executor.execute(
                RoundExecutionRequest(
                    repo_root=repo_root,
                    evaluation_dir=evaluation_dir,
                    evaluation_id=evaluation_id,
                    round_plan=round_plan,
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
                    policy_backend=args.policy_backend,
                    runtime_target=claim_first_initial_target,
                    policy_server_port=(
                        args.hyvla_port
                        if args.policy_backend == "hyvla"
                        else args.smolvla_port
                    ),
                )
            )
            child_manifest = round_result.child_manifest
            child_dir = round_result.child_dir
            round_summary = round_result.round_summary
            tool_evaluation = round_result.tool_evaluation
            returncode = round_result.returncode
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

            plan_before_decision = plan
            observation_history = [
                item["round_summary"] for item in round_runs
            ]
            dynamic_step_session = (
                plan_session is not None
                and adaptive_step_agent is not None
                and planning_context is not None
            )
            if (
                args.max_agent_rounds is not None
                and executed_rounds >= args.max_agent_rounds
            ):
                plan, decision, _cap_assessment = (
                    apply_external_hard_round_cap(
                        evaluation_dir=evaluation_dir,
                        plan=plan,
                        round_runs=round_runs,
                        executed_rounds=executed_rounds,
                        max_agent_rounds=args.max_agent_rounds,
                        user_request=args.request,
                        bound_plan_session=plan_session,
                    )
                )
                break
            if dynamic_step_session:
                active_failure_stage = (
                    f"adaptive_decision_after_round_{executed_rounds}"
                )
                navigation_options = plan_session.navigation_options(
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
                                target=plan_session.target,
                                planning_context=planning_context,
                                round_plan=materialized_round,
                                evaluation_dir=evaluation_dir,
                                round_number=next_round_number,
                            )
                        )
                active_failure_stage = (
                    f"plan_transition_after_round_{executed_rounds}"
                )
                plan, decision, runtime_directive = plan_session.apply_plan_step(
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
                    plan_session is not None
                    and not fixed_click_bell
                    and not legacy_click_bell
                    and candidate_decision.get("action") in {"continue", "stop"}
                )
                if common_adaptive_session:
                    plan, decision, runtime_directive = adjudicate_bounded_transition(
                        plan_session=plan_session,
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
            if plan_session is not None:
                # Persist and execute the exact normalized proposal-bearing plan;
                # snapshot() alone normalizes only a deep copy for reporting.
                plan = plan_session.normalize_plan(plan)
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
                if plan_session is not None:
                    write_json(
                        evaluation_dir / "plan/bound_task_session.json",
                        plan_session.snapshot(
                            args.request, plan, observation_history
                        ),
                    )
                break
            if plan_session is not None:
                write_json(
                    evaluation_dir / "plan/bound_task_session.json",
                    plan_session.snapshot(
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
        flagship_acceptance = None
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
