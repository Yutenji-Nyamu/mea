"""Cold bounded Proposal and legacy PlanStep compatibility helpers.

These routines preserve historical catalog/fixed paper protocols.  The
production Plan Agent authors typed Proposals directly and must not import
this task-specific transport at module import time.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from experiments.paper.compat_capability_adapter import taskgen_route
from mea.planner import AdaptivePlanStepAgent, BoundTaskPlanSession
from mea.planner import validate_open_query_plan_proposal
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
from mea.toolgen import route_tool_request


COMPAT_PROPOSAL_ERRORS = (ProposalError, ProposalAgentError)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_bounded_proposal_agent(
    provider: Any,
    *,
    model: str,
) -> BoundedProposalAgent:
    return BoundedProposalAgent(provider, model=model)


def create_adaptive_plan_step_agent(
    provider: Any,
    *,
    model: str,
) -> AdaptivePlanStepAgent:
    return AdaptivePlanStepAgent(provider, model=model)


def create_bound_task_plan_session(
    catalog: Mapping[str, Any],
    task_name: str,
    *,
    max_rounds: int | None = None,
) -> BoundTaskPlanSession:
    if max_rounds is None:
        return BoundTaskPlanSession.from_catalog(catalog, task_name)
    return BoundTaskPlanSession.from_catalog(
        catalog,
        task_name,
        max_rounds=max_rounds,
    )


def _bound_target_task_name(target: Mapping[str, Any]) -> str:
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
    """Author one historical catalog-bounded Task/Tool Proposal."""

    round_plan = deepcopy(round_plan)
    bound_task_name = _bound_target_task_name(target)
    supplied_task_name = round_plan.get("task_name")
    if supplied_task_name not in {None, bound_task_name}:
        raise RuntimeError("round proposal cannot change the bound task")
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
    taskgen_requested = any(
        isinstance(need, Mapping) and need.get("required") is True
        for need in (scene_need, checker_need)
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
        semantic_rule_tool_requested and not tool_satisfied_by_task_checker
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
            "tool_satisfied_by_task_checker": tool_satisfied_by_task_checker,
        }
    proposal_dir = (
        evaluation_dir / "plan/bounded_proposal" / f"round_{round_number:02d}"
    )
    proposal_dir.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] | None = None
    proposal_source = "BoundedProposalAgent"
    prompt_text = ""
    try:
        if provider_scene_checker_requested and semantic_proposal is not None:
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
                    "proposal_candidate_bundle.json"
                    if bundle is not None
                    else None
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
            "tool_satisfied_by_task_checker": tool_satisfied_by_task_checker,
        },
    }
    write_json(proposal_dir / "proposal_bundle.json", artifact)
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


__all__ = [
    "COMPAT_PROPOSAL_ERRORS",
    "adjudicate_bounded_transition",
    "apply_bounded_round_proposal",
    "create_adaptive_plan_step_agent",
    "create_bound_task_plan_session",
    "create_bounded_proposal_agent",
    "persist_adaptive_step_selection",
]
