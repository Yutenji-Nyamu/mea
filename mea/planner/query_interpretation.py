"""Query interpretation and Proposal binding for the Plan Agent.

This module owns the semantic boundary between an open user Query and a typed
Proposal. Retrieval is advisory: it may identify reusable artifacts, but it
never restricts what the Plan Agent is allowed to propose.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping

from mea.artifact_retrieval_index import resolve_task_retrieval_index

from .plan_agent_schema import validate_open_query_plan_proposal
from .experiment_candidate import build_experiment_candidate
from .open_task_resolver import (
    validate_free_concern,
    validate_free_concern_experiment_needs,
)
from .plan_agent_errors import PlanAgentSessionError
from .policy_task_binding import (
    PolicyTaskBindingError,
    policy_task_binding_from_target,
)
from .semantic_coverage import build_evaluation_intent, validate_evaluation_intent


def build_initial_semantic_proposal_bundle(
    *,
    user_query: str,
    concern: Mapping[str, Any],
    experiment_needs: Mapping[str, Any],
    evaluation_intent: Mapping[str, Any],
    provider_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize a Query-only candidate before any runtime evidence exists.

    Query interpretation already chose the first sub-aspect, hypothesis,
    perturbation, observation, and independent Task/Tool needs before catalog
    retrieval.  This adapter preserves that Query-authored seed for a no-control
    first round or a legacy protocol.  It is explicitly marked
    ``pre_evidence_query_candidate`` and must never be reported as a Fig. 5
    evidence-conditioned refinement.
    """

    query = _nonempty_text(user_query, "user_query")
    trusted_concern = validate_free_concern(
        concern,
        expected_query=query,
    )
    trusted_needs = validate_free_concern_experiment_needs(
        experiment_needs
    )
    intent = validate_evaluation_intent(evaluation_intent)
    if intent["source_query"] != query:
        raise PlanAgentSessionError(
            "EvaluationIntent source_query differs from the original Query"
        )
    slug = re.sub(
        r"[^a-z0-9]+",
        ".",
        trusted_concern["sub_aspect"].casefold(),
    ).strip(".")
    if not slug:
        slug = intent["intent_id"].removeprefix("intent.")
    proposal = validate_open_query_plan_proposal(
        {
            "schema_version": 2,
            "action": "continue",
            "sub_aspect": f"query_interpretation.{slug[:96]}",
            "hypothesis": intent["hypothesis"],
            "requested_perturbation": {
                "description": intent["requested_change"],
                "controlled_changes": [intent["requested_change"]],
                "preserve": deepcopy(intent["preserved_conditions"]),
            },
            **deepcopy(trusted_needs),
            "rationale": (
                "Directly execute the catalog-free first concern selected for "
                "the original Query; no second Planner may replace it before "
                "the control evidence is observed."
            ),
            "answer": None,
            "claim_verdict": None,
            "evidence_sufficient": False,
        },
        has_evidence=False,
    )
    return {
        "schema_version": 1,
        "source": "provider_plan_agent_direct_materialization",
        "input_intent_id": intent["intent_id"],
        "proposal": proposal,
        "provider": deepcopy(dict(provider_record or {})),
    }


def build_dynamic_experiment_candidate(
    *,
    user_query: str,
    task_name: str,
    proposal: Mapping[str, Any],
    evaluation_intent: Mapping[str, Any] | None = None,
    official_success_reuse: bool = False,
) -> dict[str, Any]:
    """Bind one semantic proposal without rewriting its first-candidate intent.

    The provider owns the four independent generation/tool requirements.  For
    the first Query-derived candidate, the Query interpretation owns
    the exact change, hypothesis, preserved conditions, and observation text.
    Later evidence-driven pivots receive a fresh per-candidate intent derived
    directly from their proposal, so every round carries the same traceable
    experiment contract.
    """

    if not isinstance(official_success_reuse, bool):
        raise PlanAgentSessionError("official_success_reuse must be bool")
    trusted = validate_open_query_plan_proposal(proposal, has_evidence=True)
    if trusted["action"] != "continue":
        raise PlanAgentSessionError(
            "only a continue decision can become an executable Proposal"
        )
    perturbation = trusted["requested_perturbation"]
    scene_need = trusted["scene_need"]
    checker_need = trusted["checker_need"]
    rule_tool_need = trusted["rule_tool_need"]
    vqa_tool_need = trusted["vqa_tool_need"]
    observation_parts = [
        str(need.get("description") or "").strip()
        for need in (checker_need, rule_tool_need, vqa_tool_need)
        if need["required"]
    ]
    proposal_intent = build_evaluation_intent(
        source_query=user_query,
        original_concern=trusted["sub_aspect"],
        hypothesis=trusted["hypothesis"],
        requested_change=perturbation["description"],
        preserved_conditions=perturbation["preserve"],
        required_observation=(
            " ".join(part for part in observation_parts if part)
            or "Observe the policy outcome needed to decide: "
            + trusted["hypothesis"]
        ),
    )
    intent = (
        validate_evaluation_intent(evaluation_intent)
        if evaluation_intent is not None
        else proposal_intent
    )
    exact_hypothesis = intent["hypothesis"]
    semantic_concern = (
        f"{intent['original_concern']}: {exact_hypothesis}"
    )
    scene_description = (
        str(scene_need["description"]).strip()
        if scene_need["required"]
        else ""
    )
    # Typed preservation facts travel in EvaluationIntent. Do not translate
    # them back into prose and then attempt to recover them inside TaskGen.
    return build_experiment_candidate(
        source_query=_nonempty_text(user_query, "user_query"),
        base_task=_nonempty_text(task_name, "task_name"),
        semantic_concern=semantic_concern,
        scene_need=(
            {
                "kind": "adapt",
                "description": scene_description,
                "reuse_first": True,
            }
            if scene_need["required"]
            else None
        ),
        checker_need=(
            {
                "kind": "generate",
                "description": str(checker_need["description"]).strip(),
                "reuse_first": True,
            }
            if checker_need["required"]
            else None
        ),
        rule_tool_need=(
            {
                "kind": (
                    "reuse"
                    if official_success_reuse
                    else "measure"
                ),
                "description": str(rule_tool_need["description"]).strip(),
                "reuse_first": True,
            }
            if rule_tool_need["required"]
            else None
        ),
        vqa_tool_need=(
            {
                "kind": "vqa",
                "description": str(vqa_tool_need["description"]).strip(),
                "reuse_first": True,
            }
            if vqa_tool_need["required"]
            else None
        ),
        evaluation_intent=intent,
    )

def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanAgentSessionError(f"{field} must be a non-empty string")
    return value.strip()


def _target_task_name(target: Mapping[str, Any]) -> str:
    if "policy_task_binding" in target:
        try:
            return policy_task_binding_from_target(target)["task_name"]
        except (PolicyTaskBindingError, TypeError) as exc:
            raise PlanAgentSessionError(str(exc)) from exc
    return _nonempty_text(target.get("task_name"), "target.task_name")


def control_template_id(target: Mapping[str, Any]) -> str:
    """Return the trusted official-scene control for a bound task."""

    task_name = _target_task_name(target)
    retrieval_index = resolve_task_retrieval_index(
        task_name,
        allow_unregistered=True,
    )
    template_id = retrieval_index["control_template_id"]
    if "policy_task_binding" in target:
        return template_id
    available = {
        str(item)
        for aspect in target.get("aspects", [])
        if isinstance(aspect, Mapping)
        for item in aspect.get("template_ids", [])
    }
    if template_id not in available:
        # Cached plans and older test fixtures may predate the neutral
        # official-baseline contract.  Keep them readable by accepting only an
        # already-bound unchanged-scene passthrough; newly built targets always
        # expose ``task_execution.official_baseline``.
        legacy_controls = [
            contract["template_id"]
            for contract in retrieval_index["entries"]
            if contract["template_id"] in available
            and contract["taskgen"]["operation"] == "official_passthrough"
        ]
        if len(legacy_controls) == 1:
            return legacy_controls[0]
        raise PlanAgentSessionError(
            f"control template {template_id!r} is outside the bound task"
        )
    return template_id


def resolve_concern_candidate_domain(
    concern: Mapping[str, Any],
    *,
    experiment_needs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Admit typed needs without reinterpreting their prose against a catalog.

    A no-TaskGen request can execute the unchanged official task directly.
    Every other concern remains owned by the Plan Agent, whose Proposal names
    the scene, checker, and Tool work to retrieve or generate.
    """

    if not isinstance(concern, Mapping):
        raise PlanAgentSessionError("Query interpretation must be an object")
    for field in (
        "source_query",
        "sub_aspect",
        "hypothesis",
        "requested_variation",
        "measurement_need",
    ):
        _nonempty_text(
            concern.get(field), f"QueryInterpretation.{field}"
        )
    trusted_needs = (
        validate_free_concern_experiment_needs(experiment_needs)
        if experiment_needs is not None
        else None
    )
    if (
        trusted_needs is not None
        and trusted_needs["scene_need"]["required"] is False
        and trusted_needs["checker_need"]["required"] is False
    ):
        rule_description = str(
            trusted_needs["rule_tool_need"].get("description") or ""
        ).casefold()
        official_success_reuse = bool(
            trusted_needs["rule_tool_need"]["required"]
            and "official" in rule_description
            and "check_success" in rule_description
        )
        return {
            "schema_version": 1,
            "decision": "official_execution",
            "resolution": "official_execution_from_typed_needs",
            "concern": deepcopy(dict(concern)),
            "experiment_needs": deepcopy(trusted_needs),
            "taskgen_required": False,
            "official_success_reuse": official_success_reuse,
            "execution_authorized": True,
        }
    return {
        "schema_version": 1,
        "decision": "proposal_reuse_or_generate",
        "resolution": "proposal_reuse_or_generate",
        "concern": deepcopy(dict(concern)),
        "experiment_needs": (
            deepcopy(trusted_needs) if trusted_needs is not None else None
        ),
        "taskgen_required": bool(
            trusted_needs is not None
            and (
                trusted_needs["scene_need"]["required"]
                or trusted_needs["checker_need"]["required"]
            )
        ),
        "proposal_selection_required": True,
        "execution_authorized": True,
    }


__all__ = [
    "build_dynamic_experiment_candidate",
    "build_initial_semantic_proposal_bundle",
    "control_template_id",
    "resolve_concern_candidate_domain",
]
