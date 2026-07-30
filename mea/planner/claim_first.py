"""Plan Agent open-Query planning without a predeclared aspect itinerary.

The existing adaptive planner deliberately selects from trusted executable
templates.  That is useful for production routing, but it cannot demonstrate
the paper's stronger claim that the Plan Agent discovers a small set of
sub-aspects while evaluation is in progress.  This module owns that earlier,
semantic planning step.

Only three kinds of information enter the model prompt:

* the original Query;
* a projection of policy/simulator generation capabilities that removes
  aspect ids, template ids, and navigation order;
* evidence from rounds that have already completed.

The resulting semantic proposal is intentionally not executable.  TaskGen and
ToolGen must subsequently resolve its requested perturbation and tool need.
There is no deterministic aspect fallback: provider failure is surfaced rather
than silently restoring a scripted route.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.planner.semantic_coverage import (
    SemanticCoverageError,
    build_candidate_intent_alignment,
    validate_evaluation_intent,
)
from mea.planner.open_task_resolver import (
    EXPERIMENTAL_SUCCESS_CHECKER_GUIDANCE,
    query_requires_experimental_checker,
)
from mea.planner.proposal_execution import (
    ProposalExecutionError,
    validate_plan_agent_proposal_execution,
)
from mea.providers.json_response import extract_json_response


class PlanAgentError(ValueError):
    """Raised when open-Query inputs or a semantic proposal are invalid."""


# Compatibility name for callers and historical exception handlers.
ClaimFirstPlanError = PlanAgentError


_CAPABILITY_KEYS = {
    "schema_version",
    "policy_card",
    "simulator_card",
    "generation_card",
}
_GENERATION_KEYS = {"backend_primitives"}
_BACKEND_PRIMITIVE_KEYS = {
    "scene",
    "checker",
    "telemetry",
    "rule",
    "vqa",
    "retrieve",
    "generate",
}
_LEGACY_GENERATION_KEYS = {"taskgen_operations", "toolgen"}
_TASKGEN_OPERATION_KEYS = {
    "operation",
    "controlled_axis",
    "generation_mode",
    "allowed_change_roots",
}
_TOOLGEN_KEYS = {
    "retrieve_first",
    "can_generate_rule_metric",
    "can_generate_vqa_question",
}
_EVIDENCE_KEYS = {
    "schema_version",
    "round_id",
    "tested_sub_aspect",
    "tested_hypothesis",
    "tested_perturbation",
    "outcome",
    "evidence_summary",
    "limitations",
}
_PROPOSAL_BASE_KEYS = {
    "schema_version",
    "action",
    "sub_aspect",
    "hypothesis",
    "requested_perturbation",
    "rationale",
}
_LEGACY_PROPOSAL_KEYS = _PROPOSAL_BASE_KEYS | {
    "task_need",
    "tool_need",
}
_TYPED_PROPOSAL_KEYS = _PROPOSAL_BASE_KEYS | {
    "scene_need",
    "checker_need",
    "rule_tool_need",
    "vqa_tool_need",
}
_CANONICAL_PROPOSAL_KEYS = (
    _LEGACY_PROPOSAL_KEYS | _TYPED_PROPOSAL_KEYS
)
_PERTURBATION_KEYS = {"description", "controlled_changes", "preserve"}
_NEED_KEYS = {"required", "description"}
_TOOL_NEED_KEYS = _NEED_KEYS | {"reuse_first"}
_OUTCOMES = {"success", "failure", "ambiguous"}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORBIDDEN_CAPABILITY_KEYS = {
    "aspect_id",
    "available_aspect_ids",
    "template_id",
    "template_ids",
    "fallback_step",
    "navigation_options",
}
_PLANNING_LINEAGE_KEYS = {
    "schema_version",
    "decision_kind",
    "evidence_conditioned",
    "completed_round_ids",
    "completed_round_count",
    "input_digest",
}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClaimFirstPlanError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _text_list(value: Any, field: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise ClaimFirstPlanError(f"{field} must be a list")
    result = [_text(item, f"{field}[]") for item in value]
    if not allow_empty and not result:
        raise ClaimFirstPlanError(f"{field} must not be empty")
    if len(result) != len(set(result)):
        raise ClaimFirstPlanError(f"{field} must not contain duplicates")
    return result


def _assert_no_navigation_keys(value: Any, *, field: str) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(set(value) & _FORBIDDEN_CAPABILITY_KEYS)
        if forbidden:
            raise ClaimFirstPlanError(
                f"{field} contains predeclared navigation fields: {forbidden}"
            )
        for key, nested in value.items():
            _assert_no_navigation_keys(nested, field=f"{field}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_navigation_keys(nested, field=f"{field}[{index}]")


def validate_open_query_capabilities(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a semantic capability projection with no aspect itinerary.

    Schema v2 exposes only backend primitives.  Schema v1 is retained as a
    reader for historical evidence bundles; it is never emitted by the current
    production projection.
    """

    if not isinstance(value, Mapping) or set(value) != _CAPABILITY_KEYS:
        raise ClaimFirstPlanError(
            f"OpenQueryCapabilities fields must be exactly "
            f"{sorted(_CAPABILITY_KEYS)}"
        )
    result = deepcopy(dict(value))
    if result.get("schema_version") not in {1, 2}:
        raise ClaimFirstPlanError(
            "OpenQueryCapabilities.schema_version must be 1 or 2"
        )
    _assert_no_navigation_keys(result, field="OpenQueryCapabilities")

    policy = result.get("policy_card")
    simulator = result.get("simulator_card")
    generation = result.get("generation_card")
    if not isinstance(policy, Mapping) or not policy:
        raise ClaimFirstPlanError("policy_card must be a non-empty object")
    if not isinstance(simulator, Mapping) or not simulator:
        raise ClaimFirstPlanError("simulator_card must be a non-empty object")
    if not isinstance(generation, Mapping):
        raise ClaimFirstPlanError("generation_card must be an object")
    generation_keys = set(generation)
    if generation_keys == _GENERATION_KEYS:
        if result["schema_version"] != 2:
            raise ClaimFirstPlanError(
                "backend_primitives generation cards require schema_version 2"
            )
        primitives = generation.get("backend_primitives")
        if (
            not isinstance(primitives, Mapping)
            or set(primitives) != _BACKEND_PRIMITIVE_KEYS
        ):
            raise ClaimFirstPlanError(
                "backend_primitives fields must be exactly "
                f"{sorted(_BACKEND_PRIMITIVE_KEYS)}"
            )
        if any(
            not isinstance(primitives.get(key), bool)
            for key in _BACKEND_PRIMITIVE_KEYS
        ):
            raise ClaimFirstPlanError(
                "all backend primitive capability flags must be bool"
            )
        result["policy_card"] = deepcopy(dict(policy))
        result["simulator_card"] = deepcopy(dict(simulator))
        result["generation_card"] = {
            "backend_primitives": deepcopy(dict(primitives))
        }
        return result

    if generation_keys != _LEGACY_GENERATION_KEYS:
        raise ClaimFirstPlanError(
            "generation_card fields must match either the current "
            f"{sorted(_GENERATION_KEYS)} or historical "
            f"{sorted(_LEGACY_GENERATION_KEYS)} shape"
        )
    if result["schema_version"] != 1:
        raise ClaimFirstPlanError(
            "historical taskgen_operations generation cards require "
            "schema_version 1"
        )
    operations = generation.get("taskgen_operations")
    if not isinstance(operations, list):
        raise ClaimFirstPlanError("taskgen_operations must be a list")
    normalized_operations: list[dict[str, Any]] = []
    for index, raw in enumerate(operations):
        if not isinstance(raw, Mapping) or set(raw) != _TASKGEN_OPERATION_KEYS:
            raise ClaimFirstPlanError(
                "each taskgen operation must have exactly "
                f"{sorted(_TASKGEN_OPERATION_KEYS)}"
            )
        operation = {
            "operation": _text(raw.get("operation"), f"taskgen_operations[{index}].operation"),
            "controlled_axis": _optional_text(
                raw.get("controlled_axis"),
                f"taskgen_operations[{index}].controlled_axis",
            ),
            "generation_mode": _optional_text(
                raw.get("generation_mode"),
                f"taskgen_operations[{index}].generation_mode",
            ),
            "allowed_change_roots": _text_list(
                raw.get("allowed_change_roots"),
                f"taskgen_operations[{index}].allowed_change_roots",
            ),
        }
        if operation not in normalized_operations:
            normalized_operations.append(operation)
    toolgen = generation.get("toolgen")
    if not isinstance(toolgen, Mapping) or set(toolgen) != _TOOLGEN_KEYS:
        raise ClaimFirstPlanError(
            f"toolgen fields must be exactly {sorted(_TOOLGEN_KEYS)}"
        )
    if any(not isinstance(toolgen.get(key), bool) for key in _TOOLGEN_KEYS):
        raise ClaimFirstPlanError("all toolgen capability flags must be bool")
    result["policy_card"] = deepcopy(dict(policy))
    result["simulator_card"] = deepcopy(dict(simulator))
    result["generation_card"] = {
        "taskgen_operations": normalized_operations,
        "toolgen": deepcopy(dict(toolgen)),
    }
    return result


def project_open_query_capabilities(
    planning_context: Mapping[str, Any],
    *,
    allowed_aspect_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Project one runtime into paper-level backend primitives.

    The caller is expected to validate/build ``planning_context`` through the
    repository-owned context adapter first.  Templates remain a retrieval
    index downstream; they do not define what the Plan Agent may propose.
    """

    if not isinstance(planning_context, Mapping):
        raise ClaimFirstPlanError("PlanningContext must be an object")
    policy = planning_context.get("policy_card")
    simulator = planning_context.get("simulator_card")
    if not isinstance(policy, Mapping):
        raise ClaimFirstPlanError("PlanningContext.policy_card must be an object")
    if not isinstance(simulator, Mapping):
        raise ClaimFirstPlanError("PlanningContext.simulator_card must be an object")
    if allowed_aspect_ids is not None:
        # Kept only so historical callers do not break.  The production
        # capability boundary is deliberately independent of routed aspects.
        _text_list(list(allowed_aspect_ids), "allowed_aspect_ids")

    projected_simulator = {
        key: deepcopy(nested)
        for key, nested in simulator.items()
        if key != "available_aspect_ids"
    }
    success_contract = simulator.get("success_contract")
    telemetry_available = bool(
        simulator.get("tracked_actors")
        or simulator.get("semantic_fields")
        or (
            isinstance(success_contract, Mapping)
            and success_contract.get("semantic_telemetry_available") is True
        )
    )
    return validate_open_query_capabilities(
        {
            "schema_version": 2,
            "policy_card": deepcopy(dict(policy)),
            "simulator_card": projected_simulator,
            "generation_card": {
                "backend_primitives": {
                    "scene": True,
                    "checker": True,
                    "telemetry": telemetry_available,
                    "rule": True,
                    "vqa": True,
                    "retrieve": True,
                    "generate": True,
                },
            },
        }
    )


def validate_open_query_evidence(
    value: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the compact evidence-only history shown to the Plan Agent."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ClaimFirstPlanError("evidence_history must be a sequence")
    result: list[dict[str, Any]] = []
    seen_round_ids: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _EVIDENCE_KEYS:
            raise ClaimFirstPlanError(
                f"evidence_history[{index}] fields must be exactly "
                f"{sorted(_EVIDENCE_KEYS)}"
            )
        item = deepcopy(dict(raw))
        if item.get("schema_version") != 1:
            raise ClaimFirstPlanError(
                f"evidence_history[{index}].schema_version must be 1"
            )
        round_id = _text(item.get("round_id"), f"evidence_history[{index}].round_id")
        if round_id in seen_round_ids:
            raise ClaimFirstPlanError(f"duplicate evidence round_id: {round_id!r}")
        seen_round_ids.add(round_id)
        outcome = item.get("outcome")
        if outcome not in _OUTCOMES:
            raise ClaimFirstPlanError(
                f"evidence_history[{index}].outcome must be one of "
                f"{sorted(_OUTCOMES)}"
            )
        item.update(
            {
                "round_id": round_id,
                "tested_sub_aspect": _text(
                    item.get("tested_sub_aspect"),
                    f"evidence_history[{index}].tested_sub_aspect",
                ),
                "tested_hypothesis": _text(
                    item.get("tested_hypothesis"),
                    f"evidence_history[{index}].tested_hypothesis",
                ),
                "tested_perturbation": _text(
                    item.get("tested_perturbation"),
                    f"evidence_history[{index}].tested_perturbation",
                ),
                "outcome": outcome,
                "evidence_summary": _text(
                    item.get("evidence_summary"),
                    f"evidence_history[{index}].evidence_summary",
                ),
                "limitations": _text_list(
                    item.get("limitations"),
                    f"evidence_history[{index}].limitations",
                ),
            }
        )
        result.append(item)
    return result


def _validate_need(
    value: Any,
    *,
    field: str,
    tool: bool,
) -> dict[str, Any]:
    keys = _TOOL_NEED_KEYS if tool else _NEED_KEYS
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ClaimFirstPlanError(f"{field} fields must be exactly {sorted(keys)}")
    required = value.get("required")
    if not isinstance(required, bool):
        raise ClaimFirstPlanError(f"{field}.required must be bool")
    description = _optional_text(value.get("description"), f"{field}.description")
    if required != (description is not None):
        raise ClaimFirstPlanError(
            f"{field}.description must be present exactly when required=true"
        )
    result = {"required": required, "description": description}
    if tool:
        if value.get("reuse_first") is not True:
            raise ClaimFirstPlanError(f"{field}.reuse_first must be true")
        result["reuse_first"] = True
    return result


def _empty_need(*, tool: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "required": False,
        "description": None,
    }
    if tool:
        result["reuse_first"] = True
    return result


def _compatibility_need(
    needs: Sequence[Mapping[str, Any]],
    *,
    tool: bool,
) -> dict[str, Any]:
    requested = [need for need in needs if need["required"]]
    if not requested:
        return _empty_need(tool=tool)
    descriptions = [
        str(need["description"])
        for need in requested
        if need.get("description") is not None
    ]
    result: dict[str, Any] = {
        "required": True,
        "description": " | ".join(descriptions),
    }
    if tool:
        result["reuse_first"] = True
    return result


def validate_open_query_plan_proposal(
    value: Mapping[str, Any],
    *,
    has_evidence: bool,
) -> dict[str, Any]:
    """Validate one semantic next-step decision without catalog enumeration."""

    if not isinstance(value, Mapping) or frozenset(value) not in {
        frozenset(_LEGACY_PROPOSAL_KEYS),
        frozenset(_TYPED_PROPOSAL_KEYS),
        frozenset(_CANONICAL_PROPOSAL_KEYS),
    }:
        raise ClaimFirstPlanError(
            "OpenQueryPlanProposal fields must match either the legacy "
            f"{sorted(_LEGACY_PROPOSAL_KEYS)} or typed "
            f"{sorted(_TYPED_PROPOSAL_KEYS)} shape"
        )
    result = deepcopy(dict(value))
    if result.get("schema_version") not in {1, 2}:
        raise ClaimFirstPlanError(
            "OpenQueryPlanProposal.schema_version must be 1 or 2"
        )
    action = result.get("action")
    if action not in {"continue", "stop"}:
        raise ClaimFirstPlanError("OpenQueryPlanProposal.action must be continue or stop")
    result["hypothesis"] = _text(result.get("hypothesis"), "hypothesis")
    result["rationale"] = _text(result.get("rationale"), "rationale")
    if "scene_need" in result:
        result["scene_need"] = _validate_need(
            result.get("scene_need"), field="scene_need", tool=False
        )
        result["checker_need"] = _validate_need(
            result.get("checker_need"), field="checker_need", tool=False
        )
        result["rule_tool_need"] = _validate_need(
            result.get("rule_tool_need"),
            field="rule_tool_need",
            tool=True,
        )
        result["vqa_tool_need"] = _validate_need(
            result.get("vqa_tool_need"),
            field="vqa_tool_need",
            tool=True,
        )
    else:
        # A legacy task_need did not say whether code generation concerned the
        # scene or success semantics.  Preserve it as one scene request only;
        # never silently expand one bit into both scene and checker work.
        legacy_task = _validate_need(
            result.get("task_need"), field="task_need", tool=False
        )
        legacy_tool = _validate_need(
            result.get("tool_need"), field="tool_need", tool=True
        )
        result["scene_need"] = deepcopy(legacy_task)
        result["checker_need"] = _empty_need(tool=False)
        result["rule_tool_need"] = deepcopy(legacy_tool)
        result["vqa_tool_need"] = _empty_need(tool=True)
    result["task_need"] = _compatibility_need(
        [result["scene_need"], result["checker_need"]],
        tool=False,
    )
    result["tool_need"] = _compatibility_need(
        [result["rule_tool_need"], result["vqa_tool_need"]],
        tool=True,
    )
    result["schema_version"] = 2

    if action == "stop":
        if not has_evidence:
            raise ClaimFirstPlanError("stop requires at least one completed evidence round")
        if result.get("sub_aspect") is not None:
            raise ClaimFirstPlanError("stop must set sub_aspect to null")
        if result.get("requested_perturbation") is not None:
            raise ClaimFirstPlanError("stop must set requested_perturbation to null")
        if any(
            result[field]["required"]
            for field in (
                "scene_need",
                "checker_need",
                "rule_tool_need",
                "vqa_tool_need",
            )
        ):
            raise ClaimFirstPlanError("stop cannot request TaskGen or ToolGen work")
        return result

    sub_aspect = _text(result.get("sub_aspect"), "sub_aspect")
    if not _IDENTIFIER.fullmatch(sub_aspect):
        raise ClaimFirstPlanError(
            "sub_aspect must be a semantic identifier using letters, digits, . _ or -"
        )
    perturbation = result.get("requested_perturbation")
    if not isinstance(perturbation, Mapping) or set(perturbation) != _PERTURBATION_KEYS:
        raise ClaimFirstPlanError(
            "requested_perturbation fields must be exactly "
            f"{sorted(_PERTURBATION_KEYS)}"
        )
    result["sub_aspect"] = sub_aspect
    result["requested_perturbation"] = {
        "description": _text(
            perturbation.get("description"), "requested_perturbation.description"
        ),
        "controlled_changes": _text_list(
            perturbation.get("controlled_changes"),
            "requested_perturbation.controlled_changes",
            allow_empty=False,
        ),
        "preserve": _text_list(
            perturbation.get("preserve"), "requested_perturbation.preserve"
        ),
    }
    return result


def open_query_input_digest(
    user_query: str,
    capabilities: Mapping[str, Any],
    evidence_history: Sequence[Mapping[str, Any]],
    evaluation_intent: Mapping[str, Any] | None = None,
) -> str:
    """Hash the exact semantic inputs used for one provider decision."""

    payload = {
        "user_query": _text(user_query, "user_query"),
        "capabilities": validate_open_query_capabilities(capabilities),
        "evidence_history": validate_open_query_evidence(evidence_history),
    }
    if evaluation_intent is not None:
        payload["evaluation_intent"] = validate_evaluation_intent(
            evaluation_intent
        )
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_open_query_planning_lineage(
    user_query: str,
    capabilities: Mapping[str, Any],
    evidence_history: Sequence[Mapping[str, Any]],
    evaluation_intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the exact completed evidence used to author one proposal.

    This is deliberately separate from execution provenance.  Its purpose is
    to distinguish a Query-only first proposal from a later Fig. 5 refinement
    that was authored only after the preceding Aggregate/Evidence was read.
    """

    trusted_evidence = validate_open_query_evidence(evidence_history)
    digest = open_query_input_digest(
        user_query,
        capabilities,
        trusted_evidence,
        evaluation_intent,
    )
    completed_round_ids = [
        item["round_id"] for item in trusted_evidence
    ]
    evidence_conditioned = bool(completed_round_ids)
    return {
        "schema_version": 1,
        "decision_kind": (
            "evidence_conditioned_refinement"
            if evidence_conditioned
            else "query_initial_candidate"
        ),
        "evidence_conditioned": evidence_conditioned,
        "completed_round_ids": completed_round_ids,
        "completed_round_count": len(completed_round_ids),
        "input_digest": digest,
    }


def validate_open_query_proposal_lineage(
    proposal_bundle: Mapping[str, Any],
    *,
    user_query: str,
    capabilities: Mapping[str, Any],
    evidence_history: Sequence[Mapping[str, Any]],
    evaluation_intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed when a proposal was not authored from current evidence.

    A provider response generated before round ``n`` completed must not be
    relabelled as the decision for round ``n + 1``.  The digest covers the
    original Query, capability boundary, complete evidence history, and an
    optional frozen EvaluationIntent.
    """

    if not isinstance(proposal_bundle, Mapping):
        raise ClaimFirstPlanError("proposal_bundle must be an object")
    expected = build_open_query_planning_lineage(
        user_query,
        capabilities,
        evidence_history,
        evaluation_intent,
    )
    actual_digest = proposal_bundle.get("input_digest")
    if actual_digest != expected["input_digest"]:
        raise ClaimFirstPlanError(
            "proposal input_digest does not match the current completed "
            "Aggregate/Evidence history"
        )
    raw_lineage = proposal_bundle.get("planning_lineage")
    if (
        not isinstance(raw_lineage, Mapping)
        or set(raw_lineage) != _PLANNING_LINEAGE_KEYS
    ):
        raise ClaimFirstPlanError(
            "provider proposal must carry complete planning_lineage"
        )
    lineage = deepcopy(dict(raw_lineage))
    if lineage != expected:
        raise ClaimFirstPlanError(
            "proposal planning_lineage does not match the current completed "
            "rounds"
        )
    return lineage


class PlanAgent:
    """Ask a provider to discover the next sub-aspect from evidence."""

    def __init__(self, provider: Any, *, model: str):
        self.provider = provider
        self.model = _text(model, "model")
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_errors: list[str] = []

    @staticmethod
    def _prompt(
        user_query: str,
        capabilities: Mapping[str, Any],
        evidence_history: Sequence[Mapping[str, Any]],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> str:
        checker_required = query_requires_experimental_checker(user_query)
        example = {
            "schema_version": 2,
            "action": "continue",
            "sub_aspect": "semantic.sub_aspect_discovered_now",
            "hypothesis": "A falsifiable statement this one round will test.",
            "requested_perturbation": {
                "description": (
                    "Set one advertised factor from its baseline to one "
                    "bounded diagnostic value."
                ),
                "controlled_changes": ["factor: baseline -> diagnostic value"],
                "preserve": ["task identity", "policy checkpoint"],
            },
            "scene_need": {
                "required": True,
                "description": "Scene construction or adaptation needed.",
            },
            "checker_need": {
                "required": checker_required,
                "description": (
                    "The additional experimental success predicate."
                    if checker_required
                    else None
                ),
            },
            "rule_tool_need": {
                "required": True,
                "description": "Numeric or symbolic Rule Tool observable needed.",
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "rationale": "Why this is the most informative next test for the Query.",
        }
        intent_section = ""
        if evaluation_intent is not None:
            intent_section = f"""
FROZEN EVALUATION INTENT (must be implemented directly):
{json.dumps(validate_evaluation_intent(evaluation_intent), ensure_ascii=False, indent=2)}

Every action=continue proposal must directly implement this frozen intent.
Do not silently replace it with a nearby diagnostic proxy.  Preserve its
requested change, preserved conditions, hypothesis, and required observation.
"""
        return f"""You are the Plan Agent in ManipEvalAgent.
Discover a small set of evaluation sub-aspects online.  There is no predeclared
candidate/template-ID itinerary, success-then-switch script, or fallback route.
The capability card exposes only backend primitives such as scene/checker
generation, telemetry, Rule/VQA Tools, and artifact retrieval.  It is an
execution boundary, not an operation menu or prescribed test order.  Choose
only the single most informative next experiment for the original Query, using
the policy/simulator capabilities and completed evidence below.

For action=continue, invent a precise semantic sub_aspect identifier and one
falsifiable hypothesis.  Request a bounded perturbation supported by the
capability cards.  Independently state whether the scene, success checker,
Rule Tool, and VQA Tool must be retrieved, created, or altered.  Do not request
a scene or checker merely because a Tool is needed, and do not couple scene
and checker needs.  A new Tool need may be named even when it is not in an
existing metric/question list.  Avoid repeating a tested perturbation unless
ambiguous evidence requires a more observable version.
Each Rule/VQA need must name one primary scalar or boolean observation for this
round.  Leave independent measurements for a later evidence-conditioned round
instead of bundling them into one Tool request.
For both rule_tool_need and vqa_tool_need, reuse_first MUST always be true,
including when required=false: retrieve-first is the ToolGen method contract,
not a choice to bypass reuse.
State the intentional delta in requested_perturbation.description and
controlled_changes with an explicit operation and concrete value or direction;
put unchanged conditions only in preserve.  When scene_need.required is true,
repeat that same explicit delta in scene_need.description.  Preserve only the
isolation-critical factors supported by an advertised simulator, frozen-binding,
or visual authority; do not claim that every unspecified state is unchanged.
When an additional experimental checker must retain the official goal, name
"official core predicate as a required conjunct" in preserve.  Do not call the
extended checker "official success semantics" or claim full equivalence.
TaskGen may retrieve or generate scene and checker code; ToolGen may retrieve
or generate Rule/VQA Tools.  These artifact primitives do not authorize policy
or controller intervention: do not reduce gripper precision, inject action
noise or latency, or change policy weights.  After successful evidence, refine
to another executable scene/checker/tool concern instead of relabelling a scene
change as an unavailable policy intervention.
{EXPERIMENTAL_SUCCESS_CHECKER_GUIDANCE}

Use success to probe the most consequential remaining uncertainty; use failure
to discriminate a causal failure hypothesis; use ambiguous evidence to improve
observability or isolate the confound.  When completed evidence is non-empty,
the rationale must cite a concrete observed outcome or limitation and explain
why it changed the priority of this sub-aspect.  Do not present a candidate
that was already frozen before seeing that evidence as evidence-conditioned
refinement.  If completed evidence contains a finite scalar, bracket the next
intervention or falsifiable threshold around that observed scale; do not invent
a distant numeric boundary unrelated to the measurement.  For a broad
robustness Query, for example, a successful control
can justify selecting the highest-risk supported perturbation, while a failed
control should redirect to baseline reliability or failure diagnosis.

Stop only when the completed evidence already answers the original Query.  For
action=stop set sub_aspect and
requested_perturbation to null, all four needs to
required=false/description=null, and express the evidence-supported conclusion
in hypothesis.

ORIGINAL QUERY:
{user_query}
{intent_section}

POLICY AND SIMULATOR CAPABILITIES:
{json.dumps(capabilities, ensure_ascii=False, indent=2)}

COMPLETED ROUND EVIDENCE (chronological; empty means first proposal):
{json.dumps(evidence_history, ensure_ascii=False, indent=2)}

Return strict JSON with exactly these fields:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""

    def propose(
        self,
        user_query: str,
        *,
        capabilities: Mapping[str, Any],
        evidence_history: Sequence[Mapping[str, Any]],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        query = _text(user_query, "user_query")
        trusted_capabilities = validate_open_query_capabilities(capabilities)
        trusted_evidence = validate_open_query_evidence(evidence_history)
        trusted_intent = (
            validate_evaluation_intent(evaluation_intent)
            if evaluation_intent is not None
            else None
        )
        prompt = self._prompt(
            query,
            trusted_capabilities,
            trusted_evidence,
            trusted_intent,
        )
        self.last_prompt = prompt
        self.last_responses = []
        self.last_errors = []

        proposal: dict[str, Any] | None = None
        for _attempt in range(2):
            attempt_prompt = prompt
            if self.last_errors:
                attempt_prompt += (
                    "\nPREVIOUS VALIDATION ERROR:\n"
                    + self.last_errors[-1]
                    + "\nReturn one complete corrected JSON object.\n"
                )
            try:
                response = self.provider.text(
                    attempt_prompt,
                    model=self.model,
                    system="Return only strict OpenQueryPlanProposal JSON.",
                    max_tokens=900,
                    temperature=0.0,
                )
                self.last_responses.append(response)
                proposal = validate_open_query_plan_proposal(
                    extract_json_response(response),
                    has_evidence=bool(trusted_evidence),
                )
                try:
                    proposal = validate_plan_agent_proposal_execution(
                        proposal,
                        capabilities=trusted_capabilities,
                    )
                except ProposalExecutionError as exc:
                    raise PlanAgentError(str(exc)) from exc
                if (
                    proposal["action"] == "continue"
                    and query_requires_experimental_checker(query)
                    and proposal["checker_need"]["required"] is not True
                ):
                    raise PlanAgentError(
                        "the original Query explicitly defines experimental "
                        "success semantics, so checker_need.required must be true"
                    )
                if trusted_intent is not None and proposal["action"] == "continue":
                    scene_need = proposal["scene_need"]
                    checker_need = proposal["checker_need"]
                    rule_tool_need = proposal["rule_tool_need"]
                    vqa_tool_need = proposal["vqa_tool_need"]
                    observation_contract = (
                        trusted_intent["required_observation"]
                        + "\n"
                        + trusted_intent["hypothesis"]
                    )
                    alignment = build_candidate_intent_alignment(
                        trusted_intent,
                        semantic_concern=(
                            trusted_intent["original_concern"]
                            + "\n"
                            + trusted_intent["hypothesis"]
                        ),
                        scene_need=(
                            {
                                "description": (
                                    trusted_intent["requested_change"]
                                    + "\nPreserve: "
                                    + "; ".join(
                                        trusted_intent[
                                            "preserved_conditions"
                                        ]
                                    )
                                )
                            }
                            if scene_need["required"]
                            else None
                        ),
                        checker_need=(
                            {"description": observation_contract}
                            if checker_need["required"]
                            else None
                        ),
                        rule_tool_need=(
                            {"description": observation_contract}
                            if rule_tool_need["required"]
                            else None
                        ),
                        vqa_tool_need=(
                            {"description": observation_contract}
                            if vqa_tool_need["required"]
                            else None
                        ),
                    )
                    if alignment["relationship"] != "direct":
                        raise ClaimFirstPlanError(
                            "proposal silently pivots to a diagnostic proxy; "
                            "it must directly implement the frozen "
                            "EvaluationIntent. Unmatched intent fields: "
                            + ", ".join(
                                alignment["unmatched_intent_fields"]
                            )
                        )
                break
            except SemanticCoverageError as exc:
                proposal = None
                self.last_errors.append(
                    f"{type(exc).__name__}: {exc}"
                )
            except Exception as exc:
                proposal = None
                self.last_errors.append(f"{type(exc).__name__}: {exc}")
        if proposal is None:
            raise ClaimFirstPlanError(
                "provider failed two open-Query proposal attempts: "
                + " | ".join(self.last_errors)
            )
        planning_lineage = build_open_query_planning_lineage(
            query,
            trusted_capabilities,
            trusted_evidence,
            trusted_intent,
        )
        return {
            "schema_version": 1,
            "source": "provider_plan_agent_open_query",
            "input_digest": planning_lineage["input_digest"],
            "planning_lineage": planning_lineage,
            "proposal": proposal,
            "provider": {
                "model_requested": self.model,
                "called": True,
                "attempt_count": len(self.last_responses),
                "errors": list(self.last_errors),
                "last_metadata": deepcopy(
                    dict(getattr(self.provider, "last_metadata", {}))
                ),
            },
        }


# Compatibility class name; new callers should use ``PlanAgent``.
ClaimFirstOpenQueryAgent = PlanAgent


__all__ = [
    "PlanAgent",
    "PlanAgentError",
    "ClaimFirstOpenQueryAgent",
    "ClaimFirstPlanError",
    "build_open_query_planning_lineage",
    "open_query_input_digest",
    "project_open_query_capabilities",
    "validate_open_query_capabilities",
    "validate_open_query_evidence",
    "validate_open_query_plan_proposal",
    "validate_open_query_proposal_lineage",
]
