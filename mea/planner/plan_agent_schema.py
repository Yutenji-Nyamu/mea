"""Small schemas for open-Query Plan Agent decisions."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.planner.experiment_candidate import (
    ExperimentCandidateError,
    validate_scene_deltas,
)
from mea.planner.proposal_execution import validate_plan_agent_proposal_execution
from mea.taskgen.preservation_facts import (
    PreservationFactError,
    normalize_preservation_conditions,
)


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
    "answer",
    "claim_verdict",
    "evidence_sufficient",
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


def _controlled_changes(value: Any, field: str) -> list[str | dict[str, Any]]:
    """Accept frozen prose readers but make new numeric scene edits typed."""

    if not isinstance(value, list) or not value:
        raise ClaimFirstPlanError(f"{field} must be a non-empty list")
    if all(isinstance(item, str) for item in value):
        return _text_list(value, field, allow_empty=False)
    if not all(isinstance(item, Mapping) for item in value):
        raise ClaimFirstPlanError(
            f"{field} must contain either legacy strings or typed scene deltas"
        )
    try:
        return validate_scene_deltas(value)
    except ExperimentCandidateError as exc:
        raise ClaimFirstPlanError(f"{field}: {exc}") from exc


def _preservation_facts(value: Any, field: str) -> list[dict[str, str | None]]:
    if not isinstance(value, list):
        raise ClaimFirstPlanError(f"{field} must be a list")
    try:
        return normalize_preservation_conditions(value)
    except PreservationFactError as exc:
        raise ClaimFirstPlanError(f"{field}: {exc}") from exc


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
    result["answer"] = _optional_text(result.get("answer"), "answer")
    verdict = result.get("claim_verdict")
    if verdict not in {None, "supported", "refuted", "inconclusive"}:
        raise ClaimFirstPlanError(
            "claim_verdict must be null, supported, refuted, or inconclusive"
        )
    result["claim_verdict"] = verdict
    if not isinstance(result.get("evidence_sufficient"), bool):
        raise ClaimFirstPlanError("evidence_sufficient must be bool")
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
        if result["evidence_sufficient"]:
            if result["claim_verdict"] not in {"supported", "refuted"}:
                raise ClaimFirstPlanError(
                    "an evidence-sufficient stop needs a supported or refuted verdict"
                )
            if result["answer"] is None:
                raise ClaimFirstPlanError(
                    "an evidence-sufficient stop needs an answer"
                )
        elif result["claim_verdict"] != "inconclusive":
            raise ClaimFirstPlanError(
                "an evidence-insufficient stop must use an inconclusive verdict"
            )
        return result

    if result["answer"] is not None or result["claim_verdict"] is not None:
        raise ClaimFirstPlanError(
            "continue must set answer and claim_verdict to null"
        )
    if result["evidence_sufficient"]:
        raise ClaimFirstPlanError(
            "continue must set evidence_sufficient to false"
        )

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
        "controlled_changes": _controlled_changes(
            perturbation.get("controlled_changes"),
            "requested_perturbation.controlled_changes",
        ),
        "preserve": _preservation_facts(
            perturbation.get("preserve"),
            "requested_perturbation.preserve",
        ),
    }
    return result


__all__ = [
    "PlanAgentError",
    "project_open_query_capabilities",
    "validate_open_query_capabilities",
    "validate_open_query_evidence",
    "validate_open_query_plan_proposal",
]
