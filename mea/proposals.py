"""Task-agnostic proposal contracts between Plan, TaskGen, and ToolGen.

The paper-level proposal is intentionally kept above the existing materialized
capability contract.  A proposal describes *what* one bound-task evaluation
round should test.  TaskGen and ToolGen then resolve it through their own
reuse-first registries; executable paths, checkpoints, seeds, and gates remain
runtime-owned.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping

from mea.aspects import AspectError, canonicalize_aspect_id
from mea.capability_adapter import (
    CapabilityAdapterError,
    registered_capability_contracts,
    taskgen_route,
    validate_capability_contract,
    validate_contract_changes,
)
from mea.taskgen.capabilities import CapabilityError, get_capability


class ProposalError(ValueError):
    """Raised when a semantic proposal exceeds the bound task contract."""


_TASK_PROPOSAL_V1_KEYS = {
    "schema_version",
    "proposal_id",
    "task_name",
    "aspect_id",
    "intent",
    "capability_id",
    "reuse_first",
    "changes",
    "preserve_success_semantics",
}
_TASK_PROPOSAL_V2_KEYS = _TASK_PROPOSAL_V1_KEYS | {"success_spec"}
_TOOL_PROPOSAL_V1_KEYS = {
    "schema_version",
    "proposal_id",
    "task_name",
    "aspect_id",
    "evaluation_goal",
    "metric",
    "question",
    "vqa_phenomenon_ids",
    "reuse_first",
}
_TOOL_PROPOSAL_V2_KEYS = _TOOL_PROPOSAL_V1_KEYS | {"vqa_question_specs"}
_TOOL_PROPOSAL_V3_KEYS = _TOOL_PROPOSAL_V2_KEYS | {"metric_spec"}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProposalError(f"{field} must be a non-empty string")
    return value.strip()


def _proposal_id(value: Any, field: str) -> str:
    normalized = _text(value, field)
    if not _IDENTIFIER.fullmatch(normalized):
        raise ProposalError(f"{field} must contain only letters, digits, . _ or -")
    return normalized


def validate_task_proposal(
    value: Mapping[str, Any],
    *,
    expected_task_name: str | None = None,
) -> dict[str, Any]:
    """Validate one TaskGen request without accepting executable fields."""

    if not isinstance(value, Mapping):
        raise ProposalError("TaskProposal must be an object")
    schema_version = value.get("schema_version")
    expected_keys = (
        _TASK_PROPOSAL_V1_KEYS
        if schema_version == 1
        else _TASK_PROPOSAL_V2_KEYS
        if schema_version == 2
        else None
    )
    if expected_keys is None:
        raise ProposalError("TaskProposal.schema_version must be 1 or 2")
    if set(value) != expected_keys:
        raise ProposalError(
            f"TaskProposal v{schema_version} fields must be exactly "
            f"{sorted(expected_keys)}"
        )
    proposal = deepcopy(dict(value))
    proposal["proposal_id"] = _proposal_id(
        proposal.get("proposal_id"), "TaskProposal.proposal_id"
    )
    task_name = _text(proposal.get("task_name"), "TaskProposal.task_name")
    if expected_task_name is not None and task_name != expected_task_name:
        raise ProposalError(
            f"TaskProposal cannot switch bound task {expected_task_name!r} to "
            f"{task_name!r}"
        )
    if schema_version == 2 and task_name != "beat_block_hammer":
        raise ProposalError(
            "TaskProposal v2 SuccessSpec only supports beat_block_hammer"
        )
    try:
        aspect_id = canonicalize_aspect_id(proposal.get("aspect_id"))
    except AspectError as exc:
        raise ProposalError(str(exc)) from exc
    capability_id = _text(
        proposal.get("capability_id"), "TaskProposal.capability_id"
    )
    if schema_version == 2 and (
        aspect_id != "object_appearance.color"
        or capability_id != "object_appearance.color"
    ):
        raise ProposalError(
            "TaskProposal v2 is capability-gated to "
            "beat_block_hammer/object_appearance.color"
        )
    official_passthrough = capability_id == "task_execution.official_passthrough"
    try:
        capability = (
            {"allowed_change_roots": []}
            if official_passthrough
            else get_capability(task_name, capability_id)
        )
    except CapabilityError as exc:
        raise ProposalError(str(exc)) from exc
    changes = proposal.get("changes")
    if not isinstance(changes, Mapping) or (not changes and not official_passthrough):
        raise ProposalError(
            "TaskProposal.changes must be non-empty outside official passthrough"
        )
    if official_passthrough and changes:
        raise ProposalError("official passthrough TaskProposal.changes must be empty")
    unknown_roots = sorted(set(changes) - set(capability["allowed_change_roots"]))
    if unknown_roots:
        raise ProposalError(
            f"TaskProposal changes exceed capability roots: {unknown_roots}"
        )
    if proposal.get("reuse_first") is not True:
        raise ProposalError("TaskProposal.reuse_first must be true")
    provider_scene_checker = (
        schema_version == 1
        and task_name == "beat_block_hammer"
        and capability_id == "robustness.distractor_avoidance"
        and aspect_id == "robustness.distractor_avoidance"
    )
    expected_preserve = not provider_scene_checker
    if schema_version == 1 and proposal.get(
        "preserve_success_semantics"
    ) is not expected_preserve:
        raise ProposalError(
            "TaskProposal.preserve_success_semantics must be false only for "
            "the BBH provider scene+checker capability"
        )
    if schema_version == 2:
        if official_passthrough:
            raise ProposalError(
                "TaskProposal v2 SuccessSpec requires a generated BBH capability"
            )
        if proposal.get("preserve_success_semantics") is not False:
            raise ProposalError(
                "TaskProposal v2 preserve_success_semantics must be false"
            )
        try:
            from mea.taskgen.success_spec import (
                SUCCESS_SPEC_V2_EXPERIMENTAL_ACT_ENVELOPE,
                success_spec_validation_report,
                validate_success_spec,
            )

            success_spec = validate_success_spec(proposal.get("success_spec"))
            success_report = success_spec_validation_report(success_spec)
        except ValueError as exc:
            raise ProposalError(f"invalid TaskProposal SuccessSpec: {exc}") from exc
        if (
            success_spec.get("envelope_id")
            != SUCCESS_SPEC_V2_EXPERIMENTAL_ACT_ENVELOPE
            or success_report["official_equivalent"]
            or not success_report["act_eligible"]
            or not success_report["experimental_bounded"]
        ):
            raise ProposalError(
                "TaskProposal v2 requires the trusted experimental bounded "
                "ACT SuccessSpec envelope"
            )
        proposal["success_spec"] = success_spec
    proposal.update(
        {
            "task_name": task_name,
            "aspect_id": aspect_id,
            "intent": _text(proposal.get("intent"), "TaskProposal.intent"),
            "capability_id": capability_id,
            "changes": deepcopy(dict(changes)),
        }
    )
    return proposal


def validate_tool_proposal(
    value: Mapping[str, Any],
    *,
    expected_task_name: str | None = None,
    expected_aspect_id: str | None = None,
) -> dict[str, Any]:
    """Validate the semantic Rule/VQA assignment for one TaskProposal."""

    if not isinstance(value, Mapping):
        raise ProposalError(
            "ToolProposal must be an object"
        )
    proposal = deepcopy(dict(value))
    schema_version = proposal.get("schema_version")
    expected_keys = (
        _TOOL_PROPOSAL_V1_KEYS
        if schema_version == 1
        else _TOOL_PROPOSAL_V2_KEYS
        if schema_version == 2
        else _TOOL_PROPOSAL_V3_KEYS
        if schema_version == 3
        else None
    )
    if expected_keys is None:
        raise ProposalError("ToolProposal.schema_version must be 1, 2, or 3")
    if set(proposal) != expected_keys:
        raise ProposalError(
            f"ToolProposal v{schema_version} fields must be exactly "
            f"{sorted(expected_keys)}"
        )
    proposal["proposal_id"] = _proposal_id(
        proposal.get("proposal_id"), "ToolProposal.proposal_id"
    )
    task_name = _text(proposal.get("task_name"), "ToolProposal.task_name")
    if expected_task_name is not None and task_name != expected_task_name:
        raise ProposalError(
            f"ToolProposal cannot switch bound task {expected_task_name!r} to "
            f"{task_name!r}"
        )
    try:
        aspect_id = canonicalize_aspect_id(proposal.get("aspect_id"))
        expected_aspect = (
            canonicalize_aspect_id(expected_aspect_id)
            if expected_aspect_id is not None
            else None
        )
    except AspectError as exc:
        raise ProposalError(str(exc)) from exc
    if expected_aspect is not None and aspect_id != expected_aspect:
        raise ProposalError(
            f"ToolProposal aspect {aspect_id!r} does not match TaskProposal "
            f"aspect {expected_aspect!r}"
        )
    phenomena = proposal.get("vqa_phenomenon_ids")
    if (
        not isinstance(phenomena, list)
        or not phenomena
        or any(not isinstance(item, str) or not item.strip() for item in phenomena)
        or len(phenomena) != len(set(phenomena))
    ):
        raise ProposalError(
            "ToolProposal.vqa_phenomenon_ids must be a non-empty unique string list"
        )
    if proposal.get("reuse_first") is not True:
        raise ProposalError("ToolProposal.reuse_first must be true")
    if schema_version in {2, 3}:
        raw_specs = proposal.get("vqa_question_specs")
        if not isinstance(raw_specs, list) or not raw_specs:
            raise ProposalError(
                "ToolProposal.vqa_question_specs must be a non-empty list"
            )
        try:
            from mea.execution_vqa import (
                is_run_local_phenomenon_id,
                validate_run_local_question_spec,
            )

            specs = [validate_run_local_question_spec(item) for item in raw_specs]
        except ValueError as exc:
            raise ProposalError(f"invalid run-local VQA question: {exc}") from exc
        ids = [item["id"] for item in specs]
        if len(ids) != len(set(ids)):
            raise ProposalError("ToolProposal.vqa_question_specs ids must be unique")
        selected_local_ids = [
            item for item in phenomena if is_run_local_phenomenon_id(item)
        ]
        if set(selected_local_ids) != set(ids):
            raise ProposalError(
                "ToolProposal run-local phenomenon ids must match "
                "vqa_question_specs exactly"
            )
        proposal["vqa_question_specs"] = specs
    if schema_version == 3:
        try:
            from mea.toolgen.metric_spec import validate_metric_spec

            proposal["metric_spec"] = validate_metric_spec(
                proposal.get("metric_spec")
            )
        except RuntimeError as exc:
            raise ProposalError(f"invalid ToolProposal MetricSpec: {exc}") from exc
    proposal.update(
        {
            "task_name": task_name,
            "aspect_id": aspect_id,
            "evaluation_goal": _text(
                proposal.get("evaluation_goal"), "ToolProposal.evaluation_goal"
            ),
            "metric": _text(proposal.get("metric"), "ToolProposal.metric"),
            "question": _text(
                proposal.get("question"), "ToolProposal.question"
            ),
            "vqa_phenomenon_ids": [item.strip() for item in phenomena],
        }
    )
    # Reuse the existing route-free Tool request validator so a proposal and
    # normal execution enter ToolGen through exactly the same public boundary.
    try:
        from mea.toolgen import validate_tool_request

        request = {
            "schema_version": 1,
            "task_name": str(proposal["task_name"]),
            "metric": str(proposal["metric"]),
            "question": str(proposal["question"]),
        }
        if schema_version == 3:
            request["schema_version"] = 2
            request["metric_spec"] = proposal["metric_spec"]
        validate_tool_request(request)
    except RuntimeError as exc:
        raise ProposalError(f"ToolProposal cannot form a Tool request: {exc}") from exc
    return proposal


def task_proposal_from_contract(
    contract: Mapping[str, Any],
    *,
    intent: str,
) -> dict[str, Any]:
    """Lift one materialized capability contract into the paper-level schema."""

    taskgen = contract.get("taskgen") if isinstance(contract, Mapping) else None
    aspect = contract.get("aspect") if isinstance(contract, Mapping) else None
    if not isinstance(taskgen, Mapping) or not isinstance(aspect, Mapping):
        raise ProposalError("capability contract lacks taskgen/aspect objects")
    proposal = {
        "schema_version": 1,
        "proposal_id": str(
            taskgen.get("task_variant_id") or contract.get("template_id")
        ),
        "task_name": contract.get("task_name"),
        "aspect_id": aspect.get("aspect_id"),
        "intent": intent,
        "capability_id": taskgen.get("capability_id"),
        "reuse_first": True,
        "changes": deepcopy(dict(taskgen.get("changes") or {})),
        "preserve_success_semantics": (
            taskgen.get("operation") != "provider_scene_checker_codegen"
        ),
    }
    return validate_task_proposal(
        proposal, expected_task_name=str(contract.get("task_name"))
    )


def tool_proposal_from_contract(
    contract: Mapping[str, Any],
    *,
    evaluation_goal: str,
) -> dict[str, Any]:
    """Lift the selected Rule/VQA assignment into one ToolProposal."""

    try:
        from mea.capability_adapter import build_contract_tool_request

        request = build_contract_tool_request(contract)
    except (RuntimeError, ValueError) as exc:
        raise ProposalError(f"cannot materialize contract Tool request: {exc}") from exc
    aspect = contract.get("aspect") if isinstance(contract, Mapping) else None
    vqa = contract.get("vqa") if isinstance(contract, Mapping) else None
    if not isinstance(aspect, Mapping) or not isinstance(vqa, Mapping):
        raise ProposalError("capability contract lacks aspect/vqa objects")
    proposal = {
        "schema_version": 1,
        "proposal_id": f"{contract.get('template_id')}.tool",
        "task_name": contract.get("task_name"),
        "aspect_id": aspect.get("aspect_id"),
        "evaluation_goal": evaluation_goal,
        "metric": request["metric"],
        "question": request["question"],
        "vqa_phenomenon_ids": deepcopy(list(vqa.get("phenomenon_ids") or [])),
        "reuse_first": True,
    }
    return validate_tool_proposal(
        proposal,
        expected_task_name=str(contract.get("task_name")),
        expected_aspect_id=str(aspect.get("aspect_id")),
    )


def tool_request_from_proposal(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the existing route-free Tool request consumed by ToolGen."""

    proposal = deepcopy(dict(value))
    proposal = validate_tool_proposal(proposal)
    request = {
        "schema_version": 1,
        "task_name": str(proposal["task_name"]),
        "metric": str(proposal["metric"]),
        "question": str(proposal["question"]),
    }
    if proposal["schema_version"] == 3:
        request["schema_version"] = 2
        request["metric_spec"] = deepcopy(proposal["metric_spec"])
    return request


def attach_round_proposals(round_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Make TaskProposal and ToolProposal authoritative for a materialized round."""

    result = deepcopy(dict(round_plan))
    contract = result.get("capability_contract")
    if not isinstance(contract, Mapping):
        raise ProposalError("round has no capability_contract")
    result["task_proposal"] = task_proposal_from_contract(
        contract,
        intent=str(result.get("task_instruction") or result.get("rationale") or ""),
    )
    result["tool_proposal"] = tool_proposal_from_contract(
        contract,
        evaluation_goal=str(result.get("rationale") or result["task_proposal"]["intent"]),
    )
    # Preserve old readers while ensuring the duplicated execution fields are
    # exact projections of the proposal boundary.
    result["tool_request"] = tool_request_from_proposal(result["tool_proposal"])
    result["vqa_phenomenon_ids"] = list(
        result["tool_proposal"]["vqa_phenomenon_ids"]
    )
    return result


def materialize_round_proposals(
    round_plan: Mapping[str, Any],
    task_proposal: Mapping[str, Any],
    tool_proposal: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one bounded model proposal onto a trusted materializer.

    The registered capability contract remains the authority for executable
    code paths, change roots, gates, and metric families.  The proposals own
    only this round's semantic variation and visual assignment.  This lets a
    query request an unseen-but-bounded variation without pretending that the
    new variation was already a registry template.
    """

    result = deepcopy(dict(round_plan))
    task_name = _text(result.get("task_name"), "round.task_name")
    task = validate_task_proposal(task_proposal, expected_task_name=task_name)
    tool = validate_tool_proposal(
        tool_proposal,
        expected_task_name=task_name,
        expected_aspect_id=task["aspect_id"],
    )
    try:
        from mea.execution_vqa import is_run_local_phenomenon_id

        catalog_phenomena = [
            item
            for item in tool["vqa_phenomenon_ids"]
            if not is_run_local_phenomenon_id(item)
        ]
    except ImportError:  # pragma: no cover - package is present in normal runs.
        catalog_phenomena = list(tool["vqa_phenomenon_ids"])
    candidates = []
    for raw_contract in registered_capability_contracts(task_name):
        contract = validate_capability_contract(raw_contract)
        if (
            contract["aspect"]["aspect_id"] == task["aspect_id"]
            and contract["taskgen"]["capability_id"] == task["capability_id"]
            and (
                contract["tool"]["metric"] == tool["metric"]
                or tool.get("metric_spec") is not None
            )
            and set(catalog_phenomena)
            <= set(contract["vqa"]["phenomenon_ids"])
        ):
            candidates.append(contract)
    if not candidates:
        raise ProposalError(
            "no trusted TaskGen/ToolGen materializer can satisfy this proposal"
        )
    current_template = str(result.get("template_id") or "")
    contract = next(
        (
            item
            for item in candidates
            if item["template_id"] == current_template
        ),
        candidates[0],
    )
    try:
        task["changes"] = validate_contract_changes(contract, task["changes"])
    except CapabilityAdapterError as exc:
        raise ProposalError(f"TaskProposal exceeds its capability envelope: {exc}") from exc
    if task_name == "click_bell" and task["capability_id"] == "object_position.fixed_xy":
        try:
            from mea.taskgen.click_bell import validate_click_bell_variant_hint

            task["changes"] = validate_click_bell_variant_hint(task["changes"])
        except RuntimeError as exc:
            raise ProposalError(f"invalid click_bell position proposal: {exc}") from exc

    result.update(
        {
            "template_id": contract["template_id"],
            "capability_id": contract["taskgen"]["capability_id"],
            "task_variant_id": task["proposal_id"],
            "capability_contract": contract,
            "sub_aspect": task["aspect_id"],
            "aspect_id": task["aspect_id"],
            "route": taskgen_route(contract),
            "variant_hint": deepcopy(task["changes"]),
            "task_proposal": task,
            "tool_proposal": tool,
            "tool_request": tool_request_from_proposal(tool),
            "vqa_phenomenon_ids": list(tool["vqa_phenomenon_ids"]),
            "proposal_materialization": {
                "schema_version": 1,
                "mode": "query_generated_bounded_variation",
                "base_template_id": contract["template_id"],
                "capability_contract_is_authority_envelope": True,
                "task_proposal_is_round_variation_authority": True,
            },
        }
    )
    execution = result.get("execution")
    if isinstance(execution, Mapping):
        result["execution"] = deepcopy(dict(execution))
        result["execution"]["gates"] = list(contract["required_gates"])
    instruction = str(result.get("task_instruction") or "").strip()
    result["task_instruction"] = (
        f"{instruction} Query-generated bounded variation: {task['intent']}"
    ).strip()
    return result


__all__ = [
    "ProposalError",
    "attach_round_proposals",
    "materialize_round_proposals",
    "task_proposal_from_contract",
    "tool_proposal_from_contract",
    "tool_request_from_proposal",
    "validate_task_proposal",
    "validate_tool_proposal",
]
