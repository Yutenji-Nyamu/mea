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
from mea.taskgen.capabilities import CapabilityError, get_capability


class ProposalError(ValueError):
    """Raised when a semantic proposal exceeds the bound task contract."""


_TASK_PROPOSAL_KEYS = {
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
_TOOL_PROPOSAL_KEYS = {
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

    if not isinstance(value, Mapping) or set(value) != _TASK_PROPOSAL_KEYS:
        raise ProposalError(
            f"TaskProposal fields must be exactly {sorted(_TASK_PROPOSAL_KEYS)}"
        )
    proposal = deepcopy(dict(value))
    if proposal.get("schema_version") != 1:
        raise ProposalError("TaskProposal.schema_version must be 1")
    proposal["proposal_id"] = _proposal_id(
        proposal.get("proposal_id"), "TaskProposal.proposal_id"
    )
    task_name = _text(proposal.get("task_name"), "TaskProposal.task_name")
    if expected_task_name is not None and task_name != expected_task_name:
        raise ProposalError(
            f"TaskProposal cannot switch bound task {expected_task_name!r} to "
            f"{task_name!r}"
        )
    try:
        aspect_id = canonicalize_aspect_id(proposal.get("aspect_id"))
    except AspectError as exc:
        raise ProposalError(str(exc)) from exc
    capability_id = _text(
        proposal.get("capability_id"), "TaskProposal.capability_id"
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
    if proposal.get("preserve_success_semantics") is not True:
        raise ProposalError(
            "TaskProposal.preserve_success_semantics must be true"
        )
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

    if not isinstance(value, Mapping) or set(value) != _TOOL_PROPOSAL_KEYS:
        raise ProposalError(
            f"ToolProposal fields must be exactly {sorted(_TOOL_PROPOSAL_KEYS)}"
        )
    proposal = deepcopy(dict(value))
    if proposal.get("schema_version") != 1:
        raise ProposalError("ToolProposal.schema_version must be 1")
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

        validate_tool_request(tool_request_from_proposal(proposal))
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
        "preserve_success_semantics": True,
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
    if set(proposal) != _TOOL_PROPOSAL_KEYS:
        proposal = validate_tool_proposal(proposal)
    return {
        "schema_version": 1,
        "task_name": str(proposal["task_name"]),
        "metric": str(proposal["metric"]),
        "question": str(proposal["question"]),
    }


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


__all__ = [
    "ProposalError",
    "attach_round_proposals",
    "task_proposal_from_contract",
    "tool_proposal_from_contract",
    "tool_request_from_proposal",
    "validate_task_proposal",
    "validate_tool_proposal",
]
