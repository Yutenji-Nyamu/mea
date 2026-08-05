"""Validate the legacy reviewed capability envelope attached to one round."""

from __future__ import annotations

from typing import Any

from mea.execution_vqa.query import is_run_local_phenomenon_id
from mea.proposals import (
    ProposalError,
    tool_request_from_proposal,
    validate_task_proposal,
    validate_tool_proposal,
)
from mea.toolgen import route_tool_request


def validate_round_capability_contract(
    round_plan: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind every duplicated runtime field to one trusted adapter contract."""

    raw = round_plan.get("capability_contract")
    if raw is None:
        return None
    from experiments.paper.compat_capability_adapter import taskgen_route
    from mea.artifact_retrieval_index import (
        ArtifactRetrievalIndexError,
        build_contract_tool_request,
        validate_capability_contract,
        validate_contract_changes,
    )

    try:
        contract = validate_capability_contract(raw)
        registered_tool = build_contract_tool_request(contract)
    except (ArtifactRetrievalIndexError, ValueError) as exc:
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
        except (ProposalError, ArtifactRetrievalIndexError) as exc:
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


__all__ = ["validate_round_capability_contract"]
