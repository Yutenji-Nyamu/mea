"""Reuse-first resolution for one validated TaskProposal.

The planner's capability contract describes the permitted materializer, but it
does not by itself prove that TaskGen attempted reuse before generation.  This
module keeps that decision explicit and side-effect free.  Built-in official
and bounded-overlay materializers win immediately; generated capabilities may
then consult an exact, explicitly reviewed artifact lookup before falling back
to code generation.

The semantic key deliberately excludes ``proposal_id`` and free-form
``intent``.  Rewording a Query may therefore reuse the same executable task,
while any change to task, aspect, capability, bounded changes, success
preservation, or the trusted materializer contract produces a different key.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable, Mapping

from mea.capability_adapter import (
    CapabilityAdapterError,
    taskgen_route,
    validate_capability_contract,
    validate_contract_changes,
)
from mea.proposals import ProposalError, validate_task_proposal


class TaskResolutionError(RuntimeError):
    """Raised when a proposal or reviewed reuse result is not an exact match."""


ReviewedTaskLookup = Callable[[dict[str, Any]], Mapping[str, Any] | None]


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TaskResolutionError(
            f"task resolution identity is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def task_semantic_key(
    task_proposal: Mapping[str, Any],
    capability_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact executable identity used by reviewed task lookup."""

    try:
        contract = validate_capability_contract(capability_contract)
        proposal = validate_task_proposal(
            task_proposal, expected_task_name=contract["task_name"]
        )
        proposal["changes"] = validate_contract_changes(
            contract, proposal["changes"]
        )
    except (CapabilityAdapterError, ProposalError, ValueError) as exc:
        raise TaskResolutionError(
            f"invalid TaskProposal resolution input: {exc}"
        ) from exc

    taskgen = contract["taskgen"]
    aspect = contract["aspect"]
    if proposal["capability_id"] != taskgen["capability_id"]:
        raise TaskResolutionError(
            "TaskProposal capability does not match the materializer contract"
        )
    if proposal["aspect_id"] != aspect["aspect_id"]:
        raise TaskResolutionError(
            "TaskProposal aspect does not match the materializer contract"
        )

    return {
        "schema_version": 1,
        "task_name": proposal["task_name"],
        "aspect_id": proposal["aspect_id"],
        "capability_id": proposal["capability_id"],
        "changes": deepcopy(proposal["changes"]),
        "preserve_success_semantics": proposal["preserve_success_semantics"],
        # TaskProposal v1 has no SuccessSpec.  Keeping an explicit null here
        # makes the reuse identity forward-compatible with a bounded v2
        # proposal without interpreting or validating that later schema here.
        "success_spec": deepcopy(proposal.get("success_spec")),
        "capability_contract_sha256": _canonical_sha256(contract),
    }


def _reviewed_registration(
    raw: Mapping[str, Any],
    *,
    semantic_key: Mapping[str, Any],
    semantic_key_sha256: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TaskResolutionError("reviewed task lookup must return an object or null")
    if raw.get("schema_version") != 1:
        raise TaskResolutionError("reviewed task registration schema_version must be 1")
    if raw.get("status") != "approved":
        raise TaskResolutionError("reviewed task registration is not approved")
    registration_id = raw.get("registration_id")
    artifact_id = raw.get("artifact_id")
    if not isinstance(registration_id, str) or not registration_id.strip():
        raise TaskResolutionError(
            "reviewed task registration_id must be a non-empty string"
        )
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise TaskResolutionError(
            "reviewed task artifact_id must be a non-empty string"
        )
    if raw.get("semantic_key_sha256") != semantic_key_sha256:
        raise TaskResolutionError("reviewed task semantic key hash does not match")
    if raw.get("semantic_key") != dict(semantic_key):
        raise TaskResolutionError("reviewed task semantic contract does not match")
    return {
        "registration_id": registration_id.strip(),
        "artifact_id": artifact_id.strip(),
        "status": "approved",
        "semantic_key_sha256": semantic_key_sha256,
    }


def resolve_task_proposal(
    task_proposal: Mapping[str, Any],
    capability_contract: Mapping[str, Any],
    *,
    find_reviewed: ReviewedTaskLookup | None = None,
) -> dict[str, Any]:
    """Resolve official/reuse/reviewed/generate without executing any stage.

    ``find_reviewed`` is intentionally a narrow injected interface.  Persistent
    registry storage and explicit admission remain separate concerns; the
    resolver accepts only an approved candidate that repeats the exact semantic
    key and hash supplied to the lookup.
    """

    try:
        contract = validate_capability_contract(capability_contract)
        proposal = validate_task_proposal(
            task_proposal, expected_task_name=contract["task_name"]
        )
    except (CapabilityAdapterError, ProposalError, ValueError) as exc:
        raise TaskResolutionError(
            f"invalid TaskProposal resolution input: {exc}"
        ) from exc
    semantic_key = task_semantic_key(proposal, contract)
    semantic_key_sha256 = _canonical_sha256(semantic_key)
    requested_route = taskgen_route(contract)
    operation = contract["taskgen"]["operation"]

    base = {
        "schema_version": 1,
        "task_name": proposal["task_name"],
        "proposal_id": proposal["proposal_id"],
        "requested_route": requested_route,
        "semantic_key": semantic_key,
        "semantic_key_sha256": semantic_key_sha256,
    }
    if operation == "official_passthrough":
        return {
            **base,
            "resolved_route": "official",
            "materialization": "official_reuse",
            "reason": "trusted official task satisfies the proposal",
            "provider_required": False,
            "reviewed_lookup_attempted": False,
            "reviewed_registration": None,
        }
    if operation in {"bounded_variant_overlay", "reuse_variant"}:
        return {
            **base,
            "resolved_route": "reuse",
            "materialization": (
                "bounded_overlay"
                if operation == "bounded_variant_overlay"
                else "built_in_variant_reuse"
            ),
            "reason": "trusted built-in materializer satisfies the proposal",
            "provider_required": False,
            "reviewed_lookup_attempted": False,
            "reviewed_registration": None,
        }
    if operation != "force_codegen":  # pragma: no cover - validated contract.
        raise TaskResolutionError(f"unsupported TaskGen operation: {operation!r}")

    query = {
        "schema_version": 1,
        "semantic_key": deepcopy(semantic_key),
        "semantic_key_sha256": semantic_key_sha256,
    }
    match = find_reviewed(deepcopy(query)) if find_reviewed is not None else None
    if match is not None:
        registration = _reviewed_registration(
            match,
            semantic_key=semantic_key,
            semantic_key_sha256=semantic_key_sha256,
        )
        return {
            **base,
            "resolved_route": "reviewed_generated_reuse",
            "materialization": "reviewed_generated_artifact",
            "reason": "exact approved generated task artifact match",
            "provider_required": False,
            "reviewed_lookup_attempted": True,
            "reviewed_registration": registration,
        }
    return {
        **base,
        "resolved_route": "force_codegen",
        "materialization": "generate",
        "reason": (
            "no exact approved reusable task artifact matched"
            if find_reviewed is not None
            else "reviewed task lookup is not configured; generate under the trusted contract"
        ),
        "provider_required": True,
        "reviewed_lookup_attempted": find_reviewed is not None,
        "reviewed_registration": None,
    }


__all__ = [
    "ReviewedTaskLookup",
    "TaskResolutionError",
    "resolve_task_proposal",
    "task_semantic_key",
]
