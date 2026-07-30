"""Bind open Plan Agent Proposals to executable method surfaces.

The Plan Agent may discover concerns outside a retrieval catalog. Openness does
not make every intervention executable by every backend: RoboTwin TaskGen
currently writes scene construction and success checking code, while
policy/controller perturbations require an explicit runtime hook.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


class ProposalExecutionError(ValueError):
    """Raised when a Proposal asks one backend to implement another surface."""


_RUNTIME_ONLY_PHRASES = (
    "action chunk",
    "action delay",
    "action latency",
    "action noise",
    "actuation noise",
    "control frequency",
    "control latency",
    "control precision",
    "controller gain",
    "controller precision",
    "gripper accuracy",
    "gripper precision",
    "inference latency",
    "policy parameter",
    "policy weight",
    "precision of the gripper",
    "动作延迟",
    "动作噪声",
    "控制器",
    "控制精度",
    "夹爪精度",
    "推理延迟",
    "策略参数",
    "策略权重",
)
_RUNTIME_HOOK_ROOTS = {
    "action_transform",
    "controller",
    "policy_runtime",
    "runtime_intervention",
}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _proposal_change_text(proposal: Mapping[str, Any]) -> str:
    perturbation = proposal.get("requested_perturbation")
    if not isinstance(perturbation, Mapping):
        return ""
    parts = [_text(perturbation.get("description"))]
    controlled = perturbation.get("controlled_changes")
    if isinstance(controlled, list):
        parts.extend(_text(item) for item in controlled)
    return "\n".join(part for part in parts if part)


def _candidate_change_text(candidate: Mapping[str, Any]) -> str:
    intent = candidate.get("evaluation_intent")
    if isinstance(intent, Mapping):
        requested_change = _text(intent.get("requested_change"))
        if requested_change:
            return requested_change
    scene_need = candidate.get("scene_need")
    if isinstance(scene_need, Mapping):
        return _text(scene_need.get("description"))
    return _text(scene_need)


def runtime_only_change_phrases(text: str) -> list[str]:
    """Return runtime-only intervention phrases present in one change."""

    normalized = _text(text).casefold()
    return [
        phrase
        for phrase in _RUNTIME_ONLY_PHRASES
        if phrase.casefold() in normalized
    ]


def _runtime_hooks_from_capabilities(
    capabilities: Mapping[str, Any],
) -> set[str]:
    generation = capabilities.get("generation_card")
    operations = (
        generation.get("taskgen_operations")
        if isinstance(generation, Mapping)
        else None
    )
    roots: set[str] = set()
    if isinstance(operations, list):
        for operation in operations:
            if not isinstance(operation, Mapping):
                continue
            allowed = operation.get("allowed_change_roots")
            if isinstance(allowed, list):
                roots.update(
                    _text(item).casefold()
                    for item in allowed
                    if _text(item)
                )
    return roots & _RUNTIME_HOOK_ROOTS


def validate_plan_agent_proposal_execution(
    proposal: Mapping[str, Any],
    *,
    capabilities: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a Proposal that needs an unadvertised runtime intervention."""

    result = deepcopy(dict(proposal))
    if result.get("action") != "continue":
        return result
    phrases = runtime_only_change_phrases(_proposal_change_text(result))
    if phrases and not _runtime_hooks_from_capabilities(capabilities):
        raise ProposalExecutionError(
            "requested perturbation requires a policy/controller runtime "
            "intervention that the capability cards do not advertise: "
            + ", ".join(phrases)
            + ". Propose a scene/checker/tool change implementable by the "
            "advertised roots, or report the unsupported capability."
        )
    return result


def validate_taskgen_candidate_execution(
    candidate: Mapping[str, Any],
    *,
    allowed_change_roots: Sequence[str],
) -> dict[str, Any]:
    """Reject TaskGen input whose requested change cannot live in its roots."""

    result = deepcopy(dict(candidate))
    phrases = runtime_only_change_phrases(_candidate_change_text(result))
    roots = {_text(item).casefold() for item in allowed_change_roots if _text(item)}
    if phrases and not (roots & _RUNTIME_HOOK_ROOTS):
        raise ProposalExecutionError(
            "TaskGen cannot implement the requested runtime intervention "
            "through the available change roots "
            f"{sorted(roots)}: "
            + ", ".join(phrases)
        )
    return result


__all__ = [
    "ProposalExecutionError",
    "runtime_only_change_phrases",
    "validate_plan_agent_proposal_execution",
    "validate_taskgen_candidate_execution",
]
