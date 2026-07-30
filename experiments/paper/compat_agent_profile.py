"""Legacy Agent profile parsing and validation for explicit paper protocols.

The production entry point exposes the paper's Plan Agent.  This module owns
only the hidden
catalog/fixed/registered and task-specific compatibility surface retained for
paper experiments.  Production code imports it lazily after such a profile is
actually requested or resolved.
"""

from __future__ import annotations

from typing import Any


class CompatAgentProfileError(ValueError):
    """Raised when an explicit legacy paper profile is internally inconsistent."""


def compat_agent_profile_requested(
    args: Any,
    *,
    requested_open_query_planner: str | None,
) -> bool:
    """Return whether parsed arguments select the paper compatibility surface."""

    registered_values = (
        getattr(args, "evidence_manifest", None),
        getattr(args, "command_plan", None),
        getattr(args, "registered_route", None),
        getattr(args, "registered_strategy", None),
    )
    return bool(
        requested_open_query_planner == "catalog_step_v1"
        or getattr(args, "task_profile", "official") != "official"
        or getattr(args, "planning_policy", "dynamic_evidence_v1")
        != "dynamic_evidence_v1"
        or getattr(args, "proposal_mode", "catalog") != "catalog"
        or any(value is not None for value in registered_values)
    )


def resolve_compat_agent_profile(
    args: Any,
    *,
    requested_open_query_planner: str | None,
) -> dict[str, Any]:
    """Resolve and validate one explicitly requested paper compatibility profile."""

    if not compat_agent_profile_requested(
        args,
        requested_open_query_planner=requested_open_query_planner,
    ):
        raise CompatAgentProfileError(
            "paper compatibility resolution requires an explicit legacy profile"
        )
    planner = requested_open_query_planner or "catalog_step_v1"
    if planner == "claim_first_v1":
        planner = "plan_agent_v1"
    claim_first_mode = planner == "plan_agent_v1"
    proposal_mode = getattr(args, "proposal_mode", "catalog")
    planning_policy = getattr(
        args,
        "planning_policy",
        "dynamic_evidence_v1",
    )
    if proposal_mode != "catalog" and not getattr(args, "auto_route", False):
        raise CompatAgentProfileError(
            "--proposal-mode novel_first_round requires --auto-route"
        )
    if claim_first_mode and planning_policy != "dynamic_evidence_v1":
        raise CompatAgentProfileError(
            "plan_agent_v1 requires --planning-policy dynamic_evidence_v1"
        )
    if claim_first_mode and proposal_mode != "catalog":
        raise CompatAgentProfileError(
            "plan_agent_v1 resolves its semantic proposal after evidence; "
            "do not also select a predeclared --proposal-mode"
        )

    registered_values = (
        getattr(args, "evidence_manifest", None),
        getattr(args, "command_plan", None),
        getattr(args, "registered_route", None),
        getattr(args, "registered_strategy", None),
    )
    if any(value is not None for value in registered_values) and not all(
        value is not None for value in registered_values
    ):
        raise CompatAgentProfileError(
            "registered execution requires --evidence-manifest, --command-plan, "
            "--registered-route, and --registered-strategy together"
        )
    registered_strategy = getattr(args, "registered_strategy", None)
    if registered_strategy is not None and getattr(args, "auto_route", False):
        raise CompatAgentProfileError(
            "registered execution forbids live --auto-route"
        )
    if (
        registered_strategy is not None
        and getattr(args, "evaluation_id", None) is None
    ):
        raise CompatAgentProfileError(
            "registered execution requires an explicit --evaluation-id"
        )
    return {
        "schema_version": 1,
        "open_query_planner": planner,
        "claim_first_mode": claim_first_mode,
        "proposal_mode": proposal_mode,
        "planning_policy": planning_policy,
        "registered_strategy": registered_strategy,
    }


def resolve_task_specific_runtime_profile(
    args: Any,
    *,
    claim_first_mode: bool,
) -> dict[str, Any]:
    """Validate the retained BBH/ClickBell transport and select its backend."""

    task_name = str(getattr(args, "task_name", ""))
    task_profile = str(getattr(args, "task_profile", "official"))
    execution_backend_arg = getattr(args, "execution_backend", None)
    legacy_click_bell = task_profile == "position_lr"
    adaptive_click_bell = task_profile == "adaptive_properties"
    fixed_click_bell = task_profile == "fixed_suite"
    bounded_click_bell = bool(
        legacy_click_bell or adaptive_click_bell or fixed_click_bell
    )
    if bounded_click_bell and task_name != "click_bell":
        raise CompatAgentProfileError(
            "click_bell generated task profiles require --task-name click_bell"
        )
    if task_name == "beat_block_hammer" and task_profile != "official":
        raise CompatAgentProfileError(
            "beat_block_hammer does not use click_bell task profiles"
        )
    if task_name == "beat_block_hammer" and execution_backend_arg:
        raise CompatAgentProfileError(
            "--execution-backend currently applies to schema-backed official "
            "tasks; beat_block_hammer keeps its bounded generated-task flow"
        )
    if bounded_click_bell and execution_backend_arg not in {None, "act"}:
        raise CompatAgentProfileError(
            "click_bell generated profiles are ACT-only"
        )
    if (
        legacy_click_bell
        and getattr(args, "generated_rounds", 2) not in {1, 2}
    ):
        raise CompatAgentProfileError(
            "click_bell position_lr supports at most 2 rounds"
        )
    execution_backend = (
        "act"
        if claim_first_mode
        or task_name == "beat_block_hammer"
        or bounded_click_bell
        else (execution_backend_arg or "expert")
    )
    return {
        "schema_version": 1,
        "legacy_click_bell": legacy_click_bell,
        "adaptive_click_bell": adaptive_click_bell,
        "fixed_click_bell": fixed_click_bell,
        "bounded_click_bell": bounded_click_bell,
        "execution_backend": execution_backend,
    }


__all__ = [
    "CompatAgentProfileError",
    "compat_agent_profile_requested",
    "resolve_compat_agent_profile",
    "resolve_task_specific_runtime_profile",
]
