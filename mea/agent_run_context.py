"""Typed state shared by the thin Agent CLI orchestration phases.

The public command is intentionally a dispatcher.  Each phase below owns a
real method responsibility while this object only carries the artifacts that
must cross phase boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentRunContext:
    """Mutable lifecycle state for one Agent command invocation."""

    args: Any
    bound_plan_only: bool
    repo_root: Path
    runtime_policy_spec: Any | None
    plan_runtime_limits: dict[str, Any] | None
    models: dict[str, str]
    history_path: Path

    provider: Any | None = None
    runtime_plan_agent_targets: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    runtime_binding_excluded: list[dict[str, str]] = field(default_factory=list)
    global_route_result: dict[str, Any] | None = None
    global_history_retrieval: dict[str, Any] = field(default_factory=dict)
    query_interpreter: Any | None = None
    query_interpretation_bundle: dict[str, Any] | None = None
    concern_candidate_resolution: dict[str, Any] | None = None
    open_task_inventory: list[dict[str, Any]] | None = None
    open_task_resolution: dict[str, Any] | None = None
    execution_backend: str = "act"
    history_database: Any | None = None
    history_context: list[dict[str, Any]] = field(default_factory=list)
    history_retrieval: dict[str, Any] = field(default_factory=dict)
    planner_kwargs: dict[str, Any] = field(default_factory=dict)

    initial_target: dict[str, Any] | None = None
    round_budget: int | None = None
    control_required: bool = True
    evaluation_intent: dict[str, Any] | None = None
    initial_semantic_bundle: dict[str, Any] | None = None
    frozen_first_candidate: dict[str, Any] | None = None
    initial_open_candidate: dict[str, Any] | None = None
    direct_single_candidate_query: bool = False

    manifest: dict[str, Any] | None = None
    evaluation_id: str | None = None
    evaluation_dir: Path | None = None
    plan: dict[str, Any] | None = None
    plan_session: Any | None = None
    plan_session_path: str | None = None
    evaluation_target: dict[str, Any] | None = None
    planning_context: dict[str, Any] | None = None
    plan_agent: Any | None = None
    plan_agent_capabilities: dict[str, Any] | None = None


__all__ = ["AgentRunContext"]
