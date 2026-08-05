"""Stable public imports for the paper-aligned Plan Agent.

Validation and lineage ownership lives in :mod:`plan_agent_schema`; provider
prompting and bounded generation lives in :mod:`plan_agent_provider`.
Historical ClaimFirst names remain read-only aliases for persisted artifacts
and external callers.
"""

from __future__ import annotations

from .plan_agent_provider import PlanAgent
from .plan_agent_schema import (
    ClaimFirstPlanError,
    PlanAgentError,
    build_open_query_planning_lineage,
    open_query_input_digest,
    project_open_query_capabilities,
    validate_open_query_capabilities,
    validate_open_query_evidence,
    validate_open_query_plan_proposal,
    validate_open_query_proposal_lineage,
)


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
