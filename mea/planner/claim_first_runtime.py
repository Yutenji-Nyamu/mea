"""Compatibility imports for the historical ClaimFirst runtime name.

New code should import the paper-aligned query_interpretation,
plan_agent_evidence, and plan_agent_session modules. Historical evidence and
callers remain readable through this module without rewriting stored artifacts.
"""

from __future__ import annotations

from .plan_agent_errors import ClaimFirstRuntimeError, PlanAgentSessionError
from .plan_agent_evidence import build_claim_first_evidence_record, render_query_answer
from .plan_agent_session import ClaimFirstRuntimeController, PlanAgentSession
from .query_interpretation import (
    build_dynamic_experiment_candidate,
    build_initial_semantic_proposal_bundle,
    control_template_id,
    resolve_concern_candidate_domain,
    resolve_semantic_proposal,
)


__all__ = [
    "PlanAgentSession",
    "PlanAgentSessionError",
    "ClaimFirstRuntimeController",
    "ClaimFirstRuntimeError",
    "build_claim_first_evidence_record",
    "build_dynamic_experiment_candidate",
    "build_initial_semantic_proposal_bundle",
    "control_template_id",
    "render_query_answer",
    "resolve_concern_candidate_domain",
    "resolve_semantic_proposal",
]
