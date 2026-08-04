"""Outer planning API with legacy planners loaded only on explicit use."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .context import (
    PlanningContextError,
    build_planning_context,
    validate_planning_context,
)
from .evidence_policy import (
    EvidencePacketError,
    assess_conditional_transition,
    assess_evidence,
    build_evidence_aggregate,
    build_evidence_packet,
    validate_evidence_aggregate,
    validate_evidence_packet,
)
from .prototype import (
    BLUE_TASK_INSTRUCTION,
    MAX_ROUNDS,
    POSITION_TASK_INSTRUCTION,
    SCALE_TASK_INSTRUCTION,
    SAFETY_TASK_INSTRUCTION,
    SUB_ASPECT_CATALOG,
    TIMING_TASK_INSTRUCTION,
    PlanAgentError,
    PlanAgentPrototype,
    make_evaluation_id,
    validate_evaluation_plan,
    validate_next_round_decision,
)
from .catalog import (
    ACTCatalogError,
    ACT_ROUTE_TASKS,
    build_act_catalog,
    catalog_task,
    validate_act_catalog,
)
from .global_query import (
    GlobalQueryRouter,
    GlobalRouteError,
    build_global_route_prompt,
    route_to_bbh_proposal,
    route_to_click_proposal,
    route_to_official_proposal,
    route_to_planner_proposal,
    validate_route_selection,
)
from .session import (
    BoundTaskPlanSession,
    PlanSessionError,
    build_adaptive_directive,
    build_evaluation_target,
    validate_adaptive_choice,
    validate_evaluation_target,
)
from .adaptive_step import (
    AdaptivePlanStepAgent,
    AdaptiveStepError,
    validate_plan_step_proposal,
)
from .query_contract import (
    CLAIM_TYPES,
    OUTCOMES,
    QuerySufficiencyError,
    assess_query_sufficiency,
    build_query_sufficiency_contract,
    extend_query_candidate_universe,
    infer_claim_type,
    validate_query_sufficiency_contract,
)
from .experiment_candidate import (
    ExperimentCandidateError,
    build_experiment_candidate,
    validate_experiment_candidate,
)
from .semantic_coverage import (
    SemanticCoverageError,
    advance_implementation_trace_with_tool,
    build_candidate_intent_alignment,
    build_evaluation_intent,
    build_implementation_trace,
    evaluation_intent_from_query_interpretation,
    evaluation_intent_from_free_concern,
    validate_evaluation_intent,
    validate_implementation_trace,
    validate_intent_alignment,
)
from .claim_first import (
    PlanAgent,
    PlanAgentError as OpenQueryPlanAgentError,
    ClaimFirstOpenQueryAgent,
    ClaimFirstPlanError,
    build_open_query_planning_lineage,
    open_query_input_digest,
    project_open_query_capabilities,
    validate_open_query_capabilities,
    validate_open_query_evidence,
    validate_open_query_plan_proposal,
    validate_open_query_proposal_lineage,
)
from .plan_agent_errors import (
    PlanAgentSessionError,
    ClaimFirstRuntimeError,
)
from .plan_agent_evidence import (
    build_claim_first_evidence_record,
    render_query_answer,
)
from .plan_agent_session import (
    PlanAgentSession,
    ClaimFirstRuntimeController,
)
from .query_interpretation import (
    build_dynamic_experiment_candidate,
    build_initial_semantic_proposal_bundle,
    control_template_id,
    resolve_concern_candidate_domain,
    resolve_semantic_proposal,
)
from .claim_first_initial import (
    PlanAgentInitialPlanBuilder,
    PlanAgentInitialPlanError,
    ClaimFirstInitialPlanBuilder,
    ClaimFirstInitialPlanError,
    build_plan_agent_control_round,
    build_plan_agent_execution_binding,
)
from .open_task_resolver import (
    PlanAgentQueryInterpreter,
    FreeConcernAgent,
    discover_robotwin_task_inventory,
    resolve_open_task,
)
from .open_world_session import (
    OpenWorldSessionError,
    build_open_world_evaluation_target,
    validate_open_world_evaluation_target,
)
from .policy_task_binding import (
    PolicyTaskBindingError,
    build_policy_task_binding,
    policy_task_binding_from_target,
    validate_policy_task_binding,
)


# These task-specific and catalog planners are compatibility/paper protocols.
# Preserve the public import API while keeping normal Plan Agent imports free of
# their modules and construction side effects.
_LEGACY_EXPORTS = {
    "OFFICIAL_GATES": (".official", "OFFICIAL_GATES"),
    "OFFICIAL_TEMPLATE_ID": (".official", "OFFICIAL_TEMPLATE_ID"),
    "OfficialTaskPlanAgent": (".official", "OfficialTaskPlanAgent"),
    "CLICK_BELL_ADAPTIVE_ASPECTS": (
        ".click_bell_catalog",
        "CLICK_BELL_ADAPTIVE_ASPECTS",
    ),
    "CLICK_BELL_ADAPTIVE_TEMPLATES": (
        ".click_bell_catalog",
        "CLICK_BELL_ADAPTIVE_TEMPLATES",
    ),
    "CLICK_BELL_POSITIONS": (".click_bell_catalog", "CLICK_BELL_POSITIONS"),
    "CLICK_BELL_TEMPLATE_IDS": (".click_bell_catalog", "CLICK_BELL_TEMPLATE_IDS"),
    "ClickBellAdaptivePlanAgent": (".click_bell", "ClickBellAdaptivePlanAgent"),
    "ClickBellFixedSuitePlanAgent": (".click_bell", "ClickBellFixedSuitePlanAgent"),
    "ClickBellPositionPlanAgent": (".click_bell", "ClickBellPositionPlanAgent"),
    "CATALOG_PLAN_TASKS": (".catalog_plan", "CATALOG_PLAN_TASKS"),
    "CatalogPlanAgent": (".catalog_plan", "CatalogPlanAgent"),
    "CatalogPlanError": (".catalog_plan", "CatalogPlanError"),
    "PlanMaterializer": (".catalog_plan", "PlanMaterializer"),
}


def __getattr__(name: str) -> Any:
    """Resolve compatibility planner exports only when explicitly requested."""

    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LEGACY_EXPORTS))


__all__ = [
    "BLUE_TASK_INSTRUCTION",
    "MAX_ROUNDS",
    "POSITION_TASK_INSTRUCTION",
    "SCALE_TASK_INSTRUCTION",
    "SAFETY_TASK_INSTRUCTION",
    "SUB_ASPECT_CATALOG",
    "TIMING_TASK_INSTRUCTION",
    "assess_evidence",
    "assess_conditional_transition",
    "EvidencePacketError",
    "build_evidence_aggregate",
    "build_evidence_packet",
    "validate_evidence_aggregate",
    "validate_evidence_packet",
    "PlanningContextError",
    "build_planning_context",
    "validate_planning_context",
    "PlanAgentError",
    "PlanAgentPrototype",
    "make_evaluation_id",
    "validate_evaluation_plan",
    "validate_next_round_decision",
    "OFFICIAL_GATES",
    "OFFICIAL_TEMPLATE_ID",
    "OfficialTaskPlanAgent",
    "CLICK_BELL_POSITIONS",
    "CLICK_BELL_TEMPLATE_IDS",
    "CLICK_BELL_ADAPTIVE_ASPECTS",
    "CLICK_BELL_ADAPTIVE_TEMPLATES",
    "ClickBellAdaptivePlanAgent",
    "ClickBellFixedSuitePlanAgent",
    "ClickBellPositionPlanAgent",
    "ACTCatalogError",
    "ACT_ROUTE_TASKS",
    "build_act_catalog",
    "catalog_task",
    "validate_act_catalog",
    "CATALOG_PLAN_TASKS",
    "CatalogPlanAgent",
    "CatalogPlanError",
    "PlanMaterializer",
    "GlobalQueryRouter",
    "GlobalRouteError",
    "build_global_route_prompt",
    "route_to_bbh_proposal",
    "route_to_click_proposal",
    "route_to_official_proposal",
    "route_to_planner_proposal",
    "validate_route_selection",
    "BoundTaskPlanSession",
    "PlanSessionError",
    "build_adaptive_directive",
    "build_evaluation_target",
    "validate_adaptive_choice",
    "validate_evaluation_target",
    "AdaptivePlanStepAgent",
    "AdaptiveStepError",
    "validate_plan_step_proposal",
    "CLAIM_TYPES",
    "OUTCOMES",
    "QuerySufficiencyError",
    "assess_query_sufficiency",
    "build_query_sufficiency_contract",
    "extend_query_candidate_universe",
    "infer_claim_type",
    "validate_query_sufficiency_contract",
    "ExperimentCandidateError",
    "build_experiment_candidate",
    "validate_experiment_candidate",
    "SemanticCoverageError",
    "advance_implementation_trace_with_tool",
    "build_candidate_intent_alignment",
    "build_evaluation_intent",
    "build_implementation_trace",
    "evaluation_intent_from_free_concern",
    "evaluation_intent_from_query_interpretation",
    "validate_evaluation_intent",
    "validate_implementation_trace",
    "validate_intent_alignment",
    "PlanAgent",
    "OpenQueryPlanAgentError",
    "ClaimFirstOpenQueryAgent",
    "ClaimFirstPlanError",
    "build_open_query_planning_lineage",
    "open_query_input_digest",
    "project_open_query_capabilities",
    "validate_open_query_capabilities",
    "validate_open_query_evidence",
    "validate_open_query_plan_proposal",
    "validate_open_query_proposal_lineage",
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
    "PlanAgentInitialPlanBuilder",
    "PlanAgentInitialPlanError",
    "build_plan_agent_control_round",
    "build_plan_agent_execution_binding",
    "ClaimFirstInitialPlanBuilder",
    "ClaimFirstInitialPlanError",
    "FreeConcernAgent",
    "PlanAgentQueryInterpreter",
    "discover_robotwin_task_inventory",
    "resolve_open_task",
    "OpenWorldSessionError",
    "build_open_world_evaluation_target",
    "validate_open_world_evaluation_target",
    "PolicyTaskBindingError",
    "build_policy_task_binding",
    "policy_task_binding_from_target",
    "validate_policy_task_binding",
]
