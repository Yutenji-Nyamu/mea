"""Evidence-conditioned Proposal authoring and binding for Plan Agent sessions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .plan_agent_schema import (
    PlanAgentError,
    validate_open_query_capabilities,
    validate_open_query_plan_proposal,
)
from .plan_agent_errors import PlanAgentSessionError
from .plan_agent_evidence import _current_planning_evidence, render_query_answer
from .runtime_limits import PlanRuntimeError, validate_agent_stop
from .query_interpretation import (
    build_dynamic_experiment_candidate,
)
from .semantic_coverage import SemanticCoverageError, validate_evaluation_intent


class PlanAgentDecisionMixin:
    def bind_evidence_conditioned_semantic_step(
        self,
        proposal_bundle: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        capabilities: Mapping[str, Any],
        executed_candidate_ids: Sequence[str],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Bind a proposal after validating the current evidence snapshot."""

        try:
            trusted_capabilities = validate_open_query_capabilities(
                capabilities
            )
            history = _current_planning_evidence(observation)
            if self.require_control_anchor and not history:
                raise PlanAgentSessionError(
                    "control-required Plan Agent needs observed control evidence "
                    "before binding the next sub-aspect"
                )
            trusted_intent = (
                validate_evaluation_intent(evaluation_intent)
                if evaluation_intent is not None
                else None
            )
        except (PlanAgentError, SemanticCoverageError) as exc:
            raise PlanAgentSessionError(str(exc)) from exc
        return self.bind_semantic_step(
            proposal_bundle,
            observation,
            executed_template_ids=executed_candidate_ids,
            evaluation_intent=trusted_intent,
        )

    def propose_semantic_step(
        self,
        planner: Any,
        observation: Mapping[str, Any],
        *,
        capabilities: Mapping[str, Any],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Read completed evidence, then author one validated next decision.

        Keeping the provider call inside the session makes the temporal order
        explicit: round evidence is validated first; only then may the Plan
        Agent choose stop or a next semantic concern.  Binding a continuing
        Proposal remains a separate step so an outer execution cap can reject
        it without pre-empting an evidence-backed stop decision.
        """

        assessment = observation.get("assessment")
        if not isinstance(assessment, Mapping):
            raise PlanAgentSessionError(
                "Plan Agent observation has no assessment"
            )
        history = _current_planning_evidence(observation)
        if self.require_control_anchor and not history:
            raise PlanAgentSessionError(
                "control-required Plan Agent needs observed control evidence "
                "before authoring the next sub-aspect"
            )
        try:
            trusted_capabilities = validate_open_query_capabilities(
                capabilities
            )
            trusted_intent = (
                validate_evaluation_intent(evaluation_intent)
                if evaluation_intent is not None
                else None
            )
        except (PlanAgentError, SemanticCoverageError) as exc:
            raise PlanAgentSessionError(str(exc)) from exc
        propose = getattr(planner, "propose", None)
        if not callable(propose):
            raise PlanAgentSessionError(
                "Plan Agent must expose a callable propose()"
            )
        # The Query contract validates a decision after the Agent authors it.
        # Before that call, expose only concrete runtime limits; forwarding the
        # contract's stop/verdict fields would let the validator pre-author the
        # very decision that the Plan Agent is meant to make from evidence.
        decision_context = {
            key: deepcopy(assessment[key])
            for key in ("budget_remaining", "limitations")
            if assessment.get(key) is not None
        }
        try:
            proposal_bundle = propose(
                self.user_query,
                capabilities=trusted_capabilities,
                evidence_history=history,
                evaluation_intent=trusted_intent,
                decision_context=decision_context,
            )
        except Exception as exc:
            if isinstance(exc, PlanAgentSessionError):
                raise
            raise PlanAgentSessionError(
                "evidence-conditioned Plan Agent failed after completed "
                f"rounds {[item['round_id'] for item in history]}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(proposal_bundle, Mapping):
            raise PlanAgentSessionError(
                "Plan Agent returned no proposal bundle"
            )
        return deepcopy(dict(proposal_bundle))

    def propose_and_bind_semantic_step(
        self,
        planner: Any,
        observation: Mapping[str, Any],
        *,
        capabilities: Mapping[str, Any],
        executed_candidate_ids: Sequence[str],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Author from completed evidence, then bind exactly one next step."""

        proposal_bundle = self.propose_semantic_step(
            planner,
            observation,
            capabilities=capabilities,
            evaluation_intent=evaluation_intent,
        )
        return self.bind_evidence_conditioned_semantic_step(
            proposal_bundle,
            observation,
            capabilities=capabilities,
            executed_candidate_ids=executed_candidate_ids,
            evaluation_intent=evaluation_intent,
        )

    def bind_semantic_step(
        self,
        proposal_bundle: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        executed_template_ids: Sequence[str],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate and bind one semantic Proposal to the generic runtime.

        Retrieval may identify an exact historical artifact, but it is only a
        reuse hint.  It never changes the Proposal into a catalog-authorized
        execution step.  Consequently every production Plan Agent round has
        the same typed Proposal boundary and the generic TaskGen/ToolGen
        materializer decides whether to reuse or generate each requested
        artifact.
        """

        assessment = observation.get("assessment")
        if not isinstance(assessment, Mapping):
            raise PlanAgentSessionError("Plan Agent observation has no assessment")
        raw_proposal = proposal_bundle.get("proposal")
        if not isinstance(raw_proposal, Mapping):
            raise PlanAgentSessionError(
                "Plan Agent proposal bundle has no proposal object"
            )
        try:
            proposal = validate_open_query_plan_proposal(
                raw_proposal,
                has_evidence=bool(observation.get("records")),
            )
        except PlanAgentError as exc:
            raise PlanAgentSessionError(str(exc)) from exc

        if proposal["action"] == "stop":
            baseline_valid = bool(
                not self.require_control_anchor
                or observation.get("control_passed") is True
            )
            if proposal["evidence_sufficient"] and not baseline_valid:
                raise PlanAgentSessionError(
                    "a supported or refuted answer requires a valid unchanged "
                    "official baseline"
                )
            try:
                stop_assessment = validate_agent_stop(
                    assessment,
                    rationale=proposal["rationale"],
                    answer=proposal["answer"],
                    claim_verdict=proposal["claim_verdict"],
                    evidence_sufficient=proposal["evidence_sufficient"],
                )
            except PlanRuntimeError as exc:
                raise PlanAgentSessionError(str(exc)) from exc
            stop_answer = render_query_answer(
                self.user_query,
                stop_assessment,
                observation.get("records") or [],
                baseline_valid=baseline_valid,
            )
            resolution = "plan_agent_stop"
            answered_query = bool(proposal["evidence_sufficient"])
            stop_answer["answered"] = answered_query
            stop_answer["stop_reason"] = "agent_stop"
            stop_answer["answer"] = (
                proposal["answer"]
                if answered_query
                else proposal["rationale"]
            )
            return {
                "schema_version": 2,
                "semantic_proposal_bundle": deepcopy(dict(proposal_bundle)),
                "semantic_needs": {
                    "scene_need": deepcopy(proposal["scene_need"]),
                    "checker_need": deepcopy(proposal["checker_need"]),
                    "rule_tool_need": deepcopy(proposal["rule_tool_need"]),
                    "vqa_tool_need": deepcopy(proposal["vqa_tool_need"]),
                    "task_need": deepcopy(proposal["task_need"]),
                    "tool_need": deepcopy(proposal["tool_need"]),
                },
                "resolution": {
                    "schema_version": 1,
                    "semantic_sub_aspect": None,
                    "resolved_aspect_id": None,
                    "resolved_template_id": None,
                    "resolved_candidate_id": None,
                    "resolution": resolution,
                    "hidden": False,
                    "matched_tokens": [],
                    "catalog_was_model_visible": False,
                    "catalog_resolution_error": None,
                    "retrieval_aspect_id": None,
                    "retrieval_template_id": None,
                    "retrieval_resolution": None,
                },
                "runtime_limits": deepcopy(self.runtime_limits),
                "query_assessment": stop_assessment,
                "query_answer": stop_answer,
                "plan_step": {
                    "schema_version": 2,
                    "action": "stop",
                    "aspect_id": None,
                    "candidate_id": None,
                    "execution_mode": "none",
                    "proposal": None,
                    "rationale": proposal["rationale"],
                    "hypothesis": proposal["hypothesis"],
                    "answered_query": answered_query,
                    "answer": stop_assessment.get("agent_answer"),
                    "claim_verdict": stop_assessment.get("claim_verdict"),
                    "stop_reason": stop_assessment.get("stop_reason"),
                    "next_round": None,
                },
            }

        if assessment.get("should_stop"):
            raise PlanAgentSessionError(
                "cannot continue after the external round cap"
            )
        candidate = build_dynamic_experiment_candidate(
            user_query=self.user_query,
            task_name=self.task_name,
            proposal=proposal,
            evaluation_intent=evaluation_intent,
            candidate_id=(
                f"dynamic.{self.task_name}.round_"
                f"{len(executed_template_ids) + 1}"
            ),
        )
        return self._bind_dynamic_candidate(
            proposal_bundle=proposal_bundle,
            proposal=proposal,
            candidate=candidate,
            executed_candidate_ids=executed_template_ids,
            resolution="proposal_reuse_or_generate",
            catalog_resolution_error=None,
            retrieval_hint=None,
        )


__all__ = ["PlanAgentDecisionMixin"]
