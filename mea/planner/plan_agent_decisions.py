"""Evidence-conditioned Proposal authoring and binding for Plan Agent sessions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .plan_agent_schema import (
    ClaimFirstPlanError,
    validate_open_query_capabilities,
    validate_open_query_plan_proposal,
    validate_open_query_proposal_lineage,
)
from .plan_agent_errors import ClaimFirstRuntimeError
from .plan_agent_evidence import (
    _attach_planning_lineage,
    _current_planning_evidence,
    render_query_answer,
)
from .query_contract import (
    QuerySufficiencyError,
    project_agent_inconclusive_stop,
)
from .query_interpretation import (
    build_dynamic_experiment_candidate,
    resolve_semantic_proposal,
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
        """Bind a proposal only if it consumed the current evidence history.

        This is the auditable boundary for cached/provider proposals.  A bundle
        authored before the latest completed round fails its input digest or
        completed-round lineage check instead of being released as the next
        sub-aspect.
        """

        try:
            trusted_capabilities = validate_open_query_capabilities(
                capabilities
            )
            history = _current_planning_evidence(observation)
            if self.require_control_anchor and not history:
                raise ClaimFirstRuntimeError(
                    "control-required Plan Agent needs observed control evidence "
                    "before binding the next sub-aspect"
                )
            trusted_intent = (
                validate_evaluation_intent(evaluation_intent)
                if evaluation_intent is not None
                else None
            )
            lineage = validate_open_query_proposal_lineage(
                proposal_bundle,
                user_query=self.user_query,
                capabilities=trusted_capabilities,
                evidence_history=history,
                evaluation_intent=trusted_intent,
            )
        except (ClaimFirstPlanError, SemanticCoverageError) as exc:
            raise ClaimFirstRuntimeError(str(exc)) from exc
        bound = self.bind_semantic_step(
            proposal_bundle,
            observation,
            executed_template_ids=executed_candidate_ids,
            evaluation_intent=trusted_intent,
        )
        return _attach_planning_lineage(bound, lineage)

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
            raise ClaimFirstRuntimeError(
                "Plan Agent observation has no assessment"
            )
        history = _current_planning_evidence(observation)
        if self.require_control_anchor and not history:
            raise ClaimFirstRuntimeError(
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
        except (ClaimFirstPlanError, SemanticCoverageError) as exc:
            raise ClaimFirstRuntimeError(str(exc)) from exc
        propose = getattr(planner, "propose", None)
        if not callable(propose):
            raise ClaimFirstRuntimeError(
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
            if isinstance(exc, ClaimFirstRuntimeError):
                raise
            raise ClaimFirstRuntimeError(
                "evidence-conditioned Plan Agent failed after completed "
                f"rounds {[item['round_id'] for item in history]}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(proposal_bundle, Mapping):
            raise ClaimFirstRuntimeError(
                "Plan Agent returned no proposal bundle"
            )
        try:
            validate_open_query_proposal_lineage(
                proposal_bundle,
                user_query=self.user_query,
                capabilities=trusted_capabilities,
                evidence_history=history,
                evaluation_intent=trusted_intent,
            )
        except ClaimFirstPlanError as exc:
            raise ClaimFirstRuntimeError(str(exc)) from exc
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
            raise ClaimFirstRuntimeError("Plan Agent observation has no assessment")
        raw_proposal = proposal_bundle.get("proposal")
        if not isinstance(raw_proposal, Mapping):
            raise ClaimFirstRuntimeError(
                "Plan Agent proposal bundle has no proposal object"
            )
        try:
            proposal = validate_open_query_plan_proposal(
                raw_proposal,
                has_evidence=bool(observation.get("records")),
            )
        except ClaimFirstPlanError as exc:
            raise ClaimFirstRuntimeError(str(exc)) from exc

        if proposal["action"] == "stop":
            query_answer = observation.get("query_answer")
            sufficient_stop = bool(
                assessment.get("should_stop") is True
                and assessment.get("evidence_sufficient") is True
                and assessment.get("stop_reason") == "evidence_sufficient"
                and isinstance(query_answer, Mapping)
                and query_answer.get("answered") is True
            )
            if sufficient_stop:
                stop_assessment = deepcopy(dict(assessment))
                stop_answer = deepcopy(dict(query_answer))
                resolution = "query_contract_validated_stop"
                answered_query = True
            else:
                try:
                    stop_assessment = project_agent_inconclusive_stop(
                        assessment,
                        rationale=proposal["rationale"],
                    )
                except QuerySufficiencyError as exc:
                    raise ClaimFirstRuntimeError(str(exc)) from exc
                stop_answer = render_query_answer(
                    self.user_query,
                    stop_assessment,
                    observation.get("records") or [],
                    baseline_valid=observation.get("control_passed") is not False,
                )
                resolution = "plan_agent_inconclusive_stop"
                answered_query = False
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
                "query_contract": deepcopy(self.query_contract),
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
                    "claim_verdict": stop_assessment.get("claim_verdict"),
                    "stop_reason": stop_assessment.get("stop_reason"),
                    "next_round": None,
                },
            }

        budget_remaining = assessment.get("budget_remaining")
        may_continue_after_sufficiency = bool(
            assessment.get("should_stop") is True
            and assessment.get("evidence_sufficient") is True
            and assessment.get("stop_reason") == "evidence_sufficient"
            and isinstance(budget_remaining, int)
            and not isinstance(budget_remaining, bool)
            and budget_remaining > 0
        )
        if assessment.get("should_stop") and not may_continue_after_sufficiency:
            raise ClaimFirstRuntimeError(
                "cannot bind a semantic step after the query contract stopped"
            )
        if (
            self.require_control_anchor
            and observation.get("control_passed") is not True
        ):
            raise ClaimFirstRuntimeError(
                "cannot attribute a property before the control passes"
            )
        retrieval_hint: dict[str, Any] | None = None
        retrieval_error: str | None = None
        try:
            retrieval_hint = resolve_semantic_proposal(
                proposal,
                target={"aspects": self.retrieval_aspects},
                executed_template_ids=executed_template_ids,
                control_template=self.control_template,
            )
        except ClaimFirstRuntimeError as catalog_error:
            retrieval_error = str(catalog_error)
        candidate = build_dynamic_experiment_candidate(
            user_query=self.user_query,
            task_name=self.task_name,
            proposal=proposal,
            evaluation_intent=evaluation_intent,
        )
        return self._bind_dynamic_candidate(
            proposal_bundle=proposal_bundle,
            proposal=proposal,
            candidate=candidate,
            executed_candidate_ids=executed_template_ids,
            resolution=(
                "retrieval_hint_then_reuse_or_generate"
                if retrieval_hint is not None
                else "proposal_reuse_or_generate"
            ),
            catalog_resolution_error=retrieval_error,
            retrieval_hint=retrieval_hint,
        )



__all__ = ["PlanAgentDecisionMixin"]
