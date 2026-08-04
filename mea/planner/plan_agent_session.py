"""Canonical multi-round Plan Agent session.

The session owns evidence observation, Proposal authoring and binding, and the
QueryContract stop-validation boundary. Simulator and policy execution remain
outside this module.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .claim_first import (
    ClaimFirstPlanError,
    validate_open_query_capabilities,
    validate_open_query_evidence,
    validate_open_query_plan_proposal,
    validate_open_query_proposal_lineage,
)
from .experiment_candidate import validate_experiment_candidate
from .open_world_session import _FrozenExecutionTransport
from .plan_agent_errors import ClaimFirstRuntimeError, PlanAgentSessionError
from .plan_agent_evidence import (
    _attach_planning_lineage,
    _current_planning_evidence,
    build_claim_first_evidence_record,
    render_query_answer,
)
from .query_contract import (
    assess_query_sufficiency,
    build_query_sufficiency_contract,
    extend_query_candidate_universe,
    infer_claim_type,
    validate_query_sufficiency_contract,
)
from .query_interpretation import (
    _adapter_retrieval_aspects,
    _failure_seeking_existential,
    _nonempty_text,
    _target_task_name,
    _template_aspect,
    build_dynamic_experiment_candidate,
    control_template_id,
    resolve_semantic_proposal,
)
from .semantic_coverage import SemanticCoverageError, validate_evaluation_intent

class PlanAgentSession:
    """Own one complete Plan Agent session.

    The public session owns both semantic decisions and the frozen execution
    transport.  Legacy semantic-only fixtures may still pass a catalog-shaped
    target, but production targets construct their execution transport here so
    callers cannot keep two independently evolving session states.
    """

    def __init__(
        self,
        user_query: str,
        target: Mapping[str, Any],
        *,
        query_contract: Mapping[str, Any] | None = None,
        candidate_aspect_ids: Sequence[str] | None = None,
        require_control_anchor: bool | None = None,
        retrieval_aspects: Sequence[Mapping[str, Any]] | None = None,
        control_round: Mapping[str, Any] | None = None,
    ):
        self.user_query = _nonempty_text(user_query, "user_query")
        self.target = deepcopy(dict(target))
        self.task_name = _target_task_name(self.target)
        if retrieval_aspects is None:
            raw_retrieval_aspects = self.target.get("aspects")
            if not isinstance(raw_retrieval_aspects, list):
                raw_retrieval_aspects = _adapter_retrieval_aspects(
                    self.task_name
                )
        else:
            raw_retrieval_aspects = list(retrieval_aspects)
        if any(
            not isinstance(aspect, Mapping)
            for aspect in raw_retrieval_aspects
        ):
            raise ClaimFirstRuntimeError(
                "retrieval_aspects must contain only artifact-hint objects"
            )
        if (
            not raw_retrieval_aspects
            and "policy_task_binding" not in self.target
        ):
            raise ClaimFirstRuntimeError(
                "legacy retrieval_aspects must contain artifact hints"
            )
        self.retrieval_aspects = [
            deepcopy(dict(aspect)) for aspect in raw_retrieval_aspects
        ]
        if require_control_anchor is not None and not isinstance(
            require_control_anchor, bool
        ):
            raise ClaimFirstRuntimeError(
                "require_control_anchor must be bool or None"
            )
        self.control_template = control_template_id(self.target)
        if candidate_aspect_ids is not None:
            allowed_aspects = {
                _nonempty_text(item, "candidate_aspect_ids[]")
                for item in candidate_aspect_ids
            }
            known_aspects = {
                str(aspect.get("aspect_id") or "")
                for aspect in self.retrieval_aspects
                if isinstance(aspect, Mapping)
            }
            unknown_aspects = allowed_aspects - known_aspects
            if unknown_aspects:
                raise ClaimFirstRuntimeError(
                    "routed candidate aspects leave the bound task catalog: "
                    f"{sorted(unknown_aspects)}"
                )
            self.retrieval_aspects = [
                deepcopy(dict(aspect))
                for aspect in self.retrieval_aspects
                if isinstance(aspect, Mapping)
                and (
                    str(aspect.get("aspect_id") or "") in allowed_aspects
                    or self.control_template
                    in {str(item) for item in aspect.get("template_ids", [])}
                )
            ]
        retrieval_target = {"aspects": self.retrieval_aspects}
        self.template_to_aspect = _template_aspect(retrieval_target)
        candidates = [
            template_id
            for template_id in self.template_to_aspect
            if template_id != self.control_template
        ]
        supplied_contract = (
            validate_query_sufficiency_contract(query_contract)
            if query_contract is not None
            else None
        )
        contract_requires_control = (
            supplied_contract["control_requirement"] == "required"
            if supplied_contract is not None
            else (
                True
                if require_control_anchor is None
                else require_control_anchor
            )
        )
        if (
            supplied_contract is not None
            and require_control_anchor is not None
            and require_control_anchor != contract_requires_control
        ):
            raise ClaimFirstRuntimeError(
                "require_control_anchor conflicts with QueryContract "
                "control_requirement"
            )
        self.require_control_anchor = contract_requires_control
        round_budget = int(self.target.get("max_rounds") or 0) - (
            1 if self.require_control_anchor else 0
        )
        # Legacy official-only adapters use max_rounds=1 because their catalog
        # contains only the control.  That catalog size must not prohibit one
        # Query-derived generation round.
        if not candidates and round_budget < 1:
            round_budget = 1
        if round_budget < 1:
            raise ClaimFirstRuntimeError(
                "Plan Agent runtime needs budget for at least one candidate round"
            )
        if query_contract is None:
            claim_type = infer_claim_type(self.user_query)
            if claim_type == "comparative":
                raise ClaimFirstRuntimeError(
                    "comparative Query requires an explicit preregistered "
                    "query-sufficiency contract with two groups"
                )
            failure_witness = bool(
                claim_type == "existential"
                and _failure_seeking_existential(self.user_query)
            )
            contract = build_query_sufficiency_contract(
                self.user_query,
                candidate_universe=candidates,
                round_budget=round_budget,
                claim_type=claim_type,
                # Bound-task templates are retrieval seeds, never proof that an
                # open Query's semantic candidate universe is exhaustive.
                candidate_universe_closed=False,
                existential_witness_outcome=(
                    "fail" if failure_witness else None
                ),
                control_requirement=(
                    "required"
                    if self.require_control_anchor
                    else "not_required"
                ),
            )
        else:
            assert supplied_contract is not None
            contract = supplied_contract
            # The task catalog is a retrieval index, not an authorization
            # boundary. A Query-derived candidate may therefore be absent from
            # the registered template inventory; its executable Task/Tool needs
            # are validated later by the materialization and evidence gates.
            if int(contract["round_budget"]) > round_budget:
                raise ClaimFirstRuntimeError(
                    "query contract spends rounds reserved for the control anchor"
                )
        self.query_contract = contract
        self.dynamic_candidates: dict[str, dict[str, Any]] = {}
        self._frozen_execution = (
            _FrozenExecutionTransport.from_target(
                self.target,
                control_round=control_round,
                query_contract=self.query_contract,
            )
            if "policy_task_binding" in self.target
            else None
        )
        if self._frozen_execution is not None:
            self.target = deepcopy(self._frozen_execution.target)

    def _require_frozen_execution(self) -> _FrozenExecutionTransport:
        transport = self._frozen_execution
        if transport is None:
            raise ClaimFirstRuntimeError(
                "this legacy Plan Agent fixture has no frozen execution binding"
            )
        return transport

    @property
    def execution_binding(self) -> dict[str, Any]:
        """Return the frozen policy/task binding owned by this session."""

        return deepcopy(self._require_frozen_execution().binding)

    def normalize_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize the executable plan through this session's transport."""

        return self._require_frozen_execution().normalize_plan(plan)

    def planning_context(self, repo_root: str | Path) -> dict[str, Any]:
        """Project trusted runtime capabilities for retrieval and generation."""

        return self._require_frozen_execution().planning_context(repo_root)

    def apply_plan_step(
        self,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]],
        proposal: Mapping[str, Any],
        *,
        materialized_round: Mapping[str, Any] | None = None,
        source: str = "provider",
        query_contract: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Apply one semantic decision to the frozen executable plan."""

        contract = (
            query_contract
            if query_contract is not None
            else self.query_contract
        )
        result = self._require_frozen_execution().apply_plan_step(
            plan,
            observation_history,
            proposal,
            materialized_round=materialized_round,
            source=source,
            query_contract=contract,
        )
        normalized_contract = result[0].get("query_contract")
        if isinstance(normalized_contract, Mapping):
            self.query_contract = deepcopy(dict(normalized_contract))
        return result

    def snapshot(
        self,
        user_query: str,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return one combined semantic/execution session snapshot."""

        return self._require_frozen_execution().snapshot(
            user_query,
            plan,
            observation_history,
        )

    def _register_dynamic_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        require_direct: bool,
    ) -> dict[str, Any]:
        trusted = validate_experiment_candidate(candidate)
        expected_task = _nonempty_text(
            self.task_name,
            "target.task_name",
        )
        if trusted["base_task"] != expected_task:
            raise ClaimFirstRuntimeError(
                "frozen candidate base_task differs from the bound policy task"
            )
        if (
            require_direct
            and trusted.get("intent_alignment", {}).get("relationship")
            != "direct"
        ):
            raise ClaimFirstRuntimeError(
                "frozen candidate must directly implement its EvaluationIntent"
            )
        candidate_id = trusted["candidate_id"]
        existing = self.dynamic_candidates.get(candidate_id)
        if existing is not None and existing != trusted:
            raise ClaimFirstRuntimeError(
                f"dynamic candidate id collision: {candidate_id}"
            )
        self.dynamic_candidates[candidate_id] = trusted
        if candidate_id not in self.query_contract["candidate_universe"]:
            # A genuinely new evidence-conditioned Proposal reopens the
            # candidate universe.  A Query-frozen candidate that is already
            # present in a closed contract must not silently discard that
            # caller-authored finite scope merely by being registered for
            # execution.
            self.query_contract = extend_query_candidate_universe(
                self.query_contract,
                [candidate_id],
                candidate_universe_closed=False,
            )
        return deepcopy(trusted)

    def register_frozen_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Register a Query-only first candidate for a no-control session.

        A control-required session must leave its first semantic sub-aspect
        unfrozen until the control Aggregate has been observed.  Otherwise a
        Query-interpreter candidate authored before execution could be relabelled
        as Fig. 5 evidence-conditioned planning after the control merely passes.
        """

        if self.require_control_anchor:
            raise ClaimFirstRuntimeError(
                "control-required Plan Agent cannot freeze a pre-evidence "
                "candidate; author the next sub-aspect from observed control "
                "evidence"
            )

        return self._register_dynamic_candidate(
            candidate,
            require_direct=True,
        )

    def _bind_dynamic_candidate(
        self,
        *,
        proposal_bundle: Mapping[str, Any],
        proposal: Mapping[str, Any],
        candidate: Mapping[str, Any],
        executed_candidate_ids: Sequence[str],
        resolution: str,
        catalog_resolution_error: str | None,
        retrieval_hint: Mapping[str, Any] | None = None,
        require_direct: bool = False,
    ) -> dict[str, Any]:
        trusted = self._register_dynamic_candidate(
            candidate,
            require_direct=require_direct,
        )
        candidate_id = trusted["candidate_id"]
        if candidate_id in {str(item) for item in executed_candidate_ids}:
            raise ClaimFirstRuntimeError(
                f"dynamic candidate was already executed: {candidate_id}"
            )
        hint = dict(retrieval_hint or {})
        dynamic_resolution = {
            "schema_version": 1,
            "semantic_sub_aspect": proposal["sub_aspect"],
            "resolved_aspect_id": hint.get("resolved_aspect_id"),
            "resolved_template_id": None,
            "resolved_candidate_id": candidate_id,
            "resolution": resolution,
            "hidden": bool(hint.get("hidden", False)),
            "matched_tokens": list(hint.get("matched_tokens") or []),
            "catalog_was_model_visible": False,
            "catalog_resolution_error": catalog_resolution_error,
            "retrieval_aspect_id": hint.get("resolved_aspect_id"),
            "retrieval_template_id": hint.get("resolved_template_id"),
            "retrieval_resolution": hint.get("resolution"),
        }
        return {
            "schema_version": 2,
            "semantic_proposal_bundle": deepcopy(dict(proposal_bundle)),
            "semantic_needs": {
                "scene_need": {
                    "required": trusted["scene_need"] is not None,
                    "description": (
                        trusted["scene_need"]["description"]
                        if trusted["scene_need"] is not None
                        else None
                    ),
                },
                "checker_need": {
                    "required": trusted["checker_need"] is not None,
                    "description": (
                        trusted["checker_need"]["description"]
                        if trusted["checker_need"] is not None
                        else None
                    ),
                },
                "rule_tool_need": {
                    "required": trusted["rule_tool_need"] is not None,
                    "description": (
                        trusted["rule_tool_need"]["description"]
                        if trusted["rule_tool_need"] is not None
                        else None
                    ),
                    "reuse_first": True,
                },
                "vqa_tool_need": {
                    "required": trusted["vqa_tool_need"] is not None,
                    "description": (
                        trusted["vqa_tool_need"]["description"]
                        if trusted["vqa_tool_need"] is not None
                        else None
                    ),
                    "reuse_first": True,
                },
                "task_need": deepcopy(proposal["task_need"]),
                "tool_need": deepcopy(proposal["tool_need"]),
            },
            "resolution": dynamic_resolution,
            "query_contract": deepcopy(self.query_contract),
            "plan_step": {
                "schema_version": 2,
                "action": "propose",
                "aspect_id": proposal["sub_aspect"],
                "candidate_id": candidate_id,
                "execution_mode": "reuse_or_generate",
                "proposal": trusted,
                "rationale": proposal["rationale"],
                "answered_query": False,
            },
        }

    def bind_frozen_candidate(
        self,
        proposal_bundle: Mapping[str, Any],
        candidate: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        executed_candidate_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Authorize a no-control Query-only candidate without claiming refinement.

        Its semantic choice predates rollout evidence.  The returned lineage
        therefore remains ``pre_evidence`` and must not be counted as Fig. 5
        evidence-conditioned sub-aspect selection.
        """

        if self.require_control_anchor:
            raise ClaimFirstRuntimeError(
                "control-required Plan Agent cannot bind a pre-evidence "
                "candidate; author the next sub-aspect from observed control "
                "evidence"
            )
        assessment = observation.get("assessment")
        if not isinstance(assessment, Mapping):
            raise ClaimFirstRuntimeError(
                "Plan Agent observation has no assessment"
            )
        if assessment.get("should_stop"):
            raise ClaimFirstRuntimeError(
                "cannot bind a frozen candidate after the query contract stopped"
            )
        if (
            self.require_control_anchor
            and observation.get("control_passed") is not True
        ):
            raise ClaimFirstRuntimeError(
                "cannot execute a frozen property candidate before control passes"
            )
        raw_proposal = proposal_bundle.get("proposal")
        if not isinstance(raw_proposal, Mapping):
            raise ClaimFirstRuntimeError(
                "frozen proposal bundle has no proposal object"
            )
        proposal = validate_open_query_plan_proposal(
            raw_proposal,
            has_evidence=bool(observation.get("records")),
        )
        if proposal["action"] != "continue":
            raise ClaimFirstRuntimeError(
                "a frozen first candidate must be an action=continue proposal"
            )
        raw_lineage = proposal_bundle.get("planning_lineage")
        if not isinstance(raw_lineage, Mapping):
            raw_lineage = {
                "schema_version": 1,
                "decision_kind": "pre_evidence_query_candidate",
                "evidence_conditioned": False,
                "completed_round_ids": [],
                "completed_round_count": 0,
                "input_digest": None,
            }
        lineage = deepcopy(dict(raw_lineage))
        if (
            lineage.get("decision_kind")
            != "pre_evidence_query_candidate"
            or lineage.get("evidence_conditioned") is not False
            or lineage.get("completed_round_ids") != []
            or lineage.get("completed_round_count") != 0
        ):
            raise ClaimFirstRuntimeError(
                "bind_frozen_candidate accepts only an explicitly pre-evidence "
                "Query candidate"
            )
        bound = self._bind_dynamic_candidate(
            proposal_bundle=proposal_bundle,
            proposal=proposal,
            candidate=candidate,
            executed_candidate_ids=executed_candidate_ids,
            resolution="pre_evidence_query_proposal",
            catalog_resolution_error=None,
            require_direct=True,
        )
        return _attach_planning_lineage(bound, lineage)

    def observe(
        self,
        round_plans: Sequence[Mapping[str, Any]],
        round_summaries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Normalize all completed rounds and decide whether execution stops."""

        if len(round_plans) != len(round_summaries):
            raise ClaimFirstRuntimeError(
                "completed plans and summaries must be aligned"
            )
        if self.require_control_anchor and not round_plans:
            raise ClaimFirstRuntimeError(
                "control-first observation requires one completed control round"
            )
        records = [
            build_claim_first_evidence_record(plan, summary)
            for plan, summary in zip(round_plans, round_summaries)
        ]
        control_semantics: Mapping[str, Any] = {}
        control_authority_valid = True
        control_pipeline_valid = True
        baseline_valid = True
        if self.require_control_anchor:
            if records[0]["template_id"] != self.control_template:
                raise ClaimFirstRuntimeError(
                    "Plan Agent property attribution requires the control "
                    "template first"
                )
            control_packet = records[0]["evidence_packet"]
            control_outcome = records[0]["evaluation_outcome"]
            control_semantics = records[0].get("outcome_semantics") or {}
            control_authority_valid = bool(
                control_outcome.get("metric") == "official_check_success"
                and control_outcome.get("official_equivalent") is not False
                and control_semantics.get("status") != "conflict"
            )
            control_pipeline_valid = bool(
                control_packet["pipeline"]["passed"]
                and control_packet["policy"]["reported"]
                and control_packet["policy"]["success_rate"] is not None
            )
            baseline_valid = bool(
                control_authority_valid
                and control_pipeline_valid
                and float(control_packet["policy"]["success_rate"]) >= 1.0
            )
        candidate_records = (
            records[1:] if self.require_control_anchor else records
        )
        outside_candidate_ids = [
            record["candidate_id"]
            for record in candidate_records
            if record["candidate_id"]
            not in self.query_contract["candidate_universe"]
        ]
        if outside_candidate_ids:
            raise ClaimFirstRuntimeError(
                "completed candidate evidence is outside the active "
                "QueryContract universe: "
                f"{list(dict.fromkeys(outside_candidate_ids))}"
            )
        policy_candidate_records = [
            record
            for record in candidate_records
            if record.get("planning_observation") is None
        ]
        candidate_evidence = [
            deepcopy(record["candidate_evidence"])
            for record in policy_candidate_records
        ]
        assessment = assess_query_sufficiency(
            self.query_contract,
            candidate_evidence,
            completed_rounds=len(policy_candidate_records),
        )
        transport_conflict_ids = [
            record["candidate_id"]
            for record in candidate_records
            if (
                record["candidate_id"]
                in self.query_contract["candidate_universe"]
                and (
                    record.get("candidate_evidence", {}).get("outcome")
                    == "conflict"
                    or record.get("evidence_packet", {}).get(
                        "evidence_strength"
                    )
                    == "conflicting"
                )
            )
        ]
        if transport_conflict_ids:
            assessment = {
                **assessment,
                "should_stop": True,
                "stop_reason": "evidence_conflict",
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": (
                    "Rule, VQA, or execution evidence conflicts for a "
                    "completed candidate; the Query cannot be answered from "
                    "this evidence."
                ),
                "conflict_candidate_ids": list(
                    dict.fromkeys(
                        list(assessment.get("conflict_candidate_ids") or [])
                        + transport_conflict_ids
                    )
                ),
                "recommended_candidate_ids": [],
            }
        semantic_conflict_ids = [
            record["candidate_id"]
            for record in candidate_records
            if (
                record["candidate_id"]
                in self.query_contract["candidate_universe"]
                and record.get("outcome_semantics", {}).get("status")
                == "conflict"
            )
        ]
        if semantic_conflict_ids:
            assessment = {
                **assessment,
                "should_stop": True,
                "stop_reason": "outcome_semantics_conflict",
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": (
                    "Generated and official/core success semantics disagree for "
                    "a completed candidate; the Query cannot be answered from "
                    "this evidence."
                ),
                "conflict_candidate_ids": list(
                    dict.fromkeys(
                        list(assessment.get("conflict_candidate_ids") or [])
                        + semantic_conflict_ids
                    )
                ),
                "recommended_candidate_ids": [],
            }
        semantic_non_comparable_ids = [
            record["candidate_id"]
            for record in candidate_records
            if (
                record["candidate_id"]
                in self.query_contract["candidate_universe"]
                and record.get("evaluation_outcome", {}).get("metric")
                == "generated_check_success"
                and record.get("outcome_semantics", {}).get("status")
                == "non_comparable"
            )
        ]
        if semantic_non_comparable_ids:
            assessment = {
                **assessment,
                "should_stop": True,
                "stop_reason": "outcome_semantics_non_comparable",
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": (
                    "The generated checker lacks a comparable official/core "
                    "projection, so this Query cannot be answered from the "
                    "completed evidence."
                ),
                "non_comparable_candidate_ids": list(
                    dict.fromkeys(semantic_non_comparable_ids)
                ),
                "recommended_candidate_ids": [],
            }
        if self.require_control_anchor and not baseline_valid:
            reason = (
                "control_baseline_semantics_conflict"
                if control_semantics.get("status") == "conflict"
                else
                "control_baseline_non_official_outcome"
                if not control_authority_valid
                else
                "control_baseline_pipeline_invalid"
                if not control_pipeline_valid
                else "control_baseline_policy_failed"
            )
            assessment = {
                **assessment,
                "should_stop": True,
                "stop_reason": reason,
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": (
                    "The unchanged-scene control must pass before property "
                    "attribution; no candidate experiment is authorized."
                ),
                "recommended_candidate_ids": [],
            }
        answer = (
            render_query_answer(
                self.user_query,
                assessment,
                records,
                baseline_valid=baseline_valid,
                baseline_stop_reason=assessment["stop_reason"],
            )
            if assessment["should_stop"]
            else None
        )
        return {
            "schema_version": 1,
            "control_template_id": self.control_template,
            "control_required": self.require_control_anchor,
            "control_passed": (
                baseline_valid if self.require_control_anchor else None
            ),
            "query_contract": deepcopy(self.query_contract),
            "assessment": assessment,
            "records": records,
            "open_query_evidence_history": validate_open_query_evidence(
                [record["open_query_evidence"] for record in records]
            ),
            "query_answer": answer,
        }

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
        if (
            assessment.get("should_stop")
            and assessment.get("evidence_sufficient") is not True
        ):
            raise ClaimFirstRuntimeError(
                "cannot propose a semantic step after the query contract stopped"
            )
        if (
            self.require_control_anchor
            and observation.get("control_passed") is not True
        ):
            raise ClaimFirstRuntimeError(
                "cannot propose a property experiment before the control passes"
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
        try:
            proposal_bundle = propose(
                self.user_query,
                capabilities=trusted_capabilities,
                evidence_history=history,
                evaluation_intent=trusted_intent,
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
            if not (
                assessment.get("should_stop") is True
                and assessment.get("evidence_sufficient") is True
                and assessment.get("stop_reason") == "evidence_sufficient"
                and isinstance(query_answer, Mapping)
                and query_answer.get("answered") is True
            ):
                raise ClaimFirstRuntimeError(
                    "Plan Agent stop rejected by QueryContract: completed "
                    "evidence does not yet support an answer to the original "
                    "Query"
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
                    "resolution": "query_contract_validated_stop",
                    "hidden": False,
                    "matched_tokens": [],
                    "catalog_was_model_visible": False,
                    "catalog_resolution_error": None,
                    "retrieval_aspect_id": None,
                    "retrieval_template_id": None,
                    "retrieval_resolution": None,
                },
                "query_contract": deepcopy(self.query_contract),
                "query_assessment": deepcopy(dict(assessment)),
                "query_answer": deepcopy(dict(query_answer)),
                "plan_step": {
                    "schema_version": 2,
                    "action": "stop",
                    "aspect_id": None,
                    "candidate_id": None,
                    "execution_mode": "none",
                    "proposal": None,
                    "rationale": proposal["rationale"],
                    "hypothesis": proposal["hypothesis"],
                    "answered_query": True,
                    "claim_verdict": assessment.get("claim_verdict"),
                    "stop_reason": assessment.get("stop_reason"),
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

# Compatibility aliases retain object identity for historical callers.
ClaimFirstRuntimeController = PlanAgentSession


__all__ = [
    "PlanAgentSession",
    "PlanAgentSessionError",
    "ClaimFirstRuntimeController",
    "ClaimFirstRuntimeError",
]
