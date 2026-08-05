"""Canonical multi-round Plan Agent session.

The session owns evidence observation, Proposal authoring and binding, and the
QueryContract stop-validation boundary. Simulator and policy execution remain
outside this module.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from mea.method_runtime import BackendTaskBinding

from .experiment_candidate import validate_experiment_candidate
from .open_world_session import _FrozenExecutionTransport
from .plan_agent_errors import ClaimFirstRuntimeError, PlanAgentSessionError
from .plan_agent_decisions import PlanAgentDecisionMixin
from .plan_agent_evidence_session import PlanAgentEvidenceMixin
from .plan_agent_evidence import (
    _attach_planning_lineage,
)
from .plan_agent_schema import validate_open_query_plan_proposal
from .query_contract import (
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
    control_template_id,
)


class PlanAgentSession(PlanAgentEvidenceMixin, PlanAgentDecisionMixin):
    """Own one complete Plan Agent session.

    The public session owns both semantic decisions and the frozen execution
    transport.  Legacy semantic-only fixtures may still pass a catalog-shaped
    target, but production targets construct their execution transport here so
    callers cannot keep two independently evolving session states.
    """

    def __init__(
        self,
        user_query: str,
        target: Mapping[str, Any] | None = None,
        *,
        method_binding: BackendTaskBinding | None = None,
        method_max_rounds: int | None = None,
        query_contract: Mapping[str, Any] | None = None,
        candidate_aspect_ids: Sequence[str] | None = None,
        require_control_anchor: bool | None = None,
        retrieval_aspects: Sequence[Mapping[str, Any]] | None = None,
        control_round: Mapping[str, Any] | None = None,
    ):
        self.user_query = _nonempty_text(user_query, "user_query")
        if method_binding is not None:
            if target is not None:
                raise ClaimFirstRuntimeError(
                    "target and method_binding are mutually exclusive"
                )
            if (
                isinstance(method_max_rounds, bool)
                or not isinstance(method_max_rounds, int)
                or method_max_rounds < 1
            ):
                raise ClaimFirstRuntimeError(
                    "method_max_rounds must be a positive integer"
                )
            raw_task_name = str(
                method_binding.metadata.get("task_name")
                or method_binding.binding_id
            ).casefold()
            method_task_name = re.sub(
                r"[^a-z0-9_]+", "_", raw_task_name
            ).strip("_")
            if not method_task_name or not method_task_name[0].isalpha():
                method_task_name = f"task_{method_task_name or 'bound'}"
            self.target = {
                "schema_version": 1,
                "task_name": method_task_name,
                "method_binding": method_binding.to_dict(),
                "max_rounds": method_max_rounds,
            }
            self.task_name = method_task_name
            raw_retrieval_aspects: list[Mapping[str, Any]] = []
        else:
            if target is None:
                raise ClaimFirstRuntimeError(
                    "target or method_binding is required"
                )
            self.target = deepcopy(dict(target))
            self.task_name = _target_task_name(self.target)
            raw_retrieval_aspects = []
        if method_binding is None and retrieval_aspects is None:
            raw_retrieval_aspects = self.target.get("aspects")
            if not isinstance(raw_retrieval_aspects, list):
                raw_retrieval_aspects = _adapter_retrieval_aspects(
                    self.task_name
                )
        elif retrieval_aspects is not None:
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
            and method_binding is None
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
        self.control_template = (
            "official_control"
            if method_binding is not None
            else control_template_id(self.target)
        )
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

# Compatibility aliases retain object identity for historical callers.
ClaimFirstRuntimeController = PlanAgentSession


__all__ = [
    "PlanAgentSession",
    "PlanAgentSessionError",
    "ClaimFirstRuntimeController",
    "ClaimFirstRuntimeError",
]
