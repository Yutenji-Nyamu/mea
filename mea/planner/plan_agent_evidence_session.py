"""Evidence adaptation and runtime-limit observation for Plan Agent sessions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .plan_agent_errors import PlanAgentSessionError
from .plan_agent_evidence import build_plan_agent_evidence_record, render_query_answer
from .plan_agent_schema import validate_open_query_evidence
from .runtime_limits import summarize_plan_evidence
from .query_interpretation import _nonempty_text


class PlanAgentEvidenceMixin:
    def observe_method_evidence(
        self,
        evidence_history: Sequence[Mapping[str, Any]],
        *,
        candidate_evidence: Sequence[Mapping[str, Any]],
        baseline_valid: bool,
        records: Sequence[Mapping[str, Any]] | None = None,
        baseline_stop_reason: str | None = None,
    ) -> dict[str, Any]:
        """Observe direct ``MethodRuntime`` evidence without a simulator schema.

        RoboTwin's production round summary carries a rich EvidencePacket.  A
        benchmark with a different native artifact format should not fabricate
        that transport merely to reuse the Plan Agent.  This boundary consumes
        the simulator-neutral evidence schema, applies the same runtime limits,
        and returns the same semantic state used by ``propose_semantic_step``.
        """

        if not isinstance(baseline_valid, bool):
            raise PlanAgentSessionError("baseline_valid must be bool")
        history = validate_open_query_evidence(list(evidence_history))
        trusted_records = [deepcopy(dict(item)) for item in (records or [])]
        if trusted_records:
            record_round_ids = [
                _nonempty_text(
                    record.get("round_id"),
                    f"records[{index}].round_id",
                )
                for index, record in enumerate(trusted_records)
            ]
            evidence_round_ids = [item["round_id"] for item in history]
            if record_round_ids != evidence_round_ids:
                raise PlanAgentSessionError(
                    "MethodRuntime evidence and answer records must have the "
                    "same ordered round ids"
                )
        else:
            trusted_records = [
                {"round_id": item["round_id"], "evidence_refs": []}
                for item in history
            ]

        assessment = summarize_plan_evidence(
            self.runtime_limits,
            candidate_evidence,
            completed_rounds=len(candidate_evidence),
        )
        if self.require_control_anchor and not baseline_valid:
            assessment = {
                **assessment,
                "should_stop": True,
                "stop_reason": (
                    baseline_stop_reason or "control_baseline_policy_failed"
                ),
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": (
                    "The unchanged control did not provide a valid policy "
                    "baseline, so generated-candidate attribution is blocked."
                ),
                "recommended_candidate_ids": [],
            }
        answer = (
            render_query_answer(
                self.user_query,
                assessment,
                trusted_records,
                baseline_valid=baseline_valid,
                baseline_stop_reason=baseline_stop_reason,
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
            "runtime_limits": deepcopy(self.runtime_limits),
            "assessment": assessment,
            "records": trusted_records,
            "open_query_evidence_history": history,
            "query_answer": answer,
        }

    def observe(
        self,
        round_plans: Sequence[Mapping[str, Any]],
        round_summaries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Normalize all completed rounds and decide whether execution stops."""

        if len(round_plans) != len(round_summaries):
            raise PlanAgentSessionError(
                "completed plans and summaries must be aligned"
            )
        if self.require_control_anchor and not round_plans:
            raise PlanAgentSessionError(
                "control-first observation requires one completed control round"
            )
        records = [
            build_plan_agent_evidence_record(plan, summary)
            for plan, summary in zip(round_plans, round_summaries)
        ]
        control_semantics: Mapping[str, Any] = {}
        control_authority_valid = True
        control_pipeline_valid = True
        baseline_valid = True
        if self.require_control_anchor:
            if records[0]["template_id"] != self.control_template:
                raise PlanAgentSessionError(
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
        policy_candidate_records = [
            record
            for record in candidate_records
            if record.get("planning_observation") is None
        ]
        candidate_evidence = [
            deepcopy(record["candidate_evidence"])
            for record in policy_candidate_records
        ]
        assessment = summarize_plan_evidence(
            self.runtime_limits,
            candidate_evidence,
            completed_rounds=len(policy_candidate_records),
        )
        semantic_non_comparable_ids = [
            record["candidate_id"]
            for record in candidate_records
            if (
                record.get("evaluation_outcome", {}).get("metric")
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
            "runtime_limits": deepcopy(self.runtime_limits),
            "assessment": assessment,
            "records": records,
            "open_query_evidence_history": validate_open_query_evidence(
                [record["open_query_evidence"] for record in records]
            ),
            "query_answer": answer,
        }


__all__ = ["PlanAgentEvidenceMixin"]
