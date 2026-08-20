"""Evidence adaptation and runtime-limit observation for Plan Agent sessions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .plan_agent_errors import PlanAgentSessionError
from .plan_agent_evidence import (
    build_plan_agent_evidence_record,
    render_query_answer,
)
from .plan_agent_schema import validate_open_query_evidence
from .runtime_limits import summarize_plan_evidence
from .query_interpretation import _nonempty_text


_BASELINE_LIMITATION = (
    "A supported or refuted answer requires a successful unchanged official "
    "baseline; no valid baseline has completed yet."
)
_VALID_OFFICIAL_BASELINE_IDENTITIES = frozenset(
    {
        (
            "official_check_success",
            "official_check_success",
            True,
            "official_check_success",
            "official_check_success",
            True,
            "official_only",
        ),
        (
            "official_check_success",
            "official_check_success_reused",
            True,
            "official_check_success",
            "official_check_success_reused",
            True,
            "official_only",
        ),
    }
)


def _baseline_passed(record: Mapping[str, Any]) -> bool:
    evidence = record["round_evidence"]
    policy = evidence["policy"]
    evaluation_outcome = record["evaluation_outcome"]
    semantics = record.get("outcome_semantics") or {}
    official_identity = (
        evaluation_outcome.get("metric"),
        evaluation_outcome.get("authority"),
        evaluation_outcome.get("official_equivalent"),
        policy.get("metric"),
        policy.get("authority"),
        policy.get("official_equivalent"),
        semantics.get("status"),
    )
    authority_valid = official_identity in _VALID_OFFICIAL_BASELINE_IDENTITIES
    return bool(
        authority_valid
        and policy["success_rate"] is not None
        and record["candidate_evidence"]["outcome"] == "pass"
        and float(policy["success_rate"]) >= 1.0
    )


def _is_unchanged_official_retry(round_plan: Mapping[str, Any]) -> bool:
    if round_plan.get("route") != "official":
        return False
    proposal = round_plan.get("proposal") or round_plan.get(
        "experiment_candidate"
    )
    return bool(
        isinstance(proposal, Mapping)
        and all(
            proposal.get(field) is None
            for field in (
                "scene_need",
                "checker_need",
                "rule_tool_need",
                "vqa_tool_need",
                "tool_need",
            )
        )
    )


def _with_baseline_limitation(
    assessment: Mapping[str, Any],
    *,
    baseline_valid: bool,
) -> dict[str, Any]:
    result = deepcopy(dict(assessment))
    if not baseline_valid:
        result["limitations"] = list(
            dict.fromkeys(
                [*(result.get("limitations") or []), _BASELINE_LIMITATION]
            )
        )
    return result


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

        RoboTwin's production round summary carries typed RoundEvidence. A
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
        assessment = _with_baseline_limitation(
            assessment,
            baseline_valid=(
                baseline_valid if self.require_control_anchor else True
            ),
        )
        answer = (
            render_query_answer(
                self.user_query,
                assessment,
                trusted_records,
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
        baseline_valid = True
        baseline_attempt_indexes: set[int] = set()
        if self.require_control_anchor:
            if records[0]["template_id"] != self.control_template:
                raise PlanAgentSessionError(
                    "Plan Agent property attribution requires the control "
                    "template first"
                )
            baseline_attempt_indexes = {
                index
                for index, plan in enumerate(round_plans)
                if index == 0 or _is_unchanged_official_retry(plan)
            }
            baseline_valid = any(
                _baseline_passed(records[index])
                for index in sorted(baseline_attempt_indexes)
            )
        candidate_start = 1 if self.require_control_anchor else 0
        charged_round_records = [
            (index, records[index])
            for index in range(candidate_start, len(records))
            if (
                index in baseline_attempt_indexes
                or records[index].get("planning_observation") is None
            )
        ]
        candidate_evidence = [
            deepcopy(record["candidate_evidence"])
            for index, record in charged_round_records
            if index not in baseline_attempt_indexes
        ]
        assessment = summarize_plan_evidence(
            self.runtime_limits,
            candidate_evidence,
            completed_rounds=len(charged_round_records),
        )
        assessment = _with_baseline_limitation(
            assessment,
            baseline_valid=baseline_valid,
        )
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
