"""Runtime bridge for claim-first open-Query planning.

``ClaimFirstOpenQueryAgent`` deliberately emits a semantic experiment rather
than an executable catalog step.  This module connects that semantic proposal
to the existing bounded ACT runtime without letting a language model invent
execution details or decide when evidence is sufficient.

The bridge has four explicit responsibilities:

* run one unchanged official-scene control before property attribution;
* derive OpenQueryEvidence and finite-domain candidate evidence directly from
  the runtime-owned EvidencePacket and round-provenance sidecar;
* apply the query-sufficiency contract before accepting a model-authored stop;
* resolve a semantic sub-aspect to one still-legal trusted template only after
  the model has made its claim-first proposal.

This remains a bounded finite-domain protocol.  It is not a statistical
generalization guarantee and does not make the hidden executable catalog part
of the model prompt.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.round_provenance import canonical_sha256

from .claim_first import (
    ClaimFirstPlanError,
    validate_open_query_evidence,
    validate_open_query_plan_proposal,
)
from .evidence_policy import build_evidence_packet, validate_evidence_packet
from .query_contract import (
    assess_query_sufficiency,
    build_query_sufficiency_contract,
    infer_claim_type,
    validate_query_sufficiency_contract,
)


class ClaimFirstRuntimeError(ValueError):
    """Raised when semantic planning cannot be bound to trusted evidence."""


CONTROL_TEMPLATE_BY_TASK = {
    # Both templates preserve the upstream scene.  Their Rule metric may be
    # task-specific, while the policy-success field supplies the common
    # clean-control gate used here.
    "beat_block_hammer": "safety.hammer_left_camera_contact.official",
    "click_bell": "performance.completion_time_stability.official",
}

_SEMANTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "under",
    "with",
    "semantic",
    "sub",
    "aspect",
    "test",
    "task",
    "policy",
}


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClaimFirstRuntimeError(f"{field} must be a non-empty string")
    return value.strip()


def control_template_id(target: Mapping[str, Any]) -> str:
    """Return the trusted official-scene control for a bound task."""

    task_name = _nonempty_text(target.get("task_name"), "target.task_name")
    template_id = CONTROL_TEMPLATE_BY_TASK.get(task_name)
    if template_id is None:
        raise ClaimFirstRuntimeError(
            f"claim-first control anchor is not defined for {task_name!r}"
        )
    available = {
        str(item)
        for aspect in target.get("aspects", [])
        if isinstance(aspect, Mapping)
        for item in aspect.get("template_ids", [])
    }
    if template_id not in available:
        raise ClaimFirstRuntimeError(
            f"control template {template_id!r} is outside the bound task"
        )
    return template_id


def build_control_anchor_proposal(
    target: Mapping[str, Any],
    user_query: str,
) -> dict[str, Any]:
    """Build the cached first-round proposal consumed by legacy materializers.

    No provider call is needed to choose a control: it is a protocol
    prerequisite rather than an answer to the open Query.
    """

    query = _nonempty_text(user_query, "user_query")
    task_name = _nonempty_text(target.get("task_name"), "target.task_name")
    template_id = control_template_id(target)
    if task_name == "click_bell":
        return {
            "schema_version": 1,
            "task_name": "click_bell",
            "evaluation_goal": (
                "establish_clean_control_before_claim_first_attribution: "
                + query
            ),
            "requested_aspect_ids": [
                "performance.completion_time_stability"
            ],
            "first_aspect_id": "performance.completion_time_stability",
        }
    if task_name == "beat_block_hammer":
        return {
            "schema_version": 5,
            "task_name": "beat_block_hammer",
            "policy": deepcopy(dict(target["policy"])),
            "evaluation_goal": (
                "establish_clean_control_before_claim_first_attribution: "
                + query
            ),
            "requested_template_ids": [template_id],
            "first_template_id": template_id,
            "max_rounds": int(target["max_rounds"]),
        }
    raise ClaimFirstRuntimeError(
        f"claim-first control proposal is not supported for {task_name!r}"
    )


def _template_aspect(target: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(template_id): str(aspect["aspect_id"])
        for aspect in target.get("aspects", [])
        if isinstance(aspect, Mapping)
        for template_id in aspect.get("template_ids", [])
    }


def _semantic_tokens(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return set().union(
            *(_semantic_tokens(key) | _semantic_tokens(item) for key, item in value.items())
        ) if value else set()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return set().union(*(_semantic_tokens(item) for item in value)) if value else set()
    if not isinstance(value, str):
        return set()
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", value.casefold())
        if token not in _SEMANTIC_STOPWORDS and len(token) > 1
    }


def resolve_semantic_proposal(
    proposal: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    executed_template_ids: Sequence[str],
    control_template: str,
) -> dict[str, Any]:
    """Resolve one semantic proposal to a unique unexecuted trusted template.

    Exact aspect identity wins.  Otherwise a deterministic lexical score over
    the semantic request and runtime-private aspect descriptions is used.  A
    tie or zero score fails closed; there is no first-item fallback.
    """

    normalized = validate_open_query_plan_proposal(proposal, has_evidence=True)
    if normalized["action"] != "continue":
        raise ClaimFirstRuntimeError(
            "the query contract, not the model, owns claim-first stopping"
        )
    executed = {str(item) for item in executed_template_ids}
    candidates: list[dict[str, Any]] = []
    proposal_aspect = str(normalized["sub_aspect"])
    proposal_tokens = _semantic_tokens(normalized)
    for aspect in target.get("aspects", []):
        if not isinstance(aspect, Mapping):
            continue
        aspect_id = str(aspect.get("aspect_id") or "")
        aspect_tokens = _semantic_tokens(
            {
                "aspect_id": aspect_id,
                "description": aspect.get("description"),
            }
        )
        exact = proposal_aspect == aspect_id
        for raw_template in aspect.get("template_ids", []):
            template_id = str(raw_template)
            if template_id == control_template or template_id in executed:
                continue
            template_tokens = aspect_tokens | _semantic_tokens(template_id)
            overlap = sorted(proposal_tokens & template_tokens)
            exact_template = proposal_aspect == template_id
            score = (
                (2000 if exact_template else 1000 if exact else 0)
                + len(overlap)
            )
            candidates.append(
                {
                    "aspect_id": aspect_id,
                    "template_id": template_id,
                    "score": score,
                    "matched_tokens": overlap,
                }
            )
    if not candidates:
        raise ClaimFirstRuntimeError(
            "no unexecuted non-control template remains in the bound task"
        )
    candidates.sort(key=lambda item: (-int(item["score"]), item["template_id"]))
    best = candidates[0]
    tied = [item for item in candidates if item["score"] == best["score"]]
    if int(best["score"]) <= 0 or len(tied) != 1:
        raise ClaimFirstRuntimeError(
            "semantic proposal does not resolve uniquely to one trusted "
            f"template; top candidates={[(item['template_id'], item['score']) for item in tied]}"
        )
    return {
        "schema_version": 1,
        "semantic_sub_aspect": proposal_aspect,
        "resolved_aspect_id": best["aspect_id"],
        "resolved_template_id": best["template_id"],
        "resolution": (
            "exact_template"
            if best["score"] >= 2000
            else "exact_aspect"
            if best["score"] >= 1000
            else "unique_lexical"
        ),
        "matched_tokens": best["matched_tokens"],
        "catalog_was_model_visible": False,
    }


def _round_artifact_refs(
    round_summary: Mapping[str, Any],
    round_provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pointer = round_summary.get("provenance")
    if not isinstance(pointer, Mapping):
        raise ClaimFirstRuntimeError(
            "claim-first evidence requires a round provenance pointer"
        )
    binding = round_provenance.get("binding")
    if not isinstance(binding, Mapping):
        raise ClaimFirstRuntimeError(
            "claim-first evidence requires a round provenance binding"
        )
    if pointer.get("binding_sha256") != round_provenance.get("binding_sha256"):
        raise ClaimFirstRuntimeError(
            "round provenance pointer and binding sha256 disagree"
        )
    if binding.get("round_id") != round_summary.get("round_id"):
        raise ClaimFirstRuntimeError(
            "round provenance binding does not match the round summary"
        )
    refs = [
        {
            "kind": "round_provenance",
            "path": _nonempty_text(pointer.get("path"), "provenance.path"),
            "sha256": _nonempty_text(pointer.get("sha256"), "provenance.sha256"),
        }
    ]
    for raw in binding.get("artifacts", []):
        if not isinstance(raw, Mapping):
            raise ClaimFirstRuntimeError("provenance artifact ref must be an object")
        kind = str(raw.get("kind") or "")
        if kind not in {
            "act_metadata",
            "act_result",
            "child_manifest",
            "execution_vqa_result",
            "round_aggregate",
            "tool_execution",
        }:
            continue
        refs.append(
            {
                "kind": kind,
                "path": _nonempty_text(raw.get("path"), f"{kind}.path"),
                "sha256": _nonempty_text(raw.get("sha256"), f"{kind}.sha256"),
            }
        )
    if not any(item["kind"] == "child_manifest" for item in refs):
        raise ClaimFirstRuntimeError(
            "round provenance has no child_manifest evidence ref"
        )
    return refs


def build_claim_first_evidence_record(
    round_plan: Mapping[str, Any],
    round_summary: Mapping[str, Any],
    round_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive compact semantic/query evidence from one completed runtime round."""

    if round_plan.get("round_id") != round_summary.get("round_id"):
        raise ClaimFirstRuntimeError("round plan and summary ids disagree")
    binding = round_provenance.get("binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("round_plan_sha256") != canonical_sha256(round_plan)
    ):
        raise ClaimFirstRuntimeError(
            "round provenance is not bound to the supplied round plan"
        )
    packet = validate_evidence_packet(
        build_evidence_packet(
            {"rounds": [deepcopy(dict(round_plan))], "max_rounds": 1},
            [deepcopy(dict(round_summary))],
        )
    )
    refs = _round_artifact_refs(round_summary, round_provenance)
    strength = packet["evidence_strength"]
    success_rate = packet["policy"]["success_rate"]
    if strength == "conflicting":
        semantic_outcome = "ambiguous"
        candidate_outcome = "conflict"
    elif strength != "sufficient" or success_rate is None:
        semantic_outcome = "ambiguous"
        candidate_outcome = "unknown"
    elif float(success_rate) >= 1.0:
        semantic_outcome = "success"
        candidate_outcome = "pass"
    else:
        semantic_outcome = "failure"
        candidate_outcome = "fail"

    task_proposal = round_plan.get("task_proposal") or {}
    sub_aspect = str(
        task_proposal.get("aspect_id")
        or round_plan.get("sub_aspect")
        or round_plan.get("aspect_id")
        or "unknown"
    )
    hypothesis = str(
        task_proposal.get("intent")
        or round_plan.get("task_instruction")
        or f"Evaluate {sub_aspect}."
    ).strip()
    changes = task_proposal.get("changes")
    perturbation = (
        json.dumps(changes, ensure_ascii=False, sort_keys=True)
        if isinstance(changes, Mapping) and changes
        else "unchanged official-scene control"
        if str(round_plan.get("template_id")) in CONTROL_TEMPLATE_BY_TASK.values()
        else str(round_plan.get("template_id") or sub_aspect)
    )
    limitations = [
        "One bounded runtime round is not a statistical generalization estimate."
    ]
    if strength != "sufficient":
        limitations.append(
            "The typed Rule/VQA/pipeline evidence is not sufficient: "
            + ", ".join(packet["reason_codes"] or [strength])
        )
    if success_rate is None:
        limitations.append("Policy success was not reported for this round.")
    summary_text = (
        f"EvidencePacket strength={strength}; policy_success_rate="
        f"{success_rate}; Rule metric={packet['rule']['metric']}; "
        f"VQA status={packet['vqa']['status']}."
    )
    open_query = validate_open_query_evidence(
        [
            {
                "schema_version": 1,
                "round_id": str(round_plan["round_id"]),
                "tested_sub_aspect": sub_aspect,
                "tested_hypothesis": hypothesis,
                "tested_perturbation": perturbation,
                "outcome": semantic_outcome,
                "evidence_summary": summary_text,
                "limitations": limitations,
            }
        ]
    )[0]
    diagnosis = None
    if candidate_outcome == "fail":
        diagnosis = (
            f"Observed policy success_rate={float(success_rate):.6g} for "
            f"{round_plan.get('template_id')} with complete Rule metric "
            f"{packet['rule']['metric']}; this localizes an observed weakness "
            "but does not establish a causal mechanism."
        )
    candidate = {
        "candidate_id": str(round_plan.get("template_id") or ""),
        "outcome": candidate_outcome,
        "score": (
            float(success_rate) if success_rate is not None else None
        ),
        "diagnosis": diagnosis,
    }
    binding_payload = {
        "round_id": round_plan["round_id"],
        "template_id": round_plan.get("template_id"),
        "evidence_packet": packet,
        "evidence_refs": refs,
    }
    return {
        "schema_version": 1,
        "round_id": str(round_plan["round_id"]),
        "template_id": str(round_plan.get("template_id") or ""),
        "open_query_evidence": open_query,
        "candidate_evidence": candidate,
        "evidence_packet": packet,
        "evidence_refs": refs,
        "binding_sha256": canonical_sha256(binding_payload),
    }


def render_query_answer(
    user_query: str,
    assessment: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_valid: bool,
    baseline_stop_reason: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic query answer/limitation projection."""

    query = _nonempty_text(user_query, "user_query")
    if not baseline_valid:
        answered = False
        stop_reason = baseline_stop_reason or "control_baseline_invalid"
        verdict = "inconclusive"
        answer = (
            "The original Query cannot be attributed to a tested property "
            "because the required unchanged-scene control did not produce "
            "complete successful policy evidence."
        )
        untested = list(
            assessment.get("contract", {}).get("candidate_universe", [])
        )
        limitations = [
            "No property attribution is allowed without a passing control.",
            "The observed control result may reflect policy, simulator, or pipeline effects.",
        ]
    else:
        answered = bool(assessment.get("evidence_sufficient"))
        stop_reason = str(assessment.get("stop_reason") or "continue")
        verdict = str(assessment.get("claim_verdict") or "inconclusive")
        if answered:
            answer = (
                f"For the finite registered candidate domain, the Query verdict "
                f"is {verdict}."
            )
        else:
            answer = (
                "The bounded evidence does not yet satisfy the truth conditions "
                "needed to answer the original Query."
            )
        untested = list(assessment.get("untested_candidate_ids") or [])
        limitations = list(assessment.get("limitations") or [])
    if untested:
        limitations.append(
            "Untested finite-domain candidates: " + ", ".join(untested)
        )
    limitations.extend(
        [
            "This answer is limited to the bound task, checkpoint, variants, and recorded seeds.",
            "A finite-domain N-small result is not a broad generalization guarantee.",
        ]
    )
    refs = [
        deepcopy(ref)
        for record in records
        for ref in record.get("evidence_refs", [])
        if isinstance(ref, Mapping)
    ]
    return {
        "schema_version": 1,
        "original_query": query,
        "answered": answered,
        "stop_reason": stop_reason,
        "claim_type": assessment.get("contract", {}).get("claim_type"),
        "claim_verdict": verdict,
        "answer": answer,
        "tested_candidate_ids": list(
            assessment.get("observed_candidate_ids") or []
        ),
        "untested_candidate_ids": untested,
        "limitations": list(dict.fromkeys(limitations)),
        "evidence_refs": refs,
        "evidence_binding_sha256": canonical_sha256(
            [record.get("binding_sha256") for record in records]
        ),
    }


class ClaimFirstRuntimeController:
    """Own control gating, query sufficiency, and semantic catalog resolution."""

    def __init__(
        self,
        user_query: str,
        target: Mapping[str, Any],
        *,
        query_contract: Mapping[str, Any] | None = None,
    ):
        self.user_query = _nonempty_text(user_query, "user_query")
        self.target = deepcopy(dict(target))
        self.control_template = control_template_id(self.target)
        self.template_to_aspect = _template_aspect(self.target)
        candidates = [
            template_id
            for template_id in self.template_to_aspect
            if template_id != self.control_template
        ]
        round_budget = int(self.target.get("max_rounds") or 0) - 1
        if not candidates or round_budget < 1:
            raise ClaimFirstRuntimeError(
                "claim-first runtime needs one control and at least one candidate round"
            )
        if query_contract is None:
            claim_type = infer_claim_type(self.user_query)
            if claim_type == "comparative":
                raise ClaimFirstRuntimeError(
                    "comparative Query requires an explicit preregistered "
                    "query-sufficiency contract with two groups"
                )
            contract = build_query_sufficiency_contract(
                self.user_query,
                candidate_universe=candidates,
                round_budget=round_budget,
                claim_type=claim_type,
            )
        else:
            contract = validate_query_sufficiency_contract(query_contract)
            if set(contract["candidate_universe"]) - set(candidates):
                raise ClaimFirstRuntimeError(
                    "query contract leaves the non-control bound candidate domain"
                )
            if int(contract["round_budget"]) > round_budget:
                raise ClaimFirstRuntimeError(
                    "query contract spends rounds reserved for the control anchor"
                )
        self.query_contract = contract

    def observe(
        self,
        round_plans: Sequence[Mapping[str, Any]],
        round_summaries: Sequence[Mapping[str, Any]],
        round_provenances: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Normalize all completed rounds and decide whether execution stops."""

        if not (
            len(round_plans)
            == len(round_summaries)
            == len(round_provenances)
            and round_plans
        ):
            raise ClaimFirstRuntimeError(
                "completed plans, summaries, and provenance must be non-empty and aligned"
            )
        records = [
            build_claim_first_evidence_record(plan, summary, provenance)
            for plan, summary, provenance in zip(
                round_plans, round_summaries, round_provenances
            )
        ]
        if records[0]["template_id"] != self.control_template:
            raise ClaimFirstRuntimeError(
                "claim-first property attribution requires the control template first"
            )
        control_packet = records[0]["evidence_packet"]
        baseline_valid = bool(
            control_packet["evidence_strength"] == "sufficient"
            and control_packet["policy"]["success_rate"] is not None
            and float(control_packet["policy"]["success_rate"]) >= 1.0
        )
        candidate_records = records[1:]
        candidate_evidence = [
            deepcopy(record["candidate_evidence"])
            for record in candidate_records
            if record["template_id"] in self.query_contract["candidate_universe"]
        ]
        assessment = assess_query_sufficiency(
            self.query_contract,
            candidate_evidence,
            completed_rounds=len(candidate_records),
        )
        if not baseline_valid:
            reason = (
                "control_baseline_pipeline_invalid"
                if control_packet["evidence_strength"] != "sufficient"
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
            "control_passed": baseline_valid,
            "query_contract": deepcopy(self.query_contract),
            "assessment": assessment,
            "records": records,
            "open_query_evidence_history": validate_open_query_evidence(
                [record["open_query_evidence"] for record in records]
            ),
            "query_answer": answer,
        }

    def bind_semantic_step(
        self,
        proposal_bundle: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        executed_template_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Validate and resolve a provider/cached semantic next-step bundle."""

        assessment = observation.get("assessment")
        if not isinstance(assessment, Mapping):
            raise ClaimFirstRuntimeError("claim-first observation has no assessment")
        if assessment.get("should_stop"):
            raise ClaimFirstRuntimeError(
                "cannot bind a semantic step after the query contract stopped"
            )
        if observation.get("control_passed") is not True:
            raise ClaimFirstRuntimeError(
                "cannot attribute a property before the control passes"
            )
        raw_proposal = proposal_bundle.get("proposal")
        if not isinstance(raw_proposal, Mapping):
            raise ClaimFirstRuntimeError(
                "claim-first proposal bundle has no proposal object"
            )
        try:
            proposal = validate_open_query_plan_proposal(
                raw_proposal, has_evidence=True
            )
        except ClaimFirstPlanError as exc:
            raise ClaimFirstRuntimeError(str(exc)) from exc
        resolution = resolve_semantic_proposal(
            proposal,
            target=self.target,
            executed_template_ids=executed_template_ids,
            control_template=self.control_template,
        )
        current_aspect = self.template_to_aspect.get(
            str(executed_template_ids[-1])
        )
        return {
            "schema_version": 1,
            "semantic_proposal_bundle": deepcopy(dict(proposal_bundle)),
            "resolution": resolution,
            "plan_step": {
                "schema_version": 1,
                "action": (
                    "refine"
                    if resolution["resolved_aspect_id"] == current_aspect
                    else "propose"
                ),
                "aspect_id": resolution["resolved_aspect_id"],
                "template_id": resolution["resolved_template_id"],
                "rationale": proposal["rationale"],
                "answered_query": False,
            },
        }


__all__ = [
    "CONTROL_TEMPLATE_BY_TASK",
    "ClaimFirstRuntimeController",
    "ClaimFirstRuntimeError",
    "build_claim_first_evidence_record",
    "build_control_anchor_proposal",
    "control_template_id",
    "render_query_answer",
    "resolve_semantic_proposal",
]
