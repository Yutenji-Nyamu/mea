"""Evidence projection and answer rendering for the Plan Agent."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.artifact_retrieval_index import resolve_task_retrieval_index

from .plan_agent_schema import PlanAgentError, validate_open_query_evidence
from .evidence_policy import build_evidence_packet, validate_evidence_packet
from .plan_agent_errors import PlanAgentSessionError
from .query_interpretation import _nonempty_text


def _round_candidate_id(round_plan: Mapping[str, Any]) -> str:
    """Return the planner-owned candidate identity for one executed round."""

    return _nonempty_text(
        round_plan.get("candidate_id") or round_plan.get("template_id"),
        "round_plan.candidate_id",
    )


def _uses_task_control_template(round_plan: Mapping[str, Any]) -> bool:
    """Recognize the bound task's unchanged official control artifact."""

    task_name = round_plan.get("task_name")
    template_id = round_plan.get("template_id")
    if (
        not isinstance(task_name, str)
        or not task_name.strip()
        or not isinstance(template_id, str)
        or not template_id.strip()
    ):
        return False
    try:
        retrieval_index = resolve_task_retrieval_index(
            task_name.strip(),
            allow_unregistered=True,
        )
    except ValueError:
        return False
    return template_id.strip() == retrieval_index["control_template_id"]


def _round_artifact_refs(
    round_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project the small set of artifacts needed to inspect one round.

    These are navigation pointers, not integrity claims.  Experimental
    preregistration may hash a bundle separately without making the normal
    Plan Agent runtime depend on a provenance subsystem.
    """

    refs: list[dict[str, Any]] = []
    child_run_id = str(round_summary.get("taskgen_run_id") or "").strip()
    if child_run_id:
        refs.append(
            {
                "kind": "child_manifest",
                "path": f"mea/generated_tasks/{child_run_id}/manifest.json",
            }
        )

    execution_dir = str(
        round_summary.get("execution_artifact_dir") or ""
    ).strip().rstrip("/\\")
    explicit_artifacts = round_summary.get("evidence_artifact_paths")
    explicit_artifacts = (
        explicit_artifacts
        if isinstance(explicit_artifacts, Mapping)
        else {}
    )
    observations = round_summary.get("observations")
    observations = observations if isinstance(observations, Mapping) else {}
    evidence_aggregate_path = str(
        explicit_artifacts.get("evidence_aggregate") or ""
    ).strip()
    if not evidence_aggregate_path and execution_dir:
        evidence_aggregate_path = f"{execution_dir}/evidence_aggregate.json"
    if evidence_aggregate_path and isinstance(
        observations.get("evidence_aggregate"), Mapping
    ):
        refs.append(
            {
                "kind": "evidence_aggregate",
                "path": evidence_aggregate_path,
            }
        )
    aggregate_path = str(
        explicit_artifacts.get("round_aggregate") or ""
    ).strip()
    if not aggregate_path and execution_dir:
        aggregate_path = f"{execution_dir}/aggregate_result.json"
    if aggregate_path and isinstance(observations.get("aggregate"), Mapping):
        refs.append(
            {
                "kind": "round_aggregate",
                "path": aggregate_path,
            }
        )
    planned_tool = observations.get("planned_tool")
    tool_path = str(
        explicit_artifacts.get("tool_execution") or ""
    ).strip()
    if not tool_path and execution_dir:
        tool_path = f"{execution_dir}/planned_tool/tool_execution.json"
    if (
        tool_path
        and isinstance(planned_tool, Mapping)
        and planned_tool.get("status") != "skipped"
    ):
        refs.append(
            {
                "kind": "tool_execution",
                "path": tool_path,
            }
        )
    execution_vqa = observations.get("execution_vqa")
    if isinstance(execution_vqa, Mapping):
        artifacts = execution_vqa.get("artifacts")
        result_path = (
            str(artifacts.get("result") or "").strip()
            if isinstance(artifacts, Mapping)
            else ""
        )
        if result_path:
            refs.append(
                {
                    "kind": "execution_vqa_result",
                    "path": result_path,
                }
            )
    return refs


def _compact_planned_tool_evidence(
    observations: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expose measured Tool values to the next semantic planning step.

    The typed Tool execution remains authoritative. This projection is only a
    compact prompt-facing observation; it never replaces official success or
    changes the deterministic runtime evidence summary by itself.
    """

    planned = observations.get("planned_tool")
    if not isinstance(planned, Mapping):
        return []
    route_decision = planned.get("route_decision")
    route_decision = (
        route_decision if isinstance(route_decision, Mapping) else {}
    )
    validation = planned.get("validation")
    validation = validation if isinstance(validation, Mapping) else {}
    route = (
        planned.get("route")
        or route_decision.get("resolved_route")
        or route_decision.get("route")
    )
    provider_called = route_decision.get(
        "provider_called",
        validation.get("provider_called"),
    )
    tool_request = planned.get("tool_request")
    tool_request = (
        tool_request if isinstance(tool_request, Mapping) else {}
    )
    metric_spec = tool_request.get("metric_spec")
    metric_spec = metric_spec if isinstance(metric_spec, Mapping) else {}
    metric_description = str(metric_spec.get("description") or "").strip()
    semantic_review = validation.get("semantic_review")
    semantic_review = (
        semantic_review if isinstance(semantic_review, Mapping) else {}
    )
    semantic_checks = semantic_review.get("checks")
    semantic_checks = (
        semantic_checks if isinstance(semantic_checks, Mapping) else {}
    )
    compact: list[dict[str, Any]] = []
    episodes = planned.get("episodes")
    if not isinstance(episodes, list):
        return compact
    for episode in episodes:
        if not isinstance(episode, Mapping):
            continue
        result = episode.get("result")
        result = result if isinstance(result, Mapping) else episode
        details = result.get("details")
        details = details if isinstance(details, Mapping) else {}
        item = {
            "metric": str(
                result.get("tool")
                or result.get("metric")
                or planned.get("reference_tool")
                or ""
            ),
            "value": result.get("value"),
            "unit": result.get("unit"),
            "passed": result.get("passed"),
            "route": route,
            "provider_called": provider_called,
            "null_reason": details.get("reason"),
        }
        if metric_description:
            item["description"] = metric_description
        if semantic_checks.get("returns_diagnostic_not_success") is True:
            item["returns_diagnostic_not_success"] = True
        compact.append(item)
    return compact


def build_plan_agent_evidence_record(
    round_plan: Mapping[str, Any],
    round_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive compact semantic/query evidence from one completed runtime round."""

    if round_plan.get("round_id") != round_summary.get("round_id"):
        raise PlanAgentSessionError("round plan and summary ids disagree")
    candidate_id = _round_candidate_id(round_plan)
    evidence_round = deepcopy(dict(round_plan))
    # EvidencePacket v1 calls this execution identity ``template_id``.  A
    # dynamic plan has no catalog template, so project its candidate id into
    # that legacy transport field without mutating the runtime plan.
    if not evidence_round.get("template_id"):
        evidence_round["template_id"] = candidate_id
    packet = validate_evidence_packet(
        build_evidence_packet(
            {"rounds": [evidence_round], "max_rounds": 1},
            [deepcopy(dict(round_summary))],
        )
    )
    refs = _round_artifact_refs(round_summary)
    observations = round_summary.get("observations")
    observations = observations if isinstance(observations, Mapping) else {}
    planning_observation = observations.get("planning_observation")
    planning_observation = (
        deepcopy(dict(planning_observation))
        if isinstance(planning_observation, Mapping)
        else None
    )
    policy_outcome = (
        observations.get("policy_outcome")
        if isinstance(observations.get("policy_outcome"), Mapping)
        else {
            "metric": "official_check_success",
            "authority": "official_check_success",
            "binding": None,
            "value": None,
            "official_equivalent": True,
            "execution_scope": "legacy_unspecified_official",
        }
    )
    outcome_semantics = observations.get("outcome_semantics")
    if not isinstance(outcome_semantics, Mapping):
        outcome_semantics = policy_outcome.get("outcome_semantics")
    outcome_semantics = (
        deepcopy(dict(outcome_semantics))
        if isinstance(outcome_semantics, Mapping)
        else {
            "schema_version": 1,
            "status": "non_comparable",
            "evidence_conflict": False,
            "official_equivalent": policy_outcome.get("official_equivalent"),
            "episodes": [],
            "reason_codes": ["outcome_semantics_not_recorded"],
        }
    )
    outcome_semantics_status = str(
        outcome_semantics.get("status") or "non_comparable"
    )
    strength = packet["evidence_strength"]
    success_rate = packet["policy"]["success_rate"]
    generated_metric = (
        policy_outcome.get("metric") == "generated_check_success"
    )
    if planning_observation is not None:
        semantic_outcome = "ambiguous"
        candidate_outcome = "unknown"
    elif outcome_semantics_status == "conflict":
        semantic_outcome = "ambiguous"
        candidate_outcome = "conflict"
    elif generated_metric and outcome_semantics_status not in {
        "equivalent_agreement",
        "expected_semantic_extension",
    }:
        semantic_outcome = "ambiguous"
        candidate_outcome = "unknown"
    elif strength == "conflicting":
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
    dynamic_proposal = round_plan.get("proposal") or round_plan.get(
        "experiment_candidate"
    )
    dynamic_proposal = (
        dynamic_proposal if isinstance(dynamic_proposal, Mapping) else {}
    )
    evaluation_intent = dynamic_proposal.get("evaluation_intent")
    evaluation_intent = (
        evaluation_intent if isinstance(evaluation_intent, Mapping) else {}
    )
    sub_aspect = str(
        evaluation_intent.get("original_concern")
        or task_proposal.get("aspect_id")
        or round_plan.get("sub_aspect")
        or round_plan.get("aspect_id")
        or "unknown"
    )
    hypothesis = str(
        evaluation_intent.get("hypothesis")
        or task_proposal.get("intent")
        or dynamic_proposal.get("semantic_concern")
        or round_plan.get("task_instruction")
        or f"Evaluate {sub_aspect}."
    ).strip()
    changes = task_proposal.get("changes")
    scene_need = dynamic_proposal.get("scene_need")
    scene_description = (
        str(scene_need.get("description") or "").strip()
        if isinstance(scene_need, Mapping)
        else ""
    )
    perturbation = str(
        evaluation_intent.get("requested_change") or scene_description
    ).strip()
    if not perturbation:
        perturbation = (
            json.dumps(changes, ensure_ascii=False, sort_keys=True)
            if isinstance(changes, Mapping) and changes
            else "unchanged official-scene control"
            if _uses_task_control_template(round_plan)
            else candidate_id
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
    if planning_observation is not None:
        planning_kind = str(planning_observation.get("kind") or "")
        limitations.append(
            (
                "The scripted expert could not certify this candidate or its "
                "same-seed comparator; the candidate outcome remains unknown. "
                if planning_kind == "expert_oracle_unavailable"
                else "TaskGen rejected this candidate before policy execution. "
            )
            + "This is planning evidence only and contributes N=0 policy samples."
        )
    if policy_outcome.get("official_equivalent") is False:
        limitations.append(
            "This round is judged by the bounded generated_check_success "
            "predicate and is not an official RoboTwin success result."
        )
    if outcome_semantics_status == "expected_semantic_extension":
        limitations.append(
            "The generated checker has not been certified as equivalent to "
            "the official core predicate; its verdict must be treated as "
            "experimental."
        )
    elif outcome_semantics_status == "conflict":
        limitations.append(
            "Generated and official/core success semantics conflict; this "
                "round cannot support an answered Plan Agent stop."
        )
    planned_tool_evidence = _compact_planned_tool_evidence(observations)
    tool_summary = (
        json.dumps(
            planned_tool_evidence,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        if planned_tool_evidence
        else "[]"
    )
    summary_text = (
        f"EvidencePacket strength={strength}; "
        f"authoritative_candidate_outcome={semantic_outcome}; "
        f"success_predicate_metric={policy_outcome.get('metric')}; "
        f"success_predicate_value={policy_outcome.get('value')}; "
        f"success_predicate_authority={policy_outcome.get('authority')}; "
        f"success_predicate_semantics={outcome_semantics_status}; "
        f"policy_success_rate={success_rate}; "
        f"Rule metric={packet['rule']['metric']}; "
        f"VQA status={packet['vqa']['status']}; "
        "diagnostic_tool_role=supporting_measurement_not_success_authority; "
        f"diagnostic_tool_measurements={tool_summary}."
    )
    if planning_observation is not None:
        planning_summary = {
            key: planning_observation.get(key)
            for key in (
                "kind",
                "failure_stage",
                "reason_code",
                "diagnosis",
                "policy_rollouts_started",
                "policy_sample_count",
            )
            if planning_observation.get(key) is not None
        }
        repair_evidence = planning_observation.get("bounded_repair_evidence")
        if isinstance(repair_evidence, list) and repair_evidence:
            planning_summary["bounded_repair_evidence"] = repair_evidence[-1:]
        failure_stage = str(
            planning_observation.get("failure_stage")
            or round_summary.get("failure_stage")
            or ""
        )
        validation_label = (
            "expert_certification_failed"
            if "expert" in failure_stage
            else "pre_policy_validation_failed"
        )
        summary_text += (
            " planning_observation="
            + json.dumps(
                planning_summary,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + f"; {validation_label}; policy_not_executed; N=0."
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
    diagnosis = (
        str(planning_observation.get("diagnosis") or "").strip() or None
        if planning_observation is not None
        else None
    )
    if candidate_outcome == "fail":
        diagnosis = (
            f"Observed policy success_rate={float(success_rate):.6g} for "
            f"{candidate_id} with complete Rule metric "
            f"{packet['rule']['metric']}; this localizes an observed weakness "
            "but does not establish a causal mechanism."
        )
    candidate = {
        "candidate_id": candidate_id,
        "outcome": candidate_outcome,
        "score": (
            float(success_rate) if success_rate is not None else None
        ),
        "diagnosis": diagnosis,
    }
    return {
        "schema_version": 1,
        "round_id": str(round_plan["round_id"]),
        "template_id": str(round_plan.get("template_id") or ""),
        "candidate_id": candidate_id,
        "open_query_evidence": open_query,
        "candidate_evidence": candidate,
        "evaluation_outcome": deepcopy(dict(policy_outcome)),
        "outcome_semantics": outcome_semantics,
        "planned_tool_evidence": planned_tool_evidence,
        "planning_observation": planning_observation,
        "policy_sample_count": (
            0 if planning_observation is not None else None
        ),
        "evidence_packet": packet,
        "evidence_refs": refs,
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
        untested = []
        limitations = [
            "No property attribution is allowed without a passing control.",
            "The observed control result may reflect policy, simulator, or pipeline effects.",
        ]
    else:
        answered = bool(assessment.get("evidence_sufficient"))
        stop_reason = str(assessment.get("stop_reason") or "continue")
        verdict = str(assessment.get("claim_verdict") or "inconclusive")
        if answered:
            answer = f"For the completed evidence, the Query verdict is {verdict}."
        else:
            answer = (
                "The bounded evidence does not yet support an answer to the "
                "original Query."
            )
        untested = list(assessment.get("untested_candidate_ids") or [])
        limitations = list(assessment.get("limitations") or [])
    if untested:
        limitations.append(
            "Untested candidates: " + ", ".join(untested)
        )
    limitations.extend(
        [
            "This answer is limited to the bound task, checkpoint, variants, and recorded seeds.",
            "A bounded N-small result is not a broad generalization guarantee.",
        ]
    )
    refs = [
        deepcopy(ref)
        for record in records
        for ref in record.get("evidence_refs", [])
        if isinstance(ref, Mapping)
    ]
    outcome_authorities = [
        deepcopy(record["evaluation_outcome"])
        for record in records
        if isinstance(record.get("evaluation_outcome"), Mapping)
    ]
    non_official = [
        item
        for item in outcome_authorities
        if item.get("official_equivalent") is False
    ]
    if non_official:
        limitations.append(
            "At least one candidate verdict uses generated_check_success; "
            "it must not be interpreted as official benchmark success."
        )
    outcome_semantics = [
        deepcopy(record["outcome_semantics"])
        for record in records
        if isinstance(record.get("outcome_semantics"), Mapping)
    ]
    semantic_conflicts = [
        item for item in outcome_semantics if item.get("status") == "conflict"
    ]
    semantic_extensions = [
        item
        for item in outcome_semantics
        if item.get("status") == "expected_semantic_extension"
    ]
    if semantic_conflicts:
        limitations.append(
            "At least one round has conflicting generated versus "
            "official/core success semantics."
        )
    if semantic_extensions:
        limitations.append(
            "At least one generated checker has not been certified as "
            "official-equivalent; its verdict must be treated as experimental."
        )
    answer_scope = (
        "bounded_experimental_query_semantics"
        if non_official
        else "official_equivalent"
    )
    return {
        "schema_version": 1,
        "original_query": query,
        "answered": answered,
        "answer_scope": answer_scope,
        "official_benchmark_answered": bool(answered and not non_official),
        "stop_reason": stop_reason,
        "claim_type": None,
        "claim_verdict": verdict,
        "answer": answer,
        "tested_candidate_ids": list(
            assessment.get("observed_candidate_ids") or []
        ),
        "untested_candidate_ids": untested,
        "limitations": list(dict.fromkeys(limitations)),
        "evidence_refs": refs,
        "evaluation_outcomes": outcome_authorities,
        "outcome_semantics": outcome_semantics,
        "evidence_conflict": bool(semantic_conflicts),
    }


def _current_planning_evidence(
    observation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return evidence whose round lineage agrees with runtime records."""

    if not isinstance(observation, Mapping):
        raise PlanAgentSessionError("Plan Agent observation must be an object")
    raw_history = observation.get("open_query_evidence_history")
    if not isinstance(raw_history, list):
        raise PlanAgentSessionError(
            "Plan Agent observation has no open_query_evidence_history"
        )
    try:
        history = validate_open_query_evidence(raw_history)
    except PlanAgentError as exc:
        raise PlanAgentSessionError(str(exc)) from exc
    records = observation.get("records")
    if not isinstance(records, list):
        raise PlanAgentSessionError("Plan Agent observation has no records")
    record_round_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PlanAgentSessionError(
                f"Plan Agent observation record {index} must be an object"
            )
        record_round_ids.append(
            _nonempty_text(
                record.get("round_id"),
                f"observation.records[{index}].round_id",
            )
        )
    evidence_round_ids = [item["round_id"] for item in history]
    if evidence_round_ids != record_round_ids:
        raise PlanAgentSessionError(
            "open_query_evidence_history does not align with completed "
            "runtime records"
        )
    return history


__all__ = ["build_plan_agent_evidence_record", "render_query_answer"]
