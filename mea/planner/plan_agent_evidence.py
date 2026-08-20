"""Evidence projection and answer rendering for the Plan Agent."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.artifact_retrieval_index import resolve_task_retrieval_index

from .evidence_policy import (
    _GENERATED_POLICY_AUTHORITY,
    _OFFICIAL_POLICY_AUTHORITIES,
    validate_round_evidence,
)
from .plan_agent_schema import PlanAgentError, validate_open_query_evidence
from .plan_agent_errors import PlanAgentSessionError
from .query_interpretation import _nonempty_text


def _policy_outcome_is_decidable(
    policy: Mapping[str, Any],
    outcome_semantics_status: str,
) -> bool:
    """Return whether one completed rate has a trusted success identity."""

    metric = policy.get("metric")
    authority = policy.get("authority")
    official_equivalent = policy.get("official_equivalent")
    if metric == "official_check_success":
        return bool(
            authority in _OFFICIAL_POLICY_AUTHORITIES
            and official_equivalent is True
            and outcome_semantics_status == "official_only"
        )
    if metric != "generated_check_success":
        return False
    if authority != _GENERATED_POLICY_AUTHORITY:
        return False
    return bool(
        (
            official_equivalent is True
            and outcome_semantics_status == "equivalent_agreement"
        )
        or (
            official_equivalent is False
            and outcome_semantics_status == "expected_semantic_extension"
        )
    )


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
    round_evidence_path = str(
        explicit_artifacts.get("round_evidence") or ""
    ).strip()
    if not round_evidence_path and execution_dir:
        round_evidence_path = f"{execution_dir}/round_evidence.json"
    if round_evidence_path and isinstance(
        observations.get("round_evidence"), Mapping
    ):
        refs.append(
            {
                "kind": "round_evidence",
                "path": round_evidence_path,
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


def build_plan_agent_evidence_record(
    round_plan: Mapping[str, Any],
    round_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive compact semantic/query evidence from one completed runtime round."""

    if round_plan.get("round_id") != round_summary.get("round_id"):
        raise PlanAgentSessionError("round plan and summary ids disagree")
    candidate_id = _round_candidate_id(round_plan)
    observations = round_summary.get("observations")
    observations = observations if isinstance(observations, Mapping) else {}
    raw_round_evidence = observations.get("round_evidence")
    if not isinstance(raw_round_evidence, Mapping):
        raise PlanAgentSessionError(
            "completed round has no typed RoundEvidence"
        )
    evidence = validate_round_evidence(raw_round_evidence)
    if evidence["round_id"] != str(round_plan["round_id"]):
        raise PlanAgentSessionError(
            "RoundEvidence round_id differs from the executed plan"
        )
    if evidence["candidate_id"] != candidate_id:
        raise PlanAgentSessionError(
            "RoundEvidence candidate_id differs from the executed plan"
        )
    refs = _round_artifact_refs(round_summary)
    planning_observation = deepcopy(evidence["planning_observation"])
    policy = evidence["policy"]
    policy_outcome = {
        "metric": policy["metric"],
        "authority": policy["authority"],
        "value": policy["success_rate"],
        "official_equivalent": policy["official_equivalent"],
        "execution_scope": policy["execution_scope"],
    }
    outcome_semantics = deepcopy(evidence["outcome_semantics"])
    outcome_semantics_status = str(
        outcome_semantics.get("status") or "non_comparable"
    )
    success_rate = policy["success_rate"]
    policy_outcome_decidable = _policy_outcome_is_decidable(
        policy,
        outcome_semantics_status,
    )
    rule_ready = bool(
        not evidence["rule"]["requested"]
        or (
            evidence["rule"]["status"] == "passed"
            and evidence["rule"]["results"]
        )
    )
    vqa_ready = bool(
        not evidence["vqa"]["required"]
        or evidence["vqa"]["status"] == "passed"
    )
    evidence_conflict = bool(
        outcome_semantics_status == "conflict"
        or outcome_semantics.get("evidence_conflict") is True
        or evidence["vqa"]["evidence_conflict"]
    )
    if planning_observation is not None:
        semantic_outcome = "ambiguous"
        candidate_outcome = "unknown"
    elif evidence_conflict:
        semantic_outcome = "ambiguous"
        candidate_outcome = "conflict"
    elif not policy_outcome_decidable:
        semantic_outcome = "ambiguous"
        candidate_outcome = "unknown"
    elif (
        not evidence["pipeline"]["passed"]
        or not rule_ready
        or not vqa_ready
        or success_rate is None
    ):
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
    if not evidence["pipeline"]["passed"]:
        limitations.append(
            "The execution pipeline did not complete"
            + (
                f" at {evidence['pipeline']['failure_stage']}."
                if evidence["pipeline"]["failure_stage"]
                else "."
            )
        )
    if evidence["rule"]["requested"] and not rule_ready:
        limitations.append(
            "The requested Rule evidence did not produce a passed result set."
        )
    if evidence["vqa"]["required"] and not vqa_ready:
        limitations.append(
            "The requested VQA evidence status is "
            f"{evidence['vqa']['status']}."
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
    if evidence["vqa"]["evidence_conflict"]:
        limitations.append(
            "VQA evidence conflicts with the recorded runtime evidence; this "
            "round cannot support an answered Plan Agent stop."
        )
    planned_tool_evidence = deepcopy(evidence["rule"]["results"])
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
        f"RoundEvidence pipeline_passed={evidence['pipeline']['passed']}; "
        f"candidate_outcome={semantic_outcome}; "
        f"success_predicate_metric={policy_outcome.get('metric')}; "
        f"success_predicate_value={policy_outcome.get('value')}; "
        f"success_predicate_authority={policy_outcome.get('authority')}; "
        f"success_predicate_semantics={outcome_semantics_status}; "
        f"policy_success_rate={success_rate}; "
        f"Rule status={evidence['rule']['status']}; "
        f"Rule metric={evidence['rule']['metric']}; "
        f"VQA status={evidence['vqa']['status']}; "
        f"VQA evidence_conflict={evidence['vqa']['evidence_conflict']}; "
        "diagnostic_tool_role=supporting_measurement_not_success_authority; "
        f"diagnostic_tool_measurements={tool_summary}."
    )
    scene_change = evidence["scene_change"]
    if (
        isinstance(scene_change, Mapping)
        and isinstance(scene_change.get("tracked_actor_changes"), list)
        and scene_change["tracked_actor_changes"]
    ):
        summary_text += (
            " simulator_scene_change="
            + json.dumps(
                scene_change,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "."
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
        rule_metric = evidence["rule"]["metric"]
        diagnosis = (
            f"Observed policy success_rate={float(success_rate):.6g} for "
            f"{candidate_id}"
            + (
                f" with Rule metric {rule_metric}"
                if rule_metric is not None
                else ""
            )
            + "; this localizes an observed weakness "
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
        "round_evidence": evidence,
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
    candidate_conflicts = [
        record
        for record in records
        if isinstance(record.get("candidate_evidence"), Mapping)
        and record["candidate_evidence"].get("outcome") == "conflict"
    ]
    vqa_conflicts = [
        record
        for record in records
        if isinstance(record.get("round_evidence"), Mapping)
        and isinstance(record["round_evidence"].get("vqa"), Mapping)
        and record["round_evidence"]["vqa"].get("evidence_conflict") is True
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
    if vqa_conflicts:
        limitations.append(
            "At least one round has VQA evidence that conflicts with its "
            "recorded runtime evidence."
        )
    elif candidate_conflicts and not semantic_conflicts:
        limitations.append(
            "At least one round contains conflicting runtime evidence."
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
        "evidence_conflict": bool(semantic_conflicts or candidate_conflicts),
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
