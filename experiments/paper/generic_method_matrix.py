"""Audit one frozen cross-task ManipEvalAgent method batch.

This script is deliberately read-only with respect to evaluation bundles.  It
separates method completion, policy outcomes, and answer sufficiency instead
of collapsing them into one success flag.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _answer_path(evaluation_dir: Path, manifest: Mapping[str, Any]) -> Path | None:
    stop = manifest.get("plan_agent_stop")
    if isinstance(stop, Mapping) and isinstance(stop.get("answer_path"), str):
        candidate = evaluation_dir / str(stop["answer_path"])
        if candidate.is_file():
            return candidate
    for relative in (
        "plan/plan_agent_session/query_answer.json",
        "plan/claim_first_runtime/query_answer.json",
    ):
        candidate = evaluation_dir / relative
        if candidate.is_file():
            return candidate
    return None


def _child_runs(
    evaluation_dir: Path, repo_root: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    children: list[dict[str, Any]] = []
    errors: list[str] = []
    for record_path in sorted((evaluation_dir / "execution").glob("*/child_run.json")):
        record = _load_json(record_path)
        relative_record = record_path.relative_to(repo_root).as_posix()
        if record is None:
            errors.append(f"invalid child record: {relative_record}")
            continue
        manifest_path = record.get("manifest_path")
        if not isinstance(manifest_path, str):
            errors.append(f"child record lacks manifest_path: {relative_record}")
            continue
        child_path = Path(manifest_path)
        if not child_path.is_absolute():
            child_path = repo_root / child_path
        child = _load_json(child_path)
        if child is None:
            errors.append(f"missing or invalid child manifest: {manifest_path}")
            continue
        children.append(
            {
                "record": record,
                "manifest": child,
                "round_id": record_path.parent.name,
            }
        )
    return children, errors


def _round_summaries(evaluation_dir: Path) -> list[dict[str, Any]]:
    return [
        summary
        for path in sorted((evaluation_dir / "summary").glob("round_*.json"))
        for summary in [_load_json(path)]
        if summary is not None
    ]


def _canonical_plan_steps(evaluation_dir: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for path in sorted(
        (evaluation_dir / "plan/plan_agent_steps").glob("*/bound_semantic_step.json")
    ):
        step = _load_json(path)
        if step is not None:
            steps.append(step)
    return steps


def _last_authority(
    evaluation_dir: Path,
    *,
    manifest: Mapping[str, Any] | None,
    answer: Mapping[str, Any] | None,
) -> str:
    if manifest is None:
        return "launch"
    failure = manifest.get("failure")
    if isinstance(failure, Mapping):
        failure_type = str(failure.get("type") or "").lower()
        if "taskgen" in failure_type or "taskgeneration" in failure_type:
            return "taskgen"
        if manifest.get("failure_stage"):
            return "round_execution"
    if answer is not None:
        return "answer"
    if list((evaluation_dir / "summary").glob("round_*.json")):
        return "aggregate"
    if list((evaluation_dir / "execution").glob("*/child_run.json")):
        return "round_execution"
    if (evaluation_dir / "plan/evaluation_plan.json").is_file():
        return "planning"
    return "route_binding"


def _policy_observations(
    children: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for child in children:
        manifest = child["manifest"]
        evaluation = manifest.get("act_evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        seeds = list(evaluation.get("actual_seeds") or [])
        if not seeds:
            continue
        generation_kind = manifest.get("generation_kind")
        observations.append(
            {
                "round_id": child["round_id"],
                "scope": (
                    "official"
                    if generation_kind == "official_passthrough"
                    else "generated"
                ),
                "pipeline_passed": evaluation.get("passed") is not False,
                "outcome": evaluation.get("outcome_value"),
                "seeds": seeds,
            }
        )
    return observations


def _method_failures(
    manifest: Mapping[str, Any] | None,
    children: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    artifact_errors: list[str],
) -> list[str]:
    failures = list(artifact_errors)
    if manifest is None:
        failures.append("missing evaluation manifest")
        return failures
    status = str(manifest.get("status") or "missing")
    lifecycle = str(manifest.get("lifecycle_status") or "")
    if isinstance(manifest.get("failure"), Mapping):
        failures.append("evaluation manifest records failure")
        failure = manifest["failure"]
        failures.append(
            "failure="
            + str(failure.get("type") or "unknown")
            + ": "
            + str(failure.get("message") or "no message")
        )
    if status.startswith("failed") or status in {"registration_failed", "missing"}:
        failures.append(f"evaluation status={status}")
    if lifecycle and lifecycle not in {"completed", "finalized"}:
        failures.append(f"lifecycle_status={lifecycle}")
    planning_statuses = {
        "candidate_unexecutable",
        "taskgen_materialization_failed",
        "unsupported",
    }
    for child in children:
        record = child["record"]
        child_manifest = child["manifest"]
        record_status = str(record.get("status") or "")
        child_status = str(child_manifest.get("status") or "")
        if record.get("returncode") not in {None, 0}:
            failures.append(f"{child['round_id']} returncode={record.get('returncode')}")
        if record_status not in {"", "completed", *planning_statuses}:
            failures.append(f"{child['round_id']} record_status={record_status}")
        if child_status not in {"", "completed", *planning_statuses}:
            failures.append(f"{child['round_id']} child_status={child_status}")
    for summary in summaries:
        observations = summary.get("observations")
        if isinstance(observations, Mapping) and observations.get("act_pipeline_status") is False:
            failures.append(f"{summary.get('round_id')} act_pipeline_status=false")
    return sorted(set(failures))


def summarize_evaluation(
    *,
    repo_root: Path,
    task: str,
    task_run_dir: Path,
) -> dict[str, Any]:
    evaluation_id_path = task_run_dir / "evaluation_id.txt"
    evaluation_id = (
        evaluation_id_path.read_text(encoding="utf-8").strip()
        if evaluation_id_path.is_file()
        else ""
    )
    evaluation_dir = repo_root / "mea/evaluation_runs" / evaluation_id
    manifest = _load_json(evaluation_dir / "manifest.json")
    answer_path = _answer_path(evaluation_dir, manifest or {})
    answer = _load_json(answer_path) if answer_path is not None else None
    children, artifact_errors = _child_runs(evaluation_dir, repo_root)
    summaries = _round_summaries(evaluation_dir)
    plan_steps = _canonical_plan_steps(evaluation_dir)
    policy_observations = _policy_observations(children)
    failures = _method_failures(manifest, children, summaries, artifact_errors)

    bounded_child_statuses = {
        None,
        "completed",
        "candidate_unexecutable",
        "taskgen_materialization_failed",
        "unsupported",
    }
    bounded_children = [
        child
        for child in children
        if child["record"].get("returncode") in {None, 0}
        and child["record"].get("status") in bounded_child_statuses
        and child["manifest"].get("status") in bounded_child_statuses
    ]
    completed_children = [
        child
        for child in bounded_children
        if child["record"].get("status") in {None, "completed"}
        and child["manifest"].get("status") in {None, "completed"}
    ]
    generated_children = [
        child
        for child in completed_children
        if child["manifest"].get("status") in {None, "completed"}
        and child["manifest"].get("generation_kind")
        not in {None, "official_passthrough", "unsupported", "candidate_unexecutable"}
    ]
    rejected_round_count = sum(
        child["record"].get("status") in {"unsupported", "candidate_unexecutable"}
        or child["manifest"].get("status") in {"unsupported", "candidate_unexecutable"}
        for child in children
    )
    failure = (manifest or {}).get("failure")
    materialization_failure_count = sum(
        child["manifest"].get("status") == "taskgen_materialization_failed"
        for child in children
    ) + int(
        isinstance(failure, Mapping)
        and (
            "taskgen" in str(failure.get("type") or "").lower()
            or "taskgeneration" in str(failure.get("type") or "").lower()
        )
    )
    evidence_conditioned_steps = sum(
        isinstance(step.get("semantic_proposal_bundle"), Mapping)
        and isinstance(
            step["semantic_proposal_bundle"].get("planning_lineage"), Mapping
        )
        and step["semantic_proposal_bundle"]["planning_lineage"].get(
            "evidence_conditioned"
        )
        is True
        for step in plan_steps
    )
    sub_aspects = sorted(
        {
            str(value)
            for step in plan_steps
            for value in [
                (
                    step.get("semantic_proposal_bundle", {})
                    .get("proposal", {})
                    .get("sub_aspect")
                )
            ]
            if value
        }
        | {
            str(summary["sub_aspect"])
            for summary in summaries
            if summary.get("sub_aspect")
        }
    )
    exact_reuse_rounds = [
        str(summary.get("round_id"))
        for summary in summaries
        if isinstance(summary.get("observations"), Mapping)
        and isinstance(summary["observations"].get("planned_tool"), Mapping)
        and summary["observations"]["planned_tool"].get("status") == "passed"
        and summary["observations"]["planned_tool"].get("route")
        == "run_local_reuse"
    ]
    stop = (manifest or {}).get("plan_agent_stop")
    stop = dict(stop) if isinstance(stop, Mapping) else {}
    answered = bool((answer or {}).get("answered"))
    evidence_sufficient = stop.get("evidence_sufficient") is True
    if failures:
        method_status = "method_system_failure"
    elif answer is None:
        method_status = "incomplete_without_answer"
    elif answered and evidence_sufficient:
        method_status = "completed_supported_answer"
    else:
        method_status = "completed_inconclusive_answer"

    schema_origins = sorted(
        {
            str(context["schema_origin"])
            for child in completed_children
            for context in [child["manifest"].get("runtime_task_context")]
            if isinstance(context, Mapping)
            and isinstance(context.get("schema_origin"), str)
        }
    )
    negative_scopes = sorted(
        {
            str(observation["scope"])
            for observation in policy_observations
            if observation["pipeline_passed"] and observation["outcome"] is False
        }
    )
    return {
        "task": task,
        "evaluation_id": evaluation_id or None,
        "method_status": method_status,
        "manifest_status": (manifest or {}).get("status"),
        "failure_stage": (manifest or {}).get("failure_stage"),
        "method_failure_reasons": failures,
        "last_authoritative_stage": _last_authority(
            evaluation_dir, manifest=manifest, answer=answer
        ),
        "policy_rollouts": sum(len(item["seeds"]) for item in policy_observations),
        "policy_observations": policy_observations,
        "policy_negative_observed": bool(negative_scopes),
        "policy_negative_scopes": negative_scopes,
        "bounded_rounds": len(bounded_children),
        "completed_rounds": len(completed_children),
        "generated_rounds": len(generated_children),
        "rejected_rounds": rejected_round_count,
        "materialization_failures": materialization_failure_count,
        "runtime_schema_origins": schema_origins,
        "distinct_sub_aspects": sub_aspects,
        "evidence_conditioned_steps": evidence_conditioned_steps,
        "agent_stop_proposed": stop.get("plan_agent_stop_proposed"),
        "answered": answered,
        "evidence_sufficient": stop.get("evidence_sufficient"),
        "stop_reason": stop.get("stop_reason") or (answer or {}).get("stop_reason"),
        "claim_verdict": (answer or {}).get("claim_verdict"),
        "exact_tool_reuse_rounds": exact_reuse_rounds,
        "answer_path": (
            answer_path.relative_to(repo_root).as_posix()
            if answer_path is not None
            else None
        ),
    }


def build_batch_summary(
    repo_root: Path, batch_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
        raise ValueError("batch config tasks must be a list of task names")
    evaluations = [
        summarize_evaluation(
            repo_root=repo_root,
            task=task,
            task_run_dir=batch_root / task,
        )
        for task in tasks
    ]
    statuses = Counter(item["method_status"] for item in evaluations)
    last_stages = Counter(item["last_authoritative_stage"] for item in evaluations)
    return {
        "schema_version": 2,
        "batch_id": config.get("batch_id") or batch_root.name,
        "source_commit": config.get("source_commit"),
        "evaluation_count": len(evaluations),
        "policy_rollout_count": sum(item["policy_rollouts"] for item in evaluations),
        "generated_round_count": sum(item["generated_rounds"] for item in evaluations),
        "rejected_round_count": sum(item["rejected_rounds"] for item in evaluations),
        "materialization_failure_count": sum(
            item["materialization_failures"] for item in evaluations
        ),
        "method_status_counts": dict(sorted(statuses.items())),
        "last_authoritative_stage_counts": dict(sorted(last_stages.items())),
        "policy_negative_task_count": sum(
            item["policy_negative_observed"] for item in evaluations
        ),
        "evaluations": evaluations,
        "claim_boundary": (
            "Cross-task N=1 method characterization; method completion, policy "
            "outcomes, and answer sufficiency are orthogonal. This batch does not "
            "establish benchmark generalization or reproduce a paper table."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _load_json(args.config.expanduser().resolve())
    if config is None:
        raise ValueError("batch config must be a valid JSON object")
    summary = build_batch_summary(
        args.repo_root.expanduser().resolve(),
        args.batch_root.expanduser().resolve(),
        config,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
