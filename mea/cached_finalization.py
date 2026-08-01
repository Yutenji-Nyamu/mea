"""Finish an evaluation from immutable cached evidence without new rollouts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from mea.feedback import (
    PlanAgentFinalSummary,
    render_evaluation_report,
    write_evidence_report,
)
from mea.plan_agent_application import update_manifest


class CachedFinalizationError(RuntimeError):
    """Raised when an evaluation is not at a safe final-answer boundary."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CachedFinalizationError(f"cannot read cached artifact: {path}") from exc
    if not isinstance(value, dict):
        raise CachedFinalizationError(f"cached artifact must be an object: {path}")
    return value


def _validate_cached_boundary(
    evaluation_id: str,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    for label, artifact in (("manifest", manifest), ("summary", summary)):
        if artifact.get("evaluation_id") != evaluation_id:
            raise CachedFinalizationError(
                f"{label} evaluation_id does not match {evaluation_id}"
            )
    evidence_id = evidence.get("evaluation_id")
    if evidence_id is not None and evidence_id != evaluation_id:
        raise CachedFinalizationError(
            f"evidence evaluation_id does not match {evaluation_id}"
        )
    if not isinstance(summary.get("status"), str):
        raise CachedFinalizationError("cached summary has no terminal status")
    if not isinstance(evidence.get("rounds"), list) or not evidence["rounds"]:
        raise CachedFinalizationError("cached evidence contains no completed round")
    if manifest.get("lifecycle_status") == "completed":
        raise CachedFinalizationError("evaluation is already completed")
    failure_stage = manifest.get("failure_stage")
    allowed_stages = {
        None,
        "evaluation_aggregation",
        "final_answer",
        "cached_evidence_final_answer",
    }
    if failure_stage not in allowed_stages:
        raise CachedFinalizationError(
            "evaluation failed before the cached final-answer boundary: "
            f"{failure_stage}"
        )


def finalize_cached_evaluation(
    repo_root: str | Path,
    evaluation_id: str,
    *,
    provider: Any,
    feedback_model: str,
) -> dict[str, Any]:
    """Generate only final answer/report artifacts from completed evidence.

    The function never constructs a RoundExecutor or calls a policy backend.
    Existing round, Aggregate, summary, and evidence artifacts are read-only.
    """

    root = Path(repo_root).expanduser().resolve()
    evaluation_dir = root / "mea" / "evaluation_runs" / evaluation_id
    manifest = _read_json(evaluation_dir / "manifest.json")
    summary = _read_json(evaluation_dir / "summary" / "summary.json")
    evidence = _read_json(evaluation_dir / "summary" / "evidence_bundle.json")
    _validate_cached_boundary(evaluation_id, manifest, summary, evidence)

    original_failure = manifest.get("failure")
    update_manifest(
        evaluation_dir,
        status="generating_answer",
        lifecycle_status="finalizing_cached_evidence",
        failure=None,
        failure_stage=None,
        cached_finalization={
            "rollouts_executed": 0,
            "source_summary": "summary/summary.json",
            "source_evidence": "summary/evidence_bundle.json",
            "original_failure": original_failure,
        },
    )
    try:
        feedback = PlanAgentFinalSummary(
            root,
            provider,
            model=feedback_model,
        ).generate(evidence, output_dir=evaluation_dir / "answer")
        report_path = evaluation_dir / "evaluation_report.md"
        report_path.write_text(
            render_evaluation_report(evidence, feedback),
            encoding="utf-8",
        )
        update_manifest(
            evaluation_dir,
            status=summary["status"],
            lifecycle_status="completed",
            execution_finished_at=datetime.now().astimezone().isoformat(),
            answer_path="answer/answer.json",
            report_path="evaluation_report.md",
            answer=feedback,
            summary=summary,
            flagship_acceptance=summary.get("flagship_acceptance"),
            failure=None,
            failure_stage=None,
        )
        report_bundle = write_evidence_report(
            root,
            evaluation_dir,
            destination=evaluation_dir / "evidence_report.md",
        )
        update_manifest(
            evaluation_dir,
            evidence_report_path="evidence_report.md",
            evidence_report_bundle=report_bundle,
            history_index=(
                {"status": "disabled"}
                if manifest.get("history_retrieval_status") == "disabled"
                else manifest.get("history_index", {"status": "not_indexed"})
            ),
        )
    except Exception as exc:
        update_manifest(
            evaluation_dir,
            status="failed",
            lifecycle_status="failed",
            failure_stage="cached_evidence_final_answer",
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise

    return {
        "evaluation_id": evaluation_id,
        "status": summary["status"],
        "lifecycle_status": "completed",
        "rollouts_executed": 0,
        "answer_path": "answer/answer.json",
        "report_path": "evaluation_report.md",
        "evidence_report_path": "evidence_report.md",
    }
