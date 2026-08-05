"""Compact illustrated report for one paper-level MEA data flow."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .evidence_final import publish_final_evidence
from .evidence_projection import (
    EvidenceReportError,
    _read_json,
    _resolve_child_ids,
    _safe_artifact_id,
    _target_scope,
)
from .evidence_publication import EvidencePublisher, publish_plan_artifacts
from .evidence_rendering import render_final, render_report_header, render_round
from .evidence_round import publish_round_evidence


def write_evidence_report(
    repo_root: str | Path,
    evaluation_dir: str | Path,
    *,
    destination: str | Path | None = None,
    publish: bool = False,
    max_video_bytes: int = 2_000_000,
    include_repair_id: str | None = None,
) -> dict[str, Any]:
    """Write one compact Markdown report and its displayed artifacts."""

    root = Path(repo_root).expanduser().resolve()
    evaluation = Path(evaluation_dir).expanduser().resolve()
    try:
        evaluation.relative_to(root)
    except ValueError as exc:
        raise EvidenceReportError(
            "evaluation_dir must be inside repo_root"
        ) from exc

    manifest = _read_json(evaluation / "manifest.json", required=True)
    plan = _read_json(evaluation / "plan/evaluation_plan.json") or deepcopy(
        manifest.get("plan") if isinstance(manifest.get("plan"), dict) else {}
    )
    summary = _read_json(evaluation / "summary/summary.json")
    evidence = _read_json(evaluation / "summary/evidence_bundle.json")
    canonical_answer = evaluation / "answer/answer.json"
    feedback = (
        _read_json(canonical_answer)
        if canonical_answer.is_file()
        else _read_json(evaluation / "feedback/feedback.json")
    )
    session = _read_json(evaluation / "plan/bound_task_session.json")

    requested_report = (
        Path(destination).expanduser()
        if destination is not None
        else evaluation / "evidence_report.md"
    )
    report_path = requested_report.resolve()
    if requested_report.is_symlink() or requested_report.absolute() != report_path:
        raise EvidenceReportError("report destination must not use symlinks")
    try:
        report_path.relative_to(root)
    except ValueError as exc:
        raise EvidenceReportError("destination must be inside repo_root") from exc

    publisher = EvidencePublisher(
        root=root,
        bundle_root=report_path.parent,
        publish=publish,
    )
    publisher.prepare()
    rounds = [
        deepcopy(item)
        for item in plan.get("rounds") or []
        if isinstance(item, dict)
    ]
    round_summaries = [
        deepcopy(item)
        for item in summary.get("rounds") or evidence.get("rounds") or []
        if isinstance(item, dict)
    ]
    child_ids = _resolve_child_ids(manifest, round_summaries or rounds)
    safe_round_ids = [
        _safe_artifact_id(
            item.get("round_id") or f"round_{index}",
            label="round_id",
        )
        for index, item in enumerate(rounds, start=1)
    ]
    safe_child_ids = [
        _safe_artifact_id(item, label="child_id")
        if item is not None
        else None
        for item in child_ids
    ]
    decisions = [
        item
        for item in plan.get("round_decisions") or []
        if isinstance(item, dict)
    ]
    query = (
        session.get("user_query")
        or manifest.get("user_request")
        or evidence.get("user_request")
        or "N/A"
    )
    raw_target = (
        session.get("target")
        if isinstance(session.get("target"), dict)
        else {
            "binding_mode": "single_task_single_checkpoint",
            "task_name": manifest.get("task_name"),
            "task_profile": manifest.get("task_profile"),
            "policy": (
                plan.get("policy")
                if isinstance(plan.get("policy"), dict)
                else None
            ),
            "checkpoint": None,
        }
    )
    target = _target_scope(raw_target)

    publish_plan_artifacts(
        publisher,
        evaluation=evaluation,
        round_count=len(rounds),
    )
    round_lines: list[str] = []
    compact_rounds: list[dict[str, Any]] = []
    for index, round_plan in enumerate(rounds, start=1):
        round_id = safe_round_ids[index - 1]
        round_summary = (
            round_summaries[index - 1]
            if index - 1 < len(round_summaries)
            else _read_json(evaluation / "summary" / f"{round_id}.json")
        )
        child_id = (
            safe_child_ids[index - 1]
            if index - 1 < len(safe_child_ids)
            else None
        )
        published_round = publish_round_evidence(
            publisher=publisher,
            evaluation=evaluation,
            round_id=round_id,
            round_plan=round_plan,
            round_summary=round_summary,
            child_id=child_id,
            next_decision=(
                decisions[index - 1]
                if index - 1 < len(decisions)
                else None
            ),
            max_video_bytes=max_video_bytes,
        )
        compact_rounds.append(published_round.compact)
        round_lines.extend(
            render_round(
                index=index,
                round_id=round_id,
                round_plan=round_plan,
                target=target,
                compact=published_round.compact,
                proposal_copy=published_round.proposal_copy,
                taskgen_destination=published_round.taskgen_destination,
                code_copy=published_round.code_copy,
                variant_copy=published_round.variant_copy,
                scene_copy=published_round.scene_copy,
                video_copy=published_round.video_copy,
                video_source=published_round.video_source,
                tool_code_copy=published_round.tool_code_copy,
                montage_copy=published_round.montage_copy,
                report_path=report_path,
            )
        )

    episode_counts = [
        (
            len(item.get("seeds") or [])
            if item.get("policy_success") is not None
            else 0
        )
        for item in compact_rounds
    ]
    lines = render_report_header(
        evaluation_id=manifest.get("evaluation_id"),
        evaluation_name=evaluation.name,
        query=query,
        target=target,
        session=session,
        plan=plan,
        rounds=rounds,
        episode_counts=episode_counts,
        semantic_dir=publisher.semantic_dir,
        report_path=report_path,
    )
    lines.extend(round_lines)

    final = publish_final_evidence(
        publisher=publisher,
        evaluation=evaluation,
        rounds=rounds,
        feedback=feedback,
        manifest=manifest,
        query=query,
        target=target,
        plan=plan,
        session=session,
        summary=summary,
        compact_rounds=compact_rounds,
        include_repair_id=include_repair_id,
    )
    lines.extend(
        render_final(
            final_payload=final.final_payload,
            alignment_reaudit_summary=final.alignment_reaudit_summary,
            run_summary_path=final.run_summary_path,
            repair_result=final.repair_result,
            reuse_summary=final.reuse_summary,
            report_path=report_path,
            evaluation=evaluation,
            root=root,
        )
    )
    return publisher.finish(
        report_path=report_path,
        report_lines=lines,
        run_summary_path=final.run_summary_path,
        manifest=manifest,
        evaluation=evaluation,
        round_count=len(compact_rounds),
        max_video_bytes=max_video_bytes,
        include_repair_id=include_repair_id,
    )


__all__ = ["EvidenceReportError", "write_evidence_report"]
