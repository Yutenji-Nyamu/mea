"""Compact, illustrated report for the paper-level MEA data flow.

The normal evaluation report remains the complete machine audit.  This module
selects the small set of real artifacts a user needs to inspect the method:
query, plan, generated/reused task, render, rollout, Tool/VQA evidence,
Aggregate, next-round decision, and final answer.  It never fabricates a
missing image, code file, metric, or model answer.
"""

from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


class EvidenceReportError(RuntimeError):
    """Raised when an evaluation cannot be represented without guessing."""


def _read_json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise EvidenceReportError(f"required JSON artifact is missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceReportError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceReportError(f"JSON artifact must contain an object: {path}")
    return value


def _repo_file(repo_root: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        resolved = candidate.expanduser().resolve()
        resolved.relative_to(repo_root)
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _relative_link(path: Path, report_path: Path) -> str:
    return Path(os.path.relpath(path, report_path.parent)).as_posix()


def _copy_file(
    source: Path | None,
    destination: Path,
    *,
    max_bytes: int | None = None,
) -> Path | None:
    if source is None or not source.is_file():
        return None
    if max_bytes is not None and source.stat().st_size > max_bytes:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def _code_excerpt(path: Path, *, max_lines: int = 80, max_chars: int = 9000) -> str:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""
    source_lines = source.splitlines()
    lines = source_lines[:max_lines]
    excerpt = "\n".join(lines)
    truncated = len(source_lines) > max_lines or len(excerpt) > max_chars
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rsplit("\n", 1)[0]
    if truncated:
        excerpt += "\n# ... truncated; open the linked artifact for the full source"
    return excerpt


def _quote(value: Any) -> list[str]:
    text = str(value or "N/A").strip() or "N/A"
    return [f"> {line}" if line else ">" for line in text.splitlines()]


def _compact_tool_rows(tool: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in tool.get("episodes") or []:
        if not isinstance(episode, Mapping):
            continue
        result = episode.get("result") if isinstance(episode.get("result"), Mapping) else {}
        rows.append(
            {
                "role": episode.get("role"),
                "policy_name": episode.get("policy_name"),
                "seed": episode.get("seed"),
                "value": result.get("value"),
                "unit": result.get("unit"),
                "passed": result.get("passed"),
            }
        )
    return rows


def _compact_vqa(vqa: Mapping[str, Any]) -> dict[str, Any]:
    query = vqa.get("query") if isinstance(vqa.get("query"), Mapping) else {}
    observation = (
        vqa.get("observation")
        if isinstance(vqa.get("observation"), Mapping)
        else {}
    )
    return {
        "status": vqa.get("status"),
        "questions": [
            {"id": item.get("id"), "question": item.get("question")}
            for item in query.get("questions") or []
            if isinstance(item, Mapping)
        ],
        "phenomena": [
            {
                "id": item.get("id"),
                "observed": item.get("observed"),
                "description": item.get("description"),
                "confidence": item.get("confidence"),
                "frame_ids": item.get("frame_ids"),
            }
            for item in observation.get("phenomena") or []
            if isinstance(item, Mapping)
        ],
        "numeric_consistency": observation.get("numeric_consistency"),
        "evidence_conflict": vqa.get("evidence_conflict"),
    }


def _resolve_child_ids(manifest: Mapping[str, Any], rounds: list[dict[str, Any]]) -> list[str | None]:
    ids = list(manifest.get("child_run_ids") or [])
    result: list[str | None] = []
    for index, round_plan in enumerate(rounds):
        run_id = ids[index] if index < len(ids) else None
        if run_id is None:
            run_id = round_plan.get("taskgen_run_id")
        result.append(str(run_id) if run_id else None)
    return result


def write_evidence_report(
    repo_root: str | Path,
    evaluation_dir: str | Path,
    *,
    destination: str | Path | None = None,
    publish: bool = False,
    max_video_bytes: int = 2_000_000,
) -> dict[str, Any]:
    """Write one compact Markdown report and copy only its displayed artifacts."""

    root = Path(repo_root).expanduser().resolve()
    evaluation = Path(evaluation_dir).expanduser().resolve()
    try:
        evaluation.relative_to(root)
    except ValueError as exc:
        raise EvidenceReportError("evaluation_dir must be inside repo_root") from exc
    manifest = _read_json(evaluation / "manifest.json", required=True)
    plan = _read_json(evaluation / "plan/evaluation_plan.json") or deepcopy(
        manifest.get("plan") if isinstance(manifest.get("plan"), dict) else {}
    )
    summary = _read_json(evaluation / "summary/summary.json")
    evidence = _read_json(evaluation / "summary/evidence_bundle.json")
    feedback = _read_json(evaluation / "feedback/feedback.json")
    session = _read_json(evaluation / "plan/bound_task_session.json")
    report_path = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else evaluation / "evidence_report.md"
    )
    try:
        report_path.relative_to(root)
    except ValueError as exc:
        raise EvidenceReportError("destination must be inside repo_root") from exc
    report_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_root = report_path.parent
    asset_dir = bundle_root / "assets"
    code_dir = bundle_root / "code"
    data_dir = bundle_root / "data"

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
    target = session.get("target") if isinstance(session.get("target"), dict) else {
        "binding_mode": "single_task_single_checkpoint",
        "task_name": manifest.get("task_name"),
        "task_profile": manifest.get("task_profile"),
        "policy": (plan.get("policy") if isinstance(plan.get("policy"), dict) else None),
        "checkpoint": None,
    }

    lines = [
        f"# MEA method evidence: {manifest.get('evaluation_id', evaluation.name)}",
        "",
        "> This is a compact view of real run artifacts. The complete machine audit remains in the evaluation directory.",
        "",
        "## 1. Query and fixed policy scope",
        "",
        *_quote(query),
        "",
        _json_block(
            {
                "binding_mode": target.get("binding_mode"),
                "task_name": target.get("task_name"),
                "task_profile": target.get("task_profile"),
                "policy": target.get("policy"),
                "checkpoint": target.get("checkpoint"),
                "round_budget": session.get("round_budget") or plan.get("max_rounds"),
                "episodes_per_round": [
                    (item.get("execution") or {}).get("num_episodes") for item in rounds
                ],
            }
        ),
        "",
        "One evaluation keeps this task and ACT checkpoint fixed. Adaptation happens only across this task's sub-aspects/variants.",
        "",
        "## 2. Paper-level data flow",
        "",
        "```mermaid",
        "flowchart LR",
        '  Q["Open Query"] --> P["Plan Agent / sub-aspect"]',
        '  P --> T["TaskGen: reuse or generate"]',
        '  T --> I["Render / visual reflection"]',
        '  I --> E["ACT rollout"]',
        '  E --> V["Rule Tool + dynamic VQA"]',
        '  V --> A["Aggregate"]',
        '  A -->|"evidence"| P',
        '  A --> R["Final answer"]',
        "```",
        "",
        "## 3. Initial decomposition",
        "",
        _json_block(
            {
                "evaluation_goal": plan.get("evaluation_goal"),
                "selected_aspect_ids": (
                    session.get("selected_aspect_ids")
                    or plan.get("requested_aspect_ids")
                ),
                "requested_template_ids": plan.get("requested_template_ids"),
                "first_round": rounds[0].get("round_id") if rounds else None,
                "planning_state": plan.get("planning_state"),
            }
        ),
        "",
    ]

    published_files: list[str] = []
    compact_rounds: list[dict[str, Any]] = []
    for index, round_plan in enumerate(rounds, start=1):
        round_id = str(round_plan.get("round_id") or f"round_{index}")
        round_summary = (
            round_summaries[index - 1]
            if index - 1 < len(round_summaries)
            else _read_json(evaluation / "summary" / f"{round_id}.json")
        )
        child_id = child_ids[index - 1] if index - 1 < len(child_ids) else None
        child = root / "mea/generated_tasks" / child_id if child_id else None
        child_manifest = _read_json(child / "manifest.json") if child else {}
        execution = evaluation / "execution" / round_id
        tool = _read_json(execution / "planned_tool/tool_execution.json")
        vqa = _read_json(execution / "execution_vqa/execution_vqa.json")
        aggregate = _read_json(execution / "aggregate_result.json")

        task_code = child / "task.py" if child and (child / "task.py").is_file() else None
        overlay = child / "overlay.yml" if child and (child / "overlay.yml").is_file() else None
        variant_spec = child / "variant_spec.json" if child and (child / "variant_spec.json").is_file() else None
        display_code = task_code or overlay
        code_copy = _copy_file(
            display_code,
            code_dir / f"{round_id}_{display_code.name}" if display_code else code_dir / "missing",
        )
        if code_copy:
            published_files.append(str(code_copy.relative_to(root)).replace("\\", "/"))
        variant_copy = _copy_file(
            variant_spec,
            data_dir / f"{round_id}_variant_spec.json",
        )
        if variant_copy:
            published_files.append(
                str(variant_copy.relative_to(root)).replace("\\", "/")
            )

        scene_source = None
        if child:
            for candidate in (
                child / "evidence/initial_head.png",
                child / "reflection/attempt_00/render.png",
            ):
                if candidate.is_file():
                    scene_source = candidate
                    break
        scene_copy = _copy_file(scene_source, asset_dir / f"{round_id}_scene.png")
        montage_copy = _copy_file(
            execution / "execution_vqa/execution_montage.png",
            asset_dir / f"{round_id}_vqa_montage.png",
        )
        video_source = None
        if child:
            for candidate in (
                child / "evaluation/episode0.mp4",
                *sorted((child / "evaluation/telemetry/act").glob("*/video.mp4")),
            ):
                if candidate.is_file():
                    video_source = candidate
                    break
        video_copy = _copy_file(
            video_source,
            asset_dir / f"{round_id}_act.mp4",
            max_bytes=max_video_bytes,
        )
        for copied in (scene_copy, montage_copy, video_copy):
            if copied:
                published_files.append(str(copied.relative_to(root)).replace("\\", "/"))

        generated_tool = _repo_file(
            root,
            (tool.get("source") or {}).get("artifact")
            if isinstance(tool.get("source"), dict)
            else None,
        )
        tool_code_copy = _copy_file(
            generated_tool,
            code_dir / f"{round_id}_tool.py",
        )
        if tool_code_copy:
            published_files.append(str(tool_code_copy.relative_to(root)).replace("\\", "/"))

        observations = (
            round_summary.get("observations")
            if isinstance(round_summary.get("observations"), dict)
            else {}
        )
        compact = {
            "round_id": round_id,
            "aspect_id": round_plan.get("aspect_id") or round_plan.get("sub_aspect"),
            "template_id": round_plan.get("template_id"),
            "taskgen_route": round_plan.get("route"),
            "taskgen_kind": child_manifest.get("generation_kind")
            or child_manifest.get("mode"),
            "execution_backend": observations.get("execution_backend"),
            "seeds": (round_plan.get("execution") or {}).get("seeds"),
            "pipeline_passed": round_summary.get("pipeline_passed"),
            "policy_success": observations.get("policy_success"),
            "tool_metric": (tool.get("tool_request") or {}).get("metric"),
            "tool_route": tool.get("route"),
            "tool_rows": _compact_tool_rows(tool),
            "vqa": _compact_vqa(vqa),
            "aggregate_status": aggregate.get("status"),
            "next_decision": decisions[index - 1] if index - 1 < len(decisions) else None,
        }
        compact_rounds.append(compact)
        data_path = data_dir / f"{round_id}.json"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        published_files.append(str(data_path.relative_to(root)).replace("\\", "/"))

        persisted_task_proposal = round_plan.get("task_proposal")
        if isinstance(persisted_task_proposal, Mapping):
            task_proposal_heading = "### Plan -> TaskProposal"
            task_proposal_payload = persisted_task_proposal
        else:
            task_proposal_heading = "### Legacy plan intent"
            task_proposal_payload = {
                "proposal_status": "missing_legacy_projection",
                "task_name": round_plan.get("task_name") or target.get("task_name"),
                "aspect_id": compact["aspect_id"],
                "task_instruction": round_plan.get("task_instruction"),
            }
        persisted_tool_proposal = round_plan.get("tool_proposal")
        if isinstance(persisted_tool_proposal, Mapping):
            tool_proposal_heading = "### ToolProposal -> ToolGen / reuse"
            tool_proposal_payload = persisted_tool_proposal
        else:
            tool_proposal_heading = "### Legacy Tool request"
            tool_proposal_payload = {
                "proposal_status": "missing_legacy_projection",
                "tool_request": tool.get("tool_request") or None,
            }

        lines.extend(
            [
                f"## 4.{index}. {round_id}: {compact['aspect_id']}",
                "",
                task_proposal_heading,
                "",
                _json_block(task_proposal_payload),
                "",
                "### TaskGen output",
                "",
                f"- Route: `{compact['taskgen_route']}`",
                f"- Materialization: `{compact['taskgen_kind'] or 'not recorded'}`",
                f"- Child run: `{child_id or 'N/A'}`",
            ]
        )
        if code_copy:
            language = "python" if code_copy.suffix == ".py" else "yaml"
            lines.extend(
                [
                    f"- Full task artifact: [{code_copy.name}]({_relative_link(code_copy, report_path)})",
                    "",
                    f"```{language}",
                    _code_excerpt(code_copy),
                    "```",
                ]
            )
        else:
            lines.append("- Generated/reused source: N/A (artifact was not present)")
        if variant_copy:
            lines.append(
                f"- VariantSpec: [{variant_copy.name}]({_relative_link(variant_copy, report_path)})"
            )
        if scene_copy:
            lines.extend(
                [
                    "",
                    "### Render / scene check",
                    "",
                    f"![{round_id} initial scene]({_relative_link(scene_copy, report_path)})",
                ]
            )
        else:
            lines.extend(["", "### Render / scene check", "", "N/A - no real scene image was found."])

        lines.extend(
            [
                "",
                "### ACT rollout",
                "",
                _json_block(
                    {
                        "backend": compact["execution_backend"],
                        "seeds": compact["seeds"],
                        "pipeline_passed": compact["pipeline_passed"],
                        "policy_success": compact["policy_success"],
                    }
                ),
            ]
        )
        if video_copy:
            link = _relative_link(video_copy, report_path)
            lines.extend(["", f"[Open ACT video]({link})", "", f'<video src="{link}" controls width="720"></video>'])
        elif video_source is not None:
            lines.append("\nVideo exists in the raw run but exceeded the publish size limit.")
        else:
            lines.append("\nN/A - no ACT video was found.")

        lines.extend(
            [
                "",
                tool_proposal_heading,
                "",
                _json_block(tool_proposal_payload),
                "",
                _json_block(
                    {
                        "route": compact["tool_route"],
                        "metric": compact["tool_metric"],
                        "episodes": compact["tool_rows"],
                    }
                ),
            ]
        )
        if tool_code_copy:
            lines.extend(
                [
                    f"\n[Open generated/reused Tool source]({_relative_link(tool_code_copy, report_path)})",
                    "",
                    "```python",
                    _code_excerpt(tool_code_copy),
                    "```",
                ]
            )
        lines.extend(["", "### Dynamic VQA", "", _json_block(compact["vqa"])])
        if montage_copy:
            lines.extend(
                [
                    "",
                    f"![{round_id} VQA keyframes]({_relative_link(montage_copy, report_path)})",
                ]
            )
        lines.extend(
            [
                "",
                "### Aggregate -> next decision",
                "",
                _json_block(
                    {
                        "aggregate_status": compact["aggregate_status"],
                        "policy_success": compact["policy_success"],
                        "decision": compact["next_decision"],
                    }
                ),
                "",
            ]
        )

    final_payload = {
        "answer": feedback.get("answer"),
        "findings": feedback.get("findings"),
        "recommended_next_step": feedback.get("recommended_next_step"),
        "limitations": feedback.get("limitations"),
    }
    lines.extend(
        [
            "## 5. Final answer to the original Query",
            "",
            *_quote(final_payload["answer"]),
            "",
            _json_block(
                {
                    "findings": final_payload["findings"],
                    "recommended_next_step": final_payload["recommended_next_step"],
                    "limitations": final_payload["limitations"],
                }
            ),
            "",
            "## 6. Boundaries",
            "",
            "- Policy results and pipeline status are reported separately.",
            "- Expert evidence, when present, is a solvability/instrumentation gate, not ACT performance.",
            "- Few-shot N=1 rounds demonstrate method wiring, not benchmark-level generalization.",
            "- Missing artifacts are shown as N/A; this report never substitutes proxy images or invented values.",
            "",
            "## 7. Raw artifact index",
            "",
        ]
    )
    for raw in (
        evaluation / "manifest.json",
        evaluation / "plan/evaluation_plan.json",
        evaluation / "plan/bound_task_session.json",
        evaluation / "summary/evidence_bundle.json",
        evaluation / "feedback/feedback.json",
        evaluation / "evaluation_report.md",
    ):
        if raw.is_file():
            if publish:
                lines.append(
                    f"- Server source: `{raw.relative_to(root).as_posix()}`"
                )
            else:
                lines.append(f"- [{raw.name}]({_relative_link(raw, report_path)})")

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    published_files.append(str(report_path.relative_to(root)).replace("\\", "/"))
    bundle_manifest = {
        "schema_version": 1,
        "evaluation_id": manifest.get("evaluation_id"),
        "source_evaluation": str(evaluation.relative_to(root)).replace("\\", "/"),
        "report": str(report_path.relative_to(root)).replace("\\", "/"),
        "publish_mode": bool(publish),
        "files": sorted(set(published_files)),
        "round_count": len(compact_rounds),
        "video_size_limit_bytes": int(max_video_bytes),
    }
    manifest_path = bundle_root / "evidence_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    bundle_manifest["files"] = sorted(
        set(
            [
                *bundle_manifest["files"],
                str(manifest_path.relative_to(root)).replace("\\", "/"),
            ]
        )
    )
    manifest_path.write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle_manifest


__all__ = ["EvidenceReportError", "write_evidence_report"]
