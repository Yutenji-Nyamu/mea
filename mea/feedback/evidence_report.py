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
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


class EvidenceReportError(RuntimeError):
    """Raised when an evaluation cannot be represented without guessing."""


_PLAN_ARTIFACTS = (
    "evaluation_plan.json", "bound_task_session.json",
    "query_sufficiency_contract.json", "global_query_route.json",
    "free_concern.json", "free_concern_prompt.md",
)

_TASKGEN_ARTIFACTS = (
    "manifest.json", "generation/experiment_candidate.json",
    "generation/code_prompt.md", "generation/provider_response.txt",
    "validation/static.json", "validation/checker_fixtures.json",
    "validation/implementation_trace.json", "validation/setup_preflight.json",
    "validation/expert_preflight.json", "validation/vision.json",
    "validation/vision_prompt.md", "validation/vision_response.txt",
)

_TASKGEN_RENDER_ARTIFACTS = (
    "evidence/official_initial_head.png", "evidence/expert_head.png",
    "evidence/scene_comparison.png",
)

_TOOL_ARTIFACT_ROLES = {
    "tool_request": frozenset({".json"}),
    "route_decision": frozenset({".json"}),
    "resolved_tool_spec": frozenset({".json"}),
    "toolgen_manifest": frozenset({".json"}),
    "registration": frozenset({".json"}),
    "property_validation": frozenset({".json"}),
    "metric_spec_execution": frozenset({".json"}),
    "run_local_registration": frozenset({".json"}),
    "review_manifest": frozenset({".json"}),
}

_FINAL_ARTIFACTS = (
    ("feedback/feedback.json", "answer/feedback.json"),
    ("evaluation_report.md", "answer/evaluation_report.md"),
    ("summary/aggregate_result.json", "aggregate/final.json"),
    ("summary/summary.json", "audit/summary.json"),
    ("summary/evidence_bundle.json", "audit/evidence_bundle.json"),
    ("manifest.json", "audit/source_manifest.json"),
    ("plan/semantic_preservation_audit.json",
     "audit/semantic_preservation_audit.json"),
    ("summary/semantic_preservation_audit.json",
     "audit/summary_semantic_preservation_audit.json"),
    ("audit/final_audit.json", "audit/final_audit.json"),
    ("audit/protocol_audit.json", "audit/protocol_audit.json"),
)

_SAFE_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_MAX_PUBLIC_ARTIFACT_BYTES = 5_000_000


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


def _safe_artifact_id(raw: Any, *, label: str) -> str:
    value = str(raw)
    if value in {".", ".."} or _SAFE_ARTIFACT_ID.fullmatch(value) is None:
        raise EvidenceReportError(f"{label} is not a safe artifact id: {value!r}")
    return value


def _target_scope(target: Mapping[str, Any]) -> dict[str, Any]:
    """Project legacy or schema-v3 targets into one report-only scope."""

    binding = target.get("policy_task_binding")
    if isinstance(binding, Mapping):
        return {
            "binding_mode": target.get("binding_mode"),
            "task_name": binding.get("task_name"),
            "task_profile": None,
            "policy": deepcopy(
                dict(binding.get("policy"))
                if isinstance(binding.get("policy"), Mapping)
                else {}
            ),
            "checkpoint": deepcopy(
                dict(binding.get("checkpoint"))
                if isinstance(binding.get("checkpoint"), Mapping)
                else {}
            ),
        }
    return {
        "binding_mode": target.get("binding_mode"),
        "task_name": target.get("task_name"),
        "task_profile": target.get("task_profile"),
        "policy": deepcopy(
            dict(target.get("policy"))
            if isinstance(target.get("policy"), Mapping)
            else {}
        ),
        "checkpoint": deepcopy(
            dict(target.get("checkpoint"))
            if isinstance(target.get("checkpoint"), Mapping)
            else {}
        ),
    }


def _relative_link(path: Path, report_path: Path) -> str:
    return Path(os.path.relpath(path, report_path.parent)).as_posix()


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
    include_repair_id: str | None = None,
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
    bundle_root = report_path.parent
    bundle_root.mkdir(parents=True, exist_ok=True)
    previous_manifest_path = bundle_root / "evidence_bundle_manifest.json"
    if previous_manifest_path.is_file():
        previous = _read_json(previous_manifest_path, required=True)
        old_files = previous.get("files")
        if not isinstance(old_files, list):
            raise EvidenceReportError("previous evidence manifest has no files list")
        old_paths: list[Path] = []
        for raw in old_files:
            if not isinstance(raw, str) or not raw:
                raise EvidenceReportError("previous evidence manifest has invalid path")
            old_path = root / raw
            old_resolved = old_path.resolve()
            if bundle_root not in old_resolved.parents:
                raise EvidenceReportError(
                    "previous evidence manifest points outside its bundle"
                )
            if old_path.is_symlink() or old_path.absolute() != old_resolved:
                raise EvidenceReportError("refusing to clear a symlinked old artifact")
            old_paths.append(old_resolved)
        current_files = {
            p.resolve()
            for p in bundle_root.rglob("*")
            if p.is_file() or p.is_symlink()
        }
        if publish and current_files != set(old_paths):
            raise EvidenceReportError(
                "previous manifest did not account for every published file"
            )
        for old_path in old_paths:
            if old_path.is_file():
                old_path.unlink()
    elif publish and any(bundle_root.iterdir()):
        raise EvidenceReportError(
            "publish destination must be fresh or contain its prior "
            "evidence_bundle_manifest.json"
        )

    asset_dir = bundle_root / "assets"
    code_dir = bundle_root / "code"
    data_dir = bundle_root / "data"
    semantic_dir = bundle_root / "artifacts"
    published_files: list[str] = []
    copied_destinations: set[Path] = set()

    def publish_copy(
        source: Path | str | None,
        destination: Path,
        *,
        allowed_suffixes: frozenset[str] | None = None,
        max_bytes: int = _MAX_PUBLIC_ARTIFACT_BYTES,
        skip_oversize: bool = False,
    ) -> Path | None:
        if source is None or (isinstance(source, str) and not source.strip()):
            return None
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = root / source_path
        if not source_path.exists():
            return None
        source_resolved = source_path.resolve()
        if root not in source_resolved.parents:
            raise EvidenceReportError("artifact source is outside repo root")
        if (
            not source_resolved.is_file()
            or source_path.is_symlink()
            or source_path.absolute() != source_resolved
        ):
            raise EvidenceReportError("artifact source must be a regular non-symlink file")
        if allowed_suffixes and source_resolved.suffix.lower() not in allowed_suffixes:
            raise EvidenceReportError(
                f"artifact has invalid role suffix: {source_resolved.name}"
            )

        destination_resolved = destination.resolve()
        if bundle_root not in destination_resolved.parents:
            raise EvidenceReportError("artifact destination is outside bundle")
        if (
            destination.is_symlink()
            or destination.absolute() != destination_resolved
        ):
            raise EvidenceReportError(
                "artifact destination must be a fresh non-symlink path"
            )
        if destination_resolved in copied_destinations:
            raise EvidenceReportError("duplicate artifact destination")
        if destination_resolved.exists():
            raise EvidenceReportError("artifact destination already exists")
        if source_resolved.stat().st_size > max_bytes:
            if skip_oversize:
                return None
            raise EvidenceReportError(
                f"artifact exceeds {max_bytes} byte limit: {source_resolved.name}"
            )
        destination_resolved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_resolved, destination_resolved)
        copied_destinations.add(destination_resolved)
        relative = destination_resolved.relative_to(root).as_posix()
        published_files.append(relative)
        return destination_resolved

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
        _safe_artifact_id(item, label="child_id") if item is not None else None
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

    publish_copy(
        evaluation / "request.json",
        semantic_dir / "query/request.json",
    )
    for relative in _PLAN_ARTIFACTS:
        publish_copy(
            evaluation / "plan" / relative,
            semantic_dir / "plan" / relative,
        )
    for stem in ("free_concern_response",):
        for response_index in range(1, 5):
            publish_copy(
                evaluation / f"plan/{stem}_{response_index}.txt",
                semantic_dir / f"plan/{stem}_{response_index}.txt",
            )
        publish_copy(
            evaluation / f"plan/{stem}.txt",
            semantic_dir / f"plan/{stem}.txt",
        )

    for step_index in range(1, len(rounds) + 1):
        source_root = (
            evaluation / f"plan/claim_first_steps/after_round_{step_index:02d}"
        )
        destination_root = (
            semantic_dir / f"plan/claim_first_steps/after_round_{step_index:02d}"
        )
        for name in (
            "prompt.md",
            "semantic_proposal_bundle.json",
            "bound_semantic_step.json",
        ):
            publish_copy(
                source_root / name,
                destination_root / name,
            )
        for response_index in range(1, 5):
            publish_copy(
                source_root / f"response_{response_index}.txt",
                destination_root / f"response_{response_index}.txt",
            )

    claim_runtime = evaluation / "plan/claim_first_runtime"
    for round_index in range(1, len(rounds) + 1):
        publish_copy(
            claim_runtime / f"evidence_after_round_{round_index:02d}.json",
            semantic_dir
            / "plan/claim_first_runtime"
            / f"evidence_after_round_{round_index:02d}.json",
        )
    publish_copy(
        claim_runtime / "query_answer.json",
        semantic_dir / "answer/query_answer.json",
    )

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
        code_copy = publish_copy(
            display_code,
            code_dir / f"{round_id}_{display_code.name}" if display_code else code_dir / "missing",
        )
        variant_copy = publish_copy(
            variant_spec,
            data_dir / f"{round_id}_variant_spec.json",
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
        scene_copy = publish_copy(
            scene_source,
            asset_dir / f"{round_id}_scene.png",
        )
        montage_copy = publish_copy(
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
        video_copy = publish_copy(
            video_source,
            asset_dir / f"{round_id}_act.mp4",
            max_bytes=max_video_bytes,
            skip_oversize=True,
        )

        generated_tool = (
            (tool.get("source") or {}).get("artifact")
            if isinstance(tool.get("source"), dict)
            else None
        )
        tool_code_copy = publish_copy(
            generated_tool,
            code_dir / f"{round_id}_tool.py",
            allowed_suffixes=frozenset({".py"}),
        )

        taskgen_destination = semantic_dir / "taskgen" / round_id
        if child is not None:
            for relative in _TASKGEN_ARTIFACTS:
                publish_copy(
                    child / relative,
                    taskgen_destination / relative,
                )
            for relative in _TASKGEN_RENDER_ARTIFACTS:
                publish_copy(
                    child / relative,
                    taskgen_destination / relative,
                )

        tool_destination = semantic_dir / "tool" / round_id
        publish_copy(
            execution / "planned_tool/tool_execution.json",
            tool_destination / "tool_execution.json",
        )
        tool_artifacts = (
            tool.get("artifacts")
            if isinstance(tool.get("artifacts"), Mapping)
            else {}
        )
        copied_tool_references: dict[str, Path] = {}
        for artifact_name, allowed_suffixes in sorted(
            _TOOL_ARTIFACT_ROLES.items()
        ):
            raw_source = tool_artifacts.get(artifact_name)
            if not isinstance(raw_source, str) or not raw_source.strip():
                continue
            source = Path(raw_source)
            if not source.is_absolute():
                source = root / source
            suffix = source.suffix.lower()
            destination = tool_destination / f"{artifact_name}{suffix}"
            copied = publish_copy(
                source,
                destination,
                allowed_suffixes=allowed_suffixes,
            )
            if copied is not None:
                copied_tool_references[artifact_name] = source.resolve()

        toolgen_manifest_source = copied_tool_references.get("toolgen_manifest")
        if toolgen_manifest_source is not None:
            toolgen_manifest = _read_json(toolgen_manifest_source)
            successful_attempt = toolgen_manifest.get("successful_attempt")
            if isinstance(successful_attempt, int) and successful_attempt >= 0:
                attempt = (
                    toolgen_manifest_source.parent
                    / "attempts"
                    / f"attempt_{successful_attempt}"
                )
                for source_name, destination_name in (
                    ("prompt.md", "codegen_prompt.md"),
                    ("response.txt", "codegen_response.txt"),
                    ("validation.json", "codegen_validation.json"),
                ):
                    publish_copy(
                        attempt / source_name,
                        tool_destination / destination_name,
                        allowed_suffixes=frozenset(
                            {Path(source_name).suffix.lower()}
                        ),
                    )

        publish_copy(
            execution / "execution_vqa/execution_vqa.json",
            semantic_dir / "vqa" / f"{round_id}.json",
        )
        publish_copy(
            execution / "aggregate_result.json",
            semantic_dir / "aggregate" / f"{round_id}.json",
        )

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
        data_relative = data_path.relative_to(root).as_posix()
        published_files.append(data_relative)

        persisted_task_proposal = round_plan.get("task_proposal")
        if isinstance(persisted_task_proposal, Mapping):
            task_proposal_heading = "### Plan -> TaskProposal"
            task_proposal_payload = persisted_task_proposal
        else:
            task_proposal_heading = (
                "### Compatibility task projection (not used for planning)"
            )
            task_proposal_payload = {
                "proposal_status": "not_projected_in_compatibility_view",
                "task_name": round_plan.get("task_name") or target.get("task_name"),
                "aspect_id": compact["aspect_id"],
                "task_instruction": round_plan.get("task_instruction"),
            }
        persisted_tool_proposal = round_plan.get("tool_proposal")
        if isinstance(persisted_tool_proposal, Mapping):
            tool_proposal_heading = "### ToolProposal -> ToolGen / reuse"
            tool_proposal_payload = persisted_tool_proposal
        else:
            tool_proposal_heading = (
                "### Compatibility Tool projection (not used for planning)"
            )
            tool_proposal_payload = {
                "proposal_status": "not_projected_in_compatibility_view",
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

    for source, destination in _FINAL_ARTIFACTS:
        publish_copy(
            evaluation / source,
            semantic_dir / destination,
        )
    for round_plan in rounds:
        round_id = str(round_plan.get("round_id") or "")
        if not round_id:
            continue
        publish_copy(
            evaluation / f"plan/decision_after_{round_id}.json",
            semantic_dir / "plan/decisions" / f"after_{round_id}.json",
        )

    repair_result: dict[str, Any] | None = None
    if include_repair_id is not None:
        repair_id = _safe_artifact_id(
            include_repair_id,
            label="include_repair_id",
        )
        repair_root = evaluation / "repairs" / repair_id
        repair_result = _read_json(repair_root / "result.json", required=True)
        if repair_result.get("status") != "completed":
            raise EvidenceReportError(
                f"requested repair is not completed: {repair_id}"
            )
        for source, destination in (
            ("result.json", "result.json"),
            ("repair_provenance.json", "repair_provenance.json"),
            (
                "first_query/planned_tool/tool_execution.json",
                "first_query_tool_execution.json",
            ),
            (
                "second_query_exact_reuse/planned_tool/tool_execution.json",
                "exact_reuse_tool_execution.json",
            ),
        ):
            publish_copy(
                repair_root / source,
                semantic_dir / "audit" / "completed_round_reuse" / destination,
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
    if repair_result is not None:
        lines.extend(
            [
                "### Append-only completed-round Tool reuse audit",
                "",
                _json_block(
                    {
                        "repair_id": repair_result.get("repair_id"),
                        "act_rollouts_started": repair_result.get(
                            "act_rollouts_started"
                        ),
                        "first_query_route": repair_result.get(
                            "first_query_route"
                        ),
                        "first_query_measurements": repair_result.get(
                            "first_query_measurements"
                        ),
                        "exact_reuse_route": repair_result.get(
                            "exact_reuse_route"
                        ),
                        "exact_reuse_provider_called": repair_result.get(
                            "exact_reuse_provider_called"
                        ),
                        "aggregate_status": repair_result.get(
                            "aggregate_status"
                        ),
                    }
                ),
                "",
                "This audit reuses completed ACT telemetry and starts no "
                "simulator or policy rollout. It proves exact run-local reuse, "
                "not independent cross-evaluation reuse.",
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
        "source_server_path": str(evaluation.resolve()) if publish else None,
        "report": str(report_path.relative_to(root)).replace("\\", "/"),
        "publish_mode": bool(publish),
        "files": sorted(set(published_files)),
        "round_count": len(compact_rounds),
        "video_size_limit_bytes": int(max_video_bytes),
        "included_repair_id": include_repair_id,
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
