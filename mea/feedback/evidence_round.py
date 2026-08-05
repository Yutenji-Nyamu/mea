"""Per-round evidence projection and artifact publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .evidence_projection import (
    _compact_aggregate,
    _compact_decision,
    _compact_tool_rows,
    _compact_vqa,
    _read_json,
    _semantic_aggregate,
)
from .evidence_publication import EvidencePublisher


TASKGEN_ARTIFACTS = (
    "generation/code_prompt.md",
    "generation/provider_response.txt",
    "validation/static.json",
    "validation/checker_fixtures.json",
    "validation/implementation_trace.json",
    "validation/setup_preflight.json",
    "validation/expert_preflight.json",
    "validation/vision.json",
    "validation/vision_prompt.md",
    "validation/vision_response.txt",
)
TASKGEN_RENDER_ARTIFACTS = ("evidence/scene_comparison.png",)
TOOL_ARTIFACT_ROLES = {
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


@dataclass(frozen=True)
class PublishedRound:
    round_id: str
    compact: dict[str, Any]
    proposal_copy: Path | None
    taskgen_destination: Path
    code_copy: Path | None
    variant_copy: Path | None
    scene_copy: Path | None
    video_copy: Path | None
    video_source: Path | None
    tool_code_copy: Path | None
    montage_copy: Path | None


def publish_round_evidence(
    *,
    publisher: EvidencePublisher,
    evaluation: Path,
    round_id: str,
    round_plan: Mapping[str, Any],
    round_summary: Mapping[str, Any],
    child_id: str | None,
    next_decision: Mapping[str, Any] | None,
    max_video_bytes: int,
) -> PublishedRound:
    """Publish real round artifacts and return their compact projection."""

    root = publisher.root
    child = root / "mea/generated_tasks" / child_id if child_id else None
    child_manifest = _read_json(child / "manifest.json") if child else {}
    execution = evaluation / "execution" / round_id
    tool = _read_json(execution / "planned_tool/tool_execution.json")
    vqa = _read_json(execution / "execution_vqa/execution_vqa.json")
    aggregate = _read_json(execution / "aggregate_result.json")

    task_code = (
        child / "task.py"
        if child and (child / "task.py").is_file()
        else None
    )
    overlay = (
        child / "overlay.yml"
        if child and (child / "overlay.yml").is_file()
        else None
    )
    variant_spec = (
        child / "variant_spec.json"
        if child and (child / "variant_spec.json").is_file()
        else None
    )
    display_code = task_code or (
        overlay
        if overlay is not None and overlay.stat().st_size > 3
        else None
    )
    code_copy = publisher.publish_copy(
        display_code,
        (
            publisher.code_dir / f"{round_id}_{display_code.name}"
            if display_code
            else publisher.code_dir / "missing"
        ),
    )
    variant_copy = publisher.publish_copy(
        variant_spec,
        publisher.data_dir / f"{round_id}_variant_spec.json",
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
    scene_copy = publisher.publish_copy(
        scene_source,
        publisher.asset_dir / f"{round_id}_scene.png",
    )
    montage_copy = publisher.publish_copy(
        execution / "execution_vqa/execution_montage.png",
        publisher.asset_dir / f"{round_id}_vqa_montage.png",
    )
    video_source = None
    if child:
        for candidate in (
            child / "evaluation/episode0.mp4",
            *sorted(
                (child / "evaluation/telemetry/act").glob("*/video.mp4")
            ),
        ):
            if candidate.is_file():
                video_source = candidate
                break
    video_copy = publisher.publish_copy(
        video_source,
        publisher.asset_dir / f"{round_id}_act.mp4",
        max_bytes=max_video_bytes,
        skip_oversize=True,
    )

    generated_tool = (
        (tool.get("source") or {}).get("artifact")
        if isinstance(tool.get("source"), dict)
        else None
    )
    tool_code_copy = publisher.publish_copy(
        generated_tool,
        publisher.code_dir / f"{round_id}_tool.py",
        allowed_suffixes=frozenset({".py"}),
    )

    taskgen_destination = publisher.semantic_dir / "taskgen" / round_id
    proposal_copy = None
    if child is not None:
        proposal_copy = publisher.publish_first(
            (
                child / "generation/proposal.json",
                child / "generation/experiment_candidate.json",
            ),
            taskgen_destination / "generation/proposal.json",
        )
        for relative in TASKGEN_ARTIFACTS:
            publisher.publish_copy(
                child / relative,
                taskgen_destination / relative,
            )
        for relative in TASKGEN_RENDER_ARTIFACTS:
            publisher.publish_copy(
                child / relative,
                taskgen_destination / relative,
            )

    tool_destination = publisher.semantic_dir / "tool" / round_id
    publisher.publish_copy(
        execution / "planned_tool/tool_execution.json",
        tool_destination / "tool_execution.json",
    )
    tool_artifacts = (
        tool.get("artifacts")
        if isinstance(tool.get("artifacts"), Mapping)
        else {}
    )
    copied_tool_references: dict[str, Path] = {}
    for artifact_name, allowed_suffixes in sorted(TOOL_ARTIFACT_ROLES.items()):
        raw_source = tool_artifacts.get(artifact_name)
        if not isinstance(raw_source, str) or not raw_source.strip():
            continue
        source = Path(raw_source)
        if not source.is_absolute():
            source = root / source
        suffix = source.suffix.lower()
        copied = publisher.publish_copy(
            source,
            tool_destination / f"{artifact_name}{suffix}",
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
                publisher.publish_copy(
                    attempt / source_name,
                    tool_destination / destination_name,
                    allowed_suffixes=frozenset(
                        {Path(source_name).suffix.lower()}
                    ),
                )
    publisher.publish_copy(
        execution / "execution_vqa/execution_vqa.json",
        publisher.semantic_dir / "vqa" / f"{round_id}.json",
    )

    observations = (
        round_summary.get("observations")
        if isinstance(round_summary.get("observations"), dict)
        else {}
    )
    scene_validation = (
        child_manifest.get("scene_validation")
        if isinstance(child_manifest.get("scene_validation"), Mapping)
        else {}
    )
    generic_preflight = (
        scene_validation.get("generic_preflight")
        if isinstance(scene_validation.get("generic_preflight"), Mapping)
        else {}
    )
    checker_contract = (
        child_manifest.get("checker_contract")
        if isinstance(child_manifest.get("checker_contract"), Mapping)
        else {}
    )
    taskgen_provider = (
        child_manifest.get("provider")
        if isinstance(child_manifest.get("provider"), Mapping)
        else {}
    )
    vision_validation = (
        generic_preflight.get("vision_validation")
        if isinstance(generic_preflight.get("vision_validation"), Mapping)
        else {}
    )
    compact = {
        "round_id": round_id,
        "aspect_id": round_plan.get("aspect_id")
        or round_plan.get("sub_aspect"),
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
        "aggregate": _compact_aggregate(aggregate),
        "next_decision": _compact_decision(next_decision),
        "taskgen_gates": {
            "generation_attempts": taskgen_provider.get(
                "provider_call_count"
            ),
            "checker_fixtures": (
                f"{checker_contract.get('fixture_pass_count')}/"
                f"{checker_contract.get('fixture_count')}"
                if checker_contract.get("fixture_count") is not None
                else None
            ),
            "vision_passed": vision_validation.get("passed"),
            "expert_passed": generic_preflight.get("expert_passed"),
        },
    }
    publisher.publish_json(
        _semantic_aggregate(aggregate),
        publisher.semantic_dir / "aggregate" / f"{round_id}.json",
    )
    return PublishedRound(
        round_id=round_id,
        compact=compact,
        proposal_copy=proposal_copy,
        taskgen_destination=taskgen_destination,
        code_copy=code_copy,
        variant_copy=variant_copy,
        scene_copy=scene_copy,
        video_copy=video_copy,
        video_source=video_source,
        tool_code_copy=tool_code_copy,
        montage_copy=montage_copy,
    )


__all__ = ["PublishedRound", "publish_round_evidence"]
