"""Policy invocation and RolloutObservation projection for RoboTwin."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import (
    MaterializedCandidate,
    RolloutObservation,
    RolloutRequest,
)

from .runtime_contracts import _RoboTwinNativeCandidate, _json_object


def rollout_candidate(
    backend: Any,
    candidate: MaterializedCandidate,
    request: RolloutRequest,
) -> RolloutObservation:
    native = candidate.native_task
    if not isinstance(native, _RoboTwinNativeCandidate):
        raise TypeError(
            "RoboTwin candidate native_task has the wrong runtime type"
        )
    manifest = deepcopy(dict(native.rollout_manifest))
    request.output_dir.mkdir(parents=True, exist_ok=True)
    overlay = request.output_dir / "overlay.yml"
    source_value = manifest.get("overlay")
    source_overlay = (
        Path(str(source_value)).expanduser()
        if isinstance(source_value, str) and source_value.strip()
        else None
    )
    if source_overlay is not None and not source_overlay.is_absolute():
        source_overlay = (backend.repo_root / source_overlay).resolve()
    if source_overlay is not None and source_overlay.is_file():
        if source_overlay.resolve() != overlay.resolve():
            overlay.write_bytes(source_overlay.read_bytes())
    elif not overlay.is_file():
        overlay.write_text("{}\n", encoding="utf-8")
    manifest["overlay"] = str(overlay)
    result = backend.rollout_runner(
        candidate=candidate,
        request=request,
        manifest=manifest,
    )
    value = _json_object(result, "RoboTwin rollout result")
    if not isinstance(value.get("success"), bool):
        raise TypeError(
            "RoboTwin rollout result requires explicit boolean success"
        )
    episode = _json_object(
        value.get("episode"),
        "RoboTwin rollout result.episode",
    )
    raw_artifacts = value.get("artifacts") or {}
    if not isinstance(raw_artifacts, Mapping):
        raise TypeError(
            "RoboTwin rollout result.artifacts must be an object"
        )
    artifacts = {
        str(key): str(item)
        for key, item in raw_artifacts.items()
    }
    metadata = _json_object(
        value.get("metadata") or {},
        "RoboTwin rollout result.metadata",
    )
    return RolloutObservation(
        benchmark=backend.benchmark,
        round_id=request.round_id,
        candidate_id=candidate.candidate_id,
        seed=request.seed,
        success=value["success"],
        episode=episode,
        native_episode=result,
        artifacts=artifacts,
        metadata={
            "taskgen_route": native.taskgen_resolution["route"],
            **metadata,
        },
    )
