"""Run retrieved trusted tools over every completed telemetry episode."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .retrieval import TrustedToolRetriever
from .tools import TrajectoryView, run_trusted_tools


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_telemetry_root(
    telemetry_root: str | Path,
    *,
    user_request: str,
    task_name: str = "beat_block_hammer",
    outcome_metric: str = "official_check_success",
    outcome_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_outcome_binding: dict[str, str] | None = None
    if outcome_metric == "generated_check_success":
        expected = {
            "metric",
            "authority",
            "success_spec_sha256",
            "task_module",
        }
        if not isinstance(outcome_binding, Mapping) or set(outcome_binding) != expected:
            raise RuntimeError(
                "generated_check_success requires an exact outcome binding"
            )
        normalized_outcome_binding = {
            key: str(outcome_binding[key]).strip() for key in expected
        }
        if (
            normalized_outcome_binding["metric"] != "generated_check_success"
            or normalized_outcome_binding["authority"]
            != "compiled_success_spec_experimental_bounded"
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                normalized_outcome_binding["success_spec_sha256"],
            )
            or not normalized_outcome_binding["task_module"]
        ):
            raise RuntimeError("invalid generated outcome binding")
    elif outcome_binding is not None:
        raise RuntimeError(
            "outcome_binding is only valid for generated_check_success"
        )
    root = Path(telemetry_root).expanduser().resolve()
    selection = TrustedToolRetriever().select(
        user_request,
        task_name=task_name,
        outcome_metric=outcome_metric,
    )
    episodes: list[dict[str, Any]] = []
    for metadata_path in sorted(root.rglob("episode.json")):
        episode_dir = metadata_path.parent
        trajectory = TrajectoryView(episode_dir)
        episode_task = trajectory.metadata.get("task_name")
        if episode_task != task_name:
            raise RuntimeError(
                "telemetry root 混入其他任务: "
                f"requested={task_name!r}, episode={episode_task!r}, "
                f"path={episode_dir}"
            )
        if normalized_outcome_binding is not None:
            if (
                trajectory.metadata.get("task_module")
                != normalized_outcome_binding["task_module"]
            ):
                raise RuntimeError(
                    "generated outcome binding task_module differs from episode"
                )
            trajectory.outcome_binding = dict(normalized_outcome_binding)
        results = run_trusted_tools(
            trajectory, selection["selected_tools"]
        )
        artifact_names = (
            "episode.json",
            "states.csv",
            "semantic_trace.npz",
            "events.jsonl",
            "dynamics_trace.npz",
            "telemetry_profile.json",
        )
        artifact_hashes = {
            name: _sha256(episode_dir / name)
            for name in artifact_names
            if (episode_dir / name).is_file()
        }
        episode_result = {
            "episode_dir": str(episode_dir.relative_to(root)),
            "metadata": trajectory.metadata,
            "artifact_sha256": artifact_hashes,
            "tool_results": results,
        }
        (episode_dir / "tool_results.json").write_text(
            json.dumps(episode_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        episodes.append(episode_result)
    if not episodes:
        raise RuntimeError(f"没有在 {root} 下发现完整 telemetry episode")
    summary = {
        "schema_version": 1,
        "task_name": task_name,
        "user_request": user_request,
        "outcome_binding": normalized_outcome_binding,
        "tool_retrieval": selection,
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    (root / "tool_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
