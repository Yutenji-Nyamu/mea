"""Run retrieved trusted tools over every completed telemetry episode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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
) -> dict[str, Any]:
    root = Path(telemetry_root).expanduser().resolve()
    selection = TrustedToolRetriever().select(
        user_request, task_name=task_name
    )
    episodes: list[dict[str, Any]] = []
    for metadata_path in sorted(root.rglob("episode.json")):
        episode_dir = metadata_path.parent
        trajectory = TrajectoryView(episode_dir)
        results = run_trusted_tools(
            trajectory, selection["selected_tools"]
        )
        artifact_hashes = {
            name: _sha256(episode_dir / name)
            for name in ("episode.json", "states.csv", "semantic_trace.npz", "events.jsonl")
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
        "tool_retrieval": selection,
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    (root / "tool_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
