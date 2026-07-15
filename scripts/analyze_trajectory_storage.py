#!/usr/bin/env python3
"""Report the real storage footprint and signal shapes of one trajectory."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACTS = (
    "episode.json",
    "schema.json",
    "states.csv",
    "semantic_trace.npz",
    "events.jsonl",
    "video.mp4",
    "tool_results.json",
)


def _file_size(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "logical_bytes": int(stat.st_size),
        "disk_bytes": int(getattr(stat, "st_blocks", 0) * 512),
    }


def analyze_episode(episode_dir: str | Path) -> dict[str, Any]:
    episode = Path(episode_dir).expanduser().resolve()
    if not episode.is_dir():
        raise FileNotFoundError(f"trajectory directory does not exist: {episode}")

    files = {
        name: _file_size(episode / name)
        for name in ARTIFACTS
        if (episode / name).is_file()
    }
    trace_path = episode / "semantic_trace.npz"
    if not trace_path.is_file():
        raise FileNotFoundError(f"missing semantic_trace.npz: {episode}")
    with np.load(trace_path) as archive:
        arrays = {
            name: {
                "shape": list(archive[name].shape),
                "dtype": str(archive[name].dtype),
                "raw_bytes": int(archive[name].nbytes),
            }
            for name in archive.files
        }
    raw_trace_bytes = sum(item["raw_bytes"] for item in arrays.values())

    states_path = episode / "states.csv"
    states: dict[str, Any] = {}
    if states_path.is_file():
        data = states_path.read_bytes()
        with states_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            row_count = sum(1 for _ in reader)
        states = {
            "rows": row_count,
            "columns": len(header),
            "logical_bytes": len(data),
            "gzip_level_9_estimate_bytes": len(gzip.compress(data, compresslevel=9)),
        }

    total_logical = sum(item["logical_bytes"] for item in files.values())
    return {
        "schema_version": 1,
        "episode_dir": str(episode),
        "files": files,
        "total_logical_bytes": total_logical,
        "states_csv": states,
        "semantic_trace": {
            "rows": int(arrays["physics_step"]["shape"][0]),
            "arrays": arrays,
            "raw_array_bytes": raw_trace_bytes,
            "compressed_npz_bytes": files["semantic_trace.npz"]["logical_bytes"],
            "compression_ratio": (
                raw_trace_bytes / files["semantic_trace.npz"]["logical_bytes"]
            ),
        },
        "interpretation": {
            "time_axis": "250 Hz task-semantic slice after recorder attachment",
            "not_a_full_simulator_dump": True,
            "rgb_storage": "compressed MP4 when video.mp4 exists",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_episode(args.episode_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
