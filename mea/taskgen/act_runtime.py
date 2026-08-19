"""ACT runtime invocation and artifact-alignment contract for TaskGen.

This module owns the simulator-facing ACT mechanics.  TaskGen orchestration
continues to decide *when* ACT is allowed to run and injects the command runner
so the public CLI and its existing test seams remain stable.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


CommandRunner = Callable[..., int]
JsonWriter = Callable[[Path, Any], None]


def newest_eval_dir(
    repo_root: Path,
    before: set[Path],
    *,
    task_name: str = "beat_block_hammer",
    task_config: str = "demo_clean",
    checkpoint_setting: str = "demo_clean",
) -> Path | None:
    """Return only an evaluation directory created after the current launch."""

    eval_root = (
        repo_root / "eval_result" / task_name / "ACT" / task_config / checkpoint_setting
    )
    after = (
        {path for path in eval_root.glob("*") if path.is_dir()}
        if eval_root.exists()
        else set()
    )
    created = after - before
    return max(created, key=lambda path: path.stat().st_mtime) if created else None


def archive_previous_act_attempt(run_dir: Path) -> Path | None:
    """Preserve stale retry artifacts without mixing them into a new result."""

    evaluation_dir = run_dir / "evaluation"
    candidates = [
        *evaluation_dir.glob("episode*.mp4"),
        *(evaluation_dir / name for name in ("_result.txt", "act.json", "act.log")),
        evaluation_dir / "telemetry/act",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    archive_dir = evaluation_dir / "previous_act_attempts" / stamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), archive_dir / path.name)
    return archive_dir


def build_act_command(
    *,
    python_executable: str,
    task_name: str,
    task_config: str,
    checkpoint_setting: str,
    expert_data_num: int,
    policy_seed: int,
    gpu: int,
    num_episodes: int,
    task_module: Any,
    overlay_path: Path,
    seed: int,
    telemetry_root: Path,
    telemetry_profile: str,
) -> list[str]:
    """Build the positional ``eval_mea.sh`` contract without launching it."""

    command = [
        "env",
        f"PYTHON_BIN={python_executable}",
        "bash",
        "policy/ACT/eval_mea.sh",
        task_name,
        task_config,
        checkpoint_setting,
        str(expert_data_num),
        str(policy_seed),
        str(gpu),
        str(num_episodes),
        task_module,
        str(overlay_path),
        str(seed),
        str(telemetry_root),
        telemetry_profile,
    ]
    return command


def run_act(
    repo_root: Path,
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    seed: int,
    gpu: int,
    num_episodes: int,
    command_runner: CommandRunner,
    json_writer: JsonWriter,
    telemetry_profile: str = "balanced_v1",
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Run a task-specific ACT checkpoint and attach videos to telemetry."""

    task_name = str(manifest["task_name"])
    task_config = str(manifest.get("task_config") or "demo_clean")
    checkpoint_setting = str(manifest.get("checkpoint_setting") or "demo_clean")
    expert_data_num = int(manifest.get("expert_data_num") or 50)
    policy_seed = int(manifest.get("policy_seed") or 0)
    checkpoint_dir = (
        repo_root
        / "policy/ACT/act_ckpt"
        / f"act-{task_name}"
        / f"{checkpoint_setting}-{expert_data_num}"
    )
    required_checkpoint_files = [
        checkpoint_dir / "policy_last.ckpt",
        checkpoint_dir / "dataset_stats.pkl",
    ]
    missing_checkpoint_files = [
        path for path in required_checkpoint_files if not path.is_file()
    ]
    if missing_checkpoint_files:
        missing = ", ".join(
            str(path.relative_to(repo_root)) for path in missing_checkpoint_files
        )
        raise RuntimeError(
            f"ACT checkpoint preflight failed for {task_name}: {missing}. "
            "Download it on the server with "
            f"`python scripts/download_act_checkpoint.py {task_name}`; "
            "do not relay routine checkpoints through a local workstation."
        )

    previous_attempt = archive_previous_act_attempt(run_dir)
    telemetry_root = run_dir / "evaluation/telemetry/act"
    eval_root = (
        repo_root / "eval_result" / task_name / "ACT" / task_config / checkpoint_setting
    )
    before = (
        {path for path in eval_root.glob("*") if path.is_dir()}
        if eval_root.exists()
        else set()
    )
    command = build_act_command(
        python_executable=python_executable,
        task_name=task_name,
        task_config=task_config,
        checkpoint_setting=checkpoint_setting,
        expert_data_num=expert_data_num,
        policy_seed=policy_seed,
        gpu=gpu,
        num_episodes=num_episodes,
        task_module=manifest["task_module"],
        overlay_path=run_dir / "overlay.yml",
        seed=seed,
        telemetry_root=telemetry_root,
        telemetry_profile=telemetry_profile,
    )
    started = datetime.now().astimezone().isoformat()
    returncode = command_runner(
        command,
        cwd=repo_root,
        log_path=run_dir / "evaluation/act.log",
    )
    source_dir = newest_eval_dir(
        repo_root,
        before,
        task_name=task_name,
        task_config=task_config,
        checkpoint_setting=checkpoint_setting,
    )
    copied = []
    result_file_copied = False
    if source_dir:
        sources = sorted(source_dir.glob("episode*.mp4"))
        result_file = source_dir / "_result.txt"
        if result_file.is_file():
            sources.append(result_file)
        for source in sources:
            if source.is_file():
                destination = run_dir / "evaluation" / source.name
                shutil.copy2(source, destination)
                copied.append(str(destination.relative_to(repo_root)))
                if source.name == "_result.txt":
                    result_file_copied = True

    copied_video_paths = list((run_dir / "evaluation").glob("episode*.mp4"))
    telemetry_episode_paths = list(
        metadata.parent for metadata in telemetry_root.glob("episode_*/episode.json")
    )
    index_issues: list[str] = []
    video_by_index: dict[int, Path] = {}
    telemetry_by_index: dict[int, Path] = {}
    for video in copied_video_paths:
        match = re.fullmatch(r"episode(\d+)\.mp4", video.name)
        if match is None:
            index_issues.append(f"unrecognized ACT video name: {video.name}")
            continue
        episode_index = int(match.group(1))
        if episode_index in video_by_index:
            index_issues.append(f"duplicate ACT video index: {episode_index}")
            continue
        if video.stat().st_size <= 0:
            index_issues.append(f"empty ACT video: {video.name}")
        video_by_index[episode_index] = video
    for episode_dir in telemetry_episode_paths:
        match = re.match(r"episode_(\d+)(?:_|$)", episode_dir.name)
        if match is None:
            index_issues.append(
                f"unrecognized ACT telemetry directory: {episode_dir.name}"
            )
            continue
        episode_index = int(match.group(1))
        if episode_index in telemetry_by_index:
            index_issues.append(f"duplicate ACT telemetry index: {episode_index}")
            continue
        telemetry_by_index[episode_index] = episode_dir
    video_indices = set(video_by_index)
    telemetry_indices = set(telemetry_by_index)
    if video_indices != telemetry_indices:
        index_issues.append(
            "ACT video/telemetry indices differ: "
            f"videos={sorted(video_indices)}, telemetry={sorted(telemetry_indices)}"
        )
    paired_indices = sorted(video_indices & telemetry_indices)
    copied_videos = [video_by_index[index] for index in sorted(video_indices)]
    telemetry_episodes = [
        telemetry_by_index[index] for index in sorted(telemetry_indices)
    ]
    video_associations = []
    actual_seeds: list[int] = []
    for episode_index in paired_indices:
        episode_dir = telemetry_by_index[episode_index]
        video = video_by_index[episode_index]
        destination = episode_dir / "video.mp4"
        shutil.copy2(video, destination)
        metadata_path = episode_dir / "episode.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("seed") is not None:
                actual_seeds.append(int(metadata["seed"]))
            metadata.setdefault("artifacts", {})["video"] = "video.mp4"
            metadata["video_alignment"] = {
                "policy_frame_rate_hz": 10,
                "frame_semantics": (
                    "pre-action; contact in policy step k lies between adjacent frames"
                ),
            }
            json_writer(metadata_path, metadata)
        video_associations.append(
            {
                "episode_dir": str(episode_dir.relative_to(repo_root)),
                "video": str(destination.relative_to(repo_root)),
                "episode_index": episode_index,
            }
        )

    result = {
        "command": command,
        "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(),
        "returncode": returncode,
        "task_name": task_name,
        "task_config": task_config,
        "checkpoint_setting": checkpoint_setting,
        "expert_data_num": expert_data_num,
        "policy_seed": policy_seed,
        "num_episodes": num_episodes,
        "actual_seeds": actual_seeds,
        "checkpoint": {
            "directory": str(checkpoint_dir.relative_to(repo_root)),
            "required_files": [
                str(path.relative_to(repo_root)) for path in required_checkpoint_files
            ],
            "preflight_passed": True,
        },
        "source_eval_dir": str(source_dir) if source_dir else None,
        "copied_artifacts": copied,
        "copied_video_count": len(copied_videos),
        "telemetry_root": str(telemetry_root.relative_to(repo_root)),
        "telemetry_episode_count": len(telemetry_episodes),
        "video_associations": video_associations,
        "episode_index_alignment": {
            "passed": not index_issues,
            "video_indices": sorted(video_indices),
            "telemetry_indices": sorted(telemetry_indices),
            "issues": index_issues,
        },
        "previous_attempt_archive": (
            str(previous_attempt.relative_to(repo_root))
            if previous_attempt is not None
            else None
        ),
        "passed": (
            returncode == 0
            and source_dir is not None
            and result_file_copied
            and not index_issues
            and len(copied_videos) == num_episodes
            and len(telemetry_episodes) == num_episodes
            and len(actual_seeds) == num_episodes
        ),
    }
    json_writer(run_dir / "evaluation/act.json", result)
    if not result["passed"]:
        raise RuntimeError(f"ACT {num_episodes}-episode 未通过: {result}")
    return result
