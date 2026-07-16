"""Generate, validate, render, and optionally evaluate one TaskGen variant."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.providers import OpenAICompatibleProvider
from mea.toolkit import evaluate_telemetry_root
from mea.taskgen import (
    TaskGenPrototype,
    VisualReflectionError,
    execute_reflection_loop,
    inject_oversized_block_fixture,
    inject_wrong_color_fixture,
    repair_generated_method,
    validate_vision_observation,
    create_official_task_run,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_manifest(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    write_json(path, manifest)
    return manifest


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode


def run_probe(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    seed: int,
    episode_index: int = 0,
    expert: bool,
    scene_json: Path | None = None,
    image: Path | None = None,
    log_path: Path | None = None,
    raise_on_failure: bool = True,
    max_expert_attempts: int = 3,
    telemetry_dir: Path | None = None,
    telemetry_profile: str = "balanced_v1",
    visual_capture_profile_id: str | None = None,
) -> dict[str, Any]:
    scene_json = scene_json or run_dir / "validation/scene.json"
    image = image or run_dir / "evidence/initial_head.png"
    log_path = log_path or run_dir / "validation/probe.log"
    command = [
        sys.executable,
        "-m",
        "mea.taskgen.probe",
        "--repo-root",
        str(repo_root),
        "--task-name",
        manifest["task_name"],
        "--task-module",
        manifest["task_module"],
        "--task-config",
        "demo_clean",
        "--ckpt-setting",
        "demo_clean",
        "--overlay",
        str(run_dir / "overlay.yml"),
        "--seed",
        str(seed),
        "--episode-index",
        str(episode_index),
        "--image",
        str(image),
        "--output",
        str(scene_json),
        "--telemetry-profile",
        telemetry_profile,
    ]
    if expert:
        command.append("--expert")
    if telemetry_dir is not None:
        command.extend(["--telemetry-dir", str(telemetry_dir)])
    if visual_capture_profile_id is not None:
        command.extend(
            ["--visual-capture-profile", visual_capture_profile_id]
        )

    attempts: list[dict[str, Any]] = []
    attempt_logs: list[Path] = []
    attempt_limit = max(1, max_expert_attempts) if expert else 1
    scene: dict[str, Any] = {}
    returncode = 1
    for attempt_index in range(attempt_limit):
        attempt_log = (
            log_path.with_name(
                f"{log_path.stem}_attempt_{attempt_index}{log_path.suffix}"
            )
            if expert
            else log_path
        )
        attempt_logs.append(attempt_log)
        returncode = run_command(
            command,
            cwd=repo_root,
            log_path=attempt_log,
        )
        scene = (
            json.loads(scene_json.read_text(encoding="utf-8"))
            if scene_json.exists()
            else {}
        )
        attempts.append(
            {
                "attempt_index": attempt_index,
                "returncode": returncode,
                "expert": scene.get("expert"),
            }
        )
        if returncode != 2:
            break

    if expert:
        combined = []
        for attempt_index, attempt_log in enumerate(attempt_logs):
            combined.append(f"===== expert attempt {attempt_index} =====\n")
            if attempt_log.is_file():
                combined.append(attempt_log.read_text(encoding="utf-8"))
        log_path.write_text("".join(combined), encoding="utf-8")
        scene.setdefault("expert", {})["attempts_used"] = len(attempts)
        scene["expert_attempts"] = attempts
    scene["returncode"] = returncode
    write_json(scene_json, scene)
    if raise_on_failure and returncode != 0:
        raise RuntimeError(f"setup/expert probe 失败，returncode={returncode}")
    return scene


def run_official_expert_episodes(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    start_seed: int,
    num_episodes: int,
    telemetry_profile: str,
    max_seed_candidates: int | None = None,
) -> dict[str, Any]:
    """Execute unchanged expert probes on solvable official-task seeds."""

    episode_summaries: list[dict[str, Any]] = []
    rejected_seeds: list[dict[str, Any]] = []
    first_scene: dict[str, Any] | None = None
    candidate_limit = max_seed_candidates or max(num_episodes * 10, num_episodes + 5)
    for candidate_index in range(candidate_limit):
        if len(episode_summaries) >= num_episodes:
            break
        episode_index = len(episode_summaries)
        seed = start_seed + candidate_index
        is_first = episode_index == 0
        scene = run_probe(
            repo_root,
            run_dir,
            manifest,
            seed=seed,
            episode_index=episode_index,
            expert=True,
            scene_json=(
                run_dir / "validation/scene.json"
                if is_first
                else run_dir
                / f"validation/official_episodes/episode_{episode_index:03d}_seed_{seed}.json"
            ),
            image=(
                run_dir / "evidence/initial_head.png"
                if is_first
                else run_dir
                / f"evidence/official_episodes/episode_{episode_index:03d}_seed_{seed}.png"
            ),
            log_path=(
                run_dir / "validation/probe.log"
                if is_first
                else run_dir
                / f"validation/official_episodes/episode_{episode_index:03d}_seed_{seed}.log"
            ),
            telemetry_dir=(
                run_dir
                / "evaluation/telemetry/expert"
                / f"episode_{episode_index:03d}_seed_{seed}"
            ),
            telemetry_profile=telemetry_profile,
            visual_capture_profile_id="event_keyframes_v1",
            raise_on_failure=False,
            max_expert_attempts=1,
        )
        returncode = int(scene.get("returncode", 0))
        if returncode != 0:
            error = scene.get("error") or {}
            if error.get("type") == "UnStableError":
                rejected_seeds.append(
                    {
                        "seed": seed,
                        "reason": "unstable_initial_scene",
                        "error_type": error.get("type"),
                        "message": error.get("message"),
                    }
                )
                continue
            if returncode == 2:
                rejected_seeds.append(
                    {
                        "seed": seed,
                        "reason": "expert_unsolvable",
                        "error_type": error.get("type"),
                        "message": error.get("message"),
                    }
                )
                continue
            raise RuntimeError(
                "official expert probe failed for "
                f"seed={seed}, returncode={returncode}: "
                f"{error.get('type') or 'unknown error'}"
            )
        if not bool(scene.get("expert", {}).get("passed")):
            rejected_seeds.append(
                {
                    "seed": seed,
                    "reason": "expert_unsolvable",
                    "error_type": None,
                    "message": "official expert did not satisfy check_success",
                }
            )
            continue
        if first_scene is None:
            first_scene = scene
        telemetry = scene.get("telemetry", {})
        telemetry_metadata = telemetry.get("metadata", {})
        video_artifact = telemetry_metadata.get("artifacts", {}).get("video")
        episode_summaries.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "setup_success": bool(scene.get("setup_success")),
                "render_success": bool(scene.get("render_success")),
                "rule_passed": bool(scene.get("rule_check", {}).get("passed")),
                "expert_passed": bool(scene.get("expert", {}).get("passed")),
                "image": scene.get("image"),
                "telemetry": telemetry.get("episode_dir"),
                "video": (
                    str(Path(telemetry["episode_dir"]) / video_artifact)
                    if telemetry.get("episode_dir") and video_artifact
                    else None
                ),
                "visual_capture": telemetry_metadata.get("visual_capture"),
            }
        )
    if first_scene is None or len(episode_summaries) < num_episodes:
        raise RuntimeError(
            "official expert seed scan exhausted before collecting "
            f"{num_episodes} episodes; accepted={len(episode_summaries)}, "
            f"rejected={len(rejected_seeds)}, candidates={candidate_limit}"
        )
    first_scene["expert_batch"] = {
        "passed": all(item["expert_passed"] for item in episode_summaries),
        "episode_count": len(episode_summaries),
        "candidate_count": len(episode_summaries) + len(rejected_seeds),
        "rejected_seed_count": len(rejected_seeds),
        "rejected_seeds": rejected_seeds,
        "episodes": episode_summaries,
    }
    write_json(run_dir / "validation/scene.json", first_scene)
    write_json(
        run_dir / "validation/official_expert_episodes.json",
        first_scene["expert_batch"],
    )
    return first_scene


def collect_position_samples(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    start_seed: int,
    num_episodes: int,
    first_scene: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collect simulator-native block poses for every evaluation seed."""

    sample_root = run_dir / "validation/position_samples"
    samples: list[dict[str, Any]] = []
    for episode_index in range(num_episodes):
        seed = start_seed + episode_index
        if episode_index == 0 and first_scene:
            scene = first_scene
        else:
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=seed,
                expert=True,
                scene_json=sample_root / f"seed_{seed}.json",
                image=sample_root / f"seed_{seed}.png",
                log_path=sample_root / f"seed_{seed}.log",
            )
        position = scene.get("block_pose", {}).get("position")
        if not isinstance(position, list) or len(position) < 2:
            raise RuntimeError(f"seed={seed} 缺少 block_pose.position")
        samples.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "block_position": [float(value) for value in position],
                "block_quaternion": scene.get("block_pose", {}).get("quaternion"),
                "rule_passed": bool(scene.get("rule_check", {}).get("passed")),
                "expert_passed": bool(scene.get("expert", {}).get("passed")),
                "image": scene.get("image"),
            }
        )

    xs = [item["block_position"][0] for item in samples]
    ys = [item["block_position"][1] for item in samples]
    unique_xy = {
        (round(item["block_position"][0], 8), round(item["block_position"][1], 8))
        for item in samples
    }
    result = {
        "start_seed": start_seed,
        "num_episodes": num_episodes,
        "samples": samples,
        "metrics": {
            "unique_xy_count": len(unique_xy),
            "x_span": max(xs) - min(xs),
            "y_span": max(ys) - min(ys),
            "position_varied": len(unique_xy) > 1,
        },
        "passed": len(samples) == num_episodes
        and all(
            item["rule_passed"] and item["expert_passed"] for item in samples
        ),
    }
    write_json(run_dir / "validation/position_samples.json", result)
    return result


def run_vision_check(
    provider: OpenAICompatibleProvider,
    run_dir: Path,
    spec: dict[str, Any],
    *,
    model: str,
    image_path: Path | None = None,
    prompt_path: Path | None = None,
    response_path: Path | None = None,
    result_path: Path | None = None,
) -> dict[str, Any]:
    image_path = image_path or run_dir / "evidence/initial_head.png"
    prompt_path = prompt_path or run_dir / "validation/vision_prompt.md"
    response_path = response_path or run_dir / "validation/vision_response.txt"
    result_path = result_path or run_dir / "validation/vision.json"
    expected_half_size = 0.025 * float(spec["changes"]["block"]["scale"])
    prompt = f"""这是 RoboTwin beat_block_hammer 的初始场景首帧。
请检查被锤子敲击的方块是否符合下面的 VariantSpec，并检查场景是否有明显异常：
{json.dumps(spec, ensure_ascii=False, indent=2)}

官方 scale=1.0 的方块 half_size 是 (0.025, 0.025, 0.025) 米；本次预期
half_size 是 ({expected_half_size:.6f}, {expected_half_size:.6f}, {expected_half_size:.6f}) 米。
请结合方块与锤子的相对尺寸判断是否明显偏大或偏小。

只输出 JSON：
{{
  "aligned": true,
  "target_actor": "block",
  "observed_color": "blue",
  "unexpected_changes": [],
  "diagnosis": "场景与需求是否一致，以及不一致的具体原因",
  "suggestions": ["若不一致，给出只修改 load_actors() 的具体建议"],
  "confidence": 0.0
}}
"""
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    response = provider.vision(
        prompt,
        image_path,
        model=model,
        max_tokens=512,
        temperature=0.0,
    )
    response_path.write_text(response + "\n", encoding="utf-8")
    from mea.taskgen import extract_json_response

    result = validate_vision_observation(extract_json_response(response), spec)
    result["provider_metadata"] = dict(provider.last_metadata)
    write_json(result_path, result)
    return result


def run_visual_self_reflection(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    provider: OpenAICompatibleProvider,
    *,
    seed: int,
    text_model: str,
    vision_model: str,
    max_repairs: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = json.loads((run_dir / "variant_spec.json").read_text(encoding="utf-8"))
    reflection_dir = run_dir / "reflection"
    reflection_dir.mkdir(parents=True, exist_ok=True)

    def observe(attempt_index: int) -> dict[str, Any]:
        attempt_dir = reflection_dir / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        scene_path = attempt_dir / "scene.json"
        image_path = attempt_dir / "render.png"
        scene = run_probe(
            repo_root,
            run_dir,
            manifest,
            seed=seed,
            expert=False,
            scene_json=scene_path,
            image=image_path,
            log_path=attempt_dir / "probe.log",
            raise_on_failure=False,
        )
        probe_passed = bool(
            scene.get("setup_success")
            and scene.get("render_success")
            and scene.get("rule_check", {}).get("passed")
            and scene.get("returncode") == 0
        )
        if probe_passed:
            vision = run_vision_check(
                provider,
                run_dir,
                spec,
                model=vision_model,
                image_path=image_path,
                prompt_path=attempt_dir / "vision_prompt.md",
                response_path=attempt_dir / "vision_response.txt",
                result_path=attempt_dir / "vision.json",
            )
        else:
            error = scene.get("error") or {}
            vision = {
                "aligned": False,
                "target_actor": "block",
                "expected_color": "blue",
                "observed_color": "unavailable",
                "color_matches": False,
                "unexpected_changes": ["scene_probe_failed"],
                "diagnosis": (
                    f"Scene setup/render/rule probe failed: "
                    f"{error.get('type', 'unknown')}: {error.get('message', '')}"
                ),
                "suggestions": [
                    "Repair load_actors() so setup, render, hammer/block actor checks pass."
                ],
                "confidence": 1.0,
                "passed": False,
                "provider_metadata": {},
            }
            write_json(attempt_dir / "vision.json", vision)
        return {
            "passed": bool(probe_passed and vision.get("passed")),
            "probe_passed": probe_passed,
            "scene_path": str(scene_path.relative_to(run_dir)),
            "image_path": str(image_path.relative_to(run_dir)),
            "vision_path": str((attempt_dir / "vision.json").relative_to(run_dir)),
            "vision": vision,
        }

    def repair(repair_index: int, observation: dict[str, Any]) -> dict[str, Any]:
        update_manifest(
            run_dir,
            status=f"visual_reflection_repair_{repair_index}",
        )
        result = repair_generated_method(
            repo_root,
            run_dir,
            provider,
            model=text_model,
            spec=spec,
            observation=observation,
            repair_index=repair_index,
            protected_before=manifest["protected_hashes_before"],
        )
        update_manifest(
            run_dir,
            static_validation=result["static_validation"],
        )
        return result

    summary = execute_reflection_loop(
        max_repairs=max_repairs,
        observe=observe,
        repair=repair,
    )
    write_json(reflection_dir / "summary.json", summary)
    if not summary["passed"]:
        raise VisualReflectionError(
            f"Visual Self-Reflection 用尽 {max_repairs} 次 repair: {summary}"
        )

    final_attempt = reflection_dir / f"attempt_{summary['final_attempt']:02d}"
    shutil.copy2(final_attempt / "render.png", run_dir / "evidence/initial_head.png")
    shutil.copy2(final_attempt / "vision.json", run_dir / "validation/vision.json")
    if (final_attempt / "vision_prompt.md").is_file():
        shutil.copy2(
            final_attempt / "vision_prompt.md",
            run_dir / "validation/vision_prompt.md",
        )
    if (final_attempt / "vision_response.txt").is_file():
        shutil.copy2(
            final_attempt / "vision_response.txt",
            run_dir / "validation/vision_response.txt",
        )
    final_scene = json.loads((final_attempt / "scene.json").read_text(encoding="utf-8"))
    final_vision = json.loads((final_attempt / "vision.json").read_text(encoding="utf-8"))
    return summary, final_scene, final_vision


def newest_eval_dir(
    repo_root: Path,
    before: set[Path],
    *,
    task_name: str = "beat_block_hammer",
    task_config: str = "demo_clean",
    checkpoint_setting: str = "demo_clean",
) -> Path | None:
    eval_root = (
        repo_root
        / "eval_result"
        / task_name
        / "ACT"
        / task_config
        / checkpoint_setting
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


def run_act(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    seed: int,
    gpu: int,
    num_episodes: int,
    telemetry_profile: str = "balanced_v1",
) -> dict[str, Any]:
    """Run a task-specific ACT checkpoint and attach videos to telemetry."""

    task_name = str(manifest["task_name"])
    task_config = str(manifest.get("task_config") or "demo_clean")
    checkpoint_setting = str(
        manifest.get("checkpoint_setting") or "demo_clean"
    )
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
        repo_root
        / "eval_result"
        / task_name
        / "ACT"
        / task_config
        / checkpoint_setting
    )
    before = {path for path in eval_root.glob("*") if path.is_dir()} if eval_root.exists() else set()
    command = [
        "env",
        f"PYTHON_BIN={sys.executable}",
        "bash",
        "policy/ACT/eval_mea.sh",
        task_name,
        task_config,
        checkpoint_setting,
        str(expert_data_num),
        str(policy_seed),
        str(gpu),
        str(num_episodes),
        manifest["task_module"],
        str(run_dir / "overlay.yml"),
        str(seed),
        str(telemetry_root),
        telemetry_profile,
    ]
    started = datetime.now().astimezone().isoformat()
    returncode = run_command(
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
        metadata.parent
        for metadata in telemetry_root.glob("episode_*/episode.json")
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
            index_issues.append(
                f"duplicate ACT telemetry index: {episode_index}"
            )
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
                "frame_semantics": "pre-action; contact in policy step k lies between adjacent frames",
            }
            write_json(metadata_path, metadata)
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
                str(path.relative_to(repo_root))
                for path in required_checkpoint_files
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
    write_json(run_dir / "evaluation/act.json", result)
    if not result["passed"]:
        raise RuntimeError(f"ACT {num_episodes}-episode 未通过: {result}")
    return result


def evaluate_run_telemetry(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    telemetry_root = run_dir / "evaluation/telemetry"
    summary = evaluate_telemetry_root(
        telemetry_root,
        user_request=manifest["user_request"],
        task_name=manifest["task_name"],
    )
    return {
        "artifact": str(
            (telemetry_root / "tool_results.json").relative_to(repo_root)
        ),
        "episode_count": summary["episode_count"],
        "tool_retrieval": summary["tool_retrieval"],
        "episodes": [
            {
                "episode_dir": episode["episode_dir"],
                "policy_name": episode["metadata"].get("policy_name"),
                "seed": episode["metadata"].get("seed"),
                "success": episode["metadata"].get("success"),
                "tool_results": episode["tool_results"],
            }
            for episode in summary["episodes"]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume-run",
        help="Resume an existing run_id without calling the text-generation stages again.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-module")
    parser.add_argument(
        "--mode",
        choices=["reuse", "force_codegen", "official"],
        default="force_codegen",
    )
    parser.add_argument("--text-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--vision-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--telemetry-profile",
        choices=["balanced_v1", "legacy_v1"],
        default="balanced_v1",
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--vision-check", action="store_true")
    parser.add_argument(
        "--max-reflections",
        type=int,
        default=2,
        help="Maximum number of CodeGen repairs after failed visual observations.",
    )
    parser.add_argument(
        "--reflection-fixture",
        choices=["wrong_color", "oversized_block"],
        help="Test-only injected visual mismatch used to exercise the repair loop.",
    )
    parser.add_argument("--run-act", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise SystemExit("--num-episodes 必须是正整数")
    repo_root = args.repo_root.expanduser().resolve()
    provider = None
    if (not args.resume_run and args.mode != "official") or args.vision_check:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=args.text_model,
            vision_model=args.vision_model,
            timeout=180.0,
        )
    if args.resume_run:
        if args.run_id:
            raise SystemExit("--resume-run 与 --run-id 不能同时使用")
        run_dir = repo_root / "mea/generated_tasks" / args.resume_run
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"run manifest 不存在: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        if not args.request:
            raise SystemExit("新 TaskGen run 必须提供 --request")
        if args.mode == "official":
            manifest = create_official_task_run(
                repo_root,
                args.request,
                task_name=args.task_name,
                task_module=args.task_module,
                run_id=args.run_id,
                telemetry_profile=args.telemetry_profile,
            )
        else:
            prototype = TaskGenPrototype(repo_root, provider, model=args.text_model)
            manifest = prototype.generate(
                args.request,
                task_name=args.task_name,
                mode=args.mode,
                run_id=args.run_id,
            )
        run_dir = repo_root / "mea/generated_tasks" / manifest["run_id"]

    requested_execution_backend = (
        (
            "both" if args.expert and args.run_act
            else "act" if args.run_act
            else "expert" if args.expert
            else "setup_probe"
        )
        if manifest.get("mode") == "official"
        else ("act" if args.run_act else "expert" if args.expert else "setup_probe")
    )
    update_manifest(
        run_dir,
        requested_execution_backend=requested_execution_backend,
    )

    try:
        if manifest.get("mode") == "official" and (
            args.vision_check or args.reflection_fixture
        ):
            raise RuntimeError(
                "official route bypasses generated-scene vision/reflection; "
                "use expert, act, or both execution without scene codegen"
            )
        if args.reflection_fixture:
            if args.resume_run:
                raise RuntimeError("reflection fixture 只允许用于新的 TaskGen run")
            if not args.vision_check:
                raise RuntimeError("reflection fixture 必须与 --vision-check 一起使用")
            spec = json.loads((run_dir / "variant_spec.json").read_text(encoding="utf-8"))
            fixture_function = {
                "wrong_color": inject_wrong_color_fixture,
                "oversized_block": inject_oversized_block_fixture,
            }[args.reflection_fixture]
            fixture = fixture_function(
                repo_root, run_dir, spec, manifest["protected_hashes_before"]
            )
            update_manifest(run_dir, reflection_fixture=fixture)

        scene = None
        if args.vision_check:
            if provider is None:
                raise RuntimeError("vision check 缺少 provider")
            reflection, reflected_scene, vision = run_visual_self_reflection(
                repo_root,
                run_dir,
                manifest,
                provider,
                seed=args.seed,
                text_model=args.text_model,
                vision_model=args.vision_model,
                max_repairs=args.max_reflections,
            )
            update_manifest(
                run_dir,
                status="vision_passed",
                visual_self_reflection=reflection,
                vision_validation=vision,
            )
            scene = reflected_scene

        if manifest.get("mode") == "official" and args.expert:
            scene = run_official_expert_episodes(
                repo_root,
                run_dir,
                manifest,
                start_seed=args.seed,
                num_episodes=args.num_episodes,
                telemetry_profile=args.telemetry_profile,
            )
            update_manifest(run_dir, status="probe_passed", scene_validation=scene)
        elif manifest.get("mode") == "official" and args.run_act:
            # ACT-only evaluates the learned policy; this probe validates only
            # simulator setup/render/rules and does not create expert evidence.
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                expert=False,
                telemetry_profile=args.telemetry_profile,
            )
            update_manifest(run_dir, status="probe_passed", scene_validation=scene)
        elif args.expert or args.run_act:
            expert_telemetry_dir = (
                run_dir
                / "evaluation/telemetry/expert"
                / f"episode_000_seed_{args.seed}"
            )
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                expert=True,
                telemetry_dir=expert_telemetry_dir,
                telemetry_profile=args.telemetry_profile,
            )
            update_manifest(run_dir, status="probe_passed", scene_validation=scene)
        elif args.probe and not args.vision_check:
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                expert=False,
                telemetry_profile=args.telemetry_profile,
            )
            update_manifest(run_dir, status="probe_passed", scene_validation=scene)
        elif scene is not None:
            write_json(run_dir / "validation/scene.json", scene)
            update_manifest(run_dir, scene_validation=scene)

        if args.run_act:
            if manifest["task_name"] == "beat_block_hammer":
                position_samples = collect_position_samples(
                    repo_root,
                    run_dir,
                    manifest,
                    start_seed=args.seed,
                    num_episodes=args.num_episodes,
                    first_scene=scene,
                )
            else:
                position_samples = {
                    "status": "not_applicable",
                    "reason": (
                        "official passthrough tasks have no BBH block-position "
                        "contract"
                    ),
                    "passed": True,
                    "samples": [],
                    "metrics": {},
                }
                write_json(
                    run_dir / "validation/position_samples.json",
                    position_samples,
                )
            update_manifest(run_dir, position_samples=position_samples)
            act = run_act(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                gpu=args.gpu,
                num_episodes=args.num_episodes,
                telemetry_profile=args.telemetry_profile,
            )
            alignment = {
                "status": "not_applicable",
                "passed": True,
                "reason": "paired expert/ACT execution was not requested",
                "expert_seeds": [],
                "act_seeds": act.get("actual_seeds", []),
            }
            if manifest.get("mode") == "official" and args.expert:
                expert_seeds = [
                    int(item["seed"])
                    for item in (scene or {}).get("expert_batch", {}).get(
                        "episodes", []
                    )
                ]
                act_seeds = [int(value) for value in act.get("actual_seeds", [])]
                aligned = expert_seeds == act_seeds
                alignment = {
                    "status": "passed" if aligned else "failed",
                    "passed": aligned,
                    "reason": (
                        "expert and ACT used the same ordered seeds"
                        if aligned
                        else "expert and ACT ordered seeds differ"
                    ),
                    "expert_seeds": expert_seeds,
                    "act_seeds": act_seeds,
                }
            write_json(
                run_dir / "evaluation/backend_seed_alignment.json",
                alignment,
            )
            update_manifest(
                run_dir,
                act_evaluation=act,
                backend_seed_alignment=alignment,
            )
            if not alignment["passed"]:
                raise RuntimeError(
                    "paired expert/ACT seed alignment failed: "
                    f"expert={alignment['expert_seeds']}, "
                    f"ACT={alignment['act_seeds']}"
                )
            trusted_tools = evaluate_run_telemetry(
                repo_root,
                run_dir,
                manifest,
            )
            update_manifest(
                run_dir,
                status="completed",
                failure=None,
                act_evaluation=act,
                execution_backends=(
                    ["expert", "ACT"] if args.expert else ["ACT"]
                ),
                backend_seed_alignment=alignment,
                trusted_tool_evaluation=trusted_tools,
            )
        else:
            updates: dict[str, Any] = {
                "status": "completed_without_act",
                "failure": None,
            }
            if args.expert:
                updates["execution_backends"] = ["expert"]
                updates["trusted_tool_evaluation"] = evaluate_run_telemetry(
                    repo_root,
                    run_dir,
                    manifest,
                )
            update_manifest(run_dir, **updates)
    except Exception as exc:
        update_manifest(
            run_dir,
            status="failed",
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise

    print(json.dumps(json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
