"""RoboTwin TaskGen simulator probe and terminal-authority helpers."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

CommandRunner = Callable[..., int]
JsonWriter = Callable[[Path, Any], None]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

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


def _generated_checker_error_payload(
    scene: Mapping[str, Any],
) -> dict[str, str] | None:
    error = scene.get("error")
    if not isinstance(error, Mapping):
        return None
    traceback = str(error.get("traceback") or "")
    if "generated_checker_success = bool(task.check_success())" not in traceback:
        return None
    return {
        "type": str(error.get("type") or "checker_error"),
        "message": str(error.get("message") or "").strip(),
    }


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
    discover_task_context: bool = False,
    task_context: Path | None = None,
    action_dimension: int = 0,
    command_runner: CommandRunner | None = None,
    json_writer: JsonWriter | None = None,
) -> dict[str, Any]:
    command_runner = command_runner or run_command
    json_writer = json_writer or _write_json
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
    if manifest.get("capability_id") == "scene_background_texture":
        # RoboTwin selects assets/background_texture/unseen only in eval mode.
        command.append("--eval-mode")
    if telemetry_dir is not None:
        command.extend(["--telemetry-dir", str(telemetry_dir)])
    if visual_capture_profile_id is not None:
        command.extend(["--visual-capture-profile", visual_capture_profile_id])
    if discover_task_context:
        command.append("--discover-task-context")
        command.extend(["--action-dimension", str(action_dimension)])
    if task_context is not None:
        command.extend(["--task-context", str(task_context)])

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
        returncode = command_runner(
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
        if (
            returncode != 2
            or _generated_checker_error_payload(scene) is not None
        ):
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
    json_writer(scene_json, scene)
    if raise_on_failure and returncode != 0:
        error = scene.get("error")
        detail = ""
        if isinstance(error, Mapping):
            error_type = str(error.get("type") or "probe_error")
            error_message = str(error.get("message") or "").strip()
            detail = (
                f": {error_type}: {error_message}"
                if error_message
                else f": {error_type}"
            )
        raise RuntimeError(
            f"setup/expert probe failed, returncode={returncode}{detail}"
        )
    return scene
def _tracked_actor_heights(scene: Mapping[str, Any]) -> dict[str, float]:
    """Return compact simulator-authoritative actor heights for repair feedback."""

    heights: dict[str, float] = {}
    terminal_actors = scene.get("expert_terminal_tracked_actors")
    tracked_actors = (
        terminal_actors
        if isinstance(terminal_actors, list)
        else scene.get("tracked_actors")
    )
    for actor in tracked_actors or []:
        if not isinstance(actor, Mapping):
            continue
        actor_id = str(actor.get("id") or "").strip()
        position = actor.get("position")
        if (
            not actor_id
            or not isinstance(position, list)
            or len(position) < 3
            or isinstance(position[2], bool)
            or not isinstance(position[2], (int, float))
        ):
            continue
        heights[actor_id] = round(float(position[2]), 6)
    return heights


def _tracked_actor_positions(
    scene: Mapping[str, Any],
) -> dict[str, list[float]]:
    """Return compact current or expert-terminal xyz state for repair."""

    positions: dict[str, list[float]] = {}
    terminal_actors = scene.get("expert_terminal_tracked_actors")
    tracked_actors = (
        terminal_actors
        if isinstance(terminal_actors, list)
        else scene.get("tracked_actors")
    )
    for actor in tracked_actors or []:
        if not isinstance(actor, Mapping):
            continue
        actor_id = str(actor.get("id") or "").strip()
        position = actor.get("position")
        if (
            not actor_id
            or not isinstance(position, list)
            or len(position) < 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in position[:3]
            )
        ):
            continue
        positions[actor_id] = [
            round(float(value), 6) for value in position[:3]
        ]
    return positions


def _robot_tcp_positions(
    scene: Mapping[str, Any],
    field: str,
) -> dict[str, list[float]]:
    """Return finite optional left/right TCP coordinates from probe evidence."""

    raw_positions = scene.get(field)
    if not isinstance(raw_positions, Mapping):
        return {}
    positions: dict[str, list[float]] = {}
    for side in ("left", "right"):
        position = raw_positions.get(side)
        if (
            not isinstance(position, (list, tuple))
            or len(position) < 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in position[:3]
            )
        ):
            continue
        positions[side] = [round(float(value), 6) for value in position[:3]]
    return positions


def _expert_terminal_authority_failure(
    expert: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Separate an unsolved expert scene from a generated-checker mismatch."""

    outcome = expert.get("expert")
    if not isinstance(outcome, Mapping):
        outcome = {}
    plan_success = outcome.get("plan_success")
    official_success = outcome.get("official_core_predicate_satisfied")
    if not isinstance(official_success, bool):
        official_success = outcome.get("official_check_success")
    error = expert.get("error")
    if isinstance(error, Mapping):
        reason = "expert_execution_error"
    elif plan_success is False:
        reason = "expert_plan_unsuccessful"
    elif official_success is False:
        reason = "official_success_false_after_expert_plan"
    elif plan_success is not True or official_success is not True:
        reason = "expert_outcome_incomplete"
    else:
        return None
    result: dict[str, Any] = {
        "reason": reason,
        "plan_success": plan_success,
        "generated_checker_success": outcome.get("check_success"),
        "official_core_predicate_satisfied": official_success,
        "expert_terminal_actor_z_m": _tracked_actor_heights(expert),
        "repair_scope": "scene_or_expert_plan_not_checker_only",
    }
    if isinstance(error, Mapping):
        result["expert_error"] = {
            "type": str(error.get("type") or "probe_error"),
            "message": str(error.get("message") or "").strip(),
        }
    return result


def _generated_checker_execution_failure(
    expert: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Identify errors raised after the expert plan enters generated checker code."""

    error = _generated_checker_error_payload(expert)
    if error is None:
        return None
    return {
        "reason": "generated_checker_execution_error",
        "error": error,
        "repair_scope": "checker_only_after_expert_action",
    }


def _checker_fixture_failure_diagnosis(
    fixtures: list[dict[str, Any]],
    *,
    setup: Mapping[str, Any],
    expert: Mapping[str, Any],
    success_contract: Mapping[str, Any] | None = None,
) -> str:
    """Give one bounded regeneration concrete semantic evidence, not a return code."""

    failed = [
        {
            "fixture_id": item["fixture_id"],
            "expected": item["expected"],
            "observed": item["observed"],
        }
        for item in fixtures
        if not item["passed"]
    ]
    initial_heights = _tracked_actor_heights(setup)
    terminal_heights = _tracked_actor_heights(expert)
    contract = dict(success_contract or {})
    target_actor_id = str(contract.get("target_actor_id") or "").strip()
    minimum_height = contract.get("minimum_height_m")
    lift_boundary = (
        float(minimum_height)
        if not isinstance(minimum_height, bool)
        and isinstance(minimum_height, (int, float))
        and math.isfinite(float(minimum_height))
        else None
    )
    evidence: dict[str, Any] = {
        "failed_fixtures": failed,
        "initial_actor_z_m": initial_heights,
        "expert_terminal_actor_z_m": terminal_heights,
        "initial_actor_xyz_m": _tracked_actor_positions(setup),
        "expert_terminal_actor_xyz_m": _tracked_actor_positions(expert),
    }
    initial_robot_tcp = _robot_tcp_positions(
        setup,
        "initial_robot_tcp_xyz_m",
    )
    if initial_robot_tcp:
        evidence["initial_robot_tcp_xyz_m"] = initial_robot_tcp
    expert_terminal_robot_tcp = _robot_tcp_positions(
        expert,
        "expert_terminal_robot_tcp_xyz_m",
    )
    if expert_terminal_robot_tcp:
        evidence["expert_terminal_robot_tcp_xyz_m"] = expert_terminal_robot_tcp
    if lift_boundary is not None:
        evidence["official_lift_contract"] = {
            "target_actor_id": target_actor_id or None,
            "minimum_height_m": lift_boundary,
        }
    return (
        "generated checker failed live negative/positive fixtures: "
        + json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    )
__all__ = ["run_command", "run_probe"]
