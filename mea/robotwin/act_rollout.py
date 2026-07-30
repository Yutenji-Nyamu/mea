"""Native ACT rollout runner for the shared RoboTwin MethodRuntime.

TaskGen owns candidate materialization.  This adapter owns only the policy
episode: invoke the existing ACT evaluator, read the single aligned telemetry
episode, and return the compact observation expected by
``RoboTwinMethodBackend``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from mea.method_runtime import MaterializedCandidate, RolloutRequest
from mea.taskgen.act_runtime import run_act


class ACTRobotwinRolloutError(RuntimeError):
    """Raised when an ACT episode cannot become MethodRuntime evidence."""


CommandRunner = Callable[..., int]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_logged(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        return process.wait()


def _policy_success(path: Path) -> bool:
    if not path.is_file():
        raise ACTRobotwinRolloutError(
            f"ACT result file is unavailable: {path}"
        )
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    if len(values) != 1:
        raise ACTRobotwinRolloutError(
            "native ACT MethodRuntime requires exactly one policy result"
        )
    return values[0] > 0.5


class ACTRobotwinRolloutRunner:
    """Execute one already-materialized RoboTwin candidate with ACT."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        gpu: int = 0,
        telemetry_profile: str = "balanced_v1",
        python_executable: str = sys.executable,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.gpu = int(gpu)
        self.telemetry_profile = str(telemetry_profile).strip()
        self.python_executable = str(python_executable)
        self.command_runner = command_runner or _run_logged
        if not self.telemetry_profile:
            raise ACTRobotwinRolloutError(
                "telemetry_profile must be non-empty"
            )

    def __call__(
        self,
        *,
        candidate: MaterializedCandidate,
        request: RolloutRequest,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        policy = candidate.task_contract.get("policy")
        if not isinstance(policy, Mapping) or policy.get("backend") != "act":
            raise ACTRobotwinRolloutError(
                "ACT runner requires an ACT policy binding"
            )
        if request.seed < 0:
            raise ACTRobotwinRolloutError("ACT seed must be non-negative")
        overlay = Path(str(manifest.get("overlay") or "")).expanduser()
        if not overlay.is_absolute():
            overlay = (self.repo_root / overlay).resolve()
        run_dir = overlay.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        if not overlay.is_file():
            overlay.write_text("{}\n", encoding="utf-8")

        runtime_manifest = {
            **dict(manifest),
            "task_name": candidate.task_contract["task_name"],
            "task_module": candidate.task_contract["task_module"],
            "task_config": str(policy.get("task_config") or "demo_clean"),
            "checkpoint_setting": str(
                policy.get("checkpoint_setting") or "demo_clean"
            ),
            "expert_data_num": int(policy.get("expert_data_num") or 50),
            "policy_seed": int(policy.get("policy_seed") or 0),
        }
        result = run_act(
            self.repo_root,
            run_dir,
            runtime_manifest,
            seed=request.seed,
            gpu=self.gpu,
            num_episodes=1,
            command_runner=self.command_runner,
            json_writer=_write_json,
            telemetry_profile=self.telemetry_profile,
            python_executable=self.python_executable,
        )
        telemetry_root = self.repo_root / str(result["telemetry_root"])
        episode_dirs = sorted(
            path.parent for path in telemetry_root.glob("episode_*/episode.json")
        )
        if len(episode_dirs) != 1:
            raise ACTRobotwinRolloutError(
                "native ACT MethodRuntime requires one telemetry episode"
            )
        episode_dir = episode_dirs[0]
        metadata_path = episode_dir / "episode.json"
        try:
            episode_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ACTRobotwinRolloutError(
                f"ACT episode metadata is invalid: {metadata_path}"
            ) from exc
        success = _policy_success(run_dir / "evaluation/_result.txt")
        video = episode_dir / "video.mp4"
        schema = episode_dir / "schema.json"
        semantic_trace = episode_dir / "semantic_trace.npz"
        events = episode_dir / "events.jsonl"
        semantic_ready = bool(
            schema.is_file()
            and semantic_trace.is_file()
            and schema.stat().st_size > 0
            and semantic_trace.stat().st_size > 0
        )
        return {
            "success": success,
            "episode": {
                **episode_metadata,
                "policy_success": success,
                "episode_dir": str(episode_dir),
                "schema": str(schema) if schema.is_file() else None,
                "semantic_trace": (
                    str(semantic_trace) if semantic_trace.is_file() else None
                ),
                "events": str(events) if events.is_file() else None,
            },
            "artifacts": {
                "episode_dir": str(episode_dir),
                "video": str(video) if video.is_file() else "",
                "schema": str(schema) if schema.is_file() else "",
                "semantic_trace": (
                    str(semantic_trace) if semantic_trace.is_file() else ""
                ),
                "events": str(events) if events.is_file() else "",
                "act_result": str(run_dir / "evaluation/act.json"),
            },
            "metadata": {
                "policy_backend": "act",
                "semantic_telemetry_ready": semantic_ready,
                "actual_seeds": list(result["actual_seeds"]),
                "policy_success": success,
            },
        }


__all__ = [
    "ACTRobotwinRolloutError",
    "ACTRobotwinRolloutRunner",
]
