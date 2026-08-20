"""Hy-VLA rollout hook for the backend-neutral RoboTwin MethodRuntime.

The official Hy-VLA policy remains in its own Python environment and is
started explicitly with ``experiments/paper/robotwin_hyvla/policy_server.py``.
This module owns only the RoboTwin side of the validated loopback protocol.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import MaterializedCandidate, RolloutRequest
from mea.visual_capture import EVENT_KEYFRAMES_PROFILE
from mea.robotwin.smolvla_rollout import (
    _PolicyClient,
    _checker_outcome_snapshot,
    _persist_telemetry_outcome_semantics,
    _require_generated_task_simulator_source,
    _resolved_demo_clean_args,
)


class HyVLARolloutError(RuntimeError):
    """Raised when the external Hy-VLA contract or episode is invalid."""


def _request_payload(
    observation: Mapping[str, Any],
    instruction: str,
) -> dict[str, Any]:
    import numpy as np

    camera_observation = observation["observation"]
    images = {
        name: np.ascontiguousarray(
            camera_observation[name]["rgb"],
            dtype=np.uint8,
        )
        for name in ("head_camera", "left_camera", "right_camera")
    }
    endpose = observation["endpose"]
    return {
        "command": "act",
        "images": {
            key: {"shape": list(value.shape), "data": value.tobytes()}
            for key, value in images.items()
        },
        "endpose": {
            "left_endpose": np.asarray(endpose["left_endpose"]).tolist(),
            "left_gripper": float(endpose["left_gripper"]),
            "right_endpose": np.asarray(endpose["right_endpose"]).tolist(),
            "right_gripper": float(endpose["right_gripper"]),
        },
        "task": instruction,
    }


def run_hyvla_robotwin_episode(
    *,
    task_name: str,
    task_module: str,
    seed: int,
    output_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 18781,
    timeout_seconds: float = 180.0,
    policy_instruction: str | None = None,
    repo_root: str | Path | None = None,
    telemetry_profile: str | None = None,
    task_schema: Mapping[str, Any] | None = None,
    visual_capture_profile_id: str = EVENT_KEYFRAMES_PROFILE,
    outcome_metric: str = "official_check_success",
) -> dict[str, Any]:
    """Run one official or TaskGen-materialized task via external Hy-VLA."""

    import imageio.v3 as iio
    import numpy as np

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _require_generated_task_simulator_source(
        task_module=task_module,
        repo_root=repo_root,
    )
    module = importlib.import_module(task_module)
    task_class = getattr(module, task_name, None)
    if not isinstance(task_class, type):
        raise HyVLARolloutError(
            f"{task_module!r} does not expose task class {task_name!r}"
        )
    instruction = (
        policy_instruction.strip()
        if isinstance(policy_instruction, str) and policy_instruction.strip()
        else task_name.replace("_", " ")
    )
    task = task_class()
    started = time.perf_counter()
    actions: list[list[float]] = []
    action_latencies: list[float] = []
    video_frames: list[Any] = []
    network_forwards = 0
    recorder = None
    recorder_metadata: dict[str, Any] | None = None
    telemetry_episode_dir: Path | None = None
    rollout_error: BaseException | None = None
    try:
        task.setup_demo(
            now_ep_num=0,
            seed=int(seed),
            is_test=True,
            **_resolved_demo_clean_args(task_name),
        )
        if telemetry_profile is not None:
            if repo_root is None:
                raise HyVLARolloutError(
                    "semantic telemetry requires repo_root"
                )
            from mea.toolkit import EpisodeRecorder

            telemetry_episode_dir = (
                destination
                / "telemetry"
                / "act"
                / f"episode_000_seed_{int(seed)}"
            )
            recorder = EpisodeRecorder(
                repo_root,
                telemetry_episode_dir,
                task_name=task_name,
                seed=int(seed),
                episode_index=0,
                policy_name="Hy-VLA",
                task_module=task_module,
                task_config="demo_clean",
                checkpoint_setting="shared_official",
                telemetry_profile_id=telemetry_profile,
                visual_capture_profile_id=visual_capture_profile_id,
                task_schema=task_schema,
            )
            task._mea_recorder = recorder
            recorder.start(task)

        observation = task.get_obs()
        initial_head = np.ascontiguousarray(
            observation["observation"]["head_camera"]["rgb"],
            dtype=np.uint8,
        )
        iio.imwrite(destination / "initial_head.png", initial_head)
        video_frames.append(initial_head)

        with _PolicyClient(host, port, timeout_seconds) as client:
            client.request({"command": "reset", "seed": int(seed)})
            while len(actions) < int(task.step_lim) and not task.eval_success:
                response = client.request(
                    _request_payload(observation, instruction)
                )
                action = np.asarray(response.get("action"), dtype=np.float32)
                if action.shape != (16,) or not np.isfinite(action).all():
                    raise HyVLARolloutError(
                        f"invalid Hy-VLA action: shape={action.shape}"
                    )
                task.take_action(action, action_type="ee")
                actions.append(action.tolist())
                action_latencies.append(float(response["latency_seconds"]))
                network_forwards += int(response["network_forward"])
                observation = task.get_obs()
                if len(actions) % 25 == 0:
                    video_frames.append(
                        np.ascontiguousarray(
                            observation["observation"]["head_camera"]["rgb"],
                            dtype=np.uint8,
                        )
                    )

            checker_outcomes = _checker_outcome_snapshot(
                task,
                active_checker_metric=outcome_metric,
            )
            success = bool(checker_outcomes["active_checker_success"])
            final_head = np.ascontiguousarray(
                observation["observation"]["head_camera"]["rgb"],
                dtype=np.uint8,
            )
            iio.imwrite(destination / "final_head.png", final_head)
            if not np.array_equal(video_frames[-1], final_head):
                video_frames.append(final_head)
            video_path = destination / "episode0.mp4"
            iio.imwrite(video_path, video_frames, fps=5)
            if recorder is not None:
                task._mea_recorder = None
                recorder_metadata = recorder.finish(task, success=success)
                recorder_metadata = _persist_telemetry_outcome_semantics(
                    telemetry_episode_dir,
                    recorder_metadata,
                    checker_outcomes,
                )
                recorder = None

            result = {
                "schema_version": 1,
                "task": task_name,
                "task_module": task_module,
                "task_config": "demo_clean",
                "seed": int(seed),
                "policy": "Hy-VLA",
                "policy_instruction": instruction,
                "action_semantics": "dual-arm 16D EE wxyz",
                "step_limit": int(task.step_lim),
                "actions_executed": len(actions),
                "network_forwards": network_forwards,
                "action_latencies_seconds": action_latencies,
                "success": success,
                "eval_success": checker_outcomes["episode_latched_success"],
                **checker_outcomes,
                "actions": actions,
                "wall_seconds": time.perf_counter() - started,
                "semantic_telemetry": {
                    "ready": recorder_metadata is not None,
                    "profile": (
                        telemetry_profile
                        if recorder_metadata is not None
                        else None
                    ),
                    "episode_dir": (
                        str(telemetry_episode_dir)
                        if recorder_metadata is not None
                        else None
                    ),
                },
            }
            result_path = destination / "result.json"
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifacts = {
                "result": str(result_path),
                "video": str(video_path),
                "initial_frame": str(destination / "initial_head.png"),
                "final_frame": str(destination / "final_head.png"),
            }
            if telemetry_episode_dir is not None:
                artifacts["telemetry_episode"] = str(telemetry_episode_dir)
            return {
                "success": success,
                "episode": result,
                "artifacts": artifacts,
                "metadata": {
                    "policy_backend": "hyvla",
                    "policy_transport": "localhost_official_wrapper",
                    "trace_level": (
                        "task_schema_semantic_telemetry"
                        if recorder_metadata is not None
                        else "untyped_policy_io"
                    ),
                    "semantic_telemetry_ready": recorder_metadata is not None,
                    "video_sampling": "every_25_actions",
                },
            }
    except BaseException as exc:
        rollout_error = exc
        if recorder is not None:
            recorder.record_error(exc)
        raise
    finally:
        if recorder is not None:
            task._mea_recorder = None
            try:
                recorder.finish(
                    task,
                    success=False,
                    error=(
                        {
                            "type": type(rollout_error).__name__,
                            "message": str(rollout_error),
                        }
                        if rollout_error is not None
                        else None
                    ),
                )
            except Exception:
                if rollout_error is None:
                    raise
        task.close_env(clear_cache=True)


class HyVLARobotwinRolloutRunner:
    """MethodRuntime callable for an explicitly started Hy-VLA server."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 18781,
        timeout_seconds: float = 180.0,
        repo_root: str | Path | None = None,
        telemetry_profile: str = "balanced_v1",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_seconds = float(timeout_seconds)
        self.repo_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else None
        )
        self.telemetry_profile = str(telemetry_profile)

    def __call__(
        self,
        *,
        candidate: MaterializedCandidate,
        request: RolloutRequest,
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        policy = candidate.task_contract.get("policy")
        if not isinstance(policy, Mapping) or policy.get("backend") != "hyvla":
            raise HyVLARolloutError(
                "Hy-VLA runner requires a hyvla policy binding"
            )
        task_name = str(manifest.get("task_name") or "").strip()
        task_module = str(manifest.get("task_module") or "").strip()
        if not task_name or not task_module:
            raise HyVLARolloutError(
                "rollout manifest requires task_name and task_module"
            )
        semantic_telemetry = bool(
            candidate.task_contract.get("task_schema_available")
        )
        runtime_schema = candidate.task_contract.get("task_schema")
        artifact_summary = manifest.get("task_artifact_summary")
        artifact_summary = (
            artifact_summary if isinstance(artifact_summary, Mapping) else {}
        )
        return run_hyvla_robotwin_episode(
            task_name=task_name,
            task_module=task_module,
            seed=request.seed,
            output_dir=request.output_dir,
            host=self.host,
            port=self.port,
            timeout_seconds=self.timeout_seconds,
            policy_instruction=policy.get("task_instruction"),
            repo_root=self.repo_root if semantic_telemetry else None,
            telemetry_profile=(
                self.telemetry_profile if semantic_telemetry else None
            ),
            task_schema=(
                runtime_schema
                if semantic_telemetry and isinstance(runtime_schema, Mapping)
                else None
            ),
            visual_capture_profile_id=str(
                candidate.task_contract.get("visual_capture_profile_id")
                or EVENT_KEYFRAMES_PROFILE
            ),
            outcome_metric=str(
                artifact_summary.get("success_outcome_label")
                or "official_check_success"
            ),
        )


__all__ = [
    "HyVLARobotwinRolloutRunner",
    "HyVLARolloutError",
    "run_hyvla_robotwin_episode",
]
