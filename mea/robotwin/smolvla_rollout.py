"""SmolVLA rollout runner for the production RoboTwin MethodRuntime.

LeRobot and RoboTwin currently require incompatible Python environments.  A
LeRobot policy server therefore exposes action chunks on localhost while this
module owns the native RoboTwin episode.  The wire contract is intentionally
the same one validated by the paper pilot, but production code does not import
from ``experiments``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import MaterializedCandidate, RolloutRequest
from mea.visual_capture import EVENT_KEYFRAMES_PROFILE


class SmolVLARolloutError(RuntimeError):
    """Raised when policy transport or simulator execution fails."""


_HEADER = struct.Struct("!Q")
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise SmolVLARolloutError("SmolVLA policy server closed the socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(connection: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=5)
    connection.sendall(_HEADER.pack(len(payload)))
    connection.sendall(payload)


def _recv_message(connection: socket.socket) -> Mapping[str, Any]:
    size = _HEADER.unpack(_recv_exact(connection, _HEADER.size))[0]
    value = pickle.loads(_recv_exact(connection, size))
    if not isinstance(value, Mapping):
        raise SmolVLARolloutError("SmolVLA policy response must be an object")
    return value


@dataclass
class _PolicyClient:
    host: str
    port: int
    timeout_seconds: float
    connection: socket.socket | None = None

    def __enter__(self) -> "_PolicyClient":
        if self.host not in _LOOPBACK_HOSTS:
            raise SmolVLARolloutError(
                "SmolVLA policy transport must use a loopback address"
            )
        self.connection = socket.create_connection(
            (self.host, self.port),
            timeout=self.timeout_seconds,
        )
        self.connection.settimeout(self.timeout_seconds)
        return self

    def request(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.connection is None:
            raise SmolVLARolloutError("SmolVLA policy client is not connected")
        _send_message(self.connection, dict(value))
        response = _recv_message(self.connection)
        if response.get("ok") is not True:
            raise SmolVLARolloutError(
                str(response.get("error") or "SmolVLA policy request failed")
            )
        return response

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.connection is not None:
            try:
                self.request({"command": "close"})
            except Exception:
                pass
            self.connection.close()
            self.connection = None


def _resolved_demo_clean_args(task_name: str) -> dict[str, Any]:
    import yaml
    from envs import CONFIGS_PATH

    def read_yaml(path: str | Path) -> dict[str, Any]:
        with Path(path).open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, Mapping):
            raise SmolVLARolloutError(f"RoboTwin config is invalid: {path}")
        return dict(value)

    args = read_yaml(Path(CONFIGS_PATH) / "demo_clean.yml")
    embodiments = read_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")
    embodiment = args.get("embodiment", ["aloha-agilex"])
    if not isinstance(embodiment, list) or len(embodiment) != 1:
        raise SmolVLARolloutError(
            f"expected one dual-arm embodiment, got {embodiment!r}"
        )
    robot_file = Path(embodiments[embodiment[0]]["file_path"])
    robot_config = read_yaml(robot_file / "config.yml")
    cameras = read_yaml(Path(CONFIGS_PATH) / "_camera_config.yml")
    head_type = args["camera"]["head_camera_type"]
    args.update(
        left_robot_file=str(robot_file),
        right_robot_file=str(robot_file),
        dual_arm_embodied=True,
        left_embodiment_config=robot_config,
        right_embodiment_config=robot_config,
        head_camera_h=cameras[head_type]["h"],
        head_camera_w=cameras[head_type]["w"],
        task_name=task_name,
        task_config="demo_clean",
        render_freq=0,
        eval_mode=True,
        save_data=False,
        eval_video_save_dir=None,
    )
    return args


def _encode_observation(
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    import numpy as np

    camera_observation = observation["observation"]
    images = {
        name: np.ascontiguousarray(
            camera_observation[name]["rgb"],
            dtype=np.uint8,
        )
        for name in ("head_camera", "left_camera", "right_camera")
    }
    state = np.asarray(
        observation["joint_action"]["vector"],
        dtype=np.float32,
    )
    if state.shape != (14,):
        raise SmolVLARolloutError(
            f"expected RoboTwin state shape (14,), got {state.shape}"
        )
    return images, state


def _checker_outcome_snapshot(
    task: Any,
    *,
    active_checker_metric: str,
) -> dict[str, Any]:
    """Separate the active checker, untouched official core, and episode latch."""

    if active_checker_metric not in {
        "official_check_success",
        "generated_check_success",
    }:
        raise SmolVLARolloutError(
            f"unsupported active checker metric: {active_checker_metric!r}"
        )
    active_checker_final = bool(task.check_success())
    episode_latched_success = bool(getattr(task, "eval_success", False))
    active_checker_success = bool(
        episode_latched_success or active_checker_final
    )
    official_core_checker = getattr(
        task,
        "mea_official_check_success",
        None,
    )
    generated_checker_active = (
        active_checker_metric == "generated_check_success"
    )
    if generated_checker_active and not callable(official_core_checker):
        raise SmolVLARolloutError(
            "generated checker lacks untouched official-core authority"
        )
    official_core = bool(
        official_core_checker()
        if callable(official_core_checker)
        else active_checker_final
    )
    return {
        "active_checker_metric": active_checker_metric,
        "active_checker_success": active_checker_success,
        "active_checker_final_predicate": active_checker_final,
        "generated_checker_success": (
            active_checker_success if generated_checker_active else None
        ),
        "official_check_success": (
            official_core
            if generated_checker_active
            else active_checker_success
        ),
        "official_core_predicate_satisfied": official_core,
        "episode_latched_success": episode_latched_success,
    }


def _persist_telemetry_outcome_semantics(
    episode_dir: Path,
    recorder_metadata: Mapping[str, Any],
    outcomes: Mapping[str, Any],
) -> dict[str, Any]:
    """Make the recorder episode expose the same checker semantics as rollout."""

    metadata = dict(recorder_metadata)
    metadata.update(dict(outcomes))
    if outcomes.get("generated_checker_success") is None:
        metadata.pop("generated_checker_success", None)
    episode_path = episode_dir / "episode.json"
    episode_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _require_generated_task_simulator_source(
    *,
    task_module: str,
    repo_root: str | Path | None,
) -> None:
    """Keep TaskGen validation and rollout on the same RoboTwin source."""

    if not task_module.startswith("mea.generated_tasks."):
        return
    if repo_root is None:
        raise SmolVLARolloutError(
            "generated TaskGen rollout requires the MEA repository root"
        )
    envs = importlib.import_module("envs")
    package_file = getattr(envs, "__file__", None)
    if not isinstance(package_file, str):
        raise SmolVLARolloutError(
            "cannot identify the active RoboTwin envs package"
        )
    actual = Path(package_file).resolve().parent
    expected = Path(repo_root).expanduser().resolve() / "envs"
    if actual != expected:
        raise SmolVLARolloutError(
            "TaskGen was validated against the MEA RoboTwin fork but rollout "
            f"resolved envs from {actual}; place {expected.parent} first on "
            "PYTHONPATH"
        )


def run_smolvla_robotwin_episode(
    *,
    task_name: str,
    task_module: str,
    seed: int,
    output_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 18771,
    timeout_seconds: float = 120.0,
    policy_instruction: str | None = None,
    repo_root: str | Path | None = None,
    telemetry_profile: str | None = None,
    task_schema: Mapping[str, Any] | None = None,
    visual_capture_profile_id: str = EVENT_KEYFRAMES_PROFILE,
    outcome_metric: str = "official_check_success",
) -> dict[str, Any]:
    """Execute one importable RoboTwin task module with SmolVLA.

    Official modules are the validated production path.  A generated module is
    eligible only after TaskGen has registered it as an importable module; this
    runner does not itself install run-local source files.
    """

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
        raise SmolVLARolloutError(
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
    observed_states: list[list[float]] = []
    chunk_latencies: list[float] = []
    video_frames: list[Any] = []
    actions_executed = 0
    chunk_count = 0
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
                raise SmolVLARolloutError(
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
                policy_name="SmolVLA",
                task_module=task_module,
                task_config="demo_clean",
                checkpoint_setting="shared_official",
                telemetry_profile_id=telemetry_profile,
                visual_capture_profile_id=visual_capture_profile_id,
                task_schema=task_schema,
            )
            task._mea_recorder = recorder
            try:
                recorder.start(task)
            except Exception:
                task._mea_recorder = None
                recorder = None
                raise
        observation = task.get_obs()
        initial_images, initial_state = _encode_observation(observation)
        iio.imwrite(
            destination / "initial_head.png",
            initial_images["head_camera"],
        )
        video_frames.append(initial_images["head_camera"])
        observed_states.append(initial_state.tolist())

        with _PolicyClient(host, port, timeout_seconds) as client:
            client.request({"command": "reset", "seed": int(seed)})
            while (
                actions_executed < int(task.step_lim)
                and not task.eval_success
            ):
                images, state = _encode_observation(observation)
                encoded_images = {
                    key: {
                        "shape": list(value.shape),
                        "data": value.tobytes(),
                    }
                    for key, value in images.items()
                }
                response = client.request(
                    {
                        "command": "act_chunk",
                        "images": encoded_images,
                        "state": state.tolist(),
                        "task": instruction,
                    }
                )
                if response.get("state_dim") != 14:
                    raise SmolVLARolloutError(
                        "SmolVLA server processed an unexpected state shape"
                    )
                chunk = np.asarray(response.get("actions"), dtype=np.float32)
                if chunk.shape != (50, 14) or not np.isfinite(chunk).all():
                    raise SmolVLARolloutError(
                        f"invalid SmolVLA action chunk: {chunk.shape}"
                    )
                chunk_count += 1
                chunk_latencies.append(float(response["latency_seconds"]))
                for action in chunk:
                    if (
                        actions_executed >= int(task.step_lim)
                        or task.eval_success
                    ):
                        break
                    task.take_action(action)
                    actions.append(action.tolist())
                    actions_executed += 1
                observation = task.get_obs()
                chunk_images, chunk_state = _encode_observation(observation)
                video_frames.append(chunk_images["head_camera"])
                observed_states.append(chunk_state.tolist())

            checker_outcomes = _checker_outcome_snapshot(
                task,
                active_checker_metric=outcome_metric,
            )
            success = bool(checker_outcomes["active_checker_success"])
            final_images, final_state = _encode_observation(observation)
            iio.imwrite(
                destination / "final_head.png",
                final_images["head_camera"],
            )
            if not np.array_equal(
                video_frames[-1],
                final_images["head_camera"],
            ):
                video_frames.append(final_images["head_camera"])
            video_path = destination / "episode0.mp4"
            iio.imwrite(video_path, video_frames, fps=5)
            if recorder is not None:
                task._mea_recorder = None
                recorder_metadata = recorder.finish(
                    task,
                    success=success,
                )
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
                "policy": "SmolVLA",
                "policy_instruction": instruction,
                "step_limit": int(task.step_lim),
                "actions_executed": actions_executed,
                "chunk_count": chunk_count,
                "chunk_latencies_seconds": chunk_latencies,
                "success": success,
                # Compatibility alias retained for old result readers.  New
                # evidence uses the three explicit checker channels below.
                "eval_success": checker_outcomes[
                    "episode_latched_success"
                ],
                **checker_outcomes,
                "initial_state": initial_state.tolist(),
                "final_state": final_state.tolist(),
                "chunk_boundary_states": observed_states,
                "actions": actions,
                "wall_seconds": time.perf_counter() - started,
                "semantic_telemetry": (
                    {
                        "ready": True,
                        "profile": telemetry_profile,
                        "episode_dir": str(telemetry_episode_dir),
                    }
                    if recorder_metadata is not None
                    else {
                        "ready": False,
                        "profile": None,
                        "episode_dir": None,
                    }
                ),
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
                artifacts["telemetry_episode"] = str(
                    telemetry_episode_dir
                )
            return {
                "success": success,
                "episode": result,
                "artifacts": artifacts,
                "metadata": {
                    "policy_backend": "smolvla",
                    "policy_transport": "localhost_action_chunks",
                    "trace_level": (
                        "task_schema_semantic_telemetry"
                        if recorder_metadata is not None
                        else "untyped_policy_io"
                    ),
                    "semantic_telemetry_ready": (
                        recorder_metadata is not None
                    ),
                    "video_sampling": "action_chunk_boundaries",
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
            error_payload = (
                {
                    "type": type(rollout_error).__name__,
                    "message": str(rollout_error),
                }
                if rollout_error is not None
                else None
            )
            try:
                recorder.finish(
                    task,
                    success=False,
                    error=error_payload,
                )
            except Exception:
                if rollout_error is None:
                    raise
        task.close_env(clear_cache=True)


class SmolVLARobotwinRolloutRunner:
    """MethodRuntime rollout callable for a running SmolVLA policy server."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 18771,
        timeout_seconds: float = 120.0,
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
        if not isinstance(policy, Mapping) or policy.get("backend") != "smolvla":
            raise SmolVLARolloutError(
                "SmolVLA runner requires a smolvla policy binding"
            )
        task_name = str(manifest.get("task_name") or "").strip()
        task_module = str(manifest.get("task_module") or "").strip()
        if not task_name or not task_module:
            raise SmolVLARolloutError(
                "rollout manifest requires task_name and task_module"
            )
        semantic_telemetry = bool(
            candidate.task_contract.get("task_schema_available")
        )
        runtime_schema = candidate.task_contract.get("task_schema")
        if semantic_telemetry and runtime_schema is not None and not isinstance(
            runtime_schema,
            Mapping,
        ):
            raise SmolVLARolloutError(
                "candidate task_schema must be an object"
            )
        artifact_summary = manifest.get("task_artifact_summary")
        artifact_summary = (
            artifact_summary
            if isinstance(artifact_summary, Mapping)
            else {}
        )
        outcome_metric = str(
            artifact_summary.get("success_outcome_label")
            or "official_check_success"
        )
        return run_smolvla_robotwin_episode(
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
                if semantic_telemetry
                and isinstance(runtime_schema, Mapping)
                else None
            ),
            visual_capture_profile_id=str(
                candidate.task_contract.get("visual_capture_profile_id")
                or EVENT_KEYFRAMES_PROFILE
            ),
            outcome_metric=outcome_metric,
        )


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-module")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18771)
    args = parser.parse_args()
    value = run_smolvla_robotwin_episode(
        task_name=args.task,
        task_module=args.task_module or f"envs.{args.task}",
        seed=args.seed,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
    )
    episode = value["episode"]
    print(
        "ROLLOUT_RESULT_JSON="
        + json.dumps(
            {
                "task": episode["task"],
                "seed": episode["seed"],
                "policy": episode["policy"],
                "policy_instruction": episode["policy_instruction"],
                "success": episode["success"],
                "official_check_success": episode[
                    "official_check_success"
                ],
                "actions_executed": episode["actions_executed"],
                "chunk_count": episode["chunk_count"],
                "wall_seconds": episode["wall_seconds"],
                "result_artifact": value["artifacts"]["result"],
                "video_artifact": value["artifacts"]["video"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "SmolVLARobotwinRolloutRunner",
    "SmolVLARolloutError",
    "run_smolvla_robotwin_episode",
]
