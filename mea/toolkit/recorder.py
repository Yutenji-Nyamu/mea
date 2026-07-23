"""Schema-driven, multi-rate RoboTwin episode recorder.

The additive ``balanced_v1`` profile preserves the original policy-boundary
CSV, 250 Hz semantic NPZ and contact JSONL.  It additionally records selected
robot/actor dynamics at 50 Hz in a typed, compressed NPZ file.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mea.execution_receipt import (
    validate_execution_invocation,
    validate_imported_task_binding,
)

from .profiles import load_telemetry_profile, telemetry_profile_sha256
from .schema import load_task_schema


class RecorderError(RuntimeError):
    """Raised when a task cannot satisfy its declared telemetry schema."""


_VISUAL_CAPTURE_PROFILES = {"event_keyframes_v1"}
_VISUAL_CAPTURE_FPS = 2


def _numbers(value: Any) -> list[float]:
    if value is None:
        return []
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [float(item) for item in array]


def _pose_parts(value: Any) -> tuple[list[float], list[float]]:
    return _numbers(value.p), _numbers(value.q)


def _body_name(body: Any) -> str:
    entity = getattr(body, "entity", None)
    if entity is None:
        return str(getattr(body, "name", ""))
    getter = getattr(entity, "get_name", None)
    return str(getter() if callable(getter) else getattr(entity, "name", ""))


def _dynamic_velocity(actor_wrapper: Any) -> tuple[list[float | None], list[float | None]]:
    entity = getattr(actor_wrapper, "actor", actor_wrapper)
    direct_linear_getter = getattr(entity, "get_linear_velocity", None)
    direct_angular_getter = getattr(entity, "get_angular_velocity", None)
    direct_linear = (
        direct_linear_getter()
        if callable(direct_linear_getter)
        else getattr(entity, "linear_velocity", None)
    )
    direct_angular = (
        direct_angular_getter()
        if callable(direct_angular_getter)
        else getattr(entity, "angular_velocity", None)
    )
    if direct_linear is not None and direct_angular is not None:
        return _numbers(direct_linear), _numbers(direct_angular)
    components = []
    getter = getattr(entity, "get_components", None)
    if callable(getter):
        components = list(getter())
    else:
        components = list(getattr(entity, "components", []))
    for component in components:
        linear_getter = getattr(component, "get_linear_velocity", None)
        angular_getter = getattr(component, "get_angular_velocity", None)
        linear = (
            linear_getter()
            if callable(linear_getter)
            else getattr(component, "linear_velocity", None)
        )
        angular = (
            angular_getter()
            if callable(angular_getter)
            else getattr(component, "angular_velocity", None)
        )
        if linear is not None and angular is not None:
            return _numbers(linear), _numbers(angular)
    return [None, None, None], [None, None, None]


class EpisodeRecorder:
    """Collect one episode without changing policy or task semantics."""

    def __init__(
        self,
        repo_root: str | Path,
        output_dir: str | Path,
        *,
        task_name: str,
        seed: int,
        episode_index: int,
        policy_name: str,
        task_module: str | None = None,
        task_config: str | None = None,
        checkpoint_setting: str | None = None,
        telemetry_profile_id: str = "balanced_v1",
        visual_capture_profile_id: str | None = None,
        execution_receipt: Mapping[str, Any] | None = None,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schema = load_task_schema(self.repo_root, task_name)
        self.task_name = task_name
        self.seed = int(seed)
        self.episode_index = int(episode_index)
        self.policy_name = policy_name
        self.task_module = task_module
        self.task_config = task_config
        self.checkpoint_setting = checkpoint_setting
        self.execution_receipt = (
            validate_execution_invocation(
                execution_receipt,
                task_name=task_name,
                task_module=task_module,
                task_config=task_config,
                checkpoint_setting=checkpoint_setting,
                policy_name=policy_name,
                seed=self.seed,
                episode_index=self.episode_index,
                checkpoint_dir=(
                    execution_receipt.get("checkpoint", {}).get("root")
                    if execution_receipt.get("checkpoint", {}).get("kind")
                    == "act_checkpoint_bundle"
                    else None
                ),
                verify_checkpoint_files=True,
            )
            if execution_receipt is not None
            else None
        )
        self.executed_binding: dict[str, Any] | None = None
        self.telemetry_profile = load_telemetry_profile(telemetry_profile_id)
        self.telemetry_profile_id = telemetry_profile_id
        self.telemetry_profile_hash = telemetry_profile_sha256(
            self.telemetry_profile
        )
        if (
            visual_capture_profile_id is not None
            and visual_capture_profile_id not in _VISUAL_CAPTURE_PROFILES
        ):
            raise ValueError(
                "unknown visual capture profile: "
                f"{visual_capture_profile_id!r}"
            )
        self.visual_capture_profile_id = visual_capture_profile_id
        dynamics_stream = self.telemetry_profile.get("streams", {}).get(
            "dynamics_trace"
        )
        self.dynamics_period = (
            int(dynamics_stream["every_physics_steps"])
            if dynamics_stream is not None
            else None
        )
        if self.dynamics_period is not None and self.dynamics_period <= 0:
            raise RecorderError("dynamics sampling period must be positive")
        self.physics_dt = float(self.schema.get("physics_timestep_seconds", 0.004))
        self.action_dimension = int(self.schema.get("action_dimension", 0))
        self.policy_rows: list[dict[str, Any]] = []
        self.semantic_rows: list[dict[str, Any]] = []
        self.dynamics_rows: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.physics_step = 0
        self.policy_step = -1
        self.pending_action: list[float] | None = None
        self.pending_action_type: str | None = None
        self.active_contacts: dict[tuple[str, str], dict[str, Any]] = {}
        self.success_seen = False
        self.started_at = time.time()
        self.finished = False
        self._task: Any = None
        self.visual_frames: list[dict[str, Any]] = []
        self.visual_capture_errors: list[dict[str, str]] = []
        self.first_physical_contact_seen = False
        self.initial_physical_contacts: set[tuple[str, str]] = set()
        self.visual_keyframe_dir = self.output_dir / "visual_keyframes"
        if self.visual_capture_profile_id is not None:
            self._prepare_visual_capture()

        (self.output_dir / "schema.json").write_text(
            json.dumps(self.schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "telemetry_profile.json").write_text(
            json.dumps(self.telemetry_profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if self.execution_receipt is not None:
            (self.output_dir / "execution_receipt.json").write_text(
                json.dumps(
                    self.execution_receipt,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def start(self, task: Any) -> None:
        self._task = task
        self._validate_task(task)
        if self.execution_receipt is not None:
            self.executed_binding = validate_imported_task_binding(
                self.execution_receipt,
                task,
            )
        if self.visual_capture_profile_id is not None:
            # Resting/support contacts exist before expert motion and must not
            # consume the first action-induced contact keyframe.
            self.initial_physical_contacts = {
                pair
                for pair, sample in self._contact_samples(task).items()
                if sample["physical_contact"]
            }
        self._capture_visual_keyframe(task, reason="initial")
        self.policy_rows.append(self._full_state(task, phase="initial", action=None))
        self.semantic_rows.append(self._semantic_state(task))
        self._record_dynamics(task, force=True)

    def _visual_capture_error(self, stage: str, error: Exception) -> None:
        self.visual_capture_errors.append(
            {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            }
        )

    def _prepare_visual_capture(self) -> None:
        """Remove stale visual artifacts before a retried expert probe."""

        try:
            self.visual_keyframe_dir.mkdir(parents=True, exist_ok=True)
            for path in self.visual_keyframe_dir.glob("frame_*.png"):
                path.unlink()
            for path in (
                self.output_dir / "visual_keyframes.json",
                self.output_dir / "video.mp4",
                self.output_dir / "video.partial.mp4",
            ):
                if path.exists():
                    path.unlink()
        except Exception as exc:
            self._visual_capture_error("prepare", exc)

    def _capture_visual_keyframe(
        self,
        task: Any,
        *,
        reason: str,
    ) -> int | None:
        """Capture one sparse head-camera frame without affecting telemetry."""

        if self.visual_capture_profile_id is None:
            return None
        if (
            self.visual_frames
            and self.visual_frames[-1]["physics_step"] == self.physics_step
        ):
            reasons = self.visual_frames[-1]["reasons"]
            if reason not in reasons:
                reasons.append(reason)
            return int(self.visual_frames[-1]["frame_index"])
        frame_index = len(self.visual_frames)
        relative_image = Path("visual_keyframes") / f"frame_{frame_index:03d}.png"
        destination = self.output_dir / relative_image
        try:
            save_camera_rgb = getattr(task, "save_camera_rgb")
            save_camera_rgb(str(destination), camera_name="head_camera")
            if not destination.is_file() or destination.stat().st_size <= 0:
                raise RecorderError(
                    f"head-camera keyframe was not written: {destination}"
                )
        except Exception as exc:
            self._visual_capture_error(f"capture:{reason}", exc)
            return None
        self.visual_frames.append(
            {
                "frame_index": frame_index,
                "physics_step": self.physics_step,
                "simulation_time_seconds": self.physics_step * self.physics_dt,
                "reasons": [reason],
                "image": relative_image.as_posix(),
            }
        )
        return frame_index

    def _finalize_visual_capture(self) -> dict[str, Any] | None:
        if self.visual_capture_profile_id is None:
            return None

        video = self.output_dir / "video.mp4"
        partial_video = self.output_dir / "video.partial.mp4"
        if not self.visual_frames:
            self.visual_capture_errors.append(
                {
                    "stage": "encode",
                    "type": "RecorderError",
                    "message": "no visual keyframes were captured",
                }
            )
        if not self.visual_capture_errors:
            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(_VISUAL_CAPTURE_FPS),
                "-start_number",
                "0",
                "-i",
                str(self.visual_keyframe_dir / "frame_%03d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                str(partial_video),
            ]
            try:
                process = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=60,
                )
                if process.returncode != 0:
                    raise RecorderError(
                        "ffmpeg failed with return code "
                        f"{process.returncode}: {process.stderr.strip()}"
                    )
                if not partial_video.is_file() or partial_video.stat().st_size <= 0:
                    raise RecorderError("ffmpeg did not produce a non-empty video")
                partial_video.replace(video)
            except Exception as exc:
                self._visual_capture_error("encode", exc)
        if partial_video.exists():
            try:
                partial_video.unlink()
            except OSError:
                pass

        completed = not self.visual_capture_errors and video.is_file()
        result = {
            "schema_version": 1,
            "profile_id": self.visual_capture_profile_id,
            "status": "completed" if completed else "failed",
            "camera": "head_camera",
            "frame_count": len(self.visual_frames),
            "nominal_frame_rate_hz": _VISUAL_CAPTURE_FPS,
            "frames": self.visual_frames,
            "errors": self.visual_capture_errors,
        }
        self._write_visual_manifest(result)
        return result

    def _write_visual_manifest(self, result: dict[str, Any]) -> None:
        (self.output_dir / "visual_keyframes.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _validate_task(self, task: Any) -> None:
        missing = [
            item["task_attribute"]
            for item in self.schema["tracked_actors"]
            if not hasattr(task, item["task_attribute"])
        ]
        if missing:
            raise RecorderError(f"TaskSchema actor attributes 缺失: {missing}")
        if not hasattr(task, "robot") or not hasattr(task, "scene"):
            raise RecorderError("task 缺少 robot 或 scene")

    def _actor(self, task: Any, actor_spec: dict[str, Any]) -> Any:
        return getattr(task, actor_spec["task_attribute"])

    @staticmethod
    def _put_vector(row: dict[str, Any], prefix: str, values: Any) -> None:
        for index, value in enumerate(_numbers(values)):
            row[f"{prefix}.{index}"] = value

    def _full_state(
        self,
        task: Any,
        *,
        phase: str,
        action: list[float] | None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "phase": phase,
            "policy_step": self.policy_step,
            "physics_step": self.physics_step,
            "simulation_time_seconds": self.physics_step * self.physics_dt,
            "wall_time_seconds": time.time() - self.started_at,
            "video_frame_index": max(self.policy_step, 0),
            "success": int(bool(getattr(task, "eval_success", False))),
            "action_type": self.pending_action_type or "",
        }
        action_values = action or []
        for index in range(self.action_dimension):
            row[f"action.{index}"] = (
                action_values[index] if index < len(action_values) else None
            )

        robot = task.robot
        for side in ("left", "right"):
            entity = getattr(robot, f"{side}_entity")
            self._put_vector(row, f"robot.{side}.qpos", entity.get_qpos())
            self._put_vector(row, f"robot.{side}.qvel", entity.get_qvel())
            self._put_vector(
                row, f"robot.{side}.ee", getattr(robot, f"get_{side}_ee_pose")()
            )
            self._put_vector(
                row, f"robot.{side}.tcp", getattr(robot, f"get_{side}_tcp_pose")()
            )
            row[f"robot.{side}.gripper"] = float(
                getattr(robot, f"get_{side}_gripper_val")()
            )

        for actor_spec in self.schema["tracked_actors"]:
            actor_id = actor_spec["id"]
            actor = self._actor(task, actor_spec)
            position, quaternion = _pose_parts(actor.get_pose())
            self._put_vector(row, f"actor.{actor_id}.position", position)
            self._put_vector(row, f"actor.{actor_id}.quaternion", quaternion)
            linear, angular = _dynamic_velocity(actor)
            for index, value in enumerate(linear):
                row[f"actor.{actor_id}.linear_velocity.{index}"] = value
            for index, value in enumerate(angular):
                row[f"actor.{actor_id}.angular_velocity.{index}"] = value
            for point_id in actor_spec.get("functional_points", []):
                point = actor.get_functional_point(point_id, "pose")
                p, q = _pose_parts(point)
                self._put_vector(
                    row, f"actor.{actor_id}.functional.{point_id}.position", p
                )
                self._put_vector(
                    row, f"actor.{actor_id}.functional.{point_id}.quaternion", q
                )
        return row

    def _semantic_state(self, task: Any) -> dict[str, Any]:
        values: dict[str, Any] = {
            "physics_step": self.physics_step,
            "policy_step": self.policy_step,
            "simulation_time_seconds": self.physics_step * self.physics_dt,
            "success": bool(getattr(task, "eval_success", False)),
        }
        if self.visual_capture_profile_id is not None:
            values["video_frame_index"] = (
                int(self.visual_frames[-1]["frame_index"])
                if self.visual_frames
                else 0
            )
        actor_specs = {
            item["id"]: item for item in self.schema["tracked_actors"]
        }
        fields = self.schema.get("semantic_fields")
        if not fields:
            # Backward-compatible fallback for schema snapshots written before
            # semantic_fields became explicit.  New schemas should always list
            # their fields so the recorder remains task-independent.
            fields = self._fallback_semantic_fields()
        for field in fields:
            name = str(field["name"])
            source = field["source"]
            if source == "actor_position":
                actor = self._actor(task, actor_specs[field["actor_id"]])
                value = actor.get_pose().p
            elif source == "actor_functional_position":
                actor = self._actor(task, actor_specs[field["actor_id"]])
                point = actor.get_functional_point(
                    int(field["point_id"]), "pose"
                )
                value = point.p
            elif source == "actor_contact_position":
                actor = self._actor(task, actor_specs[field["actor_id"]])
                point = actor.get_contact_point(int(field["point_id"]), "pose")
                value = point.p
            elif source == "robot_tcp_position":
                side = str(field["side"])
                if side not in {"left", "right"}:
                    raise RecorderError(f"invalid robot side in semantic field: {side}")
                value = getattr(task.robot, f"get_{side}_tcp_pose")()[:3]
            else:
                raise RecorderError(
                    f"unsupported semantic field source {source!r} for {name!r}"
                )
            values[name] = _numbers(value)
        return values

    def _fallback_semantic_fields(self) -> list[dict[str, Any]]:
        if self.task_name == "beat_block_hammer":
            contract = self.schema.get("success_contract", {})
            return [
                {
                    "name": "hammer_position",
                    "source": "actor_position",
                    "actor_id": "hammer",
                },
                {
                    "name": "block_position",
                    "source": "actor_position",
                    "actor_id": "block",
                },
                {
                    "name": "hammer_functional_position",
                    "source": "actor_functional_position",
                    "actor_id": "hammer",
                    "point_id": contract.get("hammer_functional_point", 0),
                },
                {
                    "name": "block_functional_position",
                    "source": "actor_functional_position",
                    "actor_id": "block",
                    "point_id": contract.get("block_functional_point", 1),
                },
                {
                    "name": "left_tcp_position",
                    "source": "robot_tcp_position",
                    "side": "left",
                },
                {
                    "name": "right_tcp_position",
                    "source": "robot_tcp_position",
                    "side": "right",
                },
            ]
        fields: list[dict[str, Any]] = []
        for actor_spec in self.schema["tracked_actors"]:
            actor_id = actor_spec["id"]
            fields.append(
                {
                    "name": f"actor.{actor_id}.position",
                    "source": "actor_position",
                    "actor_id": actor_id,
                }
            )
            for point_id in actor_spec.get("functional_points", []):
                fields.append(
                    {
                        "name": f"actor.{actor_id}.functional.{point_id}.position",
                        "source": "actor_functional_position",
                        "actor_id": actor_id,
                        "point_id": point_id,
                    }
                )
            for point_id in actor_spec.get("contact_points", []):
                fields.append(
                    {
                        "name": f"actor.{actor_id}.contact.{point_id}.position",
                        "source": "actor_contact_position",
                        "actor_id": actor_id,
                        "point_id": point_id,
                    }
                )
        for side in ("left", "right"):
            fields.append(
                {
                    "name": f"{side}_tcp_position",
                    "source": "robot_tcp_position",
                    "side": side,
                }
            )
        return fields

    def _dynamics_state(
        self,
        task: Any,
        *,
        success_override: bool | None = None,
    ) -> dict[str, Any]:
        """Capture one fixed-schema selected-actor dynamics sample."""

        values: dict[str, Any] = {
            "physics_step": int(self.physics_step),
            "policy_step": int(self.policy_step),
            "simulation_time_seconds": float(self.physics_step * self.physics_dt),
            "success": (
                bool(success_override)
                if success_override is not None
                else bool(getattr(task, "eval_success", False))
            ),
        }
        robot = task.robot
        for side in ("left", "right"):
            entity = getattr(robot, f"{side}_entity")
            values[f"robot.{side}.qpos"] = _numbers(entity.get_qpos())
            values[f"robot.{side}.qvel"] = _numbers(entity.get_qvel())
            values[f"robot.{side}.ee_pose"] = _numbers(
                getattr(robot, f"get_{side}_ee_pose")()
            )
            values[f"robot.{side}.tcp_pose"] = _numbers(
                getattr(robot, f"get_{side}_tcp_pose")()
            )
            values[f"robot.{side}.gripper"] = float(
                getattr(robot, f"get_{side}_gripper_val")()
            )

        for actor_spec in self.schema["tracked_actors"]:
            actor_id = actor_spec["id"]
            actor = self._actor(task, actor_spec)
            position, quaternion = _pose_parts(actor.get_pose())
            values[f"actor.{actor_id}.position"] = position
            values[f"actor.{actor_id}.quaternion"] = quaternion
            linear, angular = _dynamic_velocity(actor)
            values[f"actor.{actor_id}.linear_velocity"] = linear
            values[f"actor.{actor_id}.angular_velocity"] = angular
            for point_id in actor_spec.get("functional_points", []):
                point = actor.get_functional_point(point_id, "pose")
                point_position, point_quaternion = _pose_parts(point)
                prefix = f"actor.{actor_id}.functional.{point_id}"
                values[f"{prefix}.position"] = point_position
                values[f"{prefix}.quaternion"] = point_quaternion
            for point_id in actor_spec.get("contact_points", []):
                point = actor.get_contact_point(point_id, "pose")
                point_position, point_quaternion = _pose_parts(point)
                prefix = f"actor.{actor_id}.contact.{point_id}"
                values[f"{prefix}.position"] = point_position
                values[f"{prefix}.quaternion"] = point_quaternion
        return values

    def _record_dynamics(
        self,
        task: Any,
        *,
        force: bool = False,
        success_override: bool | None = None,
    ) -> None:
        if self.dynamics_period is None:
            return
        if not force and self.physics_step % self.dynamics_period != 0:
            return
        row = self._dynamics_state(
            task,
            success_override=success_override,
        )
        if (
            self.dynamics_rows
            and self.dynamics_rows[-1]["physics_step"] == self.physics_step
        ):
            # finish() may force a final sample on a regular sample boundary;
            # replace it so the final snapshot reflects the latest task state.
            self.dynamics_rows[-1] = row
        else:
            self.dynamics_rows.append(row)

    def on_policy_action_start(
        self,
        task: Any,
        *,
        action: Any,
        action_type: str,
    ) -> None:
        self.policy_step += 1
        self.pending_action = _numbers(action)
        self.pending_action_type = str(action_type)

    def on_policy_action_end(self, task: Any, *, success: bool) -> None:
        self.policy_rows.append(
            self._full_state(
                task,
                phase="post_action",
                action=self.pending_action,
            )
        )
        if success:
            self._record_success(task)
        self.pending_action = None

    def _contact_samples(self, task: Any) -> dict[tuple[str, str], dict[str, Any]]:
        scene_names = {
            item["scene_name"]
            for item in self.schema["tracked_actors"]
            if item["id"] in self.schema.get("contact_focus_actor_ids", [])
        }
        samples: dict[tuple[str, str], dict[str, Any]] = {}
        for contact in task.scene.get_contacts():
            bodies = list(getattr(contact, "bodies", []))
            if len(bodies) != 2:
                continue
            names = (_body_name(bodies[0]), _body_name(bodies[1]))
            if not names[0] or not names[1] or not scene_names.intersection(names):
                continue
            pair = tuple(sorted(names))
            point_count = 0
            max_impulse = 0.0
            min_separation: float | None = None
            peak_position: list[float] | None = None
            peak_normal: list[float] | None = None
            physical_contact = False
            for point in getattr(contact, "points", []):
                point_count += 1
                impulse = _numbers(getattr(point, "impulse", [0.0, 0.0, 0.0]))
                impulse_norm = math.sqrt(sum(value * value for value in impulse))
                separation = float(getattr(point, "separation", math.inf))
                if math.isfinite(separation):
                    min_separation = (
                        separation
                        if min_separation is None
                        else min(min_separation, separation)
                    )
                if impulse_norm > max_impulse:
                    max_impulse = impulse_norm
                    peak_position = _numbers(
                        getattr(point, "position", [0.0, 0.0, 0.0])
                    )
                    peak_normal = _numbers(
                        getattr(point, "normal", [0.0, 0.0, 0.0])
                    )
                physical_contact = physical_contact or (
                    impulse_norm > 1e-8 or separation <= 0.0
                )
            value = samples.setdefault(
                pair,
                {
                    "point_count": 0,
                    "max_impulse": 0.0,
                    "min_separation": None,
                    "physical_contact": False,
                    "peak_position": None,
                    "peak_normal": None,
                },
            )
            value["point_count"] += point_count
            if max_impulse > value["max_impulse"]:
                value["max_impulse"] = max_impulse
                value["peak_position"] = peak_position
                value["peak_normal"] = peak_normal
            if min_separation is not None:
                previous = value["min_separation"]
                value["min_separation"] = (
                    min_separation
                    if previous is None
                    else min(previous, min_separation)
                )
            value["physical_contact"] = (
                value["physical_contact"] or physical_contact
            )
        return samples

    def _update_contact_events(self, task: Any) -> None:
        samples = self._contact_samples(task)
        current = set(samples)
        previous = set(self.active_contacts)
        visual_capture_enabled = (
            getattr(self, "visual_capture_profile_id", None) is not None
        )
        for pair in sorted(current - previous):
            sample = samples[pair]
            contact_frame_index = None
            if (
                visual_capture_enabled
                and sample["physical_contact"]
                and pair not in self.initial_physical_contacts
                and not getattr(self, "first_physical_contact_seen", False)
            ):
                self.first_physical_contact_seen = True
                contact_frame_index = self._capture_visual_keyframe(
                    task,
                    reason="first_physical_contact",
                )
            self.active_contacts[pair] = {
                "type": "contact_interval",
                "actors": list(pair),
                "start_policy_step": self.policy_step,
                "start_physics_step": self.physics_step,
                "start_simulation_time_seconds": self.physics_step * self.physics_dt,
                "max_impulse": float(sample["max_impulse"]),
                "max_point_count": int(sample["point_count"]),
                "min_separation": sample["min_separation"],
                "physical_contact": bool(sample["physical_contact"]),
                "first_physical_policy_step": (
                    self.policy_step if sample["physical_contact"] else None
                ),
                "first_physical_physics_step": (
                    self.physics_step if sample["physical_contact"] else None
                ),
                "first_physical_simulation_time_seconds": (
                    self.physics_step * self.physics_dt
                    if sample["physical_contact"]
                    else None
                ),
                **(
                    {
                        "first_physical_video_frame_index": (
                            contact_frame_index
                        )
                    }
                    if visual_capture_enabled
                    else {}
                ),
                "peak_policy_step": self.policy_step,
                "peak_physics_step": self.physics_step,
                "peak_position": sample["peak_position"],
                "peak_normal": sample["peak_normal"],
            }
        for pair in sorted(current & previous):
            sample = samples[pair]
            interval = self.active_contacts[pair]
            interval["max_point_count"] = max(
                interval["max_point_count"], int(sample["point_count"])
            )
            if sample["min_separation"] is not None:
                previous_min_separation = interval["min_separation"]
                interval["min_separation"] = (
                    sample["min_separation"]
                    if previous_min_separation is None
                    else min(
                        previous_min_separation,
                        sample["min_separation"],
                    )
                )
            if sample["physical_contact"] and not interval["physical_contact"]:
                interval["physical_contact"] = True
                interval["first_physical_policy_step"] = self.policy_step
                interval["first_physical_physics_step"] = self.physics_step
                interval["first_physical_simulation_time_seconds"] = (
                    self.physics_step * self.physics_dt
                )
                contact_frame_index = None
                if (
                    visual_capture_enabled
                    and pair not in self.initial_physical_contacts
                    and not getattr(self, "first_physical_contact_seen", False)
                ):
                    self.first_physical_contact_seen = True
                    contact_frame_index = self._capture_visual_keyframe(
                        task,
                        reason="first_physical_contact",
                    )
                if visual_capture_enabled:
                    interval["first_physical_video_frame_index"] = (
                        contact_frame_index
                    )
            if float(sample["max_impulse"]) > interval["max_impulse"]:
                interval["max_impulse"] = float(sample["max_impulse"])
                interval["peak_policy_step"] = self.policy_step
                interval["peak_physics_step"] = self.physics_step
                interval["peak_position"] = sample["peak_position"]
                interval["peak_normal"] = sample["peak_normal"]
        for pair in sorted(previous - current):
            self._close_contact(pair, reason="separated")

    def _close_contact(self, pair: tuple[str, str], *, reason: str) -> None:
        interval = self.active_contacts.pop(pair)
        interval.update(
            {
                "end_policy_step": self.policy_step,
                "end_physics_step": self.physics_step,
                "end_simulation_time_seconds": self.physics_step * self.physics_dt,
                "end_reason": reason,
            }
        )
        self.events.append(interval)

    def _record_success(self, task: Any) -> None:
        if self.success_seen:
            return
        self.success_seen = True
        video_frame_index = (
            self._capture_visual_keyframe(task, reason="success_transition")
            if getattr(self, "visual_capture_profile_id", None) is not None
            else max(self.policy_step, 0)
        )
        self.events.append(
            {
                "type": "success_transition",
                "policy_step": self.policy_step,
                "physics_step": self.physics_step,
                "simulation_time_seconds": self.physics_step * self.physics_dt,
                "video_frame_index": video_frame_index,
            }
        )

    def on_physics_step(self, task: Any) -> None:
        self.physics_step += 1
        try:
            success = bool(task.check_success())
        except Exception:
            success = bool(getattr(task, "eval_success", False))
        if success:
            self._record_success(task)
        self._update_contact_events(task)
        state = self._semantic_state(task)
        state["success"] = success or bool(getattr(task, "eval_success", False))
        self.semantic_rows.append(state)
        self._record_dynamics(task, success_override=state["success"])

    def record_error(self, error: BaseException) -> None:
        self.events.append(
            {
                "type": "error",
                "policy_step": self.policy_step,
                "physics_step": self.physics_step,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    def finish(
        self,
        task: Any,
        *,
        success: bool,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.finished:
            raise RecorderError("EpisodeRecorder.finish() 不能重复调用")
        self.finished = True
        for pair in list(self.active_contacts):
            self._close_contact(pair, reason="episode_end")
        if success:
            self._record_success(task)
        self._capture_visual_keyframe(task, reason="final")
        self.policy_rows.append(
            self._full_state(task, phase="final", action=None)
        )
        self._record_dynamics(task, force=True, success_override=success)
        self._write_policy_csv()
        semantic_arrays = self._write_semantic_npz()
        dynamics_arrays = self._write_dynamics_npz()
        self._write_events()
        try:
            visual_capture = self._finalize_visual_capture()
        except Exception as exc:  # visual evidence is always best-effort
            self._visual_capture_error("finalize", exc)
            # A container produced before the manifest failed is not a
            # contract-complete artifact. Remove it so path-only consumers
            # cannot mistake best-effort output for approved visual evidence.
            for path in (
                self.output_dir / "video.mp4",
                self.output_dir / "video.partial.mp4",
                self.output_dir / "visual_keyframes.json",
            ):
                try:
                    path.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    self._visual_capture_error("cleanup", cleanup_exc)
            visual_capture = {
                "schema_version": 1,
                "profile_id": self.visual_capture_profile_id,
                "status": "failed",
                "camera": "head_camera",
                "frame_count": len(self.visual_frames),
                "nominal_frame_rate_hz": _VISUAL_CAPTURE_FPS,
                "frames": self.visual_frames,
                "errors": self.visual_capture_errors,
            }
        stream_metadata: dict[str, Any] = {
            "policy_state": {
                "artifact": "states.csv",
                "sampling": "policy_boundary",
                "rows": len(self.policy_rows),
            },
            "semantic_trace": {
                "artifact": "semantic_trace.npz",
                "sampling": "physics_period",
                "every_physics_steps": 1,
                "rows": len(self.semantic_rows),
                "arrays": semantic_arrays,
            },
            "contact_events": {
                "artifact": "events.jsonl",
                "sampling": "physics_period",
                "every_physics_steps": 1,
                "mode": "interval_summary",
                "rows": len(self.events),
            },
        }
        if self.dynamics_period is not None:
            stream_metadata["dynamics_trace"] = {
                "artifact": "dynamics_trace.npz",
                "sampling": "physics_period",
                "every_physics_steps": self.dynamics_period,
                "force_initial_sample": True,
                "force_final_sample": True,
                "rows": len(self.dynamics_rows),
                "arrays": dynamics_arrays,
            }
        visual_completed = bool(
            visual_capture and visual_capture.get("status") == "completed"
        )
        metadata = {
            "schema_version": 1,
            "recorder_schema_version": 2,
            "task_name": self.task_name,
            "task_module": self.task_module,
            "task_config": self.task_config,
            "checkpoint_setting": self.checkpoint_setting,
            "policy_name": self.policy_name,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "success": bool(success),
            "policy_steps": max(self.policy_step + 1, 0),
            "physics_steps": self.physics_step,
            "physics_timestep_seconds": self.physics_dt,
            "simulation_duration_seconds": self.physics_step * self.physics_dt,
            "wall_duration_seconds": time.time() - self.started_at,
            "policy_state_rows": len(self.policy_rows),
            "semantic_trace_rows": len(self.semantic_rows),
            "dynamics_trace_rows": len(self.dynamics_rows),
            "telemetry_profile_id": self.telemetry_profile_id,
            "telemetry_profile_sha256": self.telemetry_profile_hash,
            "telemetry": {
                "profile_id": self.telemetry_profile_id,
                "profile_sha256": self.telemetry_profile_hash,
                "profile_artifact": "telemetry_profile.json",
                "streams": stream_metadata,
            },
            "contact_interval_count": sum(
                item.get("type") == "contact_interval" for item in self.events
            ),
            "error": error,
            **(
                {
                    "execution_receipt": deepcopy(self.execution_receipt),
                    "execution_receipt_sha256": self.execution_receipt[
                        "receipt_sha256"
                    ],
                    "executed_binding": deepcopy(self.executed_binding),
                    "executed_task_module_sha256": self.executed_binding[
                        "task_source_sha256"
                    ],
                    "executed_checkpoint_bundle_sha256": (
                        self.executed_binding["checkpoint_bundle_sha256"]
                    ),
                }
                if self.execution_receipt is not None
                and self.executed_binding is not None
                else {}
            ),
            **(
                {"visual_capture": visual_capture}
                if visual_capture is not None
                else {}
            ),
            **(
                {
                    "video_alignment": {
                        "schema_version": 1,
                        "mode": "event_keyframes",
                        "nominal_frame_rate_hz": _VISUAL_CAPTURE_FPS,
                        "frame_manifest": "visual_keyframes.json",
                        "frame_semantics": (
                            "ordered sparse event evidence; not continuous-time "
                            "video"
                        ),
                    }
                }
                if visual_completed
                else {}
            ),
            "artifacts": {
                "policy_states": "states.csv",
                "semantic_trace": "semantic_trace.npz",
                **(
                    {"dynamics_trace": "dynamics_trace.npz"}
                    if self.dynamics_period is not None
                    else {}
                ),
                "events": "events.jsonl",
                "task_schema": "schema.json",
                "telemetry_profile": "telemetry_profile.json",
                **(
                    {"execution_receipt": "execution_receipt.json"}
                    if self.execution_receipt is not None
                    else {}
                ),
                **(
                    {"visual_keyframes": "visual_keyframes.json"}
                    if (self.output_dir / "visual_keyframes.json").is_file()
                    else {}
                ),
                **({"video": "video.mp4"} if visual_completed else {}),
            },
        }
        (self.output_dir / "episode.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return metadata

    def _write_policy_csv(self) -> None:
        columns = [
            "phase",
            "policy_step",
            "physics_step",
            "simulation_time_seconds",
            "wall_time_seconds",
            "video_frame_index",
            "success",
            "action_type",
        ]
        remaining = sorted(
            set().union(*(row.keys() for row in self.policy_rows)) - set(columns)
        )
        with (self.output_dir / "states.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns + remaining)
            writer.writeheader()
            writer.writerows(self.policy_rows)

    @staticmethod
    def _array_manifest(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        }

    @staticmethod
    def _typed_arrays(
        rows: list[dict[str, Any]],
        *,
        step_dtype: Any = np.int64,
    ) -> dict[str, np.ndarray]:
        if not rows:
            return {}
        expected = set(rows[0])
        for index, row in enumerate(rows[1:], start=1):
            if set(row) != expected:
                missing = sorted(expected - set(row))
                extra = sorted(set(row) - expected)
                raise RecorderError(
                    f"telemetry row {index} schema drift: missing={missing}, extra={extra}"
                )
        arrays: dict[str, np.ndarray] = {}
        for key in sorted(expected):
            if key in {"physics_step", "policy_step"}:
                dtype: Any = step_dtype
            elif key == "success":
                dtype = np.bool_
            elif key == "simulation_time_seconds":
                dtype = np.float64
            else:
                dtype = np.float32
            try:
                arrays[key] = np.asarray([row[key] for row in rows], dtype=dtype)
            except (TypeError, ValueError) as exc:
                raise RecorderError(f"cannot encode telemetry array {key!r}") from exc
        return arrays

    def _write_semantic_npz(self) -> dict[str, Any]:
        # Keep the legacy semantic axis dtype byte-compatible with recorder v1.
        arrays = self._typed_arrays(self.semantic_rows, step_dtype=np.float64)
        np.savez_compressed(self.output_dir / "semantic_trace.npz", **arrays)
        return self._array_manifest(arrays)

    def _write_dynamics_npz(self) -> dict[str, Any]:
        if self.dynamics_period is None:
            return {}
        arrays = self._typed_arrays(self.dynamics_rows)
        np.savez_compressed(self.output_dir / "dynamics_trace.npz", **arrays)
        return self._array_manifest(arrays)

    def _write_events(self) -> None:
        with (self.output_dir / "events.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for event in self.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
