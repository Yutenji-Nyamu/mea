"""Schema-driven RoboTwin episode recorder.

Full robot/action snapshots are stored at policy boundaries in CSV.  A compact
semantic trace is stored at every 250 Hz physics step in NPZ, while variable
contact intervals and success transitions are stored in JSONL.
"""

from __future__ import annotations

import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .schema import load_task_schema


class RecorderError(RuntimeError):
    """Raised when a task cannot satisfy its declared telemetry schema."""


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
        self.physics_dt = float(self.schema.get("physics_timestep_seconds", 0.004))
        self.action_dimension = int(self.schema.get("action_dimension", 0))
        self.policy_rows: list[dict[str, Any]] = []
        self.semantic_rows: list[dict[str, Any]] = []
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

        (self.output_dir / "schema.json").write_text(
            json.dumps(self.schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def start(self, task: Any) -> None:
        self._task = task
        self._validate_task(task)
        self.policy_rows.append(self._full_state(task, phase="initial", action=None))
        self.semantic_rows.append(self._semantic_state(task))

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
        actor_specs = {item["id"]: item for item in self.schema["tracked_actors"]}
        hammer = self._actor(task, actor_specs["hammer"])
        block = self._actor(task, actor_specs["block"])
        values["hammer_position"] = _numbers(hammer.get_pose().p)
        values["block_position"] = _numbers(block.get_pose().p)
        success_contract = self.schema["success_contract"]
        values["hammer_functional_position"] = _numbers(
            hammer.get_functional_point(
                success_contract["hammer_functional_point"], "pose"
            ).p
        )
        values["block_functional_position"] = _numbers(
            block.get_functional_point(
                success_contract["block_functional_point"], "pose"
            ).p
        )
        values["left_tcp_position"] = _numbers(task.robot.get_left_tcp_pose()[:3])
        values["right_tcp_position"] = _numbers(task.robot.get_right_tcp_pose()[:3])
        return values

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
        for pair in sorted(current - previous):
            sample = samples[pair]
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
        self.events.append(
            {
                "type": "success_transition",
                "policy_step": self.policy_step,
                "physics_step": self.physics_step,
                "simulation_time_seconds": self.physics_step * self.physics_dt,
                "video_frame_index": max(self.policy_step, 0),
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
        self.policy_rows.append(
            self._full_state(task, phase="final", action=None)
        )
        self._write_policy_csv()
        self._write_semantic_npz()
        self._write_events()
        metadata = {
            "schema_version": 1,
            "recorder_schema_version": 1,
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
            "contact_interval_count": sum(
                item.get("type") == "contact_interval" for item in self.events
            ),
            "error": error,
            "artifacts": {
                "policy_states": "states.csv",
                "semantic_trace": "semantic_trace.npz",
                "events": "events.jsonl",
                "task_schema": "schema.json",
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

    def _write_semantic_npz(self) -> None:
        scalar_keys = (
            "physics_step",
            "policy_step",
            "simulation_time_seconds",
            "success",
        )
        vector_keys = (
            "hammer_position",
            "block_position",
            "hammer_functional_position",
            "block_functional_position",
            "left_tcp_position",
            "right_tcp_position",
        )
        arrays: dict[str, Any] = {}
        for key in scalar_keys:
            dtype = np.bool_ if key == "success" else np.float64
            arrays[key] = np.asarray(
                [row[key] for row in self.semantic_rows], dtype=dtype
            )
        for key in vector_keys:
            arrays[key] = np.asarray(
                [row[key] for row in self.semantic_rows], dtype=np.float32
            )
        np.savez_compressed(self.output_dir / "semantic_trace.npz", **arrays)

    def _write_events(self) -> None:
        with (self.output_dir / "events.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for event in self.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
