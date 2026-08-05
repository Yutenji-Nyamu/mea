"""Task-independent telemetry normalization and sampling."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .recorder_contracts import RecorderError
from .schema import resolve_task_actor


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


def _body_entity(body: Any) -> Any:
    return getattr(body, "entity", body)


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

class RecorderTelemetryMixin:
    def _validate_task(self, task: Any) -> None:
        missing: list[str] = []
        for item in self.schema["tracked_actors"]:
            try:
                resolve_task_actor(task, item)
            except (AttributeError, IndexError, KeyError, TypeError):
                missing.append(str(item["id"]))
        if missing:
            raise RecorderError(f"TaskSchema actor access paths 缺失: {missing}")
        if not hasattr(task, "robot") or not hasattr(task, "scene"):
            raise RecorderError("task 缺少 robot 或 scene")

    def _actor(self, task: Any, actor_spec: dict[str, Any]) -> Any:
        return resolve_task_actor(task, actor_spec)

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



__all__ = [
    "RecorderTelemetryMixin",
    "_body_entity",
    "_body_name",
    "_numbers",
]

