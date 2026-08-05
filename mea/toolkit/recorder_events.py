"""Contact, success, and error event normalization."""

from __future__ import annotations

import math
from typing import Any

from .recorder_telemetry import _body_entity, _body_name, _numbers
from .schema import TaskSchemaError


class RecorderEventMixin:
    def _contact_samples(self, task: Any) -> dict[tuple[str, str], dict[str, Any]]:
        tracked_actors = self.schema["tracked_actors"]
        focus_ids = set(self.schema.get("contact_focus_actor_ids", []))
        scene_name_counts: dict[str, int] = {}
        for item in tracked_actors:
            scene_name = item["scene_name"]
            scene_name_counts[scene_name] = (
                scene_name_counts.get(scene_name, 0) + 1
            )
        tracked_entities: dict[int, tuple[str, str]] = {}
        focused_entity_ids: set[int] = set()
        unique_focus_names: set[str] = set()
        for item in tracked_actors:
            scene_name = item["scene_name"]
            if scene_name_counts[scene_name] == 1:
                # Keep existing contact labels and articulation-name fallback
                # for legacy direct actors.
                label = scene_name
            else:
                label = f"{scene_name}#{item['id']}"
            if item["id"] in focus_ids and scene_name_counts[scene_name] == 1:
                unique_focus_names.add(scene_name)
            try:
                actor = self._actor(task, item)
            except (
                TaskSchemaError,
                AttributeError,
                IndexError,
                KeyError,
                TypeError,
            ):
                continue
            entity = getattr(actor, "actor", actor)
            entity_identity = id(entity)
            tracked_entities[entity_identity] = (label, str(item["id"]))
            if item["id"] in focus_ids:
                focused_entity_ids.add(entity_identity)
        samples: dict[tuple[str, str], dict[str, Any]] = {}
        for contact in task.scene.get_contacts():
            bodies = list(getattr(contact, "bodies", []))
            if len(bodies) != 2:
                continue
            raw_names = (_body_name(bodies[0]), _body_name(bodies[1]))
            identities = tuple(id(_body_entity(body)) for body in bodies)
            names = tuple(
                tracked_entities.get(identity, (raw_name, raw_name))[0]
                for identity, raw_name in zip(identities, raw_names)
            )
            actor_ids = tuple(
                tracked_entities.get(identity, (raw_name, raw_name))[1]
                for identity, raw_name in zip(identities, raw_names)
            )
            identity_focus = any(
                identity in focused_entity_ids for identity in identities
            )
            name_focus = bool(unique_focus_names.intersection(raw_names))
            if (
                not names[0]
                or not names[1]
                or not (identity_focus or name_focus)
            ):
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
                    "actors": list(sorted(raw_names)),
                    "actor_ids": list(sorted(actor_ids)),
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
                "actors": list(sample.get("actors", pair)),
                "actor_ids": list(sample.get("actor_ids", pair)),
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



__all__ = ["RecorderEventMixin"]

