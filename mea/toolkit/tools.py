"""Trusted, deterministic tools over recorded BeatBlockHammer trajectories."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


class TrajectoryError(RuntimeError):
    """Raised when an episode is incomplete or incompatible."""


class TrajectoryView:
    """Read-only view over one recorder episode."""

    REQUIRED_TRACE_KEYS = {
        "physics_step",
        "policy_step",
        "simulation_time_seconds",
        "success",
        "hammer_position",
        "block_position",
        "hammer_functional_position",
        "block_functional_position",
        "left_tcp_position",
        "right_tcp_position",
    }

    def __init__(self, episode_dir: str | Path):
        self.episode_dir = Path(episode_dir).expanduser().resolve()
        self.metadata = json.loads(
            (self.episode_dir / "episode.json").read_text(encoding="utf-8")
        )
        self.schema = json.loads(
            (self.episode_dir / "schema.json").read_text(encoding="utf-8")
        )
        with np.load(self.episode_dir / "semantic_trace.npz") as archive:
            self.trace = {key: archive[key].copy() for key in archive.files}
        missing = sorted(self.REQUIRED_TRACE_KEYS - set(self.trace))
        if missing:
            raise TrajectoryError(f"semantic trace 缺少字段: {missing}")
        self.events = []
        events_path = self.episode_dir / "events.jsonl"
        if events_path.is_file():
            self.events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        with (self.episode_dir / "states.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            self.policy_states = list(csv.DictReader(handle))
        lengths = {len(value) for value in self.trace.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise TrajectoryError("semantic trace 为空或数组长度不一致")
        physics_steps = self.trace["physics_step"].astype(np.int64)
        policy_steps = self.trace["policy_step"].astype(np.int64)
        if physics_steps[0] != 0 or np.any(np.diff(physics_steps) <= 0):
            raise TrajectoryError("physics_step 必须从 0 开始且严格递增")
        if np.any(np.diff(policy_steps) < 0):
            raise TrajectoryError("policy_step 不得倒退")
        expected_physics_steps = int(self.metadata.get("physics_steps", -1))
        if expected_physics_steps != int(physics_steps[-1]):
            raise TrajectoryError(
                "episode.json physics_steps 与 semantic trace 不一致"
            )
        expected_rows = self.metadata.get("semantic_trace_rows")
        if expected_rows is not None and int(expected_rows) != len(physics_steps):
            raise TrajectoryError(
                "episode.json semantic_trace_rows 与 semantic trace 不一致"
            )
        for key in self.REQUIRED_TRACE_KEYS - {"success"}:
            if not np.all(np.isfinite(self.trace[key])):
                raise TrajectoryError(f"semantic trace 包含非有限值: {key}")

    @property
    def contact_intervals(self) -> list[dict[str, Any]]:
        return [item for item in self.events if item.get("type") == "contact_interval"]

    @property
    def success_events(self) -> list[dict[str, Any]]:
        return [item for item in self.events if item.get("type") == "success_transition"]

    def hammer_block_contacts(self) -> list[dict[str, Any]]:
        expected = {"020_hammer", "box"}
        return [
            item
            for item in self.contact_intervals
            if set(item.get("actors", [])) == expected
        ]


def _tool_hash(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _evidence(
    trajectory: TrajectoryView,
    index: int,
) -> dict[str, Any]:
    policy_step = int(trajectory.trace["policy_step"][index])
    physics_step = int(trajectory.trace["physics_step"][index])
    return {
        "trace_index": int(index),
        "policy_step": policy_step,
        "physics_step": physics_step,
        "simulation_time_seconds": float(
            trajectory.trace["simulation_time_seconds"][index]
        ),
        "video_frame_before": max(policy_step, 0),
        "video_frame_after": max(policy_step + 1, 0),
    }


def _result(
    name: str,
    function: Callable[..., Any],
    *,
    value: Any,
    unit: str | None,
    evidence: list[dict[str, Any]],
    details: dict[str, Any] | None = None,
    passed: bool | None = None,
) -> dict[str, Any]:
    result = {
        "tool": name,
        "version": 1,
        "tool_sha256": _tool_hash(function),
        "value": value,
        "unit": unit,
        "evidence_steps": [
            int(item["physics_step"])
            for item in evidence
            if item and item.get("physics_step") is not None
        ],
        "evidence": evidence,
        "details": details or {},
    }
    if passed is not None:
        result["passed"] = bool(passed)
    return result


def hammer_pickup_height(trajectory: TrajectoryView) -> dict[str, Any]:
    z = trajectory.trace["hammer_position"][:, 2]
    index = int(np.argmax(z))
    rise = float(z[index] - z[0])
    threshold = float(trajectory.schema.get("pickup_height_threshold_m", 0.03))
    return _result(
        "hammer_pickup_height",
        hammer_pickup_height,
        value=rise,
        unit="m",
        evidence=[_evidence(trajectory, index)],
        details={"initial_z_m": float(z[0]), "maximum_z_m": float(z[index]), "threshold_m": threshold},
        passed=rise >= threshold,
    )


def hammer_block_min_xy_error(trajectory: TrajectoryView) -> dict[str, Any]:
    delta = (
        trajectory.trace["hammer_functional_position"][:, :2]
        - trajectory.trace["block_functional_position"][:, :2]
    )
    linf = np.max(np.abs(delta), axis=1)
    index = int(np.argmin(linf))
    l2 = float(np.linalg.norm(delta[index]))
    threshold = max(
        float(value)
        for value in trajectory.schema["success_contract"]["xy_tolerance_m"]
    )
    return _result(
        "hammer_block_min_xy_error",
        hammer_block_min_xy_error,
        value=float(linf[index]),
        unit="m",
        evidence=[_evidence(trajectory, index)],
        details={
            "linf_error_m": float(linf[index]),
            "l2_error_m": l2,
            "delta_xy_m": [float(value) for value in delta[index]],
            "official_linf_threshold_m": threshold,
        },
        passed=float(linf[index]) < threshold,
    )


def hammer_block_contact_ever(trajectory: TrajectoryView) -> dict[str, Any]:
    contacts = trajectory.hammer_block_contacts()
    physical_contacts = [
        item for item in contacts if item.get("physical_contact", False)
    ]
    first = (
        min(
            physical_contacts,
            key=lambda item: item["first_physical_physics_step"],
        )
        if physical_contacts
        else None
    )
    physical = bool(physical_contacts)
    evidence = []
    if first:
        policy_step = int(first["first_physical_policy_step"])
        evidence.append(
            {
                "policy_step": policy_step,
                "physics_step": int(first["first_physical_physics_step"]),
                "simulation_time_seconds": float(
                    first["first_physical_simulation_time_seconds"]
                ),
                "video_frame_before": max(policy_step, 0),
                "video_frame_after": max(policy_step + 1, 0),
            }
        )
    return _result(
        "hammer_block_contact_ever",
        hammer_block_contact_ever,
        value=physical,
        unit=None,
        evidence=evidence,
        details={
            "reported_contact": bool(contacts),
            "physical_contact": physical,
            "contact_interval_count": len(contacts),
        },
        passed=physical,
    )


def first_contact_step(trajectory: TrajectoryView) -> dict[str, Any]:
    contacts = [
        item
        for item in trajectory.hammer_block_contacts()
        if item.get("physical_contact", False)
    ]
    first = (
        min(contacts, key=lambda item: item["first_physical_physics_step"])
        if contacts
        else None
    )
    evidence = []
    value = None
    if first:
        value = int(first["first_physical_physics_step"])
        policy_step = int(first["first_physical_policy_step"])
        evidence = [
            {
                "policy_step": policy_step,
                "physics_step": value,
                "simulation_time_seconds": float(
                    first["first_physical_simulation_time_seconds"]
                ),
                "video_frame_before": max(policy_step, 0),
                "video_frame_after": max(policy_step + 1, 0),
            }
        ]
    return _result(
        "first_contact_step",
        first_contact_step,
        value=value,
        unit="physics_step",
        evidence=evidence,
        details={
            "policy_step": int(first["first_physical_policy_step"])
            if first
            else None,
            "simulation_time_seconds": float(
                first["first_physical_simulation_time_seconds"]
            )
            if first
            else None,
        },
    )


def max_contact_impulse(trajectory: TrajectoryView) -> dict[str, Any]:
    contacts = trajectory.hammer_block_contacts()
    peak = max(contacts, key=lambda item: item.get("max_impulse", 0.0)) if contacts else None
    value = float(peak.get("max_impulse", 0.0)) if peak else 0.0
    evidence = []
    if peak:
        policy_step = int(peak["peak_policy_step"])
        evidence = [
            {
                "policy_step": policy_step,
                "physics_step": int(peak["peak_physics_step"]),
                "simulation_time_seconds": int(peak["peak_physics_step"])
                * float(trajectory.schema["physics_timestep_seconds"]),
                "video_frame_before": max(policy_step, 0),
                "video_frame_after": max(policy_step + 1, 0),
            }
        ]
    return _result(
        "max_contact_impulse",
        max_contact_impulse,
        value=value,
        unit="N*s",
        evidence=evidence,
        details={
            "estimated_peak_force_N": value
            / float(trajectory.schema["physics_timestep_seconds"]),
            "min_separation_m": peak.get("min_separation") if peak else None,
        },
    )


def ee_path_length(trajectory: TrajectoryView) -> dict[str, Any]:
    block_x = float(trajectory.trace["block_position"][0, 0])
    side = "left" if block_x < 0 else "right"
    positions = trajectory.trace[f"{side}_tcp_position"]
    increments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    value = float(np.sum(increments))
    index = int(np.argmax(increments)) + 1 if len(increments) else 0
    return _result(
        "ee_path_length",
        ee_path_length,
        value=value,
        unit="m",
        evidence=[_evidence(trajectory, index)],
        details={"active_arm": side, "initial_block_x_m": block_x},
    )


def official_check_success(trajectory: TrajectoryView) -> dict[str, Any]:
    final_success = bool(trajectory.metadata.get("success"))
    first = trajectory.success_events[0] if trajectory.success_events else None
    evidence = [first] if first else []
    return _result(
        "official_check_success",
        official_check_success,
        value=final_success,
        unit=None,
        evidence=evidence,
        details={
            "latched_eval_success": final_success,
            "success_transition_recorded": first is not None,
        },
        passed=final_success,
    )


def time_to_success(trajectory: TrajectoryView) -> dict[str, Any]:
    first = trajectory.success_events[0] if trajectory.success_events else None
    value = float(first["simulation_time_seconds"]) if first else None
    return _result(
        "time_to_success",
        time_to_success,
        value=value,
        unit="s",
        evidence=[first] if first else [],
        details={"physics_step": first.get("physics_step") if first else None},
    )


TOOL_CATALOG: dict[str, dict[str, Any]] = {
    "hammer_pickup_height": {
        "function": hammer_pickup_height,
        "description": "Maximum hammer center height rise from the initial state.",
        "tags": ["hammer", "pickup", "grasp", "拿起", "抬起"],
    },
    "hammer_block_min_xy_error": {
        "function": hammer_block_min_xy_error,
        "description": "Minimum official functional-point XY alignment error.",
        "tags": ["distance", "alignment", "接近", "距离", "敲"],
    },
    "hammer_block_contact_ever": {
        "function": hammer_block_contact_ever,
        "description": "Whether hammer and block ever had physical contact.",
        "tags": ["contact", "hit", "接触", "敲"],
    },
    "first_contact_step": {
        "function": first_contact_step,
        "description": "First hammer-block contact physics step and time.",
        "tags": ["first", "contact", "首次", "接触", "时间"],
    },
    "max_contact_impulse": {
        "function": max_contact_impulse,
        "description": "Maximum contact-point impulse during hammer-block contact.",
        "tags": ["impulse", "force", "contact", "冲量", "力度", "接触"],
    },
    "ee_path_length": {
        "function": ee_path_length,
        "description": "Active-arm TCP path length at physics resolution.",
        "tags": ["path", "motion", "轨迹", "路径", "运动"],
    },
    "official_check_success": {
        "function": official_check_success,
        "description": "Latched official RoboTwin task success.",
        "tags": ["success", "result", "成功", "结果", "评估"],
    },
    "time_to_success": {
        "function": time_to_success,
        "description": "Physics simulation time of the first official success.",
        "tags": ["time", "success", "耗时", "成功", "时间"],
    },
}


def public_tool_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": item["description"],
            "tags": item["tags"],
            "version": 1,
            "sha256": _tool_hash(item["function"]),
        }
        for name, item in TOOL_CATALOG.items()
    ]


def run_trusted_tools(
    trajectory: TrajectoryView,
    tool_names: list[str],
) -> list[dict[str, Any]]:
    unknown = [name for name in tool_names if name not in TOOL_CATALOG]
    if unknown:
        raise TrajectoryError(f"未知 trusted tools: {unknown}")
    return [TOOL_CATALOG[name]["function"](trajectory) for name in tool_names]
