"""Verified, standalone few-shot examples for ToolGen prompts.

Unlike the permanent Trusted Tools, these functions return only the payload a
generated tool is allowed to return.  ToolGen adds provenance and source hashes
after execution.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def first_hammer_pickup_step_example(trajectory):
    z = trajectory.trace["hammer_position"][:, 2]
    initial_z = float(z[0])
    rise = z - initial_z
    threshold = float(trajectory.schema.get("pickup_height_threshold_m", 0.03))
    indices = np.where(rise >= threshold)[0]
    index = int(indices[0]) if len(indices) else None
    physics_step = (
        int(trajectory.trace["physics_step"][index])
        if index is not None
        else None
    )
    return {
        "value": physics_step,
        "unit": "physics_step",
        "passed": index is not None,
        "evidence_steps": [physics_step] if physics_step is not None else [],
        "details": {
            "simulation_time_seconds": (
                float(trajectory.trace["simulation_time_seconds"][index])
                if index is not None
                else None
            ),
            "initial_z_m": initial_z,
            "pickup_height_threshold_m": threshold,
            "height_rise_at_pickup_m": (
                float(rise[index]) if index is not None else None
            ),
            "maximum_height_rise_m": float(np.max(rise)),
        },
    }


def hammer_block_contact_example(trajectory):
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
    return {
        "value": physical,
        "unit": None,
        "passed": physical,
        "evidence_steps": (
            [int(first["first_physical_physics_step"])] if first else []
        ),
        "details": {
            "reported_contact": bool(contacts),
            "physical_contact": physical,
            "contact_interval_count": len(contacts),
        },
    }


def first_contact_step_example(trajectory):
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
    value = int(first["first_physical_physics_step"]) if first else None
    return {
        "value": value,
        "unit": "physics_step",
        "passed": None,
        "evidence_steps": [value] if value is not None else [],
        "details": {
            "policy_step": (
                int(first["first_physical_policy_step"]) if first else None
            ),
            "simulation_time_seconds": (
                float(first["first_physical_simulation_time_seconds"])
                if first
                else None
            ),
        },
    }


def max_contact_impulse_example(trajectory):
    contacts = trajectory.hammer_block_contacts()
    peak = (
        max(contacts, key=lambda item: item.get("max_impulse", 0.0))
        if contacts
        else None
    )
    value = float(peak.get("max_impulse", 0.0)) if peak else 0.0
    physics_step = int(peak["peak_physics_step"]) if peak else None
    return {
        "value": value,
        "unit": "N*s",
        "passed": None,
        "evidence_steps": [physics_step] if physics_step is not None else [],
        "details": {
            "estimated_peak_force_N": value
            / float(trajectory.schema["physics_timestep_seconds"]),
            "min_separation_m": peak.get("min_separation") if peak else None,
        },
    }


def time_to_success_example(trajectory):
    first = trajectory.success_events[0] if trajectory.success_events else None
    return {
        "value": float(first["simulation_time_seconds"]) if first else None,
        "unit": "s",
        "passed": None,
        "evidence_steps": (
            [int(first["physics_step"])] if first else []
        ),
        "details": {
            "physics_step": first.get("physics_step") if first else None
        },
    }


EXAMPLE_CATALOG: dict[str, dict[str, Any]] = {
    "first_hammer_pickup_step": {
        "function": first_hammer_pickup_step_example,
        "description": (
            "First physics step where hammer height rise reaches the task "
            "pickup threshold."
        ),
        "tags": ["hammer", "pickup", "first", "step", "拿起", "首次", "时间"],
    },
    "hammer_block_contact_ever": {
        "function": hammer_block_contact_example,
        "description": "Whether hammer and block ever had physical contact.",
        "tags": ["hammer", "block", "contact", "hit", "接触", "敲击"],
    },
    "first_contact_step": {
        "function": first_contact_step_example,
        "description": "First physical hammer-block contact step.",
        "tags": ["first", "contact", "step", "首次", "接触", "时间"],
    },
    "max_contact_impulse": {
        "function": max_contact_impulse_example,
        "description": "Maximum hammer-block contact impulse.",
        "tags": ["contact", "impulse", "force", "接触", "冲量", "力度"],
    },
    "time_to_success": {
        "function": time_to_success_example,
        "description": (
            "First official-success time, including an explicit null result "
            "when the event is absent."
        ),
        "tags": ["time", "duration", "success", "null", "耗时", "时间", "缺失"],
    },
}


def example_function(name: str) -> Callable[..., dict[str, Any]]:
    return EXAMPLE_CATALOG[name]["function"]
