"""Allowlisted telemetry profiles used by :mod:`mea.toolkit.recorder`.

Profiles are deliberately trusted repository data.  A Plan Agent may select a
profile id, but it cannot provide arbitrary sampling code or field paths.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


_PROFILES: dict[str, dict[str, Any]] = {
    "legacy_v1": {
        "schema_version": 1,
        "profile_id": "legacy_v1",
        "preserve_legacy_artifacts": True,
        "force_initial_sample": True,
        "force_final_sample": True,
        "streams": {
            "policy_state": {
                "sampling": "policy_boundary",
                "field_groups": ["legacy_full_state"],
                "artifact": "states.csv",
            },
            "semantic_trace": {
                "sampling": "physics_period",
                "every_physics_steps": 1,
                "field_groups": ["task_schema_semantic_fields"],
                "artifact": "semantic_trace.npz",
            },
            "contact_events": {
                "sampling": "physics_period",
                "every_physics_steps": 1,
                "mode": "interval_summary",
                "scope": "task_schema_contact_focus",
                "artifact": "events.jsonl",
            },
        },
    },
    "balanced_v1": {
        "schema_version": 1,
        "profile_id": "balanced_v1",
        "preserve_legacy_artifacts": True,
        "force_initial_sample": True,
        "force_final_sample": True,
        "streams": {
            "policy_state": {
                "sampling": "policy_boundary",
                "field_groups": ["legacy_full_state"],
                "artifact": "states.csv",
            },
            "semantic_trace": {
                "sampling": "physics_period",
                "every_physics_steps": 1,
                "field_groups": ["task_schema_semantic_fields"],
                "artifact": "semantic_trace.npz",
            },
            "dynamics_trace": {
                "sampling": "physics_period",
                "every_physics_steps": 5,
                "field_groups": [
                    "robot_joint_state",
                    "robot_end_effector_state",
                    "tracked_actor_rigid_state",
                    "tracked_actor_functional_pose",
                    "tracked_actor_contact_pose",
                ],
                "artifact": "dynamics_trace.npz",
                "float_dtype": "float32",
            },
            "contact_events": {
                "sampling": "physics_period",
                "every_physics_steps": 1,
                "mode": "interval_summary",
                "scope": "task_schema_contact_focus",
                "artifact": "events.jsonl",
            },
        },
    },
}


def load_telemetry_profile(profile_id: str) -> dict[str, Any]:
    """Return a defensive copy of one trusted telemetry profile."""

    try:
        return deepcopy(_PROFILES[profile_id])
    except KeyError as exc:
        raise ValueError(
            f"unknown telemetry profile {profile_id!r}; "
            f"expected one of {sorted(_PROFILES)}"
        ) from exc


def telemetry_profile_sha256(profile: dict[str, Any]) -> str:
    """Hash the canonical JSON representation stored beside an episode."""

    payload = json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
