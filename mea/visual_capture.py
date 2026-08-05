"""Task-independent visual evidence profiles for policy rollouts."""

from __future__ import annotations

from typing import Any, Mapping


EVENT_KEYFRAMES_PROFILE = "event_keyframes_v1"
TEMPORAL_KEYFRAMES_PROFILE = "temporal_keyframes_v1"

VISUAL_CAPTURE_PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    EVENT_KEYFRAMES_PROFILE: {
        "mode": "event_keyframes",
        "policy_step_period": None,
        "max_periodic_frames": 0,
    },
    TEMPORAL_KEYFRAMES_PROFILE: {
        "mode": "temporal_keyframes",
        "policy_step_period": 10,
        "max_periodic_frames": 8,
    },
}
VISUAL_CAPTURE_PROFILES = frozenset(VISUAL_CAPTURE_PROFILE_CONFIGS)


def visual_capture_profile_for_proposal(
    proposal: Mapping[str, Any],
) -> str:
    """Require bounded temporal evidence exactly when a VQA Tool is needed."""

    return (
        TEMPORAL_KEYFRAMES_PROFILE
        if proposal.get("vqa_tool_need") is not None
        else EVENT_KEYFRAMES_PROFILE
    )


__all__ = [
    "EVENT_KEYFRAMES_PROFILE",
    "TEMPORAL_KEYFRAMES_PROFILE",
    "VISUAL_CAPTURE_PROFILE_CONFIGS",
    "VISUAL_CAPTURE_PROFILES",
    "visual_capture_profile_for_proposal",
]
