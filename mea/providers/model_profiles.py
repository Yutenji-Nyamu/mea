"""Named model profiles with explicit per-stage overrides."""

from __future__ import annotations

from typing import Mapping


MODEL_STAGES = (
    "planner",
    "taskgen",
    "toolgen",
    "vision",
    "feedback",
)

MODEL_PROFILES: dict[str, dict[str, str]] = {
    "legacy": {
        stage: "gpt-4o-2024-11-20" for stage in MODEL_STAGES
    },
    "economy": {
        stage: "gpt-5.6-luna" for stage in MODEL_STAGES
    },
    "balanced": {
        "planner": "gpt-5.6-luna",
        "taskgen": "gpt-5.6-terra",
        "toolgen": "gpt-5.6-terra",
        "vision": "gpt-5.6-luna",
        "feedback": "gpt-5.6-luna",
    },
    "quality": {
        stage: "gpt-5.6-sol" for stage in MODEL_STAGES
    },
}


class ModelProfileError(ValueError):
    """Raised when a profile or override is invalid."""


def available_model_profiles() -> tuple[str, ...]:
    return tuple(MODEL_PROFILES)


def resolve_model_profile(
    profile: str,
    overrides: Mapping[str, str | None] | None = None,
) -> dict[str, str]:
    """Resolve one trusted profile and then apply explicit stage overrides."""

    if profile not in MODEL_PROFILES:
        raise ModelProfileError(f"未知 model profile: {profile}")
    resolved = dict(MODEL_PROFILES[profile])
    for stage, model in (overrides or {}).items():
        if stage not in MODEL_STAGES:
            raise ModelProfileError(f"未知 model stage: {stage}")
        if model is None:
            continue
        if not isinstance(model, str) or not model.strip():
            raise ModelProfileError(f"{stage} model 必须是非空字符串")
        resolved[stage] = model.strip()
    return resolved
