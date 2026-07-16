"""Auditable retrieval over task-compatible trusted tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import load_task_schema
from .tools import TOOL_CATALOG, public_tool_catalog


class TrustedToolRetriever:
    """Select a deterministic task profile before any optional keyword tools."""

    BEAT_BLOCK_HAMMER_BASELINE = (
        "hammer_pickup_height",
        "first_hammer_pickup_step",
        "hammer_block_min_xy_error",
        "hammer_block_contact_ever",
        "first_contact_step",
        "official_check_success",
    )
    # Every schema-backed task can at least expose its official outcome and
    # optional first-success time without a task-specific metric implementation.
    GENERIC_BASELINE = (
        "official_check_success",
        "time_to_success",
    )

    def __init__(self, repo_root: str | Path | None = None):
        self.repo_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )

    def select(self, user_request: str, *, task_name: str) -> dict[str, Any]:
        schema = load_task_schema(self.repo_root, task_name)
        profile = schema.get("trusted_tool_profile", "generic_success")
        text = user_request.lower()
        selected = list(
            self.BEAT_BLOCK_HAMMER_BASELINE
            if profile == "beat_block_hammer"
            else self.GENERIC_BASELINE
        )
        reason = (
            "BeatBlockHammer baseline evidence"
            if profile == "beat_block_hammer"
            else "schema-backed generic outcome evidence"
        )
        reasons = {name: reason for name in selected}
        optional = {
            "max_contact_impulse": (
                "impulse", "force", "力度", "冲量", "接触", "敲"
            ),
            "ee_path_length": ("path", "motion", "路径", "轨迹", "运动"),
            "time_to_success": ("time", "耗时", "时间", "第几步", "成功"),
        }
        # These optional tools currently have BBH-specific contracts.  Generic
        # tasks never receive them merely because the wording happens to match.
        if profile == "beat_block_hammer":
            for name, terms in optional.items():
                matches = [term for term in terms if term in text]
                if matches:
                    selected.append(name)
                    reasons[name] = f"matched request terms: {matches}"
        selected = [name for name in TOOL_CATALOG if name in selected]
        return {
            "schema_version": 1,
            "task_name": task_name,
            "selected_tools": selected,
            "reasons": reasons,
            "catalog": public_tool_catalog(),
            "task_schema": {
                "schema_version": schema["schema_version"],
                "task_family": schema.get("task_family"),
                "trusted_tool_profile": profile,
            },
            "selection_mode": "schema_profile_plus_deterministic_keywords",
        }
