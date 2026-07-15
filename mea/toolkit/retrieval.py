"""Auditable retrieval over the trusted tool catalog."""

from __future__ import annotations

from typing import Any

from .tools import TOOL_CATALOG, public_tool_catalog


class TrustedToolRetriever:
    """Select a compact tool set without another model call in the first MVP."""

    BASELINE = (
        "hammer_pickup_height",
        "first_hammer_pickup_step",
        "hammer_block_min_xy_error",
        "hammer_block_contact_ever",
        "first_contact_step",
        "official_check_success",
    )

    def select(self, user_request: str, *, task_name: str) -> dict[str, Any]:
        if task_name != "beat_block_hammer":
            raise ValueError("第一版 Tool Retriever 只支持 beat_block_hammer")
        text = user_request.lower()
        selected = list(self.BASELINE)
        reasons = {name: "BeatBlockHammer baseline evidence" for name in selected}
        optional = {
            "max_contact_impulse": ("impulse", "force", "力度", "冲量", "接触", "敲"),
            "ee_path_length": ("path", "motion", "路径", "轨迹", "运动"),
            "time_to_success": ("time", "耗时", "时间", "第几步", "成功"),
        }
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
            "selection_mode": "deterministic_keyword_plus_task_baseline",
        }
