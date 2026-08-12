"""Small retrieval helper for simulator-backed task implementation guides."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


_TASK_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def load_task_guide(repo_root: str | Path, task_name: str) -> str:
    """Return one optional task guide; absence means no task-local guidance."""

    if not isinstance(task_name, str) or not _TASK_NAME.fullmatch(task_name):
        return ""
    path = (
        Path(repo_root).expanduser().resolve()
        / "mea"
        / "knowledge"
        / "tasks"
        / f"{task_name}.md"
    )
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def task_guide_from_capabilities(
    repo_root: str | Path,
    capabilities: Mapping[str, Any],
) -> str:
    """Resolve the guide after the runtime has bound a concrete task."""

    if not isinstance(capabilities, Mapping):
        return ""
    for card_name in ("simulator_card", "policy_card"):
        card = capabilities.get(card_name)
        if isinstance(card, Mapping) and isinstance(card.get("task_name"), str):
            return load_task_guide(repo_root, card["task_name"])
    return ""


__all__ = ["load_task_guide", "task_guide_from_capabilities"]
