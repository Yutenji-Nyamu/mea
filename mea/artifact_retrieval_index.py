"""Non-authoritative retrieval index for reviewed MEA artifacts.

The paper-aligned production method may retrieve a previously reviewed
Task/Tool/VQA artifact when its semantic key matches the current Query.  This
index is only that retrieval surface: it neither declares which RoboTwin tasks
are executable nor restricts which concerns the Planner may propose.

Known records are temporarily sourced from ``capability_adapter`` while legacy
paper protocols still use its complete task views.  Keeping the projection
here gives production code a task-menu-free API and lets the legacy module
remain a compatibility export during caller migration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


class ArtifactRetrievalIndexError(ValueError):
    """Raised when an artifact retrieval index request is malformed."""


OFFICIAL_CONTROL_TEMPLATE_ID = "task_execution.official_baseline"


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactRetrievalIndexError(f"{field} must be a non-empty string")
    return value.strip()


def _load_legacy_known_artifact_adapter(
    task_name: str,
) -> Mapping[str, Any] | None:
    """Read one reviewed legacy record without treating it as authority.

    The import is intentionally lazy: ``capability_adapter`` compatibility
    callers delegate back to this module.  The bridge can be deleted after its
    remaining paper-protocol callers move to a dedicated compatibility layer.
    """

    from .capability_adapter import _known_artifact_adapter

    return _known_artifact_adapter(task_name)


def resolve_task_retrieval_index(
    task_name: Any,
    *,
    allow_unregistered: bool = False,
) -> dict[str, Any]:
    """Return reviewed retrieval hints for ``task_name``.

    Unknown tasks may request an empty index when their source/schema/checkpoint
    execution binding has already been validated elsewhere.  In both cases
    ``execution_authority`` is always false.
    """

    if not isinstance(allow_unregistered, bool):
        raise ArtifactRetrievalIndexError("allow_unregistered must be bool")
    normalized = _text(task_name, field="task_name")
    adapter = _load_legacy_known_artifact_adapter(normalized)
    if adapter is None:
        if not allow_unregistered:
            raise ArtifactRetrievalIndexError(
                f"no reviewed artifact index for task: {normalized!r}"
            )
        return {
            "schema_version": 1,
            "index_role": "retrieval_only",
            "execution_authority": False,
            "task_name": normalized,
            "control_template_id": OFFICIAL_CONTROL_TEMPLATE_ID,
            "entries": [],
            "vqa_questions": {},
            "vqa_metric_rules": {},
        }
    return {
        "schema_version": 1,
        "index_role": "retrieval_only",
        "execution_authority": False,
        "task_name": adapter["task_name"],
        "control_template_id": adapter["control_template_id"],
        "entries": deepcopy(adapter["capability_contracts"]),
        "vqa_questions": deepcopy(adapter["vqa_questions"]),
        "vqa_metric_rules": deepcopy(adapter["vqa_metric_rules"]),
    }


__all__ = [
    "ArtifactRetrievalIndexError",
    "OFFICIAL_CONTROL_TEMPLATE_ID",
    "resolve_task_retrieval_index",
]
