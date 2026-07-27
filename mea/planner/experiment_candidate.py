"""Runtime experiment candidates discovered from an open Query.

An ``ExperimentCandidate`` is the semantic hand-off between planning and the
reuse-or-generate TaskGen/ToolGen stages.  It deliberately carries no catalog
template identifier: a registered task may be retrieved as the base program,
but the requested scene, checker, and measurement remain Query-derived.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping

class ExperimentCandidateError(ValueError):
    """Raised when a runtime experiment candidate is malformed."""


_CANDIDATE_KEYS = {
    "schema_version",
    "candidate_id",
    "source_query",
    "base_task",
    "semantic_concern",
    "scene_need",
    "checker_need",
    "tool_need",
}
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentCandidateError(f"{field} must be a non-empty string")
    return value.strip()


def _candidate_id(value: Any) -> str:
    candidate_id = _text(value, "ExperimentCandidate.candidate_id")
    if not _CANDIDATE_ID.fullmatch(candidate_id):
        raise ExperimentCandidateError(
            "ExperimentCandidate.candidate_id must contain only letters, "
            "digits, dot, underscore, colon, or hyphen"
        )
    return candidate_id


def _slug(value: str, *, field: str) -> str:
    slug = _SLUG_SEPARATOR.sub(".", _text(value, field).casefold()).strip(".")
    if not slug:
        raise ExperimentCandidateError(f"{field} cannot produce a candidate id")
    return slug


def validate_experiment_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the small planner-to-generation hand-off exactly."""

    if not isinstance(value, Mapping) or set(value) != _CANDIDATE_KEYS:
        raise ExperimentCandidateError(
            "ExperimentCandidate fields must be exactly "
            f"{sorted(_CANDIDATE_KEYS)}"
        )
    candidate = deepcopy(dict(value))
    if candidate.get("schema_version") != 1:
        raise ExperimentCandidateError(
            "ExperimentCandidate.schema_version must be 1"
        )
    candidate["candidate_id"] = _candidate_id(candidate.get("candidate_id"))
    for field in sorted(
        _CANDIDATE_KEYS - {"schema_version", "candidate_id"}
    ):
        candidate[field] = _text(
            candidate.get(field), f"ExperimentCandidate.{field}"
        )
    return candidate


def build_experiment_candidate(
    *,
    source_query: str,
    base_task: str,
    semantic_concern: str,
    scene_need: str,
    checker_need: str,
    tool_need: str,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Build one catalog-independent runtime experiment candidate."""

    task = _text(base_task, "base_task")
    concern = _text(semantic_concern, "semantic_concern")
    scene = _text(scene_need, "scene_need")
    checker = _text(checker_need, "checker_need")
    experiment_digest = hashlib.sha256(
        json.dumps(
            {
                "base_task": task,
                "semantic_concern": concern,
                "scene_need": scene,
                "checker_need": checker,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    resolved_id = (
        _candidate_id(candidate_id)
        if candidate_id is not None
        else "dynamic."
        + _slug(task, field="base_task")
        + "."
        + _slug(concern, field="semantic_concern")
        + "."
        + experiment_digest
    )
    return validate_experiment_candidate(
        {
            "schema_version": 1,
            "candidate_id": resolved_id,
            "source_query": source_query,
            "base_task": task,
            "semantic_concern": concern,
            "scene_need": scene,
            "checker_need": checker,
            "tool_need": tool_need,
        }
    )


__all__ = [
    "ExperimentCandidateError",
    "build_experiment_candidate",
    "validate_experiment_candidate",
]
