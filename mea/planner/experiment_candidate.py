"""Runtime experiment candidates discovered from an open Query.

An ``ExperimentCandidate`` is the semantic hand-off between planning and the
TaskGen, ToolGen, and VQA stages.  It deliberately carries no catalog template
identifier.  Version 2 makes each materialization need optional and typed, so
a trajectory-only Query can request a Tool without inventing a new scene or
checker.  Version-1 string candidates remain accepted and are normalized to
the version-2 representation.
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
_NEED_KEYS = {"kind", "description", "reuse_first"}
_NEED_KINDS = frozenset({"reuse", "adapt", "generate", "measure", "vqa"})
_FIELD_KINDS = {
    "scene_need": frozenset({"reuse", "adapt", "generate"}),
    "checker_need": frozenset({"reuse", "adapt", "generate"}),
    "tool_need": frozenset({"reuse", "adapt", "generate", "measure", "vqa"}),
}
_LEGACY_NEED_KIND = {
    "scene_need": "adapt",
    "checker_need": "generate",
    "tool_need": "measure",
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


def _need(
    value: Any,
    *,
    field: str,
    legacy_string: bool,
) -> dict[str, Any] | None:
    if value is None:
        if legacy_string:
            raise ExperimentCandidateError(
                f"ExperimentCandidate v1 {field} must be a non-empty string"
            )
        return None
    if isinstance(value, str):
        description = _text(value, f"ExperimentCandidate.{field}")
        return {
            "kind": _LEGACY_NEED_KIND[field],
            "description": description,
            "reuse_first": field == "tool_need",
        }
    if not isinstance(value, Mapping) or set(value) != _NEED_KEYS:
        raise ExperimentCandidateError(
            f"ExperimentCandidate.{field} must be null, a legacy string, or "
            f"an object with exactly {sorted(_NEED_KEYS)}"
        )
    need = deepcopy(dict(value))
    kind = need.get("kind")
    if kind not in _NEED_KINDS or kind not in _FIELD_KINDS[field]:
        raise ExperimentCandidateError(
            f"ExperimentCandidate.{field}.kind must be one of "
            f"{sorted(_FIELD_KINDS[field])}"
        )
    need["description"] = _text(
        need.get("description"),
        f"ExperimentCandidate.{field}.description",
    )
    if not isinstance(need.get("reuse_first"), bool):
        raise ExperimentCandidateError(
            f"ExperimentCandidate.{field}.reuse_first must be bool"
        )
    return need


def experiment_candidate_need_kinds(
    value: Mapping[str, Any],
) -> frozenset[str]:
    """Return the stages explicitly requested by one normalized candidate."""

    candidate = validate_experiment_candidate(value)
    return frozenset(
        field.removesuffix("_need")
        for field in ("scene_need", "checker_need", "tool_need")
        if candidate[field] is not None
    )


def validate_experiment_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the planner-to-generation hand-off.

    Version 1 required three strings.  It is accepted only as a compatibility
    input and always returned as version 2 with typed need objects.
    """

    if not isinstance(value, Mapping) or set(value) != _CANDIDATE_KEYS:
        raise ExperimentCandidateError(
            "ExperimentCandidate fields must be exactly "
            f"{sorted(_CANDIDATE_KEYS)}"
        )
    candidate = deepcopy(dict(value))
    schema_version = candidate.get("schema_version")
    if schema_version not in {1, 2}:
        raise ExperimentCandidateError(
            "ExperimentCandidate.schema_version must be 1 or 2"
        )
    candidate["candidate_id"] = _candidate_id(candidate.get("candidate_id"))
    for field in (
        "source_query",
        "base_task",
        "semantic_concern",
    ):
        candidate[field] = _text(
            candidate.get(field), f"ExperimentCandidate.{field}"
        )
    for field in ("scene_need", "checker_need", "tool_need"):
        candidate[field] = _need(
            candidate.get(field),
            field=field,
            legacy_string=schema_version == 1,
        )
    if not any(
        candidate[field] is not None
        for field in ("scene_need", "checker_need", "tool_need")
    ):
        raise ExperimentCandidateError(
            "ExperimentCandidate must request at least one scene, checker, or "
            "tool need"
        )
    candidate["schema_version"] = 2
    return candidate


def build_experiment_candidate(
    *,
    source_query: str,
    base_task: str,
    semantic_concern: str,
    scene_need: str | Mapping[str, Any] | None = None,
    checker_need: str | Mapping[str, Any] | None = None,
    tool_need: str | Mapping[str, Any] | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Build one catalog-independent runtime experiment candidate."""

    task = _text(base_task, "base_task")
    concern = _text(semantic_concern, "semantic_concern")
    scene = _need(scene_need, field="scene_need", legacy_string=False)
    checker = _need(checker_need, field="checker_need", legacy_string=False)
    tool = _need(tool_need, field="tool_need", legacy_string=False)
    if scene is None and checker is None and tool is None:
        raise ExperimentCandidateError(
            "ExperimentCandidate must request at least one scene, checker, or "
            "tool need"
        )
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
            "schema_version": 2,
            "candidate_id": resolved_id,
            "source_query": source_query,
            "base_task": task,
            "semantic_concern": concern,
            "scene_need": scene,
            "checker_need": checker,
            "tool_need": tool,
        }
    )


__all__ = [
    "ExperimentCandidateError",
    "build_experiment_candidate",
    "experiment_candidate_need_kinds",
    "validate_experiment_candidate",
]
