"""Hash-pinned reviewed selectors for the trusted Execution VQA catalog.

The registry is intentionally read-only.  A VQAQuerySpec can select existing
``QUESTION_CATALOG`` phenomena, but it cannot add question text or alter the
answer contract.  Admission artifacts are prepared out of band and must carry
an explicit human or development-agent-proxy approval.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .query import QUESTION_CATALOG


REGISTRY_SCHEMA_VERSION = 1
SPEC_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = 1
REGISTRY_SCOPE = "reviewed_persistent_vqa_query_specs"
REVIEW_SCOPE = "persistent_vqa_query_spec_reuse"
ANSWER_CONTRACT_ID = "execution_vqa_binary_v1"
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")

INDEX_KEYS = {"schema_version", "scope", "entries"}
INDEX_ENTRY_KEYS = {
    "spec_id",
    "spec_artifact",
    "spec_artifact_sha256",
    "spec_sha256",
    "review_artifact",
    "review_artifact_sha256",
}
SPEC_KEYS = {
    "schema_version",
    "spec_id",
    "task_name",
    "template_ids",
    "sub_aspect_prefixes",
    "tool_metrics",
    "phenomenon_ids",
    "answer_contract_id",
}
REVIEW_KEYS = {
    "schema_version",
    "decision",
    "review_scope",
    "reviewer",
    "reviewed_at",
    "spec_sha256",
    "checks",
    "notes",
}
REVIEW_CHECK_KEYS = {
    "task_scope_reviewed",
    "phenomena_reviewed",
    "numeric_authority_reviewed",
    "answer_contract_reviewed",
    "tests_reviewed",
}
REVIEWER_KINDS = {"human", "development_agent_proxy"}
LOADED_ENTRY_KEYS = {
    "spec",
    "review",
    "spec_sha256",
    "spec_artifact",
    "review_artifact",
}


class ReviewedVQAQuerySpecError(RuntimeError):
    """Raised when a reviewed VQAQuerySpec registry is not trustworthy."""


def canonical_sha256(value: Any) -> str:
    """Hash semantic JSON content independently of pretty-print formatting."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise ReviewedVQAQuerySpecError(f"{field} must be a lowercase SHA-256")
    return value


def _string_list(
    value: Any,
    *,
    field: str,
    nonempty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ReviewedVQAQuerySpecError(f"{field} must be a string list")
    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        raise ReviewedVQAQuerySpecError(f"{field} must contain unique values")
    if nonempty and not normalized:
        raise ReviewedVQAQuerySpecError(f"{field} must be non-empty")
    return normalized


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewedVQAQuerySpecError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewedVQAQuerySpecError(f"{label} must be a JSON object")
    return value


def _safe_file(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ReviewedVQAQuerySpecError(f"{label} path must be non-empty")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ReviewedVQAQuerySpecError(f"{label} path escapes registry root")
    current = root
    for part in candidate_relative.parts:
        current = current / part
        if current.is_symlink():
            raise ReviewedVQAQuerySpecError(f"{label} path must not contain symlinks")
    candidate = (root / candidate_relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ReviewedVQAQuerySpecError(f"{label} path escapes registry root") from exc
    if not candidate.is_file():
        raise ReviewedVQAQuerySpecError(f"{label} file is missing: {candidate}")
    return candidate


def validate_vqa_query_spec(value: Any) -> dict[str, Any]:
    """Validate a selection-only VQAQuerySpec against the trusted catalog."""

    if not isinstance(value, Mapping) or set(value) != SPEC_KEYS:
        raise ReviewedVQAQuerySpecError(
            f"VQAQuerySpec fields must be exactly {sorted(SPEC_KEYS)}"
        )
    if value.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ReviewedVQAQuerySpecError("VQAQuerySpec schema_version must be 1")
    spec_id = value.get("spec_id")
    if not isinstance(spec_id, str) or re.fullmatch(r"vqa_[a-z0-9_]+", spec_id) is None:
        raise ReviewedVQAQuerySpecError("spec_id must match vqa_[a-z0-9_]+")
    task_name = value.get("task_name")
    if not isinstance(task_name, str) or not task_name.strip():
        raise ReviewedVQAQuerySpecError("task_name must be non-empty")
    template_ids = _string_list(value.get("template_ids"), field="template_ids")
    aspect_prefixes = _string_list(
        value.get("sub_aspect_prefixes"), field="sub_aspect_prefixes"
    )
    tool_metrics = _string_list(value.get("tool_metrics"), field="tool_metrics")
    if not any((template_ids, aspect_prefixes, tool_metrics)):
        raise ReviewedVQAQuerySpecError(
            "VQAQuerySpec needs at least one context selector"
        )
    phenomenon_ids = _string_list(
        value.get("phenomenon_ids"),
        field="phenomenon_ids",
        nonempty=True,
    )
    unknown = sorted(set(phenomenon_ids) - set(QUESTION_CATALOG))
    if unknown:
        raise ReviewedVQAQuerySpecError(
            f"VQAQuerySpec contains non-allowlisted phenomena: {unknown}"
        )
    if value.get("answer_contract_id") != ANSWER_CONTRACT_ID:
        raise ReviewedVQAQuerySpecError("answer_contract_id is not allowlisted")
    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "spec_id": spec_id,
        "task_name": task_name.strip(),
        "template_ids": template_ids,
        "sub_aspect_prefixes": aspect_prefixes,
        "tool_metrics": tool_metrics,
        "phenomenon_ids": phenomenon_ids,
        "answer_contract_id": ANSWER_CONTRACT_ID,
    }


def validate_vqa_query_review(
    value: Any,
    *,
    spec_sha256: str,
) -> dict[str, Any]:
    """Require an explicit review pinned to the exact semantic spec hash."""

    expected_spec_hash = _require_hash(spec_sha256, field="spec_sha256")
    if not isinstance(value, Mapping) or set(value) != REVIEW_KEYS:
        raise ReviewedVQAQuerySpecError(
            f"review fields must be exactly {sorted(REVIEW_KEYS)}"
        )
    if value.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ReviewedVQAQuerySpecError("review schema_version must be 1")
    if value.get("decision") != "approved":
        raise ReviewedVQAQuerySpecError("review decision must be approved")
    if value.get("review_scope") != REVIEW_SCOPE:
        raise ReviewedVQAQuerySpecError(f"review_scope must be {REVIEW_SCOPE}")
    reviewer = value.get("reviewer")
    if not isinstance(reviewer, Mapping) or set(reviewer) != {"id", "kind"}:
        raise ReviewedVQAQuerySpecError("reviewer must contain exactly id and kind")
    if not isinstance(reviewer.get("id"), str) or not reviewer["id"].strip():
        raise ReviewedVQAQuerySpecError("reviewer.id must be non-empty")
    if reviewer.get("kind") not in REVIEWER_KINDS:
        raise ReviewedVQAQuerySpecError(
            "reviewer.kind must be human or development_agent_proxy"
        )
    reviewed_at = value.get("reviewed_at")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise ReviewedVQAQuerySpecError("reviewed_at must be ISO-8601")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewedVQAQuerySpecError("reviewed_at must be ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise ReviewedVQAQuerySpecError("reviewed_at must include a timezone")
    if value.get("spec_sha256") != expected_spec_hash:
        raise ReviewedVQAQuerySpecError("review is not pinned to the exact spec")
    checks = value.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != REVIEW_CHECK_KEYS:
        raise ReviewedVQAQuerySpecError(
            "review checks must contain exactly the required fields"
        )
    failed = sorted(key for key in REVIEW_CHECK_KEYS if checks.get(key) is not True)
    if failed:
        raise ReviewedVQAQuerySpecError(f"review checks were not approved: {failed}")
    if not isinstance(value.get("notes"), str):
        raise ReviewedVQAQuerySpecError("review notes must be a string")
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "decision": "approved",
        "review_scope": REVIEW_SCOPE,
        "reviewer": {
            "id": reviewer["id"].strip(),
            "kind": reviewer["kind"],
        },
        "reviewed_at": reviewed_at,
        "spec_sha256": expected_spec_hash,
        "checks": {key: True for key in sorted(REVIEW_CHECK_KEYS)},
        "notes": value["notes"],
    }


def load_reviewed_vqa_query_specs(
    registry_dir: str | Path,
) -> list[dict[str, Any]]:
    """Load and fully verify every immutable reviewed registry entry."""

    raw_root = Path(registry_dir).expanduser()
    if raw_root.is_symlink():
        raise ReviewedVQAQuerySpecError("registry root must not be a symlink")
    root = raw_root.resolve()
    index_path = root / "index.json"
    if index_path.is_symlink():
        raise ReviewedVQAQuerySpecError("registry index path must not contain symlinks")
    if not index_path.exists():
        return []
    index_path = _safe_file(root, "index.json", label="registry index")
    index = _read_json(index_path, label="registry index")
    if set(index) != INDEX_KEYS:
        raise ReviewedVQAQuerySpecError(
            f"registry index fields must be exactly {sorted(INDEX_KEYS)}"
        )
    if (
        index.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or index.get("scope") != REGISTRY_SCOPE
        or not isinstance(index.get("entries"), list)
    ):
        raise ReviewedVQAQuerySpecError("unsupported reviewed VQA registry index")

    loaded: list[dict[str, Any]] = []
    seen_spec_ids: set[str] = set()
    for entry_index, entry in enumerate(index["entries"]):
        if not isinstance(entry, Mapping) or set(entry) != INDEX_ENTRY_KEYS:
            raise ReviewedVQAQuerySpecError(
                f"registry entries[{entry_index}] has invalid fields"
            )
        spec_artifact_hash = _require_hash(
            entry.get("spec_artifact_sha256"),
            field="spec_artifact_sha256",
        )
        semantic_spec_hash = _require_hash(
            entry.get("spec_sha256"), field="spec_sha256"
        )
        review_artifact_hash = _require_hash(
            entry.get("review_artifact_sha256"),
            field="review_artifact_sha256",
        )
        spec_path = _safe_file(root, entry.get("spec_artifact"), label="spec")
        review_path = _safe_file(root, entry.get("review_artifact"), label="review")
        if file_sha256(spec_path) != spec_artifact_hash:
            raise ReviewedVQAQuerySpecError("spec artifact hash mismatch")
        if file_sha256(review_path) != review_artifact_hash:
            raise ReviewedVQAQuerySpecError("review artifact hash mismatch")
        spec = validate_vqa_query_spec(_read_json(spec_path, label="VQAQuerySpec"))
        actual_semantic_hash = canonical_sha256(spec)
        if actual_semantic_hash != semantic_spec_hash:
            raise ReviewedVQAQuerySpecError("semantic spec hash mismatch")
        if entry.get("spec_id") != spec["spec_id"]:
            raise ReviewedVQAQuerySpecError("registry spec identity mismatch")
        review = validate_vqa_query_review(
            _read_json(review_path, label="VQAQuerySpec review"),
            spec_sha256=actual_semantic_hash,
        )
        if spec["spec_id"] in seen_spec_ids:
            raise ReviewedVQAQuerySpecError("registry contains duplicate spec_id")
        seen_spec_ids.add(spec["spec_id"])
        loaded.append(
            {
                "spec": spec,
                "review": review,
                "spec_sha256": actual_semantic_hash,
                "spec_artifact": str(spec_path),
                "review_artifact": str(review_path),
            }
        )
    return loaded


def match_reviewed_vqa_query_spec(
    entries: list[dict[str, Any]],
    *,
    task_name: str | None,
    template_id: str | None,
    sub_aspect: str | None = None,
    tool_metric: str | None = None,
) -> dict[str, Any] | None:
    """Return the unique exact-context match, or ``None`` on a safe miss.

    Non-empty selector groups are conjunctive.  For example, a clutter spec
    that also names ``official_check_success`` cannot match a clean rollout
    merely because both conditions use that generic metric.
    """

    matches: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != LOADED_ENTRY_KEYS:
            raise ReviewedVQAQuerySpecError("loaded registry entry fields are invalid")
        spec = validate_vqa_query_spec(entry.get("spec"))
        spec_hash = canonical_sha256(spec)
        if entry.get("spec_sha256") != spec_hash:
            raise ReviewedVQAQuerySpecError(
                "loaded registry entry semantic hash mismatch"
            )
        validate_vqa_query_review(entry.get("review"), spec_sha256=spec_hash)
        if spec["task_name"] != task_name:
            continue
        template_matches = (
            not spec["template_ids"] or template_id in spec["template_ids"]
        )
        aspect_matches = not spec["sub_aspect_prefixes"] or bool(
            sub_aspect
            and any(
                sub_aspect == prefix or sub_aspect.startswith(prefix + ".")
                for prefix in spec["sub_aspect_prefixes"]
            )
        )
        metric_matches = not spec["tool_metrics"] or tool_metric in spec["tool_metrics"]
        if template_matches and aspect_matches and metric_matches:
            matches.append(deepcopy(dict(entry)))
    if len(matches) > 1:
        ids = sorted(match["spec"]["spec_id"] for match in matches)
        raise ReviewedVQAQuerySpecError(
            f"multiple reviewed VQAQuerySpecs match context: {ids}"
        )
    return matches[0] if matches else None
