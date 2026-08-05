"""Reviewed persistent storage for Query-induced VQA questions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .query import (
    ExecutionVQAQueryError,
    validate_run_local_question_spec,
    vqa_need_semantic_key,
)
from .reviewed_registry import (
    REVIEWER_KINDS,
    ReviewedVQAQuerySpecError,
    _read_json,
    _safe_file,
    canonical_sha256,
    file_sha256,
)

GENERATED_QUESTION_SCOPE = "reviewed_persistent_generated_vqa_questions"
GENERATED_QUESTION_REVIEW_SCOPE = "persistent_generated_vqa_question_reuse"
GENERATED_QUESTION_REVIEW_CHECK_KEYS = {
    "proposal_alignment_reviewed",
    "question_observability_reviewed",
    "numeric_authority_reviewed",
    "execution_evidence_reviewed",
    "tests_reviewed",
}
GENERATED_QUESTION_INDEX_ENTRY_KEYS = {
    "registration_id",
    "semantic_key",
    "registration_artifact",
    "registration_artifact_sha256",
    "question_artifact",
    "question_artifact_sha256",
    "review_artifact",
    "review_artifact_sha256",
    "source_entry_artifact",
    "source_entry_artifact_sha256",
}

def _generated_registry_root(registry_dir: str | Path) -> Path:
    return Path(registry_dir).expanduser().resolve() / "generated_questions"


def _generated_source_entry(
    source_registry_dir: str | Path,
    semantic_key: str,
) -> dict[str, Any]:
    # Import lazily because ``open_question`` consumes this reviewed registry.
    from .open_question import load_run_local_vqa_questions

    matches = [
        entry
        for entry in load_run_local_vqa_questions(source_registry_dir)
        if entry["semantic_key"] == semantic_key
    ]
    if len(matches) != 1:
        raise ReviewedVQAQuerySpecError(
            "source run-local VQA question must have one exact semantic match"
        )
    return matches[0]


def build_generated_vqa_question_review_template(
    source_registry_dir: str | Path,
    semantic_key: str,
) -> dict[str, Any]:
    """Build a non-approved review pinned to one generated question."""

    return build_generated_vqa_question_review_template_from_entry(
        _generated_source_entry(source_registry_dir, semantic_key)
    )


def validate_generated_vqa_question_review(
    value: Any,
    *,
    source_entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an explicit exact-hash review before persistent admission."""

    template = build_generated_vqa_question_review_template_from_entry(
        source_entry
    )
    expected_keys = set(template)
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ReviewedVQAQuerySpecError(
            "generated VQA review fields are invalid"
        )
    if value.get("schema_version") != 1 or value.get("decision") != "approved":
        raise ReviewedVQAQuerySpecError(
            "generated VQA review decision must be approved"
        )
    if value.get("review_scope") != GENERATED_QUESTION_REVIEW_SCOPE:
        raise ReviewedVQAQuerySpecError(
            "generated VQA review_scope is invalid"
        )
    for field in (
        "semantic_key",
        "source_entry_sha256",
        "question_spec_sha256",
    ):
        if value.get(field) != template[field]:
            raise ReviewedVQAQuerySpecError(
                f"generated VQA review {field} does not match source"
            )
    reviewer = value.get("reviewer")
    if not isinstance(reviewer, Mapping) or set(reviewer) != {"id", "kind"}:
        raise ReviewedVQAQuerySpecError(
            "generated VQA reviewer must contain exactly id and kind"
        )
    if not isinstance(reviewer.get("id"), str) or not reviewer["id"].strip():
        raise ReviewedVQAQuerySpecError("generated VQA reviewer.id is empty")
    if reviewer.get("kind") not in REVIEWER_KINDS:
        raise ReviewedVQAQuerySpecError(
            "generated VQA reviewer.kind is invalid"
        )
    reviewed_at = value.get("reviewed_at")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise ReviewedVQAQuerySpecError(
            "generated VQA reviewed_at must be ISO-8601"
        )
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewedVQAQuerySpecError(
            "generated VQA reviewed_at must be ISO-8601"
        ) from exc
    if timestamp.tzinfo is None:
        raise ReviewedVQAQuerySpecError(
            "generated VQA reviewed_at must include a timezone"
        )
    checks = value.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != GENERATED_QUESTION_REVIEW_CHECK_KEYS
        or any(checks[key] is not True for key in checks)
    ):
        raise ReviewedVQAQuerySpecError(
            "generated VQA review checks were not all approved"
        )
    if not isinstance(value.get("notes"), str):
        raise ReviewedVQAQuerySpecError(
            "generated VQA review notes must be a string"
        )
    return json.loads(json.dumps(dict(value), ensure_ascii=False))


def build_generated_vqa_question_review_template_from_entry(
    source_entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact expected review fields for an already loaded entry."""

    if not isinstance(source_entry, Mapping):
        raise ReviewedVQAQuerySpecError(
            "generated VQA source entry must be an object"
        )
    task_name = source_entry.get("task_name")
    vqa_need = source_entry.get("vqa_need")
    if (
        not isinstance(task_name, str)
        or not task_name.strip()
        or not isinstance(vqa_need, str)
        or not vqa_need.strip()
    ):
        raise ReviewedVQAQuerySpecError(
            "generated VQA source task_name/vqa_need is invalid"
        )
    semantic_key = vqa_need_semantic_key(
        task_name=task_name,
        vqa_need=vqa_need,
    )
    if source_entry.get("semantic_key") != semantic_key:
        raise ReviewedVQAQuerySpecError(
            "generated VQA source semantic key is invalid"
        )
    try:
        question = validate_run_local_question_spec(
            source_entry.get("question_spec")
        )
    except ExecutionVQAQueryError as exc:
        raise ReviewedVQAQuerySpecError(
            f"generated VQA source question is invalid: {exc}"
        ) from exc
    canonical_source = json.loads(
        json.dumps(dict(source_entry), ensure_ascii=False)
    )
    canonical_source["question_spec"] = question
    return {
        "schema_version": 1,
        "decision": "pending",
        "review_scope": GENERATED_QUESTION_REVIEW_SCOPE,
        "reviewer": {"id": "", "kind": "development_agent_proxy"},
        "reviewed_at": None,
        "semantic_key": semantic_key,
        "source_entry_sha256": canonical_sha256(canonical_source),
        "question_spec_sha256": canonical_sha256(question),
        "checks": {
            key: False
            for key in sorted(GENERATED_QUESTION_REVIEW_CHECK_KEYS)
        },
        "notes": "",
    }


def _load_generated_question_index(root: Path) -> dict[str, Any]:
    path = root / "index.json"
    if not path.is_file():
        return {
            "schema_version": 1,
            "scope": GENERATED_QUESTION_SCOPE,
            "entries": [],
        }
    value = _read_json(path, label="reviewed generated VQA index")
    if (
        set(value) != {"schema_version", "scope", "entries"}
        or value.get("schema_version") != 1
        or value.get("scope") != GENERATED_QUESTION_SCOPE
        or not isinstance(value.get("entries"), list)
    ):
        raise ReviewedVQAQuerySpecError(
            "reviewed generated VQA index is invalid"
        )
    return value


def _write_generated_question_index(root: Path, value: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "index.json"
    temporary = root / "index.json.tmp"
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def install_reviewed_generated_vqa_question(
    source_registry_dir: str | Path,
    semantic_key: str,
    review: Mapping[str, Any] | str | Path,
    reviewed_registry_dir: str | Path,
) -> dict[str, Any]:
    """Install one explicitly reviewed generated question for later Queries."""

    source = _generated_source_entry(source_registry_dir, semantic_key)
    if isinstance(review, (str, Path)):
        review_value = _read_json(
            Path(review).expanduser().resolve(),
            label="generated VQA review",
        )
    else:
        review_value = dict(review)
    review_value = validate_generated_vqa_question_review(
        review_value,
        source_entry=source,
    )
    question = validate_run_local_question_spec(source["question_spec"])
    registration_id = "reviewed_vqa_" + canonical_sha256(
        {
            "semantic_key": semantic_key,
            "question_spec": question,
            "review": review_value,
        }
    )[:20]
    root = _generated_registry_root(reviewed_registry_dir)
    index = _load_generated_question_index(root)
    existing = next(
        (
            item
            for item in index["entries"]
            if item.get("registration_id") == registration_id
        ),
        None,
    )
    if existing is not None:
        return find_reviewed_generated_vqa_question(
            reviewed_registry_dir,
            task_name=source["task_name"],
            vqa_need=source["vqa_need"],
        ) or {}

    registration = {
        "schema_version": 1,
        "registration_id": registration_id,
        "scope": GENERATED_QUESTION_SCOPE,
        "status": "approved",
        "semantic_key": semantic_key,
        "task_name": source["task_name"],
        "vqa_need": source["vqa_need"],
        "question_spec_sha256": canonical_sha256(question),
        "source_entry_sha256": canonical_sha256(source),
        "review_sha256": canonical_sha256(review_value),
        "source_query": source["source_query"],
        "semantic_concern": source["semantic_concern"],
    }
    relative_root = Path("entries") / registration_id
    entry_dir = root / relative_root
    temporary_dir = root / "entries" / f"{registration_id}.tmp"
    if entry_dir.exists() or temporary_dir.exists():
        raise ReviewedVQAQuerySpecError(
            "unindexed reviewed generated VQA entry already exists"
        )
    temporary_dir.mkdir(parents=True)
    artifacts = {
        "registration": registration,
        "question": question,
        "review": review_value,
        "source_entry": source,
    }
    for name, value in artifacts.items():
        (temporary_dir / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    temporary_dir.replace(entry_dir)
    entry = {
        "registration_id": registration_id,
        "semantic_key": semantic_key,
        "registration_artifact": str(relative_root / "registration.json"),
        "registration_artifact_sha256": file_sha256(
            entry_dir / "registration.json"
        ),
        "question_artifact": str(relative_root / "question.json"),
        "question_artifact_sha256": file_sha256(entry_dir / "question.json"),
        "review_artifact": str(relative_root / "review.json"),
        "review_artifact_sha256": file_sha256(entry_dir / "review.json"),
        "source_entry_artifact": str(relative_root / "source_entry.json"),
        "source_entry_artifact_sha256": file_sha256(
            entry_dir / "source_entry.json"
        ),
    }
    index["entries"].append(entry)
    index["entries"].sort(key=lambda item: item["registration_id"])
    _write_generated_question_index(root, index)
    return find_reviewed_generated_vqa_question(
        reviewed_registry_dir,
        task_name=source["task_name"],
        vqa_need=source["vqa_need"],
    ) or {}


def load_reviewed_generated_vqa_questions(
    registry_dir: str | Path,
) -> list[dict[str, Any]]:
    """Load exact approved generated questions from the existing VQA registry."""

    root = _generated_registry_root(registry_dir)
    loaded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in _load_generated_question_index(root)["entries"]:
        if not isinstance(raw, Mapping) or set(raw) != GENERATED_QUESTION_INDEX_ENTRY_KEYS:
            raise ReviewedVQAQuerySpecError(
                "reviewed generated VQA index entry is invalid"
            )
        paths = {
            "registration": _safe_file(
                root, raw["registration_artifact"], label="VQA registration"
            ),
            "question": _safe_file(
                root, raw["question_artifact"], label="VQA question"
            ),
            "review": _safe_file(
                root, raw["review_artifact"], label="VQA review"
            ),
            "source_entry": _safe_file(
                root,
                raw["source_entry_artifact"],
                label="VQA source entry",
            ),
        }
        for name, path in paths.items():
            if file_sha256(path) != raw[f"{name}_artifact_sha256"]:
                raise ReviewedVQAQuerySpecError(
                    f"reviewed generated VQA {name} hash mismatch"
                )
        registration = _read_json(paths["registration"], label="VQA registration")
        try:
            question = validate_run_local_question_spec(
                _read_json(paths["question"], label="VQA question")
            )
        except ExecutionVQAQueryError as exc:
            raise ReviewedVQAQuerySpecError(
                f"reviewed generated VQA question is invalid: {exc}"
            ) from exc
        review = _read_json(paths["review"], label="VQA review")
        source_entry = _read_json(
            paths["source_entry"], label="VQA source entry"
        )
        semantic_key = registration.get("semantic_key")
        build_generated_vqa_question_review_template_from_entry(source_entry)
        if source_entry.get("question_spec") != question:
            raise ReviewedVQAQuerySpecError(
                "reviewed generated VQA source question is inconsistent"
            )
        review = validate_generated_vqa_question_review(
            review,
            source_entry=source_entry,
        )
        checks = {
            "registration_id": registration.get("registration_id")
            == raw.get("registration_id"),
            "scope": registration.get("scope") == GENERATED_QUESTION_SCOPE,
            "status": registration.get("status") == "approved",
            "semantic_key": semantic_key == raw.get("semantic_key"),
            "task_name": registration.get("task_name")
            == source_entry.get("task_name"),
            "vqa_need": registration.get("vqa_need")
            == source_entry.get("vqa_need"),
            "source_query": registration.get("source_query")
            == source_entry.get("source_query"),
            "semantic_concern": registration.get("semantic_concern")
            == source_entry.get("semantic_concern"),
            "question": registration.get("question_spec_sha256")
            == canonical_sha256(question),
            "source": registration.get("source_entry_sha256")
            == canonical_sha256(source_entry),
            "review": registration.get("review_sha256")
            == canonical_sha256(review),
        }
        if any(value is not True for value in checks.values()):
            raise ReviewedVQAQuerySpecError(
                "reviewed generated VQA registration is inconsistent"
            )
        if semantic_key in seen:
            raise ReviewedVQAQuerySpecError(
                "reviewed generated VQA registry has duplicate semantic keys"
            )
        seen.add(str(semantic_key))
        loaded.append(
            {
                "registration": registration,
                "question_spec": question,
                "review": review,
                "question_spec_sha256": canonical_sha256(question),
                "review_sha256": canonical_sha256(review),
                "artifacts": {key: str(path) for key, path in paths.items()},
            }
        )
    return loaded


def find_reviewed_generated_vqa_question(
    registry_dir: str | Path,
    *,
    task_name: str,
    vqa_need: str,
) -> dict[str, Any] | None:
    """Return the exact reviewed question contract for a later evaluation."""

    semantic_key = vqa_need_semantic_key(
        task_name=task_name,
        vqa_need=vqa_need,
    )
    matches = [
        item
        for item in load_reviewed_generated_vqa_questions(registry_dir)
        if item["registration"]["semantic_key"] == semantic_key
    ]
    if len(matches) > 1:
        raise ReviewedVQAQuerySpecError(
            "multiple reviewed generated VQA questions match one need"
        )
    return matches[0] if matches else None
