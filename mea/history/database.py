"""Rebuildable SQLite history of completed MEA evaluation plans.

The database is a runtime cache.  Each completed evaluation owns a canonical
``summary/history_record.json`` artifact, from which the SQLite index can be
rebuilt.  Historical outcomes are deliberately kept separate from current-run
evidence: retrieval returns compact planning context, never trajectory data.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from contextlib import closing
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping


class HistoryDatabaseError(RuntimeError):
    """Raised when the persistent history index cannot be used."""


class HistoryRecordError(ValueError):
    """Raised when evaluation artifacts cannot form a valid history record."""


class IncompleteEvaluationError(HistoryRecordError):
    """Raised when an evaluation has not reached completed lifecycle state."""


HISTORY_RECORD_SCHEMA_VERSION = 1
HISTORY_DATABASE_SCHEMA_VERSION = 1
LEGACY_SUB_ASPECT_TEMPLATES = {
    "object_appearance.color": "object_appearance.color_blue",
    "object_position": "object_position.official_random",
    "performance.pickup_to_contact_timing": (
        "performance.pickup_to_contact_timing"
    ),
}


def _canonical_json(value: Any, *, pretty: bool = False) -> str:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    try:
        return json.dumps(value, **options)
    except (TypeError, ValueError) as exc:
        raise HistoryRecordError(
            f"history value is not deterministic JSON data: {exc}"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HistoryRecordError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HistoryRecordError(f"{label} must be a JSON object: {path}")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HistoryRecordError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise HistoryRecordError(f"{field} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _relative_path(repo_root: Path, path: Path, field: str) -> str:
    try:
        return path.expanduser().resolve().relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise HistoryRecordError(
            f"{field} must stay inside repository root: {path}"
        ) from exc


def _artifact_path(
    repo_root: Path,
    evaluation_dir: Path,
    manifest: Mapping[str, Any],
    manifest_key: str,
    default_relative: str,
    *,
    required: bool,
) -> tuple[Path | None, str | None]:
    raw = manifest.get(manifest_key) or default_relative
    if not isinstance(raw, str) or not raw.strip():
        if required:
            raise HistoryRecordError(f"manifest.{manifest_key} is invalid")
        return None, None
    path = Path(raw)
    if not path.is_absolute():
        path = evaluation_dir / path
    path = path.expanduser().resolve()
    relative = _relative_path(repo_root, path, manifest_key)
    if not path.is_file():
        if required:
            raise HistoryRecordError(f"required artifact is missing: {path}")
        return None, None
    return path, relative


def _request_from_artifacts(
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    evaluation_dir: Path,
) -> str:
    for value in (manifest.get("user_request"), evidence.get("user_request")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    request_path = evaluation_dir / "request.json"
    if request_path.is_file():
        request = _read_json(request_path, "request")
        return _required_text(request.get("user_request"), "request.user_request")
    raise HistoryRecordError("completed evaluation is missing user_request")


def _executed_rounds(
    plan: Mapping[str, Any], evidence: Mapping[str, Any]
) -> list[dict[str, Any]]:
    planned = plan.get("rounds")
    if not isinstance(planned, list):
        raise HistoryRecordError("plan.rounds must be a list")
    evidence_plan = evidence.get("plan")
    if not isinstance(evidence_plan, Mapping):
        evidence_plan = {}
    evidence_rounds = evidence.get("rounds")
    if evidence_rounds is not None and not isinstance(evidence_rounds, list):
        raise HistoryRecordError("evidence.rounds must be a list when present")
    executed_count = evidence_plan.get("executed_rounds")
    if not isinstance(executed_count, int) or executed_count < 0:
        executed_count = (
            len(evidence_rounds)
            if isinstance(evidence_rounds, list)
            else len(planned)
        )
    if executed_count > len(planned):
        raise HistoryRecordError(
            "evidence executed_rounds exceeds the number of planned rounds"
        )
    if isinstance(evidence_rounds, list) and executed_count != len(evidence_rounds):
        raise HistoryRecordError(
            "evidence.plan.executed_rounds must equal len(evidence.rounds)"
        )

    result: list[dict[str, Any]] = []
    for index, value in enumerate(planned[:executed_count], start=1):
        if not isinstance(value, Mapping):
            raise HistoryRecordError(f"plan.rounds[{index - 1}] must be an object")
        planned_round_id = str(value.get("round_id") or f"round_{index}")
        if isinstance(evidence_rounds, list):
            evidence_round = evidence_rounds[index - 1]
            if not isinstance(evidence_round, Mapping):
                raise HistoryRecordError(
                    f"evidence.rounds[{index - 1}] must be an object"
                )
            evidence_round_id = evidence_round.get("round_id")
            if (
                evidence_round_id is not None
                and str(evidence_round_id) != planned_round_id
            ):
                raise HistoryRecordError(
                    "evidence and plan round_id mismatch at "
                    f"round index {index - 1}"
                )
        sub_aspect = value.get("sub_aspect")
        template_id = value.get("template_id") or LEGACY_SUB_ASPECT_TEMPLATES.get(
            sub_aspect
        )
        result.append(
            {
                "round_id": planned_round_id,
                "template_id": template_id,
                "sub_aspect": sub_aspect,
                "route": value.get("route"),
                "verification_of": value.get("verification_of"),
                "verification_trigger": value.get("verification_trigger"),
            }
        )
    return result


def _compact_round_decisions(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    decisions = plan.get("round_decisions", [])
    if not isinstance(decisions, list):
        raise HistoryRecordError("plan.round_decisions must be a list")
    compact: list[dict[str, Any]] = []
    for index, value in enumerate(decisions):
        if not isinstance(value, Mapping):
            raise HistoryRecordError(
                f"plan.round_decisions[{index}] must be an object"
            )
        assessment = value.get("evidence_assessment")
        if not isinstance(assessment, Mapping):
            assessment = {}
        compact.append(
            {
                "action": value.get("action"),
                "observation_summary": value.get("observation_summary"),
                "decision_reason": value.get("decision_reason"),
                "next_template_id": value.get("next_template_id"),
                "evidence_state": assessment.get("state"),
                "required_action": assessment.get("required_action"),
                "unresolved": assessment.get("unresolved"),
            }
        )
    return compact


def validate_history_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the durable, compact history contract."""

    if not isinstance(record, Mapping):
        raise HistoryRecordError("history record must be an object")
    normalized = dict(record)
    if normalized.get("schema_version") != HISTORY_RECORD_SCHEMA_VERSION:
        raise HistoryRecordError(
            f"history record schema_version must be {HISTORY_RECORD_SCHEMA_VERSION}"
        )
    _required_text(normalized.get("evaluation_id"), "evaluation_id")
    if normalized.get("lifecycle_status") != "completed":
        raise IncompleteEvaluationError(
            "only lifecycle_status=completed evaluations may enter history"
        )
    _required_text(normalized.get("task_name"), "task_name")
    _required_text(normalized.get("user_request"), "user_request")
    policy = normalized.get("policy")
    if not isinstance(policy, Mapping):
        raise HistoryRecordError("policy must be an object")
    _required_text(policy.get("name"), "policy.name")
    planning = normalized.get("planning")
    if not isinstance(planning, Mapping):
        raise HistoryRecordError("planning must be an object")
    _string_list(planning.get("requested_template_ids"), "planning.requested_template_ids")
    if not isinstance(planning.get("executed_rounds"), list):
        raise HistoryRecordError("planning.executed_rounds must be a list")
    artifacts = normalized.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise HistoryRecordError("artifacts must be an object")
    for key, value in artifacts.items():
        if value is None:
            continue
        if not isinstance(value, str) or Path(value).is_absolute():
            raise HistoryRecordError(
                f"artifacts.{key} must be a repository-relative path or null"
            )
    _canonical_json(normalized)
    return normalized


def build_history_record(
    repo_root: str | Path, evaluation_dir: str | Path
) -> dict[str, Any]:
    """Build one canonical record from a completed evaluation directory."""

    root = Path(repo_root).expanduser().resolve()
    directory = Path(evaluation_dir).expanduser().resolve()
    _relative_path(root, directory, "evaluation_dir")
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path, "evaluation manifest")
    lifecycle_status = manifest.get("lifecycle_status")
    legacy_completion_inferred = bool(
        lifecycle_status is None
        and manifest.get("status") == "completed"
        and isinstance(manifest.get("execution_finished_at"), str)
        and manifest.get("execution_finished_at", "").strip()
    )
    if lifecycle_status != "completed" and not legacy_completion_inferred:
        raise IncompleteEvaluationError(
            "evaluation must declare lifecycle_status=completed, or satisfy "
            "the legacy status=completed plus execution_finished_at contract"
        )

    plan_path, plan_relative = _artifact_path(
        root,
        directory,
        manifest,
        "plan_path",
        "plan/evaluation_plan.json",
        required=True,
    )
    evidence_path, evidence_relative = _artifact_path(
        root,
        directory,
        manifest,
        "evidence_path",
        "summary/evidence_bundle.json",
        required=True,
    )
    report_path, report_relative = _artifact_path(
        root,
        directory,
        manifest,
        "report_path",
        "evaluation_report.md",
        required=False,
    )
    assert plan_path is not None and evidence_path is not None
    plan = _read_json(plan_path, "evaluation plan")
    evidence = _read_json(evidence_path, "evidence bundle")

    evaluation_id = _required_text(
        manifest.get("evaluation_id") or evidence.get("evaluation_id"),
        "evaluation_id",
    )
    if directory.name != evaluation_id:
        raise HistoryRecordError(
            "evaluation_id must match its evaluation directory name"
        )
    for label, artifact_evaluation_id in (
        ("manifest", manifest.get("evaluation_id")),
        ("evidence", evidence.get("evaluation_id")),
    ):
        if (
            artifact_evaluation_id is not None
            and _required_text(
                artifact_evaluation_id, f"{label}.evaluation_id"
            )
            != evaluation_id
        ):
            raise HistoryRecordError(
                f"{label}.evaluation_id does not match evaluation directory"
            )
    task_name = _required_text(plan.get("task_name"), "plan.task_name")
    policy = plan.get("policy")
    if not isinstance(policy, Mapping):
        raise HistoryRecordError("plan.policy must be an object")
    policy_name = _required_text(policy.get("name"), "plan.policy.name")
    requested = _string_list(
        plan.get("requested_template_ids"), "plan.requested_template_ids"
    )
    executed = _executed_rounds(plan, evidence)
    legacy_template_ids_inferred = False
    if not requested:
        requested = list(
            dict.fromkeys(
                value["template_id"]
                for value in executed
                if value.get("template_id")
            )
        )
        legacy_template_ids_inferred = bool(requested)
    observations = evidence.get("observations")
    if not isinstance(observations, Mapping):
        observations = {}

    record: dict[str, Any] = {
        "schema_version": HISTORY_RECORD_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "lifecycle_status": "completed",
        "created_at": manifest.get("created_at"),
        "completed_at": manifest.get("execution_finished_at"),
        "task_name": task_name,
        "policy": {
            "name": policy_name,
            "checkpoint_setting": policy.get("checkpoint_setting"),
            "expert_data_num": policy.get("expert_data_num"),
            "language_conditioned": policy.get("language_conditioned"),
        },
        "user_request": _request_from_artifacts(
            manifest, evidence, directory
        ),
        "planning": {
            "evaluation_goal": plan.get("evaluation_goal"),
            "requested_template_ids": requested,
            "first_template_id": (
                executed[0].get("template_id") if executed else None
            ),
            "executed_rounds": executed,
            "round_decisions": _compact_round_decisions(plan),
            "planning_state": plan.get("planning_state"),
        },
        "outcome": {
            "status": manifest.get("status"),
            "pipeline_passed": observations.get("pipeline_passed"),
            "evidence_conflict": bool(
                observations.get("execution_vqa_conflict", False)
            ),
        },
        "compatibility": {
            "base_commit": manifest.get("base_commit"),
            "evidence_schema_version": evidence.get("schema_version"),
            "legacy_completion_inferred": legacy_completion_inferred,
            "legacy_template_ids_inferred": legacy_template_ids_inferred,
            "manifest_lifecycle_status": lifecycle_status,
        },
        "artifacts": {
            "manifest": _relative_path(root, manifest_path, "manifest"),
            "plan": plan_relative,
            "evidence": evidence_relative,
            "report": report_relative if report_path is not None else None,
        },
    }
    return validate_history_record(record)


def write_history_record(
    evaluation_dir: str | Path, record: Mapping[str, Any]
) -> Path:
    """Write the canonical rebuild source for one completed evaluation."""

    normalized = validate_history_record(record)
    path = Path(evaluation_dir).expanduser().resolve() / "summary/history_record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(normalized, pretty=True) + "\n", encoding="utf-8")
    return path


def read_history_record(
    evaluation_dir: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Read the canonical durable record owned by one evaluation directory."""

    directory = Path(evaluation_dir).expanduser().resolve()
    path = directory / "summary/history_record.json"
    normalized = validate_history_record(_read_json(path, "history record"))
    if normalized["evaluation_id"] != directory.name:
        raise HistoryRecordError(
            "history record evaluation_id must match its evaluation directory name"
        )
    return normalized, path


def _normalize_query(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(text.split())


def _character_ngrams(value: str, size: int = 2) -> set[str]:
    compact = "".join(character for character in value if not character.isspace())
    if not compact:
        return set()
    if len(compact) < size:
        return {compact}
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _query_similarity(left: str, right: str) -> float:
    left_normalized = _normalize_query(left)
    right_normalized = _normalize_query(right)
    sequence = SequenceMatcher(
        None, left_normalized, right_normalized, autojunk=False
    ).ratio()
    left_grams = _character_ngrams(left_normalized)
    right_grams = _character_ngrams(right_normalized)
    union = left_grams | right_grams
    jaccard = len(left_grams & right_grams) / len(union) if union else 1.0
    return round(0.8 * sequence + 0.2 * jaccard, 8)


class EvaluationHistoryDB:
    """SQLite index over canonical completed-evaluation history records."""

    def __init__(self, database_path: str | Path, *, repo_root: str | Path):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        try:
            with closing(self._connect()) as connection, connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS history_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS evaluations (
                        evaluation_id TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL,
                        completed_at TEXT,
                        task_name TEXT NOT NULL,
                        policy_name TEXT NOT NULL,
                        checkpoint_setting TEXT,
                        user_query TEXT NOT NULL,
                        normalized_query TEXT NOT NULL,
                        record_sha256 TEXT NOT NULL,
                        record_json TEXT NOT NULL,
                        indexed_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS evaluations_task_idx
                        ON evaluations(task_name);
                    CREATE INDEX IF NOT EXISTS evaluations_task_policy_idx
                        ON evaluations(task_name, policy_name);
                    """
                )
                connection.execute(
                    "INSERT INTO history_meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(HISTORY_DATABASE_SCHEMA_VERSION),),
                )
        except sqlite3.Error as exc:
            raise HistoryDatabaseError(
                f"cannot initialize history database {self.database_path}: {exc}"
            ) from exc

    def clear(self) -> None:
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("DELETE FROM evaluations")
        except sqlite3.Error as exc:
            raise HistoryDatabaseError(f"cannot clear history database: {exc}") from exc

    def upsert_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_history_record(record)
        encoded = _canonical_json(normalized)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        policy = normalized["policy"]
        indexed_at = datetime.now().astimezone().isoformat()
        try:
            with closing(self._connect()) as connection, connection:
                previous = connection.execute(
                    "SELECT record_sha256 FROM evaluations WHERE evaluation_id = ?",
                    (normalized["evaluation_id"],),
                ).fetchone()
                if previous is None:
                    action = "inserted"
                elif previous["record_sha256"] == digest:
                    action = "unchanged"
                else:
                    action = "updated"
                connection.execute(
                    """
                    INSERT INTO evaluations(
                        evaluation_id, schema_version, completed_at, task_name,
                        policy_name, checkpoint_setting, user_query,
                        normalized_query, record_sha256, record_json, indexed_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(evaluation_id) DO UPDATE SET
                        schema_version=excluded.schema_version,
                        completed_at=excluded.completed_at,
                        task_name=excluded.task_name,
                        policy_name=excluded.policy_name,
                        checkpoint_setting=excluded.checkpoint_setting,
                        user_query=excluded.user_query,
                        normalized_query=excluded.normalized_query,
                        record_sha256=excluded.record_sha256,
                        record_json=excluded.record_json,
                        indexed_at=CASE
                            WHEN evaluations.record_sha256 = excluded.record_sha256
                            THEN evaluations.indexed_at
                            ELSE excluded.indexed_at
                        END
                    """,
                    (
                        normalized["evaluation_id"],
                        normalized["schema_version"],
                        normalized.get("completed_at"),
                        normalized["task_name"],
                        policy["name"],
                        policy.get("checkpoint_setting"),
                        normalized["user_request"],
                        _normalize_query(normalized["user_request"]),
                        digest,
                        encoded,
                        indexed_at,
                    ),
                )
        except sqlite3.Error as exc:
            raise HistoryDatabaseError(f"cannot upsert history record: {exc}") from exc
        return {
            "evaluation_id": normalized["evaluation_id"],
            "action": action,
            "record_sha256": digest,
        }

    def index_evaluation_dir(self, evaluation_dir: str | Path) -> dict[str, Any]:
        directory = Path(evaluation_dir).expanduser().resolve()
        record = build_history_record(self.repo_root, directory)
        record_path = write_history_record(directory, record)
        result = self.upsert_record(record)
        result["source"] = "generated_from_evaluation_artifacts"
        result["history_record"] = _relative_path(
            self.repo_root, record_path, "history_record"
        )
        return result

    def count(self) -> int:
        try:
            with closing(self._connect()) as connection, connection:
                row = connection.execute("SELECT COUNT(*) AS count FROM evaluations").fetchone()
        except sqlite3.Error as exc:
            raise HistoryDatabaseError(f"cannot count history records: {exc}") from exc
        return int(row["count"])

    def retrieve_similar(
        self,
        user_query: str,
        *,
        task_name: str,
        policy_name: str | None = None,
        checkpoint_setting: str | None = None,
        limit: int = 3,
        exclude_evaluation_id: str | None = None,
        min_similarity: float = 0.0,
    ) -> dict[str, Any]:
        """Retrieve deterministic, compact planning examples for one task.

        ``task_name`` is always an SQL filter.  Policy is intentionally not a
        filter so that cross-policy planning remains comparable; every result
        preserves its policy label and compatibility flags.
        """

        query = _required_text(user_query, "user_query")
        task = _required_text(task_name, "task_name")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not isinstance(min_similarity, (int, float)) or not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be in [0, 1]")
        sql = "SELECT * FROM evaluations WHERE task_name = ?"
        parameters: list[Any] = [task]
        if exclude_evaluation_id:
            sql += " AND evaluation_id != ?"
            parameters.append(exclude_evaluation_id)
        try:
            with closing(self._connect()) as connection, connection:
                rows = connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            raise HistoryDatabaseError(f"cannot query evaluation history: {exc}") from exc

        candidates: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        for row in rows:
            try:
                record = validate_history_record(json.loads(row["record_json"]))
            except (json.JSONDecodeError, HistoryRecordError) as exc:
                issues.append(
                    {
                        "evaluation_id": row["evaluation_id"],
                        "reason": f"invalid cached record: {exc}",
                    }
                )
                continue
            score = _query_similarity(query, record["user_request"])
            if score < float(min_similarity):
                continue
            policy = record["policy"]
            candidates.append(
                {
                    "evaluation_id": record["evaluation_id"],
                    "similarity": score,
                    "user_request": record["user_request"],
                    "task_name": record["task_name"],
                    "policy": policy,
                    "compatibility": {
                        "same_policy": (
                            policy_name is None or policy.get("name") == policy_name
                        ),
                        "same_checkpoint": (
                            checkpoint_setting is None
                            or policy.get("checkpoint_setting") == checkpoint_setting
                        ),
                        **record.get("compatibility", {}),
                    },
                    "planning": record["planning"],
                    "outcome": record.get("outcome", {}),
                    "artifacts": record["artifacts"],
                }
            )
        candidates.sort(
            key=lambda item: (
                -item["similarity"],
                not item["compatibility"]["same_policy"],
                not item["compatibility"]["same_checkpoint"],
                item["evaluation_id"],
            )
        )
        selected = candidates[:limit]
        return {
            "schema_version": 1,
            "query": query,
            "task_name": task,
            "policy_name": policy_name,
            "checkpoint_setting": checkpoint_setting,
            "selection_policy": {
                "task_filter": "exact",
                "policy_filter": False,
                "similarity": "0.8_sequence_matcher_plus_0.2_character_bigram_jaccard",
                "tie_break": "same_policy_then_same_checkpoint_then_evaluation_id",
            },
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "candidates": selected,
            "issues": sorted(issues, key=lambda item: item["evaluation_id"]),
        }

    def retrieve_similar_global(
        self,
        user_query: str,
        *,
        allowed_task_names: Iterable[str],
        policy_name: str | None = None,
        checkpoint_setting: str | None = None,
        limit: int = 3,
        exclude_evaluation_id: str | None = None,
        min_similarity: float = 0.0,
    ) -> dict[str, Any]:
        """Retrieve completed planning priors across a trusted task allowlist.

        The allowlist comes from the global evaluation catalog; arbitrary task
        names never enter routing context.  Each task contributes at most the
        global limit because an item outside a task's top-k cannot enter the
        global top-k.
        """

        query = _required_text(user_query, "user_query")
        tasks = sorted(
            {
                _required_text(task_name, "allowed_task_name")
                for task_name in allowed_task_names
            }
        )
        if not tasks:
            raise ValueError("allowed_task_names must not be empty")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

        task_results = [
            self.retrieve_similar(
                query,
                task_name=task_name,
                policy_name=policy_name,
                checkpoint_setting=checkpoint_setting,
                limit=limit,
                exclude_evaluation_id=exclude_evaluation_id,
                min_similarity=min_similarity,
            )
            for task_name in tasks
        ]
        candidates = [
            candidate
            for result in task_results
            for candidate in result.get("candidates", [])
        ]
        candidates.sort(
            key=lambda item: (
                -item["similarity"],
                not item["compatibility"]["same_policy"],
                not item["compatibility"]["same_checkpoint"],
                item["evaluation_id"],
            )
        )
        issues = sorted(
            (
                issue
                for result in task_results
                for issue in result.get("issues", [])
            ),
            key=lambda item: item["evaluation_id"],
        )
        return {
            "schema_version": 1,
            "query": query,
            "allowed_task_names": tasks,
            "policy_name": policy_name,
            "checkpoint_setting": checkpoint_setting,
            "selection_policy": {
                "task_filter": "trusted_allowlist",
                "policy_filter": False,
                "similarity": "0.8_sequence_matcher_plus_0.2_character_bigram_jaccard",
                "tie_break": "same_policy_then_same_checkpoint_then_evaluation_id",
            },
            "candidate_count": sum(
                int(result.get("candidate_count", 0))
                for result in task_results
            ),
            "selected_count": min(len(candidates), limit),
            "candidates": candidates[:limit],
            "issues": issues,
        }

    def rebuild(
        self,
        evaluation_root: str | Path,
        *,
        reset: bool = False,
    ) -> dict[str, Any]:
        """Scan evaluation directories; one corrupt run never blocks others."""

        root = Path(evaluation_root).expanduser().resolve()
        _relative_path(self.repo_root, root, "evaluation_root")
        if reset:
            self.clear()
        counts = {"inserted": 0, "updated": 0, "unchanged": 0, "skipped": 0}
        indexed: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        directories: Iterable[Path] = sorted(
            (path for path in root.glob("eval_*") if path.is_dir()),
            key=lambda path: path.name,
        )
        for directory in directories:
            try:
                canonical_path = directory / "summary/history_record.json"
                if canonical_path.is_file():
                    record, record_path = read_history_record(directory)
                    result = self.upsert_record(record)
                    result["source"] = "canonical_history_record"
                    result["history_record"] = _relative_path(
                        self.repo_root, record_path, "history_record"
                    )
                else:
                    result = self.index_evaluation_dir(directory)
            except IncompleteEvaluationError as exc:
                counts["skipped"] += 1
                issues.append(
                    {
                        "evaluation_dir": _relative_path(
                            self.repo_root, directory, "evaluation_dir"
                        ),
                        "kind": "incomplete",
                        "reason": str(exc),
                    }
                )
                continue
            except (HistoryRecordError, HistoryDatabaseError, OSError) as exc:
                counts["skipped"] += 1
                issues.append(
                    {
                        "evaluation_dir": _relative_path(
                            self.repo_root, directory, "evaluation_dir"
                        ),
                        "kind": "invalid",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            counts[result["action"]] += 1
            indexed.append(result)
        return {
            "schema_version": 1,
            "database": _relative_path(
                self.repo_root, self.database_path, "database"
            ),
            "evaluation_root": _relative_path(
                self.repo_root, root, "evaluation_root"
            ),
            "reset": reset,
            "counts": counts,
            "database_record_count": self.count(),
            "indexed": indexed,
            "issues": issues,
        }


# Concise alias for callers that prefer the full word.
EvaluationHistoryDatabase = EvaluationHistoryDB
