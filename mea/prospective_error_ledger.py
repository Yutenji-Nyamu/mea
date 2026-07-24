"""Prospective, append-only operation/error ledger.

The denominator is frozen as operation starts.  A context manager lets future
runs write start and terminal events without remembering two separate calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


LEDGER_PROTOCOL = "prospective_operation_error_ledger_v1"
FROZEN_CATEGORIES = ("plan_agent", "taskgen", "toolgen", "simulator")
PAPER_ERROR_PROTOCOL = "prospective_paper_defined_error_study_v2"
PAPER_ERROR_CATEGORIES = (
    "plan_agent",
    "taskgen",
    "toolgen",
    "simulator",
    "other",
)
PAPER_ERROR_TERMINAL_STATUSES = (
    "passed",
    "paper_defined_error",
    "infrastructure_error",
    "aborted_upstream",
)


class ProspectiveLedgerError(ValueError):
    """Raised when ledger metadata or events violate the frozen protocol."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or not all(
        ch.isalnum() or ch in "._-" for ch in value
    ):
        raise ProspectiveLedgerError(f"{field} must be an identifier")
    return value


def initialize_ledger(directory: str | Path, *, study_id: str, frozen_at_utc: str | None = None) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "protocol": LEDGER_PROTOCOL,
        "study_id": _identifier(study_id, field="study_id"),
        "frozen_at_utc": frozen_at_utc or _now(),
        "denominator_unit": "unique_operation_started_before_outcome_known",
        "error_numerator_unit": "unique_operation_with_terminal_error",
        "categories": list(FROZEN_CATEGORIES),
        "terminal_statuses": ["completed", "error"],
        "paper_fig6_eligible": False,
    }
    metadata["metadata_sha256"] = _canonical_sha256(metadata)
    (root / "ledger_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "events.jsonl").touch(exist_ok=False)
    return metadata


def _load_metadata(directory: Path) -> dict[str, Any]:
    try:
        value = json.loads((directory / "ledger_meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveLedgerError(f"cannot read ledger metadata: {exc}") from exc
    if value.get("schema_version") != 1 or value.get("protocol") != LEDGER_PROTOCOL:
        raise ProspectiveLedgerError("unsupported ledger metadata")
    supplied = value.get("metadata_sha256")
    body = deepcopy(value)
    body.pop("metadata_sha256", None)
    if supplied != _canonical_sha256(body):
        raise ProspectiveLedgerError("ledger metadata hash mismatch")
    if value.get("categories") != list(FROZEN_CATEGORIES):
        raise ProspectiveLedgerError("ledger categories are not frozen")
    return value


class ProspectiveOperationLedger:
    """Append operation events and validate the frozen denominator."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.metadata = _load_metadata(self.directory)
        self.events_path = self.directory / "events.jsonl"

    def append(
        self,
        *,
        operation_id: str,
        run_id: str,
        category: str,
        status: str,
        evidence_ref: str | None = None,
        error_class: str | None = None,
        timestamp_utc: str | None = None,
    ) -> dict[str, Any]:
        operation_id = _identifier(operation_id, field="operation_id")
        run_id = _identifier(run_id, field="run_id")
        if category not in FROZEN_CATEGORIES:
            raise ProspectiveLedgerError(f"category must be one of {FROZEN_CATEGORIES}")
        if status not in {"started", "completed", "error"}:
            raise ProspectiveLedgerError("status must be started, completed, or error")
        if status == "error":
            error_class = _identifier(error_class, field="error_class")
        elif error_class is not None:
            raise ProspectiveLedgerError("error_class is allowed only for error events")
        event = {
            "schema_version": 1,
            "protocol": LEDGER_PROTOCOL,
            "event_id": uuid.uuid4().hex,
            "metadata_sha256": self.metadata["metadata_sha256"],
            "timestamp_utc": timestamp_utc or _now(),
            "operation_id": operation_id,
            "run_id": run_id,
            "category": category,
            "status": status,
            "evidence_ref": evidence_ref,
            "error_class": error_class,
        }
        payload = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        descriptor = os.open(self.events_path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return event

    @contextmanager
    def operation(
        self,
        *,
        operation_id: str,
        run_id: str,
        category: str,
        evidence_ref: str | None = None,
    ) -> Iterator[None]:
        self.append(
            operation_id=operation_id,
            run_id=run_id,
            category=category,
            status="started",
            evidence_ref=evidence_ref,
        )
        try:
            yield
        except BaseException as exc:
            self.append(
                operation_id=operation_id,
                run_id=run_id,
                category=category,
                status="error",
                evidence_ref=evidence_ref,
                error_class=type(exc).__name__,
            )
            raise
        else:
            self.append(
                operation_id=operation_id,
                run_id=run_id,
                category=category,
                status="completed",
                evidence_ref=evidence_ref,
            )

    def summarize(self) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        try:
            lines = self.events_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ProspectiveLedgerError(f"cannot read ledger events: {exc}") from exc
        for line_number, line in enumerate(lines, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProspectiveLedgerError(f"invalid JSONL at line {line_number}") from exc
            if event.get("metadata_sha256") != self.metadata["metadata_sha256"]:
                raise ProspectiveLedgerError("event metadata hash mismatch")
            events.append(event)
        by_operation: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            operation_id = _identifier(event.get("operation_id"), field="event.operation_id")
            by_operation.setdefault(operation_id, []).append(event)
        category_rows = {
            category: {"started": 0, "completed": 0, "errors": 0, "in_flight": 0}
            for category in FROZEN_CATEGORIES
        }
        for operation_id, rows in by_operation.items():
            starts = [row for row in rows if row.get("status") == "started"]
            terminals = [row for row in rows if row.get("status") in {"completed", "error"}]
            if len(starts) != 1 or len(terminals) > 1:
                raise ProspectiveLedgerError(f"operation {operation_id} has invalid lifecycle")
            category = starts[0].get("category")
            if category not in category_rows or any(row.get("category") != category for row in rows):
                raise ProspectiveLedgerError(f"operation {operation_id} changes category")
            category_rows[category]["started"] += 1
            if not terminals:
                category_rows[category]["in_flight"] += 1
            elif terminals[0]["status"] == "completed":
                category_rows[category]["completed"] += 1
            else:
                category_rows[category]["errors"] += 1
        denominator = sum(row["started"] for row in category_rows.values())
        numerator = sum(row["errors"] for row in category_rows.values())
        return {
            "schema_version": 1,
            "protocol": f"{LEDGER_PROTOCOL}_summary",
            "study_id": self.metadata["study_id"],
            "metadata_sha256": self.metadata["metadata_sha256"],
            "denominator_operation_starts": denominator,
            "numerator_terminal_errors": numerator,
            "prospective_error_rate": numerator / denominator if denominator else None,
            "categories": category_rows,
            "paper_fig6_eligible": False,
            "limitations": [
                "The denominator is prospective but does not yet match the paper run count.",
                "In-flight operations remain in the denominator and outside the error numerator.",
            ],
        }


def build_paper_error_study_v2(
    *,
    study_id: str,
    frozen_at_utc: str,
    operations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze a semantic-error denominator before any listed operation runs."""

    roster: list[dict[str, Any]] = []
    for index, raw in enumerate(operations):
        if not isinstance(raw, Mapping):
            raise ProspectiveLedgerError(f"operations[{index}] must be an object")
        expected = {
            "operation_id",
            "run_id",
            "category",
            "paper_error_definition_id",
        }
        if set(raw) != expected:
            raise ProspectiveLedgerError(
                f"operations[{index}] fields must be exactly {sorted(expected)}"
            )
        category = raw.get("category")
        if category not in PAPER_ERROR_CATEGORIES:
            raise ProspectiveLedgerError(
                f"operations[{index}].category must be one of "
                f"{PAPER_ERROR_CATEGORIES}"
            )
        roster.append(
            {
                "operation_id": _identifier(
                    raw.get("operation_id"),
                    field=f"operations[{index}].operation_id",
                ),
                "run_id": _identifier(
                    raw.get("run_id"), field=f"operations[{index}].run_id"
                ),
                "category": category,
                "paper_error_definition_id": _identifier(
                    raw.get("paper_error_definition_id"),
                    field=(
                        f"operations[{index}].paper_error_definition_id"
                    ),
                ),
            }
        )
    if not roster:
        raise ProspectiveLedgerError("operations must freeze a non-empty roster")
    operation_ids = [row["operation_id"] for row in roster]
    if len(operation_ids) != len(set(operation_ids)):
        raise ProspectiveLedgerError("operation_id must be unique in the roster")
    body = {
        "schema_version": 2,
        "protocol": PAPER_ERROR_PROTOCOL,
        "study_id": _identifier(study_id, field="study_id"),
        "frozen_at_utc": _text_timestamp(frozen_at_utc),
        "denominator_contract": (
            "all frozen operations must terminate as passed or "
            "paper_defined_error before an error rate is reported"
        ),
        "paper_error_contract": (
            "only a violation of the operation's preregistered semantic "
            "definition enters the numerator; exceptions and infrastructure "
            "failures are reported separately"
        ),
        "categories": list(PAPER_ERROR_CATEGORIES),
        "terminal_statuses": list(PAPER_ERROR_TERMINAL_STATUSES),
        "operations": roster,
        "paper_fig6_eligible": False,
    }
    body["study_sha256"] = _canonical_sha256(body)
    return body


def _text_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectiveLedgerError("frozen_at_utc must be non-empty text")
    text = value.strip()
    if not text.endswith("Z") or "T" not in text:
        raise ProspectiveLedgerError("frozen_at_utc must be an RFC3339 UTC timestamp")
    return text


def _validate_paper_error_study_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProspectiveLedgerError("study must be an object")
    supplied = value.get("study_sha256")
    body = deepcopy(dict(value))
    body.pop("study_sha256", None)
    if supplied != _canonical_sha256(body):
        raise ProspectiveLedgerError("paper error study hash mismatch")
    rebuilt = build_paper_error_study_v2(
        study_id=body.get("study_id"),
        frozen_at_utc=body.get("frozen_at_utc"),
        operations=body.get("operations") or [],
    )
    if rebuilt != value:
        raise ProspectiveLedgerError("paper error study contract was modified")
    return deepcopy(dict(value))


def summarize_paper_error_study_v2(
    study: Mapping[str, Any],
    latest_statuses: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize a frozen roster without turning exceptions into paper errors."""

    normalized = _validate_paper_error_study_v2(study)
    roster = {
        row["operation_id"]: row for row in normalized["operations"]
    }
    observed: dict[str, dict[str, Any]] = {}
    expected_fields = {
        "operation_id",
        "status",
        "evidence_ref",
        "observed_error_definition_id",
    }
    for index, raw in enumerate(latest_statuses):
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ProspectiveLedgerError(
                f"latest_statuses[{index}] fields must be exactly "
                f"{sorted(expected_fields)}"
            )
        operation_id = _identifier(
            raw.get("operation_id"),
            field=f"latest_statuses[{index}].operation_id",
        )
        if operation_id not in roster:
            raise ProspectiveLedgerError(
                f"status is outside the frozen roster: {operation_id}"
            )
        if operation_id in observed:
            raise ProspectiveLedgerError(
                f"latest status is duplicated: {operation_id}"
            )
        status = raw.get("status")
        if status not in {"started", *PAPER_ERROR_TERMINAL_STATUSES}:
            raise ProspectiveLedgerError(
                f"latest_statuses[{index}].status is invalid"
            )
        definition = raw.get("observed_error_definition_id")
        if status == "paper_defined_error":
            definition = _identifier(
                definition,
                field=(
                    f"latest_statuses[{index}]."
                    "observed_error_definition_id"
                ),
            )
            if definition != roster[operation_id]["paper_error_definition_id"]:
                raise ProspectiveLedgerError(
                    "paper-defined error does not match the frozen definition"
                )
        elif definition is not None:
            raise ProspectiveLedgerError(
                "only paper_defined_error may name an observed error definition"
            )
        evidence_ref = raw.get("evidence_ref")
        if evidence_ref is not None and (
            not isinstance(evidence_ref, str) or not evidence_ref.strip()
        ):
            raise ProspectiveLedgerError("evidence_ref must be non-empty or null")
        observed[operation_id] = {
            "operation_id": operation_id,
            "status": status,
            "evidence_ref": evidence_ref,
            "observed_error_definition_id": definition,
        }

    category_rows = {
        category: {
            "planned": 0,
            "not_started": 0,
            "in_flight": 0,
            "passed": 0,
            "paper_defined_errors": 0,
            "infrastructure_errors": 0,
            "aborted_upstream": 0,
        }
        for category in PAPER_ERROR_CATEGORIES
    }
    for operation_id, frozen in roster.items():
        row = category_rows[frozen["category"]]
        row["planned"] += 1
        event = observed.get(operation_id)
        if event is None:
            row["not_started"] += 1
        elif event["status"] == "started":
            row["in_flight"] += 1
        elif event["status"] == "passed":
            row["passed"] += 1
        elif event["status"] == "paper_defined_error":
            row["paper_defined_errors"] += 1
        elif event["status"] == "infrastructure_error":
            row["infrastructure_errors"] += 1
        else:
            row["aborted_upstream"] += 1

    totals = {
        key: sum(row[key] for row in category_rows.values())
        for key in next(iter(category_rows.values()))
    }
    semantic_terminal = totals["passed"] + totals["paper_defined_errors"]
    evidence_complete = semantic_terminal == totals["planned"]
    numerator = totals["paper_defined_errors"]
    distribution = {
        category: (
            row["paper_defined_errors"] / numerator if numerator else None
        )
        for category, row in category_rows.items()
    }
    return {
        "schema_version": 2,
        "protocol": f"{PAPER_ERROR_PROTOCOL}_summary",
        "study_id": normalized["study_id"],
        "study_sha256": normalized["study_sha256"],
        "frozen_denominator_operations": totals["planned"],
        "semantic_terminal_operations": semantic_terminal,
        "paper_defined_error_numerator": numerator,
        "prospective_error_rate": (
            numerator / totals["planned"] if evidence_complete else None
        ),
        "evidence_complete": evidence_complete,
        "totals": totals,
        "categories": category_rows,
        "paper_error_distribution": distribution,
        "paper_fig6_eligible": False,
        "limitations": [
            "This pilot uses a small project-defined operation roster.",
            "Infrastructure errors and upstream aborts never enter the semantic error numerator.",
            "A rate is withheld until every frozen operation has a semantic terminal status.",
        ],
    }


__all__ = [
    "FROZEN_CATEGORIES",
    "LEDGER_PROTOCOL",
    "PAPER_ERROR_CATEGORIES",
    "PAPER_ERROR_PROTOCOL",
    "PAPER_ERROR_TERMINAL_STATUSES",
    "ProspectiveLedgerError",
    "ProspectiveOperationLedger",
    "build_paper_error_study_v2",
    "initialize_ledger",
    "summarize_paper_error_study_v2",
]
