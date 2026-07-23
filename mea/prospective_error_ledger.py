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
from typing import Any, Iterator, Mapping


LEDGER_PROTOCOL = "prospective_operation_error_ledger_v1"
FROZEN_CATEGORIES = ("plan_agent", "taskgen", "toolgen", "simulator")


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
