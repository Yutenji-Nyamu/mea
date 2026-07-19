"""Durable, secret-free call-start accounting for external runtimes.

The ledger is deliberately narrower than a request trace.  It records only
the identity needed to count provider calls and their transport retries.  It
never receives prompts, images, credentials, headers, or endpoint URLs.

When the two environment variables below are unset, provider use outside an
MEA evaluation keeps its existing behavior.  When either variable is set,
both must be valid and every transport attempt is appended and fsynced before
the external request starts.  A ledger failure therefore prevents the call
instead of silently under-counting it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


LEDGER_PATH_ENV = "MEA_RUNTIME_LEDGER_PATH"
LEDGER_CONTEXT_ENV = "MEA_RUNTIME_LEDGER_CONTEXT"

_SCHEMA_VERSION = 1
_CONTEXT_KEYS = {
    "schema_version",
    "evaluation_id",
    "logical_round_id",
    "round_attempt_index",
    "child_run_id",
}
_PROVIDER_EVENT_KEYS = _CONTEXT_KEYS | {
    "event_type",
    "recorded_at",
    "logical_call_id",
    "transport_attempt",
    "modality",
    "model",
}
_ACT_EVENT_KEYS = _CONTEXT_KEYS | {
    "event_type",
    "recorded_at",
    "act_batch_id",
    "task_name",
    "policy_name",
    "start_seed",
    "num_rollouts",
}
_PROVIDER_EVENT_TYPE = "provider_transport_started"
_ACT_EVENT_TYPE = "act_batch_started"
_MODALITIES = {"text", "vision"}
_LOGICAL_CALL_ID = re.compile(r"[0-9a-f]{32}")


class RuntimeLedgerError(RuntimeError):
    """Raised when call-start evidence cannot be written or validated."""


def _non_empty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeLedgerError(f"{field} must be a non-empty string")
    if "\x00" in value:
        raise RuntimeLedgerError(f"{field} must not contain NUL")
    return value


def validate_runtime_context(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return one exact, canonical round-attempt context."""

    if not isinstance(value, Mapping) or set(value) != _CONTEXT_KEYS:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise RuntimeLedgerError(
            f"runtime ledger context fields must be exactly "
            f"{sorted(_CONTEXT_KEYS)}; got {actual}"
        )
    if value.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeLedgerError(
            f"runtime ledger context schema_version must be {_SCHEMA_VERSION}"
        )
    attempt = value.get("round_attempt_index")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise RuntimeLedgerError(
            "runtime ledger round_attempt_index must be a positive integer"
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "evaluation_id": _non_empty_text(value.get("evaluation_id"), "evaluation_id"),
        "logical_round_id": _non_empty_text(
            value.get("logical_round_id"), "logical_round_id"
        ),
        "round_attempt_index": attempt,
        "child_run_id": _non_empty_text(value.get("child_run_id"), "child_run_id"),
    }


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeLedgerError(f"runtime ledger value is not canonical JSON: {exc}") from exc


def _context_from_environment() -> tuple[Path, dict[str, Any]] | None:
    raw_path = os.getenv(LEDGER_PATH_ENV)
    raw_context = os.getenv(LEDGER_CONTEXT_ENV)
    if raw_path is None and raw_context is None:
        return None
    if not raw_path or not raw_context:
        raise RuntimeLedgerError(
            f"{LEDGER_PATH_ENV} and {LEDGER_CONTEXT_ENV} must be set together"
        )
    try:
        decoded = json.loads(raw_context)
    except json.JSONDecodeError as exc:
        raise RuntimeLedgerError(
            f"{LEDGER_CONTEXT_ENV} is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise RuntimeLedgerError(f"{LEDGER_CONTEXT_ENV} must contain a JSON object")
    path = Path(raw_path).expanduser().resolve(strict=False)
    if path.exists() and (not path.is_file() or path.is_symlink()):
        raise RuntimeLedgerError(f"runtime ledger path must be a regular file: {path}")
    if not path.parent.is_dir():
        raise RuntimeLedgerError(
            f"runtime ledger parent directory does not exist: {path.parent}"
        )
    return path, validate_runtime_context(decoded)


@contextmanager
def runtime_ledger_context(
    path: str | Path, context: Mapping[str, Any]
) -> Iterator[Path]:
    """Bind a validated ledger to the current synchronous evaluation scope.

    The previous environment is restored even when the provider raises.  This
    helper does not truncate or otherwise rewrite an existing ledger.
    """

    validated = validate_runtime_context(context)
    resolved = Path(path).expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and (not resolved.is_file() or resolved.is_symlink()):
        raise RuntimeLedgerError(f"runtime ledger path must be a regular file: {resolved}")
    previous_path = os.environ.get(LEDGER_PATH_ENV)
    previous_context = os.environ.get(LEDGER_CONTEXT_ENV)
    os.environ[LEDGER_PATH_ENV] = str(resolved)
    os.environ[LEDGER_CONTEXT_ENV] = _canonical_json(validated)
    try:
        yield resolved
    finally:
        if previous_path is None:
            os.environ.pop(LEDGER_PATH_ENV, None)
        else:
            os.environ[LEDGER_PATH_ENV] = previous_path
        if previous_context is None:
            os.environ.pop(LEDGER_CONTEXT_ENV, None)
        else:
            os.environ[LEDGER_CONTEXT_ENV] = previous_context


def new_logical_call_id() -> str:
    """Return a non-secret identifier shared by retries of one provider call."""

    return uuid.uuid4().hex


def _parse_timestamp(value: Any, *, line_number: int) -> str:
    text = _non_empty_text(value, f"ledger line {line_number} recorded_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeLedgerError(
            f"ledger line {line_number} recorded_at is not ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise RuntimeLedgerError(
            f"ledger line {line_number} recorded_at must include a timezone"
        )
    return text


def _validate_event(
    value: Mapping[str, Any],
    *,
    line_number: int,
    expected_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeLedgerError(f"ledger line {line_number} must be an object")
    event_type = value.get("event_type")
    expected_keys = (
        _PROVIDER_EVENT_KEYS
        if event_type == _PROVIDER_EVENT_TYPE
        else _ACT_EVENT_KEYS
        if event_type == _ACT_EVENT_TYPE
        else None
    )
    if expected_keys is None:
        raise RuntimeLedgerError(
            f"ledger line {line_number} has unsupported event_type: {event_type!r}"
        )
    if set(value) != expected_keys:
        raise RuntimeLedgerError(
            f"ledger line {line_number} fields must be exactly "
            f"{sorted(expected_keys)}; got {sorted(value)}"
        )
    context = validate_runtime_context(
        {key: value[key] for key in _CONTEXT_KEYS}
    )
    if expected_context is not None and context != dict(expected_context):
        raise RuntimeLedgerError(
            f"ledger line {line_number} context does not match the bound attempt"
        )
    recorded_at = _parse_timestamp(value.get("recorded_at"), line_number=line_number)
    if event_type == _ACT_EVENT_TYPE:
        act_batch_id = value.get("act_batch_id")
        if not isinstance(act_batch_id, str) or not _LOGICAL_CALL_ID.fullmatch(
            act_batch_id
        ):
            raise RuntimeLedgerError(
                f"ledger line {line_number} act_batch_id must be 32 lowercase hex characters"
            )
        start_seed = value.get("start_seed")
        num_rollouts = value.get("num_rollouts")
        if isinstance(start_seed, bool) or not isinstance(start_seed, int):
            raise RuntimeLedgerError(
                f"ledger line {line_number} start_seed must be an integer"
            )
        if (
            isinstance(num_rollouts, bool)
            or not isinstance(num_rollouts, int)
            or num_rollouts < 1
        ):
            raise RuntimeLedgerError(
                f"ledger line {line_number} num_rollouts must be positive"
            )
        return {
            **context,
            "event_type": _ACT_EVENT_TYPE,
            "recorded_at": recorded_at,
            "act_batch_id": act_batch_id,
            "task_name": _non_empty_text(
                value.get("task_name"), f"ledger line {line_number} task_name"
            ),
            "policy_name": _non_empty_text(
                value.get("policy_name"), f"ledger line {line_number} policy_name"
            ),
            "start_seed": start_seed,
            "num_rollouts": num_rollouts,
        }
    logical_call_id = value.get("logical_call_id")
    if not isinstance(logical_call_id, str) or not _LOGICAL_CALL_ID.fullmatch(
        logical_call_id
    ):
        raise RuntimeLedgerError(
            f"ledger line {line_number} logical_call_id must be 32 lowercase hex characters"
        )
    attempt = value.get("transport_attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise RuntimeLedgerError(
            f"ledger line {line_number} transport_attempt must be a positive integer"
        )
    modality = value.get("modality")
    if modality not in _MODALITIES:
        raise RuntimeLedgerError(
            f"ledger line {line_number} modality must be one of {sorted(_MODALITIES)}"
        )
    model = _non_empty_text(value.get("model"), f"ledger line {line_number} model")
    return {
        **context,
        "event_type": _PROVIDER_EVENT_TYPE,
        "recorded_at": recorded_at,
        "logical_call_id": logical_call_id,
        "transport_attempt": attempt,
        "modality": modality,
        "model": model,
    }


def read_runtime_ledger(
    path: str | Path,
    *,
    expected_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Read and strictly validate every append-only provider event."""

    resolved = Path(path).expanduser().resolve(strict=False)
    context = (
        validate_runtime_context(expected_context)
        if expected_context is not None
        else None
    )
    if not resolved.exists():
        return []
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeLedgerError(f"runtime ledger path must be a regular file: {resolved}")
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeLedgerError(f"cannot read runtime ledger {resolved}: {exc}") from exc
    events: list[dict[str, Any]] = []
    inferred_context: dict[str, Any] | None = context
    state: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            raise RuntimeLedgerError(f"ledger line {line_number} must not be blank")
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeLedgerError(
                f"ledger line {line_number} is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise RuntimeLedgerError(f"ledger line {line_number} must be a JSON object")
        event = _validate_event(
            decoded,
            line_number=line_number,
            expected_context=inferred_context,
        )
        event_context = {key: event[key] for key in _CONTEXT_KEYS}
        if inferred_context is None:
            inferred_context = event_context
        if event["event_type"] == _ACT_EVENT_TYPE:
            if any(
                previous.get("act_batch_id") == event["act_batch_id"]
                for previous in events
                if previous["event_type"] == _ACT_EVENT_TYPE
            ):
                raise RuntimeLedgerError(
                    f"ledger line {line_number} duplicates ACT batch {event['act_batch_id']}"
                )
            events.append(event)
            continue
        call_id = event["logical_call_id"]
        previous = state.get(call_id)
        if previous is None:
            if event["transport_attempt"] != 1:
                raise RuntimeLedgerError(
                    f"ledger line {line_number} starts a logical call at transport attempt "
                    f"{event['transport_attempt']} instead of 1"
                )
            state[call_id] = {
                "transport_attempt": 1,
                "modality": event["modality"],
                "model": event["model"],
            }
        else:
            expected_attempt = int(previous["transport_attempt"]) + 1
            if event["transport_attempt"] != expected_attempt:
                raise RuntimeLedgerError(
                    f"ledger line {line_number} transport attempt must be "
                    f"{expected_attempt} for logical call {call_id}"
                )
            if (
                event["modality"] != previous["modality"]
                or event["model"] != previous["model"]
            ):
                raise RuntimeLedgerError(
                    f"ledger line {line_number} changes modality or model within one logical call"
                )
            previous["transport_attempt"] = expected_attempt
        events.append(event)
    return events


def _append_durable(path: Path, encoded: bytes) -> None:
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeLedgerError(f"cannot open runtime ledger {path}: {exc}") from exc
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise RuntimeLedgerError(
                f"short write to runtime ledger {path}: {written}/{len(encoded)} bytes"
            )
        os.fsync(descriptor)
    except OSError as exc:
        raise RuntimeLedgerError(f"cannot durably append runtime ledger {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def record_provider_transport_start(
    *,
    logical_call_id: str,
    transport_attempt: int,
    modality: str,
    model: str,
) -> dict[str, Any] | None:
    """Durably record one provider transport attempt before it starts.

    Returns ``None`` only when ledger accounting is entirely disabled.  Any
    partial or invalid configuration raises before the caller contacts the
    provider.
    """

    configured = _context_from_environment()
    if configured is None:
        return None
    path, context = configured
    if not isinstance(logical_call_id, str) or not _LOGICAL_CALL_ID.fullmatch(
        logical_call_id
    ):
        raise RuntimeLedgerError(
            "logical_call_id must be 32 lowercase hexadecimal characters"
        )
    if (
        isinstance(transport_attempt, bool)
        or not isinstance(transport_attempt, int)
        or transport_attempt < 1
    ):
        raise RuntimeLedgerError("transport_attempt must be a positive integer")
    if modality not in _MODALITIES:
        raise RuntimeLedgerError(f"modality must be one of {sorted(_MODALITIES)}")
    safe_model = _non_empty_text(model, "model")

    existing = read_runtime_ledger(path, expected_context=context)
    same_call = [
        event for event in existing if event["logical_call_id"] == logical_call_id
    ]
    expected_attempt = len(same_call) + 1
    if transport_attempt != expected_attempt:
        raise RuntimeLedgerError(
            f"transport_attempt must be {expected_attempt} for logical call "
            f"{logical_call_id}"
        )
    if same_call and (
        same_call[0]["modality"] != modality or same_call[0]["model"] != safe_model
    ):
        raise RuntimeLedgerError("modality and model must remain stable across retries")

    event = {
        **context,
        "event_type": _PROVIDER_EVENT_TYPE,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "logical_call_id": logical_call_id,
        "transport_attempt": transport_attempt,
        "modality": modality,
        "model": safe_model,
    }
    # Validate the exact object before any bytes reach the append-only file.
    event = _validate_event(
        event,
        line_number=len(existing) + 1,
        expected_context=context,
    )
    encoded = (_canonical_json(event) + "\n").encode("utf-8")
    _append_durable(path, encoded)
    return event


def record_act_batch_start(
    *,
    task_name: str,
    policy_name: str,
    start_seed: int,
    num_rollouts: int,
    act_batch_id: str | None = None,
) -> dict[str, Any] | None:
    """Durably account for an ACT batch before its subprocess starts."""

    configured = _context_from_environment()
    if configured is None:
        return None
    path, context = configured
    batch_id = act_batch_id or new_logical_call_id()
    event = {
        **context,
        "event_type": _ACT_EVENT_TYPE,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "act_batch_id": batch_id,
        "task_name": task_name,
        "policy_name": policy_name,
        "start_seed": start_seed,
        "num_rollouts": num_rollouts,
    }
    existing = read_runtime_ledger(path, expected_context=context)
    event = _validate_event(
        event,
        line_number=len(existing) + 1,
        expected_context=context,
    )
    if any(
        item.get("act_batch_id") == batch_id
        for item in existing
        if item["event_type"] == _ACT_EVENT_TYPE
    ):
        raise RuntimeLedgerError(f"duplicate ACT batch id: {batch_id}")
    _append_durable(path, (_canonical_json(event) + "\n").encode("utf-8"))
    return event


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_runtime_ledger(
    path: str | Path,
    *,
    expected_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize unique logical calls separately from transport retries."""

    context = (
        validate_runtime_context(expected_context)
        if expected_context is not None
        else None
    )
    events = read_runtime_ledger(path, expected_context=context)
    if context is None and events:
        context = {key: events[0][key] for key in _CONTEXT_KEYS}
    calls: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["event_type"] != _PROVIDER_EVENT_TYPE:
            continue
        call = calls.setdefault(
            event["logical_call_id"],
            {
                "logical_call_id": event["logical_call_id"],
                "modality": event["modality"],
                "model": event["model"],
                "transport_attempts_started": 0,
            },
        )
        call["transport_attempts_started"] += 1
    by_modality = {
        modality: {
            "logical_calls_started": sum(
                item["modality"] == modality for item in calls.values()
            ),
            "transport_attempts_started": sum(
                event["event_type"] == _PROVIDER_EVENT_TYPE
                and event["modality"] == modality
                for event in events
            ),
        }
        for modality in sorted(_MODALITIES)
    }
    resolved = Path(path).expanduser().resolve(strict=False)
    return {
        "schema_version": _SCHEMA_VERSION,
        "context": context,
        "provider_called": bool(calls),
        "provider_calls_started": len(calls),
        "provider_transport_attempts_started": sum(
            event["event_type"] == _PROVIDER_EVENT_TYPE for event in events
        ),
        "act_batches_started": sum(
            event["event_type"] == _ACT_EVENT_TYPE for event in events
        ),
        "act_rollouts_started": sum(
            int(event["num_rollouts"])
            for event in events
            if event["event_type"] == _ACT_EVENT_TYPE
        ),
        "by_modality": by_modality,
        "logical_calls": list(calls.values()),
        "ledger_sha256": file_sha256(resolved) if resolved.is_file() else None,
    }


__all__ = [
    "LEDGER_CONTEXT_ENV",
    "LEDGER_PATH_ENV",
    "RuntimeLedgerError",
    "file_sha256",
    "new_logical_call_id",
    "read_runtime_ledger",
    "record_act_batch_start",
    "record_provider_transport_start",
    "runtime_ledger_context",
    "summarize_runtime_ledger",
    "validate_runtime_context",
]
