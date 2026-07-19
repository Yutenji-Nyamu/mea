"""Hash-bind one completed Agent round to the artifacts that support it.

The sidecar deliberately avoids a self-hash cycle: it hashes the round summary
*before* the ``provenance`` pointer is added.  Verification removes that
pointer again before recomputing the digest.  Existing evaluations remain
readable; new Agent rounds opt into this stricter contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class RoundProvenanceError(RuntimeError):
    """A round provenance sidecar is missing, unsafe, or has been tampered with."""


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RoundProvenanceError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(repo_root: Path, path: Path) -> Path:
    root = repo_root.expanduser().resolve()
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    cursor = root
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise RoundProvenanceError(f"artifact escapes repository: {path}") from exc
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RoundProvenanceError(f"artifact contains symlink component: {path}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise RoundProvenanceError(f"artifact is not a repository file: {path}")
    return resolved


def _artifact_ref(repo_root: Path, path: Path, *, kind: str) -> dict[str, Any]:
    resolved = _safe_file(repo_root, path)
    relative = resolved.relative_to(repo_root.resolve()).as_posix()
    return {
        "kind": kind,
        "path": relative,
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _summary_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(summary)
    value.pop("provenance", None)
    return value


def _existing_optional_artifacts(
    child_dir: Path, execution_dir: Path, ledger_paths: Iterable[Path]
) -> list[tuple[str, Path]]:
    logical_execution_dir = (
        execution_dir.parent
        if execution_dir.name.startswith("round_attempt_")
        else execution_dir
    )
    candidates = [
        ("child_manifest", child_dir / "manifest.json"),
        ("variant_spec", child_dir / "variant_spec.json"),
        ("reflection_summary", child_dir / "reflection/summary.json"),
        ("act_result", child_dir / "evaluation/_result.txt"),
        ("act_metadata", child_dir / "evaluation/act.json"),
        ("round_aggregate", execution_dir / "aggregate_result.json"),
        ("taskgen_command", execution_dir / "taskgen_command.json"),
        ("child_run_pointer", execution_dir / "child_run.json"),
        ("tool_execution", execution_dir / "planned_tool/tool_execution.json"),
        ("execution_vqa_query", execution_dir / "execution_vqa_query.json"),
        ("execution_vqa_result", execution_dir / "execution_vqa/execution_vqa.json"),
        ("execution_vqa_error", execution_dir / "execution_vqa_error.json"),
        (
            "round_recovery",
            logical_execution_dir / "whole_round_recovery/recovery_summary.json",
        ),
    ]
    candidates.extend(("runtime_call_ledger", path) for path in ledger_paths)
    seen: set[Path] = set()
    result: list[tuple[str, Path]] = []
    for kind, path in candidates:
        resolved = path.resolve(strict=False)
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        result.append((kind, path))
    return result


def bind_round_provenance(
    repo_root: str | Path,
    evaluation_dir: str | Path,
    *,
    round_plan: Mapping[str, Any],
    child_dir: str | Path,
    round_summary: Mapping[str, Any],
    ledger_paths: Iterable[str | Path] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write one append-only sidecar and return ``(summary, provenance)``.

    The caller persists the returned summary in its normal location.  The
    provenance sidecar itself is created exclusively and therefore cannot
    silently replace evidence from an earlier attempt.
    """

    root = Path(repo_root).expanduser().resolve()
    evaluation = Path(evaluation_dir).expanduser().resolve()
    if not evaluation.is_relative_to(root):
        raise RoundProvenanceError("evaluation directory must stay inside repository")
    round_id = str(round_plan.get("round_id") or "").strip()
    evaluation_id = str(
        round_summary.get("evaluation_id") or evaluation.name
    ).strip()
    if not round_id or not evaluation_id:
        raise RoundProvenanceError("round_id and evaluation_id are required")
    child = Path(child_dir).expanduser().resolve()
    execution_value = round_summary.get("execution_artifact_dir")
    execution_dir = (
        root / str(execution_value)
        if isinstance(execution_value, str) and execution_value
        else evaluation / "execution" / round_id
    )
    normalized_ledgers = [Path(path) for path in ledger_paths]
    artifacts = [
        _artifact_ref(root, path, kind=kind)
        for kind, path in _existing_optional_artifacts(
            child, execution_dir, normalized_ledgers
        )
    ]
    if not any(item["kind"] == "child_manifest" for item in artifacts):
        raise RoundProvenanceError("completed round has no child manifest")

    recovery = (round_summary.get("observations") or {}).get(
        "whole_round_recovery"
    ) or {}
    attempts = recovery.get("attempts") or []
    attempt_index = (
        int(attempts[-1].get("attempt_index", 0)) if attempts else 0
    )
    binding = {
        "evaluation_id": evaluation_id,
        "round_id": round_id,
        "child_run_id": (
            round_summary.get("child_run_id")
            or round_summary.get("taskgen_run_id")
        ),
        "final_round_attempt_index": attempt_index,
        "round_plan_sha256": canonical_sha256(round_plan),
        "round_summary_payload_sha256": canonical_sha256(
            _summary_payload(round_summary)
        ),
        "artifacts": artifacts,
    }
    provenance = {
        "schema_version": 1,
        "kind": "mea_round_provenance_v1",
        "binding": binding,
        "binding_sha256": canonical_sha256(binding),
        "paper_table_eligible": False,
        "claim_scope": "runtime provenance integrity; not a policy outcome",
    }
    sidecar = evaluation / "summary" / f"{round_id}.provenance.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sidecar.open("x", encoding="utf-8") as handle:
            json.dump(provenance, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
    except FileExistsError as exc:
        raise RoundProvenanceError(f"provenance sidecar already exists: {sidecar}") from exc

    summary_with_pointer = dict(round_summary)
    summary_with_pointer["provenance"] = {
        "path": sidecar.relative_to(root).as_posix(),
        "sha256": file_sha256(sidecar),
        "binding_sha256": provenance["binding_sha256"],
    }
    return summary_with_pointer, provenance


def verify_round_provenance(
    repo_root: str | Path,
    provenance_path: str | Path,
    *,
    round_plan: Mapping[str, Any],
    round_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly verify the hashes reachable from one provenance sidecar."""

    root = Path(repo_root).expanduser().resolve()
    path = _safe_file(root, Path(provenance_path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoundProvenanceError(f"cannot read provenance sidecar: {exc}") from exc
    if not isinstance(value, dict) or value.get("kind") != "mea_round_provenance_v1":
        raise RoundProvenanceError("unsupported round provenance sidecar")
    binding = value.get("binding")
    if not isinstance(binding, dict) or value.get("binding_sha256") != canonical_sha256(binding):
        raise RoundProvenanceError("round provenance binding hash mismatch")
    if binding.get("round_plan_sha256") != canonical_sha256(round_plan):
        raise RoundProvenanceError("round plan hash mismatch")
    if binding.get("round_summary_payload_sha256") != canonical_sha256(
        _summary_payload(round_summary)
    ):
        raise RoundProvenanceError("round summary payload hash mismatch")
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, list):
        raise RoundProvenanceError("provenance artifacts must be a list")
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {
            "kind",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RoundProvenanceError("invalid provenance artifact entry")
        artifact = _safe_file(root, Path(str(item["path"])))
        if artifact.stat().st_size != item["size_bytes"] or file_sha256(artifact) != item["sha256"]:
            raise RoundProvenanceError(f"artifact hash mismatch: {item['path']}")
    pointer = round_summary.get("provenance")
    if not isinstance(pointer, Mapping):
        raise RoundProvenanceError("round summary has no provenance pointer")
    if pointer.get("sha256") != file_sha256(path):
        raise RoundProvenanceError("provenance pointer hash mismatch")
    if pointer.get("binding_sha256") != value["binding_sha256"]:
        raise RoundProvenanceError("provenance pointer binding mismatch")
    return value
