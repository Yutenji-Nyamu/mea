"""Fail-closed pre-execution receipts for auditable MEA episodes.

The receipt is created before simulator or policy execution.  It freezes the
task source and every checkpoint artifact that can affect the policy.  Runtime
callers validate the receipt before setup/model execution, while the recorder
adds the actual imported-module binding to ``episode.json``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


class ExecutionReceiptError(RuntimeError):
    """Raised when an execution receipt or runtime binding is ambiguous."""


_HASH = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_TYPE = "mea_task_execution_preflight"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ExecutionReceiptError(f"receipt artifact is not a file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ExecutionReceiptError(f"{field} must be an integer")
    return value


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionReceiptError(f"{field} must be an object")
    return deepcopy(dict(value))


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ExecutionReceiptError(f"{field} must be a lowercase SHA-256")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    field: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise ExecutionReceiptError(
            f"{field} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def build_checkpoint_bundle(
    checkpoint_dir: str | Path | None,
    *,
    required_files: Sequence[str] = ("policy_last.ckpt", "dataset_stats.pkl"),
    kind: str = "act_checkpoint_bundle",
) -> dict[str, Any]:
    """Freeze the exact checkpoint file set consumed by a policy.

    Expert/setup probes have no model checkpoint and must explicitly use
    ``kind="expert_no_checkpoint"`` with ``checkpoint_dir=None``.
    """

    if kind == "expert_no_checkpoint":
        if checkpoint_dir is not None:
            raise ExecutionReceiptError(
                "expert_no_checkpoint cannot name a checkpoint directory"
            )
        payload = {
            "kind": kind,
            "root": None,
            "artifacts": [],
        }
        return {
            **payload,
            "bundle_sha256": canonical_sha256(payload),
        }
    if kind != "act_checkpoint_bundle":
        raise ExecutionReceiptError(f"unsupported checkpoint kind: {kind!r}")
    if checkpoint_dir is None:
        raise ExecutionReceiptError("ACT receipt requires checkpoint_dir")
    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        raise ExecutionReceiptError(
            f"checkpoint directory does not exist: {root}"
        )
    names = list(required_files)
    if (
        not names
        or len(names) != len(set(names))
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ExecutionReceiptError(
            "checkpoint required_files must be unique non-empty paths"
        )
    artifacts: list[dict[str, Any]] = []
    for relative_name in sorted(names):
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ExecutionReceiptError(
                f"unsafe checkpoint relative path: {relative_name!r}"
            )
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ExecutionReceiptError(
                f"checkpoint artifact escapes root: {relative_name!r}"
            ) from exc
        artifacts.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": artifact.stat().st_size,
                "sha256": file_sha256(artifact),
            }
        )
    payload = {
        "kind": kind,
        "root": str(root),
        "artifacts": artifacts,
    }
    return {
        **payload,
        "bundle_sha256": canonical_sha256(payload),
    }


def validate_checkpoint_bundle(
    value: Mapping[str, Any],
    *,
    verify_files: bool,
) -> dict[str, Any]:
    bundle = _object(value, field="checkpoint")
    if bundle.get("kind") == "expert_no_checkpoint":
        _require_exact_keys(
            bundle,
            {"kind", "root", "artifacts", "bundle_sha256"},
            field="checkpoint",
        )
        expected = build_checkpoint_bundle(
            None,
            kind="expert_no_checkpoint",
        )
        if bundle != expected:
            raise ExecutionReceiptError(
                "invalid expert_no_checkpoint checkpoint bundle"
            )
        return bundle
    if bundle.get("kind") != "act_checkpoint_bundle":
        raise ExecutionReceiptError("invalid checkpoint bundle kind")
    _require_exact_keys(
        bundle,
        {"kind", "root", "artifacts", "bundle_sha256"},
        field="checkpoint",
    )
    root_value = bundle.get("root")
    artifacts_value = bundle.get("artifacts")
    if not isinstance(root_value, str) or not root_value:
        raise ExecutionReceiptError("checkpoint.root must be a path")
    if not isinstance(artifacts_value, list) or not artifacts_value:
        raise ExecutionReceiptError("checkpoint.artifacts must be non-empty")
    root = Path(root_value).expanduser().resolve()
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(artifacts_value):
        artifact = _object(raw, field=f"checkpoint.artifacts[{index}]")
        _require_exact_keys(
            artifact,
            {"relative_path", "size_bytes", "sha256"},
            field=f"checkpoint.artifacts[{index}]",
        )
        relative_name = artifact.get("relative_path")
        size = artifact.get("size_bytes")
        digest = artifact.get("sha256")
        if (
            not isinstance(relative_name, str)
            or not relative_name
            or relative_name in seen
        ):
            raise ExecutionReceiptError(
                "checkpoint relative paths must be unique strings"
            )
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ExecutionReceiptError(
                f"unsafe checkpoint relative path: {relative_name!r}"
            )
        _strict_int(size, field=f"checkpoint.artifacts[{index}].size_bytes")
        if size < 0:
            raise ExecutionReceiptError("checkpoint size cannot be negative")
        _sha(digest, field=f"checkpoint.artifacts[{index}].sha256")
        normalized.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": size,
                "sha256": digest,
            }
        )
        seen.add(relative_name)
    if normalized != sorted(
        normalized, key=lambda item: item["relative_path"]
    ):
        raise ExecutionReceiptError(
            "checkpoint artifacts must use deterministic sorted order"
        )
    payload = {
        "kind": "act_checkpoint_bundle",
        "root": str(root),
        "artifacts": normalized,
    }
    if _sha(
        bundle.get("bundle_sha256"),
        field="checkpoint.bundle_sha256",
    ) != canonical_sha256(payload):
        raise ExecutionReceiptError("checkpoint bundle seal differs")
    if verify_files:
        for artifact in normalized:
            path = (root / artifact["relative_path"]).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ExecutionReceiptError(
                    "checkpoint artifact escapes its frozen root"
                ) from exc
            if (
                not path.is_file()
                or path.stat().st_size != artifact["size_bytes"]
                or file_sha256(path) != artifact["sha256"]
            ):
                raise ExecutionReceiptError(
                    "checkpoint artifact changed after receipt freeze: "
                    f"{artifact['relative_path']}"
                )
    return {
        **payload,
        "bundle_sha256": bundle["bundle_sha256"],
    }


def seal_execution_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(payload))
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def validate_execution_receipt(
    value: Mapping[str, Any],
    *,
    verify_checkpoint_files: bool = False,
) -> dict[str, Any]:
    receipt = _object(value, field="execution receipt")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_type") != _RECEIPT_TYPE
    ):
        raise ExecutionReceiptError("invalid execution receipt identity")
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "receipt_type",
            "candidate",
            "episode",
            "checkpoint",
            "receipt_sha256",
        },
        field="execution receipt",
    )
    candidate = _object(receipt.get("candidate"), field="candidate")
    episode = _object(receipt.get("episode"), field="episode")
    _require_exact_keys(
        candidate,
        {
            "task_name",
            "task_class",
            "task_module",
            "module_origin",
            "task_source_sha256",
            "proposal_sha256",
            "scene_method_sha256",
            "success_method_sha256",
            "candidate_manifest_sha256",
            "execution_module_sha256",
        },
        field="candidate",
    )
    _require_exact_keys(
        episode,
        {
            "task_name",
            "task_module",
            "task_config",
            "checkpoint_setting",
            "policy_name",
            "seed",
            "episode_index",
        },
        field="episode",
    )
    checkpoint = validate_checkpoint_bundle(
        _object(receipt.get("checkpoint"), field="checkpoint"),
        verify_files=verify_checkpoint_files,
    )
    for field in (
        "task_name",
        "task_class",
        "task_module",
        "module_origin",
    ):
        if not isinstance(candidate.get(field), str) or not candidate[field]:
            raise ExecutionReceiptError(f"candidate.{field} must be a string")
    origin = Path(candidate["module_origin"]).expanduser().resolve()
    if str(origin) != candidate["module_origin"]:
        raise ExecutionReceiptError(
            "candidate.module_origin must be an absolute normalized path"
        )
    for field in (
        "task_source_sha256",
        "proposal_sha256",
        "scene_method_sha256",
        "success_method_sha256",
        "candidate_manifest_sha256",
        "execution_module_sha256",
    ):
        _sha(candidate.get(field), field=f"candidate.{field}")
    for field in (
        "task_name",
        "task_module",
        "task_config",
        "checkpoint_setting",
        "policy_name",
    ):
        if not isinstance(episode.get(field), str) or not episode[field]:
            raise ExecutionReceiptError(f"episode.{field} must be a string")
    _strict_int(episode.get("seed"), field="episode.seed")
    episode_index = _strict_int(
        episode.get("episode_index"),
        field="episode.episode_index",
    )
    if episode_index < 0:
        raise ExecutionReceiptError("episode.episode_index cannot be negative")
    if episode["task_name"] != candidate["task_name"]:
        raise ExecutionReceiptError("receipt task_name fields differ")
    if episode["task_module"] != candidate["task_module"]:
        raise ExecutionReceiptError("receipt task_module fields differ")
    expected_module_identity = canonical_sha256(
        {
            key: candidate[key]
            for key in (
                "task_name",
                "task_class",
                "task_module",
                "task_source_sha256",
                "proposal_sha256",
                "scene_method_sha256",
                "success_method_sha256",
            )
        }
    )
    if candidate["execution_module_sha256"] != expected_module_identity:
        raise ExecutionReceiptError("candidate execution module seal differs")
    receipt["candidate"] = candidate
    receipt["episode"] = episode
    receipt["checkpoint"] = checkpoint
    seal = _sha(receipt.get("receipt_sha256"), field="receipt_sha256")
    unsealed = deepcopy(receipt)
    unsealed.pop("receipt_sha256")
    if seal != canonical_sha256(unsealed):
        raise ExecutionReceiptError("execution receipt seal differs")
    return receipt


def load_execution_receipt(
    path: str | Path,
    *,
    verify_checkpoint_files: bool = False,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionReceiptError(
            f"cannot read execution receipt: {resolved}"
        ) from exc
    return validate_execution_receipt(
        value,
        verify_checkpoint_files=verify_checkpoint_files,
    )


def validate_execution_invocation(
    receipt: Mapping[str, Any],
    *,
    task_name: str,
    task_module: str | None,
    task_config: str | None,
    checkpoint_setting: str | None,
    policy_name: str,
    seed: int,
    episode_index: int,
    checkpoint_dir: str | Path | None = None,
    verify_checkpoint_files: bool = True,
) -> dict[str, Any]:
    validated = validate_execution_receipt(
        receipt,
        verify_checkpoint_files=verify_checkpoint_files,
    )
    actual = {
        "task_name": task_name,
        "task_module": task_module,
        "task_config": task_config,
        "checkpoint_setting": checkpoint_setting,
        "policy_name": policy_name,
        "seed": seed,
        "episode_index": episode_index,
    }
    for field, observed in actual.items():
        if validated["episode"][field] != observed:
            raise ExecutionReceiptError(
                f"execution receipt {field} mismatch: "
                f"{validated['episode'][field]!r} != {observed!r}"
            )
    checkpoint = validated["checkpoint"]
    if checkpoint["kind"] == "act_checkpoint_bundle":
        if checkpoint_dir is None:
            raise ExecutionReceiptError(
                "ACT execution requires the actual checkpoint directory"
            )
        actual_root = Path(checkpoint_dir).expanduser().resolve()
        if str(actual_root) != checkpoint["root"]:
            raise ExecutionReceiptError(
                "execution checkpoint directory differs from receipt"
            )
    elif checkpoint_dir is not None:
        raise ExecutionReceiptError(
            "expert_no_checkpoint invocation cannot use checkpoint_dir"
        )
    return validated


def validate_frozen_candidate_source(
    receipt: Mapping[str, Any],
) -> Path:
    """Verify the frozen task source before importing simulator task code."""

    validated = validate_execution_receipt(
        receipt,
        verify_checkpoint_files=False,
    )
    origin = Path(
        validated["candidate"]["module_origin"]
    ).expanduser().resolve()
    if file_sha256(origin) != validated["candidate"]["task_source_sha256"]:
        raise ExecutionReceiptError(
            "candidate task source changed after receipt freeze"
        )
    return origin


def _module_origin(module: ModuleType) -> Path:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise ExecutionReceiptError("executed task module has no source file")
    path = Path(origin).expanduser().resolve()
    if path.suffix in {".pyc", ".pyo"}:
        source = path.with_suffix(".py")
        if source.is_file():
            path = source
    return path


def validate_imported_task_binding(
    receipt: Mapping[str, Any],
    task: Any,
) -> dict[str, Any]:
    validated = validate_execution_receipt(
        receipt,
        verify_checkpoint_files=False,
    )
    candidate = validated["candidate"]
    task_class = task.__class__
    actual_module_name = str(task_class.__module__)
    actual_class_name = str(task_class.__name__)
    if actual_module_name != candidate["task_module"]:
        raise ExecutionReceiptError(
            "executed task class module differs from receipt"
        )
    if actual_class_name != candidate["task_class"]:
        raise ExecutionReceiptError(
            "executed task class name differs from receipt"
        )
    module = importlib.import_module(actual_module_name)
    origin = _module_origin(module)
    if str(origin) != candidate["module_origin"]:
        raise ExecutionReceiptError(
            "executed task module origin differs from receipt"
        )
    digest = file_sha256(origin)
    if digest != candidate["task_source_sha256"]:
        raise ExecutionReceiptError(
            "executed task source changed after receipt freeze"
        )
    return {
        "task_name": candidate["task_name"],
        "task_class": actual_class_name,
        "task_module": actual_module_name,
        "module_origin": str(origin),
        "task_source_sha256": digest,
        "execution_module_sha256": candidate["execution_module_sha256"],
        "proposal_sha256": candidate["proposal_sha256"],
        "scene_method_sha256": candidate["scene_method_sha256"],
        "success_method_sha256": candidate["success_method_sha256"],
        "policy_name": validated["episode"]["policy_name"],
        "seed": validated["episode"]["seed"],
        "episode_index": validated["episode"]["episode_index"],
        "task_config": validated["episode"]["task_config"],
        "checkpoint_setting": validated["episode"]["checkpoint_setting"],
        "checkpoint_kind": validated["checkpoint"]["kind"],
        "checkpoint_bundle_sha256": validated["checkpoint"][
            "bundle_sha256"
        ],
    }


def expected_recorded_execution_binding(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the binding an episode must record for this sealed receipt."""

    validated = validate_execution_receipt(
        receipt,
        verify_checkpoint_files=False,
    )
    candidate = validated["candidate"]
    episode = validated["episode"]
    checkpoint = validated["checkpoint"]
    return {
        "task_name": candidate["task_name"],
        "task_class": candidate["task_class"],
        "task_module": candidate["task_module"],
        "module_origin": candidate["module_origin"],
        "task_source_sha256": candidate["task_source_sha256"],
        "execution_module_sha256": candidate["execution_module_sha256"],
        "proposal_sha256": candidate["proposal_sha256"],
        "scene_method_sha256": candidate["scene_method_sha256"],
        "success_method_sha256": candidate["success_method_sha256"],
        "policy_name": episode["policy_name"],
        "seed": episode["seed"],
        "episode_index": episode["episode_index"],
        "task_config": episode["task_config"],
        "checkpoint_setting": episode["checkpoint_setting"],
        "checkpoint_kind": checkpoint["kind"],
        "checkpoint_bundle_sha256": checkpoint["bundle_sha256"],
    }


def validate_recorded_execution_metadata(
    metadata: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that one recorded episode executed the frozen receipt exactly."""

    if not isinstance(metadata, Mapping):
        raise ExecutionReceiptError("episode metadata must be an object")
    expected_receipt = validate_execution_receipt(
        receipt,
        verify_checkpoint_files=False,
    )
    embedded = metadata.get("execution_receipt")
    if not isinstance(embedded, Mapping):
        raise ExecutionReceiptError(
            "legacy_episode_missing_execution_receipt"
        )
    actual_receipt = validate_execution_receipt(
        embedded,
        verify_checkpoint_files=False,
    )
    if actual_receipt != expected_receipt:
        raise ExecutionReceiptError(
            "episode execution receipt differs from preflight receipt"
        )
    if metadata.get("execution_receipt_sha256") != expected_receipt[
        "receipt_sha256"
    ]:
        raise ExecutionReceiptError(
            "episode execution receipt SHA differs"
        )
    binding = metadata.get("executed_binding")
    if not isinstance(binding, Mapping):
        raise ExecutionReceiptError(
            "legacy_episode_missing_executed_binding"
        )
    expected_binding = expected_recorded_execution_binding(expected_receipt)
    if dict(binding) != expected_binding:
        raise ExecutionReceiptError(
            "episode executed binding differs from receipt"
        )
    for field in (
        "task_name",
        "task_module",
        "task_config",
        "checkpoint_setting",
        "policy_name",
        "seed",
        "episode_index",
    ):
        if metadata.get(field) != expected_receipt["episode"][field]:
            raise ExecutionReceiptError(
                f"episode metadata {field} differs from receipt"
            )
    if metadata.get("executed_task_module_sha256") != expected_binding[
        "task_source_sha256"
    ]:
        raise ExecutionReceiptError(
            "episode executed task hash differs from receipt"
        )
    if metadata.get("executed_checkpoint_bundle_sha256") != expected_binding[
        "checkpoint_bundle_sha256"
    ]:
        raise ExecutionReceiptError(
            "episode executed checkpoint hash differs from receipt"
        )
    return expected_binding
