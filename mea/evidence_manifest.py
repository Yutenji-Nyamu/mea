"""Hash-pinned evidence preregistration for small MEA experiments.

The manifest is prepared before a rollout and is deliberately independent of
RoboTwin.  It binds the open query, candidate universe, Git revision, ACT
checkpoint bytes, telemetry contract, matched N=1 sample universe, and the
source artifacts used to define the experiment.  Preparation and validation
only read files and therefore start zero ACT rollouts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from mea.toolkit.profiles import load_telemetry_profile, telemetry_profile_sha256
from mea.planner.catalog import ACTCatalogError, build_act_catalog, catalog_task
from mea.planner.click_bell import (
    CLICK_BELL_ADAPTIVE_ASPECTS,
    CLICK_BELL_ADAPTIVE_TEMPLATES,
)


class EvidenceManifestError(RuntimeError):
    """Raised when preregistration input or a registered artifact is invalid."""


_STRATEGIES = ("fixed_predeclared_v1", "dynamic_evidence_v1")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEX_COMMIT = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_CHECKPOINT_FILES = [
    "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt",
    "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/dataset_stats.pkl",
]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require_clean_git_head(root: str | Path) -> str:
    """Return HEAD only for the exact repository root with clean tracked files."""

    repo = Path(root).expanduser().resolve()
    top = _git(repo, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        raise EvidenceManifestError(
            f"cannot resolve Git worktree: {top.stderr.strip() or 'git failed'}"
        )
    try:
        top_path = Path(top.stdout.strip()).resolve(strict=True)
    except OSError as exc:
        raise EvidenceManifestError("Git worktree root is invalid") from exc
    if top_path != repo:
        raise EvidenceManifestError("--repo-root must be the exact Git worktree root")
    head_process = _git(repo, "rev-parse", "--verify", "HEAD")
    head = head_process.stdout.strip().lower()
    if head_process.returncode != 0 or not _HEX_COMMIT.fullmatch(head):
        raise EvidenceManifestError("Git HEAD must resolve to a full 40-character commit")
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if status.returncode != 0:
        raise EvidenceManifestError(
            f"cannot inspect tracked worktree: {status.stderr.strip() or 'git failed'}"
        )
    if status.stdout.strip():
        raise EvidenceManifestError("tracked Git worktree must be clean")
    return head


def _git_head(root: Path) -> str:
    return require_clean_git_head(root)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise EvidenceManifestError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _repo_file(root: Path, value: Any, *, label: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvidenceManifestError(f"{label} must be a non-empty canonical path")
    if "\\" in value or ":" in value:
        raise EvidenceManifestError(f"{label} must be a POSIX repo-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise EvidenceManifestError(f"{label} must stay inside the repository")
    canonical = pure.as_posix()
    if canonical != value:
        raise EvidenceManifestError(f"{label} is not canonical: {value!r}")

    candidate = root
    for part in pure.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise EvidenceManifestError(f"{label} may not traverse a symlink: {value}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise EvidenceManifestError(f"{label} is missing: {value}: {exc}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise EvidenceManifestError(f"{label} is not a regular repo file: {value}")
    return resolved, canonical


def read_repo_json(root: str | Path, relative: str, *, label: str) -> dict[str, Any]:
    repo = Path(root).expanduser().resolve()
    path, _ = _repo_file(repo, relative, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceManifestError(f"cannot read {label}: {relative}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceManifestError(f"{label} must be a JSON object")
    return value


def _file_identity(root: Path, value: Any, *, label: str) -> dict[str, Any]:
    path, relative = _repo_file(root, value, label=label)
    stat = path.stat()
    return {
        "path": relative,
        "size_bytes": stat.st_size,
        "sha256": _file_sha256(path),
    }


def _checkpoint_file_identity(
    root: Path, value: Any, *, label: str
) -> dict[str, Any]:
    """Bind a fixed logical checkpoint path, its indirection, and final bytes.

    MEA commonly links ``policy`` to an adjacent RoboTwin checkout.  Source
    artifacts remain symlink-free, while the two allowlisted checkpoint paths
    may cross that deployment link.  Recording every link plus the resolved
    absolute path makes a later retarget fail even when the replacement bytes
    happen to be identical.
    """

    if not isinstance(value, str) or not value or value != value.strip():
        raise EvidenceManifestError(f"{label} must be a non-empty canonical path")
    if "\\" in value or ":" in value:
        raise EvidenceManifestError(f"{label} must be a POSIX repo-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise EvidenceManifestError(f"{label} must use a canonical logical path")
    if pure.as_posix() != value:
        raise EvidenceManifestError(f"{label} is not canonical: {value!r}")

    cursor = root
    symlink_chain: list[dict[str, str]] = []
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            try:
                target = os.readlink(cursor)
            except OSError as exc:
                raise EvidenceManifestError(
                    f"cannot read checkpoint symlink {cursor}: {exc}"
                ) from exc
            symlink_chain.append(
                {
                    "path": cursor.relative_to(root).as_posix(),
                    "target": target,
                }
            )
    try:
        resolved = cursor.resolve(strict=True)
    except OSError as exc:
        raise EvidenceManifestError(f"{label} is missing: {value}: {exc}") from exc
    if not resolved.is_file():
        raise EvidenceManifestError(f"{label} does not resolve to a regular file")
    stat = resolved.stat()
    return {
        "path": value,
        "resolved_path": resolved.as_posix(),
        "symlink_chain": symlink_chain,
        "size_bytes": stat.st_size,
        "sha256": _file_sha256(resolved),
    }


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvidenceManifestError(f"{label} must be a non-empty trimmed string")
    return value


def _identifier(value: Any, *, label: str) -> str:
    normalized = _text(value, label=label)
    if not _IDENTIFIER.fullmatch(normalized):
        raise EvidenceManifestError(f"{label} is not a safe identifier")
    return normalized


def _candidate_suite(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise EvidenceManifestError("candidate_suite must be a non-empty list")
    candidates = [_identifier(item, label="candidate_suite item") for item in value]
    if len(candidates) != len(set(candidates)):
        raise EvidenceManifestError("candidate_suite contains duplicates")
    if len(candidates) > 5:
        raise EvidenceManifestError("candidate_suite exceeds the agile maximum of 5")
    return candidates


def _expanded_aspects(candidates: list[str]) -> list[str]:
    """Infer a unique ordered list of complete trusted aspect expansions."""

    aspects: list[str] = []
    offset = 0
    while offset < len(candidates):
        template_id = candidates[offset]
        template = CLICK_BELL_ADAPTIVE_TEMPLATES.get(template_id)
        if not isinstance(template, dict):
            raise EvidenceManifestError(
                f"candidate is not a trusted click_bell template: {template_id}"
            )
        aspect_id = template.get("aspect_id")
        if aspect_id in aspects or aspect_id not in CLICK_BELL_ADAPTIVE_ASPECTS:
            raise EvidenceManifestError(
                f"candidate suite has invalid or repeated aspect: {aspect_id}"
            )
        expansion = list(CLICK_BELL_ADAPTIVE_ASPECTS[aspect_id]["template_ids"])
        if candidates[offset : offset + len(expansion)] != expansion:
            raise EvidenceManifestError(
                f"candidate suite must contain the complete ordered expansion for {aspect_id}"
            )
        if any(
            CLICK_BELL_ADAPTIVE_TEMPLATES.get(item, {}).get("aspect_id") != aspect_id
            for item in expansion
        ):
            raise EvidenceManifestError(
                f"trusted template expansion is inconsistent for {aspect_id}"
            )
        aspects.append(str(aspect_id))
        offset += len(expansion)
    return aspects


def trusted_click_bell_contract(root: str | Path) -> dict[str, Any]:
    repo = Path(root).expanduser().resolve()
    try:
        catalog = build_act_catalog(repo)
        task = catalog_task(catalog, "click_bell")
    except (ACTCatalogError, ValueError) as exc:
        raise EvidenceManifestError(
            "trusted ACT catalog does not contain checkpoint-ready click_bell"
        ) from exc
    expected_policy = {
        "policy_name": "ACT",
        "checkpoint_setting": "demo_clean",
        "expert_data_num": 50,
        "checkpoint_id": "act-click_bell/demo_clean-50",
        "ready": True,
    }
    if task.get("checkpoint") != expected_policy:
        raise EvidenceManifestError("trusted click_bell checkpoint contract changed")
    template_contract = {
        "aspects": deepcopy(CLICK_BELL_ADAPTIVE_ASPECTS),
        "templates": deepcopy(CLICK_BELL_ADAPTIVE_TEMPLATES),
    }
    return {
        "catalog_sha256": catalog["catalog_sha256"],
        "click_bell_task_sha256": canonical_sha256(task),
        "template_contract_sha256": canonical_sha256(template_contract),
    }


def _schedule(value: Any, candidates: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise EvidenceManifestError("sample_schedule must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    by_strategy: dict[str, dict[str, int]] = {name: {} for name in _STRATEGIES}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise EvidenceManifestError(f"sample_schedule[{index}] must be an object")
        _exact_keys(item, {"strategy", "variant_id", "seed"}, f"sample_schedule[{index}]")
        strategy = item["strategy"]
        if strategy not in _STRATEGIES:
            raise EvidenceManifestError(f"unknown sample strategy: {strategy!r}")
        variant = _identifier(item["variant_id"], label="sample variant_id")
        if variant not in candidates:
            raise EvidenceManifestError(f"sample variant is outside candidate_suite: {variant}")
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise EvidenceManifestError("sample seed must be a non-negative integer")
        identity = (strategy, variant, seed)
        if identity in seen or variant in by_strategy[strategy]:
            raise EvidenceManifestError(f"duplicate N=1 scheduled sample: {identity}")
        seen.add(identity)
        by_strategy[strategy][variant] = seed
        normalized.append(
            {"strategy": strategy, "variant_id": variant, "seed": seed}
        )
    expected = set(candidates)
    for strategy in _STRATEGIES:
        if set(by_strategy[strategy]) != expected:
            raise EvidenceManifestError(
                f"{strategy} must preregister exactly one sample per candidate"
            )
        ordered_variants = [
            item["variant_id"]
            for item in normalized
            if item["strategy"] == strategy
        ]
        if ordered_variants != candidates:
            raise EvidenceManifestError(
                f"{strategy} schedule must preserve candidate_suite order"
            )
    if by_strategy[_STRATEGIES[0]] != by_strategy[_STRATEGIES[1]]:
        raise EvidenceManifestError("fixed and dynamic sample identities must match")
    return normalized


def _path_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise EvidenceManifestError(f"{label} must be a non-empty list")
    result = [_text(item, label=f"{label} item") for item in value]
    if len(result) != len(set(result)):
        raise EvidenceManifestError(f"{label} contains duplicate paths")
    return result


def prepare_evidence_manifest(
    repo_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a hash-pinned manifest without starting a provider or simulator."""

    root = Path(repo_root).expanduser().resolve()
    if not isinstance(config, Mapping):
        raise EvidenceManifestError("preregistration config must be an object")
    _exact_keys(
        config,
        {
            "schema_version",
            "registration_id",
            "claim_scope",
            "task_name",
            "query",
            "base_commit",
            "candidate_suite",
            "checkpoint_setting",
            "expert_data_num",
            "checkpoint_files",
            "telemetry_profile",
            "sample_schedule",
            "source_artifacts",
        },
        "preregistration config",
    )
    if config["schema_version"] != 1:
        raise EvidenceManifestError("preregistration config schema_version must be 1")

    registration_id = _identifier(config["registration_id"], label="registration_id")
    claim_scope = _text(config["claim_scope"], label="claim_scope")
    if config["task_name"] != "click_bell":
        raise EvidenceManifestError("task_name must be click_bell")
    if config["expert_data_num"] != 50:
        raise EvidenceManifestError("expert_data_num must be exactly 50")
    query = _text(config["query"], label="query")
    base_commit = str(config["base_commit"]).lower()
    if not _HEX_COMMIT.fullmatch(base_commit):
        raise EvidenceManifestError("base_commit must be a full 40-character Git SHA")
    current_head = _git_head(root)
    if current_head != base_commit:
        raise EvidenceManifestError(
            f"base_commit does not match current Git HEAD: {base_commit} != {current_head}"
        )
    candidates = _candidate_suite(config["candidate_suite"])
    aspect_ids = _expanded_aspects(candidates)
    schedule = _schedule(config["sample_schedule"], candidates)
    checkpoint_paths = _path_list(config["checkpoint_files"], label="checkpoint_files")
    source_paths = _path_list(config["source_artifacts"], label="source_artifacts")
    if checkpoint_paths != _CHECKPOINT_FILES:
        raise EvidenceManifestError(
            "checkpoint_files must exactly equal the trusted click_bell demo_clean-50 files"
        )
    if set(checkpoint_paths) & set(source_paths):
        raise EvidenceManifestError("checkpoint_files and source_artifacts must be disjoint")

    checkpoints = [
        _checkpoint_file_identity(root, path, label=f"checkpoint_files[{index}]")
        for index, path in enumerate(checkpoint_paths)
    ]
    sources = [
        _file_identity(root, path, label=f"source_artifacts[{index}]")
        for index, path in enumerate(source_paths)
    ]
    telemetry_id = _identifier(config["telemetry_profile"], label="telemetry_profile")
    try:
        telemetry = load_telemetry_profile(telemetry_id)
    except ValueError as exc:
        raise EvidenceManifestError(str(exc)) from exc

    trusted_contract = trusted_click_bell_contract(root)
    checkpoint_setting = _identifier(
        config["checkpoint_setting"], label="checkpoint_setting"
    )
    if checkpoint_setting != "demo_clean":
        raise EvidenceManifestError("checkpoint_setting must be demo_clean")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_type": "mea_evidence_preregistration_v1",
        "registration_id": registration_id,
        "prepared_at": datetime.now().astimezone().isoformat(),
        "claim_scope": claim_scope,
        "task_name": "click_bell",
        "expert_data_num": 50,
        "paper_table_eligible": False,
        "act_rollouts_started": 0,
        "query": {"text": query, "sha256": canonical_sha256(query)},
        "candidate_suite": {
            "template_ids": candidates,
            "aspect_ids": aspect_ids,
            "first_aspect_id": aspect_ids[0],
            "sha256": canonical_sha256(candidates),
        },
        "trusted_contract": trusted_contract,
        "base_commit": base_commit,
        "checkpoint": {
            "setting": checkpoint_setting,
            "files": checkpoints,
            "file_set_sha256": canonical_sha256(checkpoints),
        },
        "telemetry": {
            "profile_id": telemetry_id,
            "profile": telemetry,
            "profile_sha256": telemetry_profile_sha256(telemetry),
        },
        "sample_schedule": {
            "mode": "matched_candidate_universe_n1",
            "entries": schedule,
            "sha256": canonical_sha256(schedule),
        },
        "source_artifacts": {
            "files": sources,
            "file_set_sha256": canonical_sha256(sources),
        },
        "limitations": [
            "This preregistration starts zero ACT rollouts and contains no result.",
            "N=1 can exercise the efficiency mechanism but cannot estimate Table 2 consistency.",
            "A dynamic strategy may stop before exhausting this matched candidate universe.",
        ],
    }
    manifest["integrity"] = {
        "algorithm": "sha256",
        "canonical_payload_sha256": canonical_sha256(manifest),
    }
    validate_evidence_manifest(root, manifest)
    return manifest


def _validate_recorded_files(
    root: Path, entries: Any, *, label: str, checkpoint: bool = False
) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        raise EvidenceManifestError(f"{label}.files must be a non-empty list")
    actual: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise EvidenceManifestError(f"{label}.files[{index}] must be an object")
        expected_keys = (
            {"path", "resolved_path", "symlink_chain", "size_bytes", "sha256"}
            if checkpoint
            else {"path", "size_bytes", "sha256"}
        )
        _exact_keys(entry, expected_keys, f"{label}.files[{index}]")
        path = entry["path"]
        if path in seen:
            raise EvidenceManifestError(f"{label} contains duplicate path: {path}")
        seen.add(path)
        identity = (
            _checkpoint_file_identity(root, path, label=f"{label}.files[{index}]")
            if checkpoint
            else _file_identity(root, path, label=f"{label}.files[{index}]")
        )
        if entry != identity:
            raise EvidenceManifestError(f"{label} artifact changed: {path}")
        actual.append(identity)
    return actual


def validate_evidence_manifest(
    repo_root: str | Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed if manifest content, paths, or pinned files have changed."""

    root = Path(repo_root).expanduser().resolve()
    if not isinstance(manifest, Mapping):
        raise EvidenceManifestError("evidence manifest must be an object")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "manifest_type",
            "registration_id",
            "prepared_at",
            "claim_scope",
            "task_name",
            "expert_data_num",
            "paper_table_eligible",
            "act_rollouts_started",
            "query",
            "candidate_suite",
            "trusted_contract",
            "base_commit",
            "checkpoint",
            "telemetry",
            "sample_schedule",
            "source_artifacts",
            "limitations",
            "integrity",
        },
        "evidence manifest",
    )
    if manifest["schema_version"] != 1 or manifest["manifest_type"] != "mea_evidence_preregistration_v1":
        raise EvidenceManifestError("unsupported evidence manifest schema/type")
    _identifier(manifest["registration_id"], label="registration_id")
    _text(manifest["claim_scope"], label="claim_scope")
    if manifest["task_name"] != "click_bell" or manifest["expert_data_num"] != 50:
        raise EvidenceManifestError("registered task/expert_data_num contract changed")
    try:
        datetime.fromisoformat(str(manifest["prepared_at"]))
    except ValueError as exc:
        raise EvidenceManifestError("prepared_at must be an ISO-8601 timestamp") from exc
    if manifest["paper_table_eligible"] is not False or manifest["act_rollouts_started"] != 0:
        raise EvidenceManifestError("preregistration cannot claim paper eligibility or ACT execution")
    if not isinstance(manifest["limitations"], list) or not manifest["limitations"]:
        raise EvidenceManifestError("limitations must be a non-empty list")

    query = manifest["query"]
    if not isinstance(query, dict):
        raise EvidenceManifestError("query must be an object")
    _exact_keys(query, {"text", "sha256"}, "query")
    query_text = _text(query["text"], label="query.text")
    if query["sha256"] != canonical_sha256(query_text):
        raise EvidenceManifestError("query hash mismatch")

    suite = manifest["candidate_suite"]
    if not isinstance(suite, dict):
        raise EvidenceManifestError("candidate_suite must be an object")
    _exact_keys(
        suite,
        {"template_ids", "aspect_ids", "first_aspect_id", "sha256"},
        "candidate_suite",
    )
    candidates = _candidate_suite(suite["template_ids"])
    aspects = _expanded_aspects(candidates)
    if suite["aspect_ids"] != aspects or suite["first_aspect_id"] != aspects[0]:
        raise EvidenceManifestError("candidate suite aspect expansion/order mismatch")
    if suite["sha256"] != canonical_sha256(candidates):
        raise EvidenceManifestError("candidate suite hash mismatch")

    trusted_contract = manifest["trusted_contract"]
    if not isinstance(trusted_contract, dict):
        raise EvidenceManifestError("trusted_contract must be an object")
    _exact_keys(
        trusted_contract,
        {"catalog_sha256", "click_bell_task_sha256", "template_contract_sha256"},
        "trusted_contract",
    )
    if trusted_contract != trusted_click_bell_contract(root):
        raise EvidenceManifestError("trusted catalog/template contract changed")

    base_commit = str(manifest["base_commit"])
    if not _HEX_COMMIT.fullmatch(base_commit):
        raise EvidenceManifestError("invalid registered base_commit")
    current_head = _git_head(root)
    if current_head != base_commit:
        raise EvidenceManifestError("registered base_commit no longer matches Git HEAD")

    checkpoint = manifest["checkpoint"]
    if not isinstance(checkpoint, dict):
        raise EvidenceManifestError("checkpoint must be an object")
    _exact_keys(checkpoint, {"setting", "files", "file_set_sha256"}, "checkpoint")
    if checkpoint["setting"] != "demo_clean":
        raise EvidenceManifestError("registered checkpoint setting changed")
    checkpoint_files = _validate_recorded_files(
        root, checkpoint["files"], label="checkpoint", checkpoint=True
    )
    if [item["path"] for item in checkpoint_files] != _CHECKPOINT_FILES:
        raise EvidenceManifestError("registered checkpoint paths changed")
    if checkpoint["file_set_sha256"] != canonical_sha256(checkpoint_files):
        raise EvidenceManifestError("checkpoint file-set hash mismatch")

    telemetry = manifest["telemetry"]
    if not isinstance(telemetry, dict):
        raise EvidenceManifestError("telemetry must be an object")
    _exact_keys(telemetry, {"profile_id", "profile", "profile_sha256"}, "telemetry")
    profile_id = _identifier(telemetry["profile_id"], label="telemetry.profile_id")
    try:
        trusted_profile = load_telemetry_profile(profile_id)
    except ValueError as exc:
        raise EvidenceManifestError(str(exc)) from exc
    if telemetry["profile"] != trusted_profile:
        raise EvidenceManifestError("registered telemetry profile differs from trusted code")
    if telemetry["profile_sha256"] != telemetry_profile_sha256(trusted_profile):
        raise EvidenceManifestError("telemetry profile hash mismatch")

    schedule = manifest["sample_schedule"]
    if not isinstance(schedule, dict):
        raise EvidenceManifestError("sample_schedule must be an object")
    _exact_keys(schedule, {"mode", "entries", "sha256"}, "sample_schedule")
    if schedule["mode"] != "matched_candidate_universe_n1":
        raise EvidenceManifestError("unsupported sample schedule mode")
    entries = _schedule(schedule["entries"], candidates)
    if schedule["sha256"] != canonical_sha256(entries):
        raise EvidenceManifestError("sample schedule hash mismatch")

    sources = manifest["source_artifacts"]
    if not isinstance(sources, dict):
        raise EvidenceManifestError("source_artifacts must be an object")
    _exact_keys(sources, {"files", "file_set_sha256"}, "source_artifacts")
    source_files = _validate_recorded_files(root, sources["files"], label="source_artifacts")
    if sources["file_set_sha256"] != canonical_sha256(source_files):
        raise EvidenceManifestError("source artifact file-set hash mismatch")
    if {item["path"] for item in checkpoint_files} & {item["path"] for item in source_files}:
        raise EvidenceManifestError("checkpoint and source artifact paths overlap")

    integrity = manifest["integrity"]
    if not isinstance(integrity, dict):
        raise EvidenceManifestError("integrity must be an object")
    _exact_keys(integrity, {"algorithm", "canonical_payload_sha256"}, "integrity")
    if integrity["algorithm"] != "sha256" or not _HEX_SHA256.fullmatch(
        str(integrity["canonical_payload_sha256"])
    ):
        raise EvidenceManifestError("invalid integrity record")
    payload = dict(manifest)
    payload.pop("integrity")
    digest = canonical_sha256(payload)
    if integrity["canonical_payload_sha256"] != digest:
        raise EvidenceManifestError("manifest payload hash mismatch")

    return {
        "schema_version": 1,
        "status": "passed",
        "registration_id": manifest["registration_id"],
        "manifest_payload_sha256": digest,
        "checkpoint_file_count": len(checkpoint_files),
        "source_artifact_count": len(source_files),
        "scheduled_candidate_count": len(candidates),
        "scheduled_strategy_count": len(_STRATEGIES),
        "act_rollouts_started": 0,
        "paper_table_eligible": False,
    }


__all__ = [
    "EvidenceManifestError",
    "canonical_sha256",
    "prepare_evidence_manifest",
    "read_repo_json",
    "require_clean_git_head",
    "trusted_click_bell_contract",
    "validate_evidence_manifest",
]
