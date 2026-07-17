#!/usr/bin/env python3
"""Prepare or validate a zero-ACT, hash-pinned evidence manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.evidence_manifest import (
    EvidenceManifestError,
    prepare_evidence_manifest,
    read_repo_json,
    validate_evidence_manifest,
)


def _relative(root: Path, value: Path, *, label: str, must_exist: bool) -> str:
    candidate = value.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        lexical = candidate.absolute()
        lexical_relative = lexical.relative_to(root)
    except ValueError as exc:
        raise EvidenceManifestError(f"{label} must stay inside --repo-root") from exc
    cursor = root
    for part in lexical_relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise EvidenceManifestError(f"{label} may not traverse a symlink")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise EvidenceManifestError(f"cannot resolve {label}: {exc}") from exc
    if not resolved.is_relative_to(root):
        raise EvidenceManifestError(f"{label} must stay inside --repo-root")
    relative = resolved.relative_to(root).as_posix()
    if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        raise EvidenceManifestError(f"invalid {label}")
    return relative


def _write_new(root: Path, output: Path, value: dict) -> Path:
    relative = _relative(root, output, label="--output", must_exist=False)
    destination = root / relative
    if destination.exists():
        raise EvidenceManifestError(f"--output already exists: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Recheck after creating parents so a pre-existing symlink cannot be used.
    _relative(root, destination, label="--output", must_exist=False)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise EvidenceManifestError(f"temporary output already exists: {temporary.name}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.expanduser().resolve()
    try:
        if args.command == "prepare":
            config_relative = _relative(
                root, args.config, label="--config", must_exist=True
            )
            config = read_repo_json(
                root, config_relative, label="preregistration config"
            )
            manifest = prepare_evidence_manifest(root, config)
            destination = _write_new(root, args.output, manifest)
            # Validate the serialized bytes, not only the in-memory object.
            serialized = read_repo_json(
                root,
                destination.relative_to(root).as_posix(),
                label="evidence manifest",
            )
            result = validate_evidence_manifest(root, serialized)
            result["manifest_path"] = destination.relative_to(root).as_posix()
        else:
            manifest_relative = _relative(
                root, args.manifest, label="--manifest", must_exist=True
            )
            manifest = read_repo_json(root, manifest_relative, label="evidence manifest")
            result = validate_evidence_manifest(root, manifest)
            result["manifest_path"] = manifest_relative
    except EvidenceManifestError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
