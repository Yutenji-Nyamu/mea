#!/usr/bin/env python3
"""Prepare or audit the bounded zero-ACT module-ablation protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.module_ablation_protocol import (
    ModuleAblationError,
    audit_module_ablation_artifacts,
    prepare_module_ablation_schedule,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ModuleAblationError(f"input must be a JSON object: {path}")
    return value


def _report(summary: dict) -> str:
    lines = [
        "# TaskGen/ToolGen Module Ablation Protocol",
        "",
        f"- protocol: `{summary['protocol']}`",
        f"- mode: `{summary['mode']}`",
        f"- status: `{summary['status']}`",
        f"- paper-table eligible: `{str(summary['paper_table_eligible']).lower()}`",
        "- calls made by this command: provider=`false`, simulator=`false`, ACT=`0`",
        "- claim boundary: functional-only; not a paper Table 3 result",
    ]
    if summary["mode"] == "prepare_only":
        lines.extend(
            [
                f"- scheduled artifacts: `{len(summary['items'])}`",
                "",
                "This is a preparation schedule only; it contains no experiment outcome.",
                "",
            ]
        )
        return "\n".join(lines)
    audit = summary["artifact_audit"]
    lines.extend(
        [
            f"- completed artifacts: `{audit['completed']}/{audit['scheduled']}`",
            f"- effect-eligible artifacts: `{audit['effect_eligible']}`",
            "- historical runtime: self-attested by completed manifests; not independently observed",
            "",
            "| Component | Reference | Module off | Matched | Effect |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in summary["comparisons"]:
        effect = row["effect"]
        rendered = (
            "null"
            if effect is None
            else str(effect["absolute_success_rate_difference"])
        )
        lines.append(
            f"| {row['component']} | {row['reference_condition']} | "
            f"{row['module_off_condition']} | "
            f"{row['eligible_matched_case_count']}/{row['scheduled_case_count']} | "
            f"{rendered} |"
        )
    lines.extend(
        [
            "",
            "Missing, incomplete, or provenance-only pairs keep the effect null.",
            "",
        ]
    )
    return "\n".join(lines)


def _destination(root: Path, output_dir: Path) -> Path:
    expanded = output_dir.expanduser()
    candidate = expanded if expanded.is_absolute() else root / expanded
    cursor = root
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise ModuleAblationError("--output-dir must stay inside --repo-root") from exc
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ModuleAblationError("--output-dir contains a symlink component")
    destination = candidate.resolve()
    if not destination.is_relative_to(root):
        raise ModuleAblationError("--output-dir must stay inside --repo-root")
    if destination.exists():
        raise ModuleAblationError(f"output directory already exists: {destination}")
    return destination


def _input(root: Path, path: Path, *, field: str) -> Path:
    expanded = path.expanduser()
    candidate = expanded if expanded.is_absolute() else root / expanded
    cursor = root
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise ModuleAblationError(f"{field} must stay inside --repo-root") from exc
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ModuleAblationError(f"{field} contains a symlink component")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ModuleAblationError(f"{field} must stay inside --repo-root")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--schedule", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    try:
        destination = _destination(root, args.output_dir)
        if args.command == "prepare":
            summary = prepare_module_ablation_schedule(
                root, _read(_input(root, args.config, field="--config"))
            )
            json_name = "schedule.json"
        else:
            summary = audit_module_ablation_artifacts(
                root, _read(_input(root, args.schedule, field="--schedule"))
            )
            json_name = "summary.json"
        destination.mkdir(parents=True)
        (destination / json_name).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (destination / "report.md").write_text(_report(summary), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, json.JSONDecodeError, ModuleAblationError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
