#!/usr/bin/env python3
"""Build the zero-new-rollout cached generation functional-gate report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.micro_ablation import (
    DEFAULT_TOOLGEN_DIR,
    MicroAblationError,
    build_cached_micro_ablation,
)


def _report(summary: dict) -> str:
    rows = [
        "# Cached Generation Functional-Gate Smoke",
        "",
        f"- status: `{summary['status']}`",
        f"- claim scope: `{summary['claim_scope']}`",
        f"- paper-table eligible: `{str(summary['paper_table_eligible']).lower()}`",
        "- new ACT rollouts: `0`",
        f"- functional gate checks: `{summary['functional_gate_checks']['passed']}/"
        f"{summary['functional_gate_checks']['total']}`",
        f"- provenance checks: `{summary['provenance_checks']['passed']}/"
        f"{summary['provenance_checks']['total']}` (no ablation effect estimate)",
        "",
        "| Setting | Evidence | Passed | Functional summary |",
        "|---|---|---:|---:|",
    ]
    rows.extend(
        f"| {item['setting']} | {item['evidence_kind']} | "
        f"{str(item['passed']).lower()} | "
        f"{str(item['counts_toward_functional_gate_summary']).lower()} |"
        for item in summary["rows"]
    )
    rows.extend(
        [
            "",
            "Cached and deterministic fault-injection evidence is a functional "
            "smoke only; it is not a paper Table 3 generation-rate estimate.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--toolgen-dir", default=DEFAULT_TOOLGEN_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve()
    if not destination.is_relative_to(root):
        raise SystemExit("--output-dir must stay inside --repo-root")
    if destination.exists():
        raise SystemExit(f"output directory already exists: {destination}")
    try:
        summary = build_cached_micro_ablation(
            root,
            toolgen_dir=args.toolgen_dir,
        )
    except MicroAblationError as exc:
        raise SystemExit(str(exc)) from exc
    destination.mkdir(parents=True)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "report.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
