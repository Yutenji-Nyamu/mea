#!/usr/bin/env python3
"""Audit the 16 top-down paper-method claims without starting runtime work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.method_coverage import (
    STATUS_IMPLEMENTED,
    build_method_coverage_report,
    render_method_coverage_markdown,
)


def _output_path(root: Path, value: Path) -> Path:
    candidate = value.expanduser()
    path = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    if not path.is_relative_to(root):
        raise ValueError("output path must stay inside --repo-root")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--require-all-implemented",
        action="store_true",
        help="Return exit status 2 while any claim is partial or evidence_pending.",
    )
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    try:
        report = build_method_coverage_report(root)
        if args.output is not None:
            output = _output_path(root, args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.markdown is not None:
            markdown = _output_path(root, args.markdown)
            markdown.parent.mkdir(parents=True, exist_ok=True)
            markdown.write_text(
                render_method_coverage_markdown(report), encoding="utf-8"
            )
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_all_implemented and any(
        item["status"] != STATUS_IMPLEMENTED for item in report["claims"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
