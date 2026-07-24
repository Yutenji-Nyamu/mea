#!/usr/bin/env python3
"""Write the read-only ACT+DP exact-seed pilot readiness report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.act_dp_pilot import build_act_dp_readiness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    report = build_act_dp_readiness(
        root,
        seeds=args.seeds if args.seeds is not None else (100600, 100601, 100602),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
        return
    output = args.output
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
