#!/usr/bin/env python3
"""Validate and summarize the 20-query unreviewed aspect draft."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.query_dataset import QueryDatasetError, summarize_query_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = json.loads(args.dataset.read_text(encoding="utf-8"))
        summary = summarize_query_dataset(value)
    except (OSError, json.JSONDecodeError, QueryDatasetError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
