"""Rebuild MEA's SQLite planning-history cache from completed evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.history import EvaluationHistoryDB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete cached rows before scanning canonical evaluation artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    evaluation_root = (
        args.evaluation_root.expanduser().resolve()
        if args.evaluation_root
        else repo_root / "mea/evaluation_runs"
    )
    database = (
        args.database.expanduser().resolve()
        if args.database
        else evaluation_root / "history.sqlite3"
    )
    result = EvaluationHistoryDB(database, repo_root=repo_root).rebuild(
        evaluation_root, reset=args.reset
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
