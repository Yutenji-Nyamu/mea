"""Build or verify the compact MEA Offline Extractor index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.knowledge import build_knowledge_index, build_knowledge_index_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    target = root / "mea/knowledge/index.json"
    expected = build_knowledge_index_data(root)
    if args.check:
        if not target.is_file():
            raise SystemExit("knowledge index 不存在，请先运行 build_mea_knowledge.py")
        actual = json.loads(target.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("knowledge index 已过期，请重新生成")
        print(f"knowledge index current: {target}")
        return
    print(build_knowledge_index(root, target))


if __name__ == "__main__":
    main()
