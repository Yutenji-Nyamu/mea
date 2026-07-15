"""Subprocess entry point for one generated offline Tool execution."""

from __future__ import annotations

import argparse
import json
import sys

from .prototype import _execute_generated_tool_in_process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--tool-name", required=True)
    args = parser.parse_args()
    source = sys.stdin.read()
    try:
        result = _execute_generated_tool_in_process(
            source,
            args.episode_dir,
            tool_name=args.tool_name,
        )
        message = {"ok": True, "result": result}
        status = 0
    except BaseException as exc:
        message = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        status = 1
    sys.stdout.write(json.dumps(message, ensure_ascii=False))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
