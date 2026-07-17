#!/usr/bin/env python3
"""Review and install generated Tools without granting automatic trust."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.toolgen import (
    ReviewedRegistryError,
    build_review_manifest_template,
    install_reviewed_registration,
    public_reviewed_registration_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-approved review template or install an explicitly "
            "approved, exact-hash generated Tool into persistent storage."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser(
        "template",
        help="Print a pending manifest; this command never approves a Tool.",
    )
    template.add_argument("--source-registry", type=Path, required=True)
    template.add_argument("--registration-id", required=True)

    install = subparsers.add_parser(
        "install",
        help="Install only when a separate manifest explicitly says approved.",
    )
    install.add_argument("--source-registry", type=Path, required=True)
    install.add_argument("--registration-id", required=True)
    install.add_argument("--review-manifest", type=Path, required=True)
    install.add_argument("--reviewed-registry", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "template":
            result = build_review_manifest_template(
                args.source_registry, args.registration_id
            )
        else:
            match = install_reviewed_registration(
                args.source_registry,
                args.registration_id,
                args.review_manifest,
                args.reviewed_registry,
            )
            result = {
                "status": "installed",
                "registration": public_reviewed_registration_summary(match),
                "artifacts": {
                    "registry": str(match["registry_dir"] / "index.json"),
                    "registration": str(match["registration_path"]),
                    "generated_tool": str(match["source_path"]),
                    "tool_spec": str(match["tool_spec_path"]),
                    "review_manifest": str(match["review_manifest_path"]),
                },
            }
    except ReviewedRegistryError as exc:
        raise SystemExit(f"reviewed registry error: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
