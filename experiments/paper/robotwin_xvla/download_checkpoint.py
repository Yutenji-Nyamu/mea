"""Download and hash the pinned X-VLA RoboTwin snapshot on AutoDL."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "2toINF/X-VLA-RoboTwin2"
REVISION = "a157c580cfe6f9f445614490f3bec1b2f9ef9f18"
DESTINATION = Path("/root/autodl-tmp/checkpoints/robotwin/xvla_robotwin2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        endpoint=os.environ.get("HF_ENDPOINT"),
        local_dir=DESTINATION,
        max_workers=4,
    )
    required = ("config.json", "model.safetensors")
    missing = [name for name in required if not (DESTINATION / name).is_file()]
    if missing:
        raise RuntimeError(f"missing X-VLA checkpoint files: {missing}")
    files = {
        name: {
            "bytes": (DESTINATION / name).stat().st_size,
            "sha256": _sha256(DESTINATION / name),
        }
        for name in required
    }
    manifest = {
        "schema_version": 1,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (DESTINATION / "mea_download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
