"""Download the pinned Hy-VLA RoboTwin snapshot on the AutoDL server.

This is the exact bounded downloader used for the 2026-08-01 deployment.  It
is intentionally server-path specific so the cold deployment ledger remains
reproducible; it never runs as part of the MEA production method.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "tencent/Hy-Embodied-0.5-VLA-RoboTwin"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
DESTINATION = Path("/root/autodl-tmp/checkpoints/robotwin/hyvla_robotwin")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    revision = HfApi(endpoint=ENDPOINT).model_info(REPO_ID).sha
    print(f"RESOLVED_REVISION={revision}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=revision,
        endpoint=ENDPOINT,
        local_dir=DESTINATION,
        max_workers=4,
    )
    required = ("config.json", "model.safetensors", "norm_stats.pkl")
    missing = [name for name in required if not (DESTINATION / name).is_file()]
    if missing:
        raise RuntimeError(f"missing checkpoint artifacts: {missing}")
    manifest = {
        "schema_version": 1,
        "repo_id": REPO_ID,
        "revision": revision,
        "endpoint": ENDPOINT,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            name: (DESTINATION / name).stat().st_size for name in required
        },
        "total_file_bytes": sum(
            path.stat().st_size
            for path in DESTINATION.rglob("*")
            if path.is_file()
        ),
    }
    (DESTINATION / "mea_download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
