"""Declarative, bounded TaskGen family for click_bell property probes."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mea.toolkit import load_task_schema


CLICK_BELL_PROTECTED_PATHS = (
    "envs/click_bell.py",
    "policy/ACT/eval.sh",
    "script/eval_policy.py",
)


class ClickBellTaskGenError(RuntimeError):
    """Raised when a click_bell bounded variant violates its contract."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {
        relative: _sha256(repo_root / relative)
        for relative in CLICK_BELL_PROTECTED_PATHS
    }


def _git_head(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _make_run_id() -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex[:8]}"


def validate_click_bell_variant_hint(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"bell"}:
        raise ClickBellTaskGenError("variant_hint must contain only bell")
    bell = value.get("bell")
    if not isinstance(bell, dict):
        raise ClickBellTaskGenError("variant_hint.bell must be an object")

    fields = set(bell)
    position_fields = {"position_mode", "xy"}
    instance_fields = {"position_mode", "instance_mode", "bell_id"}
    if fields == position_fields:
        if bell.get("position_mode") != "fixed":
            raise ClickBellTaskGenError(
                "position variants require position_mode=fixed"
            )
        xy = bell.get("xy")
        if not isinstance(xy, list) or len(xy) != 2:
            raise ClickBellTaskGenError(
                "variant_hint.bell.xy must contain two numbers"
            )
        try:
            x, y = (float(xy[0]), float(xy[1]))
        except (TypeError, ValueError) as exc:
            raise ClickBellTaskGenError("bell xy values must be numeric") from exc
        if not all(math.isfinite(item) for item in (x, y)):
            raise ClickBellTaskGenError("bell xy values must be finite")
        if not (-0.25 <= x <= 0.25) or abs(x) < 0.05:
            raise ClickBellTaskGenError(
                "bell x must be in the official safe side ranges"
            )
        if not (-0.2 <= y <= 0.0):
            raise ClickBellTaskGenError("bell y must be inside the official range")
        return {"bell": {"position_mode": "fixed", "xy": [x, y]}}

    if fields == instance_fields:
        if bell.get("position_mode") != "official_random":
            raise ClickBellTaskGenError(
                "instance variants must preserve official_random position"
            )
        if bell.get("instance_mode") != "fixed":
            raise ClickBellTaskGenError(
                "instance variants require instance_mode=fixed"
            )
        bell_id = bell.get("bell_id")
        if isinstance(bell_id, bool) or not isinstance(bell_id, int):
            raise ClickBellTaskGenError("bell_id must be integer 0 or 1")
        if bell_id not in {0, 1}:
            raise ClickBellTaskGenError("bell_id must be integer 0 or 1")
        return {
            "bell": {
                "position_mode": "official_random",
                "instance_mode": "fixed",
                "bell_id": bell_id,
            }
        }

    raise ClickBellTaskGenError(
        "variant_hint.bell must be either a strict fixed-position or "
        "fixed-instance contract"
    )


def compile_click_bell_overlay(variant_hint: Any) -> dict[str, Any]:
    normalized = validate_click_bell_variant_hint(variant_hint)
    return {"mea": {"enabled": True, "bell": normalized["bell"]}}


def create_click_bell_variant_run(
    repo_root: str | Path,
    user_request: str,
    *,
    variant_hint: Any,
    run_id: str | None = None,
    telemetry_profile: str = "balanced_v1",
) -> dict[str, Any]:
    """Package a safe single-axis overlay without pretending it is codegen."""

    root = Path(repo_root).expanduser().resolve()
    request = str(user_request).strip()
    if not request:
        raise ClickBellTaskGenError("user_request must be non-empty")
    normalized = validate_click_bell_variant_hint(variant_hint)
    schema = load_task_schema(root, "click_bell")
    resolved_run_id = run_id or _make_run_id()
    if not re.fullmatch(r"run_[A-Za-z0-9_]+", resolved_run_id):
        raise ClickBellTaskGenError(
            "run_id must be a Python package name beginning with 'run_'"
        )
    run_dir = root / "mea/generated_tasks" / resolved_run_id
    if run_dir.exists():
        raise ClickBellTaskGenError(f"run directory already exists: {run_dir}")
    for child in ("generation", "validation", "evidence", "evaluation"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    (run_dir / "__init__.py").write_text("", encoding="utf-8")

    protected_before = _protected_hashes(root)
    overlay = compile_click_bell_overlay(normalized)
    (run_dir / "overlay.yml").write_text(
        yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    bell_change = normalized["bell"]
    controlled_axis = (
        "object_instance"
        if bell_change.get("instance_mode") == "fixed"
        else "object_position"
    )
    spec = {
        "schema_version": 1,
        "task_name": "click_bell",
        "intent": f"evaluate_bell_{controlled_axis}_generalization",
        "controlled_axis": controlled_axis,
        "generation_mode": "bounded_variant_overlay",
        "changes": normalized,
        "preserve": [
            "official_pose_rng_consumption",
            "official_instance_rng_consumption",
            "official_bell_assets",
            "play_once",
            "check_success",
            "checkpoint",
        ],
    }
    protected_after = _protected_hashes(root)
    static_validation = {
        "variant_spec": {"valid": True, "normalized": normalized},
        "task_schema": {
            "valid": True,
            "schema_version": schema["schema_version"],
            "task_family": schema.get("task_family"),
        },
        "bounded_overlay": {
            "valid": True,
            "task_module": "mea.tasks.click_bell",
            "controlled_axis": controlled_axis,
            "variant_contract": normalized["bell"],
        },
        "protected_diff": {
            "valid": protected_after == protected_before,
            "hashes_after": protected_after,
        },
        "code_generation": {
            "performed": False,
            "reason": "validated declarative single-axis overlay",
        },
    }
    if not static_validation["protected_diff"]["valid"]:
        raise ClickBellTaskGenError("bounded TaskGen changed a protected file")

    manifest = {
        "schema_version": 1,
        "run_id": resolved_run_id,
        "status": "generated",
        "created_at": datetime.now().astimezone().isoformat(),
        "user_request": request,
        "task_name": "click_bell",
        "task_module": "mea.tasks.click_bell",
        "mode": "reuse",
        "generation_kind": "bounded_variant_overlay",
        "base_commit": _git_head(root),
        "protected_hashes_before": protected_before,
        "overlay": str((run_dir / "overlay.yml").relative_to(root)).replace(
            "\\", "/"
        ),
        "telemetry_profile": telemetry_profile,
        "static_validation": static_validation,
        "task_retrieval": None,
        "knowledge_retrieval": None,
        "provider": {
            "called": False,
            "reason": "bounded declarative TaskGen does not call a text model",
        },
    }
    _write_json(run_dir / "request.json", {"user_request": request})
    _write_json(run_dir / "variant_spec.json", spec)
    _write_json(run_dir / "validation/static.json", static_validation)
    _write_json(
        run_dir / "generation/bounded_overlay.json",
        {
            "generation_kind": "bounded_variant_overlay",
            "task_module": "mea.tasks.click_bell",
            "variant_hint": normalized,
            "code_generation_performed": False,
        },
    )
    _write_json(run_dir / "manifest.json", manifest)
    return manifest
