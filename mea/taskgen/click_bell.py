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
from mea.taskgen.capabilities import CapabilityError, build_variant_spec


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
    if not isinstance(value, dict):
        raise ClickBellTaskGenError("variant_hint must be an object")
    if set(value) == {"domain_randomization"}:
        randomization = value.get("domain_randomization")
        if not isinstance(randomization, dict):
            raise ClickBellTaskGenError("domain_randomization must be an object")
        fields = set(randomization)
        if fields == {"cluttered_table", "clean_background_rate"}:
            rate = randomization.get("clean_background_rate")
            if randomization.get("cluttered_table") is not True:
                raise ClickBellTaskGenError("cluttered_table must be true")
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise ClickBellTaskGenError("clean_background_rate must be numeric")
            if float(rate) != 0.0:
                raise ClickBellTaskGenError(
                    "scene clutter probes require clean_background_rate=0"
                )
            return {
                "domain_randomization": {
                    "cluttered_table": True,
                    "clean_background_rate": 0.0,
                }
            }
        if fields == {"random_background", "clean_background_rate"}:
            rate = randomization.get("clean_background_rate")
            if randomization.get("random_background") is not True:
                raise ClickBellTaskGenError("random_background must be true")
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise ClickBellTaskGenError("clean_background_rate must be numeric")
            if float(rate) != 0.0:
                raise ClickBellTaskGenError(
                    "background texture probes require clean_background_rate=0"
                )
            return {
                "domain_randomization": {
                    "random_background": True,
                    "clean_background_rate": 0.0,
                }
            }
        if fields == {"random_light", "crazy_random_light_rate"}:
            rate = randomization.get("crazy_random_light_rate")
            if randomization.get("random_light") is not True:
                raise ClickBellTaskGenError("random_light must be true")
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise ClickBellTaskGenError(
                    "crazy_random_light_rate must be numeric"
                )
            if float(rate) != 0.0:
                raise ClickBellTaskGenError(
                    "bounded lighting probes require crazy_random_light_rate=0"
                )
            return {
                "domain_randomization": {
                    "random_light": True,
                    "crazy_random_light_rate": 0.0,
                }
            }
        raise ClickBellTaskGenError(
            "domain_randomization must select exactly one trusted clutter, "
            "background-texture, or static-lighting axis"
        )
    if set(value) != {"bell"}:
        raise ClickBellTaskGenError(
            "variant_hint must contain only bell or domain_randomization"
        )
    bell = value.get("bell")
    if not isinstance(bell, dict):
        raise ClickBellTaskGenError("variant_hint.bell must be an object")

    fields = set(bell)
    position_fields = {"position_mode", "xy"}
    instance_fields = {"position_mode", "instance_mode", "bell_id"}
    if fields == position_fields:
        if bell.get("position_mode") != "fixed":
            raise ClickBellTaskGenError("position variants require position_mode=fixed")
        xy = bell.get("xy")
        if not isinstance(xy, list) or len(xy) != 2:
            raise ClickBellTaskGenError("variant_hint.bell.xy must contain two numbers")
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
            raise ClickBellTaskGenError("instance variants require instance_mode=fixed")
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
    if "bell" in normalized:
        return {"mea": {"enabled": True, "bell": normalized["bell"]}}
    return {"domain_randomization": normalized["domain_randomization"]}


def create_click_bell_variant_run(
    repo_root: str | Path,
    user_request: str,
    *,
    variant_hint: Any,
    variant_id: str | None = None,
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
    bell_change = normalized.get("bell")
    if bell_change is None:
        randomization = normalized["domain_randomization"]
        if "cluttered_table" in randomization:
            controlled_axis = "robustness.scene_clutter"
            capability_id = "robustness.scene_clutter"
            default_variant_id = "robustness.scene_clutter.official10"
        elif "random_background" in randomization:
            controlled_axis = "scene_background_texture"
            capability_id = "scene_background_texture"
            default_variant_id = "scene_background_texture.unseen"
        else:
            controlled_axis = "scene_lighting"
            capability_id = "scene_lighting"
            default_variant_id = "scene_lighting.static_random"
    else:
        controlled_axis = (
            "object_instance"
            if bell_change.get("instance_mode") == "fixed"
            else "object_position"
        )
        capability_id = {
            "object_position": "object_position.fixed_xy",
            "object_instance": "object_instance.official_id",
        }[controlled_axis]
        default_variant_id = (
            f"object_instance.base{bell_change['bell_id']}"
            if controlled_axis == "object_instance"
            else (
                "object_position.left_fixed"
                if bell_change["xy"][0] < 0
                else "object_position.right_fixed"
            )
        )
    resolved_variant_id = str(variant_id or default_variant_id)
    try:
        spec = build_variant_spec(
            task_name="click_bell",
            variant_id=resolved_variant_id,
            capability_id=capability_id,
            intent=f"evaluate_bell_{controlled_axis}_generalization",
            changes=normalized,
        )
    except CapabilityError as exc:
        raise ClickBellTaskGenError(str(exc)) from exc
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
            "variant_contract": normalized,
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
        "variant_id": resolved_variant_id,
        "capability_id": capability_id,
        "mode": "reuse",
        "generation_kind": "bounded_variant_overlay",
        "base_commit": _git_head(root),
        "protected_hashes_before": protected_before,
        "overlay": str((run_dir / "overlay.yml").relative_to(root)).replace("\\", "/"),
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
