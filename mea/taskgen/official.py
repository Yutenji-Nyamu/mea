"""Package an unchanged, schema-backed RoboTwin task for MEA execution.

This is deliberately not a TaskGen model call.  It creates the same run
envelope used by generated variants while preserving the official task source
and recording that no code generation took place.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mea.toolkit import load_task_schema
from mea.taskgen.artifacts import write_task_artifact_bundle


class OfficialTaskRunError(RuntimeError):
    """Raised when an official task cannot satisfy the bounded run contract."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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


def _module_source(repo_root: Path, task_module: str) -> Path:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", task_module):
        raise OfficialTaskRunError(f"invalid task_module: {task_module!r}")
    source = repo_root.joinpath(*task_module.split(".")).with_suffix(".py")
    if not source.is_file():
        raise OfficialTaskRunError(f"task module source does not exist: {source}")
    return source


def create_official_task_run(
    repo_root: str | Path,
    user_request: str,
    *,
    task_name: str,
    task_module: str | None = None,
    run_id: str | None = None,
    telemetry_profile: str = "balanced_v1",
) -> dict[str, Any]:
    """Create an auditable run envelope for one unchanged official task."""

    root = Path(repo_root).expanduser().resolve()
    request = str(user_request).strip()
    if not request:
        raise OfficialTaskRunError("user_request must be non-empty")
    schema = load_task_schema(root, task_name)
    module_name = task_module or f"envs.{task_name}"
    source = _module_source(root, module_name)
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError) as exc:
        raise OfficialTaskRunError(f"cannot parse official task source: {exc}") from exc
    class_found = any(
        isinstance(node, ast.ClassDef) and node.name == task_name
        for node in tree.body
    )
    if not class_found:
        raise OfficialTaskRunError(
            f"task class {task_name!r} was not found in {module_name!r}"
        )

    resolved_run_id = run_id or _make_run_id()
    if not re.fullmatch(r"run_[A-Za-z0-9_]+", resolved_run_id):
        raise OfficialTaskRunError(
            "run_id must be a Python package name beginning with 'run_'"
        )
    run_dir = root / "mea/generated_tasks" / resolved_run_id
    if run_dir.exists():
        raise OfficialTaskRunError(f"run directory already exists: {run_dir}")
    for child in ("generation", "validation", "evidence", "evaluation"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    (run_dir / "__init__.py").write_text("", encoding="utf-8")
    (run_dir / "overlay.yml").write_text("{}\n", encoding="utf-8")

    source_relative = str(source.relative_to(root)).replace("\\", "/")
    schema_relative = (
        f"mea/toolkit/schemas/{task_name}.json"
    )
    variant_spec = {
        "schema_version": 1,
        "task_name": task_name,
        "intent": "evaluate_official_task_unchanged",
        "generation_mode": "official",
        "changes": {},
        "preserve": ["official_task_source", "official_task_identity"],
    }
    static_validation = {
        "official_passthrough": {
            "valid": True,
            "task_class_found": True,
            "task_module": module_name,
            "source": source_relative,
        },
        "task_schema": {
            "valid": True,
            "schema_version": schema["schema_version"],
            "task_family": schema.get("task_family"),
            "source": schema_relative,
        },
        "code_generation": {
            "performed": False,
            "reason": "official route preserves the upstream task unchanged",
        },
    }
    manifest = {
        "schema_version": 1,
        "run_id": resolved_run_id,
        "status": "generated",
        "created_at": datetime.now().astimezone().isoformat(),
        "user_request": request,
        "task_name": task_name,
        "task_module": module_name,
        "mode": "official",
        "generation_kind": "official_passthrough",
        "base_commit": _git_head(root),
        "overlay": str((run_dir / "overlay.yml").relative_to(root)).replace(
            "\\", "/"
        ),
        "telemetry_profile": telemetry_profile,
        "static_validation": static_validation,
        "task_retrieval": None,
        "knowledge_retrieval": None,
        "provider": {
            "called": False,
            "reason": "official task bypass does not invoke TaskGen",
        },
    }
    _write_json(run_dir / "request.json", {"user_request": request})
    _write_json(run_dir / "variant_spec.json", variant_spec)
    _write_json(run_dir / "validation/static.json", static_validation)
    _write_json(
        run_dir / "generation/official_source.json",
        {
            "task_name": task_name,
            "task_module": module_name,
            "source": source_relative,
            "task_schema": schema_relative,
            "code_generation_performed": False,
        },
    )
    bundle = write_task_artifact_bundle(root, run_dir, manifest)
    manifest["task_artifact_bundle"] = "generation/task_artifact_bundle.json"
    manifest["scene_check_spec"] = "generation/scene_check_spec.json"
    manifest["task_artifact_summary"] = {
        "scene_origin": bundle["scene_method"]["origin"],
        "success_origin": bundle["success_method"]["origin"],
        "success_semantics_preserved": True,
    }
    _write_json(run_dir / "manifest.json", manifest)
    return manifest
