"""Lightweight validated Task artifact index for the generic TaskGen path."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from mea.artifact_registry import ArtifactRegistry, ArtifactRegistryError


class GenericTaskArtifactError(RuntimeError):
    """Raised when an indexed generic Task artifact is stale or inconsistent."""


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _repo_file(repo_root: Path, value: str, *, label: str) -> Path:
    path = (repo_root / value).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise GenericTaskArtifactError(f"{label} escapes the repository") from exc
    if not path.is_file():
        raise GenericTaskArtifactError(f"{label} is unavailable: {value}")
    return path


def _read_generation_proposal(
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Read the canonical Proposal, falling back to the historical filename."""

    raw_paths = [
        manifest.get("proposal_path"),
        "generation/proposal.json",
        manifest.get("experiment_candidate_path"),
        "generation/experiment_candidate.json",
    ]
    seen: set[str] = set()
    for raw in raw_paths:
        if not isinstance(raw, str) or not raw.strip() or raw in seen:
            continue
        seen.add(raw)
        path = (run_dir / raw).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError as exc:
            raise GenericTaskArtifactError(
                "generic Task Proposal path escapes its run directory"
            ) from exc
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GenericTaskArtifactError(
                "generic Task Proposal artifact is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise GenericTaskArtifactError(
                "generic Task Proposal artifact must contain an object"
            )
        return value, path
    raise GenericTaskArtifactError(
        "generic Task Proposal artifact is unavailable"
    )


class GenericTaskArtifactIndex:
    """Adapt ArtifactRegistry to GenericRoboTwinTaskGenBackend exact lookup."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        index_path: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.registry = ArtifactRegistry(
            Path(index_path).expanduser().resolve()
            if index_path is not None
            else self.repo_root / "mea/artifacts/index.json"
        )

    def find_exact(
        self,
        lookup: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        semantic_key = lookup.get("semantic_key")
        semantic_hash = lookup.get("semantic_key_sha256")
        if not isinstance(semantic_key, Mapping) or not isinstance(
            semantic_hash, str
        ):
            raise GenericTaskArtifactError("generic Task lookup is malformed")
        if _canonical_sha256(semantic_key) != semantic_hash:
            raise GenericTaskArtifactError("generic Task semantic hash differs")
        try:
            entry = self.registry.find(
                kind="task",
                semantic_key=semantic_key,
            )
        except ArtifactRegistryError as exc:
            raise GenericTaskArtifactError(str(exc)) from exc
        if entry is None:
            return None
        validation = entry["validation"]
        if (
            validation.get("status") != "passed"
            or validation.get("semantic_key_sha256") != semantic_hash
            or validation.get("scene_change_passed") is not True
        ):
            raise GenericTaskArtifactError(
                "indexed generic Task lacks passing semantic validation"
            )
        manifest_path = _repo_file(
            self.repo_root,
            entry["artifact_path"],
            label="indexed generic Task manifest",
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GenericTaskArtifactError(
                "indexed generic Task manifest is invalid"
            ) from exc
        if (
            manifest.get("generation_kind")
            != "generic_provider_scene_checker_codegen"
            or manifest.get("candidate_module_sha256")
            != validation.get("module_sha256")
            or manifest.get("task_module") != validation.get("task_module")
        ):
            raise GenericTaskArtifactError(
                "indexed generic Task manifest differs from its validation"
            )
        _read_generation_proposal(manifest_path.parent, manifest)
        task_path = manifest_path.parent / "task.py"
        if (
            not task_path.is_file()
            or hashlib.sha256(task_path.read_bytes()).hexdigest()
            != validation.get("module_sha256")
        ):
            raise GenericTaskArtifactError(
                "indexed generic Task module differs from its validation"
            )
        candidate_manifest_path = manifest_path.parent / "candidate_manifest.json"
        try:
            candidate_manifest = json.loads(
                candidate_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise GenericTaskArtifactError(
                "indexed generic Task candidate manifest is invalid"
            ) from exc
        if (
            candidate_manifest.get("module_sha256")
            != validation.get("module_sha256")
            or candidate_manifest.get("task_module")
            != validation.get("task_module")
        ):
            raise GenericTaskArtifactError(
                "indexed generic Task candidate manifest differs"
            )
        return {
            "schema_version": 1,
            "status": "approved",
            "semantic_key": deepcopy(dict(semantic_key)),
            "semantic_key_sha256": semantic_hash,
            "artifact_id": entry["artifact_path"],
            "artifact_manifest": entry["artifact_path"],
            "reuse_count": entry["reuse_count"],
        }

    def mark_reuse(
        self,
        semantic_key: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            entry = self.registry.mark_reuse(
                kind="task",
                semantic_key=semantic_key,
            )
        except ArtifactRegistryError as exc:
            raise GenericTaskArtifactError(str(exc)) from exc
        if entry is None:
            raise GenericTaskArtifactError(
                "materialized generic Task disappeared from the artifact index"
            )
        return entry

    def register_generated(
        self,
        *,
        resolution: Mapping[str, Any],
        manifest_path: str | Path,
        source_query: str,
    ) -> dict[str, Any]:
        if resolution.get("status") != "generated":
            raise GenericTaskArtifactError(
                "only a generated generic Task can be registered"
            )
        semantic_key = resolution.get("semantic_key")
        semantic_hash = resolution.get("semantic_key_sha256")
        if not isinstance(semantic_key, Mapping) or not isinstance(
            semantic_hash, str
        ):
            raise GenericTaskArtifactError(
                "generated generic Task lacks a semantic identity"
            )
        path = Path(manifest_path).expanduser().resolve()
        try:
            relative = path.relative_to(self.repo_root).as_posix()
        except ValueError as exc:
            raise GenericTaskArtifactError(
                "generic Task manifest must be inside the repository"
            ) from exc
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _read_generation_proposal(path.parent, manifest)
        task_path = path.parent / "task.py"
        candidate_manifest_path = path.parent / "candidate_manifest.json"
        try:
            candidate_manifest = json.loads(
                candidate_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise GenericTaskArtifactError(
                "generic Task candidate manifest is invalid"
            ) from exc
        actual_module_sha256 = (
            hashlib.sha256(task_path.read_bytes()).hexdigest()
            if task_path.is_file()
            else None
        )
        preflight = (
            (manifest.get("scene_validation") or {}).get(
                "generic_preflight"
            )
            or {}
        )
        fixtures = preflight.get("checker_fixtures") or []
        validation = {
            "status": "passed",
            "semantic_key_sha256": semantic_hash,
            "module_sha256": manifest.get("candidate_module_sha256"),
            "task_module": manifest.get("task_module"),
            "checker_fixture_count": len(fixtures),
            "scene_change_passed": preflight.get("scene_change_passed"),
        }
        if (
            not validation["module_sha256"]
            or actual_module_sha256 != validation["module_sha256"]
            or candidate_manifest.get("module_sha256")
            != validation["module_sha256"]
            or candidate_manifest.get("task_module")
            != validation["task_module"]
            or not validation["task_module"]
            or not fixtures
            or any(item.get("passed") is not True for item in fixtures)
            or validation["scene_change_passed"] is not True
        ):
            raise GenericTaskArtifactError(
                "generic Task cannot be indexed before all validation passes"
            )
        try:
            return self.registry.register(
                kind="task",
                semantic_key=semantic_key,
                artifact_path=relative,
                validation=validation,
                source_query=source_query,
            )
        except ArtifactRegistryError as exc:
            raise GenericTaskArtifactError(str(exc)) from exc


def materialize_reused_generic_task(
    repo_root: str | Path,
    *,
    run_id: str,
    user_request: str,
    candidate: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy validated generation evidence into a clean run-local envelope."""

    root = Path(repo_root).expanduser().resolve()
    exact = resolution.get("exact_match")
    if not isinstance(exact, Mapping):
        raise GenericTaskArtifactError("generic Task reuse lacks an exact match")
    source_manifest_path = _repo_file(
        root,
        str(exact.get("artifact_manifest") or ""),
        label="generic Task reuse manifest",
    )
    source_dir = source_manifest_path.parent
    destination = root / "mea/generated_tasks" / run_id
    if destination.exists():
        raise GenericTaskArtifactError(
            f"generic Task reuse destination already exists: {destination}"
        )
    destination.mkdir(parents=True)
    for name in ("generation",):
        source = source_dir / name
        if source.is_dir():
            shutil.copytree(source, destination / name)
    for name in ("validation", "evidence", "evaluation"):
        (destination / name).mkdir()
    for name in (
        "task.py",
        "__init__.py",
        "overlay.yml",
        "candidate_manifest.json",
    ):
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    try:
        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenericTaskArtifactError(
            "generic Task reuse source manifest is invalid"
        ) from exc
    _read_generation_proposal(source_dir, manifest)
    preserved = (
        "schema_version",
        "generation_kind",
        "mode",
        "task_name",
        "telemetry_profile",
        "static_validation",
        "vision_validation",
        "taskgen_ablation_switches",
        "taskgen_prompt_components",
        "checker_contract",
        "candidate_module_sha256",
        "candidate_manifest",
        "task_artifact_summary",
    )
    manifest = {
        key: deepcopy(manifest[key])
        for key in preserved
        if key in manifest
    }
    manifest.update(
        {
            "run_id": run_id,
            "status": "generated",
            "created_at": datetime.now().astimezone().isoformat(),
            "user_request": user_request,
            "task_module": f"mea.generated_tasks.{run_id}.task",
            "overlay": (
                f"mea/generated_tasks/{run_id}/overlay.yml"
            ),
            "proposal": deepcopy(dict(candidate)),
            "proposal_path": "generation/proposal.json",
            "generic_taskgen_resolution": (
                "generic_taskgen_resolution.json"
            ),
            "artifact_reuse": {
                "route": "exact_generated_task_reuse",
                "source_manifest": exact["artifact_manifest"],
                "reuse_count": exact.get("reuse_count"),
                "provider_called": False,
            },
        }
    )
    manifest["provider"] = {
        "model_requested": None,
        "called": False,
        "provider_call_count": 0,
        "local_regeneration_count": 0,
        "reuse_source_manifest": exact["artifact_manifest"],
    }
    manifest["task_module"] = f"mea.generated_tasks.{run_id}.task"
    candidate_manifest_path = destination / "candidate_manifest.json"
    if candidate_manifest_path.is_file():
        candidate_manifest = json.loads(
            candidate_manifest_path.read_text(encoding="utf-8")
        )
        candidate_manifest["run_id"] = run_id
        candidate_manifest["task_module"] = manifest["task_module"]
        candidate_manifest.setdefault("codegen_provenance", {})[
            "reused_from_manifest"
        ] = exact["artifact_manifest"]
        candidate_manifest_path.write_text(
            json.dumps(candidate_manifest, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    manifest["task_generation_acceptance"] = {
        "status": "pending_current_seed_revalidation",
        "scope": "exact_code_reuse_requires_fresh_simulator_acceptance",
        "act_rollouts_started_before_acceptance": 0,
    }
    generation_dir = destination / "generation"
    generation_dir.mkdir(exist_ok=True)
    (generation_dir / "experiment_candidate.json").unlink(missing_ok=True)
    (generation_dir / "proposal.json").write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "generic_taskgen_resolution.json").write_text(
        json.dumps(resolution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "GenericTaskArtifactError",
    "GenericTaskArtifactIndex",
    "materialize_reused_generic_task",
]
