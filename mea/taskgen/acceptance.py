"""Read-only acceptance checks over cached TaskGen evidence.

This module intentionally imports no provider, simulator, or policy runtime.  It
only validates previously recorded artifacts, so a smoke run is cheap and its
claim boundary is explicit: functional wiring, not a fresh experiment.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


class TaskGenAcceptanceError(RuntimeError):
    """Raised when a cached acceptance artifact is missing or malformed."""


DEFAULT_ACCEPTANCE_RUNS = {
    "official_reuse": "run_20260716_click_bell_balanced_v1_round_1",
    "click_overlay": "run_20260717_click_bell_base0_probe_p0",
    "bbh_codegen": "run_20260715_telemetry_blue_seed100000",
    "scene_error_repair": "run_20260714_161109_visual_reflection_color",
}


def _run_dir(repo_root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"run_[A-Za-z0-9_]+", run_id):
        raise TaskGenAcceptanceError(f"invalid cached run id: {run_id!r}")
    path = repo_root / "mea/generated_tasks" / run_id
    if not path.is_dir():
        raise TaskGenAcceptanceError(f"cached run does not exist: {path}")
    return path


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TaskGenAcceptanceError(f"required artifact does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskGenAcceptanceError(f"cannot read cached JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TaskGenAcceptanceError(f"cached JSON must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_ref(repo_root: Path, run_dir: Path, paths: list[str]) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for relative in paths:
        path = run_dir / relative
        if not path.is_file():
            raise TaskGenAcceptanceError(f"required artifact does not exist: {path}")
        hashes[relative] = _sha256(path)
    return {
        "run_id": run_dir.name,
        "path": str(run_dir.relative_to(repo_root)).replace("\\", "/"),
        "file_sha256": hashes,
    }


def _check(passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "evidence": evidence}


def _official_reuse_check(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    manifest = _read_object(run_dir / "manifest.json")
    static = _read_object(run_dir / "validation/static.json")
    provider = manifest.get("provider") or {}
    official = static.get("official_passthrough") or {}
    codegen = static.get("code_generation") or {}
    passed = all(
        (
            manifest.get("mode") == "official",
            manifest.get("generation_kind") == "official_passthrough",
            official.get("valid") is True,
            codegen.get("performed") is False,
            provider.get("called") is False,
            not (run_dir / "evaluation/act.json").exists(),
        )
    )
    return _check(
        passed,
        mode=manifest.get("mode"),
        generation_kind=manifest.get("generation_kind"),
        task_module=manifest.get("task_module"),
        code_generation_performed=codegen.get("performed"),
        original_provider_called=provider.get("called"),
        original_act_artifact_present=(run_dir / "evaluation/act.json").exists(),
        artifact=_artifact_ref(
            repo_root,
            run_dir,
            [
                "manifest.json",
                "validation/static.json",
                "generation/official_source.json",
            ],
        ),
    )


def _click_overlay_check(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    manifest = _read_object(run_dir / "manifest.json")
    variant_spec = _read_object(run_dir / "variant_spec.json")
    static = _read_object(run_dir / "validation/static.json")
    provider = manifest.get("provider") or {}
    bounded = static.get("bounded_overlay") or {}
    protected = static.get("protected_diff") or {}
    codegen = static.get("code_generation") or {}
    passed = all(
        (
            manifest.get("task_name") == "click_bell",
            manifest.get("mode") == "reuse",
            manifest.get("generation_kind") == "bounded_variant_overlay",
            variant_spec.get("task_name") == "click_bell",
            variant_spec.get("generation_mode") == "bounded_variant_overlay",
            variant_spec.get("controlled_axis") == bounded.get("controlled_axis"),
            bounded.get("valid") is True,
            protected.get("valid") is True,
            codegen.get("performed") is False,
            provider.get("called") is False,
            not (run_dir / "evaluation/act.json").exists(),
        )
    )
    return _check(
        passed,
        task_name=manifest.get("task_name"),
        mode=manifest.get("mode"),
        generation_kind=manifest.get("generation_kind"),
        variant_id=manifest.get("variant_id"),
        capability_id=manifest.get("capability_id"),
        controlled_axis=bounded.get("controlled_axis"),
        variant_spec_identity={
            "task_name": variant_spec.get("task_name"),
            "generation_mode": variant_spec.get("generation_mode"),
            "controlled_axis": variant_spec.get("controlled_axis"),
        },
        protected_files_unchanged=protected.get("valid"),
        code_generation_performed=codegen.get("performed"),
        original_provider_called=provider.get("called"),
        original_act_artifact_present=(run_dir / "evaluation/act.json").exists(),
        artifact=_artifact_ref(
            repo_root,
            run_dir,
            [
                "manifest.json",
                "variant_spec.json",
                "validation/static.json",
                "generation/bounded_overlay.json",
                "overlay.yml",
            ],
        ),
    )


def _bbh_codegen_provenance_check(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    manifest = _read_object(run_dir / "manifest.json")
    retrieval = _read_object(run_dir / "generation/retrieval.json")
    knowledge = _read_object(run_dir / "generation/knowledge_retrieval.json")
    static = _read_object(run_dir / "validation/static.json")
    calls = set(((manifest.get("provider") or {}).get("calls") or {}).keys())
    selected_tasks = retrieval.get("selected_tasks") or []
    selected_sources = retrieval.get("selected_sources") or []
    task_pairs_valid = (
        bool(selected_tasks)
        and len(selected_tasks) == len(selected_sources)
        and all(
            source == f"envs/{task}.py"
            for task, source in zip(selected_tasks, selected_sources)
        )
    )
    documents = knowledge.get("selected_documents") or []
    examples = knowledge.get("selected_examples") or []
    document_provenance_valid = bool(documents) and all(
        item.get("id")
        and item.get("path")
        and all(
            symbol.get("path") and symbol.get("symbol") and symbol.get("sha256")
            for symbol in item.get("source_symbols", [])
        )
        for item in documents
    )
    example_provenance_valid = all(
        item.get("id") and item.get("path") and item.get("symbol") for item in examples
    )
    manifest_task_retrieval = manifest.get("task_retrieval") or {}
    manifest_knowledge = manifest.get("knowledge_retrieval") or {}
    base_commit = str(manifest.get("base_commit") or "")
    static_ast = static.get("load_actors_ast") or {}
    protected = static.get("protected_diff") or {}
    passed = all(
        (
            manifest.get("task_name") == "beat_block_hammer",
            manifest.get("mode") == "force_codegen",
            str(manifest.get("task_module") or "").startswith("mea.generated_tasks."),
            {"proposal", "retrieval", "codegen"}.issubset(calls),
            bool(re.fullmatch(r"[0-9a-f]{40}", base_commit)),
            task_pairs_valid,
            manifest_task_retrieval.get("selected_sources") == selected_sources,
            knowledge.get("committed_index_current") is True,
            manifest_knowledge.get("selected_ids") == knowledge.get("selected_ids"),
            document_provenance_valid,
            example_provenance_valid,
            static_ast.get("valid") is True,
            static_ast.get("complete_method_generated") is True,
            protected.get("valid") is True,
            (run_dir / "task.py").is_file(),
            (run_dir / "generation/code_prompt.md").is_file(),
        )
    )
    return _check(
        passed,
        mode=manifest.get("mode"),
        task_module=manifest.get("task_module"),
        base_commit=base_commit,
        original_provider_calls=sorted(calls),
        selected_tasks=selected_tasks,
        selected_sources=selected_sources,
        selected_knowledge_ids=knowledge.get("selected_ids"),
        committed_index_current=knowledge.get("committed_index_current"),
        task_source_provenance_valid=task_pairs_valid,
        knowledge_document_provenance_valid=document_provenance_valid,
        knowledge_example_provenance_valid=example_provenance_valid,
        complete_method_generated=static_ast.get("complete_method_generated"),
        protected_files_unchanged=protected.get("valid"),
        original_act_artifact_present=(run_dir / "evaluation/act.json").exists(),
        artifact=_artifact_ref(
            repo_root,
            run_dir,
            [
                "manifest.json",
                "variant_spec.json",
                "generation/retrieval.json",
                "generation/knowledge_retrieval.json",
                "generation/code_prompt.md",
                "generation/code_response.txt",
                "validation/static.json",
                "task.py",
            ],
        ),
    )


def _scene_error_repair_check(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    manifest = _read_object(run_dir / "manifest.json")
    fixture = _read_object(run_dir / "reflection/fixture/fixture.json")
    summary = _read_object(run_dir / "reflection/summary.json")
    attempts = summary.get("attempts") or []
    first = attempts[0] if len(attempts) >= 1 else {}
    second = attempts[1] if len(attempts) >= 2 else {}
    first_observation = first.get("observation") or {}
    first_vision = first_observation.get("vision") or {}
    repair = first.get("repair") or {}
    repair_static = repair.get("static_validation") or {}
    second_observation = second.get("observation") or {}
    second_vision = second_observation.get("vision") or {}
    diagnosis = str(first_vision.get("diagnosis") or "").strip()
    unexpected = first_vision.get("unexpected_changes") or []
    static_pass = bool(
        fixture.get("injected_method_structurally_valid") is True
        and fixture.get("injected_after_normal_static_gate") is True
    )
    visual_reject = bool(
        first_observation.get("probe_passed") is True
        and first_observation.get("passed") is False
        and first_vision.get("passed") is False
        and diagnosis
        and unexpected
    )
    repair_pass = bool(
        repair.get("installed") is True
        and repair.get("method_sha256_before")
        and repair.get("method_sha256_after")
        and repair.get("method_sha256_before") != repair.get("method_sha256_after")
        and (repair_static.get("load_actors_ast") or {}).get("valid") is True
        and (repair_static.get("protected_diff") or {}).get("valid") is True
    )
    visual_pass = bool(
        second_observation.get("probe_passed") is True
        and second_observation.get("passed") is True
        and second_vision.get("passed") is True
    )
    no_act_artifact = not (run_dir / "evaluation/act.json").exists()
    passed = all(
        (
            manifest.get("task_name") == "beat_block_hammer",
            manifest.get("status") == "completed_without_act",
            fixture.get("fixture") == "wrong_color",
            fixture.get("test_only") is True,
            static_pass,
            visual_reject,
            repair_pass,
            visual_pass,
            summary.get("passed") is True,
            summary.get("repairs_used") == 1,
            summary.get("final_attempt") == 1,
            no_act_artifact,
        )
    )
    return _check(
        passed,
        fixture=fixture.get("fixture"),
        test_only=fixture.get("test_only"),
        transition=[
            "static_pass",
            "visual_reject",
            "diagnosis",
            "repair_installed",
            "static_revalidate_pass",
            "visual_pass",
        ],
        static_pass=static_pass,
        visual_reject=visual_reject,
        diagnosis_present=bool(diagnosis),
        unexpected_changes=unexpected,
        repair_installed=repair.get("installed"),
        repair_static_pass=repair_pass,
        visual_pass=visual_pass,
        repairs_used=summary.get("repairs_used"),
        original_act_artifact_present=not no_act_artifact,
        artifact=_artifact_ref(
            repo_root,
            run_dir,
            [
                "manifest.json",
                "reflection/fixture/fixture.json",
                "reflection/summary.json",
                "reflection/attempt_00/vision.json",
                "reflection/attempt_00/repair.json",
                "reflection/attempt_01/vision.json",
            ],
        ),
    )


def build_cached_taskgen_acceptance(
    repo_root: str | Path,
    *,
    official_run_id: str = DEFAULT_ACCEPTANCE_RUNS["official_reuse"],
    overlay_run_id: str = DEFAULT_ACCEPTANCE_RUNS["click_overlay"],
    codegen_run_id: str = DEFAULT_ACCEPTANCE_RUNS["bbh_codegen"],
    reflection_run_id: str = DEFAULT_ACCEPTANCE_RUNS["scene_error_repair"],
) -> dict[str, Any]:
    """Validate four cached TaskGen slices without invoking any runtime."""

    root = Path(repo_root).expanduser().resolve()
    checks = {
        "official_reuse": _official_reuse_check(root, _run_dir(root, official_run_id)),
        "click_overlay": _click_overlay_check(root, _run_dir(root, overlay_run_id)),
        "bbh_true_codegen_and_retrieval_provenance": (
            _bbh_codegen_provenance_check(root, _run_dir(root, codegen_run_id))
        ),
        "scene_error_visual_reject_diagnose_repair": _scene_error_repair_check(
            root, _run_dir(root, reflection_run_id)
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "schema_version": 1,
        "kind": "taskgen_cached_functional_acceptance_v1",
        "passed": passed,
        "cached_artifact": True,
        "no_provider": True,
        "no_simulator": True,
        "no_ACT": True,
        "paper_table_eligible": False,
        "claim_scope": "cached functional acceptance only",
        "runtime": {
            "mode": "read_only_cached_artifacts",
            "provider_called": False,
            "simulator_called": False,
            "act_called": False,
            "network_called": False,
        },
        "historical_sources_may_contain_runtime_evidence": True,
        "checks": checks,
    }


def build_scene_error_repair_acceptance(
    repo_root: str | Path,
    *,
    reflection_run_id: str,
) -> dict[str, Any]:
    """Validate one real render/diagnose/repair run without unrelated caches."""

    root = Path(repo_root).expanduser().resolve()
    check = _scene_error_repair_check(
        root, _run_dir(root, reflection_run_id)
    )
    return {
        "schema_version": 1,
        "kind": "taskgen_scene_error_repair_acceptance_v1",
        "passed": bool(check["passed"]),
        "reflection_run_id": reflection_run_id,
        "cached_artifact": True,
        "no_provider": True,
        "no_simulator": True,
        "no_ACT": True,
        "paper_table_eligible": False,
        "claim_scope": (
            "read-only acceptance of one historical real simulator and provider repair run"
        ),
        "runtime": {
            "mode": "read_only_single_run",
            "provider_called": False,
            "simulator_called": False,
            "act_called": False,
            "network_called": False,
        },
        "historical_source_contains_runtime_evidence": True,
        "check": check,
    }
