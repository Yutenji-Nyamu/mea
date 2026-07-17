"""Cached/fault-injected functional gate report with zero new ACT calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_TOOLGEN_DIR = (
    "mea/evaluation_runs/eval_20260717_click_bell_toolgen_cached_smoke_v2/"
    "execution/round_1/planned_tool/generated"
)


class MicroAblationError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MicroAblationError(f"cannot read cached artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MicroAblationError(f"cached artifact must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cached_micro_ablation(
    repo_root: str | Path,
    *,
    taskgen_acceptance: Mapping[str, Any] | None = None,
    toolgen_dir: str = DEFAULT_TOOLGEN_DIR,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    if taskgen_acceptance is None:
        from mea.taskgen.acceptance import build_cached_taskgen_acceptance

        taskgen_acceptance = build_cached_taskgen_acceptance(root)
    checks = taskgen_acceptance.get("checks")
    if not isinstance(checks, Mapping):
        raise MicroAblationError("TaskGen acceptance has no checks")
    reflection = checks.get("scene_error_visual_reject_diagnose_repair")
    rag = checks.get("bbh_true_codegen_and_retrieval_provenance")
    if not isinstance(reflection, Mapping) or not isinstance(rag, Mapping):
        raise MicroAblationError("TaskGen acceptance is missing reflection/RAG slices")
    reflection_evidence = reflection.get("evidence")
    rag_evidence = rag.get("evidence")
    if not isinstance(reflection_evidence, Mapping) or not isinstance(
        rag_evidence, Mapping
    ):
        raise MicroAblationError("TaskGen acceptance evidence is malformed")

    generated = (root / toolgen_dir).resolve()
    if not generated.is_relative_to(root):
        raise MicroAblationError("ToolGen artifact path escapes repo")
    manifest_path = generated / "manifest.json"
    first_path = generated / "attempts/attempt_0/validation.json"
    second_path = generated / "attempts/attempt_1/validation.json"
    registration_path = generated / "registration.json"
    manifest = _read(manifest_path)
    first = _read(first_path)
    second = _read(second_path)
    registration = _read(registration_path)

    taskgen_complete = bool(
        reflection.get("passed") is True
        and reflection_evidence.get("static_pass") is True
        and reflection_evidence.get("visual_reject") is True
        and reflection_evidence.get("repair_installed") is True
        and reflection_evidence.get("visual_pass") is True
    )
    without_visual_false_accept = bool(
        reflection_evidence.get("static_pass") is True
        and reflection_evidence.get("visual_reject") is True
    )
    toolgen_complete = bool(
        manifest.get("status") == "passed"
        and manifest.get("successful_attempt") == 1
        and first.get("valid") is False
        and second.get("valid") is True
        and registration.get("status") == "validated"
    )
    without_validation_false_accept = bool(
        first.get("valid") is False
        and (generated / "attempts/attempt_0/generated_tool.py").is_file()
    )
    rag_provenance = bool(
        rag.get("passed") is True
        and rag_evidence.get("task_source_provenance_valid") is True
        and rag_evidence.get("knowledge_document_provenance_valid") is True
    )
    artifact_hashes = {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in (manifest_path, first_path, second_path, registration_path)
    }
    rows = [
        {
            "setting": "taskgen_complete_cached",
            "evidence_kind": "cached_real_artifact",
            "passed": taskgen_complete,
            "counts_toward_functional_gate_summary": True,
            "observed_effect": "visual reject -> repair -> accepted scene",
        },
        {
            "setting": "taskgen_without_visual_gate_fault_injection",
            "evidence_kind": "deterministic_fault_injection",
            "passed": without_visual_false_accept,
            "counts_toward_functional_gate_summary": True,
            "counterfactual_outcome": "invalid scene would be accepted after structural gates",
        },
        {
            "setting": "toolgen_complete_cached",
            "evidence_kind": "cached_real_artifact",
            "passed": toolgen_complete,
            "counts_toward_functional_gate_summary": True,
            "observed_effect": "invalid attempt rejected -> corrected attempt validated and registered",
        },
        {
            "setting": "toolgen_without_validation_fault_injection",
            "evidence_kind": "deterministic_fault_injection",
            "passed": without_validation_false_accept,
            "counts_toward_functional_gate_summary": True,
            "counterfactual_outcome": "AST-invalid generated source would reach registration",
        },
        {
            "setting": "taskgen_rag_provenance_only",
            "evidence_kind": "cached_real_artifact",
            "passed": rag_provenance,
            "counts_toward_functional_gate_summary": False,
            "observed_effect": None,
            "unavailable_reason": "no matched no-RAG generation artifact",
        },
    ]
    functional_rows = [
        row for row in rows if row["counts_toward_functional_gate_summary"]
    ]
    provenance_rows = [
        row for row in rows if not row["counts_toward_functional_gate_summary"]
    ]
    return {
        "schema_version": 1,
        "protocol": "cached_generation_micro_ablation_v1",
        "status": "completed",
        "claim_scope": "functional gate effect smoke, not generation success rates",
        "paper_table_eligible": False,
        "table3_success_rates": None,
        "runtime": {
            "provider_called": False,
            "simulator_called": False,
            "act_rollouts_started": 0,
        },
        "historical_artifacts_include_provider_or_simulator_calls": True,
        "functional_gate_checks": {
            "passed": sum(row["passed"] for row in functional_rows),
            "total": len(functional_rows),
            "all_passed": all(row["passed"] for row in functional_rows),
        },
        "provenance_checks": {
            "passed": sum(row["passed"] for row in provenance_rows),
            "total": len(provenance_rows),
            "all_passed": all(row["passed"] for row in provenance_rows),
            "ablation_effect_estimate": None,
        },
        "rows": rows,
        "toolgen_artifacts": artifact_hashes,
        "limitations": [
            "Cached and fault-injected rows do not sample a generation distribution.",
            "RAG provenance alone cannot estimate the effect of removing RAG.",
            "Development-agent review is not independent human review.",
        ],
    }


__all__ = ["MicroAblationError", "build_cached_micro_ablation"]
