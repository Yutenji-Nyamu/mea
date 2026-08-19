"""Compatibility TaskGen CLI for frozen paper protocols and legacy callers.

The production method runtime materializes generic TaskGen candidates through
``mea.taskgen.runtime``.  This module preserves the historical BBH, ClickBell,
Table-3 and standalone CLI behavior while those paper
protocol callers remain supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from experiments.paper.compat_capability_adapter import (
    CapabilityAdapterError,
    taskgen_route,
    validate_capability_contract,
    validate_contract_changes,
)
from mea.providers import OpenAICompatibleProvider
from mea.proposals import ProposalError, validate_task_proposal
from mea.toolkit import evaluate_telemetry_root
from mea.toolkit import aggregate_tool_executions
from mea.taskgen import (
    BBHDistractorTaskGenError,
    ClickBellTaskGenError,
    TaskArtifactBundleError,
    TaskGenPrototype,
    VisualReflectionError,
    extract_json_response,
    execute_reflection_loop,
    repair_generated_method,
    create_click_bell_variant_run,
    create_official_task_run,
    validate_click_bell_variant_hint,
    build_variant_spec,
    validate_variant_spec_envelope,
    build_scene_check_spec,
    bbh_distractor_proposal_from_task_proposal,
    bbh_distractor_rollout_execution,
    build_bbh_distractor_module,
    materialize_bbh_distractor_candidate,
    validate_bbh_distractor_methods,
    validate_bbh_distractor_proposal,
    ClickBellDistractorTaskGenError,
    click_bell_distractor_from_task_proposal,
    click_bell_distractor_rollout_execution,
    materialize_click_bell_distractor_candidate,
    validate_scene_check_spec,
    write_task_artifact_bundle,
)
from mea.taskgen.act_runtime import (
    archive_previous_act_attempt,
    newest_eval_dir,
    run_act as run_act_runtime,
)
from mea.planner.experiment_candidate import (
    ExperimentCandidateError,
    validate_experiment_candidate,
)
from mea.taskgen.prototype import (
    TaskGenError,
    validate_taskgen_ablation_switches,
)
from experiments.paper.compat_taskgen.production_acceptance import (
    ProductionTaskAcceptanceError,
    record_production_task_acceptance,
    require_production_task_acceptance,
    require_task_artifact_act_runtime_eligible,
)
from mea.taskgen.reflection import protected_hashes
from mea.taskgen.runtime import (
    _checker_fixture_failure_diagnosis,
    _expert_terminal_authority_failure,
    _same_seed_tracked_actor_geometry,
    _same_seed_tracked_actor_state,
    _tracked_actor_heights,
    build_preservation_report,
    create_generic_provider_taskgen_run as _create_generic_provider_taskgen_run,
    record_generic_taskgen_generation_failure,
    run_command as _runtime_run_command,
    run_probe as _runtime_run_probe,
)
from mea.taskgen.rollout_evidence import (
    evaluate_generic_task_rollout_telemetry,
)
from mea.taskgen import visual_validation




def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_manifest(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    write_json(path, manifest)
    return manifest


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> int:
    """Compatibility wrapper preserving the historical CLI patch point."""

    return _runtime_run_command(command, cwd=cwd, log_path=log_path)


def run_probe(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    seed: int,
    episode_index: int = 0,
    expert: bool,
    scene_json: Path | None = None,
    image: Path | None = None,
    log_path: Path | None = None,
    raise_on_failure: bool = True,
    max_expert_attempts: int = 3,
    telemetry_dir: Path | None = None,
    telemetry_profile: str = "balanced_v1",
    visual_capture_profile_id: str | None = None,
    discover_task_context: bool = False,
    task_context: Path | None = None,
    action_dimension: int = 0,
) -> dict[str, Any]:
    """Compatibility wrapper with explicit command/JSON runner injection."""

    return _runtime_run_probe(
        repo_root,
        run_dir,
        manifest,
        seed=seed,
        episode_index=episode_index,
        expert=expert,
        scene_json=scene_json,
        image=image,
        log_path=log_path,
        raise_on_failure=raise_on_failure,
        max_expert_attempts=max_expert_attempts,
        telemetry_dir=telemetry_dir,
        telemetry_profile=telemetry_profile,
        visual_capture_profile_id=visual_capture_profile_id,
        discover_task_context=discover_task_context,
        task_context=task_context,
        action_dimension=action_dimension,
        command_runner=run_command,
        json_writer=write_json,
    )


def create_generic_provider_taskgen_run(
    repo_root: Path,
    *,
    user_request: str,
    provider: Any,
    model: str,
    vision_model: str,
    experiment_candidate: Mapping[str, Any],
    run_id: str,
    seed: int,
    telemetry_profile: str = "balanced_v1",
    action_dimension: int = 0,
    ablation_switches: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper preserving the script-level probe patch point."""

    return _create_generic_provider_taskgen_run(
        repo_root,
        user_request=user_request,
        provider=provider,
        model=model,
        vision_model=vision_model,
        experiment_candidate=experiment_candidate,
        run_id=run_id,
        seed=seed,
        telemetry_profile=telemetry_profile,
        action_dimension=action_dimension,
        ablation_switches=ablation_switches,
        probe_runner=run_probe,
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def task_artifact_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Expose success authority without relabeling experimental semantics."""

    semantics = bundle.get("success_semantics")
    if not isinstance(semantics, Mapping):
        raise TaskArtifactBundleError("TaskArtifactBundle success semantics are missing")
    authority = semantics.get("authority")
    experimental = authority == "compiled_success_spec_experimental_bounded"
    provider_checker = authority == "llm_generated_python_ast_validated"
    return {
        "scene_origin": bundle.get("scene_method", {}).get("origin"),
        "success_origin": bundle.get("success_method", {}).get("origin"),
        "success_semantics_preserved": bool(semantics.get("preserved")),
        "success_official_equivalent": not (
            experimental or provider_checker
        ),
        "success_compiler_eligible": not provider_checker,
        "success_act_eligible": bool(
            semantics.get("act_runtime_eligible", True)
        ),
        "success_execution_scope": (
            "provider_generated_checker"
            if provider_checker
            else "experimental_bounded_act"
            if experimental
            else "official_equivalent"
        ),
        "success_outcome_label": (
            semantics.get("outcome_label")
            if experimental or provider_checker
            else "official_check_success"
        ),
    }


def create_provider_scene_checker_taskgen_run(
    repo_root: Path,
    *,
    user_request: str,
    provider: Any,
    model: str,
    variant_spec: Mapping[str, Any],
    task_proposal: Mapping[str, Any],
    run_id: str | None = None,
    telemetry_profile: str = "balanced_v1",
    ablation_switches: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Adapt one registered provider scene+checker dialect to the run envelope."""

    request = str(user_request).strip()
    if not request:
        raise ValueError("user_request must be non-empty")
    proposal = validate_task_proposal(task_proposal)
    selected_base_task = proposal["task_name"]
    dialects = {
        "beat_block_hammer": {
            "error_type": BBHDistractorTaskGenError,
            "capability_id": "robustness.distractor_avoidance",
            "proposal_adapter": bbh_distractor_proposal_from_task_proposal,
            "materializer": materialize_bbh_distractor_candidate,
            "run_prefix": "run_bbh_distractor_",
            "proposal_artifact": "bbh_distractor_proposal.json",
            "static_artifact": "bbh_distractor_static.json",
        },
        "click_bell": {
            "error_type": ClickBellDistractorTaskGenError,
            "capability_id": "robustness.distractor_avoidance",
            "proposal_adapter": click_bell_distractor_from_task_proposal,
            "materializer": materialize_click_bell_distractor_candidate,
            "run_prefix": "run_click_bell_distractor_",
            "proposal_artifact": "click_bell_distractor_proposal.json",
            "static_artifact": "click_bell_distractor_static.json",
        },
    }
    dialect = dialects.get(selected_base_task)
    if dialect is None:
        raise ValueError(
            "provider scene+checker has no registered dialect for "
            f"{selected_base_task!r}"
        )
    error_type = dialect["error_type"]
    spec = validate_variant_spec_envelope(variant_spec)
    if (
        spec["task_name"] != selected_base_task
        or spec["capability_id"] != dialect["capability_id"]
        or spec["generation_mode"] != "provider_scene_checker_codegen"
    ):
        raise error_type(
            "VariantSpec is not the selected provider scene+checker capability"
        )
    bounded_proposal = dialect["proposal_adapter"](
        proposal, query=request
    )
    resolved_run_id = run_id or (
        dialect["run_prefix"]
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    materializer_options = (
        {"compatibility_attempt_directory": True}
        if selected_base_task == "click_bell"
        else {}
    )
    candidate = dialect["materializer"](
        repo_root=repo_root,
        run_id=resolved_run_id,
        proposal=bounded_proposal,
        provider=provider,
        model=model,
        max_regenerations=1,
        ablation_switches=ablation_switches,
        **materializer_options,
    )
    run_dir = repo_root / "mea/generated_tasks" / resolved_run_id
    for child in ("generation", "validation", "evidence", "evaluation"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    moves = {
        "proposal_prompt.md": "generation/code_prompt.md",
        "provider_response.txt": "generation/provider_response.txt",
        "proposal.json": f"generation/{dialect['proposal_artifact']}",
        "checker_fixtures.json": "validation/checker_fixtures.json",
        "provider_attempts": "generation/provider_attempts",
    }
    for source_name, destination_name in moves.items():
        source = run_dir / source_name
        destination = run_dir / destination_name
        if not source.exists():
            raise error_type(
                f"candidate artifact is missing: {source_name}"
            )
        shutil.move(str(source), str(destination))
    provider_response = extract_json_response(
        (run_dir / "generation/provider_response.txt").read_text(
            encoding="utf-8"
        )
    )
    write_json(
        run_dir / "generation/provider_response.json",
        provider_response,
    )
    write_json(run_dir / "variant_spec.json", spec)
    (run_dir / "overlay.yml").write_text("{}\n", encoding="utf-8")
    write_json(run_dir / "request.json", {"user_request": request})
    write_json(run_dir / "generation/task_proposal.json", proposal)
    attempts = json.loads(
        (
            run_dir / "generation/provider_attempts/attempts.json"
        ).read_text(encoding="utf-8")
    )
    static_validation = {
        "variant_spec": {"valid": True},
        "provider_scene_checker": {
            "valid": True,
            "ast_policy": candidate["codegen_provenance"]["ast_policy"],
            "model_written_python": True,
            "restricted_success_spec_compiler_used": False,
            "checker_fixture_count": candidate["checker_contract"][
                "fixture_count"
            ],
            "checker_fixture_pass_count": candidate["checker_contract"][
                "fixture_pass_count"
            ],
        },
    }
    write_json(
        run_dir / f"validation/{dialect['static_artifact']}",
        static_validation,
    )
    try:
        base_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        base_commit = None
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": resolved_run_id,
        "status": "generated",
        "created_at": datetime.now().astimezone().isoformat(),
        "user_request": request,
        "task_name": selected_base_task,
        "task_module": candidate["task_module"],
        "mode": "provider_scene_checker_codegen",
        "generation_kind": "provider_scene_checker_codegen",
        "base_commit": base_commit,
        "protected_hashes_before": protected_hashes(repo_root),
        "overlay": str((run_dir / "overlay.yml").relative_to(repo_root)).replace(
            "\\", "/"
        ),
        "telemetry_profile": telemetry_profile,
        "static_validation": static_validation,
        "task_retrieval": None,
        "knowledge_retrieval": None,
        "provider": {
            "model_requested": model,
            "called": True,
            "calls": {
                f"scene_checker_attempt_{index + 1}": (
                    item.get("provider_metadata") or {}
                )
                for index, item in enumerate(attempts)
            },
            "local_repair_count": candidate[
                "codegen_provenance"
            ]["local_repair_count"],
        },
        "taskgen_ablation_switches": candidate["codegen_provenance"][
            "taskgen_ablation_switches"
        ],
        "taskgen_prompt_components": candidate["codegen_provenance"][
            "prompt_components"
        ],
        "variant_spec_authority": "planner_capability_contract",
        "task_proposal": proposal,
        "task_proposal_path": "generation/task_proposal.json",
        "checker_contract": candidate["checker_contract"],
        "candidate_module_sha256": candidate["module_sha256"],
        "candidate_manifest": "candidate_manifest.json",
        "provider_validation_artifact": (
            f"validation/{dialect['static_artifact']}"
        ),
        "provider_proposal_artifact": (
            f"generation/{dialect['proposal_artifact']}"
        ),
    }
    write_json(run_dir / "manifest.json", manifest)
    bundle = write_task_artifact_bundle(
        repo_root,
        run_dir,
        manifest,
        task_proposal=proposal,
    )
    manifest.update(
        {
            "task_artifact_bundle": "generation/task_artifact_bundle.json",
            "scene_check_spec": "generation/scene_check_spec.json",
            "task_artifact_summary": task_artifact_summary(bundle),
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def prepare_planner_capability_binding(
    raw_contract: Any,
    *,
    task_name: str,
    mode: str,
    variant_id: str | None,
    task_proposal: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Fail closed before provider, simulator, or filesystem work begins."""

    try:
        contract = validate_capability_contract(raw_contract)
    except (CapabilityAdapterError, ValueError) as exc:
        raise RuntimeError(f"invalid planner capability contract: {exc}") from exc
    if contract["task_name"] != task_name:
        raise RuntimeError("planner capability task does not match --task-name")
    declared_route = taskgen_route(contract)
    if mode != declared_route:
        raise RuntimeError(
            f"TaskGen mode {mode!r} conflicts with capability route {declared_route!r}"
        )
    taskgen = contract["taskgen"]
    expected_variant = taskgen["task_variant_id"]
    proposal = None
    if task_proposal is not None:
        try:
            proposal = validate_task_proposal(
                task_proposal, expected_task_name=task_name
            )
            proposal["changes"] = validate_contract_changes(
                contract, proposal["changes"]
            )
        except (ProposalError, CapabilityAdapterError) as exc:
            raise RuntimeError(f"TaskProposal exceeds capability contract: {exc}") from exc
        if proposal["capability_id"] != taskgen["capability_id"]:
            raise RuntimeError("TaskProposal capability does not match planner contract")
        if proposal["aspect_id"] != contract["aspect"]["aspect_id"]:
            raise RuntimeError("TaskProposal aspect does not match planner contract")
        if (
            task_name == "click_bell"
            and taskgen["operation"] == "bounded_variant_overlay"
            and proposal["changes"]
        ):
            try:
                proposal["changes"] = validate_click_bell_variant_hint(
                    proposal["changes"]
                )
            except RuntimeError as exc:
                raise RuntimeError(f"invalid bounded click_bell proposal: {exc}") from exc
    if expected_variant is None:
        if mode != "official" or variant_id is not None:
            raise RuntimeError("official capability requires no task variant")
        return contract, None
    if proposal is not None:
        expected_variant = proposal["proposal_id"]
    if variant_id != expected_variant:
        raise RuntimeError("TaskGen variant id does not match planner task_variant_id")
    try:
        trusted_spec = build_variant_spec(
            task_name=task_name,
            variant_id=expected_variant,
            capability_id=taskgen["capability_id"],
            intent=(
                proposal["intent"]
                if proposal is not None
                else f"planner_capability:{contract['template_id']}"
            ),
            changes=(proposal["changes"] if proposal is not None else taskgen["changes"]),
            generation_mode=taskgen["generation_mode"],
            preserve_success_semantics=(
                proposal["preserve_success_semantics"]
                if proposal is not None
                else taskgen["operation"]
                != "provider_scene_checker_codegen"
            ),
        )
    except ValueError as exc:
        raise RuntimeError(f"planner capability cannot build VariantSpec: {exc}") from exc
    return contract, trusted_spec


def validate_planner_capability_binding(
    raw_contract: Any,
    *,
    task_name: str,
    mode: str,
    variant_id: str | None,
    run_dir: Path,
    task_proposal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a planner adapter contract to the materialized TaskGen artifact."""

    contract, trusted_spec = prepare_planner_capability_binding(
        raw_contract,
        task_name=task_name,
        mode=mode,
        variant_id=variant_id,
        task_proposal=task_proposal,
    )
    taskgen = contract["taskgen"]
    declared_route = taskgen_route(contract)
    expected_variant = (
        trusted_spec["variant_id"]
        if trusted_spec is not None
        else taskgen["task_variant_id"]
    )
    if expected_variant is None:
        manifest_path = run_dir / "manifest.json"
        spec_path = run_dir / "variant_spec.json"
        overlay_path = run_dir / "overlay.yml"
        try:
            materialized_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            overlay_text = overlay_path.read_text(encoding="utf-8").strip()
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid official TaskGen artifact: {exc}") from exc
        expected_spec = {
            "schema_version": 1,
            "task_name": task_name,
            "intent": "evaluate_official_task_unchanged",
            "generation_mode": "official",
            "changes": {},
            "preserve": ["official_task_source", "official_task_identity"],
        }
        official_static = (materialized_manifest.get("static_validation") or {}).get(
            "official_passthrough"
        ) or {}
        if (
            materialized_manifest.get("task_name") != task_name
            or materialized_manifest.get("task_module") != f"envs.{task_name}"
            or materialized_manifest.get("mode") != "official"
            or materialized_manifest.get("generation_kind") != "official_passthrough"
            or official_static.get("valid") is not True
            or official_static.get("task_module") != f"envs.{task_name}"
            or spec != expected_spec
            or overlay_text != "{}"
        ):
            raise RuntimeError(
                "official TaskGen artifact differs from capability passthrough"
            )
    else:
        spec_path = run_dir / "variant_spec.json"
        if not spec_path.is_file():
            raise RuntimeError("generated capability requires variant_spec.json")
        try:
            materialized_manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            loaded = json.loads(spec_path.read_text(encoding="utf-8"))
            spec = validate_variant_spec_envelope(loaded)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"invalid materialized VariantSpec: {exc}") from exc
        expected = {
            "task_name": task_name,
            "variant_id": expected_variant,
            "capability_id": trusted_spec["capability_id"],
            "controlled_axis": trusted_spec["controlled_axis"],
            "generation_mode": trusted_spec["generation_mode"],
            "changes": trusted_spec["changes"],
        }
        observed = {field: spec.get(field) for field in expected}
        if observed != expected:
            raise RuntimeError(
                "materialized VariantSpec differs from planner capability contract"
            )
        if (
            materialized_manifest.get("task_name") != task_name
            or materialized_manifest.get("mode") != mode
        ):
            raise RuntimeError(
                "materialized TaskGen manifest differs from capability invocation"
            )
        if taskgen["operation"] == "bounded_variant_overlay":
            if (
                materialized_manifest.get("generation_kind")
                != "bounded_variant_overlay"
                or materialized_manifest.get("task_module")
                != f"mea.tasks.{task_name}"
            ):
                raise RuntimeError(
                    "bounded TaskGen artifact differs from capability adapter"
                )
        elif taskgen["operation"] == "force_codegen":
            if (
                materialized_manifest.get("variant_spec_authority")
                != "planner_capability_contract"
                or not str(materialized_manifest.get("task_module") or "").startswith(
                    "mea.generated_tasks."
                )
            ):
                raise RuntimeError(
                    "code-generated TaskGen artifact lacks planner authority"
                )
        elif taskgen["operation"] == "provider_scene_checker_codegen":
            candidate = materialized_manifest.get("checker_contract")
            if (
                materialized_manifest.get("generation_kind")
                != "provider_scene_checker_codegen"
                or materialized_manifest.get("variant_spec_authority")
                != "planner_capability_contract"
                or not str(materialized_manifest.get("task_module") or "").startswith(
                    "mea.generated_tasks."
                )
                or not isinstance(candidate, Mapping)
                or candidate.get("official_success") is not False
                or candidate.get("authority")
                != "llm_generated_python_ast_validated"
            ):
                raise RuntimeError(
                    "provider scene+checker artifact lacks its non-official "
                    "planner/codegen authority"
                )
        elif taskgen["operation"] == "reuse_variant":
            if (
                materialized_manifest.get("variant_spec_authority")
                != "planner_capability_contract"
                or materialized_manifest.get("task_module")
                != f"mea.tasks.{task_name}"
            ):
                raise RuntimeError(
                    "reused TaskGen variant lacks trusted task/contract authority"
                )
    result = {
        "schema_version": 1,
        "status": "passed",
        "template_id": contract["template_id"],
        "task_variant_id": expected_variant,
        "declared_route": declared_route,
        "executed_route": mode,
        "variant_spec_authority": (
            "official_passthrough"
            if trusted_spec is None
            else (
                "planner_task_proposal"
                if task_proposal is not None
                else "planner_capability_contract"
            )
        ),
        "capability_contract_sha256": _canonical_sha256(contract),
        "variant_spec_sha256": _canonical_sha256(spec) if spec is not None else None,
        "materialized_task_variant_id": (
            spec.get("variant_id") if spec is not None else None
        ),
        "task_proposal_sha256": (
            _canonical_sha256(task_proposal)
            if task_proposal is not None
            else None
        ),
    }
    update_manifest(
        run_dir,
        capability_id=taskgen["capability_id"],
        capability_contract=contract,
        capability_contract_validation=result,
    )
    return result




def run_official_expert_episodes(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    start_seed: int,
    num_episodes: int,
    telemetry_profile: str,
    max_seed_candidates: int | None = None,
) -> dict[str, Any]:
    """Execute unchanged expert probes on solvable official-task seeds."""

    episode_summaries: list[dict[str, Any]] = []
    rejected_seeds: list[dict[str, Any]] = []
    first_scene: dict[str, Any] | None = None
    candidate_limit = max_seed_candidates or max(num_episodes * 10, num_episodes + 5)
    for candidate_index in range(candidate_limit):
        if len(episode_summaries) >= num_episodes:
            break
        episode_index = len(episode_summaries)
        seed = start_seed + candidate_index
        is_first = episode_index == 0
        scene = run_probe(
            repo_root,
            run_dir,
            manifest,
            seed=seed,
            episode_index=episode_index,
            expert=True,
            scene_json=(
                run_dir / "validation/scene.json"
                if is_first
                else run_dir
                / f"validation/official_episodes/episode_{episode_index:03d}_seed_{seed}.json"
            ),
            image=(
                run_dir / "evidence/initial_head.png"
                if is_first
                else run_dir
                / f"evidence/official_episodes/episode_{episode_index:03d}_seed_{seed}.png"
            ),
            log_path=(
                run_dir / "validation/probe.log"
                if is_first
                else run_dir
                / f"validation/official_episodes/episode_{episode_index:03d}_seed_{seed}.log"
            ),
            telemetry_dir=(
                run_dir
                / "evaluation/telemetry/expert"
                / f"episode_{episode_index:03d}_seed_{seed}"
            ),
            telemetry_profile=telemetry_profile,
            visual_capture_profile_id="event_keyframes_v1",
            raise_on_failure=False,
            max_expert_attempts=1,
        )
        returncode = int(scene.get("returncode", 0))
        if returncode != 0:
            error = scene.get("error") or {}
            if error.get("type") == "UnStableError":
                rejected_seeds.append(
                    {
                        "seed": seed,
                        "reason": "unstable_initial_scene",
                        "error_type": error.get("type"),
                        "message": error.get("message"),
                    }
                )
                continue
            if returncode == 2:
                rejected_seeds.append(
                    {
                        "seed": seed,
                        "reason": "expert_unsolvable",
                        "error_type": error.get("type"),
                        "message": error.get("message"),
                    }
                )
                continue
            raise RuntimeError(
                "official expert probe failed for "
                f"seed={seed}, returncode={returncode}: "
                f"{error.get('type') or 'unknown error'}"
            )
        if not bool(scene.get("expert", {}).get("passed")):
            rejected_seeds.append(
                {
                    "seed": seed,
                    "reason": "expert_unsolvable",
                    "error_type": None,
                    "message": "official expert did not satisfy check_success",
                }
            )
            continue
        if first_scene is None:
            first_scene = scene
        telemetry = scene.get("telemetry", {})
        telemetry_metadata = telemetry.get("metadata", {})
        video_artifact = telemetry_metadata.get("artifacts", {}).get("video")
        episode_summaries.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "setup_success": bool(scene.get("setup_success")),
                "render_success": bool(scene.get("render_success")),
                "rule_passed": bool(scene.get("rule_check", {}).get("passed")),
                "expert_passed": bool(scene.get("expert", {}).get("passed")),
                "image": scene.get("image"),
                "telemetry": telemetry.get("episode_dir"),
                "video": (
                    str(Path(telemetry["episode_dir"]) / video_artifact)
                    if telemetry.get("episode_dir") and video_artifact
                    else None
                ),
                "visual_capture": telemetry_metadata.get("visual_capture"),
            }
        )
    if first_scene is None or len(episode_summaries) < num_episodes:
        raise RuntimeError(
            "official expert seed scan exhausted before collecting "
            f"{num_episodes} episodes; accepted={len(episode_summaries)}, "
            f"rejected={len(rejected_seeds)}, candidates={candidate_limit}"
        )
    first_scene["expert_batch"] = {
        "passed": all(item["expert_passed"] for item in episode_summaries),
        "episode_count": len(episode_summaries),
        "candidate_count": len(episode_summaries) + len(rejected_seeds),
        "rejected_seed_count": len(rejected_seeds),
        "rejected_seeds": rejected_seeds,
        "episodes": episode_summaries,
    }
    write_json(run_dir / "validation/scene.json", first_scene)
    write_json(
        run_dir / "validation/official_expert_episodes.json",
        first_scene["expert_batch"],
    )
    return first_scene


def collect_position_samples(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    start_seed: int,
    num_episodes: int,
    first_scene: dict[str, Any] | None,
    require_expert: bool = True,
) -> dict[str, Any]:
    """Collect simulator-native block poses without inventing expert evidence."""

    sample_root = run_dir / "validation/position_samples"
    samples: list[dict[str, Any]] = []
    for episode_index in range(num_episodes):
        seed = start_seed + episode_index
        if episode_index == 0 and first_scene:
            scene = first_scene
        else:
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=seed,
                expert=require_expert,
                scene_json=sample_root / f"seed_{seed}.json",
                image=sample_root / f"seed_{seed}.png",
                log_path=sample_root / f"seed_{seed}.log",
            )
        position = scene.get("block_pose", {}).get("position")
        if not isinstance(position, list) or len(position) < 2:
            raise RuntimeError(f"seed={seed} 缺少 block_pose.position")
        samples.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "block_position": [float(value) for value in position],
                "block_quaternion": scene.get("block_pose", {}).get("quaternion"),
                "rule_passed": bool(scene.get("rule_check", {}).get("passed")),
                "expert_passed": (
                    bool(scene.get("expert", {}).get("passed"))
                    if require_expert
                    else None
                ),
                "image": scene.get("image"),
            }
        )

    xs = [item["block_position"][0] for item in samples]
    ys = [item["block_position"][1] for item in samples]
    unique_xy = {
        (round(item["block_position"][0], 8), round(item["block_position"][1], 8))
        for item in samples
    }
    result = {
        "start_seed": start_seed,
        "num_episodes": num_episodes,
        "expert_required": require_expert,
        "samples": samples,
        "metrics": {
            "unique_xy_count": len(unique_xy),
            "x_span": max(xs) - min(xs),
            "y_span": max(ys) - min(ys),
            "position_varied": len(unique_xy) > 1,
        },
        "passed": len(samples) == num_episodes
        and all(
            item["rule_passed"]
            and (not require_expert or item["expert_passed"])
            for item in samples
        ),
    }
    write_json(run_dir / "validation/position_samples.json", result)
    return result


def collect_click_bell_position_samples(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    start_seed: int,
    num_episodes: int,
    first_scene: dict[str, Any] | None,
) -> dict[str, Any]:
    """Verify the controlled click_bell axis and expert gate for each seed."""

    spec = json.loads((run_dir / "variant_spec.json").read_text(encoding="utf-8"))
    changes = spec["changes"]
    bell_change = changes.get("bell")
    randomization_change = changes.get("domain_randomization") or {}
    clutter_change = (
        randomization_change if "cluttered_table" in randomization_change else None
    )
    background_change = (
        randomization_change if "random_background" in randomization_change else None
    )
    lighting_change = (
        randomization_change if "random_light" in randomization_change else None
    )
    expected_xy = (
        [float(value) for value in bell_change["xy"]]
        if bell_change and bell_change.get("position_mode") == "fixed"
        else None
    )
    expected_bell_id = (
        int(bell_change["bell_id"])
        if bell_change and bell_change.get("instance_mode") == "fixed"
        else None
    )
    sample_root = run_dir / "validation/position_samples"
    samples: list[dict[str, Any]] = []
    for episode_index in range(num_episodes):
        seed = start_seed + episode_index
        if episode_index == 0 and first_scene:
            scene = first_scene
        else:
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=seed,
                expert=True,
                scene_json=sample_root / f"seed_{seed}.json",
                image=sample_root / f"seed_{seed}.png",
                log_path=sample_root / f"seed_{seed}.log",
            )
        variant_check = visual_validation.validate_click_bell_scene_contract(
            scene, spec
        )
        position_check = variant_check["position"]
        instance_check = variant_check["instance"]
        clutter_check = variant_check["clutter"]
        background_check = variant_check["background_texture"]
        lighting_check = variant_check["lighting"]
        samples.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "expected_xy": expected_xy,
                "bell_position": position_check.get("actual_xy"),
                "position_matched": bool(position_check.get("passed")),
                "position_authority": position_check.get("authority"),
                "expected_bell_id": expected_bell_id,
                "bell_id": instance_check.get("actual_bell_id"),
                "instance_matched": bool(instance_check.get("passed")),
                "instance_authority": instance_check.get("authority"),
                "clutter_expected": bool(clutter_change),
                "clutter_count": int(clutter_check.get("actual_count") or 0),
                "clutter_objects": clutter_check.get("actual_objects", []),
                "clutter_matched": bool(clutter_check.get("passed")),
                "clutter_authority": clutter_check.get("authority"),
                "background_texture_expected": bool(background_change),
                "background_texture_split": background_check.get("actual_split"),
                "wall_texture": background_check.get("actual_wall_texture"),
                "table_texture": background_check.get("actual_table_texture"),
                "background_texture_matched": bool(background_check.get("passed")),
                "background_texture_authority": background_check.get("authority"),
                "lighting_expected": bool(lighting_change),
                "random_light": lighting_check.get("actual_random_light"),
                "crazy_random_light_rate": lighting_check.get(
                    "actual_crazy_random_light_rate"
                ),
                "lighting_matched": bool(lighting_check.get("passed")),
                "lighting_authority": lighting_check.get("authority"),
                "variant_matched": bool(variant_check.get("passed")),
                "rule_passed": bool(scene.get("rule_check", {}).get("passed")),
                "expert_passed": bool(scene.get("expert", {}).get("passed")),
                "image": scene.get("image"),
            }
        )

    xy_values = [
        item["bell_position"][:2]
        for item in samples
        if isinstance(item.get("bell_position"), list)
        and len(item["bell_position"]) >= 2
    ]
    unique_xy = {
        (round(value[0], 8), round(value[1], 8))
        for value in xy_values
        if isinstance(value, list) and len(value) >= 2
    }
    result = {
        "start_seed": start_seed,
        "num_episodes": num_episodes,
        "controlled_axis": spec.get("controlled_axis"),
        "variant_contract": changes,
        "samples": samples,
        "metrics": {
            "expected_xy": expected_xy,
            "expected_bell_id": expected_bell_id,
            "unique_xy_count": len(unique_xy),
            "all_positions_matched": all(item["position_matched"] for item in samples),
            "position_varied": len(unique_xy) > 1,
            "observed_bell_ids": sorted(
                {
                    int(item["bell_id"])
                    for item in samples
                    if isinstance(item.get("bell_id"), int)
                    and not isinstance(item.get("bell_id"), bool)
                }
            ),
            "all_instances_matched": all(item["instance_matched"] for item in samples),
            "expected_clutter": bool(clutter_change),
            "minimum_clutter_count": 1 if clutter_change else 0,
            "all_clutter_matched": all(item["clutter_matched"] for item in samples),
            "clutter_counts": [item["clutter_count"] for item in samples],
            "expected_background_texture": bool(background_change),
            "required_texture_split": "unseen" if background_change else None,
            "all_background_textures_matched": all(
                item["background_texture_matched"] for item in samples
            ),
            "observed_texture_splits": sorted(
                {
                    str(item["background_texture_split"])
                    for item in samples
                    if item.get("background_texture_split") is not None
                }
            ),
            "expected_random_lighting": bool(lighting_change),
            "all_lighting_matched": all(item["lighting_matched"] for item in samples),
        },
        "passed": len(samples) == num_episodes
        and all(
            item["variant_matched"] and item["rule_passed"] and item["expert_passed"]
            for item in samples
        ),
    }
    write_json(run_dir / "validation/position_samples.json", result)
    return result


def regenerate_bbh_distractor_scene_checker(
    repo_root: Path,
    run_dir: Path,
    provider: Any,
    *,
    model: str,
    observation: Mapping[str, Any],
    repair_index: int,
    protected_before: Mapping[str, str],
) -> dict[str, Any]:
    """Regenerate both provider methods once after a failed visual check."""

    if repair_index != 1:
        raise VisualReflectionError(
            "provider scene+checker visual regeneration is limited to one attempt"
        )
    scene_check = validate_scene_check_spec(
        json.loads(
            (run_dir / "generation/scene_check_spec.json").read_text(
                encoding="utf-8"
            )
        )
    )
    if (
        scene_check["repair_policy"]["mode"]
        != "regenerate_scene_checker_code"
        or scene_check["repair_policy"]["max_repairs_supported"] != 1
    ):
        raise VisualReflectionError(
            "SceneCheckSpec does not authorize provider scene+checker regeneration"
        )
    task_proposal = json.loads(
        (run_dir / "generation/task_proposal.json").read_text(encoding="utf-8")
    )
    proposal = validate_bbh_distractor_proposal(
        json.loads(
            (
                run_dir / "generation/bbh_distractor_proposal.json"
            ).read_text(encoding="utf-8")
        )
    )
    current_methods = json.loads(
        (run_dir / "generation/provider_response.json").read_text(
            encoding="utf-8"
        )
    )
    validate_bbh_distractor_methods(current_methods, proposal)

    attempt_dir = run_dir / "reflection" / f"attempt_{repair_index - 1:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        "Regenerate the complete provider-written RoboTwin BeatBlockHammer "
        "scene and replacement success checker after the proposal-derived "
        "initial-frame visual check failed.\n\n"
        "TASK PROPOSAL:\n"
        + json.dumps(task_proposal, ensure_ascii=False, indent=2)
        + "\n\nSCENE CHECK SPEC:\n"
        + json.dumps(scene_check, ensure_ascii=False, indent=2)
        + "\n\nVISUAL OBSERVATION:\n"
        + json.dumps(dict(observation), ensure_ascii=False, indent=2)
        + "\n\nCURRENT METHODS:\n"
        + json.dumps(current_methods, ensure_ascii=False, indent=2)
        + "\n\nReturn one strict JSON object with exactly two string fields, "
        "load_actors and check_success. Regenerate both complete methods, not "
        "a patch. The target block and same-size lookalike distractor must both "
        "be visibly present in a physically plausible initial scene. Preserve "
        "the proposal's actor names, bounded distractor offset, target-contact "
        "requirement, and latched no-distractor-contact success semantics. "
        "Preserve the inherited RoboTwin interface: assign the official target "
        "actor to self.block, assign the lookalike to self.distractor, and read "
        "alignment from self.hammer.get_functional_point(0, \"pose\").p and "
        "self.block.get_functional_point(1, \"pose\").p. Detect contacts via "
        "self.check_actors_contact(self.hammer.get_name(), "
        "self.block.get_name()) and the analogous distractor call; the method "
        "requires actor-name strings, not actor objects. "
        "Exact actor identity, offset, and contact behavior are checked by "
        "simulator/semantic fixtures, not RGB. Do not use imports, files, "
        "network, processes, dunder access, dynamic execution, super(), or "
        "extra helpers. Do not return Markdown."
    )
    (attempt_dir / "repair_prompt.md").write_text(prompt, encoding="utf-8")
    response = provider.text(
        prompt,
        model=model,
        system=(
            "Return one strict JSON object containing the two corrected "
            "complete Python methods."
        ),
        max_tokens=5000,
        temperature=0.0,
    )
    (attempt_dir / "repair_response.txt").write_text(
        response + "\n", encoding="utf-8"
    )
    try:
        methods = json.loads(response)
    except json.JSONDecodeError as exc:
        raise BBHDistractorTaskGenError(
            "visual regeneration response must be one strict JSON object"
        ) from exc
    validation = validate_bbh_distractor_methods(methods, proposal)
    fixtures = [
        dict(item) for item in validation["checker_fixtures"]
    ]
    module_source = build_bbh_distractor_module(methods)
    compile(module_source, str(run_dir / "task.py"), "exec")
    if protected_hashes(repo_root) != dict(protected_before):
        raise BBHDistractorTaskGenError(
            "provider scene+checker regeneration changed protected files"
        )

    temporary_task = run_dir / "task.py.visual_repairing"
    temporary_task.write_text(module_source, encoding="utf-8")
    temporary_task.replace(run_dir / "task.py")
    write_json(run_dir / "generation/provider_response.json", methods)
    (run_dir / "generation/provider_response.txt").write_text(
        json.dumps(methods, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_json(run_dir / "validation/checker_fixtures.json", fixtures)

    module_sha256 = hashlib.sha256(
        (run_dir / "task.py").read_bytes()
    ).hexdigest()
    candidate_path = run_dir / "candidate_manifest.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["module_sha256"] = module_sha256
    candidate["scene_method_sha256"] = validation["scene_sha256"]
    candidate["success_method_sha256"] = validation["success_sha256"]
    provenance = candidate["codegen_provenance"]
    provenance["provider_call_count"] = int(
        provenance.get("provider_call_count", 0)
    ) + 1
    provenance["visual_regeneration_count"] = 1
    provenance["visual_regeneration_limit"] = 1
    provenance["provider_metadata"] = dict(
        getattr(provider, "last_metadata", {}) or {}
    )
    candidate["checker_contract"]["fixture_count"] = len(fixtures)
    candidate["checker_contract"]["fixture_pass_count"] = sum(
        1 for item in fixtures if item["passed"]
    )
    write_json(candidate_path, candidate)

    static_validation = {
        "variant_spec": {"valid": True},
        "provider_scene_checker": {
            "valid": True,
            "ast_policy": validation["policy"],
            "model_written_python": True,
            "restricted_success_spec_compiler_used": False,
            "checker_fixture_count": len(fixtures),
            "checker_fixture_pass_count": sum(
                1 for item in fixtures if item["passed"]
            ),
            "visual_regenerated": True,
        },
    }
    write_json(
        run_dir / "validation/bbh_distractor_static.json",
        static_validation,
    )
    result = {
        "repair_index": repair_index,
        "regenerated_methods": ["load_actors", "check_success"],
        "module_sha256": module_sha256,
        "scene_method_sha256": validation["scene_sha256"],
        "success_method_sha256": validation["success_sha256"],
        "checker_contract": candidate["checker_contract"],
        "static_validation": static_validation,
        "provider_metadata": dict(
            getattr(provider, "last_metadata", {}) or {}
        ),
        "installed": True,
    }
    write_json(attempt_dir / "repair.json", result)
    return result


def run_visual_self_reflection(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    provider: OpenAICompatibleProvider,
    *,
    seed: int,
    text_model: str,
    vision_model: str,
    max_repairs: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = json.loads((run_dir / "variant_spec.json").read_text(encoding="utf-8"))
    scene_check_path = run_dir / "generation/scene_check_spec.json"
    scene_check = (
        validate_scene_check_spec(
            json.loads(scene_check_path.read_text(encoding="utf-8"))
        )
        if scene_check_path.is_file()
        else build_scene_check_spec(spec)
    )
    is_click_bell = spec.get("task_name") == "click_bell"
    is_provider_distractor = (
        scene_check.get("success_semantics") == "provider_generated_python"
    )
    reflection_dir = run_dir / "reflection"
    reflection_dir.mkdir(parents=True, exist_ok=True)

    def observe(attempt_index: int) -> dict[str, Any]:
        attempt_dir = reflection_dir / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        scene_path = attempt_dir / "scene.json"
        image_path = attempt_dir / "render.png"
        scene = run_probe(
            repo_root,
            run_dir,
            manifest,
            seed=seed,
            expert=False,
            scene_json=scene_path,
            image=image_path,
            log_path=attempt_dir / "probe.log",
            raise_on_failure=False,
        )
        structural_probe_passed = bool(
            scene.get("setup_success")
            and scene.get("render_success")
            and scene.get("rule_check", {}).get("passed")
            and scene.get("returncode") == 0
        )
        variant_validation = (
            visual_validation.validate_click_bell_scene_contract(scene, spec)
            if is_click_bell
            and not is_provider_distractor
            and structural_probe_passed
            else {
                "status": "not_applicable",
                "passed": True,
                "authority": None,
            }
        )
        write_json(attempt_dir / "variant_validation.json", variant_validation)
        probe_passed = bool(
            structural_probe_passed and variant_validation.get("passed")
        )
        if probe_passed:
            vision = visual_validation.run_vision_check(
                provider,
                run_dir,
                spec,
                model=vision_model,
                image_path=image_path,
                prompt_path=attempt_dir / "vision_prompt.md",
                response_path=attempt_dir / "vision_response.txt",
                result_path=attempt_dir / "vision.json",
            )
        else:
            error = scene.get("error") or {}
            vision = {
                "aligned": False,
                "target_actor": "bell" if is_click_bell else "block",
                "unexpected_changes": [
                    "scene_variant_mismatch"
                    if structural_probe_passed and is_click_bell
                    else "scene_probe_failed"
                ],
                "diagnosis": (
                    "Simulator bell state did not match the validated variant."
                    if structural_probe_passed and is_click_bell
                    else f"Scene setup/render/rule probe failed: "
                    f"{error.get('type', 'unknown')}: {error.get('message', '')}"
                ),
                "suggestions": [
                    "Inspect the bounded click_bell overlay."
                    if is_click_bell
                    else "Repair load_actors() so setup, render, hammer/block actor checks pass."
                ],
                "confidence": 1.0,
                "passed": False,
                "variant_authorities": (
                    variant_validation.get("authorities") if is_click_bell else None
                ),
                "provider_metadata": {},
            }
            if not is_click_bell and not is_provider_distractor:
                vision.update(
                    {
                        "expected_color": "blue",
                        "observed_color": "unavailable",
                        "color_matches": False,
                    }
                )
            write_json(attempt_dir / "vision.json", vision)
        return {
            "passed": bool(probe_passed and vision.get("passed")),
            "probe_passed": probe_passed,
            "scene_path": str(scene_path.relative_to(run_dir)),
            "image_path": str(image_path.relative_to(run_dir)),
            "vision_path": str((attempt_dir / "vision.json").relative_to(run_dir)),
            "variant_validation_path": str(
                (attempt_dir / "variant_validation.json").relative_to(run_dir)
            ),
            "variant_validation": variant_validation,
            "vision": vision,
        }

    def repair(repair_index: int, observation: dict[str, Any]) -> dict[str, Any]:
        if is_click_bell and not is_provider_distractor:
            raise VisualReflectionError(
                "click_bell bounded overlay is validate-only and does not support repair"
            )
        if is_provider_distractor and is_click_bell:
            raise VisualReflectionError(
                "click_bell provider scene+checker visual repair requires a new "
                "candidate run; BBH repair is not reused across dialects"
            )
        if is_provider_distractor:
            update_manifest(
                run_dir,
                status=f"visual_scene_checker_regeneration_{repair_index}",
            )
            result = regenerate_bbh_distractor_scene_checker(
                repo_root,
                run_dir,
                provider,
                model=text_model,
                observation=observation,
                repair_index=repair_index,
                protected_before=manifest["protected_hashes_before"],
            )
            current = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            provider_record = dict(current.get("provider") or {})
            provider_calls = dict(provider_record.get("calls") or {})
            provider_calls[
                f"visual_scene_checker_regeneration_{repair_index}"
            ] = result["provider_metadata"]
            provider_record["calls"] = provider_calls
            provider_record["visual_regeneration_count"] = repair_index
            provider_record["visual_regeneration_limit"] = 1
            update_manifest(
                run_dir,
                static_validation=result["static_validation"],
                candidate_module_sha256=result["module_sha256"],
                checker_contract=result["checker_contract"],
                provider=provider_record,
            )
            return result
        update_manifest(
            run_dir,
            status=f"visual_reflection_repair_{repair_index}",
        )
        result = repair_generated_method(
            repo_root,
            run_dir,
            provider,
            model=text_model,
            spec=spec,
            observation=observation,
            repair_index=repair_index,
            protected_before=manifest["protected_hashes_before"],
        )
        update_manifest(
            run_dir,
            static_validation=result["static_validation"],
        )
        return result

    effective_max_repairs = min(
        0 if is_provider_distractor and is_click_bell else max_repairs,
        int(scene_check["repair_policy"]["max_repairs_supported"]),
    )
    summary = execute_reflection_loop(
        max_repairs=effective_max_repairs,
        observe=observe,
        repair=repair,
    )
    if is_click_bell and not is_provider_distractor:
        summary["requested_max_repairs"] = max_repairs
        summary["repair_supported"] = False
        summary[
            "validation_mode"
        ] = "simulator_position_or_instance_plus_visual_plausibility"
    if is_provider_distractor:
        summary["requested_max_repairs"] = max_repairs
        summary["repair_supported"] = not is_click_bell
        summary["repair_limit"] = 0 if is_click_bell else 1
        summary[
            "validation_mode"
        ] = "proposal_derived_target_distractor_visual_plausibility"
    summary["scene_check_spec"] = "generation/scene_check_spec.json"
    summary["scene_check_source"] = scene_check["source"]
    summary["repair_mode"] = scene_check["repair_policy"]["mode"]
    current_manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    task_proposal_path = run_dir / "generation/task_proposal.json"
    current_task_proposal = (
        json.loads(task_proposal_path.read_text(encoding="utf-8"))
        if task_proposal_path.is_file()
        else None
    )
    refreshed_bundle = write_task_artifact_bundle(
        repo_root,
        run_dir,
        current_manifest,
        task_proposal=current_task_proposal,
    )
    summary["task_artifact_bundle_refreshed"] = True
    update_manifest(
        run_dir,
        task_artifact_bundle="generation/task_artifact_bundle.json",
        scene_check_spec="generation/scene_check_spec.json",
        task_artifact_summary=task_artifact_summary(refreshed_bundle),
    )
    write_json(reflection_dir / "summary.json", summary)
    if not summary["passed"]:
        raise VisualReflectionError(
            f"Visual Self-Reflection 用尽 {max_repairs} 次 repair: {summary}"
        )

    final_attempt = reflection_dir / f"attempt_{summary['final_attempt']:02d}"
    shutil.copy2(final_attempt / "render.png", run_dir / "evidence/initial_head.png")
    shutil.copy2(final_attempt / "vision.json", run_dir / "validation/vision.json")
    if (final_attempt / "variant_validation.json").is_file():
        shutil.copy2(
            final_attempt / "variant_validation.json",
            run_dir / "validation/variant.json",
        )
    if (final_attempt / "vision_prompt.md").is_file():
        shutil.copy2(
            final_attempt / "vision_prompt.md",
            run_dir / "validation/vision_prompt.md",
        )
    if (final_attempt / "vision_response.txt").is_file():
        shutil.copy2(
            final_attempt / "vision_response.txt",
            run_dir / "validation/vision_response.txt",
        )
    final_scene = json.loads((final_attempt / "scene.json").read_text(encoding="utf-8"))
    final_vision = json.loads(
        (final_attempt / "vision.json").read_text(encoding="utf-8")
    )
    return summary, final_scene, final_vision


def run_act(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    seed: int,
    gpu: int,
    num_episodes: int,
    telemetry_profile: str = "balanced_v1",
) -> dict[str, Any]:
    """Compatibility wrapper around the TaskGen ACT runtime boundary."""

    return run_act_runtime(
        repo_root,
        run_dir,
        manifest,
        seed=seed,
        gpu=gpu,
        num_episodes=num_episodes,
        telemetry_profile=telemetry_profile,
        command_runner=run_command,
        json_writer=write_json,
        python_executable=sys.executable,
    )


def evaluate_run_telemetry(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    telemetry_root = run_dir / "evaluation/telemetry"
    if (
        manifest.get("generation_kind")
        == "generic_provider_scene_checker_codegen"
    ):
        return evaluate_generic_task_rollout_telemetry(
            repo_root,
            run_dir,
            manifest,
        )
    if manifest.get("generation_kind") == "provider_scene_checker_codegen":
        episode_dirs = sorted(
            metadata.parent
            for metadata in (telemetry_root / "act").glob(
                "episode_*/episode.json"
            )
        )
        if not episode_dirs:
            raise RuntimeError(
                "provider checker execution found no recorded ACT episodes"
            )
        task_name = manifest.get("task_name")
        checker_bridges = {
            "beat_block_hammer": (
                bbh_distractor_rollout_execution,
                "bbh_target_without_distractor_success",
            ),
            "click_bell": (
                click_bell_distractor_rollout_execution,
                "click_target_without_distractor_success",
            ),
        }
        bridge = checker_bridges.get(task_name)
        if bridge is None:
            raise RuntimeError(
                f"provider checker bridge is not registered: {task_name!r}"
            )
        checker_execution, outcome_metric = bridge
        executions = [
            checker_execution(
                episode_dir=episode_dir,
                candidate_dir=run_dir,
                policy_name="ACT",
            )
            for episode_dir in episode_dirs
        ]
        execution = {
            **executions[0],
            "episodes": [
                episode
                for item in executions
                for episode in item["episodes"]
            ],
        }
        artifact_stem = str(task_name)
        execution_path = (
            run_dir / f"evaluation/{artifact_stem}_checker_execution.json"
        )
        write_json(execution_path, execution)
        aggregate_path = (
            run_dir / f"evaluation/{artifact_stem}_checker_aggregate.json"
        )
        aggregate = aggregate_tool_executions(
            [execution],
            output_path=aggregate_path,
        )
        return {
            "artifact": str(execution_path.relative_to(repo_root)),
            "aggregate_artifact": str(aggregate_path.relative_to(repo_root)),
            "episode_count": len(execution["episodes"]),
            "outcome_metric": outcome_metric,
            "outcome_authority": "llm_generated_python_ast_validated",
            "outcome_binding": {
                "metric": outcome_metric,
                "authority": "llm_generated_python_ast_validated",
                "module_sha256": manifest["candidate_module_sha256"],
                "task_module": manifest["task_module"],
            },
            "tool_retrieval": {
                "route": "bound_llm_generated_checker",
                "generated_new_tool": False,
            },
            "episodes": execution["episodes"],
            "aggregate": aggregate,
        }
    bundle = json.loads(
        (run_dir / "generation/task_artifact_bundle.json").read_text(
            encoding="utf-8"
        )
    )
    semantics = bundle.get("success_semantics", {})
    outcome_metric = (
        "generated_check_success"
        if semantics.get("authority")
        == "compiled_success_spec_experimental_bounded"
        else "official_check_success"
    )
    outcome_binding = (
        {
            "metric": outcome_metric,
            "authority": semantics["authority"],
            "success_spec_sha256": semantics["success_spec_sha256"],
            "task_module": manifest["task_module"],
        }
        if outcome_metric == "generated_check_success"
        else None
    )
    summary = evaluate_telemetry_root(
        telemetry_root,
        user_request=manifest["user_request"],
        task_name=manifest["task_name"],
        outcome_metric=outcome_metric,
        outcome_binding=outcome_binding,
    )
    return {
        "artifact": str((telemetry_root / "tool_results.json").relative_to(repo_root)),
        "episode_count": summary["episode_count"],
        "outcome_metric": outcome_metric,
        "outcome_authority": semantics.get("authority"),
        "outcome_binding": outcome_binding,
        "tool_retrieval": summary["tool_retrieval"],
        "episodes": [
            {
                "episode_dir": episode["episode_dir"],
                "policy_name": episode["metadata"].get("policy_name"),
                "seed": episode["metadata"].get("seed"),
                "success": episode["metadata"].get("success"),
                "tool_results": episode["tool_results"],
            }
            for episode in summary["episodes"]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["robotwin", "libero"],
        default="robotwin",
        help="Select the existing RoboTwin TaskGen or the bounded LIBERO backend.",
    )
    parser.add_argument("--request")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume-run",
        help="Resume an existing run_id without calling the text-generation stages again.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-module")
    parser.add_argument(
        "--variant-hint-json",
        help="Trusted planner-owned JSON for a bounded declarative task variant.",
    )
    parser.add_argument(
        "--variant-id",
        help="Trusted planner template id recorded in VariantSpec v2.",
    )
    parser.add_argument(
        "--capability-contract-json",
        help=(
            "Trusted planner adapter contract; exact TaskGen identity and changes "
            "are revalidated before simulator or policy execution."
        ),
    )
    parser.add_argument(
        "--task-proposal-json",
        help=(
            "Paper-level semantic TaskProposal. TaskGen validates the fixed "
            "task/capability and consumes its changes before materialization."
        ),
    )
    parser.add_argument(
        "--proposal-json",
        help=(
            "Plan Agent Proposal. It contains no catalog template/aspect and "
            "is consumed by generic scene+checker TaskGen."
        ),
    )
    parser.add_argument(
        "--experiment-candidate-json",
        dest="legacy_experiment_candidate_json",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mode",
        choices=[
            "reuse",
            "force_codegen",
            "provider_scene_checker_codegen",
            "generic_provider_scene_checker_codegen",
            "official",
        ],
        default="force_codegen",
    )
    parser.add_argument("--text-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--vision-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--telemetry-profile",
        choices=["balanced_v1", "legacy_v1"],
        default="balanced_v1",
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--vision-check", action="store_true")
    parser.add_argument(
        "--max-reflections",
        type=int,
        default=2,
        help="Maximum number of CodeGen repairs after failed visual observations.",
    )
    parser.add_argument("--run-act", action="store_true")
    parser.add_argument(
        "--accept-task-only",
        action="store_true",
        help=(
            "Run the expert TaskGen gate and persist production Task acceptance "
            "without starting ACT."
        ),
    )
    parser.add_argument(
        "--taskgen-ablation-json",
        help=(
            "Preregistered Table 3 switches with exactly rag, "
            "visual_self_check, and readme_agent booleans."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark == "libero":
        from mea.libero.taskgen import run_libero_taskgen_cli

        run_libero_taskgen_cli(args)
        return
    if args.num_episodes <= 0:
        raise SystemExit("--num-episodes 必须是正整数")
    if args.accept_task_only and args.run_act:
        raise SystemExit("--accept-task-only cannot be combined with --run-act")
    if args.accept_task_only and not args.expert:
        raise SystemExit("--accept-task-only requires --expert")
    taskgen_ablation: dict[str, bool] | None = None
    if args.taskgen_ablation_json is not None:
        try:
            taskgen_ablation = validate_taskgen_ablation_switches(
                json.loads(args.taskgen_ablation_json)
            )
        except (json.JSONDecodeError, TaskGenError) as exc:
            raise SystemExit(f"invalid --taskgen-ablation-json: {exc}") from exc
        if (
            args.mode
            not in {
                "force_codegen",
                "provider_scene_checker_codegen",
                "generic_provider_scene_checker_codegen",
            }
            or args.resume_run
        ):
            raise SystemExit(
                "--taskgen-ablation-json requires a fresh codegen run"
            )
        if args.vision_check != taskgen_ablation["visual_self_check"]:
            raise SystemExit(
                "--vision-check must exactly match the preregistered "
                "visual_self_check switch"
            )
    repo_root = args.repo_root.expanduser().resolve()
    task_proposal: dict[str, Any] | None = None
    if args.task_proposal_json is not None:
        if args.resume_run:
            raise SystemExit("--task-proposal-json cannot be used with --resume-run")
        try:
            raw_task_proposal = json.loads(args.task_proposal_json)
            task_proposal = validate_task_proposal(
                raw_task_proposal, expected_task_name=args.task_name
            )
        except (json.JSONDecodeError, ProposalError) as exc:
            raise SystemExit(f"invalid --task-proposal-json: {exc}") from exc
    if (
        args.proposal_json is not None
        and args.legacy_experiment_candidate_json is not None
    ):
        raise SystemExit(
            "--proposal-json and the legacy candidate option are mutually exclusive"
        )
    raw_proposal_json = (
        args.proposal_json
        if args.proposal_json is not None
        else args.legacy_experiment_candidate_json
    )
    experiment_candidate: dict[str, Any] | None = None
    if raw_proposal_json is not None:
        if args.resume_run:
            raise SystemExit(
                "--proposal-json cannot be used with --resume-run"
            )
        if args.mode != "generic_provider_scene_checker_codegen":
            raise SystemExit(
                "--proposal-json requires generic provider mode"
            )
        if task_proposal is not None:
            raise SystemExit(
                "Plan Agent Proposal and legacy TaskProposal are mutually exclusive"
            )
        try:
            experiment_candidate = validate_experiment_candidate(
                json.loads(raw_proposal_json)
            )
        except (json.JSONDecodeError, ExperimentCandidateError) as exc:
            raise SystemExit(f"invalid --proposal-json: {exc}") from exc
        if experiment_candidate["base_task"] != args.task_name:
            raise SystemExit(
                "Proposal.base_task differs from --task-name"
            )
    elif args.mode == "generic_provider_scene_checker_codegen":
        raise SystemExit("generic provider mode requires --proposal-json")
    capability_contract: dict[str, Any] | None = None
    trusted_variant_spec: dict[str, Any] | None = None
    if args.capability_contract_json is not None:
        try:
            raw_contract = json.loads(args.capability_contract_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"--capability-contract-json is invalid JSON: {exc}"
            ) from exc
        try:
            if (
                task_proposal is not None
                and args.mode != "official"
                and args.variant_id is None
            ):
                args.variant_id = task_proposal["proposal_id"]
            capability_contract, trusted_variant_spec = (
                prepare_planner_capability_binding(
                    raw_contract,
                    task_name=args.task_name,
                    mode=args.mode,
                    variant_id=args.variant_id,
                    task_proposal=task_proposal,
                )
            )
        except RuntimeError as exc:
            raise SystemExit(f"capability contract preflight failed: {exc}") from exc
        if (
            args.task_module is not None
            and args.mode == "official"
            and args.task_module != f"envs.{args.task_name}"
        ):
            raise SystemExit(
                "capability-bound official execution cannot override --task-module"
            )
    if (
        task_proposal is not None
        and capability_contract is None
        and args.mode != "official"
        and args.variant_id is None
    ):
        # Preserve the standalone Proposal CLI: without a planner capability
        # envelope, the proposal id remains the only bounded variant identity.
        args.variant_id = task_proposal["proposal_id"]
    parsed_variant_hint: dict[str, Any] | None = None
    if args.variant_hint_json is not None:
        try:
            loaded_hint = json.loads(args.variant_hint_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--variant-hint-json is invalid JSON: {exc}") from exc
        if not isinstance(loaded_hint, dict):
            raise SystemExit("--variant-hint-json must encode an object")
        parsed_variant_hint = loaded_hint
        if (
            capability_contract is not None
            and task_proposal is None
            and parsed_variant_hint != capability_contract["taskgen"]["changes"]
        ):
            raise SystemExit(
                "variant hint differs from planner capability contract"
            )
    if task_proposal is not None:
        if parsed_variant_hint is not None and parsed_variant_hint != task_proposal["changes"]:
            raise SystemExit("variant hint differs from TaskProposal changes")
        parsed_variant_hint = task_proposal["changes"]
    bounded_click_bell = bool(
        not args.resume_run
        and args.task_name == "click_bell"
        and args.mode == "reuse"
        and parsed_variant_hint is not None
    )
    if (
        not args.resume_run
        and args.task_name == "click_bell"
        and args.mode == "reuse"
        and parsed_variant_hint is None
    ):
        raise SystemExit(
            "click_bell reuse requires trusted --variant-hint-json; "
            "use --mode official for the unchanged upstream task"
        )
    provider = None
    if (
        not args.resume_run
        and args.mode != "official"
        and not bounded_click_bell
        and not (trusted_variant_spec is not None and args.mode == "reuse")
    ) or args.vision_check:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=args.text_model,
            vision_model=args.vision_model,
            timeout=180.0,
        )
    if args.resume_run:
        if args.run_id:
            raise SystemExit("--resume-run 与 --run-id 不能同时使用")
        run_dir = repo_root / "mea/generated_tasks" / args.resume_run
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"run manifest 不存在: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        if not args.request:
            raise SystemExit("新 TaskGen run 必须提供 --request")
        if args.mode == "official":
            manifest = create_official_task_run(
                repo_root,
                args.request,
                task_name=args.task_name,
                task_module=args.task_module,
                run_id=args.run_id,
                telemetry_profile=args.telemetry_profile,
            )
        elif bounded_click_bell:
            manifest = create_click_bell_variant_run(
                repo_root,
                args.request,
                variant_hint=parsed_variant_hint,
                variant_id=args.variant_id,
                run_id=args.run_id,
                telemetry_profile=args.telemetry_profile,
            )
        elif args.mode == "provider_scene_checker_codegen":
            if (
                provider is None
                or trusted_variant_spec is None
                or task_proposal is None
                or capability_contract is None
                or capability_contract["taskgen"]["operation"]
                != "provider_scene_checker_codegen"
            ):
                raise SystemExit(
                    "provider scene+checker codegen requires its exact "
                    "capability contract, TaskProposal, and provider"
                )
            manifest = create_provider_scene_checker_taskgen_run(
                repo_root,
                user_request=args.request,
                provider=provider,
                model=args.text_model,
                variant_spec=trusted_variant_spec,
                task_proposal=task_proposal,
                run_id=args.run_id,
                telemetry_profile=args.telemetry_profile,
                ablation_switches=taskgen_ablation,
            )
        elif args.mode == "generic_provider_scene_checker_codegen":
            if (
                provider is None
                or experiment_candidate is None
                or not args.run_id
            ):
                raise SystemExit(
                    "generic scene+checker codegen requires a provider, "
                    "Proposal, and explicit --run-id"
                )
            try:
                manifest = create_generic_provider_taskgen_run(
                    repo_root,
                    user_request=args.request,
                    provider=provider,
                    model=args.text_model,
                    vision_model=args.vision_model,
                    experiment_candidate=experiment_candidate,
                    run_id=args.run_id,
                    seed=args.seed,
                    telemetry_profile=args.telemetry_profile,
                    ablation_switches=taskgen_ablation,
                )
            except Exception as exc:
                record_generic_taskgen_generation_failure(
                    repo_root,
                    run_id=args.run_id,
                    user_request=args.request,
                    experiment_candidate=experiment_candidate,
                    model=args.text_model,
                    telemetry_profile=args.telemetry_profile,
                    error=exc,
                )
                raise
        else:
            prototype = TaskGenPrototype(repo_root, provider, model=args.text_model)
            manifest = prototype.generate(
                args.request,
                task_name=args.task_name,
                mode=args.mode,
                run_id=args.run_id,
                variant_id=args.variant_id,
                trusted_variant_spec=trusted_variant_spec,
                task_proposal=task_proposal,
                ablation_switches=taskgen_ablation,
            )
        run_dir = repo_root / "mea/generated_tasks" / manifest["run_id"]

    if task_proposal is not None:
        write_json(run_dir / "generation/task_proposal.json", task_proposal)
        bundle = write_task_artifact_bundle(
            repo_root,
            run_dir,
            manifest,
            task_proposal=task_proposal,
        )
        update_manifest(
            run_dir,
            task_proposal=task_proposal,
            task_proposal_path="generation/task_proposal.json",
            task_artifact_bundle="generation/task_artifact_bundle.json",
            scene_check_spec="generation/scene_check_spec.json",
            task_artifact_summary=task_artifact_summary(bundle),
        )
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    if capability_contract is not None:
        try:
            validate_planner_capability_binding(
                capability_contract,
                task_name=args.task_name,
                mode=args.mode,
                variant_id=args.variant_id,
                run_dir=run_dir,
                task_proposal=task_proposal,
            )
        except RuntimeError as exc:
            update_manifest(
                run_dir,
                status="failed",
                failure_stage="capability_contract_validation",
                failure={"type": type(exc).__name__, "message": str(exc)},
            )
            raise SystemExit(f"capability contract validation failed: {exc}") from exc
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    generic_provider_mode = (
        manifest.get("mode") == "generic_provider_scene_checker_codegen"
    )
    if args.run_act and not generic_provider_mode:
        try:
            require_task_artifact_act_runtime_eligible(run_dir, manifest)
        except ProductionTaskAcceptanceError as exc:
            raise SystemExit(f"TaskGen ACT runtime gate failed: {exc}") from exc
    elif args.run_act:
        acceptance = manifest.get("task_generation_acceptance") or {}
        preflight = (
            (manifest.get("scene_validation") or {}).get("generic_preflight")
            or {}
        )
        fixtures = preflight.get("checker_fixtures") or []
        vision = manifest.get("vision_validation") or {}
        visual_required = acceptance.get(
            "visual_self_check_required", True
        )
        if (
            acceptance.get("status") != "accepted"
            or acceptance.get("act_rollouts_started_before_acceptance") != 0
            or preflight.get("render_passed") is not True
            or preflight.get("expert_passed") is not True
            or preflight.get("scene_change_passed") is not True
            or preflight.get("preserved_conditions_verified") is False
            or not fixtures
            or any(item.get("passed") is not True for item in fixtures)
            or (
                visual_required
                and (
                    vision.get("status") != "passed"
                    or vision.get("passed") is not True
                )
            )
        ):
            raise SystemExit(
                "generic TaskGen ACT runtime gate failed: live fixture/render/"
                "expert preflight is incomplete"
            )

    if args.accept_task_only:
        requested_execution_backend = "expert+task_acceptance_no_act"
    else:
        requested_execution_backend = (
            (
                "both"
                if args.expert and args.run_act
                else "act"
                if args.run_act
                else "expert"
                if args.expert
                else "setup_probe"
            )
            if manifest.get("mode") == "official"
            else (
                "expert+act"
                if args.expert and args.run_act
                else "expert_gate+act"
                if args.run_act
                else "expert"
                if args.expert
                else "setup_probe"
            )
        )
    update_manifest(
        run_dir,
        requested_execution_backend=requested_execution_backend,
    )

    position_samples: dict[str, Any] | None = None
    try:
        if manifest.get("mode") == "official" and args.vision_check:
            raise RuntimeError(
                "official route bypasses generated-scene vision/reflection; "
                "use expert, act, or both execution without scene codegen"
            )
        scene = None
        if args.vision_check and not generic_provider_mode:
            if provider is None:
                raise RuntimeError("vision check 缺少 provider")
            reflection, reflected_scene, vision = run_visual_self_reflection(
                repo_root,
                run_dir,
                manifest,
                provider,
                seed=args.seed,
                text_model=args.text_model,
                vision_model=args.vision_model,
                max_repairs=args.max_reflections,
            )
            update_manifest(
                run_dir,
                status="vision_passed",
                visual_self_reflection=reflection,
                vision_validation=vision,
            )
            scene = reflected_scene

        if generic_provider_mode:
            scene = dict(manifest.get("scene_validation") or {})
            write_json(run_dir / "validation/scene.json", scene)
        elif manifest.get("mode") == "official" and args.expert:
            scene = run_official_expert_episodes(
                repo_root,
                run_dir,
                manifest,
                start_seed=args.seed,
                num_episodes=args.num_episodes,
                telemetry_profile=args.telemetry_profile,
            )
            update_manifest(run_dir, status="probe_passed", scene_validation=scene)
        elif manifest.get("mode") == "official" and args.run_act:
            # ACT-only evaluates the learned policy; this probe validates only
            # simulator setup/render/rules and does not create expert evidence.
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                expert=False,
                telemetry_profile=args.telemetry_profile,
            )
            update_manifest(run_dir, status="probe_passed", scene_validation=scene)
        elif args.expert or args.run_act:
            expert_telemetry_dir = (
                run_dir
                / "evaluation/telemetry/expert"
                / f"episode_000_seed_{args.seed}"
            )
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                expert=True,
                telemetry_dir=expert_telemetry_dir,
                telemetry_profile=args.telemetry_profile,
            )
            update_manifest(run_dir, status="probe_passed", scene_validation=scene)
        elif args.probe and not args.vision_check:
            scene = run_probe(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                expert=False,
                telemetry_profile=args.telemetry_profile,
            )
            update_manifest(run_dir, status="probe_passed", scene_validation=scene)
        elif scene is not None:
            write_json(run_dir / "validation/scene.json", scene)
            update_manifest(run_dir, scene_validation=scene)

        if args.accept_task_only and generic_provider_mode:
            update_manifest(
                run_dir,
                status="completed_without_act",
                failure=None,
            )
        elif args.accept_task_only:
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            task_acceptance = record_production_task_acceptance(
                run_dir,
                manifest,
                scene=scene,
                position_samples=None,
                task_resolution=None,
                require_expert=True,
            )
            bundle = json.loads(
                (
                    run_dir / "generation/task_artifact_bundle.json"
                ).read_text(encoding="utf-8")
            )
            semantics = bundle.get("success_semantics")
            act_runtime_eligible = not (
                isinstance(semantics, Mapping)
                and semantics.get("act_runtime_eligible") is False
            )
            update_manifest(
                run_dir,
                task_generation_acceptance={
                    "status": task_acceptance["status"],
                    "scope": "task_generation_only_no_act",
                    "artifact": (
                        "validation/task_generation_attempts/"
                        "task_generation_attempt_summary.json"
                    ),
                    "act_rollouts_started_before_acceptance": task_acceptance[
                        "runtime"
                    ]["act_rollouts_started"],
                    "act_runtime_eligible": act_runtime_eligible,
                },
            )
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            require_production_task_acceptance(
                run_dir,
                manifest,
                task_resolution=None,
            )

        if args.run_act:
            if generic_provider_mode:
                position_samples = {
                    "status": "not_applicable",
                    "reason": (
                        "generic TaskGen validates its Query-derived scene "
                        "through live fixtures rather than a task-specific "
                        "position contract"
                    ),
                    "passed": True,
                    "samples": [],
                    "metrics": {},
                }
                write_json(
                    run_dir / "validation/position_samples.json",
                    position_samples,
                )
            elif manifest["task_name"] == "beat_block_hammer":
                position_samples = collect_position_samples(
                    repo_root,
                    run_dir,
                    manifest,
                    start_seed=args.seed,
                    num_episodes=args.num_episodes,
                    first_scene=scene,
                    require_expert=bool(
                        args.expert or manifest.get("mode") != "official"
                    ),
                )
            elif (
                manifest.get("generation_kind") == "bounded_variant_overlay"
                and manifest["task_name"] == "click_bell"
            ):
                position_samples = collect_click_bell_position_samples(
                    repo_root,
                    run_dir,
                    manifest,
                    start_seed=args.seed,
                    num_episodes=args.num_episodes,
                    first_scene=scene,
                )
            else:
                position_samples = {
                    "status": "not_applicable",
                    "reason": ("non-BBH tasks have no BBH block-position contract"),
                    "passed": True,
                    "samples": [],
                    "metrics": {},
                }
                write_json(
                    run_dir / "validation/position_samples.json",
                    position_samples,
                )
            update_manifest(run_dir, position_samples=position_samples)
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            if not generic_provider_mode:
                task_acceptance = record_production_task_acceptance(
                    run_dir,
                    manifest,
                    scene=scene,
                    position_samples=position_samples,
                    task_resolution=None,
                    require_expert=bool(
                        args.expert or manifest.get("mode") != "official"
                    ),
                )
                update_manifest(
                    run_dir,
                    task_generation_acceptance={
                        "status": task_acceptance["status"],
                        "artifact": (
                            "validation/task_generation_attempts/"
                            "task_generation_attempt_summary.json"
                        ),
                        "act_rollouts_started_before_acceptance": task_acceptance[
                            "runtime"
                        ]["act_rollouts_started"],
                    },
                )
                manifest = json.loads(
                    (run_dir / "manifest.json").read_text(encoding="utf-8")
                )
                require_production_task_acceptance(
                    run_dir,
                    manifest,
                    task_resolution=None,
                    for_act=True,
                )
            act = run_act(
                repo_root,
                run_dir,
                manifest,
                seed=args.seed,
                gpu=args.gpu,
                num_episodes=args.num_episodes,
                telemetry_profile=args.telemetry_profile,
            )
            alignment = {
                "status": "not_applicable",
                "passed": True,
                "reason": "paired expert/ACT execution was not requested",
                "expert_seeds": [],
                "act_seeds": act.get("actual_seeds", []),
            }
            if manifest.get("mode") == "official" and args.expert:
                expert_seeds = [
                    int(item["seed"])
                    for item in (scene or {})
                    .get("expert_batch", {})
                    .get("episodes", [])
                ]
                act_seeds = [int(value) for value in act.get("actual_seeds", [])]
                aligned = expert_seeds == act_seeds
                alignment = {
                    "status": "passed" if aligned else "failed",
                    "passed": aligned,
                    "reason": (
                        "expert and ACT used the same ordered seeds"
                        if aligned
                        else "expert and ACT ordered seeds differ"
                    ),
                    "expert_seeds": expert_seeds,
                    "act_seeds": act_seeds,
                }
            write_json(
                run_dir / "evaluation/backend_seed_alignment.json",
                alignment,
            )
            update_manifest(
                run_dir,
                act_evaluation=act,
                backend_seed_alignment=alignment,
            )
            if not alignment["passed"]:
                raise RuntimeError(
                    "paired expert/ACT seed alignment failed: "
                    f"expert={alignment['expert_seeds']}, "
                    f"ACT={alignment['act_seeds']}"
                )
            trusted_tools = evaluate_run_telemetry(
                repo_root,
                run_dir,
                manifest,
            )
            update_manifest(
                run_dir,
                status="completed",
                failure=None,
                act_evaluation=act,
                execution_backends=(
                    ["expert", "ACT"]
                    if args.expert or manifest.get("mode") != "official"
                    else ["ACT"]
                ),
                pre_policy_gates=(
                    ["expert_solvability", "controlled_variation_samples"]
                    if manifest.get("mode") != "official"
                    else ["expert_solvability"]
                    if args.expert
                    else ["setup_render_rule"]
                ),
                backend_seed_alignment=alignment,
                trusted_tool_evaluation=trusted_tools,
            )
        else:
            updates: dict[str, Any] = {
                "status": "completed_without_act",
                "failure": None,
            }
            if args.expert:
                updates["execution_backends"] = ["expert"]
                if not args.accept_task_only:
                    updates["trusted_tool_evaluation"] = evaluate_run_telemetry(
                        repo_root,
                        run_dir,
                        manifest,
                    )
            update_manifest(run_dir, **updates)
    except Exception as exc:
        update_manifest(
            run_dir,
            status="failed",
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise

    print(
        json.dumps(
            json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
