"""Generate, validate, render, and optionally evaluate one TaskGen variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.capability_adapter import (
    CapabilityAdapterError,
    taskgen_route,
    validate_capability_contract,
    validate_contract_changes,
)
from mea.providers import OpenAICompatibleProvider
from mea.proposals import ProposalError, validate_task_proposal
from mea.runtime_ledger import record_act_batch_start
from mea.toolkit import evaluate_telemetry_root
from mea.taskgen import (
    ClickBellTaskGenError,
    TaskGenPrototype,
    VisualReflectionError,
    execute_reflection_loop,
    inject_oversized_block_fixture,
    inject_wrong_color_fixture,
    repair_generated_method,
    validate_click_bell_vision_observation,
    validate_vision_observation,
    create_click_bell_variant_run,
    create_official_task_run,
    default_bbh_success_spec,
    validate_click_bell_variant_hint,
    build_variant_spec,
    validate_variant_spec_envelope,
    build_scene_check_spec,
    validate_scene_check_spec,
    write_task_artifact_bundle,
)


_REGISTRATION_KEYS = {
    "schema_version",
    "registration_id",
    "evidence_manifest_payload_sha256",
    "command_plan_sha256",
    "registered_route_sha256",
    "checkpoint_file_set_sha256",
    "source_artifact_file_set_sha256",
    "base_commit",
    "candidate_suite_sha256",
    "trusted_catalog_sha256",
    "trusted_template_contract_sha256",
    "strategy",
    "expected_evaluation_id",
    "expected_child_run_prefix",
}


def validate_registration_identity(value: Any, *, run_id: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _REGISTRATION_KEYS:
        raise ValueError("registration identity fields changed")
    if value.get("schema_version") != 1:
        raise ValueError("registration identity schema_version must be 1")
    for field in (
        "evidence_manifest_payload_sha256",
        "command_plan_sha256",
        "registered_route_sha256",
        "checkpoint_file_set_sha256",
        "source_artifact_file_set_sha256",
        "candidate_suite_sha256",
        "trusted_catalog_sha256",
        "trusted_template_contract_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(value.get(field) or "")):
            raise ValueError(f"invalid registration hash: {field}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(value.get("base_commit") or "")):
        raise ValueError("invalid registration base_commit")
    if value.get("strategy") not in {
        "fixed_predeclared_v1",
        "dynamic_evidence_v1",
    }:
        raise ValueError("invalid registered strategy")
    evaluation_id = value.get("expected_evaluation_id")
    if not isinstance(evaluation_id, str) or not re.fullmatch(
        r"eval_[A-Za-z0-9_]+", evaluation_id
    ):
        raise ValueError("invalid registered evaluation id")
    expected_prefix = f"run_{evaluation_id.removeprefix('eval_')}_"
    if value.get("expected_child_run_prefix") != expected_prefix:
        raise ValueError("invalid registered child run prefix")
    if run_id is not None and not run_id.startswith(expected_prefix):
        raise ValueError("TaskGen run_id differs from registered parent")
    return dict(value)


def bind_registration_to_episode_metadata(
    run_dir: Path, registration_identity: dict[str, Any]
) -> None:
    telemetry = (run_dir / "evaluation/telemetry").resolve()
    if not telemetry.is_dir():
        return
    for path in sorted(telemetry.rglob("episode.json")):
        if path.is_symlink():
            raise RuntimeError("episode metadata may not be a symlink")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(telemetry):
            raise RuntimeError("episode metadata path escapes telemetry root")
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("episode metadata must be an object")
        existing = value.get("registration_identity")
        if existing is not None and existing != registration_identity:
            raise RuntimeError("episode registration identity already differs")
        value["registration_identity"] = registration_identity
        write_json(resolved, value)


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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        if task_name == "click_bell" and proposal["changes"]:
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


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode


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
) -> dict[str, Any]:
    scene_json = scene_json or run_dir / "validation/scene.json"
    image = image or run_dir / "evidence/initial_head.png"
    log_path = log_path or run_dir / "validation/probe.log"
    command = [
        sys.executable,
        "-m",
        "mea.taskgen.probe",
        "--repo-root",
        str(repo_root),
        "--task-name",
        manifest["task_name"],
        "--task-module",
        manifest["task_module"],
        "--task-config",
        "demo_clean",
        "--ckpt-setting",
        "demo_clean",
        "--overlay",
        str(run_dir / "overlay.yml"),
        "--seed",
        str(seed),
        "--episode-index",
        str(episode_index),
        "--image",
        str(image),
        "--output",
        str(scene_json),
        "--telemetry-profile",
        telemetry_profile,
    ]
    if expert:
        command.append("--expert")
    if manifest.get("capability_id") == "scene_background_texture":
        # RoboTwin selects assets/background_texture/unseen only in eval mode.
        command.append("--eval-mode")
    if telemetry_dir is not None:
        command.extend(["--telemetry-dir", str(telemetry_dir)])
    if visual_capture_profile_id is not None:
        command.extend(["--visual-capture-profile", visual_capture_profile_id])

    attempts: list[dict[str, Any]] = []
    attempt_logs: list[Path] = []
    attempt_limit = max(1, max_expert_attempts) if expert else 1
    scene: dict[str, Any] = {}
    returncode = 1
    for attempt_index in range(attempt_limit):
        attempt_log = (
            log_path.with_name(
                f"{log_path.stem}_attempt_{attempt_index}{log_path.suffix}"
            )
            if expert
            else log_path
        )
        attempt_logs.append(attempt_log)
        returncode = run_command(
            command,
            cwd=repo_root,
            log_path=attempt_log,
        )
        scene = (
            json.loads(scene_json.read_text(encoding="utf-8"))
            if scene_json.exists()
            else {}
        )
        attempts.append(
            {
                "attempt_index": attempt_index,
                "returncode": returncode,
                "expert": scene.get("expert"),
            }
        )
        if returncode != 2:
            break

    if expert:
        combined = []
        for attempt_index, attempt_log in enumerate(attempt_logs):
            combined.append(f"===== expert attempt {attempt_index} =====\n")
            if attempt_log.is_file():
                combined.append(attempt_log.read_text(encoding="utf-8"))
        log_path.write_text("".join(combined), encoding="utf-8")
        scene.setdefault("expert", {})["attempts_used"] = len(attempts)
        scene["expert_attempts"] = attempts
    scene["returncode"] = returncode
    write_json(scene_json, scene)
    if raise_on_failure and returncode != 0:
        raise RuntimeError(f"setup/expert probe 失败，returncode={returncode}")
    return scene


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
) -> dict[str, Any]:
    """Collect simulator-native block poses for every evaluation seed."""

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
                "expert_passed": bool(scene.get("expert", {}).get("passed")),
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
        "samples": samples,
        "metrics": {
            "unique_xy_count": len(unique_xy),
            "x_span": max(xs) - min(xs),
            "y_span": max(ys) - min(ys),
            "position_varied": len(unique_xy) > 1,
        },
        "passed": len(samples) == num_episodes
        and all(item["rule_passed"] and item["expert_passed"] for item in samples),
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
        variant_check = validate_click_bell_scene_contract(scene, spec)
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


def run_vision_check(
    provider: OpenAICompatibleProvider,
    run_dir: Path,
    spec: dict[str, Any],
    *,
    model: str,
    image_path: Path | None = None,
    prompt_path: Path | None = None,
    response_path: Path | None = None,
    result_path: Path | None = None,
) -> dict[str, Any]:
    image_path = image_path or run_dir / "evidence/initial_head.png"
    prompt_path = prompt_path or run_dir / "validation/vision_prompt.md"
    response_path = response_path or run_dir / "validation/vision_response.txt"
    result_path = result_path or run_dir / "validation/vision.json"
    scene_check_path = run_dir / "generation/scene_check_spec.json"
    if scene_check_path.is_file():
        scene_check = validate_scene_check_spec(
            json.loads(scene_check_path.read_text(encoding="utf-8"))
        )
    else:
        scene_check = build_scene_check_spec(spec)
        write_json(scene_check_path, scene_check)
    scene_check_text = json.dumps(scene_check, ensure_ascii=False, indent=2)
    if spec.get("task_name") == "click_bell":
        bell_change = spec["changes"].get("bell")
        randomization_change = spec["changes"].get("domain_randomization") or {}
        clutter_change = (
            randomization_change
            if "cluttered_table" in randomization_change
            else None
        )
        background_change = (
            randomization_change
            if "random_background" in randomization_change
            else None
        )
        lighting_change = (
            randomization_change if "random_light" in randomization_change else None
        )
        if clutter_change is not None:
            contract_description = (
                "This round intentionally enables RoboTwin's simulator-native "
                "cluttered_table with clean_background_rate=0. The extra tabletop "
                "objects are expected and must not be reported as an unexpected "
                "change. Check that the target bell remains visible and the "
                "physical scene is plausible; exact clutter count is checked from "
                "simulator task state."
            )
        elif background_change is not None:
            contract_description = (
                "This round enables RoboTwin's simulator-native random_background "
                "with clean_background_rate=0 and eval_mode=true. Both table and "
                "wall therefore use the upstream unseen texture split. Check only "
                "that the bell remains visible and the rendered scene is plausible; "
                "exact texture ids and split are checked from simulator task info."
            )
        elif lighting_change is not None:
            contract_description = (
                "This round enables RoboTwin's simulator-native random_light with "
                "crazy_random_light_rate=0, so point and directional light colors "
                "are randomized once at setup without per-frame flicker. Check only "
                "that the bell remains visible and illumination is usable; the "
                "configuration branch is checked from simulator task attributes."
            )
        elif bell_change and bell_change.get("instance_mode") == "fixed":
            bell_id = int(bell_change["bell_id"])
            visual_description = (
                "白色 dome、黑色底座、较大实例" if bell_id == 0 else "蓝色 dome、棕色底座、较小实例"
            )
            contract_description = (
                f"本轮固定官方 bell base{bell_id}（{visual_description}），位置保持官方随机。"
                "精确 bell_id 已由 simulator task attribute 检查负责。"
            )
        elif bell_change:
            expected_xy = bell_change["xy"]
            contract_description = (
                f"本轮固定 workspace xy={expected_xy}，bell 实例保持官方随机。"
                "精确 XY 已由 simulator tracked actor 检查负责。"
            )
        prompt = f"""这是 RoboTwin click_bell 受限单轴变式的初始场景首帧。
请只检查目标 bell 是否清晰可见、场景是否物理合理、是否存在明显多余或缺失物体。
{contract_description}
不能仅凭 RGB 宣称精确坐标或实例 ID 是否正确。

PROPOSAL-DERIVED SCENE CHECK SPEC:
{scene_check_text}

只输出 JSON：
{{
  "aligned": true,
  "target_actor": "bell",
  "bell_visible": true,
  "unexpected_changes": [],
  "diagnosis": "目标铃是否可见以及场景是否存在明显异常",
  "suggestions": [],
  "confidence": 0.0
}}
"""
    else:
        expected_half_size = 0.025 * float(spec["changes"]["block"]["scale"])
        prompt = f"""这是 RoboTwin beat_block_hammer 的初始场景首帧。
请检查被锤子敲击的方块是否符合下面的 VariantSpec，并检查场景是否有明显异常：
{json.dumps(spec, ensure_ascii=False, indent=2)}

PROPOSAL-DERIVED SCENE CHECK SPEC:
{scene_check_text}

官方 scale=1.0 的方块 half_size 是 (0.025, 0.025, 0.025) 米；本次预期
half_size 是 ({expected_half_size:.6f}, {expected_half_size:.6f}, {expected_half_size:.6f}) 米。
请结合方块与锤子的相对尺寸判断是否明显偏大或偏小。

只输出 JSON：
{{
  "aligned": true,
  "target_actor": "block",
  "observed_color": "blue",
  "unexpected_changes": [],
  "diagnosis": "场景与需求是否一致，以及不一致的具体原因",
  "suggestions": ["若不一致，给出只修改 load_actors() 的具体建议"],
  "confidence": 0.0
}}
"""
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    response = provider.vision(
        prompt,
        image_path,
        model=model,
        max_tokens=512,
        temperature=0.0,
    )
    response_path.write_text(response + "\n", encoding="utf-8")
    from mea.taskgen import extract_json_response

    parsed = extract_json_response(response)
    result = (
        validate_click_bell_vision_observation(parsed)
        if spec.get("task_name") == "click_bell"
        else validate_vision_observation(parsed, spec)
    )
    result["provider_metadata"] = dict(provider.last_metadata)
    write_json(result_path, result)
    return result


def validate_click_bell_scene_contract(
    scene: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    """Validate the controlled axis from simulator state, never from RGB."""

    if not isinstance(spec, dict) or spec.get("task_name") != "click_bell":
        raise ClickBellTaskGenError("scene contract requires a click_bell variant spec")
    normalized = validate_click_bell_variant_hint(spec.get("changes"))
    bell_change = normalized.get("bell")
    randomization_change = normalized.get("domain_randomization") or {}
    clutter_change = (
        randomization_change if "cluttered_table" in randomization_change else None
    )
    background_change = (
        randomization_change if "random_background" in randomization_change else None
    )
    lighting_change = (
        randomization_change if "random_light" in randomization_change else None
    )
    expected_axis = (
        "robustness.scene_clutter"
        if clutter_change is not None
        else "scene_background_texture"
        if background_change is not None
        else "scene_lighting"
        if lighting_change is not None
        else "object_instance"
        if bell_change and bell_change.get("instance_mode") == "fixed"
        else "object_position"
    )
    declared_axis = spec.get("controlled_axis")
    if declared_axis is not None and declared_axis != expected_axis:
        raise ClickBellTaskGenError(
            "variant spec controlled_axis does not match its strict bell contract"
        )
    bell = next(
        (
            actor
            for actor in scene.get("tracked_actors", [])
            if actor.get("id") == "bell"
        ),
        None,
    )
    actual_xy = (
        [float(value) for value in bell.get("position", [])[:2]]
        if isinstance(bell, dict)
        else []
    )
    if bell_change and bell_change.get("position_mode") == "fixed":
        expected_xy = [float(value) for value in bell_change["xy"]]
        position_passed = len(actual_xy) == 2 and all(
            abs(left - right) <= 1e-6 for left, right in zip(actual_xy, expected_xy)
        )
        position = {
            "status": "passed" if position_passed else "failed",
            "passed": position_passed,
            "expected_xy": expected_xy,
            "actual_xy": actual_xy,
            "tolerance": 1e-6,
            "authority": "simulator_tracked_actor_xy",
        }
    else:
        position = {
            "status": "not_applicable",
            "passed": True,
            "expected_xy": None,
            "actual_xy": actual_xy,
            "tolerance": None,
            "authority": "simulator_tracked_actor_xy",
        }

    if bell_change and bell_change.get("instance_mode") == "fixed":
        expected_bell_id = int(bell_change["bell_id"])
        actual_bell_id = (scene.get("task_attributes") or {}).get("bell_id")
        instance_passed = (
            not isinstance(actual_bell_id, bool)
            and isinstance(actual_bell_id, int)
            and actual_bell_id == expected_bell_id
        )
        instance = {
            "status": "passed" if instance_passed else "failed",
            "passed": instance_passed,
            "expected_bell_id": expected_bell_id,
            "actual_bell_id": actual_bell_id,
            "authority": "simulator_task_attribute:bell_id",
        }
    else:
        instance = {
            "status": "not_applicable",
            "passed": True,
            "expected_bell_id": None,
            "actual_bell_id": (scene.get("task_attributes") or {}).get("bell_id"),
            "authority": "simulator_task_attribute:bell_id",
        }

    randomization = scene.get("domain_randomization") or {}
    actual_clutter_enabled = randomization.get("cluttered_table")
    actual_objects = randomization.get("cluttered_objects")
    if not isinstance(actual_objects, list):
        actual_objects = []
    actual_count = randomization.get("cluttered_object_count")
    if isinstance(actual_count, bool) or not isinstance(actual_count, int):
        actual_count = len(actual_objects)
    if clutter_change is not None:
        clutter_passed = bool(
            actual_clutter_enabled is True
            and randomization.get("clean_background_rate") == 0.0
            and actual_count >= 1
        )
        clutter = {
            "status": "passed" if clutter_passed else "failed",
            "passed": clutter_passed,
            "expected_enabled": True,
            "expected_clean_background_rate": 0.0,
            "minimum_object_count": 1,
            "actual_enabled": actual_clutter_enabled,
            "actual_clean_background_rate": randomization.get("clean_background_rate"),
            "actual_count": actual_count,
            "actual_objects": actual_objects,
            "authority": "simulator_task_info:cluttered_table_info",
        }
    else:
        clutter = {
            "status": "not_applicable",
            "passed": True,
            "expected_enabled": None,
            "expected_clean_background_rate": None,
            "minimum_object_count": 0,
            "actual_enabled": actual_clutter_enabled,
            "actual_clean_background_rate": randomization.get("clean_background_rate"),
            "actual_count": actual_count,
            "actual_objects": actual_objects,
            "authority": "simulator_task_info:cluttered_table_info",
        }

    actual_random_background = randomization.get("random_background")
    actual_wall_texture = randomization.get("wall_texture")
    actual_table_texture = randomization.get("table_texture")
    actual_texture_split = randomization.get("texture_split")
    if background_change is not None:
        background_passed = bool(
            scene.get("eval_mode") is True
            and actual_random_background is True
            and randomization.get("clean_background_rate") == 0.0
            and isinstance(actual_wall_texture, str)
            and actual_wall_texture.startswith("unseen/")
            and isinstance(actual_table_texture, str)
            and actual_table_texture.startswith("unseen/")
            and actual_texture_split == "unseen"
            and randomization.get("background_authority")
            == "simulator_task_info:texture_info"
        )
        background_texture = {
            "status": "passed" if background_passed else "failed",
            "passed": background_passed,
            "expected_random_background": True,
            "expected_clean_background_rate": 0.0,
            "expected_eval_mode": True,
            "expected_split": "unseen",
            "actual_random_background": actual_random_background,
            "actual_clean_background_rate": randomization.get(
                "clean_background_rate"
            ),
            "actual_eval_mode": scene.get("eval_mode"),
            "actual_split": actual_texture_split,
            "actual_wall_texture": actual_wall_texture,
            "actual_table_texture": actual_table_texture,
            "authority": "simulator_task_info:texture_info",
        }
    else:
        background_texture = {
            "status": "not_applicable",
            "passed": True,
            "expected_random_background": None,
            "expected_clean_background_rate": None,
            "expected_eval_mode": None,
            "expected_split": None,
            "actual_random_background": actual_random_background,
            "actual_clean_background_rate": randomization.get(
                "clean_background_rate"
            ),
            "actual_eval_mode": scene.get("eval_mode"),
            "actual_split": actual_texture_split,
            "actual_wall_texture": actual_wall_texture,
            "actual_table_texture": actual_table_texture,
            "authority": "simulator_task_info:texture_info",
        }

    actual_random_light = randomization.get("random_light")
    actual_crazy_rate = randomization.get("crazy_random_light_rate")
    actual_crazy_light = randomization.get("crazy_random_light")
    direction_light_count = randomization.get("direction_light_count")
    point_light_count = randomization.get("point_light_count")
    direction_light_colors = randomization.get("direction_light_colors")
    point_light_colors = randomization.get("point_light_colors")

    def valid_light_colors(colors: Any, count: Any) -> bool:
        return bool(
            isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 1
            and isinstance(colors, list)
            and len(colors) == count
            and all(
                isinstance(color, list)
                and len(color) == 3
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= 1.0
                    for value in color
                )
                for color in colors
            )
        )

    if lighting_change is not None:
        lighting_passed = bool(
            actual_random_light is True
            and actual_crazy_rate == 0.0
            and actual_crazy_light is False
            and valid_light_colors(direction_light_colors, direction_light_count)
            and valid_light_colors(point_light_colors, point_light_count)
            and randomization.get("lighting_authority")
            == (
                "simulator_task_attributes:random_light,crazy_random_light_rate,"
                "crazy_random_light;simulator_light_components:get_color"
            )
        )
        lighting = {
            "status": "passed" if lighting_passed else "failed",
            "passed": lighting_passed,
            "expected_random_light": True,
            "expected_crazy_random_light_rate": 0.0,
            "expected_temporal_flicker": False,
            "actual_random_light": actual_random_light,
            "actual_crazy_random_light_rate": actual_crazy_rate,
            "actual_crazy_random_light": actual_crazy_light,
            "direction_light_count": direction_light_count,
            "point_light_count": point_light_count,
            "direction_light_colors": direction_light_colors,
            "point_light_colors": point_light_colors,
            "authority": randomization.get("lighting_authority"),
        }
    else:
        lighting = {
            "status": "not_applicable",
            "passed": True,
            "expected_random_light": None,
            "expected_crazy_random_light_rate": None,
            "expected_temporal_flicker": None,
            "actual_random_light": actual_random_light,
            "actual_crazy_random_light_rate": actual_crazy_rate,
            "actual_crazy_random_light": actual_crazy_light,
            "direction_light_count": direction_light_count,
            "point_light_count": point_light_count,
            "direction_light_colors": direction_light_colors,
            "point_light_colors": point_light_colors,
            "authority": randomization.get("lighting_authority"),
        }

    passed = bool(
        position["passed"]
        and instance["passed"]
        and clutter["passed"]
        and background_texture["passed"]
        and lighting["passed"]
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "actor_id": "bell",
        "controlled_axis": expected_axis,
        "position": position,
        "instance": instance,
        "clutter": clutter,
        "background_texture": background_texture,
        "lighting": lighting,
        "authorities": [
            position["authority"],
            instance["authority"],
            clutter["authority"],
            background_texture["authority"],
            lighting["authority"],
        ],
    }


def validate_click_bell_scene_position(
    scene: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    """Backward-compatible view of the fixed-position contract."""

    return validate_click_bell_scene_contract(scene, spec)["position"]


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
            validate_click_bell_scene_contract(scene, spec)
            if is_click_bell and structural_probe_passed
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
            vision = run_vision_check(
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
            if not is_click_bell:
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
        if is_click_bell:
            raise VisualReflectionError(
                "click_bell bounded overlay is validate-only and does not support repair"
            )
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
        max_repairs,
        int(scene_check["repair_policy"]["max_repairs_supported"]),
    )
    summary = execute_reflection_loop(
        max_repairs=effective_max_repairs,
        observe=observe,
        repair=repair,
    )
    if is_click_bell:
        summary["requested_max_repairs"] = max_repairs
        summary["repair_supported"] = False
        summary[
            "validation_mode"
        ] = "simulator_position_or_instance_plus_visual_plausibility"
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
    summary["final_scene_source_sha256"] = refreshed_bundle["scene_method"][
        "source_sha256"
    ]
    update_manifest(
        run_dir,
        task_artifact_bundle="generation/task_artifact_bundle.json",
        scene_check_spec="generation/scene_check_spec.json",
        task_artifact_summary={
            "scene_origin": refreshed_bundle["scene_method"]["origin"],
            "success_origin": refreshed_bundle["success_method"]["origin"],
            "success_semantics_preserved": True,
        },
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


def newest_eval_dir(
    repo_root: Path,
    before: set[Path],
    *,
    task_name: str = "beat_block_hammer",
    task_config: str = "demo_clean",
    checkpoint_setting: str = "demo_clean",
) -> Path | None:
    eval_root = (
        repo_root / "eval_result" / task_name / "ACT" / task_config / checkpoint_setting
    )
    after = (
        {path for path in eval_root.glob("*") if path.is_dir()}
        if eval_root.exists()
        else set()
    )
    created = after - before
    return max(created, key=lambda path: path.stat().st_mtime) if created else None


def archive_previous_act_attempt(run_dir: Path) -> Path | None:
    """Preserve stale retry artifacts without mixing them into a new result."""

    evaluation_dir = run_dir / "evaluation"
    candidates = [
        *evaluation_dir.glob("episode*.mp4"),
        *(evaluation_dir / name for name in ("_result.txt", "act.json", "act.log")),
        evaluation_dir / "telemetry/act",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    archive_dir = evaluation_dir / "previous_act_attempts" / stamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), archive_dir / path.name)
    return archive_dir


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
    """Run a task-specific ACT checkpoint and attach videos to telemetry."""

    task_name = str(manifest["task_name"])
    task_config = str(manifest.get("task_config") or "demo_clean")
    checkpoint_setting = str(manifest.get("checkpoint_setting") or "demo_clean")
    expert_data_num = int(manifest.get("expert_data_num") or 50)
    policy_seed = int(manifest.get("policy_seed") or 0)
    checkpoint_dir = (
        repo_root
        / "policy/ACT/act_ckpt"
        / f"act-{task_name}"
        / f"{checkpoint_setting}-{expert_data_num}"
    )
    required_checkpoint_files = [
        checkpoint_dir / "policy_last.ckpt",
        checkpoint_dir / "dataset_stats.pkl",
    ]
    missing_checkpoint_files = [
        path for path in required_checkpoint_files if not path.is_file()
    ]
    if missing_checkpoint_files:
        missing = ", ".join(
            str(path.relative_to(repo_root)) for path in missing_checkpoint_files
        )
        raise RuntimeError(
            f"ACT checkpoint preflight failed for {task_name}: {missing}. "
            "Download it on the server with "
            f"`python scripts/download_act_checkpoint.py {task_name}`; "
            "do not relay routine checkpoints through a local workstation."
        )

    previous_attempt = archive_previous_act_attempt(run_dir)
    telemetry_root = run_dir / "evaluation/telemetry/act"
    eval_root = (
        repo_root / "eval_result" / task_name / "ACT" / task_config / checkpoint_setting
    )
    before = (
        {path for path in eval_root.glob("*") if path.is_dir()}
        if eval_root.exists()
        else set()
    )
    command = [
        "env",
        f"PYTHON_BIN={sys.executable}",
        "bash",
        "policy/ACT/eval_mea.sh",
        task_name,
        task_config,
        checkpoint_setting,
        str(expert_data_num),
        str(policy_seed),
        str(gpu),
        str(num_episodes),
        manifest["task_module"],
        str(run_dir / "overlay.yml"),
        str(seed),
        str(telemetry_root),
        telemetry_profile,
    ]
    started = datetime.now().astimezone().isoformat()
    record_act_batch_start(
        task_name=task_name,
        policy_name="ACT",
        start_seed=seed,
        num_rollouts=num_episodes,
    )
    returncode = run_command(
        command,
        cwd=repo_root,
        log_path=run_dir / "evaluation/act.log",
    )
    source_dir = newest_eval_dir(
        repo_root,
        before,
        task_name=task_name,
        task_config=task_config,
        checkpoint_setting=checkpoint_setting,
    )
    copied = []
    result_file_copied = False
    if source_dir:
        sources = sorted(source_dir.glob("episode*.mp4"))
        result_file = source_dir / "_result.txt"
        if result_file.is_file():
            sources.append(result_file)
        for source in sources:
            if source.is_file():
                destination = run_dir / "evaluation" / source.name
                shutil.copy2(source, destination)
                copied.append(str(destination.relative_to(repo_root)))
                if source.name == "_result.txt":
                    result_file_copied = True

    copied_video_paths = list((run_dir / "evaluation").glob("episode*.mp4"))
    telemetry_episode_paths = list(
        metadata.parent for metadata in telemetry_root.glob("episode_*/episode.json")
    )
    index_issues: list[str] = []
    video_by_index: dict[int, Path] = {}
    telemetry_by_index: dict[int, Path] = {}
    for video in copied_video_paths:
        match = re.fullmatch(r"episode(\d+)\.mp4", video.name)
        if match is None:
            index_issues.append(f"unrecognized ACT video name: {video.name}")
            continue
        episode_index = int(match.group(1))
        if episode_index in video_by_index:
            index_issues.append(f"duplicate ACT video index: {episode_index}")
            continue
        if video.stat().st_size <= 0:
            index_issues.append(f"empty ACT video: {video.name}")
        video_by_index[episode_index] = video
    for episode_dir in telemetry_episode_paths:
        match = re.match(r"episode_(\d+)(?:_|$)", episode_dir.name)
        if match is None:
            index_issues.append(
                f"unrecognized ACT telemetry directory: {episode_dir.name}"
            )
            continue
        episode_index = int(match.group(1))
        if episode_index in telemetry_by_index:
            index_issues.append(f"duplicate ACT telemetry index: {episode_index}")
            continue
        telemetry_by_index[episode_index] = episode_dir
    video_indices = set(video_by_index)
    telemetry_indices = set(telemetry_by_index)
    if video_indices != telemetry_indices:
        index_issues.append(
            "ACT video/telemetry indices differ: "
            f"videos={sorted(video_indices)}, telemetry={sorted(telemetry_indices)}"
        )
    paired_indices = sorted(video_indices & telemetry_indices)
    copied_videos = [video_by_index[index] for index in sorted(video_indices)]
    telemetry_episodes = [
        telemetry_by_index[index] for index in sorted(telemetry_indices)
    ]
    video_associations = []
    actual_seeds: list[int] = []
    for episode_index in paired_indices:
        episode_dir = telemetry_by_index[episode_index]
        video = video_by_index[episode_index]
        destination = episode_dir / "video.mp4"
        shutil.copy2(video, destination)
        metadata_path = episode_dir / "episode.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("seed") is not None:
                actual_seeds.append(int(metadata["seed"]))
            metadata.setdefault("artifacts", {})["video"] = "video.mp4"
            metadata["video_alignment"] = {
                "policy_frame_rate_hz": 10,
                "frame_semantics": "pre-action; contact in policy step k lies between adjacent frames",
            }
            write_json(metadata_path, metadata)
        video_associations.append(
            {
                "episode_dir": str(episode_dir.relative_to(repo_root)),
                "video": str(destination.relative_to(repo_root)),
                "episode_index": episode_index,
            }
        )

    result = {
        "command": command,
        "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(),
        "returncode": returncode,
        "task_name": task_name,
        "task_config": task_config,
        "checkpoint_setting": checkpoint_setting,
        "expert_data_num": expert_data_num,
        "policy_seed": policy_seed,
        "num_episodes": num_episodes,
        "actual_seeds": actual_seeds,
        "checkpoint": {
            "directory": str(checkpoint_dir.relative_to(repo_root)),
            "required_files": [
                str(path.relative_to(repo_root)) for path in required_checkpoint_files
            ],
            "preflight_passed": True,
        },
        "source_eval_dir": str(source_dir) if source_dir else None,
        "copied_artifacts": copied,
        "copied_video_count": len(copied_videos),
        "telemetry_root": str(telemetry_root.relative_to(repo_root)),
        "telemetry_episode_count": len(telemetry_episodes),
        "video_associations": video_associations,
        "episode_index_alignment": {
            "passed": not index_issues,
            "video_indices": sorted(video_indices),
            "telemetry_indices": sorted(telemetry_indices),
            "issues": index_issues,
        },
        "previous_attempt_archive": (
            str(previous_attempt.relative_to(repo_root))
            if previous_attempt is not None
            else None
        ),
        "passed": (
            returncode == 0
            and source_dir is not None
            and result_file_copied
            and not index_issues
            and len(copied_videos) == num_episodes
            and len(telemetry_episodes) == num_episodes
            and len(actual_seeds) == num_episodes
        ),
    }
    write_json(run_dir / "evaluation/act.json", result)
    if not result["passed"]:
        raise RuntimeError(f"ACT {num_episodes}-episode 未通过: {result}")
    return result


def evaluate_run_telemetry(
    repo_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    telemetry_root = run_dir / "evaluation/telemetry"
    summary = evaluate_telemetry_root(
        telemetry_root,
        user_request=manifest["user_request"],
        task_name=manifest["task_name"],
    )
    return {
        "artifact": str((telemetry_root / "tool_results.json").relative_to(repo_root)),
        "episode_count": summary["episode_count"],
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
        "--mode",
        choices=["reuse", "force_codegen", "official"],
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
    parser.add_argument(
        "--reflection-fixture",
        choices=["wrong_color", "oversized_block"],
        help="Test-only injected visual mismatch used to exercise the repair loop.",
    )
    parser.add_argument(
        "--success-spec-fixture",
        choices=["invalid_threshold"],
        help=(
            "Development-only invalid SuccessSpec used to prove diagnosis and one "
            "bounded repair before code generation."
        ),
    )
    parser.add_argument("--run-act", action="store_true")
    parser.add_argument(
        "--registration-identity-json",
        help="Parent Agent registration identity propagated to child/episode artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise SystemExit("--num-episodes 必须是正整数")
    if args.success_spec_fixture is not None and (
        args.resume_run
        or args.mode != "force_codegen"
        or args.task_name != "beat_block_hammer"
    ):
        raise SystemExit(
            "--success-spec-fixture requires a fresh beat_block_hammer force_codegen run"
        )
    repo_root = args.repo_root.expanduser().resolve()
    registration_identity: dict[str, Any] | None = None
    if args.registration_identity_json is not None:
        if args.resume_run:
            raise SystemExit("registered TaskGen execution cannot use --resume-run")
        try:
            registration_identity = validate_registration_identity(
                json.loads(args.registration_identity_json)
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"invalid --registration-identity-json: {exc}") from exc
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
        else:
            prototype = TaskGenPrototype(repo_root, provider, model=args.text_model)
            success_spec_candidate = None
            success_spec_max_repairs = 0
            if args.success_spec_fixture == "invalid_threshold":
                if args.mode != "force_codegen":
                    raise SystemExit(
                        "--success-spec-fixture requires --mode force_codegen"
                    )
                success_spec_candidate = default_bbh_success_spec()
                success_spec_candidate["predicates"][0]["thresholds_m"] = [
                    0.2,
                    0.2,
                ]
                success_spec_max_repairs = 1
            manifest = prototype.generate(
                args.request,
                task_name=args.task_name,
                mode=args.mode,
                run_id=args.run_id,
                variant_id=args.variant_id,
                trusted_variant_spec=trusted_variant_spec,
                success_spec_candidate=success_spec_candidate,
                success_spec_max_repairs=success_spec_max_repairs,
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
            task_artifact_summary={
                "scene_origin": bundle["scene_method"]["origin"],
                "success_origin": bundle["success_method"]["origin"],
                "success_semantics_preserved": True,
            },
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

    if registration_identity is not None:
        try:
            registration_identity = validate_registration_identity(
                registration_identity, run_id=str(manifest["run_id"])
            )
        except ValueError as exc:
            raise SystemExit(f"child registration binding failed: {exc}") from exc
        update_manifest(
            run_dir,
            registration_identity=registration_identity,
        )

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
        else ("act" if args.run_act else "expert" if args.expert else "setup_probe")
    )
    update_manifest(
        run_dir,
        requested_execution_backend=requested_execution_backend,
    )

    try:
        if manifest.get("mode") == "official" and (
            args.vision_check or args.reflection_fixture
        ):
            raise RuntimeError(
                "official route bypasses generated-scene vision/reflection; "
                "use expert, act, or both execution without scene codegen"
            )
        if args.reflection_fixture:
            if manifest.get("task_name") != "beat_block_hammer":
                raise RuntimeError(
                    "reflection fixtures are only defined for beat_block_hammer"
                )
            if args.resume_run:
                raise RuntimeError("reflection fixture 只允许用于新的 TaskGen run")
            if not args.vision_check:
                raise RuntimeError("reflection fixture 必须与 --vision-check 一起使用")
            spec = json.loads(
                (run_dir / "variant_spec.json").read_text(encoding="utf-8")
            )
            fixture_function = {
                "wrong_color": inject_wrong_color_fixture,
                "oversized_block": inject_oversized_block_fixture,
            }[args.reflection_fixture]
            fixture = fixture_function(
                repo_root, run_dir, spec, manifest["protected_hashes_before"]
            )
            update_manifest(run_dir, reflection_fixture=fixture)

        scene = None
        if args.vision_check:
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

        if manifest.get("mode") == "official" and args.expert:
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

        if args.run_act:
            if manifest["task_name"] == "beat_block_hammer":
                position_samples = collect_position_samples(
                    repo_root,
                    run_dir,
                    manifest,
                    start_seed=args.seed,
                    num_episodes=args.num_episodes,
                    first_scene=scene,
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
            if registration_identity is not None:
                bind_registration_to_episode_metadata(
                    run_dir, registration_identity
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
                execution_backends=(["expert", "ACT"] if args.expert else ["ACT"]),
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
                if registration_identity is not None:
                    bind_registration_to_episode_metadata(
                        run_dir, registration_identity
                    )
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
