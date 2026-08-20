"""Production runtime for catalog-independent RoboTwin TaskGen.

This module owns the importable generic TaskGen generation and simulator
preflight boundary. The legacy CLI re-exports compatibility wrappers; policy
execution and CLI argument handling remain outside this module.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from mea.planner.experiment_candidate import (
    ExperimentCandidateError,
    validate_experiment_candidate,
)
from mea.robotwin_task_context import (
    RoboTwinTaskContextError,
    resolve_robotwin_task_context,
)

from .artifact_index import (
    GenericTaskArtifactIndex,
    materialize_reused_generic_task,
)
from .attempts import CandidateUnexecutableError
from .generic_backend import (
    GenericRoboTwinTaskGenBackend,
    GenericTaskGenError,
    load_generic_robotwin_task_adapter,
)
from .generic_visual import (
    GenericVisualDiagnosisError,
    diagnose_generic_scene_render,
)
from .prototype import extract_json_response
from .provider_scene_checker import validate_provider_run_id
from .probe_runtime import (
    _checker_fixture_failure_diagnosis,
    _expert_terminal_authority_failure,
    _generated_checker_execution_failure,
    _tracked_actor_heights,
    _tracked_actor_positions,
    run_command,
    run_probe,
)
from .preservation import (
    _checker_references_official_core,
    _same_seed_tracked_actor_geometry,
    _same_seed_tracked_actor_state,
    build_preservation_report,
)


ProbeRunner = Callable[..., dict[str, Any]]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _tracked_actor_position_changes(
    official: Mapping[str, Any],
    generated: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return significant same-seed actor-position facts for later planning."""

    official_seed = official.get("seed")
    generated_seed = generated.get("seed")
    if (
        isinstance(official_seed, bool)
        or not isinstance(official_seed, int)
        or isinstance(generated_seed, bool)
        or not isinstance(generated_seed, int)
        or official_seed != generated_seed
    ):
        return []

    official_positions = _tracked_actor_positions(official)
    generated_positions = _tracked_actor_positions(generated)
    changes: list[dict[str, Any]] = []
    for actor_id in sorted(official_positions.keys() & generated_positions.keys()):
        official_position = official_positions[actor_id]
        generated_position = generated_positions[actor_id]
        for axis_index, axis in enumerate(("x", "y", "z")):
            signed_delta = (
                generated_position[axis_index]
                - official_position[axis_index]
            )
            if abs(signed_delta) <= 1e-6:
                continue
            changes.append(
                {
                    "actor_id": actor_id,
                    "property": "position",
                    "axis": axis,
                    "signed_delta": signed_delta,
                    "generated_value": generated_position[axis_index],
                    "unit": "m",
                    "comparison_seed": official_seed,
                    "authority": "same_seed_simulator_setup_state",
                }
            )
    return changes


def _requested_position_delta_report(
    candidate: Mapping[str, Any],
    official: Mapping[str, Any],
    generated: Mapping[str, Any],
    *,
    tolerance_m: float = 1e-5,
) -> dict[str, Any]:
    """Check typed Plan position deltas against same-seed simulator state."""

    scene_need = candidate.get("scene_need")
    requested = (
        scene_need.get("controlled_changes")
        if isinstance(scene_need, Mapping)
        else None
    )
    if not isinstance(requested, list) or not requested:
        return {
            "schema_version": 1,
            "status": "not_required",
            "verified": True,
            "tolerance_m": tolerance_m,
            "checks": [],
        }

    official_seed = official.get("seed")
    generated_seed = generated.get("seed")
    same_seed = bool(
        not isinstance(official_seed, bool)
        and isinstance(official_seed, int)
        and not isinstance(generated_seed, bool)
        and isinstance(generated_seed, int)
        and official_seed == generated_seed
    )
    official_positions = _tracked_actor_positions(official)
    generated_positions = _tracked_actor_positions(generated)
    axis_indices = {"x": 0, "y": 1, "z": 2}
    checks: list[dict[str, Any]] = []
    for change in requested:
        actor_id = str(change.get("actor") or "").strip()
        axis = change.get("axis")
        expected = change.get("signed_delta")
        official_position = official_positions.get(actor_id)
        generated_position = generated_positions.get(actor_id)
        observed: float | None = None
        if (
            same_seed
            and axis in axis_indices
            and official_position is not None
            and generated_position is not None
        ):
            index = axis_indices[str(axis)]
            observed = generated_position[index] - official_position[index]
        passed = bool(
            observed is not None
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and abs(observed - float(expected)) <= tolerance_m
        )
        checks.append(
            {
                "actor_id": actor_id,
                "property": "position",
                "axis": axis,
                "reference": change.get("reference"),
                "expected_signed_delta": expected,
                "observed_signed_delta": observed,
                "unit": "m",
                "comparison_seed": official_seed if same_seed else None,
                "passed": passed,
                "authority": (
                    "same_seed_simulator_setup_state"
                    if observed is not None
                    else "requested_delta_simulator_state_unavailable"
                ),
            }
        )
    verified = bool(checks) and all(item["passed"] for item in checks)
    return {
        "schema_version": 1,
        "status": "verified" if verified else "failed",
        "verified": verified,
        "tolerance_m": tolerance_m,
        "checks": checks,
    }

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
    probe_runner: ProbeRunner | None = None,
) -> dict[str, Any]:
    """Materialize one catalog-independent scene and/or checker candidate.

    The active checker is gated against two real simulator states before any
    learned-policy rollout: the untouched initial state must be negative, and
    the official expert terminal state must be positive.  A null checker need
    reuses the official method and retains official outcome authority; only a
    requested generated checker has experimental semantics.
    """

    probe_runner = probe_runner or run_probe
    run_id = validate_provider_run_id(
        run_id,
        error_type=GenericTaskGenError,
    )
    try:
        candidate = validate_experiment_candidate(experiment_candidate)
    except ExperimentCandidateError as exc:
        raise GenericTaskGenError(str(exc)) from exc
    generated_scene = candidate["scene_need"] is not None
    generated_checker = candidate["checker_need"] is not None
    outcome_label = (
        "generated_check_success"
        if generated_checker
        else "official_check_success"
    )
    request = str(user_request).strip()
    if not request:
        raise GenericTaskGenError("user_request must be non-empty")
    try:
        task_context = resolve_robotwin_task_context(
            repo_root,
            candidate["base_task"],
        )
    except RoboTwinTaskContextError as exc:
        raise GenericTaskGenError(str(exc)) from exc
    runtime_context_probe: Mapping[str, Any] | None = None
    task_context_probe_result: Mapping[str, Any] | None = None
    task_context_path: Path | None = None
    if task_context.task_schema is None:
        context_dir = (
            repo_root
            / "mea/generated_task_attempts"
            / f"{run_id}_task_context"
        )
        context_dir.mkdir(parents=True, exist_ok=True)
        context_overlay = context_dir / "overlay.yml"
        context_overlay.write_text("{}\n", encoding="utf-8")
        try:
            task_context_probe_result = probe_runner(
                repo_root,
                context_dir,
                {
                    "task_name": candidate["base_task"],
                    "task_module": f"envs.{candidate['base_task']}",
                },
                seed=seed,
                expert=False,
                scene_json=context_dir / "probe.json",
                image=context_dir / "initial_head.png",
                log_path=context_dir / "probe.log",
                telemetry_profile=telemetry_profile,
                discover_task_context=True,
                action_dimension=action_dimension,
            )
        except Exception as exc:
            raise GenericTaskGenError(
                "official reset could not establish TaskContext authority: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(task_context_probe_result, Mapping):
            raise GenericTaskGenError(
                "official reset returned an invalid TaskContext result"
            )
        raw_probe = task_context_probe_result.get("task_context_probe")
        if not isinstance(raw_probe, Mapping):
            raise GenericTaskGenError(
                "official reset returned no validated TaskContext probe"
            )
        runtime_context_probe = deepcopy(dict(raw_probe))
        try:
            task_context = resolve_robotwin_task_context(
                repo_root,
                candidate["base_task"],
                runtime_probe=runtime_context_probe,
            )
        except RoboTwinTaskContextError as exc:
            raise GenericTaskGenError(
                f"runtime TaskContext validation failed: {exc}"
            ) from exc
        task_context_path = context_dir / "task_context.json"
        write_json(task_context_path, task_context.to_dict())
    if task_context.task_schema is None:
        raise GenericTaskGenError(
            "TaskContext lacks simulator-authoritative actor/telemetry fields"
        )
    official_task_schema = deepcopy(dict(task_context.task_schema))
    accepted_preflight: dict[str, Any] = {}
    visual_attempts: list[dict[str, Any]] = []
    official_control: dict[str, Any] = {}
    visual_self_check_enabled = bool(
        True
        if ablation_switches is None
        else ablation_switches.get("visual_self_check")
    )

    def scene_projection(scene: Mapping[str, Any]) -> dict[str, Any]:
        """Keep simulator-native fields that can prove a materialized change."""

        return {
            "actors": scene.get("actors"),
            "tracked_actors": scene.get("tracked_actors"),
            "task_attributes": scene.get("task_attributes"),
            "domain_randomization": scene.get("domain_randomization"),
        }

    def scene_change_report(
        official: Mapping[str, Any],
        generated: Mapping[str, Any],
        *,
        official_image: Path,
        generated_image: Path,
    ) -> dict[str, Any]:
        official_projection = scene_projection(official)
        generated_projection = scene_projection(generated)
        changed_components = [
            key
            for key in official_projection
            if official_projection[key] != generated_projection[key]
        ]
        official_image_bytes = (
            official_image.read_bytes() if official_image.is_file() else None
        )
        generated_image_bytes = (
            generated_image.read_bytes() if generated_image.is_file() else None
        )
        image_changed = bool(
            official_image_bytes
            and generated_image_bytes
            and official_image_bytes != generated_image_bytes
        )
        scene_change_expected = candidate["scene_need"] is not None
        return {
            "schema_version": 1,
            "passed": bool(
                changed_components or image_changed
                if scene_change_expected
                else not changed_components
            ),
            "expected_state": (
                "changed" if scene_change_expected else "preserved"
            ),
            "authority": (
                "same_seed_official_vs_generated_simulator_state_and_render"
            ),
            "changed_components": changed_components,
            "tracked_actor_changes": _tracked_actor_position_changes(
                official, generated
            ),
            "render_changed": image_changed,
            "scene_need": candidate["scene_need"],
            "semantic_scope": (
                "observable_scene_difference; exact natural-language "
                "entailment remains a limitation"
            ),
        }

    def checker_fixtures(
        _methods: Mapping[str, str],
        _candidate: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        # Semantic fixtures are produced by the simulator preflight below.
        # Returning a fabricated Python-only fixture here would weaken the
        # paper claim, so the generic backend explicitly merges the live pair.
        return []

    def preflight_candidate(
        attempt_dir: Path,
        _module_source: str,
        _candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        preflight_runtime = {
            "simulator_probes": 0,
            "expert_probes": 0,
        }
        for package_dir in (
            repo_root / "mea/generated_task_attempts",
            attempt_dir.parent,
            attempt_dir,
        ):
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
        overlay_path = attempt_dir / "overlay.yml"
        overlay_path.write_text("{}\n", encoding="utf-8")
        module_path = attempt_dir / "candidate_task.py"
        if not module_path.is_file():
            module_path = attempt_dir / "task.py"
        if not module_path.is_file():
            raise GenericTaskGenError(
                "generic candidate module is unavailable for preflight"
            )
        task_module = (
            module_path.relative_to(repo_root).with_suffix("").as_posix()
            .replace("/", ".")
        )
        probe_manifest = {
            "task_name": candidate["base_task"],
            "task_module": task_module,
        }
        official_manifest = {
            "task_name": candidate["base_task"],
            "task_module": f"envs.{candidate['base_task']}",
        }
        official_control_dir = attempt_dir.parent / "official_control"
        official_setup_image = official_control_dir / "setup_head.png"
        if "setup" not in official_control:
            official_control_dir.mkdir(exist_ok=True)
            overlay_path = official_control_dir / "overlay.yml"
            if not overlay_path.exists():
                overlay_path.write_text("{}\n", encoding="utf-8")
            official_control["setup"] = probe_runner(
                repo_root,
                official_control_dir,
                official_manifest,
                seed=seed,
                expert=False,
                scene_json=official_control_dir / "setup.json",
                image=official_setup_image,
                log_path=official_control_dir / "setup.log",
                telemetry_profile=telemetry_profile,
                **(
                    {"task_context": task_context_path}
                    if task_context_path is not None
                    else {}
                ),
            )
            preflight_runtime["simulator_probes"] += 1
        official_setup = official_control["setup"]
        setup_image = attempt_dir / "setup_head.png"
        try:
            setup = probe_runner(
                repo_root,
                attempt_dir,
                probe_manifest,
                seed=seed,
                expert=False,
                scene_json=attempt_dir / "setup_preflight.json",
                image=setup_image,
                log_path=attempt_dir / "setup_preflight.log",
                telemetry_profile=telemetry_profile,
                **(
                    {"task_context": task_context_path}
                    if task_context_path is not None
                    else {}
                ),
            )
        except TimeoutError as exc:
            preflight_runtime["simulator_probes"] += 1
            raise GenericTaskGenError(
                f"generated setup probe failed: {exc}",
                runtime=preflight_runtime,
            ) from exc
        preflight_runtime["simulator_probes"] += 1
        requested_delta_report = _requested_position_delta_report(
            candidate,
            official_setup,
            setup,
        )
        if requested_delta_report["verified"] is not True:
            raise GenericTaskGenError(
                "generated scene did not realize the typed controlled "
                "position change: "
                + json.dumps(
                    requested_delta_report,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                runtime=preflight_runtime,
            )
        expert = probe_runner(
            repo_root,
            attempt_dir,
            probe_manifest,
            seed=seed,
            expert=True,
            scene_json=attempt_dir / "expert_preflight.json",
            image=attempt_dir / "expert_head.png",
            log_path=attempt_dir / "expert_preflight.log",
            max_expert_attempts=1,
            telemetry_profile=telemetry_profile,
            raise_on_failure=False,
            **(
                {"task_context": task_context_path}
                if task_context_path is not None
                else {}
            ),
        )
        preflight_runtime["expert_probes"] += max(
            len(expert.get("expert_attempts") or []), 1
        )
        checker_execution_failure = _generated_checker_execution_failure(
            expert
        )
        if checker_execution_failure is not None:
            raise GenericTaskGenError(
                "generated checker failed live execution: "
                + json.dumps(
                    checker_execution_failure,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                runtime=preflight_runtime,
            )
        expert_returncode = expert.get("returncode")
        if expert_returncode not in {None, 0, 2}:
            error = expert.get("error")
            detail = (
                json.dumps(error, ensure_ascii=False, sort_keys=True)
                if isinstance(error, Mapping)
                else "no structured simulator diagnosis"
            )
            raise GenericTaskGenError(
                "official expert probe execution failed before checker "
                f"validation: returncode={expert_returncode}; {detail}"
            )
        terminal_authority_failure = _expert_terminal_authority_failure(expert)
        if terminal_authority_failure is not None:
            if "expert" not in official_control:
                official_control["expert"] = probe_runner(
                    repo_root,
                    official_control_dir,
                    official_manifest,
                    seed=seed,
                    expert=True,
                    scene_json=official_control_dir / "expert.json",
                    image=official_control_dir / "expert_head.png",
                    log_path=official_control_dir / "expert.log",
                    max_expert_attempts=1,
                    telemetry_profile=telemetry_profile,
                    raise_on_failure=False,
                    **(
                        {"task_context": task_context_path}
                        if task_context_path is not None
                        else {}
                    ),
                )
                preflight_runtime["expert_probes"] += max(
                    len(
                        official_control["expert"].get("expert_attempts")
                        or []
                    ),
                    1,
                )
            official_failure = _expert_terminal_authority_failure(
                official_control["expert"]
            )
            if official_failure is not None:
                raise GenericTaskGenError(
                    "official same-seed expert baseline is unavailable: "
                    + json.dumps(
                        official_failure,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    runtime=preflight_runtime,
                )
            raise GenericTaskGenError(
                "generated scene/expert failed official terminal-state "
                "authority: "
                + json.dumps(
                    terminal_authority_failure,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                runtime=preflight_runtime,
            )
        fixtures = [
            {
                "fixture_id": "simulator_initial_negative",
                "expected": False,
                "observed": bool(setup.get("initial_check_success")),
                "passed": setup.get("initial_check_success") is False,
                "authority": "fresh_simulator_state_before_action",
            },
            {
                "fixture_id": "official_expert_terminal_positive",
                "expected": True,
                "observed": bool(
                    (expert.get("expert") or {}).get("check_success")
                ),
                "passed": bool(
                    (expert.get("expert") or {}).get("check_success")
                ),
                "authority": "official_expert_terminal_state",
            },
        ]
        scene_change = scene_change_report(
            official_setup,
            setup,
            official_image=official_setup_image,
            generated_image=setup_image,
        )
        visual: dict[str, Any]
        if visual_self_check_enabled:
            try:
                visual = diagnose_generic_scene_render(
                    provider,
                    candidate,
                    official_image=official_setup_image,
                    generated_image=setup_image,
                    output_dir=attempt_dir / "visual",
                    model=vision_model,
                    scene_change_passed=bool(scene_change["passed"]),
                )
            except GenericVisualDiagnosisError as exc:
                visual_attempts.append(
                    {
                        "attempt": attempt_dir.name,
                        "status": "invalid_response",
                        "passed": False,
                        "diagnosis": str(exc),
                    }
                )
                raise GenericTaskGenError(
                    f"visual diagnosis was invalid: {exc}"
                ) from exc
            visual_attempts.append(
                {
                    "attempt": attempt_dir.name,
                    "status": "passed" if visual["passed"] else "failed",
                    "passed": bool(visual["passed"]),
                    "diagnosis": visual["diagnosis"],
                    "repair_instructions": list(
                        visual["repair_instructions"]
                    ),
                }
            )
            if not visual["passed"]:
                repair = "; ".join(visual["repair_instructions"])
                raise GenericTaskGenError(
                    "visual diagnosis rejected the generated scene: "
                    + visual["diagnosis"]
                    + (f"; repair: {repair}" if repair else "")
                )
        else:
            visual = {
                "schema_version": 1,
                "status": "disabled_by_ablation",
                "passed": None,
                "model_requested": vision_model,
            }
        evaluation_intent = candidate.get("evaluation_intent")
        preserved_conditions = (
            list(evaluation_intent.get("preserved_conditions") or [])
            if isinstance(evaluation_intent, Mapping)
            else []
        )
        preservation_report = build_preservation_report(
            preserved_conditions,
            scene_generated=generated_scene,
            checker_generated=generated_checker,
            checker_references_official_core=(
                _checker_references_official_core(_module_source)
                if generated_checker
                else None
            ),
            visual_self_check_enabled=visual_self_check_enabled,
            visual=visual,
            official_setup=official_setup,
            generated_setup=setup,
        )
        result = {
            "schema_version": 1,
            "render_passed": bool(
                setup.get("render_success") and expert.get("render_success")
            ),
            "expert_passed": bool(
                (expert.get("expert") or {}).get("passed")
            ),
            "simulator_probes": preflight_runtime["simulator_probes"],
            "expert_probes": preflight_runtime["expert_probes"],
            "scene_change_passed": scene_change["passed"],
            "scene_change": scene_change,
            "requested_scene_change_validation": requested_delta_report,
            "vision_validation": visual,
            "preserved_conditions_verified": preservation_report["verified"],
            "preserved_conditions": preserved_conditions,
            "preservation_status": preservation_report["status"],
            "preservation_checks": preservation_report["checks"],
            "preservation_authority": "per_condition_authority",
            "checker_fixtures": fixtures,
            "initial_actor_z_m": _tracked_actor_heights(setup),
            "expert_terminal_actor_z_m": _tracked_actor_heights(expert),
            "official_setup_scene": str(
                (official_control_dir / "setup.json").relative_to(repo_root)
            ).replace("\\", "/"),
            "official_setup_image": str(
                official_setup_image.relative_to(repo_root)
            ).replace("\\", "/"),
            "setup_scene": str(
                (attempt_dir / "setup_preflight.json").relative_to(repo_root)
            ).replace("\\", "/"),
            "expert_scene": str(
                (attempt_dir / "expert_preflight.json").relative_to(repo_root)
            ).replace("\\", "/"),
            "setup_image": str(
                (attempt_dir / "setup_head.png").relative_to(repo_root)
            ).replace("\\", "/"),
            "expert_image": str(
                (attempt_dir / "expert_head.png").relative_to(repo_root)
            ).replace("\\", "/"),
        }
        if visual_self_check_enabled:
            for key, name in (
                ("visual_comparison_image", "official_vs_generated.png"),
                ("visual_prompt", "vision_prompt.md"),
                ("visual_response", "vision_response.txt"),
                ("visual_result", "vision.json"),
            ):
                result[key] = str(
                    (attempt_dir / "visual" / name).relative_to(repo_root)
                ).replace("\\", "/")
        if not all(item["passed"] for item in fixtures):
            raise GenericTaskGenError(
                _checker_fixture_failure_diagnosis(
                    fixtures,
                    setup=setup,
                    expert=expert,
                    success_contract=official_task_schema.get(
                        "success_contract"
                    ),
                )
            )
        if not scene_change["passed"]:
            raise GenericTaskGenError(
                "generated scene is not observably different from the "
                "same-seed official control"
            )
        if preservation_report["verified"] is False:
            failed_conditions = [
                item["condition"]
                for item in preservation_report["checks"]
                if item["verified"] is False
            ]
            raise GenericTaskGenError(
                "generated task violated a checked preservation condition: "
                + ", ".join(failed_conditions),
                runtime=preflight_runtime,
            )
        accepted_preflight.clear()
        accepted_preflight.update(result)
        return result

    adapter = load_generic_robotwin_task_adapter(
        repo_root,
        candidate["base_task"],
        checker_fixtures=checker_fixtures,
        preflight_candidate=preflight_candidate,
        resolve_metric=lambda _candidate: outcome_label,
        resolve_checker_contract=lambda _candidate: {
            "outcome_label": outcome_label,
            "act_runtime_eligible": True,
            "preserved": not generated_checker,
            "official_equivalent": not generated_checker,
            "semantic_scope": (
                "query_derived_experimental_predicate"
                if generated_checker
                else "official_check_success_reused"
            ),
        },
        prompt_constraints=(
            "Keep the official class identity and policy action interface. "
            "Use only assets and simulator APIs present in retrieved context. "
            "The generated initial scene must differ observably from the "
            "same-seed official scene in simulator state or rendered pixels "
            "when scene_need is non-null; when scene_need is null, preserve "
            "the official load_actors implementation exactly. When "
            "checker_need is null, preserve official check_success exactly. "
            "SAPIEN Pose.p and Pose.q values must not be modified by indexed "
            "assignment or +=/-= because those writes do not update the Pose; "
            "construct a new sapien.Pose from a copied position array and the "
            "original quaternion before passing it to create_actor. "
            "The upstream create_actor scale argument is normally replaced "
            "by asset model_data. scale_multiplier is the final/original "
            "size ratio: increase by 50% uses 1.5; reduce by 50%, or reduce "
            "to 50%, uses 0.5. Use scale_override only for "
            "a known absolute asset scale. Both opt-ins update the built mesh "
            "scale and Actor point metadata. "
            "If load_actors adds an actor that later measurement may need, "
            "also assign self.mea_telemetry_tracked_actors to a list of dicts "
            "with exactly id, task_attribute, scene_name, functional_points, "
            "contact_points, and contact_focus; task_attribute must name the "
            "public self attribute holding that actor, and contact_focus must "
            "be a boolean. Actors already listed in the TASK "
            "TELEMETRY/EXECUTION SCHEMA remain tracked automatically when "
            "their pose or instance is replaced: do not assign "
            "mea_telemetry_tracked_actors merely to repeat them. Include only "
            "entirely new actors in that list. Every new actor must "
            "have a unique simulator/contact identity distinct from every "
            "base actor: pass a unique runtime_name to create_actor when the "
            "asset modelname is reused, and declare that exact runtime "
            "get_name() value as scene_name. The asset "
            "modelname is not a unique runtime identity. Do not redeclare an "
            "actor already present in the TASK TELEMETRY/EXECUTION SCHEMA; "
            "that schema remains valid when the generated scene replaces the "
            "same public actor attribute and scene name. "
            "The initial state must not satisfy check_success; the official "
            "expert terminal state must satisfy it."
        ),
        runtime_probe=runtime_context_probe,
    )
    artifact_index = GenericTaskArtifactIndex(repo_root)
    resolution = GenericRoboTwinTaskGenBackend(
        repo_root,
        provider,
        model=model,
        find_exact=artifact_index.find_exact,
    ).materialize(
        candidate,
        adapter,
        run_id=run_id,
        max_regenerations=1,
        ablation_switches=ablation_switches,
    )
    reused_manifest: dict[str, Any] | None = None
    if resolution["status"] == "reused":
        reused_manifest = materialize_reused_generic_task(
            repo_root,
            run_id=run_id,
            user_request=request,
            candidate=candidate,
            resolution=resolution,
        )
    elif resolution["status"] != "generated":
        raise GenericTaskGenError(
            "generic TaskGen returned an unsupported resolution status"
        )
    run_dir = repo_root / "mea/generated_tasks" / run_id
    for child in ("generation", "validation", "evidence", "evaluation"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "validation/task_context.json",
        task_context.to_dict(),
    )
    if reused_manifest is None:
        moves = {
            "proposal_prompt.md": "generation/code_prompt.md",
            "provider_response.txt": "generation/provider_response.txt",
            "proposal.json": "generation/proposal.json",
            "checker_fixtures.json": "validation/checker_fixtures.json",
            "task_generation_result.json": (
                "generation/task_generation_result.json"
            ),
        }
        for source_name, destination_name in moves.items():
            source = run_dir / source_name
            if not source.is_file():
                raise GenericTaskGenError(
                    f"generic candidate artifact is missing: {source_name}"
                )
            shutil.move(str(source), str(run_dir / destination_name))
        write_json(
            run_dir / "generation/provider_response.json",
            extract_json_response(
                (run_dir / "generation/provider_response.txt").read_text(
                    encoding="utf-8"
                )
            ),
        )
    else:
        # Exact reuse skips provider code generation, not simulator
        # acceptance. Re-run the same-seed scene/checker gates before ACT.
        preflight_candidate(
            run_dir,
            (run_dir / "task.py").read_text(encoding="utf-8"),
            candidate,
        )
    (run_dir / "overlay.yml").write_text("{}\n", encoding="utf-8")
    write_json(run_dir / "request.json", {"user_request": request})

    def copy_preflight(key: str, destination: str) -> dict[str, Any]:
        relative = accepted_preflight.get(key)
        if not isinstance(relative, str):
            raise GenericTaskGenError(f"generic preflight lacks {key}")
        source = repo_root / relative
        target = run_dir / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return (
            json.loads(target.read_text(encoding="utf-8"))
            if target.suffix == ".json"
            else {}
        )

    setup_scene = copy_preflight(
        "setup_scene", "validation/setup_preflight.json"
    )
    copy_preflight(
        "official_setup_scene",
        "validation/official_setup_preflight.json",
    )
    expert_scene = copy_preflight(
        "expert_scene", "validation/expert_preflight.json"
    )
    copy_preflight("setup_image", "evidence/initial_head.png")
    copy_preflight(
        "official_setup_image",
        "evidence/official_initial_head.png",
    )
    copy_preflight("expert_image", "evidence/expert_head.png")
    if visual_self_check_enabled:
        copy_preflight(
            "visual_comparison_image",
            "evidence/scene_comparison.png",
        )
        copy_preflight(
            "visual_prompt",
            "validation/vision_prompt.md",
        )
        copy_preflight(
            "visual_response",
            "validation/vision_response.txt",
        )
        copy_preflight(
            "visual_result",
            "validation/vision.json",
        )
    run_local_preflight = deepcopy(accepted_preflight)
    run_local_preflight.update(
        {
            "official_setup_scene": (
                f"mea/generated_tasks/{run_id}/validation/"
                "official_setup_preflight.json"
            ),
            "setup_scene": (
                f"mea/generated_tasks/{run_id}/validation/"
                "setup_preflight.json"
            ),
            "expert_scene": (
                f"mea/generated_tasks/{run_id}/validation/"
                "expert_preflight.json"
            ),
            "official_setup_image": (
                f"mea/generated_tasks/{run_id}/evidence/"
                "official_initial_head.png"
            ),
            "setup_image": (
                f"mea/generated_tasks/{run_id}/evidence/initial_head.png"
            ),
            "expert_image": (
                f"mea/generated_tasks/{run_id}/evidence/expert_head.png"
            ),
        }
    )
    if reused_manifest is not None:
        reused_manifest["scene_validation"] = {
            **expert_scene,
            "setup_fixture": setup_scene,
            "generic_preflight": run_local_preflight,
        }
        reused_manifest["task_generation_acceptance"] = {
            **dict(
                reused_manifest.get("task_generation_acceptance") or {}
            ),
            "status": "accepted",
            "scope": (
                "exact_reuse_with_current_seed_render_expert_fixtures"
            ),
            "act_rollouts_started_before_acceptance": 0,
            "checker_fixture_count": len(
                run_local_preflight["checker_fixtures"]
            ),
            "scene_change_passed": run_local_preflight[
                "scene_change_passed"
            ],
            "scene_change_authority": run_local_preflight[
                "scene_change"
            ]["authority"],
            "visual_self_check_required": visual_self_check_enabled,
            "visual_self_check_passed": (
                run_local_preflight["vision_validation"].get("passed")
                if visual_self_check_enabled
                else None
            ),
            "preservation_status": run_local_preflight.get(
                "preservation_status"
            ),
            "preserved_conditions_verified": run_local_preflight.get(
                "preserved_conditions_verified"
            ),
        }
        reused_manifest["task_context"] = {
            "path": "validation/task_context.json",
            "schema_origin": task_context.schema_origin,
            "taskgen_ready": task_context.taskgen_ready,
            "runtime_probe_executed": task_context_probe_result is not None,
        }
        reused_manifest["vision_validation"] = (
            {
                **deepcopy(run_local_preflight["vision_validation"]),
                "status": "passed",
                "attempt_count": len(visual_attempts),
                "repairs_triggered_by_visual_failure": 0,
                "attempts": deepcopy(visual_attempts),
                "artifacts": {
                    "comparison_image": "evidence/scene_comparison.png",
                    "prompt": "validation/vision_prompt.md",
                    "response": "validation/vision_response.txt",
                    "result": "validation/vision.json",
                },
            }
            if visual_self_check_enabled
            else {
                "status": "disabled_by_ablation",
                "passed": None,
                "model_requested": vision_model,
                "attempt_count": 0,
                "repairs_triggered_by_visual_failure": 0,
            }
        )
        reused_manifest.setdefault("provider", {}).update(
            {
                "vision_provider_call_count": len(visual_attempts),
                "visual_model_requested": vision_model,
            }
        )
        write_json(run_dir / "manifest.json", reused_manifest)
        exact_match = resolution["exact_match"]
        reused_manifest["artifact_registry"] = {
            "kind": "task",
            "semantic_key": deepcopy(resolution["semantic_key"]),
            "artifact_path": exact_match["artifact_manifest"],
            "index_path": str(
                artifact_index.registry.index_path.relative_to(repo_root)
            ).replace("\\", "/"),
        }
        write_json(run_dir / "manifest.json", reused_manifest)
        for name in (
            "official_setup_preflight.json",
            "official_setup_preflight.log",
            "official_setup_head.png",
            "setup_preflight.json",
            "setup_preflight.log",
            "setup_head.png",
            "expert_preflight.json",
            "expert_preflight.log",
            "expert_head.png",
        ):
            (run_dir / name).unlink(missing_ok=True)
        return reused_manifest
    candidate_manifest = resolution["candidate_manifest"]
    validation = resolution["validation"]
    static_validation = {
        "provider_scene_checker": {
            "valid": True,
            "ast_policy": validation["policy"],
            "model_written_python": True,
            "restricted_success_spec_compiler_used": False,
            "checker_fixture_count": validation["checker_fixture_count"],
            "checker_fixture_pass_count": sum(
                1
                for item in validation["checker_fixtures"]
                if item.get("passed") is True
            ),
        }
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "generated",
        "created_at": datetime.now().astimezone().isoformat(),
        "user_request": request,
        "task_name": candidate["base_task"],
        "task_module": candidate_manifest["task_module"],
        "mode": "generic_provider_scene_checker_codegen",
        "generation_kind": "generic_provider_scene_checker_codegen",
        "overlay": str((run_dir / "overlay.yml").relative_to(repo_root)).replace(
            "\\", "/"
        ),
        "telemetry_profile": telemetry_profile,
        "task_context": {
            "path": "validation/task_context.json",
            "schema_origin": task_context.schema_origin,
            "taskgen_ready": task_context.taskgen_ready,
            "runtime_probe_executed": task_context_probe_result is not None,
        },
        "static_validation": static_validation,
        "scene_validation": {
            **expert_scene,
            "setup_fixture": setup_scene,
            "generic_preflight": run_local_preflight,
        },
        "vision_validation": (
            {
                **deepcopy(accepted_preflight["vision_validation"]),
                "status": "passed",
                "attempt_count": len(visual_attempts),
                "repairs_triggered_by_visual_failure": sum(
                    1
                    for item in visual_attempts[:-1]
                    if item.get("passed") is False
                ),
                "attempts": deepcopy(visual_attempts),
                "artifacts": {
                    "comparison_image": "evidence/scene_comparison.png",
                    "prompt": "validation/vision_prompt.md",
                    "response": "validation/vision_response.txt",
                    "result": "validation/vision.json",
                },
            }
            if visual_self_check_enabled
            else {
                "status": "disabled_by_ablation",
                "passed": None,
                "model_requested": vision_model,
                "attempt_count": 0,
                "repairs_triggered_by_visual_failure": 0,
            }
        ),
        "provider": {
            "model_requested": model,
            "called": True,
            "local_repair_count": resolution["local_repair_count"],
            "provider_call_count": resolution["provider_call_count"],
            "vision_provider_call_count": len(visual_attempts),
        },
        "taskgen_ablation_switches": candidate_manifest[
            "codegen_provenance"
        ]["taskgen_ablation_switches"],
        "taskgen_prompt_components": candidate_manifest[
            "codegen_provenance"
        ]["prompt_components"],
        "proposal": candidate,
        "proposal_path": "generation/proposal.json",
        "checker_contract": candidate_manifest["checker_contract"],
        "candidate_manifest": "candidate_manifest.json",
        "generic_taskgen_resolution": "generic_taskgen_resolution.json",
        "task_generation_acceptance": {
            "status": "accepted",
            "scope": "generic_live_fixture_render_expert_before_policy",
            "act_rollouts_started_before_acceptance": 0,
            "checker_fixture_count": validation["checker_fixture_count"],
            "scene_change_passed": accepted_preflight[
                "scene_change_passed"
            ],
            "scene_change_authority": accepted_preflight[
                "scene_change"
            ]["authority"],
            "visual_self_check_required": visual_self_check_enabled,
            "visual_self_check_passed": (
                accepted_preflight["vision_validation"].get("passed")
                if visual_self_check_enabled
                else None
            ),
            "preservation_status": accepted_preflight.get(
                "preservation_status"
            ),
            "preserved_conditions_verified": accepted_preflight.get(
                "preserved_conditions_verified"
            ),
        },
        "task_artifact_summary": {
            "scene_origin": (
                "provider_generated_code"
                if generated_scene
                else "official_method_reuse"
            ),
            "success_origin": (
                "provider_generated_python"
                if generated_checker
                else "official_method_reuse"
            ),
            "success_semantics_preserved": not generated_checker,
            "success_official_equivalent": not generated_checker,
            "success_compiler_eligible": not generated_checker,
            "success_act_eligible": True,
            "success_execution_scope": (
                "provider_generated_checker"
                if generated_checker
                else "official_equivalent"
            ),
            "success_outcome_label": outcome_label,
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    if ablation_switches is not None:
        manifest["artifact_registry"] = {
            "status": "disabled_for_codegen_ablation",
            "reason": (
                "Table 3 conditions generate independently and never "
                "occupy or reuse the production semantic artifact key"
            ),
        }
        write_json(run_dir / "manifest.json", manifest)
        return manifest
    artifact_entry = artifact_index.register_generated(
        resolution=resolution,
        manifest_path=run_dir / "manifest.json",
    )
    manifest["artifact_registry"] = {
        "kind": artifact_entry["kind"],
        "semantic_key": artifact_entry["semantic_key"],
        "artifact_path": artifact_entry["artifact_path"],
        "index_path": str(
            artifact_index.registry.index_path.relative_to(repo_root)
        ).replace("\\", "/"),
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def record_generic_taskgen_generation_failure(
    repo_root: Path,
    *,
    run_id: str,
    user_request: str,
    experiment_candidate: Mapping[str, Any],
    model: str,
    telemetry_profile: str,
    error: Exception,
) -> dict[str, Any]:
    """Leave a compact child manifest when generation exhausts its repair."""

    candidate = validate_experiment_candidate(experiment_candidate)
    run_dir = repo_root / "mea/generated_tasks" / run_id
    for child in ("generation", "validation", "evidence", "evaluation"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "request.json", {"user_request": user_request})
    write_json(
        run_dir / "generation/proposal.json",
        candidate,
    )
    result_source = (
        repo_root
        / "mea/generated_task_attempts"
        / run_id
        / "task_generation_result.json"
    )
    result_artifact = None
    generation_result: dict[str, Any] = {}
    if result_source.is_file():
        result_target = run_dir / "validation/task_generation_result.json"
        shutil.copy2(result_source, result_target)
        result_artifact = str(result_target.relative_to(repo_root)).replace(
            "\\", "/"
        )
        try:
            candidate_result = json.loads(result_source.read_text(encoding="utf-8"))
            if isinstance(candidate_result, Mapping):
                generation_result = dict(candidate_result)
        except (OSError, json.JSONDecodeError):
            pass
    candidate_unexecutable = isinstance(error, CandidateUnexecutableError)
    error_result = getattr(error, "result", {})
    if isinstance(error_result, Mapping) and error_result:
        generation_result = dict(error_result)
    attempt_runtime = generation_result.get("runtime") or {}
    final_failure = (
        (generation_result.get("repair") or {}).get("failure")
        or (generation_result.get("generation") or {}).get("failure")
        or {}
    )
    expert_failure_kind = (
        str(final_failure.get("failure_kind") or "").strip()
        if candidate_unexecutable
        else ""
    )
    if candidate_unexecutable and expert_failure_kind not in {
        "candidate_unexecutable",
        "official_baseline_unsolvable",
    }:
        expert_failure_kind = "candidate_unexecutable"
    policy_rollouts_started = (
        int(attempt_runtime.get("act_rollouts_started", 0))
        if isinstance(attempt_runtime, Mapping)
        else 0
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": (
            "candidate_unexecutable" if candidate_unexecutable else "failed"
        ),
        "created_at": datetime.now().astimezone().isoformat(),
        "user_request": str(user_request),
        "task_name": candidate["base_task"],
        "task_module": None,
        "mode": "generic_provider_scene_checker_codegen",
        "generation_kind": "generic_provider_scene_checker_codegen",
        "telemetry_profile": telemetry_profile,
        "provider": {
            "model_requested": model,
            "called": True,
        },
        "proposal": candidate,
        "proposal_path": "generation/proposal.json",
        "task_generation_result": result_artifact,
        "failure": {
            "stage": (
                "taskgen_expert_gate"
                if candidate_unexecutable
                else "provider_scene_checker_generation"
            ),
            "failure_kind": (
                expert_failure_kind
                if candidate_unexecutable
                else final_failure.get("failure_kind")
            ),
            "type": type(error).__name__,
            "message": str(error),
            "diagnosis": (
                final_failure.get("message")
                if candidate_unexecutable
                else None
            ),
        },
        "policy_execution": {
            "started": policy_rollouts_started > 0,
            "rollouts_started": policy_rollouts_started,
            "sample_count": 0,
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest
