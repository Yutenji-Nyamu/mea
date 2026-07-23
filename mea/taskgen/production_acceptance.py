"""Fail-closed TaskGen acceptance barrier used before policy execution.

Task generation already has stage-specific repair implementations (SuccessSpec
repair, visual reflection, and expert solvability checks).  This module does
not duplicate them.  It gives the production path one append-only boundary
that verifies their final artifacts and proves that no ACT rollout started
before the candidate was accepted.
"""

from __future__ import annotations

import ast
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any, Mapping

import yaml

from .artifacts import (
    TaskArtifactBundleError,
    validate_task_artifact_bundle,
    write_task_artifact_bundle,
)
from .attempts import (
    TaskGenerationRecoveryError,
    TaskGenerationStageError,
    run_bounded_task_generation,
)
from .click_bell import ClickBellTaskGenError, compile_click_bell_overlay
from .prototype import (
    TaskGenError,
    compile_overlay,
    validate_load_actors,
    validate_variant_spec,
)
from .scene_checks import SceneCheckSpecError, validate_scene_check_spec
from .capabilities import CapabilityError, validate_variant_spec_envelope
from .success_spec import SuccessSpecError, success_spec_validation_report
from .reviewed_registry import RUNTIME_DEPENDENCY_PATHS


class ProductionTaskAcceptanceError(RuntimeError):
    """Raised when the final pre-policy TaskGen candidate is not acceptable."""


_REVIEWED_BASE_ARTIFACTS = frozenset(
    {
        "task.py",
        "variant_spec.json",
        "overlay.yml",
        "generation/load_actors.py.txt",
        "generation/task_artifact_bundle.json",
        "generation/scene_check_spec.json",
        "validation/static.json",
    }
)
_REVIEWED_OPTIONAL_ARTIFACTS = frozenset({"generation/success_spec.json"})
_RUN_LOCAL_DERIVED_ARTIFACTS = frozenset(
    {
        "generation/task_artifact_bundle.json",
        "generation/scene_check_spec.json",
    }
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_reviewed_provenance(
    run_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    """Pin reviewed executable inputs while allowing declared run-local bindings.

    TaskArtifactBundle and SceneCheckSpec bind the current run module and current
    proposal id, so reviewed reuse deliberately rebuilds those two files.  Every
    executable or review-evidence input remains byte-identical to the registry
    copy and the explicitly pinned runtime Python dependencies must not drift.
    """

    if manifest.get("generation_kind") != "reviewed_generated_task_reuse":
        return {}
    registration = manifest.get("reviewed_task_registration")
    if not isinstance(registration, Mapping):
        raise ProductionTaskAcceptanceError(
            "reviewed Task registration provenance is missing"
        )
    copied = registration.get("copied_files")
    if not isinstance(copied, Mapping):
        raise ProductionTaskAcceptanceError("reviewed Task copied-file hashes are missing")
    copied_keys = set(copied)
    allowed_sets = {
        _REVIEWED_BASE_ARTIFACTS,
        _REVIEWED_BASE_ARTIFACTS | _REVIEWED_OPTIONAL_ARTIFACTS,
    }
    if copied_keys not in allowed_sets:
        raise ProductionTaskAcceptanceError(
            "reviewed Task copied-file set differs from the registry contract"
        )

    immutable_hashes: dict[str, str] = {}
    for relative in sorted(copied_keys - _RUN_LOCAL_DERIVED_ARTIFACTS):
        expected = copied.get(relative)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ProductionTaskAcceptanceError(
                f"reviewed Task hash is invalid: {relative}"
            )
        path = run_dir / relative
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ProductionTaskAcceptanceError(
                f"reviewed Task artifact is missing: {relative}: {exc}"
            ) from exc
        if path.is_symlink() or not resolved.is_relative_to(run_dir):
            raise ProductionTaskAcceptanceError(
                f"reviewed Task artifact escaped the run: {relative}"
            )
        actual = _file_sha256(resolved)
        if actual != expected:
            raise ProductionTaskAcceptanceError(
                f"reviewed Task immutable artifact changed: {relative}"
            )
        immutable_hashes[relative] = actual

    repo_root = run_dir.parents[2]
    runtime_expected = registration.get("runtime_dependency_hashes")
    if not isinstance(runtime_expected, Mapping) or set(runtime_expected) != set(
        RUNTIME_DEPENDENCY_PATHS
    ):
        raise ProductionTaskAcceptanceError(
            "reviewed Task runtime dependency contract is incomplete"
        )
    runtime_hashes: dict[str, str] = {}
    for relative in RUNTIME_DEPENDENCY_PATHS:
        expected = runtime_expected.get(relative)
        path = repo_root / relative
        if not isinstance(expected, str) or not path.is_file():
            raise ProductionTaskAcceptanceError(
                f"reviewed Task runtime dependency is missing: {relative}"
            )
        actual = _file_sha256(path)
        if actual != expected:
            raise ProductionTaskAcceptanceError(
                f"reviewed Task runtime dependency changed: {relative}"
            )
        runtime_hashes[relative] = actual

    return {
        "reviewed_immutable_artifacts_sha256": _canonical_sha256(
            immutable_hashes
        ),
        "reviewed_runtime_dependencies_sha256": _canonical_sha256(runtime_hashes),
    }


def _declared_method_source(
    path: Path, *, class_name: str, method_name: str
) -> str:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ProductionTaskAcceptanceError(f"invalid task source: {exc}") from exc
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    methods = (
        [
            node
            for node in classes[0].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        ]
        if len(classes) == 1
        else []
    )
    if len(methods) != 1:
        raise ProductionTaskAcceptanceError(
            f"task source must declare exactly one {class_name}.{method_name}"
        )
    lines = source.splitlines()
    return textwrap.dedent(
        "\n".join(lines[methods[0].lineno - 1 : methods[0].end_lineno]) + "\n"
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionTaskAcceptanceError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProductionTaskAcceptanceError(f"{label} must be a JSON object")
    return value


def _provider_calls(manifest: Mapping[str, Any], run_dir: Path) -> int:
    provider = manifest.get("provider")
    if not isinstance(provider, Mapping):
        return 0
    calls = provider.get("calls")
    text_calls = len(calls) if isinstance(calls, Mapping) else int(
        bool(provider.get("called"))
    )
    vision_calls = len(list((run_dir / "reflection").glob("attempt_*/vision_response.txt")))
    repair_calls = len(list((run_dir / "reflection").glob("attempt_*/repair_response.txt")))
    return text_calls + vision_calls + repair_calls


def _expert_probe_count(scene: Mapping[str, Any] | None) -> int:
    if not isinstance(scene, Mapping):
        return 0
    batch = scene.get("expert_batch")
    if isinstance(batch, Mapping) and isinstance(batch.get("episodes"), list):
        return len(batch["episodes"])
    return int(isinstance(scene.get("expert"), Mapping))


def _simulator_probe_count(scene: Mapping[str, Any] | None) -> int:
    if not isinstance(scene, Mapping):
        return 0
    attempts = scene.get("expert_attempts")
    if isinstance(attempts, list) and attempts:
        return len(attempts)
    batch = scene.get("expert_batch")
    if isinstance(batch, Mapping) and isinstance(batch.get("episodes"), list):
        return len(batch["episodes"])
    return 1


def _act_rollouts(manifest: Mapping[str, Any]) -> int:
    act = manifest.get("act_evaluation")
    if not isinstance(act, Mapping):
        return 0
    seeds = act.get("actual_seeds")
    return len(seeds) if isinstance(seeds, list) else 0


def _scene_checks(
    scene: Mapping[str, Any] | None,
    *,
    require_expert: bool,
) -> dict[str, bool | None]:
    if not isinstance(scene, Mapping):
        return {
            "scene_present": False,
            "setup_success": False,
            "render_success": False,
            "rule_check_passed": False,
            "expert_passed": False if require_expert else None,
        }
    rule = scene.get("rule_check")
    expert = scene.get("expert")
    batch = scene.get("expert_batch")
    expert_passed = None
    if require_expert:
        if isinstance(batch, Mapping):
            expert_passed = bool(batch.get("passed"))
        elif isinstance(expert, Mapping):
            expert_passed = bool(expert.get("passed"))
        else:
            expert_passed = False
    return {
        "scene_present": True,
        "setup_success": bool(scene.get("setup_success")),
        "render_success": bool(scene.get("render_success")),
        "rule_check_passed": bool(
            isinstance(rule, Mapping) and rule.get("passed")
        ),
        "expert_passed": expert_passed,
    }


def _require_current_scene_gates(
    scene: Mapping[str, Any] | None,
    position_samples: Mapping[str, Any] | None,
    *,
    require_expert: bool,
) -> dict[str, bool | None]:
    checks = _scene_checks(scene, require_expert=require_expert)
    if not (
        checks["scene_present"]
        and checks["setup_success"]
        and checks["render_success"]
        and checks["rule_check_passed"]
    ):
        raise ProductionTaskAcceptanceError(
            "current TaskGen setup/render/rule probe did not pass"
        )
    if require_expert and not checks["expert_passed"]:
        raise ProductionTaskAcceptanceError("current official expert gate did not pass")
    if isinstance(position_samples, Mapping) and not bool(
        position_samples.get("passed")
    ):
        raise ProductionTaskAcceptanceError(
            "current controlled-variation sample gate did not pass"
        )
    return checks


def _identity(
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    task_resolution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    spec = _read_json(run_dir / "variant_spec.json", label="VariantSpec")
    return {
        "schema_version": 1,
        "run_id": manifest.get("run_id"),
        "task_name": manifest.get("task_name"),
        "task_module": manifest.get("task_module"),
        "mode": manifest.get("mode"),
        "generation_kind": manifest.get("generation_kind"),
        "variant_spec_sha256": _canonical_sha256(spec),
        "task_resolution_sha256": (
            _canonical_sha256(task_resolution) if task_resolution is not None else None
        ),
    }


def _validate_runtime_variant_spec(
    value: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate official, generic v2, and legacy generated identities."""

    task_name = str(manifest.get("task_name") or "")
    if manifest.get("mode") == "official" or manifest.get(
        "generation_kind"
    ) == "official_passthrough":
        required = {
            "schema_version",
            "task_name",
            "intent",
            "generation_mode",
            "changes",
            "preserve",
        }
        expected = {
            "schema_version": 1,
            "task_name": task_name,
            "intent": "evaluate_official_task_unchanged",
            "generation_mode": "official",
            "changes": {},
            "preserve": ["official_task_source", "official_task_identity"],
        }
        if set(value) != required or dict(value) != expected:
            raise ProductionTaskAcceptanceError(
                "official VariantSpec no longer preserves the upstream task"
            )
        return expected
    if value.get("schema_version") == 2:
        try:
            normalized = validate_variant_spec_envelope(value)
        except CapabilityError as exc:
            raise ProductionTaskAcceptanceError(str(exc)) from exc
        if normalized["task_name"] != task_name:
            raise ProductionTaskAcceptanceError(
                "VariantSpec and manifest task identity differ"
            )
        return normalized
    return validate_variant_spec(dict(value), task_name)


def _verify_bound_artifacts(
    run_dir: Path,
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> None:
    """Recompute executable bindings and gates from the final files."""

    repo_root = run_dir.parents[2]
    proposal_path = run_dir / "generation/task_proposal.json"
    task_proposal = (
        _read_json(proposal_path, label="TaskProposal")
        if proposal_path.is_file()
        else None
    )
    try:
        expected_bundle = write_task_artifact_bundle(
            repo_root,
            run_dir,
            manifest,
            task_proposal=task_proposal,
            persist=False,
        )
    except (TaskArtifactBundleError, OSError, UnicodeError, ValueError) as exc:
        raise ProductionTaskAcceptanceError(
            f"cannot reconstruct TaskArtifactBundle: {exc}"
        ) from exc
    if dict(bundle) != expected_bundle:
        raise ProductionTaskAcceptanceError(
            "TaskArtifactBundle differs from the final executable bindings"
        )

    scene_check = bundle.get("scene_check_spec")
    if not isinstance(scene_check, Mapping):
        raise ProductionTaskAcceptanceError("SceneCheckSpec binding is missing")
    if scene_check.get("artifact") != "generation/scene_check_spec.json":
        raise ProductionTaskAcceptanceError("SceneCheckSpec path changed")
    scene_check_value = _read_json(
        run_dir / "generation/scene_check_spec.json", label="SceneCheckSpec"
    )
    try:
        validate_scene_check_spec(scene_check_value)
    except SceneCheckSpecError as exc:
        raise ProductionTaskAcceptanceError(str(exc)) from exc
    if scene_check.get("sha256") != _canonical_sha256(scene_check_value):
        raise ProductionTaskAcceptanceError("SceneCheckSpec hash changed")

    overlay_path = run_dir / "overlay.yml"
    try:
        overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ProductionTaskAcceptanceError(f"invalid overlay.yml: {exc}") from exc
    try:
        expected_overlay = (
            {}
            if manifest.get("mode") == "official"
            else compile_overlay(dict(spec))
            if spec.get("task_name") == "beat_block_hammer"
            else compile_click_bell_overlay(spec.get("changes"))
            if spec.get("task_name") == "click_bell"
            else None
        )
    except (TaskGenError, ClickBellTaskGenError, ValueError) as exc:
        raise ProductionTaskAcceptanceError(f"overlay contract is invalid: {exc}") from exc
    if expected_overlay is None or overlay != expected_overlay:
        raise ProductionTaskAcceptanceError("overlay.yml differs from VariantSpec")

    scene_binding = bundle.get("scene_method")
    if not isinstance(scene_binding, Mapping):
        raise ProductionTaskAcceptanceError("scene method binding is missing")
    if scene_binding.get("origin") == "generated_code":
        source = scene_binding.get("source")
        if not isinstance(source, str):
            raise ProductionTaskAcceptanceError("generated scene source is missing")
        source_path = (repo_root / source).resolve()
        if not source_path.is_relative_to(repo_root):
            raise ProductionTaskAcceptanceError("generated scene source escaped repo")
        method_source = _declared_method_source(
            source_path,
            class_name=str(manifest.get("task_name") or ""),
            method_name="load_actors",
        )
        try:
            validate_load_actors(method_source, dict(spec))
        except TaskGenError as exc:
            raise ProductionTaskAcceptanceError(
                f"generated load_actors is invalid: {exc}"
            ) from exc
        repair_source = run_dir / "generation/load_actors.py.txt"
        try:
            recorded_method = repair_source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ProductionTaskAcceptanceError(
                f"generated load_actors evidence is missing: {exc}"
            ) from exc
        if recorded_method.strip() != method_source.strip():
            raise ProductionTaskAcceptanceError(
                "generated load_actors evidence differs from task.py"
            )

    semantics = bundle.get("success_semantics")
    if not isinstance(semantics, Mapping):
        raise ProductionTaskAcceptanceError("success semantics binding is missing")
    success_binding = bundle.get("success_method")
    if not isinstance(success_binding, Mapping):
        raise ProductionTaskAcceptanceError("success method binding is missing")
    if success_binding.get("origin") != "compiled_success_spec":
        return
    success_spec = _read_json(
        run_dir / "generation/success_spec.json", label="SuccessSpec"
    )
    try:
        report = success_spec_validation_report(success_spec)
    except SuccessSpecError as exc:
        raise ProductionTaskAcceptanceError(f"SuccessSpec is invalid: {exc}") from exc
    if not report["act_eligible"]:
        raise ProductionTaskAcceptanceError("SuccessSpec is not ACT eligible")
    experimental = bool(report["experimental_bounded"])
    expected_authority = (
        "compiled_success_spec_experimental_bounded"
        if experimental
        else "compiled_success_spec_official_equivalent"
    )
    if (
        semantics.get("authority") != expected_authority
        or semantics.get("preserved") is not bool(report["official_equivalent"])
        or semantics.get("success_spec_sha256") != _canonical_sha256(success_spec)
    ):
        raise ProductionTaskAcceptanceError(
            "TaskArtifactBundle mislabels the final SuccessSpec authority"
        )


def _validate_current_candidate(
    run_dir: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    raw_spec = _read_json(run_dir / "variant_spec.json", label="VariantSpec")
    try:
        spec = _validate_runtime_variant_spec(raw_spec, manifest)
        bundle = validate_task_artifact_bundle(
            _read_json(
                run_dir / "generation/task_artifact_bundle.json",
                label="TaskArtifactBundle",
            )
        )
        _verify_bound_artifacts(run_dir, manifest, spec, bundle)
        reviewed_contract = _verify_reviewed_provenance(run_dir, manifest)
        contract = {
            "variant_spec_sha256": _canonical_sha256(spec),
            "task_artifact_bundle_sha256": _canonical_sha256(bundle),
            "overlay_sha256": _file_sha256(run_dir / "overlay.yml"),
            **reviewed_contract,
        }
    except (
        OSError,
        UnicodeError,
        TaskGenError,
        TaskArtifactBundleError,
        ProductionTaskAcceptanceError,
        ValueError,
    ) as exc:
        raise ProductionTaskAcceptanceError(str(exc)) from exc
    return spec, bundle, contract


def _require_current_artifact_contract(
    summary: Mapping[str, Any], contract: Mapping[str, str]
) -> None:
    accepted = summary.get("accepted_result")
    recorded = accepted.get("artifact_contract") if isinstance(accepted, Mapping) else None
    if not isinstance(recorded, Mapping) or dict(recorded) != dict(contract):
        raise ProductionTaskAcceptanceError(
            "accepted TaskGen artifact contract changed after validation"
        )


def _require_bundle_act_runtime_eligible(bundle: Mapping[str, Any]) -> None:
    """Fail closed when the final executable bundle explicitly forbids ACT.

    The TaskArtifactBundle is reconstructed from the current executable files
    before this check.  Manifest/TaskProposal copies are therefore descriptive
    metadata only and cannot make an experimental SuccessSpec ACT-eligible.
    """

    semantics = bundle.get("success_semantics")
    if not isinstance(semantics, Mapping):
        raise ProductionTaskAcceptanceError(
            "TaskArtifactBundle success semantics are missing"
        )
    if semantics.get("act_runtime_eligible") is False:
        blocker = semantics.get("runtime_blocker")
        detail = (
            str(blocker).strip()
            if isinstance(blocker, str) and blocker.strip()
            else "the final success semantics are probe-only"
        )
        raise ProductionTaskAcceptanceError(
            f"TaskArtifactBundle forbids ACT runtime execution: {detail}"
        )


def require_task_artifact_act_runtime_eligible(
    run_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the final TaskArtifactBundle and authorize ACT from it alone."""

    root = Path(run_dir).expanduser().resolve()
    _, bundle, _ = _validate_current_candidate(root, manifest)
    _require_bundle_act_runtime_eligible(bundle)
    return bundle


def validate_production_task_acceptance(
    summary: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an existing acceptance summary before ACT can start."""

    if not isinstance(summary, Mapping):
        raise ProductionTaskAcceptanceError("TaskGen acceptance summary is not an object")
    if summary.get("status") != "accepted":
        raise ProductionTaskAcceptanceError("TaskGen candidate was not accepted")
    if summary.get("recovery_scope") != "task_generation_before_policy":
        raise ProductionTaskAcceptanceError("TaskGen acceptance scope changed")
    runtime = summary.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("act_rollouts_started") != 0:
        raise ProductionTaskAcceptanceError(
            "TaskGen acceptance must precede every ACT rollout"
        )
    accepted = summary.get("accepted_result")
    if not isinstance(accepted, Mapping):
        raise ProductionTaskAcceptanceError("TaskGen accepted result is missing")
    identity = accepted.get("proposal_identity")
    if not isinstance(identity, Mapping):
        raise ProductionTaskAcceptanceError("TaskGen proposal identity is missing")
    if expected_identity is not None and dict(identity) != dict(expected_identity):
        raise ProductionTaskAcceptanceError("TaskGen proposal identity changed")
    return dict(summary)


def record_production_task_acceptance(
    run_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    scene: Mapping[str, Any] | None,
    position_samples: Mapping[str, Any] | None,
    task_resolution: Mapping[str, Any] | None = None,
    require_expert: bool,
) -> dict[str, Any]:
    """Verify final TaskGen artifacts and write one append-only acceptance trace.

    Stage-specific repair happens before this call.  The barrier records the
    resulting candidate and is deliberately non-regenerating: an exhausted
    repair stage must fail rather than silently create another policy sample.
    """

    root = Path(run_dir).expanduser().resolve()
    identity = _identity(root, manifest, task_resolution=task_resolution)
    attempt_root = root / "validation/task_generation_attempts"
    summary_path = attempt_root / "task_generation_attempt_summary.json"
    if summary_path.is_file():
        summary = validate_production_task_acceptance(
            _read_json(summary_path, label="TaskGen acceptance summary"),
            expected_identity=identity,
        )
        _, _, contract = _validate_current_candidate(root, manifest)
        _require_current_artifact_contract(summary, contract)
        _require_current_scene_gates(
            scene, position_samples, require_expert=require_expert
        )
        return summary
    if attempt_root.exists():
        raise ProductionTaskAcceptanceError(
            f"incomplete TaskGen acceptance trace already exists: {attempt_root}"
        )

    def execute_attempt(
        _attempt_dir: Path, _attempt_index: int, _requested_action: str | None
    ) -> dict[str, Any]:
        if _act_rollouts(manifest):
            raise TaskGenerationStageError(
                "policy_execution",
                "started_before_task_acceptance",
                "ACT evidence already exists before TaskGen acceptance",
                runtime={"act_rollouts_started": _act_rollouts(manifest)},
            )
        try:
            _spec, _bundle, artifact_contract = _validate_current_candidate(
                root, manifest
            )
        except ProductionTaskAcceptanceError as exc:
            raise TaskGenerationStageError(
                "static_validation",
                "failed",
                str(exc),
                diagnosis={"artifact": "TaskArtifactBundle/VariantSpec"},
            ) from exc

        checks = _scene_checks(scene, require_expert=require_expert)
        if not checks["scene_present"] or not checks["setup_success"]:
            raise TaskGenerationStageError(
                "render_probe", "failed", "TaskGen setup probe did not pass"
            )
        if not checks["render_success"] or not checks["rule_check_passed"]:
            raise TaskGenerationStageError(
                "render_probe", "failed", "TaskGen render/rule probe did not pass"
            )
        if require_expert and not checks["expert_passed"]:
            raise TaskGenerationStageError(
                "expert_gate", "unsolvable", "official expert gate did not pass"
            )
        if isinstance(position_samples, Mapping) and not bool(
            position_samples.get("passed")
        ):
            raise TaskGenerationStageError(
                "expert_gate",
                "unsolvable",
                "controlled-variation sample gate did not pass",
            )
        return {
            "status": "accepted",
            "proposal_identity": identity,
            "checks": {
                **checks,
                "variant_spec_valid": True,
                "task_artifact_bundle_valid": True,
                "position_samples_passed": (
                    bool(position_samples.get("passed"))
                    if isinstance(position_samples, Mapping)
                    else None
                ),
            },
            "artifact_contract": {
                **artifact_contract,
            },
            "runtime": {
                "provider_calls": _provider_calls(manifest, root),
                "simulator_probes": _simulator_probe_count(scene),
                "expert_probes": _expert_probe_count(scene),
                "act_rollouts_started": 0,
            },
        }

    try:
        summary = run_bounded_task_generation(
            attempt_root,
            proposal_identity=identity,
            execute_attempt=execute_attempt,
            max_regenerations=0,
        )
    except TaskGenerationRecoveryError as exc:
        raise ProductionTaskAcceptanceError(str(exc)) from exc
    return validate_production_task_acceptance(summary, expected_identity=identity)


def require_production_task_acceptance(
    run_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    task_resolution: Mapping[str, Any] | None = None,
    for_act: bool = False,
) -> dict[str, Any]:
    """Load and verify the acceptance barrier immediately before ACT launch."""

    root = Path(run_dir).expanduser().resolve()
    identity = _identity(root, manifest, task_resolution=task_resolution)
    summary = validate_production_task_acceptance(
        _read_json(
            root
            / "validation/task_generation_attempts/task_generation_attempt_summary.json",
            label="TaskGen acceptance summary",
        ),
        expected_identity=identity,
    )
    _, bundle, contract = _validate_current_candidate(root, manifest)
    _require_current_artifact_contract(summary, contract)
    if for_act:
        _require_bundle_act_runtime_eligible(bundle)
    return summary


__all__ = [
    "ProductionTaskAcceptanceError",
    "record_production_task_acceptance",
    "require_production_task_acceptance",
    "require_task_artifact_act_runtime_eligible",
    "validate_production_task_acceptance",
]
