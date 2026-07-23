"""Fail-closed registry for explicitly reviewed generated Task artifacts.

Generation never writes this registry.  A separate review must approve one
exact semantic key and the exact bytes of every executable/evidence artifact.
Persistent reuse revalidates the complete entry before returning a match or
copying files into a new run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from .artifacts import TaskArtifactBundleError, validate_task_artifact_bundle
from .capabilities import CapabilityError, validate_variant_spec_envelope
from .prototype import TaskGenError, compile_overlay, validate_load_actors
from .scene_checks import SceneCheckSpecError, validate_scene_check_spec
from .success_spec import (
    SuccessSpecError,
    success_spec_validation_report,
    validate_compiled_success_method,
    validate_success_spec,
)


REGISTRY_SCHEMA_VERSION = 1
REGISTRATION_SCHEMA_VERSION = 1
REVIEW_MANIFEST_SCHEMA_VERSION = 1
REGISTRY_SCOPE = "reviewed_generated_task_reuse"
REVIEW_SCOPE = "persistent_generated_task_reuse"
ADMISSION_POLICY = "explicit_approved_manifest_exact_artifact_hashes"
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
RUNTIME_DEPENDENCY_PATHS = (
    "envs/beat_block_hammer.py",
    "envs/_base_task.py",
    "envs/utils/__init__.py",
    "envs/utils/create_actor.py",
    "envs/utils/rand_create_actor.py",
)

BASE_ARTIFACTS = (
    "task.py",
    "variant_spec.json",
    "overlay.yml",
    "generation/load_actors.py.txt",
    "generation/task_artifact_bundle.json",
    "generation/scene_check_spec.json",
    "validation/static.json",
)
SUCCESS_SPEC_ARTIFACT = "generation/success_spec.json"
HASH_FIELDS = {
    "task.py": "task_sha256",
    "variant_spec.json": "variant_spec_sha256",
    "overlay.yml": "overlay_sha256",
    "generation/load_actors.py.txt": "load_actors_source_sha256",
    SUCCESS_SPEC_ARTIFACT: "success_spec_sha256",
    "generation/task_artifact_bundle.json": "task_artifact_bundle_sha256",
    "generation/scene_check_spec.json": "scene_check_spec_sha256",
    "validation/static.json": "static_validation_sha256",
}
SEMANTIC_KEY_FIELDS = {
    "schema_version",
    "task_name",
    "aspect_id",
    "capability_id",
    "changes",
    "preserve_success_semantics",
    "success_spec",
    "capability_contract_sha256",
}
REVIEW_CHECKS = {
    "task_source_reviewed",
    "variant_spec_matches_semantic_key",
    "overlay_matches_variant_spec",
    "repair_source_reviewed",
    "success_semantics_reviewed",
    "task_artifact_bundle_reviewed",
    "scene_check_reviewed",
    "static_validation_reviewed",
}
REVIEW_MANIFEST_FIELDS = {
    "schema_version",
    "decision",
    "review_scope",
    "reviewer",
    "reviewed_at",
    "source_run_id",
    "semantic_key_sha256",
    "runtime_dependency_hashes",
    *HASH_FIELDS.values(),
    "checks",
    "notes",
}
INDEX_FIELDS = {
    "registration_id",
    "artifact_id",
    "scope",
    "status",
    "task_name",
    "semantic_key_sha256",
    "artifact_hashes",
    "runtime_dependency_hashes",
    "registration_artifact",
    "registration_artifact_sha256",
    "review_manifest_artifact",
    "review_manifest_artifact_sha256",
    "artifacts",
}
REGISTRATION_FIELDS = {
    "schema_version",
    "registration_id",
    "artifact_id",
    "scope",
    "status",
    "source_run_id",
    "task_name",
    "semantic_key",
    "semantic_key_sha256",
    "artifact_hashes",
    "runtime_dependency_hashes",
    "review_manifest_sha256",
    "reviewer",
    "reviewed_at",
    "installed_at",
}


class ReviewedTaskRegistryError(RuntimeError):
    """Raised when admission, storage, or exact lookup is not trustworthy."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReviewedTaskRegistryError(
            f"value is not canonical JSON: {exc}"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReviewedTaskRegistryError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _runtime_dependency_hashes(repo_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in RUNTIME_DEPENDENCY_PATHS:
        path = (repo_root / relative).resolve()
        if not path.is_relative_to(repo_root) or not path.is_file():
            raise ReviewedTaskRegistryError(
                f"generated Task runtime dependency is missing: {relative}"
            )
        result[relative] = _file_sha256(path)
    return result


def _validate_runtime_dependency_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(
        RUNTIME_DEPENDENCY_PATHS
    ):
        raise ReviewedTaskRegistryError(
            "runtime_dependency_hashes must pin the generated BBH runtime ABI"
        )
    return {
        relative: _require_hash(value.get(relative), field=f"runtime dependency {relative}")
        for relative in RUNTIME_DEPENDENCY_PATHS
    }


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ReviewedTaskRegistryError(
            f"unfinished registry write already exists: {temporary}"
        )
    temporary.write_bytes(payload)
    temporary.replace(path)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewedTaskRegistryError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewedTaskRegistryError(f"{label} must be a JSON object: {path}")
    _canonical_bytes(value)
    return value


def _require_hash(value: Any, *, field: str, allow_none: bool = False) -> Any:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise ReviewedTaskRegistryError(f"{field} must be a lowercase SHA-256")
    return value


def _unresolved_root(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ReviewedTaskRegistryError(f"{label} must not be a symlink")
    return path.resolve()


def _safe_artifact(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ReviewedTaskRegistryError(f"{label} path must be a non-empty string")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ReviewedTaskRegistryError(f"{label} path escapes its root")
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ReviewedTaskRegistryError(f"{label} path must not contain symlinks")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ReviewedTaskRegistryError(f"{label} path escapes its root") from exc
    return candidate


def _validate_semantic_key(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SEMANTIC_KEY_FIELDS:
        raise ReviewedTaskRegistryError(
            "semantic key must contain exactly the TaskGen resolver fields"
        )
    result = deepcopy(dict(value))
    if result.get("schema_version") != 1:
        raise ReviewedTaskRegistryError("semantic key schema_version must be 1")
    for field in ("task_name", "aspect_id", "capability_id"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ReviewedTaskRegistryError(f"semantic key {field} must be non-empty")
    if not isinstance(result.get("changes"), Mapping):
        raise ReviewedTaskRegistryError("semantic key changes must be an object")
    preserve_success = result.get("preserve_success_semantics")
    if preserve_success is not True and preserve_success is not False:
        raise ReviewedTaskRegistryError(
            "semantic key preserve_success_semantics must be boolean"
        )
    if result.get("success_spec") is not None and not isinstance(
        result["success_spec"], Mapping
    ):
        raise ReviewedTaskRegistryError("semantic key success_spec must be object or null")
    if preserve_success and result.get("success_spec") is not None:
        raise ReviewedTaskRegistryError(
            "preserved success semantics must not carry a replacement SuccessSpec"
        )
    if not preserve_success:
        if not isinstance(result.get("success_spec"), Mapping):
            raise ReviewedTaskRegistryError(
                "non-preserved success semantics require a bounded SuccessSpec"
            )
        try:
            report = success_spec_validation_report(result["success_spec"])
        except SuccessSpecError as exc:
            raise ReviewedTaskRegistryError(
                f"semantic key SuccessSpec is invalid: {exc}"
            ) from exc
        if not report["act_eligible"] or not report["experimental_bounded"]:
            raise ReviewedTaskRegistryError(
                "semantic key replacement SuccessSpec must be experimental bounded ACT"
            )
    _require_hash(
        result.get("capability_contract_sha256"),
        field="semantic key capability_contract_sha256",
    )
    _canonical_bytes(result)
    return result


def _method_source(
    source: str,
    class_name: str,
    method_name: str,
    *,
    allowed_methods: set[str],
) -> tuple[str, str]:
    try:
        tree = ast.parse(source)
        compile(source, "<reviewed generated task>", "exec")
    except SyntaxError as exc:
        raise ReviewedTaskRegistryError(f"generated task.py is invalid: {exc}") from exc
    for node in tree.body:
        if isinstance(node, ast.Expr):
            if not (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                raise ReviewedTaskRegistryError(
                    "task.py contains executable top-level expressions"
                )
        elif isinstance(node, ast.Import):
            imported = {(item.name, item.asname) for item in node.names}
            if not imported <= {("numpy", "np"), ("sapien", None)}:
                raise ReviewedTaskRegistryError("task.py imports an unapproved module")
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module not in {
                f"envs.{class_name}",
                "envs.utils",
            }:
                raise ReviewedTaskRegistryError("task.py imports an unapproved module")
            imported = {(item.name, item.asname) for item in node.names}
            if node.module == f"envs.{class_name}" and imported != {
                (class_name, "OfficialBeatBlockHammer")
            }:
                raise ReviewedTaskRegistryError(
                    "task.py official base import is not exact"
                )
            if node.module == "envs.utils" and imported != {
                ("create_actor", None),
                ("create_box", None),
                ("rand_pose", None),
            }:
                raise ReviewedTaskRegistryError("task.py utility imports are not exact")
        elif not isinstance(node, ast.ClassDef):
            raise ReviewedTaskRegistryError(
                "task.py contains executable top-level statements"
            )
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1 or len([n for n in tree.body if isinstance(n, ast.ClassDef)]) != 1:
        raise ReviewedTaskRegistryError("task.py must declare exactly the generated task class")
    task_class = classes[0]
    if task_class.decorator_list:
        raise ReviewedTaskRegistryError("generated task class must not use decorators")
    if (
        len(task_class.bases) != 1
        or not isinstance(task_class.bases[0], ast.Name)
        or task_class.bases[0].id != "OfficialBeatBlockHammer"
        or task_class.keywords
    ):
        raise ReviewedTaskRegistryError("generated task class base is not trusted")
    class_members = [
        node
        for node in task_class.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    if any(not isinstance(node, ast.FunctionDef) for node in class_members):
        raise ReviewedTaskRegistryError("generated task class contains unsupported members")
    if {node.name for node in class_members} != allowed_methods:
        raise ReviewedTaskRegistryError(
            "generated task class declares an unexpected method set"
        )
    methods = [
        node
        for node in class_members
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if len(methods) != 1:
        raise ReviewedTaskRegistryError(
            f"task.py must declare exactly one {method_name} method"
        )
    method = methods[0]
    if method.decorator_list:
        raise ReviewedTaskRegistryError(f"{method_name} must not use decorators")
    arguments = method.args
    if (
        [argument.arg for argument in arguments.posonlyargs + arguments.args]
        != ["self"]
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.kwonlyargs
        or arguments.defaults
        or arguments.kw_defaults
        or any(
            argument.annotation is not None
            for argument in arguments.posonlyargs + arguments.args
        )
        or method.returns is not None
    ):
        raise ReviewedTaskRegistryError(f"{method_name} must be a plain self method")
    lines = source.splitlines()
    raw = "\n".join(lines[method.lineno - 1 : method.end_lineno]) + "\n"
    return textwrap.dedent(raw), hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_static_validation(
    value: Mapping[str, Any],
    *,
    compiled_success: bool,
    official_equivalent_success: bool = False,
) -> None:
    required_true = (
        ("variant_spec", "valid"),
        ("load_actors_ast", "valid"),
        ("load_actors_ast", "complete_method_generated"),
        ("protected_diff", "valid"),
    )
    for section, field in required_true:
        item = value.get(section)
        if not isinstance(item, Mapping) or item.get(field) is not True:
            raise ReviewedTaskRegistryError(
                f"static validation did not pass {section}.{field}"
            )
    load_validation = value["load_actors_ast"]
    if load_validation.get("calls_super") is not False:
        raise ReviewedTaskRegistryError("generated load_actors must not call super")
    if compiled_success:
        success = value.get("success_spec")
        for field in ("valid", "act_eligible", "complete_method_generated"):
            legacy_official_act = (
                field == "act_eligible"
                and isinstance(success, Mapping)
                and field not in success
                and official_equivalent_success
            )
            if (
                not legacy_official_act
                and (not isinstance(success, Mapping) or success.get(field) is not True)
            ):
                raise ReviewedTaskRegistryError(
                    f"static validation did not pass success_spec.{field}"
                )
        if success.get("arbitrary_code_accepted") is not False:
            raise ReviewedTaskRegistryError(
                "static validation accepted arbitrary success code"
            )


def _source_artifacts(
    source_run_dir: str | Path,
    semantic_key: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    expected_runtime_dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _unresolved_root(source_run_dir, label="source run directory")
    if not root.is_dir():
        raise ReviewedTaskRegistryError(f"source run directory does not exist: {root}")
    key = _validate_semantic_key(semantic_key)
    paths = {
        relative: _safe_artifact(root, relative, label=f"source {relative}")
        for relative in BASE_ARTIFACTS
    }
    missing = sorted(relative for relative, path in paths.items() if not path.is_file())
    if missing:
        raise ReviewedTaskRegistryError(f"source task artifacts are missing: {missing}")

    variant = _read_json(paths["variant_spec.json"], label="VariantSpec")
    bundle = _read_json(
        paths["generation/task_artifact_bundle.json"],
        label="TaskArtifactBundle",
    )
    scene_check = _read_json(
        paths["generation/scene_check_spec.json"], label="SceneCheckSpec"
    )
    static = _read_json(paths["validation/static.json"], label="static validation")
    try:
        normalized_variant = validate_variant_spec_envelope(variant)
        normalized_bundle = validate_task_artifact_bundle(bundle)
        normalized_scene_check = validate_scene_check_spec(scene_check)
    except (CapabilityError, TaskArtifactBundleError, SceneCheckSpecError) as exc:
        raise ReviewedTaskRegistryError(f"invalid generated task evidence: {exc}") from exc
    if normalized_variant != variant or normalized_bundle != bundle or normalized_scene_check != scene_check:
        raise ReviewedTaskRegistryError("generated task evidence is not canonical")
    try:
        overlay = yaml.safe_load(paths["overlay.yml"].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ReviewedTaskRegistryError(f"invalid generated overlay.yml: {exc}") from exc
    if overlay != compile_overlay(variant):
        raise ReviewedTaskRegistryError("overlay.yml does not match VariantSpec")
    controlled_axis = variant["controlled_axis"]
    aspect_id = key["aspect_id"]
    if (
        variant["task_name"] != key["task_name"]
        or variant["capability_id"] != key["capability_id"]
        or not (
            aspect_id == controlled_axis
            or aspect_id.startswith(controlled_axis + ".")
            or key["capability_id"].startswith(controlled_axis + ".")
        )
        or variant["changes"] != key["changes"]
        or variant["generation_mode"] != "force_codegen"
    ):
        raise ReviewedTaskRegistryError(
            "VariantSpec does not match the exact semantic key"
        )
    if bundle.get("task_name") != key["task_name"]:
        raise ReviewedTaskRegistryError("TaskArtifactBundle task identity differs")
    if bundle.get("scene_method", {}).get("origin") != "generated_code":
        raise ReviewedTaskRegistryError("reviewed generated task must bind generated scene code")
    if bundle.get("scene_check_spec", {}).get("artifact") != (
        "generation/scene_check_spec.json"
    ):
        raise ReviewedTaskRegistryError("TaskArtifactBundle scene check path is not fixed")
    if bundle.get("variant_spec_sha256") != _canonical_sha256(variant):
        raise ReviewedTaskRegistryError("TaskArtifactBundle VariantSpec hash differs")
    if bundle.get("scene_check_spec", {}).get("sha256") != _canonical_sha256(scene_check):
        raise ReviewedTaskRegistryError("TaskArtifactBundle SceneCheckSpec hash differs")
    if scene_check.get("task_name") != key["task_name"]:
        raise ReviewedTaskRegistryError("SceneCheckSpec task identity differs")

    task_bytes = paths["task.py"].read_bytes()
    try:
        task_source = task_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise ReviewedTaskRegistryError("task.py must be UTF-8") from exc
    task_hash = hashlib.sha256(task_bytes).hexdigest()
    compiled_success = bundle.get("success_method", {}).get("origin") == (
        "compiled_success_spec"
    )
    allowed_methods = (
        {"load_actors", "check_success"}
        if compiled_success
        else {"load_actors"}
    )
    load_source, load_symbol_hash = _method_source(
        task_source,
        key["task_name"],
        "load_actors",
        allowed_methods=allowed_methods,
    )
    try:
        repair_source = paths["generation/load_actors.py.txt"].read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError) as exc:
        raise ReviewedTaskRegistryError(
            f"invalid generated load_actors repair source: {exc}"
        ) from exc
    normalized_load = textwrap.dedent(load_source).strip() + "\n"
    normalized_repair = textwrap.dedent(repair_source).strip() + "\n"
    if normalized_repair != normalized_load:
        raise ReviewedTaskRegistryError(
            "generation/load_actors.py.txt differs from task.py load_actors"
        )
    try:
        recomputed_load = validate_load_actors(load_source, variant)
    except TaskGenError as exc:
        raise ReviewedTaskRegistryError(f"generated load_actors is invalid: {exc}") from exc
    for field in ("valid", "complete_method_generated", "calls_super"):
        if static.get("load_actors_ast", {}).get(field) != recomputed_load.get(field):
            raise ReviewedTaskRegistryError(
                f"recorded load_actors validation differs for {field}"
            )
    scene_binding = bundle["scene_method"]
    if (
        scene_binding.get("source_sha256") != task_hash
        or scene_binding.get("symbol_sha256") != load_symbol_hash
        or scene_binding.get("symbol_declared") is not True
    ):
        raise ReviewedTaskRegistryError("task.py scene binding hash differs")

    success_spec = None
    success_report: dict[str, Any] | None = None
    if compiled_success:
        success_path = _safe_artifact(
            root, SUCCESS_SPEC_ARTIFACT, label="source SuccessSpec"
        )
        if not success_path.is_file():
            raise ReviewedTaskRegistryError("compiled task SuccessSpec is missing")
        paths[SUCCESS_SPEC_ARTIFACT] = success_path
        success_spec = _read_json(success_path, label="SuccessSpec")
        try:
            normalized_success = validate_success_spec(success_spec)
            success_report = success_spec_validation_report(success_spec)
            success_source, success_symbol_hash = _method_source(
                task_source,
                key["task_name"],
                "check_success",
                allowed_methods=allowed_methods,
            )
            recomputed_success = validate_compiled_success_method(
                success_source, success_spec
            )
        except SuccessSpecError as exc:
            raise ReviewedTaskRegistryError(
                f"compiled SuccessSpec binding is invalid: {exc}"
            ) from exc
        if normalized_success != success_spec:
            raise ReviewedTaskRegistryError("SuccessSpec is not canonical")
        success_hash = _canonical_sha256(success_spec)
        semantics = bundle.get("success_semantics", {})
        success_binding = bundle["success_method"]
        if (
            semantics.get("success_spec") != SUCCESS_SPEC_ARTIFACT
            or semantics.get("success_spec_sha256") != success_hash
            or success_binding.get("source_sha256") != task_hash
            or success_binding.get("symbol_sha256") != success_symbol_hash
            or success_binding.get("symbol_declared") is not True
        ):
            raise ReviewedTaskRegistryError("task.py SuccessSpec binding hash differs")
        expected_authority = (
            "compiled_success_spec_experimental_bounded"
            if success_report["experimental_bounded"]
            else "compiled_success_spec_official_equivalent"
        )
        if (
            not success_report["act_eligible"]
            or semantics.get("authority") != expected_authority
            or semantics.get("preserved")
            is not bool(success_report["official_equivalent"])
        ):
            raise ReviewedTaskRegistryError(
                "TaskArtifactBundle mislabels SuccessSpec execution authority"
            )
        if key["preserve_success_semantics"] is not bool(
            success_report["official_equivalent"]
        ):
            raise ReviewedTaskRegistryError(
                "semantic key preservation claim differs from compiled SuccessSpec"
            )
        for field in (
            "valid",
            "act_eligible",
            "complete_method_generated",
            "arbitrary_code_accepted",
        ):
            recorded = static.get("success_spec", {}).get(field)
            legacy_official_act = (
                field == "act_eligible"
                and recorded is None
                and success_report["official_equivalent"]
            )
            if not legacy_official_act and recorded != recomputed_success.get(field):
                raise ReviewedTaskRegistryError(
                    f"recorded SuccessSpec validation differs for {field}"
                )
        if key["success_spec"] is not None and key["success_spec"] != success_spec:
            raise ReviewedTaskRegistryError(
                "generated SuccessSpec differs from semantic key"
            )
    elif key["success_spec"] is not None:
        raise ReviewedTaskRegistryError(
            "semantic key requests SuccessSpec but bundle reuses official success"
        )
    _validate_static_validation(
        static,
        compiled_success=compiled_success,
        official_equivalent_success=bool(
            success_report and success_report["official_equivalent"]
        ),
    )

    artifacts = {relative: path.read_bytes() for relative, path in paths.items()}
    artifact_hashes = {
        relative: hashlib.sha256(payload).hexdigest()
        for relative, payload in artifacts.items()
    }
    runtime_dependency_hashes = (
        _validate_runtime_dependency_hashes(expected_runtime_dependencies)
        if expected_runtime_dependencies is not None
        else _runtime_dependency_hashes(
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else root.parents[2]
        )
    )
    return {
        "root": root,
        "source_run_id": root.name,
        "semantic_key": key,
        "semantic_key_sha256": _canonical_sha256(key),
        "artifacts": artifacts,
        "artifact_hashes": artifact_hashes,
        "runtime_dependency_hashes": runtime_dependency_hashes,
        "variant_spec": variant,
        "success_spec": success_spec,
        "bundle": bundle,
    }


def build_task_review_manifest_template(
    source_run_dir: str | Path,
    semantic_key: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a pending template; this function never synthesizes approval."""

    source = _source_artifacts(
        source_run_dir, semantic_key, repo_root=repo_root
    )
    result = {
        "schema_version": REVIEW_MANIFEST_SCHEMA_VERSION,
        "decision": "pending",
        "review_scope": REVIEW_SCOPE,
        "reviewer": {"id": "", "kind": "development_agent"},
        "reviewed_at": None,
        "source_run_id": source["source_run_id"],
        "semantic_key_sha256": source["semantic_key_sha256"],
        "runtime_dependency_hashes": source["runtime_dependency_hashes"],
        "checks": {key: False for key in sorted(REVIEW_CHECKS)},
        "notes": "",
    }
    for artifact, field in HASH_FIELDS.items():
        result[field] = source["artifact_hashes"].get(artifact)
    return result


def validate_task_review_manifest(value: Any) -> dict[str, Any]:
    """Require explicit approval pinned to every reusable artifact byte."""

    if not isinstance(value, Mapping) or set(value) != REVIEW_MANIFEST_FIELDS:
        raise ReviewedTaskRegistryError(
            "review manifest fields do not match the strict schema"
        )
    result = deepcopy(dict(value))
    if result.get("schema_version") != REVIEW_MANIFEST_SCHEMA_VERSION:
        raise ReviewedTaskRegistryError("review manifest schema_version must be 1")
    if result.get("decision") != "approved":
        raise ReviewedTaskRegistryError("review manifest decision must be approved")
    if result.get("review_scope") != REVIEW_SCOPE:
        raise ReviewedTaskRegistryError(
            f"review manifest review_scope must be {REVIEW_SCOPE}"
        )
    reviewer = result.get("reviewer")
    if not isinstance(reviewer, Mapping) or set(reviewer) != {"id", "kind"}:
        raise ReviewedTaskRegistryError("reviewer must contain exactly id and kind")
    if (
        not isinstance(reviewer.get("id"), str)
        or not reviewer["id"].strip()
        or len(reviewer["id"]) > 120
    ):
        raise ReviewedTaskRegistryError("reviewer.id must be a non-empty identifier")
    if reviewer.get("kind") not in {"human", "development_agent"}:
        raise ReviewedTaskRegistryError(
            "reviewer.kind must be human or development_agent"
        )
    reviewed_at = result.get("reviewed_at")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise ReviewedTaskRegistryError("reviewed_at must be an ISO-8601 timestamp")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewedTaskRegistryError("reviewed_at is not valid ISO-8601") from exc
    if timestamp.tzinfo is None:
        raise ReviewedTaskRegistryError("reviewed_at must include a timezone")
    if not isinstance(result.get("source_run_id"), str) or IDENTIFIER_PATTERN.fullmatch(
        result["source_run_id"]
    ) is None:
        raise ReviewedTaskRegistryError("source_run_id must be a safe identifier")
    _require_hash(result.get("semantic_key_sha256"), field="semantic_key_sha256")
    result["runtime_dependency_hashes"] = _validate_runtime_dependency_hashes(
        result.get("runtime_dependency_hashes")
    )
    for artifact, field in HASH_FIELDS.items():
        _require_hash(
            result.get(field),
            field=field,
            allow_none=artifact == SUCCESS_SPEC_ARTIFACT,
        )
    checks = result.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != REVIEW_CHECKS:
        raise ReviewedTaskRegistryError(
            "review checks must contain exactly the required checks"
        )
    failed = sorted(key for key in REVIEW_CHECKS if checks.get(key) is not True)
    if failed:
        raise ReviewedTaskRegistryError(f"review checks were not approved: {failed}")
    if not isinstance(result.get("notes"), str):
        raise ReviewedTaskRegistryError("review notes must be a string")
    _canonical_bytes(result)
    return result


def _empty_index() -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "scope": REGISTRY_SCOPE,
        "admission_policy": ADMISSION_POLICY,
        "entries": [],
    }


def _read_index(root: Path) -> dict[str, Any]:
    index_path = _safe_artifact(root, "index.json", label="registry index")
    if not index_path.exists():
        if root.exists() and any(root.iterdir()):
            raise ReviewedTaskRegistryError(
                "non-empty reviewed registry has no index.json"
            )
        return _empty_index()
    if not index_path.is_file():
        raise ReviewedTaskRegistryError("reviewed registry index must be a file")
    index = _read_json(index_path, label="reviewed task registry index")
    if set(index) != {"schema_version", "scope", "admission_policy", "entries"}:
        raise ReviewedTaskRegistryError("reviewed task registry index fields are invalid")
    if (
        index.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or index.get("scope") != REGISTRY_SCOPE
        or index.get("admission_policy") != ADMISSION_POLICY
        or not isinstance(index.get("entries"), list)
    ):
        raise ReviewedTaskRegistryError("unsupported reviewed task registry index")
    registration_ids: set[str] = set()
    semantic_hashes: set[str] = set()
    for entry in index["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != INDEX_FIELDS:
            raise ReviewedTaskRegistryError("reviewed task index entry fields are invalid")
        registration_id = entry.get("registration_id")
        artifact_id = entry.get("artifact_id")
        if not isinstance(registration_id, str) or re.fullmatch(
            r"reviewed_task_[0-9a-f]{20}", registration_id
        ) is None:
            raise ReviewedTaskRegistryError("invalid reviewed task registration_id")
        if not isinstance(artifact_id, str) or re.fullmatch(
            r"task_artifact_[0-9a-f]{20}", artifact_id
        ) is None:
            raise ReviewedTaskRegistryError("invalid reviewed task artifact_id")
        semantic_hash = _require_hash(
            entry.get("semantic_key_sha256"), field="entry semantic_key_sha256"
        )
        _validate_runtime_dependency_hashes(entry.get("runtime_dependency_hashes"))
        if registration_id in registration_ids or semantic_hash in semantic_hashes:
            raise ReviewedTaskRegistryError(
                "reviewed task registry contains ambiguous duplicate entries"
            )
        registration_ids.add(registration_id)
        semantic_hashes.add(semantic_hash)
    return index


def _expected_entry_paths(registration_id: str, artifact_names: set[str]) -> set[str]:
    files = {"registration.json", "review_manifest.json"}
    files.update(f"artifacts/{name}" for name in artifact_names)
    nodes = set(files)
    for file_name in files:
        parent = Path(file_name).parent
        while str(parent) not in {"", "."}:
            nodes.add(parent.as_posix())
            parent = parent.parent
    return nodes


def _validate_entry_layout(entry_dir: Path, artifact_names: set[str]) -> None:
    if entry_dir.is_symlink() or not entry_dir.is_dir():
        raise ReviewedTaskRegistryError("reviewed task entry must be a real directory")
    actual: set[str] = set()
    for path in entry_dir.rglob("*"):
        relative = path.relative_to(entry_dir).as_posix()
        if path.is_symlink():
            raise ReviewedTaskRegistryError(
                f"reviewed task entry must not contain symlinks: {relative}"
            )
        actual.add(relative)
    expected = _expected_entry_paths(entry_dir.name, artifact_names)
    if actual != expected:
        raise ReviewedTaskRegistryError(
            "reviewed task entry contains missing or unapproved files"
        )


def _artifact_names_from_hashes(hashes: Any) -> set[str]:
    if not isinstance(hashes, Mapping):
        raise ReviewedTaskRegistryError("artifact_hashes must be an object")
    expected = set(BASE_ARTIFACTS)
    if hashes.get(SUCCESS_SPEC_ARTIFACT) is not None:
        expected.add(SUCCESS_SPEC_ARTIFACT)
    if set(hashes) != expected:
        raise ReviewedTaskRegistryError("artifact_hashes contains an invalid artifact set")
    for relative, value in hashes.items():
        _require_hash(value, field=f"artifact hash {relative}")
    return expected


def _load_entry(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    if entry.get("scope") != REGISTRY_SCOPE or entry.get("status") != "approved":
        raise ReviewedTaskRegistryError("reviewed task entry is not approved")
    registration_id = entry["registration_id"]
    artifact_names = _artifact_names_from_hashes(entry.get("artifact_hashes"))
    entry_dir = _safe_artifact(
        root, f"entries/{registration_id}", label="reviewed task entry"
    )
    _validate_entry_layout(entry_dir, artifact_names)
    expected_registration = f"entries/{registration_id}/registration.json"
    expected_review = f"entries/{registration_id}/review_manifest.json"
    if (
        entry.get("registration_artifact") != expected_registration
        or entry.get("review_manifest_artifact") != expected_review
    ):
        raise ReviewedTaskRegistryError("reviewed task metadata path is not fixed")
    registration_path = _safe_artifact(
        root, expected_registration, label="reviewed task registration"
    )
    review_path = _safe_artifact(root, expected_review, label="task review manifest")
    if _file_sha256(registration_path) != entry.get("registration_artifact_sha256"):
        raise ReviewedTaskRegistryError("reviewed task registration was tampered")
    if _file_sha256(review_path) != entry.get("review_manifest_artifact_sha256"):
        raise ReviewedTaskRegistryError("reviewed task review manifest was tampered")
    registration = _read_json(registration_path, label="reviewed task registration")
    if set(registration) != REGISTRATION_FIELDS:
        raise ReviewedTaskRegistryError("reviewed task registration fields are invalid")
    review = validate_task_review_manifest(
        _read_json(review_path, label="task review manifest")
    )
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != artifact_names:
        raise ReviewedTaskRegistryError("reviewed task artifact map is invalid")
    verified: dict[str, dict[str, str]] = {}
    for relative in sorted(artifact_names):
        descriptor = artifacts[relative]
        expected_path = f"entries/{registration_id}/artifacts/{relative}"
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "sha256"}
            or descriptor.get("path") != expected_path
            or descriptor.get("sha256") != entry["artifact_hashes"].get(relative)
        ):
            raise ReviewedTaskRegistryError(
                f"reviewed task artifact descriptor is invalid: {relative}"
            )
        path = _safe_artifact(root, expected_path, label=f"reviewed {relative}")
        if not path.is_file() or _file_sha256(path) != descriptor["sha256"]:
            raise ReviewedTaskRegistryError(
                f"reviewed task artifact was tampered: {relative}"
            )
        verified[relative] = {"path": str(path), "sha256": descriptor["sha256"]}

    review_hash = _canonical_sha256(review)
    expected_artifact_id = "task_artifact_" + _canonical_sha256(
        {
            "semantic_key_sha256": entry.get("semantic_key_sha256"),
            "artifact_hashes": entry.get("artifact_hashes"),
            "runtime_dependency_hashes": entry.get("runtime_dependency_hashes"),
        }
    )[:20]
    expected_registration_id = "reviewed_task_" + _canonical_sha256(
        {
            "artifact_id": expected_artifact_id,
            "review_manifest_sha256": review_hash,
        }
    )[:20]
    checks = {
        "registration_schema": registration.get("schema_version")
        == REGISTRATION_SCHEMA_VERSION,
        "registration_id": registration.get("registration_id") == registration_id,
        "artifact_id": registration.get("artifact_id") == entry.get("artifact_id"),
        "derived_artifact_id": entry.get("artifact_id") == expected_artifact_id,
        "derived_registration_id": registration_id == expected_registration_id,
        "scope": registration.get("scope") == REGISTRY_SCOPE,
        "status": registration.get("status") == "approved",
        "task_name": registration.get("task_name") == entry.get("task_name"),
        "semantic_hash": registration.get("semantic_key_sha256")
        == entry.get("semantic_key_sha256"),
        "semantic_key": _canonical_sha256(registration.get("semantic_key"))
        == entry.get("semantic_key_sha256"),
        "artifact_hashes": registration.get("artifact_hashes")
        == entry.get("artifact_hashes"),
        "runtime_dependency_hashes": registration.get(
            "runtime_dependency_hashes"
        )
        == entry.get("runtime_dependency_hashes"),
        "review_runtime_dependencies": review.get("runtime_dependency_hashes")
        == entry.get("runtime_dependency_hashes"),
        "review_hash": registration.get("review_manifest_sha256") == review_hash,
        "review_semantic": review.get("semantic_key_sha256")
        == entry.get("semantic_key_sha256"),
        "review_source_run": review.get("source_run_id")
        == registration.get("source_run_id"),
        "reviewer": review.get("reviewer") == registration.get("reviewer"),
        "reviewed_at": review.get("reviewed_at") == registration.get("reviewed_at"),
    }
    for artifact, field in HASH_FIELDS.items():
        checks[f"review_{field}"] = review.get(field) == entry["artifact_hashes"].get(
            artifact
        )
    failed = sorted(key for key, passed in checks.items() if passed is not True)
    if failed:
        raise ReviewedTaskRegistryError(
            f"reviewed task registration hashes are inconsistent: {failed}"
        )
    semantic_key = _validate_semantic_key(registration["semantic_key"])
    if registration["task_name"] != semantic_key["task_name"]:
        raise ReviewedTaskRegistryError(
            "reviewed task registration task differs from semantic key"
        )
    revalidated = _source_artifacts(
        entry_dir / "artifacts",
        semantic_key,
        expected_runtime_dependencies=entry["runtime_dependency_hashes"],
    )
    if revalidated["artifact_hashes"] != entry["artifact_hashes"]:
        raise ReviewedTaskRegistryError(
            "reviewed task artifact set differs after semantic revalidation"
        )
    return {
        "registration": registration,
        "review_manifest": review,
        "registry_dir": root,
        "entry_dir": entry_dir,
        "verified_artifacts": verified,
    }


def _audit_registry_layout(root: Path, index: Mapping[str, Any]) -> None:
    if not root.exists():
        return
    allowed_root = {"index.json"}
    entries_path = root / "entries"
    if entries_path.exists() or entries_path.is_symlink():
        allowed_root.add("entries")
    if {path.name for path in root.iterdir()} != allowed_root:
        raise ReviewedTaskRegistryError("reviewed task registry has unindexed files")
    expected = {entry["registration_id"] for entry in index["entries"]}
    if entries_path.is_symlink():
        raise ReviewedTaskRegistryError("reviewed task entries directory must not be a symlink")
    if expected and not entries_path.is_dir():
        raise ReviewedTaskRegistryError("reviewed task entries directory is missing")
    if entries_path.exists():
        actual = {path.name for path in entries_path.iterdir()}
        if actual != expected:
            raise ReviewedTaskRegistryError(
                "reviewed task entries directory has unindexed content"
            )


def load_reviewed_task_registry(registry_dir: str | Path) -> dict[str, Any]:
    """Load and fully verify every persistent generated-task entry."""

    root = _unresolved_root(registry_dir, label="reviewed task registry")
    if root.exists() and not root.is_dir():
        raise ReviewedTaskRegistryError(
            "reviewed task registry must be a directory"
        )
    index = _read_index(root)
    _audit_registry_layout(root, index)
    for entry in index["entries"]:
        _load_entry(root, entry)
    return deepcopy(index)


def _read_manifest_input(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ReviewedTaskRegistryError("review manifest path must not be a symlink")
    return _read_json(path.resolve(), label="task review manifest")


def install_reviewed_task(
    source_run_dir: str | Path,
    semantic_key: Mapping[str, Any],
    review_manifest: Mapping[str, Any] | str | Path,
    registry_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Install one separately approved immutable generated-task artifact set."""

    source = _source_artifacts(
        source_run_dir, semantic_key, repo_root=repo_root
    )
    review = validate_task_review_manifest(_read_manifest_input(review_manifest))
    expected = {
        "source_run_id": source["source_run_id"],
        "semantic_key_sha256": source["semantic_key_sha256"],
        "runtime_dependency_hashes": source["runtime_dependency_hashes"],
    }
    for artifact, field in HASH_FIELDS.items():
        expected[field] = source["artifact_hashes"].get(artifact)
    mismatched = sorted(key for key, value in expected.items() if review.get(key) != value)
    if mismatched:
        raise ReviewedTaskRegistryError(
            f"review manifest does not match source artifacts: {mismatched}"
        )

    artifact_id = "task_artifact_" + _canonical_sha256(
        {
            "semantic_key_sha256": source["semantic_key_sha256"],
            "artifact_hashes": source["artifact_hashes"],
            "runtime_dependency_hashes": source["runtime_dependency_hashes"],
        }
    )[:20]
    review_hash = _canonical_sha256(review)
    registration_id = "reviewed_task_" + _canonical_sha256(
        {"artifact_id": artifact_id, "review_manifest_sha256": review_hash}
    )[:20]
    root = _unresolved_root(registry_dir, label="reviewed task registry")
    index = load_reviewed_task_registry(root)
    for entry in index["entries"]:
        if entry["semantic_key_sha256"] == source["semantic_key_sha256"]:
            loaded = _load_entry(root, entry)
            if entry["artifact_id"] == artifact_id:
                if (
                    loaded["registration"].get("review_manifest_sha256")
                    == review_hash
                ):
                    return _match_from_loaded(loaded)
                raise ReviewedTaskRegistryError(
                    "task artifact already has a different review attestation; "
                    "review upgrades require an explicit multi-attestation workflow"
                )
            raise ReviewedTaskRegistryError(
                "semantic key is already approved for a different task artifact"
            )

    registration = {
        "schema_version": REGISTRATION_SCHEMA_VERSION,
        "registration_id": registration_id,
        "artifact_id": artifact_id,
        "scope": REGISTRY_SCOPE,
        "status": "approved",
        "source_run_id": source["source_run_id"],
        "task_name": source["semantic_key"]["task_name"],
        "semantic_key": source["semantic_key"],
        "semantic_key_sha256": source["semantic_key_sha256"],
        "artifact_hashes": source["artifact_hashes"],
        "runtime_dependency_hashes": source["runtime_dependency_hashes"],
        "review_manifest_sha256": review_hash,
        "reviewer": review["reviewer"],
        "reviewed_at": review["reviewed_at"],
        "installed_at": datetime.now().astimezone().isoformat(),
    }
    entries_root = root / "entries"
    entry_dir = entries_root / registration_id
    temporary_dir = entries_root / (registration_id + ".tmp")
    if entries_root.is_symlink():
        raise ReviewedTaskRegistryError(
            "reviewed task entries directory must not be a symlink"
        )
    if entries_root.exists() and not entries_root.is_dir():
        raise ReviewedTaskRegistryError("reviewed task entries path must be a directory")
    if any(path.exists() or path.is_symlink() for path in (entry_dir, temporary_dir)):
        raise ReviewedTaskRegistryError(
            f"unindexed reviewed task entry already exists: {entry_dir}"
        )
    temporary_dir.mkdir(parents=True)
    (temporary_dir / "registration.json").write_bytes(
        _pretty_json_bytes(registration)
    )
    (temporary_dir / "review_manifest.json").write_bytes(_pretty_json_bytes(review))
    for relative, payload in source["artifacts"].items():
        path = temporary_dir / "artifacts" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    temporary_dir.replace(entry_dir)

    registration_path = entry_dir / "registration.json"
    review_path = entry_dir / "review_manifest.json"
    descriptors = {
        relative: {
            "path": f"entries/{registration_id}/artifacts/{relative}",
            "sha256": digest,
        }
        for relative, digest in source["artifact_hashes"].items()
    }
    entry = {
        "registration_id": registration_id,
        "artifact_id": artifact_id,
        "scope": REGISTRY_SCOPE,
        "status": "approved",
        "task_name": source["semantic_key"]["task_name"],
        "semantic_key_sha256": source["semantic_key_sha256"],
        "artifact_hashes": source["artifact_hashes"],
        "runtime_dependency_hashes": source["runtime_dependency_hashes"],
        "registration_artifact": f"entries/{registration_id}/registration.json",
        "registration_artifact_sha256": _file_sha256(registration_path),
        "review_manifest_artifact": f"entries/{registration_id}/review_manifest.json",
        "review_manifest_artifact_sha256": _file_sha256(review_path),
        "artifacts": descriptors,
    }
    index["entries"].append(entry)
    index["entries"].sort(key=lambda item: item["registration_id"])
    _write_bytes_atomic(root / "index.json", _pretty_json_bytes(index))
    return _match_from_loaded(_load_entry(root, entry))


def _match_from_loaded(loaded: Mapping[str, Any]) -> dict[str, Any]:
    registration = loaded["registration"]
    return {
        "schema_version": 1,
        "registration_id": registration["registration_id"],
        "artifact_id": registration["artifact_id"],
        "status": "approved",
        "semantic_key": deepcopy(registration["semantic_key"]),
        "semantic_key_sha256": registration["semantic_key_sha256"],
        "runtime_dependency_hashes": deepcopy(
            registration["runtime_dependency_hashes"]
        ),
        "review_authority": deepcopy(registration["reviewer"]),
        "reviewed_at": registration["reviewed_at"],
        "review_attestation_paper_eligible": (
            registration["reviewer"].get("kind") == "human"
        ),
        "registry_dir": str(loaded["registry_dir"]),
        "verified_artifacts": deepcopy(loaded["verified_artifacts"]),
    }


def find_reviewed_task(
    registry_dir: str | Path,
    query: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Find one approved entry for the resolver's exact semantic-key query."""

    if not isinstance(query, Mapping) or set(query) != {
        "schema_version",
        "semantic_key",
        "semantic_key_sha256",
    }:
        raise ReviewedTaskRegistryError("reviewed task query fields are invalid")
    if query.get("schema_version") != 1:
        raise ReviewedTaskRegistryError("reviewed task query schema_version must be 1")
    semantic_key = _validate_semantic_key(query.get("semantic_key"))
    semantic_hash = _canonical_sha256(semantic_key)
    if query.get("semantic_key_sha256") != semantic_hash:
        raise ReviewedTaskRegistryError("reviewed task query semantic hash differs")
    root = _unresolved_root(registry_dir, label="reviewed task registry")
    index = load_reviewed_task_registry(root)
    matches = [
        entry
        for entry in index["entries"]
        if entry["semantic_key_sha256"] == semantic_hash
    ]
    if not matches:
        return None
    if len(matches) != 1:  # Defensive; load already rejects ambiguity.
        raise ReviewedTaskRegistryError("reviewed task query is ambiguous")
    loaded = _load_entry(root, matches[0])
    if loaded["registration"]["semantic_key"] != semantic_key:
        raise ReviewedTaskRegistryError("reviewed task semantic contract differs")
    match = _match_from_loaded(loaded)
    if repo_root is not None:
        validate_reviewed_task_runtime(match, repo_root)
    return match


def validate_reviewed_task_runtime(
    match: Mapping[str, Any], repo_root: str | Path
) -> dict[str, str]:
    """Verify imported official task/utils bytes before materialization."""

    expected = _validate_runtime_dependency_hashes(
        match.get("runtime_dependency_hashes")
        if isinstance(match, Mapping)
        else None
    )
    root = Path(repo_root).expanduser().resolve()
    actual = _runtime_dependency_hashes(root)
    if actual != expected:
        changed = sorted(
            relative
            for relative in RUNTIME_DEPENDENCY_PATHS
            if actual.get(relative) != expected.get(relative)
        )
        raise ReviewedTaskRegistryError(
            f"reviewed Task runtime dependencies changed: {changed}"
        )
    return actual


def copy_reviewed_task_artifacts(
    match: Mapping[str, Any],
    destination: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy only verified task artifacts into an absent or empty destination."""

    if not isinstance(match, Mapping):
        raise ReviewedTaskRegistryError("reviewed task match must be an object")
    for field in ("registry_dir", "registration_id", "artifact_id"):
        if not isinstance(match.get(field), str) or not match[field]:
            raise ReviewedTaskRegistryError(f"reviewed task match lacks {field}")
    root = _unresolved_root(match["registry_dir"], label="reviewed task registry")
    index = load_reviewed_task_registry(root)
    entry = next(
        (
            item
            for item in index["entries"]
            if item["registration_id"] == match["registration_id"]
        ),
        None,
    )
    if entry is None or entry["artifact_id"] != match["artifact_id"]:
        raise ReviewedTaskRegistryError("reviewed task match is not registered")
    loaded = _load_entry(root, entry)
    expected_match = _match_from_loaded(loaded)
    for field in (
        "semantic_key",
        "semantic_key_sha256",
        "runtime_dependency_hashes",
        "review_authority",
        "reviewed_at",
        "review_attestation_paper_eligible",
    ):
        if match.get(field) != expected_match[field]:
            raise ReviewedTaskRegistryError(f"reviewed task match {field} differs")
    if repo_root is not None:
        validate_reviewed_task_runtime(expected_match, repo_root)

    raw_destination = Path(destination).expanduser()
    if raw_destination.is_symlink():
        raise ReviewedTaskRegistryError("artifact destination must not be a symlink")
    target = raw_destination.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        pass
    else:
        raise ReviewedTaskRegistryError("artifact destination must be outside registry")
    if target.exists():
        if not target.is_dir():
            raise ReviewedTaskRegistryError("artifact destination must be a directory")
        if any(target.iterdir()):
            raise ReviewedTaskRegistryError("artifact destination must be empty")

    payloads: dict[str, bytes] = {}
    for relative, descriptor in loaded["verified_artifacts"].items():
        source_path = Path(descriptor["path"])
        payload = source_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
            raise ReviewedTaskRegistryError(
                f"reviewed task artifact changed before copy: {relative}"
            )
        payloads[relative] = payload
    target.mkdir(parents=True, exist_ok=True)
    for relative, payload in payloads.items():
        output = _safe_artifact(target, relative, label="artifact destination")
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomic(output, payload)
        if hashlib.sha256(output.read_bytes()).hexdigest() != (
            loaded["verified_artifacts"][relative]["sha256"]
        ):
            raise ReviewedTaskRegistryError(
                f"copied task artifact integrity failed: {relative}"
            )
    return {
        "registration_id": match["registration_id"],
        "artifact_id": match["artifact_id"],
        "runtime_dependency_hashes": expected_match[
            "runtime_dependency_hashes"
        ],
        "review_authority": expected_match["review_authority"],
        "reviewed_at": expected_match["reviewed_at"],
        "review_attestation_paper_eligible": expected_match[
            "review_attestation_paper_eligible"
        ],
        "destination": str(target),
        "files": {
            relative: loaded["verified_artifacts"][relative]["sha256"]
            for relative in sorted(payloads)
        },
    }


__all__ = [
    "RUNTIME_DEPENDENCY_PATHS",
    "ReviewedTaskRegistryError",
    "build_task_review_manifest_template",
    "copy_reviewed_task_artifacts",
    "find_reviewed_task",
    "install_reviewed_task",
    "load_reviewed_task_registry",
    "validate_reviewed_task_runtime",
    "validate_task_review_manifest",
]
