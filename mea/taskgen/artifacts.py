"""Auditable TaskArtifactBundle shared by all TaskGen routes."""

from __future__ import annotations

import ast
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any, Mapping

from .scene_checks import build_scene_check_spec
from .success_spec import (
    SuccessSpecError,
    validate_compiled_success_method,
    validate_success_spec,
)


class TaskArtifactBundleError(RuntimeError):
    """Raised when a materialized TaskGen run cannot be described honestly."""


SCENE_ORIGINS = {"generated_code", "bounded_overlay_wrapper", "official_reuse"}
SUCCESS_ORIGINS = {"compiled_success_spec", "official_reuse"}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_source(repo_root: Path, module: str) -> Path:
    return repo_root.joinpath(*module.split(".")).with_suffix(".py")


def _node_source(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno]) + "\n"


def _method_binding(
    repo_root: Path,
    *,
    module: str,
    class_name: str,
    method_name: str,
    origin: str,
) -> dict[str, Any]:
    source_path = _module_source(repo_root, module)
    relative = (
        str(source_path.relative_to(repo_root)).replace("\\", "/")
        if source_path.is_relative_to(repo_root)
        else str(source_path)
    )
    binding = {
        "method": method_name,
        "origin": origin,
        "module": module,
        "class_name": class_name,
        "source": relative,
        "source_available": source_path.is_file(),
        "source_sha256": None,
        "symbol_declared": False,
        "symbol_sha256": None,
        "resolution": "source_unavailable",
    }
    if not source_path.is_file():
        return binding
    source = source_path.read_text(encoding="utf-8")
    binding["source_sha256"] = _file_sha256(source_path)
    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as exc:
        raise TaskArtifactBundleError(f"cannot parse task source {source_path}: {exc}") from exc
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
    if len(methods) == 1:
        symbol = _node_source(source, methods[0])
        binding.update(
            {
                "symbol_declared": True,
                "symbol_sha256": hashlib.sha256(symbol.encode("utf-8")).hexdigest(),
                "resolution": "source_symbol",
            }
        )
    else:
        # Some unit fixtures and thin wrappers omit an inherited method.  Keep
        # the runtime binding explicit without inventing source provenance.
        binding["resolution"] = "runtime_or_inherited_method"
    return binding


def _declared_method_source(
    repo_root: Path,
    *,
    module: str,
    class_name: str,
    method_name: str,
) -> str | None:
    """Return one class-local method, excluding inherited/runtime bindings."""

    source_path = _module_source(repo_root, module)
    if not source_path.is_file():
        return None
    source = source_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as exc:
        raise TaskArtifactBundleError(
            f"cannot parse task source {source_path}: {exc}"
        ) from exc
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        return None
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    ]
    if len(methods) != 1:
        return None
    return textwrap.dedent(_node_source(source, methods[0]))


def _route_bindings(manifest: Mapping[str, Any]) -> tuple[str, str, str]:
    task_name = str(manifest.get("task_name") or "")
    task_module = str(manifest.get("task_module") or "")
    generation_kind = str(manifest.get("generation_kind") or "")
    mode = str(manifest.get("mode") or "")
    if generation_kind == "official_passthrough" or mode == "official":
        return "official_reuse", task_module, task_module
    if generation_kind == "bounded_variant_overlay" or mode == "reuse":
        return "bounded_overlay_wrapper", task_module, f"envs.{task_name}"
    return "generated_code", task_module, f"envs.{task_name}"


def write_task_artifact_bundle(
    repo_root: str | Path,
    run_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    task_proposal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write or refresh the run's scene/success/proposal evidence contract."""

    root = Path(repo_root).expanduser().resolve()
    run = Path(run_dir).expanduser().resolve()
    spec_path = run / "variant_spec.json"
    if not spec_path.is_file():
        raise TaskArtifactBundleError(f"variant spec is missing: {spec_path}")
    try:
        variant_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskArtifactBundleError("variant spec is not valid JSON") from exc
    task_name = str(manifest.get("task_name") or "").strip()
    task_module = str(manifest.get("task_module") or "").strip()
    if not task_name or not task_module or variant_spec.get("task_name") != task_name:
        raise TaskArtifactBundleError("manifest and VariantSpec task identity differ")

    scene_origin, scene_module, official_success_module = _route_bindings(manifest)
    success_spec_path = run / "generation/success_spec.json"
    if scene_origin == "generated_code" and not success_spec_path.is_file():
        unbound_generated_success = _method_binding(
            root,
            module=task_module,
            class_name=task_name,
            method_name="check_success",
            origin="compiled_success_spec",
        )
        if unbound_generated_success["symbol_declared"]:
            raise TaskArtifactBundleError(
                "generated check_success has no SuccessSpec provenance"
            )
    compiled_success = scene_origin == "generated_code" and success_spec_path.is_file()
    success_module = task_module if compiled_success else official_success_module
    success_origin = "compiled_success_spec" if compiled_success else "official_reuse"
    success_spec_sha256 = None
    if compiled_success:
        try:
            success_spec = validate_success_spec(
                json.loads(success_spec_path.read_text(encoding="utf-8"))
            )
        except (OSError, UnicodeError, json.JSONDecodeError, SuccessSpecError) as exc:
            raise TaskArtifactBundleError(
                f"generated SuccessSpec is invalid: {exc}"
            ) from exc
        if success_spec["task_name"] != task_name:
            raise TaskArtifactBundleError("SuccessSpec and task identity differ")
        compiled_method = _declared_method_source(
            root,
            module=task_module,
            class_name=task_name,
            method_name="check_success",
        )
        if compiled_method is None:
            raise TaskArtifactBundleError(
                "generated task does not declare the SuccessSpec check_success method"
            )
        try:
            validate_compiled_success_method(compiled_method, success_spec)
        except SuccessSpecError as exc:
            raise TaskArtifactBundleError(
                f"generated check_success does not match SuccessSpec: {exc}"
            ) from exc
        success_spec_sha256 = _canonical_sha256(success_spec)
    scene_check = build_scene_check_spec(
        variant_spec,
        task_proposal=task_proposal,
    )
    scene_check_path = run / "generation/scene_check_spec.json"
    scene_check_path.parent.mkdir(parents=True, exist_ok=True)
    scene_check_path.write_text(
        json.dumps(scene_check, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    bundle = {
        "schema_version": 1,
        "task_name": task_name,
        "task_module": task_module,
        "generation_kind": str(
            manifest.get("generation_kind")
            or ("generated_scene_code" if scene_origin == "generated_code" else manifest.get("mode"))
        ),
        "variant_spec_sha256": _canonical_sha256(variant_spec),
        "task_proposal_sha256": (
            _canonical_sha256(task_proposal) if task_proposal is not None else None
        ),
        "scene_method": _method_binding(
            root,
            module=scene_module,
            class_name=task_name,
            method_name="load_actors",
            origin=scene_origin,
        ),
        "success_method": _method_binding(
            root,
            module=success_module,
            class_name=task_name,
            method_name="check_success",
            origin=success_origin,
        ),
        "success_semantics": (
            {
                "preserved": True,
                "authority": "compiled_success_spec_official_equivalent",
                "generated_by_model": False,
                "generated_from_spec": True,
                "success_spec": "generation/success_spec.json",
                "success_spec_sha256": success_spec_sha256,
            }
            if compiled_success
            else {
                "preserved": True,
                "authority": "official_check_success",
                "generated_by_model": False,
            }
        ),
        "scene_check_spec": {
            "artifact": str(scene_check_path.relative_to(run)).replace("\\", "/"),
            "sha256": _canonical_sha256(scene_check),
            "source": scene_check["source"],
            "repair_mode": scene_check["repair_policy"]["mode"],
        },
        "boundary": (
            "TaskArtifactBundle binds executable scene and success methods. "
            + (
                "The success method was compiled from a restricted, validated "
                "SuccessSpec; it was not arbitrary model-written Python."
                if compiled_success
                else "It does not claim that official success logic was model-generated."
            )
        ),
    }
    validate_task_artifact_bundle(bundle)
    bundle_path = run / "generation/task_artifact_bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle


def validate_task_artifact_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "task_name",
        "task_module",
        "generation_kind",
        "variant_spec_sha256",
        "task_proposal_sha256",
        "scene_method",
        "success_method",
        "success_semantics",
        "scene_check_spec",
        "boundary",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise TaskArtifactBundleError(
            f"TaskArtifactBundle fields must be exactly {sorted(required)}"
        )
    if value.get("schema_version") != 1:
        raise TaskArtifactBundleError("TaskArtifactBundle.schema_version must be 1")
    scene = value.get("scene_method")
    success = value.get("success_method")
    if not isinstance(scene, Mapping) or scene.get("origin") not in SCENE_ORIGINS:
        raise TaskArtifactBundleError("scene method has no supported origin")
    if not isinstance(success, Mapping) or success.get("origin") not in SUCCESS_ORIGINS:
        raise TaskArtifactBundleError("success method has no supported origin")
    semantics = value.get("success_semantics")
    if success.get("origin") == "official_reuse":
        if not isinstance(semantics, Mapping) or semantics != {
            "preserved": True,
            "authority": "official_check_success",
            "generated_by_model": False,
        }:
            raise TaskArtifactBundleError("official success semantics were not preserved")
    else:
        if (
            not success.get("symbol_declared")
            or not isinstance(semantics, Mapping)
            or set(semantics)
            != {
                "preserved",
                "authority",
                "generated_by_model",
                "generated_from_spec",
                "success_spec",
                "success_spec_sha256",
            }
            or semantics.get("preserved") is not True
            or semantics.get("authority")
            != "compiled_success_spec_official_equivalent"
            or semantics.get("generated_by_model") is not False
            or semantics.get("generated_from_spec") is not True
            or semantics.get("success_spec") != "generation/success_spec.json"
            or not isinstance(semantics.get("success_spec_sha256"), str)
            or len(semantics["success_spec_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in semantics["success_spec_sha256"]
            )
        ):
            raise TaskArtifactBundleError(
                "compiled success method lacks a validated SuccessSpec binding"
            )
    return json.loads(json.dumps(value, ensure_ascii=False))


__all__ = [
    "TaskArtifactBundleError",
    "validate_task_artifact_bundle",
    "write_task_artifact_bundle",
]
