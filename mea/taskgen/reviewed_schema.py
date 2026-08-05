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
from pathlib import Path
from typing import Any, Mapping

from .success_spec import (
    SuccessSpecError,
    success_spec_validation_report,
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
