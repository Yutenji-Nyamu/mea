"""Catalog-independent RoboTwin TaskGen orchestration.

The backend consumes one runtime ``ExperimentCandidate`` and a thin task
adapter.  The adapter describes the official task program and exposes the
small simulator-specific hooks needed to validate generated code.  It does
not enumerate aspects, variants, metrics, or planner routes.

Task reuse is exact and semantic.  A miss invokes the provider once, with at
most one local regeneration covering static, fixture, render, and expert
validation.  Policy execution remains outside this module.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from mea.planner.experiment_candidate import (
    ExperimentCandidateError,
    validate_experiment_candidate,
)
from mea.toolkit.schema import TaskSchemaError, load_task_schema

from .provider_scene_checker import (
    TextProvider,
    compose_prompt,
    retrieve_class_methods,
    run_provider_codegen,
    text_sha256,
    validate_method_ast,
    validate_provider_run_id,
    write_candidate_artifacts,
)


class GenericTaskGenError(RuntimeError):
    """Raised when a dynamic candidate cannot be reused or generated."""


ValidateMethods = Callable[
    [Mapping[str, str], Mapping[str, Any]], Mapping[str, Any]
]
BuildModule = Callable[[Mapping[str, str], Mapping[str, Any]], str]
PreflightCandidate = Callable[
    [Path, str, Mapping[str, Any]], Mapping[str, Any]
]
ResolveMetric = Callable[[Mapping[str, Any]], str]
ResolveCheckerContract = Callable[
    [Mapping[str, Any]], Mapping[str, Any]
]
ExactTaskLookup = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
CheckerFixtureValidator = Callable[
    [Mapping[str, str], Mapping[str, Any]], list[Mapping[str, Any]]
]

_TASK_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_GENERIC_DIRECT_CALLS = {
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
_GENERIC_SAFE_MODULE_CALLS = {
    ("np", "abs"),
    ("np", "all"),
    ("np", "any"),
    ("np", "array"),
    ("np", "asarray"),
    ("np", "sum"),
}


@dataclass(frozen=True)
class GenericTaskGenHooks:
    """Simulator-facing hooks shared by every semantic candidate for a task.

    Hooks may depend on task APIs and fixtures, but must not select a concern,
    aspect, template, or planner route.  Those decisions already live in the
    ``ExperimentCandidate``.
    """

    validate_methods: ValidateMethods
    build_module: BuildModule
    preflight_candidate: PreflightCandidate
    resolve_metric: ResolveMetric
    resolve_checker_contract: ResolveCheckerContract
    prompt_constraints: str = ""


@dataclass(frozen=True)
class GenericRoboTwinTaskAdapter:
    """Thin description of one policy-compatible official RoboTwin task."""

    task_name: str
    official_source: str
    official_class: str
    task_schema: Mapping[str, Any]
    documentation_paths: tuple[str, ...]
    asset_paths: tuple[str, ...]
    hooks: GenericTaskGenHooks


def _attribute_parts(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    cursor = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        return None
    parts.append(cursor.id)
    return tuple(reversed(parts))


def _official_class(
    source_path: Path, *, class_name: str
) -> tuple[str, ast.ClassDef]:
    try:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise GenericTaskGenError(
            f"official task source is invalid: {source_path}: {exc}"
        ) from exc
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise GenericTaskGenError(
            f"official task must declare one class {class_name!r}"
        )
    methods = {
        node.name
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = sorted({"load_actors", "check_success"} - methods)
    if missing:
        raise GenericTaskGenError(
            f"official task class lacks required methods: {missing}"
        )
    return source, classes[0]


def _reject_pose_property_item_assignment(
    methods: Mapping[str, ast.AST],
) -> None:
    """Reject writes to indexed SAPIEN Pose properties.

    ``Pose.p`` and ``Pose.q`` are exposed as array values.  Mutating an item
    such as ``pose.p[0] += 0.08`` changes that temporary array, not the Pose
    passed to ``create_actor``.  The code is syntactically valid but silently
    leaves the generated scene identical to the official control.
    """

    for method_name, method in methods.items():
        for node in ast.walk(method):
            targets: list[ast.AST] = []
            if isinstance(node, ast.AugAssign):
                targets = [node.target]
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    list(node.targets)
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr in {"p", "q"}
                ):
                    raise GenericTaskGenError(
                        f"{method_name} mutates Pose.{target.value.attr} by "
                        "indexed assignment; construct a new sapien.Pose "
                        "instead"
                    )


def _derived_ast_policy(
    source_path: Path, *, class_name: str
) -> dict[str, Any]:
    """Derive allowed calls from the official task instead of a task menu."""

    source, class_node = _official_class(
        source_path, class_name=class_name
    )
    safe_direct_calls = set(_GENERIC_DIRECT_CALLS)
    safe_module_calls: set[tuple[str, ...]] = set(
        _GENERIC_SAFE_MODULE_CALLS
    )
    safe_method_calls: set[str] = set()
    allowed_private_attributes = {
        node.attr
        for node in ast.walk(class_node)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_")
    }
    trusted_nodes: list[ast.AST] = [class_node]
    policy_digest = hashlib.sha256(source.encode("utf-8"))
    utils_dir = source_path.parent / "utils"
    if utils_dir.is_dir():
        for utility_path in sorted(utils_dir.glob("*.py")):
            try:
                utility_source = utility_path.read_text(encoding="utf-8")
                utility_tree = ast.parse(
                    utility_source,
                    filename=str(utility_path),
                )
            except (OSError, UnicodeError, SyntaxError) as exc:
                raise GenericTaskGenError(
                    f"trusted simulator utility is invalid: "
                    f"{utility_path}: {exc}"
                ) from exc
            policy_digest.update(
                utility_path.name.encode("utf-8")
                + b"\0"
                + utility_source.encode("utf-8")
            )
            trusted_nodes.append(utility_tree)
            safe_direct_calls.update(
                node.name
                for node in utility_tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and not node.name.startswith("_")
            )
    for node in (
        descendant
        for trusted in trusted_nodes
        for descendant in ast.walk(trusted)
    ):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            safe_direct_calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            parts = _attribute_parts(node.func)
            if parts is not None:
                safe_module_calls.add(parts)
            safe_method_calls.add(node.func.attr)
    return {
        "policy_id": (
            "generic_official_api_ast_v1:"
            + policy_digest.hexdigest()[:16]
        ),
        "safe_direct_calls": safe_direct_calls,
        "safe_module_calls": safe_module_calls,
        "safe_method_calls": safe_method_calls,
        "allowed_private_attributes": allowed_private_attributes,
    }


def validate_generic_task_methods(
    methods: Mapping[str, str],
    *,
    official_source: str | Path,
    official_class: str,
    required_method_changes: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Apply one data-derived AST and compile boundary to a method pair.

    A generated experiment may need only a scene or only a checker.  Requested
    methods must differ from the official implementation; unrequested methods
    must remain bytecode-equivalent at the AST level.  Legacy callers default
    to requiring both changes.
    """

    if (
        not isinstance(methods, Mapping)
        or set(methods) != {"load_actors", "check_success"}
        or any(not isinstance(methods[name], str) for name in methods)
    ):
        raise GenericTaskGenError(
            "generated methods must be load_actors/check_success strings"
        )
    source_path = Path(official_source).expanduser().resolve()
    policy = _derived_ast_policy(source_path, class_name=official_class)
    parsed = {
        name: validate_method_ast(
            methods[name],
            name,
            safe_direct_calls=policy["safe_direct_calls"],
            safe_module_calls=policy["safe_module_calls"],
            safe_method_calls=policy["safe_method_calls"],
            allowed_private_attributes=policy[
                "allowed_private_attributes"
            ],
            error_type=GenericTaskGenError,
        )
        for name in ("load_actors", "check_success")
    }
    _reject_pose_property_item_assignment(parsed)
    _official_source, official_node = _official_class(
        source_path,
        class_name=official_class,
    )
    official_methods = {
        node.name: node
        for node in official_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"load_actors", "check_success"}
    }
    changed_from_official = {
        name: ast.dump(parsed[name], include_attributes=False)
        != ast.dump(official_methods[name], include_attributes=False)
        for name in ("load_actors", "check_success")
    }
    required_changes = (
        {"load_actors": True, "check_success": True}
        if required_method_changes is None
        else dict(required_method_changes)
    )
    if (
        set(required_changes) != {"load_actors", "check_success"}
        or any(not isinstance(value, bool) for value in required_changes.values())
    ):
        raise GenericTaskGenError(
            "required_method_changes must map load_actors/check_success to bool"
        )
    missing_changes = sorted(
        name
        for name, required in required_changes.items()
        if required and not changed_from_official[name]
    )
    unexpected_changes = sorted(
        name
        for name, required in required_changes.items()
        if not required and changed_from_official[name]
    )
    if missing_changes:
        raise GenericTaskGenError(
            "generated methods copied requested official implementations "
            f"unchanged: {missing_changes}"
        )
    if unexpected_changes:
        raise GenericTaskGenError(
            "generated methods changed unrequested official implementations: "
            f"{unexpected_changes}"
        )
    for name in ("load_actors", "check_success"):
        compile(methods[name], f"<generic-{name}>", "exec")
    return {
        "valid": True,
        "policy": policy["policy_id"],
        "scene_ast_nodes": sum(
            1 for _ in ast.walk(parsed["load_actors"])
        ),
        "success_ast_nodes": sum(
            1 for _ in ast.walk(parsed["check_success"])
        ),
        "scene_sha256": text_sha256(methods["load_actors"]),
        "success_sha256": text_sha256(methods["check_success"]),
        "changed_from_official": changed_from_official,
        "required_method_changes": required_changes,
    }


def build_generic_task_subclass_module(
    methods: Mapping[str, str],
    *,
    official_module: str,
    official_class: str,
) -> str:
    """Build a thin subclass while inheriting the official module namespace."""

    if (
        not isinstance(official_module, str)
        or not all(
            part.isidentifier() and not part.startswith("_")
            for part in official_module.split(".")
        )
    ):
        raise GenericTaskGenError(
            "official_module must be a public dotted Python identifier"
        )
    if (
        not isinstance(official_class, str)
        or not official_class.isidentifier()
        or official_class.startswith("_")
    ):
        raise GenericTaskGenError(
            "official_class must be a public Python identifier"
        )
    if (
        not isinstance(methods, Mapping)
        or set(methods) != {"load_actors", "check_success"}
    ):
        raise GenericTaskGenError(
            "subclass builder requires load_actors and check_success"
        )
    scene = textwrap.indent(
        textwrap.dedent(str(methods["load_actors"])).strip(), "    "
    )
    checker = textwrap.indent(
        textwrap.dedent(str(methods["check_success"])).strip(), "    "
    )
    source = (
        '"""Provider-generated RoboTwin task candidate."""\n\n'
        f"import {official_module} as _official_task_module\n"
        f"from {official_module} import *\n\n\n"
        f"class {official_class}("
        f"_official_task_module.{official_class}):\n"
        f"{scene}\n\n"
        f"{checker}\n\n"
        "    def mea_official_check_success(self):\n"
        "        \"\"\"Evaluate the untouched official core predicate.\"\"\"\n"
        f"        return _official_task_module.{official_class}."
        "check_success(self)\n"
    )
    compile(source, "<generic-robotwin-subclass>", "exec")
    return source


def _model_names_from_source(
    source_path: Path, *, class_name: str
) -> set[str]:
    _source, class_node = _official_class(
        source_path, class_name=class_name
    )
    result: set[str] = set()
    for node in ast.walk(class_node):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "modelname"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
                and keyword.value.value
            ):
                result.add(keyword.value.value)
    return result


def _discover_task_documents(
    repo_root: Path, *, task_name: str
) -> tuple[str, ...]:
    candidates = (
        f"description/task_instruction/{task_name}.json",
        f"mea/knowledge/tasks/{task_name}.md",
        f"envs/{task_name}/README.Agent.md",
        f"envs/{task_name}.README.Agent.md",
    )
    return tuple(
        relative for relative in candidates if (repo_root / relative).is_file()
    )


def _discover_asset_descriptions(
    repo_root: Path,
    *,
    source_path: Path,
    class_name: str,
    task_schema: Mapping[str, Any],
) -> tuple[str, ...]:
    model_names = _model_names_from_source(
        source_path, class_name=class_name
    )
    for actor in task_schema.get("tracked_actors", []):
        if isinstance(actor, Mapping):
            scene_name = actor.get("scene_name")
            if isinstance(scene_name, str) and scene_name:
                model_names.add(scene_name)
    paths: list[str] = []
    for model_name in sorted(model_names):
        directory = repo_root / "description/objects_description" / model_name
        if directory.is_dir():
            paths.extend(
                path.relative_to(repo_root).as_posix()
                for path in sorted(directory.glob("*.json"))
                if path.is_file()
            )
    return tuple(paths)


def load_generic_robotwin_task_adapter(
    repo_root: str | Path,
    task_name: str,
    *,
    checker_fixtures: CheckerFixtureValidator,
    preflight_candidate: PreflightCandidate,
    resolve_metric: ResolveMetric,
    resolve_checker_contract: ResolveCheckerContract,
    prompt_constraints: str = "",
) -> GenericRoboTwinTaskAdapter:
    """Discover a thin adapter for any source/schema-backed RoboTwin task.

    The factory has no task-name, concern, template, or metric registry.
    Semantic fixtures and simulator preflight remain explicit injected hooks;
    the factory never marks either gate as passed by default.
    """

    if not isinstance(task_name, str) or not _TASK_NAME.fullmatch(task_name):
        raise GenericTaskGenError("task_name is not a RoboTwin identifier")
    if not callable(checker_fixtures):
        raise GenericTaskGenError(
            "checker_fixtures must be an explicit callable"
        )
    if not callable(preflight_candidate):
        raise GenericTaskGenError(
            "preflight_candidate must be an explicit callable"
        )
    if not callable(resolve_metric) or not callable(
        resolve_checker_contract
    ):
        raise GenericTaskGenError(
            "metric and checker contract resolvers must be callable"
        )
    root = Path(repo_root).expanduser().resolve()
    relative_source = f"envs/{task_name}.py"
    source_path = _resolve_repo_file(
        root, relative_source, label="official task source"
    )
    _official_class(source_path, class_name=task_name)
    try:
        schema = load_task_schema(root, task_name)
    except TaskSchemaError as exc:
        raise GenericTaskGenError(
            f"task toolkit schema is unavailable: {exc}"
        ) from exc
    readme = root / "mea/taskgen/README.Agent.md"
    if not readme.is_file():
        raise GenericTaskGenError(
            "TaskGen README.Agent.md is unavailable"
        )

    def validate(
        methods: Mapping[str, str],
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        report = validate_generic_task_methods(
            methods,
            official_source=source_path,
            official_class=task_name,
            required_method_changes={
                "load_actors": candidate.get("scene_need") is not None,
                "check_success": candidate.get("checker_need") is not None,
            },
        )
        try:
            raw_fixtures = checker_fixtures(methods, candidate)
        except GenericTaskGenError:
            raise
        except Exception as exc:
            raise GenericTaskGenError(
                "checker fixture hook failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(raw_fixtures, list):
            raise GenericTaskGenError(
                "checker fixture hook must return a list"
            )
        report["checker_fixtures"] = [
            deepcopy(dict(item))
            if isinstance(item, Mapping)
            else item
            for item in raw_fixtures
        ]
        report["checker_fixture_count"] = len(raw_fixtures)
        return report

    module_name = Path(relative_source).with_suffix("").as_posix().replace(
        "/", "."
    )
    hooks = GenericTaskGenHooks(
        validate_methods=validate,
        build_module=lambda methods, _candidate: (
            build_generic_task_subclass_module(
                methods,
                official_module=module_name,
                official_class=task_name,
            )
        ),
        preflight_candidate=preflight_candidate,
        resolve_metric=resolve_metric,
        resolve_checker_contract=resolve_checker_contract,
        prompt_constraints=prompt_constraints,
    )
    return GenericRoboTwinTaskAdapter(
        task_name=task_name,
        official_source=relative_source,
        official_class=task_name,
        task_schema=schema,
        documentation_paths=_discover_task_documents(
            root, task_name=task_name
        ),
        asset_paths=_discover_asset_descriptions(
            root,
            source_path=source_path,
            class_name=task_name,
            task_schema=schema,
        ),
        hooks=hooks,
    )


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GenericTaskGenError(
            f"TaskGen semantic identity is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GenericTaskGenError(f"{field} must be a non-empty string")
    return value.strip()


def _need_description(value: Any, *, field: str) -> str:
    """Normalize a typed need returned by an adapter hook."""

    if isinstance(value, Mapping):
        value = value.get("description")
    return _text(value, field=field)


def _relative_path(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise GenericTaskGenError(f"{field} must be a repository-relative path")
    return path.as_posix()


def _normalize_adapter(
    adapter: GenericRoboTwinTaskAdapter,
) -> dict[str, Any]:
    if not isinstance(adapter, GenericRoboTwinTaskAdapter):
        raise GenericTaskGenError(
            "adapter must be a GenericRoboTwinTaskAdapter"
        )
    hooks = adapter.hooks
    if not isinstance(hooks, GenericTaskGenHooks):
        raise GenericTaskGenError(
            "adapter.hooks must be GenericTaskGenHooks"
        )
    for field in (
        "validate_methods",
        "build_module",
        "preflight_candidate",
        "resolve_metric",
        "resolve_checker_contract",
    ):
        if not callable(getattr(hooks, field)):
            raise GenericTaskGenError(f"adapter hook {field} must be callable")
    if not isinstance(adapter.task_schema, Mapping):
        raise GenericTaskGenError("adapter.task_schema must be an object")
    documentation_paths = tuple(
        _relative_path(item, field="adapter.documentation_paths")
        for item in adapter.documentation_paths
    )
    asset_paths = tuple(
        _relative_path(item, field="adapter.asset_paths")
        for item in adapter.asset_paths
    )
    if len(documentation_paths) != len(set(documentation_paths)):
        raise GenericTaskGenError("adapter documentation paths must be unique")
    if len(asset_paths) != len(set(asset_paths)):
        raise GenericTaskGenError("adapter asset paths must be unique")
    prompt_constraints = hooks.prompt_constraints
    if not isinstance(prompt_constraints, str):
        raise GenericTaskGenError(
            "adapter hook prompt_constraints must be a string"
        )
    return {
        "schema_version": 1,
        "task_name": _text(adapter.task_name, field="adapter.task_name"),
        "official_source": _relative_path(
            adapter.official_source, field="adapter.official_source"
        ),
        "official_class": _text(
            adapter.official_class, field="adapter.official_class"
        ),
        "task_schema": deepcopy(dict(adapter.task_schema)),
        "documentation_paths": list(documentation_paths),
        "asset_paths": list(asset_paths),
        "generation_hook_contract": {
            "methods": ["load_actors", "check_success"],
            "static_and_fixture_validation": True,
            "render_preflight": True,
            "expert_preflight": True,
            "local_regeneration_limit": 1,
        },
    }


def _resolve_repo_file(
    repo_root: Path, relative: str, *, label: str
) -> Path:
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise GenericTaskGenError(f"{label} escapes the repository") from exc
    if not path.is_file():
        raise GenericTaskGenError(f"{label} is unavailable: {relative}")
    return path


def generic_task_semantic_key(
    candidate: Mapping[str, Any],
    adapter: GenericRoboTwinTaskAdapter,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build the exact reuse identity without Query wording or template ids."""

    try:
        normalized_candidate = validate_experiment_candidate(candidate)
    except ExperimentCandidateError as exc:
        raise GenericTaskGenError(f"invalid ExperimentCandidate: {exc}") from exc
    normalized_adapter = _normalize_adapter(adapter)
    if normalized_candidate["base_task"] != normalized_adapter["task_name"]:
        raise GenericTaskGenError(
            "ExperimentCandidate.base_task differs from adapter.task_name"
        )
    root = Path(repo_root).expanduser().resolve()
    official = _resolve_repo_file(
        root,
        normalized_adapter["official_source"],
        label="adapter official source",
    )
    asset_dependencies = []
    for relative in normalized_adapter["asset_paths"]:
        path = _resolve_repo_file(
            root, str(relative), label="adapter asset"
        )
        asset_dependencies.append(
            {
                "path": str(relative),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    documentation_dependencies = []
    for relative in normalized_adapter["documentation_paths"]:
        path = _resolve_repo_file(
            root, str(relative), label="adapter documentation"
        )
        documentation_dependencies.append(
            {
                "path": str(relative),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    taskgen_readme = _resolve_repo_file(
        root,
        "mea/taskgen/README.Agent.md",
        label="TaskGen README.Agent",
    )
    ast_policy = _derived_ast_policy(
        official, class_name=normalized_adapter["official_class"]
    )
    return {
        "schema_version": 1,
        "base_task": normalized_candidate["base_task"],
        "semantic_concern": normalized_candidate["semantic_concern"],
        "scene_need": normalized_candidate["scene_need"],
        "checker_need": normalized_candidate["checker_need"],
        "adapter_contract": {
            "official_source": normalized_adapter["official_source"],
            "official_source_sha256": hashlib.sha256(
                official.read_bytes()
            ).hexdigest(),
            "official_class": normalized_adapter["official_class"],
            "task_schema": normalized_adapter["task_schema"],
            "asset_dependencies": asset_dependencies,
            "documentation_dependencies": documentation_dependencies,
            "taskgen_readme_sha256": hashlib.sha256(
                taskgen_readme.read_bytes()
            ).hexdigest(),
            "prompt_constraints_sha256": text_sha256(
                adapter.hooks.prompt_constraints
            ),
            "ast_policy": ast_policy["policy_id"],
            "generation_hook_contract": normalized_adapter[
                "generation_hook_contract"
            ],
        },
    }


def _validate_exact_match(
    value: Mapping[str, Any],
    *,
    semantic_key: Mapping[str, Any],
    semantic_key_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GenericTaskGenError("exact task lookup must return an object or null")
    match = deepcopy(dict(value))
    if match.get("schema_version") != 1 or match.get("status") != "approved":
        raise GenericTaskGenError("exact task match is not approved")
    if match.get("semantic_key") != dict(semantic_key):
        raise GenericTaskGenError("exact task match semantic key differs")
    if match.get("semantic_key_sha256") != semantic_key_sha256:
        raise GenericTaskGenError("exact task match semantic hash differs")
    artifact_id = match.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise GenericTaskGenError("exact task match lacks artifact_id")
    return match


def _read_generation_context(
    repo_root: Path,
    *,
    adapter: Mapping[str, Any],
) -> str:
    official_path = _resolve_repo_file(
        repo_root,
        str(adapter["official_source"]),
        label="adapter official source",
    )
    try:
        official_methods = retrieve_class_methods(
            official_path,
            class_name=str(adapter["official_class"]),
            method_names=("load_actors", "check_success"),
            error_type=GenericTaskGenError,
        )
    except GenericTaskGenError:
        raise
    sections = [
        "OFFICIAL BASE TASK METHODS:\n```python\n"
        + official_methods.rstrip()
        + "\n```",
        "TASK TELEMETRY/EXECUTION SCHEMA:\n"
        + json.dumps(
            adapter["task_schema"],
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
    ]
    for relative in adapter["documentation_paths"]:
        path = _resolve_repo_file(
            repo_root, str(relative), label="adapter documentation"
        )
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise GenericTaskGenError(
                f"adapter documentation is not UTF-8 text: {relative}"
            ) from exc
        sections.append(
            f"DOCUMENTATION `{relative}`:\n{content.strip()}"
        )
    asset_descriptors: list[dict[str, Any]] = []
    for relative in adapter["asset_paths"]:
        path = _resolve_repo_file(
            repo_root, str(relative), label="adapter asset"
        )
        asset_descriptors.append(
            {
                "path": str(relative),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    sections.append(
        "AVAILABLE ASSETS:\n"
        + json.dumps(
            asset_descriptors, ensure_ascii=False, sort_keys=True, indent=2
        )
    )
    return "\n\n".join(sections) + "\n"


def _core_prompt(
    candidate: Mapping[str, Any],
    adapter: Mapping[str, Any],
    *,
    prompt_constraints: str,
) -> str:
    constraint_section = (
        "\n\nSIMULATOR-SPECIFIC API CONSTRAINTS:\n"
        + prompt_constraints.strip()
        if prompt_constraints.strip()
        else ""
    )
    return (
        "Generate one RoboTwin experiment from the open Query-derived "
        "candidate below. Retrieve semantics from the official base program, "
        "but implement the requested scene and checker rather than selecting "
        "a catalog template.\n\n"
        "EXPERIMENT CANDIDATE:\n"
        + json.dumps(candidate, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n\nTHIN TASK ADAPTER:\n"
        + json.dumps(adapter, ensure_ascii=False, sort_keys=True, indent=2)
        + constraint_section
        + "\n\nOUTPUT CONTRACT:\n"
        "Return one strict JSON object with exactly two string fields, "
        "load_actors and check_success. Each field must contain one complete "
        "Python method with only self. A non-null scene_need requires a changed "
        "load_actors method; a null scene_need requires the exact official "
        "load_actors method from the retrieved source. A non-null checker_need "
        "requires a changed check_success method; a null checker_need requires "
        "the exact official check_success method. Do not return Markdown, a "
        "template id, or an explanation."
    )


def _normalize_validation(
    value: Mapping[str, Any],
    *,
    methods: Mapping[str, str],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GenericTaskGenError("method validation hook must return an object")
    report = deepcopy(dict(value))
    if report.get("valid") is not True:
        raise GenericTaskGenError("method validation did not return valid=true")
    policy = report.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        raise GenericTaskGenError("method validation lacks an AST policy id")
    fixtures = report.get("checker_fixtures")
    if not fixtures and isinstance(preflight.get("checker_fixtures"), list):
        fixtures = deepcopy(preflight["checker_fixtures"])
        report["checker_fixtures"] = fixtures
    if (
        not isinstance(fixtures, list)
        or not fixtures
        or any(
            not isinstance(item, Mapping) or item.get("passed") is not True
            for item in fixtures
        )
    ):
        raise GenericTaskGenError(
            "method validation requires passing checker fixtures"
        )
    if preflight.get("render_passed") is not True:
        raise GenericTaskGenError("render preflight did not pass")
    if preflight.get("expert_passed") is not True:
        raise GenericTaskGenError("expert preflight did not pass")
    if preflight.get("scene_change_passed") is not True:
        raise GenericTaskGenError(
            "preflight did not verify a scene change from the official control"
        )
    report["scene_sha256"] = text_sha256(methods["load_actors"])
    report["success_sha256"] = text_sha256(methods["check_success"])
    report["checker_fixture_count"] = len(fixtures)
    report["preflight"] = deepcopy(dict(preflight))
    report["model_written_python"] = True
    report["restricted_success_spec_compiler_used"] = False
    return report


class GenericRoboTwinTaskGenBackend:
    """Reuse or generate one TaskGen artifact without a task/aspect catalog."""

    def __init__(
        self,
        repo_root: str | Path,
        provider: TextProvider,
        *,
        model: str,
        find_exact: ExactTaskLookup | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.provider = provider
        self.model = _text(model, field="model")
        self.find_exact = find_exact

    def materialize(
        self,
        candidate: Mapping[str, Any],
        adapter: GenericRoboTwinTaskAdapter,
        *,
        run_id: str,
        max_regenerations: int = 1,
        ablation_switches: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return an exact match or one fully preflighted generated artifact."""

        run_id = validate_provider_run_id(
            run_id, error_type=GenericTaskGenError
        )
        try:
            normalized_candidate = validate_experiment_candidate(candidate)
        except ExperimentCandidateError as exc:
            raise GenericTaskGenError(
                f"invalid ExperimentCandidate: {exc}"
            ) from exc
        normalized_adapter = _normalize_adapter(adapter)
        semantic_key = generic_task_semantic_key(
            normalized_candidate, adapter, repo_root=self.repo_root
        )
        semantic_hash = _canonical_sha256(semantic_key)
        lookup_query = {
            "schema_version": 1,
            "semantic_key": deepcopy(semantic_key),
            "semantic_key_sha256": semantic_hash,
        }
        if ablation_switches is not None and (
            not isinstance(ablation_switches, Mapping)
            or any(
                not isinstance(value, bool)
                for value in ablation_switches.values()
            )
        ):
            raise GenericTaskGenError(
                "ablation_switches must map component names to booleans"
            )
        # Every Table 3 arm, including its all-enabled control, must generate
        # independently. Otherwise a prior cell can silently supply its code.
        reuse_allowed = ablation_switches is None
        match = (
            self.find_exact(deepcopy(lookup_query))
            if self.find_exact is not None and reuse_allowed
            else None
        )
        if match is not None:
            return {
                "schema_version": 1,
                "status": "reused",
                "route": "exact_generated_task_reuse",
                "candidate": normalized_candidate,
                "semantic_key": semantic_key,
                "semantic_key_sha256": semantic_hash,
                "provider_required": False,
                "provider_call_count": 0,
                "exact_match": _validate_exact_match(
                    match,
                    semantic_key=semantic_key,
                    semantic_key_sha256=semantic_hash,
                ),
            }

        rag_context = _read_generation_context(
            self.repo_root, adapter=normalized_adapter
        )
        prompt, prompt_context = compose_prompt(
            core_contract=_core_prompt(
                normalized_candidate,
                normalized_adapter,
                prompt_constraints=adapter.hooks.prompt_constraints,
            ),
            rag_context=rag_context,
            repo_root=self.repo_root,
            ablation_switches=ablation_switches,
            error_type=GenericTaskGenError,
        )
        attempt_root = (
            self.repo_root / "mea/generated_task_attempts" / run_id
        )
        validation_counter = 0
        accepted_module: dict[str, str] = {}

        def validate(methods: Mapping[str, Any]) -> dict[str, Any]:
            nonlocal validation_counter
            validation_counter += 1
            typed_methods = {
                name: str(methods[name])
                for name in ("load_actors", "check_success")
            }
            try:
                raw_validation = adapter.hooks.validate_methods(
                    typed_methods, normalized_candidate
                )
                module_source = adapter.hooks.build_module(
                    typed_methods, normalized_candidate
                )
                if not isinstance(module_source, str) or not module_source.strip():
                    raise GenericTaskGenError(
                        "module builder must return non-empty Python source"
                    )
                compile(
                    module_source,
                    f"<generic-taskgen-{run_id}-attempt-{validation_counter}>",
                    "exec",
                )
                attempt_dir = (
                    attempt_root / f"attempt_{validation_counter:02d}"
                )
                (attempt_dir / "candidate_task.py").write_text(
                    module_source, encoding="utf-8"
                )
                preflight = adapter.hooks.preflight_candidate(
                    attempt_dir, module_source, normalized_candidate
                )
                validation = _normalize_validation(
                    raw_validation,
                    methods=typed_methods,
                    preflight=preflight,
                )
            except GenericTaskGenError:
                raise
            except Exception as exc:
                raise GenericTaskGenError(
                    "generic TaskGen validation hook failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            accepted_module["source"] = module_source
            return validation

        generated = run_provider_codegen(
            attempt_root=attempt_root,
            proposal=normalized_candidate,
            prompt=prompt,
            provider=self.provider,
            model=self.model,
            validate=validate,
            error_type=GenericTaskGenError,
            max_regenerations=max_regenerations,
        )
        if "source" not in accepted_module:
            raise GenericTaskGenError("accepted TaskGen module source is missing")
        metric = _need_description(
            adapter.hooks.resolve_metric(normalized_candidate),
            field="generated metric",
        )
        checker_contract = adapter.hooks.resolve_checker_contract(
            normalized_candidate
        )
        if not isinstance(checker_contract, Mapping):
            raise GenericTaskGenError(
                "checker contract hook must return an object"
            )
        run_dir = self.repo_root / "mea/generated_tasks" / run_id
        manifest = write_candidate_artifacts(
            run_dir=run_dir,
            task_name=normalized_adapter["task_name"],
            proposal=normalized_candidate,
            prompt=prompt,
            prompt_context=prompt_context,
            generated=generated,
            module_source=accepted_module["source"],
            model=self.model,
            metric=metric,
            checker_contract=checker_contract,
        )
        resolution = {
            "schema_version": 1,
            "status": "generated",
            "route": "generic_provider_scene_checker_codegen",
            "candidate": normalized_candidate,
            "adapter": normalized_adapter,
            "semantic_key": semantic_key,
            "semantic_key_sha256": semantic_hash,
            "provider_required": True,
            "provider_call_count": generated["attempt_summary"]["runtime"][
                "provider_calls"
            ],
            "local_regeneration_count": generated["attempt_summary"][
                "regenerations_used"
            ],
            "run_dir": str(run_dir),
            "candidate_manifest": manifest,
            "validation": deepcopy(dict(generated["validation"])),
        }
        (run_dir / "generic_taskgen_resolution.json").write_text(
            json.dumps(
                resolution,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return resolution


__all__ = [
    "ExactTaskLookup",
    "GenericRoboTwinTaskAdapter",
    "GenericRoboTwinTaskGenBackend",
    "GenericTaskGenError",
    "GenericTaskGenHooks",
    "build_generic_task_subclass_module",
    "generic_task_semantic_key",
    "load_generic_robotwin_task_adapter",
    "validate_generic_task_methods",
]
