"""Catalog-independent RoboTwin TaskGen orchestration.

The backend consumes one runtime ``ExperimentCandidate`` and a thin task
adapter.  The adapter describes the official task program and exposes the
small simulator-specific hooks needed to validate generated code.  It does
not enumerate aspects, variants, metrics, or planner routes.

Task reuse is exact and semantic. A miss invokes the provider with at most one
local regeneration after static, fixture, render, or expert diagnosis. Policy
execution remains outside this module.
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
from mea.planner.proposal_execution import (
    ProposalExecutionError,
    validate_taskgen_candidate_execution,
)
from mea.planner.semantic_coverage import (
    SemanticCoverageError,
    build_implementation_trace,
)
from mea.robotwin_task_context import (
    RoboTwinTaskContextError,
    resolve_robotwin_task_context,
)

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
from .semantic_review import (
    CheckerSemanticReviewError,
    checker_review_identity,
    review_generated_checker,
    validate_checker_semantic_review_binding,
)


class GenericTaskGenError(RuntimeError):
    """Raised when a dynamic candidate cannot be reused or generated."""

    def __init__(
        self,
        message: str,
        *,
        runtime: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.runtime = dict(runtime or {})


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
_GENERIC_READ_ONLY_METHOD_CALLS = {
    "get_contact_point",
    "get_contacts",
    "get_functional_point",
    "get_links",
    "get_pose",
    "mea_official_check_success",
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
    task_context: Mapping[str, Any] | None = None


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


def _literal_number(node: ast.AST) -> float | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
    ):
        value = _literal_number(node.operand)
        if value is not None:
            return value if isinstance(node.op, ast.UAdd) else -value
    return None


def _requested_scale_multiplier(scene_need: Any) -> float | None:
    if isinstance(scene_need, Mapping):
        scene_need = scene_need.get("description")
    if not isinstance(scene_need, str):
        return None
    text = scene_need.casefold()
    if not re.search(
        r"\b(?:scale|size|diameter|radius|width|height|"
        r"larger|smaller|enlarge|shrink)\b",
        text,
    ):
        return None
    percent = r"([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)"
    patterns = (
        (
            rf"\b(?:increase|enlarge|grow|scale\s+up)\b"
            rf"[^.;\n]{{0,80}}\bby\s+{percent}",
            lambda value: 1.0 + value / 100.0,
        ),
        (
            rf"\b(?:increase|enlarge|grow|scale\s+up)\b"
            rf"[^.;\n]{{0,80}}\bto\s+{percent}",
            lambda value: value / 100.0,
        ),
        (
            rf"\b(?:reduce|decrease|shrink|scale\s+down)\b"
            rf"[^.;\n]{{0,80}}\bby\s+{percent}",
            lambda value: 1.0 - value / 100.0,
        ),
        (
            rf"\b(?:reduce|decrease|shrink|scale\s+down)\b"
            rf"[^.;\n]{{0,80}}\bto\s+{percent}",
            lambda value: value / 100.0,
        ),
        (
            rf"{percent}\s+(?:larger|bigger)",
            lambda value: 1.0 + value / 100.0,
        ),
        (
            rf"{percent}\s+smaller",
            lambda value: 1.0 - value / 100.0,
        ),
    )
    for pattern, convert in patterns:
        match = re.search(pattern, text)
        if match:
            expected = convert(float(match.group(1)))
            return expected if expected > 0.0 else None
    return None


def _validate_preservation_feasibility(
    candidate: Mapping[str, Any],
) -> None:
    """Reject one impossible scale contract before retrieval or generation."""

    multiplier = _requested_scale_multiplier(candidate.get("scene_need"))
    if multiplier is None or abs(multiplier - 1.0) <= 1e-9:
        return
    scene_need = candidate.get("scene_need")
    scene_description = (
        str(scene_need.get("description") or "")
        if isinstance(scene_need, Mapping)
        else str(scene_need or "")
    )
    if re.search(
        r"\bcustom[\s-]+pivot(?:\s+capability)?\b",
        scene_description,
        re.IGNORECASE,
    ):
        return
    intent = candidate.get("evaluation_intent")
    conditions = (
        intent.get("preserved_conditions")
        if isinstance(intent, Mapping)
        else None
    )
    if not isinstance(conditions, list):
        return
    normalized = [
        str(condition).casefold()
        for condition in conditions
        if isinstance(condition, str)
    ]
    preserves_contact_world_position = any(
        re.search(r"\bcontact[\s-]+point\b", condition)
        for condition in normalized
    )
    preserves_center_or_origin_position = any(
        re.search(
            r"\b(?:center|centre|origin)(?:[\s-]+world)?"
            r"[\s-]+(?:position|location|coordinate)s?\b",
            condition,
        )
        or re.search(
            r"\b(?:actor|object|target)(?:[\s-]+center)?"
            r"(?:[\s-]+world)?[\s-]+position\b",
            condition,
        )
        for condition in normalized
    )
    if (
        preserves_contact_world_position
        and preserves_center_or_origin_position
    ):
        raise GenericTaskGenError(
            "preservation feasibility conflict: the current "
            "origin-centered uniform scale backend cannot guarantee both "
            "contact-point world position and actor/object center/origin "
            "position; Planner must revise the candidate to preserve one "
            "condition or declare a custom pivot capability"
        )


def _validate_literal_scale_alignment(
    load_actors: ast.AST,
    *,
    scene_need: Any,
) -> None:
    """Reject only explicit, literal scale changes that contradict the need."""

    expected = _requested_scale_multiplier(scene_need)
    if expected is None:
        return
    observed: list[float] = []
    for node in ast.walk(load_actors):
        if not isinstance(node, ast.Call):
            continue
        parts = _attribute_parts(node.func)
        if parts is None or parts[-1] != "create_actor":
            continue
        for keyword in node.keywords:
            if keyword.arg == "scale_multiplier":
                value = _literal_number(keyword.value)
                if value is not None:
                    observed.append(value)
    if not observed or any(abs(value - expected) <= 1e-9 for value in observed):
        return
    raise GenericTaskGenError(
        "literal create_actor scale_multiplier contradicts scene_need: "
        f"expected {expected:g} from the requested direction/magnitude, "
        f"observed {observed}"
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
    safe_method_calls: set[str] = set(_GENERIC_READ_ONLY_METHOD_CALLS)
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


def _is_direct_official_core_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "mea_official_check_success"
        and not node.args
        and not node.keywords
    )


def _checker_enforces_official_core_conjunct(
    checker: ast.Module,
) -> bool:
    """Conservatively prove that official failure forces checker failure."""

    function = checker.body[0]
    if not isinstance(function, ast.FunctionDef):
        return False
    statements = list(function.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements = statements[1:]
    first = statements[0] if statements else None
    if (
        isinstance(first, ast.If)
        and isinstance(first.test, ast.UnaryOp)
        and isinstance(first.test.op, ast.Not)
        and _is_direct_official_core_call(first.test.operand)
        and any(
            isinstance(child, ast.Return)
            and isinstance(child.value, ast.Constant)
            and child.value.value is False
            for child in first.body
        )
    ):
        return True
    non_false_returns = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and not (
            isinstance(node.value, ast.Constant)
            and node.value.value is False
        )
    ]
    return bool(non_false_returns) and all(
        isinstance(node.value, ast.BoolOp)
        and isinstance(node.value.op, ast.And)
        and any(
            _is_direct_official_core_call(value)
            for value in node.value.values
        )
        for node in non_false_returns
    )


def validate_generic_task_methods(
    methods: Mapping[str, str],
    *,
    official_source: str | Path,
    official_class: str,
    required_method_changes: Mapping[str, bool] | None = None,
    scene_need: Any = None,
    require_official_core_conjunct: bool = False,
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
    official_core_directly_called = any(
        _is_direct_official_core_call(node)
        for node in ast.walk(parsed["check_success"])
    )
    official_core_enforced_as_conjunct = (
        _checker_enforces_official_core_conjunct(
            parsed["check_success"]
        )
    )
    if (
        require_official_core_conjunct
        and not official_core_enforced_as_conjunct
    ):
        raise GenericTaskGenError(
            "generated checker must enforce "
            "self.mea_official_check_success() as a required boolean "
            "conjunct when the Proposal preserves or composes the official "
            "task goal"
        )
    _reject_pose_property_item_assignment(parsed)
    _validate_literal_scale_alignment(
        parsed["load_actors"], scene_need=scene_need
    )
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
        name: ast.dump(parsed[name].body[0], include_attributes=False)
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
        "official_core_conjunct_required": bool(
            require_official_core_conjunct
        ),
        "official_core_directly_called": official_core_directly_called,
        "official_core_enforced_as_conjunct": (
            official_core_enforced_as_conjunct
        ),
    }


def _official_task_methods(
    source_path: Path,
    *,
    class_name: str,
) -> dict[str, str]:
    """Load the two official methods used for partial TaskGen reuse."""

    return {
        name: retrieve_class_methods(
            source_path,
            class_name=class_name,
            method_names=(name,),
            error_type=GenericTaskGenError,
        )
        for name in ("load_actors", "check_success")
    }


def build_generic_task_subclass_module(
    methods: Mapping[str, str],
    *,
    official_module: str,
    official_class: str,
    emit_overrides: Mapping[str, bool] | None = None,
) -> str:
    """Build a thin subclass and override only the requested task methods."""

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
    overrides = (
        {"load_actors": True, "check_success": True}
        if emit_overrides is None
        else dict(emit_overrides)
    )
    if (
        set(overrides) != {"load_actors", "check_success"}
        or any(not isinstance(value, bool) for value in overrides.values())
    ):
        raise GenericTaskGenError(
            "emit_overrides must map load_actors/check_success to bool"
        )
    method_blocks = [
        textwrap.indent(
            textwrap.dedent(str(methods[name])).strip(), "    "
        )
        for name in ("load_actors", "check_success")
        if overrides[name]
    ]
    generated_methods = (
        "\n\n".join(method_blocks) + "\n\n" if method_blocks else ""
    )
    source = (
        '"""Provider-generated RoboTwin task candidate."""\n\n'
        f"import {official_module} as _official_task_module\n"
        f"from {official_module} import *\n\n\n"
        f"class {official_class}("
        f"_official_task_module.{official_class}):\n"
        f"{generated_methods}"
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
    task_schema: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    model_names = _model_names_from_source(
        source_path, class_name=class_name
    )
    for actor in (
        task_schema.get("tracked_actors", [])
        if isinstance(task_schema, Mapping)
        else []
    ):
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


def discover_generic_robotwin_task_identity(
    repo_root: str | Path,
    task_name: str,
    *,
    runtime_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Discover one executable RoboTwin base without a task-name registry.

    This hook-free identity is the authority boundary shared by routing and
    TaskGen.  Official source makes the task discoverable.  A reviewed
    TaskSchema accelerates TaskGen; when absent, one fresh runtime probe may
    provide the actor/telemetry authority instead.
    """

    if not isinstance(task_name, str) or not _TASK_NAME.fullmatch(task_name):
        raise GenericTaskGenError("task_name is not a RoboTwin identifier")
    root = Path(repo_root).expanduser().resolve()
    relative_source = f"envs/{task_name}.py"
    source_path = _resolve_repo_file(
        root, relative_source, label="official task source"
    )
    _official_class(source_path, class_name=task_name)
    try:
        task_context = resolve_robotwin_task_context(
            root,
            task_name,
            runtime_probe=runtime_probe,
        )
    except RoboTwinTaskContextError as exc:
        raise GenericTaskGenError(str(exc)) from exc
    schema = task_context.task_schema
    return {
        "schema_version": 1,
        "task_name": task_name,
        "official_source": relative_source,
        "official_class": task_name,
        "task_schema": deepcopy(schema) if schema is not None else None,
        "task_context": task_context.to_dict(),
        "documentation_paths": list(
            _discover_task_documents(root, task_name=task_name)
        ),
        "asset_paths": list(
            _discover_asset_descriptions(
                root,
                source_path=source_path,
                class_name=task_name,
                task_schema=schema,
            )
        ),
    }


def _candidate_requires_official_core_conjunct(
    candidate: Mapping[str, Any],
) -> bool:
    """Recognize a Proposal that retains the official task goal.

    The generated checker may add an experimental condition, but it must not
    copy the official predicate.  A direct call to the runtime-provided
    untouched method is the only supported composition boundary.
    """

    checker_need = candidate.get("checker_need")
    if not isinstance(checker_need, Mapping):
        return False
    fragments = [str(checker_need.get("description") or "")]
    intent = candidate.get("evaluation_intent")
    if isinstance(intent, Mapping):
        preserved = intent.get("preserved_conditions")
        if isinstance(preserved, list):
            fragments.extend(str(item) for item in preserved)
    text = " ".join(fragments).casefold()
    return any(
        marker in text
        for marker in (
            "official core predicate",
            "official task goal",
            "official goal",
            "official success",
            "official check_success",
            "untouched official",
            "官方任务目标",
            "官方目标",
            "官方成功",
            "官方 check_success",
        )
    )


def load_generic_robotwin_task_adapter(
    repo_root: str | Path,
    task_name: str,
    *,
    checker_fixtures: CheckerFixtureValidator,
    preflight_candidate: PreflightCandidate,
    resolve_metric: ResolveMetric,
    resolve_checker_contract: ResolveCheckerContract,
    prompt_constraints: str = "",
    runtime_probe: Mapping[str, Any] | None = None,
) -> GenericRoboTwinTaskAdapter:
    """Discover a thin adapter for any source/context-backed RoboTwin task.

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
    identity = discover_generic_robotwin_task_identity(
        root,
        task_name,
        runtime_probe=runtime_probe,
    )
    relative_source = str(identity["official_source"])
    source_path = _resolve_repo_file(
        root, relative_source, label="official task source"
    )
    raw_schema = identity["task_schema"]
    if not isinstance(raw_schema, Mapping):
        raise GenericTaskGenError(
            "TaskContext has no simulator-authoritative actor/telemetry "
            "schema; run one official reset context probe before TaskGen"
        )
    schema = deepcopy(dict(raw_schema))
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
            scene_need=candidate.get("scene_need"),
            required_method_changes={
                "load_actors": candidate.get("scene_need") is not None,
                "check_success": candidate.get("checker_need") is not None,
            },
            require_official_core_conjunct=(
                _candidate_requires_official_core_conjunct(candidate)
            ),
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
        build_module=lambda methods, candidate: (
            build_generic_task_subclass_module(
                methods,
                official_module=module_name,
                official_class=task_name,
                emit_overrides={
                    "load_actors": (
                        candidate.get("scene_need") is not None
                    ),
                    "check_success": (
                        candidate.get("checker_need") is not None
                    ),
                },
            )
        ),
        preflight_candidate=preflight_candidate,
        resolve_metric=resolve_metric,
        resolve_checker_contract=resolve_checker_contract,
        prompt_constraints=prompt_constraints,
    )
    return GenericRoboTwinTaskAdapter(
        task_name=str(identity["task_name"]),
        official_source=relative_source,
        official_class=str(identity["official_class"]),
        task_schema=schema,
        documentation_paths=tuple(identity["documentation_paths"]),
        asset_paths=tuple(identity["asset_paths"]),
        hooks=hooks,
        task_context=deepcopy(identity["task_context"]),
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
        "task_context": (
            deepcopy(dict(adapter.task_context))
            if isinstance(adapter.task_context, Mapping)
            else {
                "schema_version": 1,
                "task_name": adapter.task_name,
                "taskgen_ready": True,
                "schema_origin": "legacy_adapter_contract",
            }
        ),
        "documentation_paths": list(documentation_paths),
        "asset_paths": list(asset_paths),
        "generation_hook_contract": {
            "methods": ["load_actors", "check_success"],
            "static_and_fixture_validation": True,
            "semantic_validation": "task_schema_contract_v2",
            "checker_semantic_review": "taskgen_checker_semantic_review_v1",
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
    review_identity = checker_review_identity(normalized_candidate)
    return {
        "schema_version": 1,
        "base_task": normalized_candidate["base_task"],
        "semantic_concern": normalized_candidate["semantic_concern"],
        "scene_need": normalized_candidate["scene_need"],
        "checker_need": normalized_candidate["checker_need"],
        "evaluation_intent": review_identity["evaluation_intent"],
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
            optional_method_names=("play_once",),
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
        "TASK CONTEXT AUTHORITY:\n"
        + json.dumps(
            adapter["task_context"],
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


def _semantic_field_access_guide(adapter: Mapping[str, Any]) -> str:
    """Render exact read-only expressions for TaskSchema semantic fields."""

    schema = adapter.get("task_schema")
    if not isinstance(schema, Mapping):
        return ""
    tracked = schema.get("tracked_actors")
    actor_attributes: dict[str, str] = {}
    if isinstance(tracked, list):
        for item in tracked:
            if not isinstance(item, Mapping):
                continue
            actor_id = item.get("id")
            task_attribute = item.get("task_attribute")
            if isinstance(actor_id, str) and isinstance(task_attribute, str):
                actor_attributes[actor_id] = task_attribute
    fields = schema.get("semantic_fields")
    if not isinstance(fields, list):
        return ""
    expressions: list[str] = []
    for field in fields:
        if not isinstance(field, Mapping):
            continue
        name = field.get("name")
        source = field.get("source")
        if not isinstance(name, str) or not isinstance(source, str):
            continue
        expression: str | None = None
        if source in {
            "actor_position",
            "actor_functional_position",
            "actor_contact_position",
        }:
            actor_id = field.get("actor_id")
            task_attribute = actor_attributes.get(str(actor_id))
            if task_attribute is None:
                continue
            if source == "actor_position":
                expression = f"self.{task_attribute}.get_pose().p"
            else:
                point_id = field.get("point_id")
                if isinstance(point_id, bool) or not isinstance(point_id, int):
                    continue
                method = (
                    "get_functional_point"
                    if source == "actor_functional_position"
                    else "get_contact_point"
                )
                expression = (
                    f'self.{task_attribute}.{method}({point_id}, "pose").p'
                )
        elif source == "robot_tcp_position":
            side = field.get("side")
            if side in {"left", "right"}:
                expression = f"self.robot.{side}_tcp.get_pose().p"
        if expression is not None:
            expressions.append(f"- {name}: `{expression}`")
    if not expressions:
        return ""
    return (
        "\n\nREAD-ONLY CURRENT-STATE FIELD ACCESS:\n"
        + "\n".join(expressions)
        + "\nWhen checker_need names one of these semantic fields, use its "
        "exact expression. Do not invent a similarly named helper such as "
        "get_contact_position, and do not replace a declared actor point "
        "identity with an arbitrary PhysX collision point."
    )


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
        + _semantic_field_access_guide(adapter)
        + constraint_section
        + "\n\nOUTPUT CONTRACT:\n"
        "Return one strict JSON object with exactly two string fields, "
        "load_actors and check_success. Each field must contain one complete "
        "Python method with only self when its corresponding need is non-null. "
        "A non-null scene_need requires a changed load_actors method. A "
        "non-null checker_need requires a changed check_success method. Both "
        "JSON fields remain required for transport, but when a need is null "
        "return an empty string for that field: the runtime ignores that text "
        "and injects the exact official method before AST, fixture, render, "
        "and expert validation. "
        "A changed load_actors method must directly implement the requested "
        "scene change. Comments or an unrelated actor/pose change are not "
        "implementation evidence. load_actors cannot alter policy weights, "
        "controller or gripper precision, action noise, latency, or inference. "
        "Those require an explicit runtime intervention and must not be "
        "simulated by relabelling a scene change. For a pose change, reuse the "
        "official pose construction and alter only the Proposal-named "
        "component. When the candidate leaves the perturbation magnitude open, "
        "derive the smallest measurable change from the retrieved spawn or "
        "workspace range, stay away from its boundary, and keep every "
        "task-critical actor fully inside the unchanged camera view. "
        "Actors already present in "
        "the TASK "
        "TELEMETRY/EXECUTION SCHEMA are tracked automatically even when their "
        "pose or instance is replaced. Do not assign "
        "self.mea_telemetry_tracked_actors merely to repeat one of those base "
        "actors. Do not add helper state beyond self assignments already "
        "present in the official method and new actor handles/telemetry; in "
        "particular, do not cache initial poses, heights, thresholds, or flags "
        "on self. Compute checker values from current simulator state and "
        "literal or Query-specified thresholds. "
        "check_success cannot read the completed trajectory or invoke a "
        "derived Rule metric such as trajectory deviation, smoothness, jerk, "
        "path length, or minimum clearance. Leave that scalar observation to "
        "ToolGen and never invent calculate_* or measure_* helper methods. "
        "Implement every checker_need relation literally. Do not replace an "
        "exact relation with a correlated proxy: a closed gripper is not "
        "target contact, height is not placement, and sequential contacts "
        "are not simultaneous contacts. If the requested predicate is not "
        "available from current simulator state or is false in the supplied "
        "expert terminal fixture, let validation reject the candidate rather "
        "than weakening its meaning. "
        "When checker_need composes the official task goal with an additional "
        "experimental condition, call self.mea_official_check_success() "
        "directly and use its result as a required conjunct; do not copy or "
        "reimplement the official predicate. This preserves the official core "
        "without claiming that the extended checker is official-equivalent. "
        "For a simulator-verifiable robot-contact condition, inspect "
        "self.scene.get_contacts(). A SAPIEN PhysxContact exposes bodies, not "
        "actor0/actor1; each body.entity is the scene entity, while a "
        "RoboTwin Actor wrapper exposes its scene entity as .actor. The "
        "RoboTwin Robot wrapper has no get_links() method; when robot link "
        "entities are needed, combine "
        "self.robot.left_entity.get_links() and "
        "self.robot.right_entity.get_links(). When the Proposal specifically "
        "requires left/right gripper contact, use "
        "tuple(item[0].child_link for item in self.robot.left_gripper) and "
        "the corresponding right_gripper expression; all arm links are not "
        "equivalent to gripper links; do not invent a helper such as "
        "self.check_contact unless that exact method appears in the retrieved "
        "official source. Build checker-local entity collections with `+` or "
        "tuple literals; do not call `.append()` or `.extend()`. These APIs "
        "are read-only and must not mutate simulator state. "
        "self.mea_telemetry_tracked_actors is the metadata exception. Assign "
        "it only when adding an entirely new actor, include "
        "only new actors, and give every entry exactly id, task_attribute, "
        "scene_name, functional_points, contact_points, and a boolean "
        "contact_focus. When adding a distractor or obstacle, inspect the "
        "retrieved asset scale/collision geometry and place it initially "
        "disjoint from the target and the official expert contact path. A "
        "small center offset is not sufficient when reused asset extents "
        "overlap. Do not return Markdown, a template id, or an "
        "explanation. When the retrieved API supports scale_multiplier, it "
        "is the final-size/original-size ratio: increasing size by 50% uses "
        "1.5, while reducing size by 50% (or to 50%) uses 0.5."
    )


def _normalize_validation(
    value: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
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
    scene_requested = candidate.get("scene_need") is not None
    expected_scene_state = "changed" if scene_requested else "preserved"
    scene_report = preflight.get("scene_change")
    if isinstance(scene_report, Mapping):
        scene_alignment_passed = bool(
            scene_report.get("passed") is True
            and scene_report.get("expected_state") == expected_scene_state
        )
        scene_alignment_authority = str(
            scene_report.get("authority") or "simulator_scene_comparison"
        )
    elif scene_requested:
        scene_alignment_passed = (
            preflight.get("scene_change_passed") is True
        )
        scene_alignment_authority = "legacy_scene_change_gate"
    else:
        method_provenance = report.get("method_provenance")
        scene_alignment_passed = bool(
            isinstance(method_provenance, Mapping)
            and method_provenance.get("load_actors")
            == "official_reused"
        )
        scene_alignment_authority = "exact_official_load_actors_reuse"
    if not scene_alignment_passed:
        raise GenericTaskGenError(
            "preflight did not verify the expected scene state "
            f"{expected_scene_state!r} relative to the official control"
        )
    report["scene_sha256"] = text_sha256(methods["load_actors"])
    report["success_sha256"] = text_sha256(methods["check_success"])
    if candidate.get("checker_need") is not None:
        try:
            report["checker_semantic_review"] = (
                validate_checker_semantic_review_binding(
                    report.get("checker_semantic_review"),
                    candidate=candidate,
                    checker_sha256=report["success_sha256"],
                )
            )
        except CheckerSemanticReviewError as exc:
            raise GenericTaskGenError(str(exc)) from exc
        report["checker_semantic_review_required"] = True
    else:
        if report.get("checker_semantic_review") is not None:
            raise GenericTaskGenError(
                "official checker reuse must not carry a generated-checker "
                "semantic review"
            )
        report["checker_semantic_review"] = None
        report["checker_semantic_review_required"] = False
    report["checker_fixture_count"] = len(fixtures)
    report["preflight"] = deepcopy(dict(preflight))
    report["scene_alignment"] = {
        "passed": True,
        "expected_state": expected_scene_state,
        "authority": scene_alignment_authority,
    }
    report["model_written_python"] = True
    report["restricted_success_spec_compiler_used"] = False
    try:
        implementation_trace = build_implementation_trace(
            candidate,
            taskgen_validation=report,
        )
    except SemanticCoverageError as exc:
        raise GenericTaskGenError(
            f"invalid semantic implementation trace: {exc}"
        ) from exc
    if implementation_trace is not None:
        report["implementation_trace"] = implementation_trace
        if implementation_trace["repair_required"]:
            raise GenericTaskGenError(
                "generated TaskGen artifact does not implement the direct "
                "EvaluationIntent; regenerate once or explicitly classify "
                "the candidate as diagnostic_proxy/unsupported"
            )
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
        try:
            normalized_candidate = validate_taskgen_candidate_execution(
                normalized_candidate,
                allowed_change_roots=("load_actors", "check_success"),
            )
        except ProposalExecutionError as exc:
            raise GenericTaskGenError(str(exc)) from exc
        _validate_preservation_feasibility(normalized_candidate)
        if (
            normalized_candidate["scene_need"] is None
            and normalized_candidate["checker_need"] is None
        ):
            raise GenericTaskGenError(
                "generic TaskGen requires a scene or checker need; "
                "Tool-only candidates must bypass TaskGen"
            )
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
            try:
                implementation_trace = build_implementation_trace(
                    normalized_candidate
                )
            except SemanticCoverageError as exc:
                raise GenericTaskGenError(
                    f"invalid semantic implementation trace: {exc}"
                ) from exc
            return {
                "schema_version": 1,
                "status": "reused",
                "route": "exact_generated_task_reuse",
                "candidate": normalized_candidate,
                "semantic_key": semantic_key,
                "semantic_key_sha256": semantic_hash,
                "provider_required": False,
                "provider_call_count": 0,
                "implementation_trace": implementation_trace,
                "exact_match": _validate_exact_match(
                    match,
                    semantic_key=semantic_key,
                    semantic_key_sha256=semantic_hash,
                ),
            }

        rag_context = _read_generation_context(
            self.repo_root, adapter=normalized_adapter
        )
        official_source_path = _resolve_repo_file(
            self.repo_root,
            normalized_adapter["official_source"],
            label="adapter official source",
        )
        official_methods = _official_task_methods(
            official_source_path,
            class_name=normalized_adapter["official_class"],
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
            provider_methods = {
                name: str(methods[name])
                for name in ("load_actors", "check_success")
            }
            method_needs = {
                "load_actors": "scene_need",
                "check_success": "checker_need",
            }
            typed_methods = {
                name: (
                    provider_methods[name]
                    if normalized_candidate[need] is not None
                    else official_methods[name]
                )
                for name, need in method_needs.items()
            }
            method_provenance = {
                name: (
                    "provider_generated"
                    if normalized_candidate[need] is not None
                    else "official_reused"
                )
                for name, need in method_needs.items()
            }
            official_reused_methods = [
                name
                for name in ("load_actors", "check_success")
                if method_provenance[name] == "official_reused"
            ]
            checker_semantic_review = None
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
                if normalized_candidate["checker_need"] is not None:
                    try:
                        checker_semantic_review = (
                            review_generated_checker(
                                provider=self.provider,
                                model=self.model,
                                candidate=normalized_candidate,
                                task_context=normalized_adapter[
                                    "task_context"
                                ],
                                method_provenance=method_provenance,
                                generated_scene=typed_methods["load_actors"],
                                official_checker=official_methods[
                                    "check_success"
                                ],
                                generated_checker=typed_methods[
                                    "check_success"
                                ],
                                attempt_dir=attempt_dir,
                            )
                        )
                    except CheckerSemanticReviewError as exc:
                        raise GenericTaskGenError(
                            str(exc),
                            runtime={
                                "semantic_review_provider_calls": (
                                    exc.provider_calls
                                )
                            },
                        ) from exc
                preflight = adapter.hooks.preflight_candidate(
                    attempt_dir, module_source, normalized_candidate
                )
                validation = _normalize_validation(
                    {
                        **deepcopy(dict(raw_validation)),
                        "method_provenance": method_provenance,
                        "official_reused_methods": (
                            official_reused_methods
                        ),
                        "checker_semantic_review": (
                            checker_semantic_review
                        ),
                        "semantic_review_provider_calls": (
                            1 if checker_semantic_review is not None else 0
                        ),
                    },
                    candidate=normalized_candidate,
                    methods=typed_methods,
                    preflight=preflight,
                )
            except GenericTaskGenError as exc:
                if checker_semantic_review is not None:
                    runtime = dict(exc.runtime)
                    runtime["semantic_review_provider_calls"] = 1
                    raise GenericTaskGenError(
                        str(exc),
                        runtime=runtime,
                    ) from exc
                raise
            except Exception as exc:
                raise GenericTaskGenError(
                    "generic TaskGen validation hook failed: "
                    f"{type(exc).__name__}: {exc}",
                    runtime=(
                        {"semantic_review_provider_calls": 1}
                        if checker_semantic_review is not None
                        else None
                    ),
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
            "implementation_trace": deepcopy(
                generated["validation"].get("implementation_trace")
            ),
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
    "discover_generic_robotwin_task_identity",
    "generic_task_semantic_key",
    "load_generic_robotwin_task_adapter",
    "validate_generic_task_methods",
]
