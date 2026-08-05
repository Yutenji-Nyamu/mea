"""AST, compile, and official-method validation for generic TaskGen."""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .generic_contracts import GenericTaskGenError
from .provider_scene_checker import (
    retrieve_class_methods,
    text_sha256,
    validate_method_ast,
)

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
    "get_left_tcp_pose",
    "get_links",
    "get_pose",
    "get_right_tcp_pose",
    "mea_official_check_success",
}


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


def _official_core_aliases(function: ast.FunctionDef) -> dict[str, int]:
    """Return single-assignment local aliases of the untouched core call."""

    store_counts: dict[str, int] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            store_counts[node.id] = store_counts.get(node.id, 0) + 1
    aliases: dict[str, int] = {}
    for statement in function.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and _is_direct_official_core_call(statement.value)
            and store_counts.get(statement.targets[0].id) == 1
        ):
            aliases[statement.targets[0].id] = statement.lineno
    return aliases


def _is_official_core_conjunct(
    node: ast.AST,
    aliases: Mapping[str, int],
    *,
    return_lineno: int,
) -> bool:
    return _is_direct_official_core_call(node) or (
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and aliases.get(node.id, return_lineno) < return_lineno
    )


def _unwrap_boolean_cast(node: ast.AST | None) -> ast.AST | None:
    """Return the value of a transparent ``bool(value)`` wrapper."""

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bool"
        and len(node.args) == 1
        and not node.keywords
    ):
        return node.args[0]
    return node


def _binds_identifier(tree: ast.AST, identifier: str) -> bool:
    """Conservatively detect a local binding that can shadow a builtin."""

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == identifier
        ):
            return True
        if isinstance(node, ast.arg) and node.arg == identifier:
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == identifier:
                return True
        if isinstance(node, ast.ExceptHandler) and node.name == identifier:
            return True
    return False


def _checker_enforces_official_core_conjunct(
    checker: ast.Module,
) -> bool:
    """Conservatively prove that official failure forces checker failure."""

    function = checker.body[0]
    if not isinstance(function, ast.FunctionDef):
        return False
    aliases = _official_core_aliases(function)
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
        and len(first.body) == 1
        and isinstance(first.body[0], ast.Return)
        and isinstance(first.body[0].value, ast.Constant)
        and first.body[0].value.value is False
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
    if not non_false_returns:
        return False
    bool_is_builtin = not _binds_identifier(function, "bool")
    for node in non_false_returns:
        value = (
            _unwrap_boolean_cast(node.value)
            if bool_is_builtin
            else node.value
        )
        if not (
            isinstance(value, ast.BoolOp)
            and isinstance(value.op, ast.And)
            and any(
                _is_official_core_conjunct(
                    conjunct,
                    aliases,
                    return_lineno=node.lineno,
                )
                for conjunct in value.values
            )
        ):
            return False
    return True


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


__all__ = [
    "build_generic_task_subclass_module",
    "validate_generic_task_methods",
]

