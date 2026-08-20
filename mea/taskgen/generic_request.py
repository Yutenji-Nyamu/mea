"""Semantic identity, retrieval context, and provider prompt for TaskGen."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from mea.planner.experiment_candidate import (
    ExperimentCandidateError,
    validate_experiment_candidate,
)
from mea.toolkit.schema import actor_access_expression

from .generic_contracts import (
    GenericRoboTwinTaskAdapter,
    GenericTaskGenError,
    GenericTaskGenHooks,
)
from .provider_scene_checker import retrieve_class_methods


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
            "render_preflight": True,
            "expert_preflight": True,
            "local_repair_limit": 1,
        },
    }


def _evaluation_intent_identity(
    candidate: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Project only the typed intent facts that define exact Task reuse."""

    intent = candidate.get("evaluation_intent")
    if not isinstance(intent, Mapping):
        return None
    return {
        field: deepcopy(intent.get(field))
        for field in (
            "original_concern",
            "hypothesis",
            "requested_change",
            "preserved_conditions",
            "required_observation",
        )
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
    _resolve_repo_file(
        root,
        normalized_adapter["official_source"],
        label="adapter official source",
    )
    for relative in normalized_adapter["asset_paths"]:
        _resolve_repo_file(root, str(relative), label="adapter asset")
    for relative in normalized_adapter["documentation_paths"]:
        _resolve_repo_file(root, str(relative), label="adapter documentation")
    return {
        "schema_version": 2,
        "base_task": normalized_candidate["base_task"],
        "semantic_concern": normalized_candidate["semantic_concern"],
        "scene_need": normalized_candidate["scene_need"],
        "checker_need": normalized_candidate["checker_need"],
        "evaluation_intent": _evaluation_intent_identity(normalized_candidate),
        "adapter_contract": {
            "official_source": normalized_adapter["official_source"],
            "official_class": normalized_adapter["official_class"],
            "task_schema": normalized_adapter["task_schema"],
            "asset_paths": list(normalized_adapter["asset_paths"]),
            "documentation_paths": list(
                normalized_adapter["documentation_paths"]
            ),
            "generation_hook_contract": normalized_adapter[
                "generation_hook_contract"
            ],
        },
    }


def _validate_exact_match(
    value: Mapping[str, Any],
    *,
    semantic_key: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GenericTaskGenError("exact task lookup must return an object or null")
    match = deepcopy(dict(value))
    if match.get("schema_version") != 2 or match.get("status") != "validated":
        raise GenericTaskGenError("exact task match is not validated")
    if match.get("semantic_key") != dict(semantic_key):
        raise GenericTaskGenError("exact task match semantic key differs")
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
    guide_relative = f"mea/knowledge/tasks/{adapter['task_name']}.md"
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
        if str(relative) == guide_relative:
            label = "BOUND TASK IMPLEMENTATION GUIDE"
        elif str(relative).startswith("description/task_instruction/"):
            label = (
                "LANGUAGE PARAPHRASE DATA (language wording only; not "
                "implementation authority)"
            )
        else:
            label = "RETRIEVED DOCUMENTATION"
        sections.append(f"{label} `{relative}`:\n{content.strip()}")
    asset_descriptors: list[dict[str, Any]] = []
    for relative in adapter["asset_paths"]:
        path = _resolve_repo_file(
            repo_root, str(relative), label="adapter asset"
        )
        asset_descriptors.append(
            {
                "path": str(relative),
                "size_bytes": path.stat().st_size,
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
    actor_expressions: dict[str, str] = {}
    if isinstance(tracked, list):
        for item in tracked:
            if not isinstance(item, Mapping):
                continue
            actor_id = item.get("id")
            if not isinstance(actor_id, str):
                continue
            try:
                actor_expressions[actor_id] = actor_access_expression(item)
            except (KeyError, TypeError, ValueError):
                continue
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
            actor_expression = actor_expressions.get(str(actor_id))
            if actor_expression is None:
                continue
            if source == "actor_position":
                expression = f"{actor_expression}.get_pose().p"
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
                    f'{actor_expression}.{method}({point_id}, "pose").p'
                )
        elif source == "robot_tcp_position":
            side = field.get("side")
            if side in {"left", "right"}:
                expression = f"self.robot.get_{side}_tcp_pose()[:3]"
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
        "identity with an arbitrary PhysX collision point. Semantic field "
        "names describe evidence; they are not necessarily Python attributes. "
        "For example, do not rewrite `stapler_position` as an assumed "
        "`self.stapler` access unless that exact expression is listed above."
    )


def _core_prompt(
    candidate: Mapping[str, Any],
    adapter: Mapping[str, Any],
    *,
    prompt_constraints: str,
) -> str:
    """Render only request-specific data and the stable transport contract.

    Cross-task semantics live in README.Agent; task-local facts arrive through
    the bound implementation guide and a repair receives its previous output
    plus the concrete error. Keeping those rules out of this function prevents
    a second drifting copy of the TaskGen prompt.
    """
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
        "Return one strict JSON object with exactly string fields "
        "load_actors and check_success. Each non-empty field is one complete "
        "Python method whose only parameter is self. A non-null scene_need "
        "requires a changed load_actors; a non-null checker_need requires a "
        "changed check_success. For a null need return an empty string: the "
        "runtime injects the exact official method before validation. "
        "Implement the Proposal literally using only retrieved APIs and "
        "current simulator state. Do not change the policy/controller, weaken "
        "a relation into a proxy, move trajectory metrics into check_success, "
        "or return prose/Markdown. The README.Agent contract supplies the "
        "shared cross-task rules; the bound task guide supplies local facts."
    )


__all__ = ["generic_task_semantic_key"]
