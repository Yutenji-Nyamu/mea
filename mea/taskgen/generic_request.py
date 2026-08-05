"""Semantic identity, retrieval context, and provider prompt for TaskGen."""

from __future__ import annotations

import hashlib
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
from .generic_validation import _derived_ast_policy
from .provider_scene_checker import retrieve_class_methods, text_sha256
from .semantic_review import checker_review_identity

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
        "component. For a fixed-angle rotation, the AST contract does not "
        "admit np.sin, np.cos, np.deg2rad, or math trigonometry; emit a "
        "normalized quaternion as numeric literals instead. When the "
        "candidate leaves the perturbation magnitude open, "
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


__all__ = ["generic_task_semantic_key"]
