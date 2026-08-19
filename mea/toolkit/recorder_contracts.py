"""Task-schema extension contract for generated RoboTwin actors."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .schema import (
    actor_access_path_key,
    resolve_task_actor,
    validate_task_schema,
)


class RecorderError(RuntimeError):
    """Raised when a task cannot satisfy its declared telemetry schema."""


def extend_task_schema_with_generated_actors(
    schema: Mapping[str, Any],
    task: Any,
) -> dict[str, Any]:
    """Append generated public actors and their measurable pose signals."""

    raw = getattr(task, "mea_telemetry_tracked_actors", None)
    if raw is None:
        return deepcopy(dict(schema))
    if not isinstance(raw, (list, tuple)):
        raise RecorderError(
            "mea_telemetry_tracked_actors must be a list or tuple"
        )
    result = deepcopy(dict(schema))
    actors = result.setdefault("tracked_actors", [])
    focus = result.setdefault("contact_focus_actor_ids", [])
    semantic_fields = result.setdefault("semantic_fields", [])
    ids = {item["id"] for item in actors}
    access_keys = {actor_access_path_key(item) for item in actors}
    scene_names = {item["scene_name"] for item in actors}
    field_names = {item["name"] for item in semantic_fields}
    expected = {
        "id",
        "task_attribute",
        "scene_name",
        "functional_points",
        "contact_points",
        "contact_focus",
    }
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise RecorderError(
                f"generated tracked actor {index} has invalid fields"
            )
        # Existing actors are already tracked by the immutable task schema.
        # Some generated scene methods redundantly repeat that declaration
        # while omitting only ``contact_focus``.  Canonicalize exactly that
        # base-schema no-op; never infer fields for a genuinely new or changed
        # actor declaration.
        if set(item) == expected - {"contact_focus"}:
            matching_base = [
                actor
                for actor in actors
                if actor.get("id") == item["id"]
                and actor.get("task_attribute") == item["task_attribute"]
                and actor.get("scene_name") == item["scene_name"]
                and actor.get("functional_points", [])
                == item["functional_points"]
                and actor.get("contact_points", [])
                == item["contact_points"]
            ]
            if len(matching_base) == 1:
                item = {
                    **dict(item),
                    "contact_focus": item["id"] in focus,
                }
        if set(item) != expected:
            raise RecorderError(
                f"generated tracked actor {index} has invalid fields"
            )
        actor_id = item["id"]
        attribute = item["task_attribute"]
        scene_name = item["scene_name"]
        if any(
            not isinstance(value, str)
            or not value
            or not value.isidentifier()
            or value.startswith("_")
            for value in (actor_id, attribute)
        ) or not isinstance(scene_name, str) or not scene_name:
            raise RecorderError(
                f"generated tracked actor {index} has invalid identity"
            )
        points: dict[str, list[int]] = {}
        for field in ("functional_points", "contact_points"):
            values = item[field]
            if (
                not isinstance(values, (list, tuple))
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in values
                )
                or len(values) != len(set(values))
            ):
                raise RecorderError(
                    f"generated tracked actor {index} has invalid {field}"
                )
            points[field] = list(values)
        if not isinstance(item["contact_focus"], bool):
            raise RecorderError(
                f"generated tracked actor {index} contact_focus must be bool"
            )
        if not hasattr(task, attribute):
            raise RecorderError(
                f"generated tracked actor attribute is missing: {attribute}"
            )
        if actor_id in ids:
            existing = next(
                actor for actor in actors if actor["id"] == actor_id
            )
            exact_redeclaration = bool(
                existing.get("task_attribute") == attribute
                and existing["scene_name"] == scene_name
                and list(existing.get("functional_points", []))
                == points["functional_points"]
                and list(existing.get("contact_points", []))
                == points["contact_points"]
                and (actor_id in focus) is item["contact_focus"]
            )
            if exact_redeclaration:
                # A generated subclass may replace the official actor instance
                # while keeping its public identity.  The base schema already
                # records that attribute, so the declaration is an idempotent
                # no-op rather than a new actor.
                continue
            raise RecorderError(
                f"generated tracked actor {index} duplicates the base schema"
            )
        runtime_actor = getattr(task, attribute)
        get_runtime_name = getattr(runtime_actor, "get_name", None)
        runtime_name = (
            get_runtime_name() if callable(get_runtime_name) else None
        )
        if runtime_name != scene_name:
            raise RecorderError(
                f"generated tracked actor {index} scene_name {scene_name!r} "
                f"does not match runtime actor name {runtime_name!r}; give "
                "the new actor a unique name and declare that exact name"
            )
        generated_access_key = actor_access_path_key(
            {"task_attribute": attribute}
        )
        if generated_access_key in access_keys or scene_name in scene_names:
            raise RecorderError(
                f"generated tracked actor {index} duplicates the base schema"
            )
        actors.append(
            {
                "id": actor_id,
                "task_attribute": attribute,
                "scene_name": scene_name,
                **points,
            }
        )
        new_fields = [
            {
                "name": f"{actor_id}_position",
                "source": "actor_position",
                "actor_id": actor_id,
            },
            *[
                {
                    "name": f"{actor_id}_functional_{point_id}_position",
                    "source": "actor_functional_position",
                    "actor_id": actor_id,
                    "point_id": point_id,
                }
                for point_id in points["functional_points"]
            ],
            *[
                {
                    "name": f"{actor_id}_contact_{point_id}_position",
                    "source": "actor_contact_position",
                    "actor_id": actor_id,
                    "point_id": point_id,
                }
                for point_id in points["contact_points"]
            ],
        ]
        duplicates = sorted(
            field["name"]
            for field in new_fields
            if field["name"] in field_names
        )
        if duplicates:
            raise RecorderError(
                "generated tracked actor duplicates semantic fields: "
                + ", ".join(duplicates)
            )
        semantic_fields.extend(new_fields)
        field_names.update(field["name"] for field in new_fields)
        if item["contact_focus"]:
            focus.append(actor_id)
        ids.add(actor_id)
        access_keys.add(generated_access_key)
        scene_names.add(scene_name)
    # Validate the final schema, not only newly appended declarations.  A
    # generated subclass may redeclare an official actor or may add an
    # untracked actor with the same SAPIEN name; both cases would otherwise
    # make contact telemetry ambiguous.
    for actor_spec in actors:
        try:
            runtime_actor = resolve_task_actor(task, actor_spec)
        except (AttributeError, IndexError, KeyError, TypeError):
            runtime_actor = None
        get_runtime_name = getattr(runtime_actor, "get_name", None)
        runtime_name = (
            get_runtime_name() if callable(get_runtime_name) else None
        )
        if runtime_name != actor_spec["scene_name"]:
            raise RecorderError(
                f"tracked actor {actor_spec['id']!r} scene_name "
                f"{actor_spec['scene_name']!r} does not match runtime actor "
                f"name {runtime_name!r}"
            )
    scene = getattr(task, "scene", None)
    get_all_actors = getattr(scene, "get_all_actors", None)
    if callable(get_all_actors):
        runtime_name_counts: dict[str, int] = {}
        scene_objects = list(get_all_actors() or [])
        get_all_articulations = getattr(scene, "get_all_articulations", None)
        if callable(get_all_articulations):
            scene_objects.extend(list(get_all_articulations() or []))
        seen_scene_identity: set[int] = set()
        for runtime_actor in scene_objects:
            if id(runtime_actor) in seen_scene_identity:
                continue
            seen_scene_identity.add(id(runtime_actor))
            get_runtime_name = getattr(runtime_actor, "get_name", None)
            runtime_name = (
                get_runtime_name() if callable(get_runtime_name) else None
            )
            if isinstance(runtime_name, str) and runtime_name:
                runtime_name_counts[runtime_name] = (
                    runtime_name_counts.get(runtime_name, 0) + 1
                )
        declared_name_counts: dict[str, int] = {}
        for actor_spec in actors:
            scene_name = actor_spec["scene_name"]
            declared_name_counts[scene_name] = (
                declared_name_counts.get(scene_name, 0) + 1
            )
        ambiguous = sorted(
            scene_name
            for scene_name, expected_count in declared_name_counts.items()
            if runtime_name_counts.get(scene_name, 0) != expected_count
        )
        if ambiguous:
            if all(
                declared_name_counts[name] == 1 for name in ambiguous
            ):
                raise RecorderError(
                    "tracked actor runtime names must occur exactly once in "
                    "the scene: " + ", ".join(ambiguous)
                )
            raise RecorderError(
                "tracked actor runtime-name multiplicity differs from the "
                "scene: " + ", ".join(ambiguous)
            )
    try:
        return validate_task_schema(
            result,
            expected_task_name=str(schema.get("task_name")),
        )
    except Exception as exc:
        raise RecorderError(
            f"generated tracked actor schema is invalid: {exc}"
        ) from exc


__all__ = ["RecorderError", "extend_task_schema_with_generated_actors"]
