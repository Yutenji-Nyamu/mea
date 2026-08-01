from __future__ import annotations

from pathlib import Path

import pytest

from mea.robotwin_task_context import (
    RoboTwinTaskContextError,
    build_runtime_task_context_probe,
    resolve_robotwin_task_context,
)
from mea.taskgen.generic_backend import (
    GenericTaskGenError,
    _semantic_field_access_guide,
    discover_generic_robotwin_task_identity,
    load_generic_robotwin_task_adapter,
)
from mea.toolkit.schema import resolve_task_actor


class _Actor:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name

    def get_pose(self) -> object:
        return object()


class _Scene:
    def __init__(self, actors: list[_Actor]) -> None:
        self._actors = actors

    def get_all_actors(self) -> list[_Actor]:
        return list(self._actors)

    def get_timestep(self) -> float:
        return 0.004

    def get_contacts(self) -> list[object]:
        return []


class _Robot:
    def get_left_tcp_pose(self) -> list[float]:
        return [0.0, 0.0, 0.0]

    def get_right_tcp_pose(self) -> list[float]:
        return [0.0, 0.0, 0.0]


class runtime_context_task:
    def __init__(self, actor: _Actor) -> None:
        self.target = actor
        self.scene = _Scene([actor])
        self.robot = _Robot()

    def check_success(self) -> bool:
        return False


class nested_context_task:
    def __init__(self, actors: list[_Actor]) -> None:
        self.actor_groups = {"targets": (list(actors),)}
        self.scene = _Scene(actors)
        self.robot = _Robot()

    def check_success(self) -> bool:
        return False


def _write_source_only_task(root: Path) -> None:
    source = root / "envs/runtime_context_task.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class runtime_context_task:\n"
        "    def load_actors(self):\n"
        "        self.target = create_actor(modelname='runtime_target')\n\n"
        "    def check_success(self):\n"
        "        return self.target.is_ready()\n",
        encoding="utf-8",
    )
    readme = root / "mea/taskgen/README.Agent.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("Generate one task candidate.\n", encoding="utf-8")


def _runtime_probe(root: Path) -> dict[str, object]:
    actor = _Actor("runtime_target")
    return build_runtime_task_context_probe(
        runtime_context_task(actor),
        repo_root=root,
        task_name="runtime_context_task",
        action_dimension=14,
    )


def _write_nested_source_task(root: Path) -> None:
    source = root / "envs/nested_context_task.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "class nested_context_task:\n"
        "    def load_actors(self):\n"
        "        self.actor_groups = {'targets': ([create_actor(), "
        "create_actor()],)}\n\n"
        "    def check_success(self):\n"
        "        return all(actor.is_ready() for actor in "
        "self.actor_groups['targets'][0])\n",
        encoding="utf-8",
    )


def test_source_discovery_is_not_taskgen_authority(tmp_path: Path) -> None:
    _write_source_only_task(tmp_path)

    identity = discover_generic_robotwin_task_identity(
        tmp_path,
        "runtime_context_task",
    )

    assert identity["task_schema"] is None
    assert identity["task_context"]["schema_origin"] == "source_only"
    assert identity["task_context"]["taskgen_ready"] is False
    with pytest.raises(
        GenericTaskGenError,
        match="official reset context probe",
    ):
        load_generic_robotwin_task_adapter(
            tmp_path,
            "runtime_context_task",
            checker_fixtures=lambda _methods, _candidate: [],
            preflight_candidate=lambda _path, _source, _candidate: {},
            resolve_metric=lambda _candidate: "metric",
            resolve_checker_contract=lambda _candidate: {},
        )


def test_live_reset_probe_unlocks_generic_taskgen_without_hand_schema(
    tmp_path: Path,
) -> None:
    _write_source_only_task(tmp_path)
    probe = _runtime_probe(tmp_path)

    context = resolve_robotwin_task_context(
        tmp_path,
        "runtime_context_task",
        runtime_probe=probe,
    )
    adapter = load_generic_robotwin_task_adapter(
        tmp_path,
        "runtime_context_task",
        runtime_probe=probe,
        checker_fixtures=lambda _methods, _candidate: [],
        preflight_candidate=lambda _path, _source, _candidate: {},
        resolve_metric=lambda _candidate: "metric",
        resolve_checker_contract=lambda _candidate: {},
    )

    assert context.schema_origin == "runtime_probe"
    assert context.taskgen_ready is True
    assert context.task_schema is not None
    assert context.task_schema["tracked_actors"] == [
        {
            "id": "target",
            "task_attribute": "target",
            "scene_name": "runtime_target",
            "functional_points": [],
            "contact_points": [],
        }
    ]
    assert context.task_schema["semantic_roles"] == {}
    assert context.task_schema["contact_focus_actor_ids"] == ["target"]
    assert [
        field["name"] for field in context.task_schema["semantic_fields"]
    ] == [
        "target_position",
        "left_tcp_position",
        "right_tcp_position",
    ]
    assert context.task_schema["success_contract"]["authority"] == (
        "official_check_success_runtime_callable"
    )
    observables = context.telemetry_observables
    assert observables["simulation_clock"]["available"] is True
    assert observables["policy_action"]["dimension"] == 14
    assert observables["robot_tcp"]["available_sides"] == [
        "left",
        "right",
    ]
    assert observables["contact_events"]["available"] is True
    assert context.task_schema["telemetry_observables"] == observables
    assert adapter.task_schema == context.task_schema
    assert adapter.task_context["schema_origin"] == "runtime_probe"


def test_runtime_probe_cannot_invent_source_actor_or_rebind_source(
    tmp_path: Path,
) -> None:
    _write_source_only_task(tmp_path)
    probe = _runtime_probe(tmp_path)
    forged_actor = {
        **probe,
        "actors": [
            {
                "task_attribute": "imagined_target",
                "scene_name": "runtime_target",
            }
        ],
    }
    with pytest.raises(
        RoboTwinTaskContextError,
        match="lacks source authority",
    ):
        resolve_robotwin_task_context(
            tmp_path,
            "runtime_context_task",
            runtime_probe=forged_actor,
        )

    forged_hash = {**probe, "official_source_sha256": "0" * 64}
    with pytest.raises(
        RoboTwinTaskContextError,
        match="source hash differs",
    ):
        resolve_robotwin_task_context(
            tmp_path,
            "runtime_context_task",
            runtime_probe=forged_hash,
        )


def test_live_probe_discovers_nested_builtin_actor_containers(
    tmp_path: Path,
) -> None:
    _write_nested_source_task(tmp_path)
    actors = [_Actor("repeated_target"), _Actor("repeated_target")]

    probe = build_runtime_task_context_probe(
        nested_context_task(actors),
        repo_root=tmp_path,
        task_name="nested_context_task",
        action_dimension=14,
    )
    context = resolve_robotwin_task_context(
        tmp_path,
        "nested_context_task",
        runtime_probe=probe,
    )

    expected_paths = [
        [
            {"attribute": "actor_groups"},
            {"key": "targets"},
            {"index": 0},
            {"index": index},
        ]
        for index in range(2)
    ]
    assert probe["actors"] == [
        {"access_path": path, "scene_name": "repeated_target"}
        for path in expected_paths
    ]
    assert context.task_schema is not None
    assert [
        actor["access_path"]
        for actor in context.task_schema["tracked_actors"]
    ] == expected_paths
    actor_ids = [
        actor["id"] for actor in context.task_schema["tracked_actors"]
    ]
    assert len(set(actor_ids)) == 2
    assert actor_ids[0].startswith(
        "actor_groups_key_str_targets_index_0_index_0_path_"
    )
    assert actor_ids[1].startswith(
        "actor_groups_key_str_targets_index_0_index_1_path_"
    )
    assert all(
        "task_attribute" not in actor
        for actor in context.task_schema["tracked_actors"]
    )
    assert resolve_task_actor(
        nested_context_task(actors),
        context.task_schema["tracked_actors"][1],
    ) is actors[1]
    guide = _semantic_field_access_guide(
        {"task_schema": context.task_schema}
    )
    assert (
        'self.actor_groups["targets"][0][0].get_pose().p' in guide
    )
