import json
import tempfile
import unittest
from pathlib import Path

from mea.planner.open_world_session import OpenWorldPlanSession
from mea.planner.runtime_task_binding import (
    RuntimeTaskBindingError,
    build_runtime_open_world_evaluation_target,
    build_runtime_policy_task_binding,
)


def _write_unregistered_runtime_task(root: Path) -> None:
    task_name = "novel_task"
    source = root / "envs" / f"{task_name}.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class novel_task:\n"
        "    def load_actors(self):\n"
        "        return None\n"
        "\n"
        "    def check_success(self):\n"
        "        return False\n",
        encoding="utf-8",
    )
    schema_path = root / "mea" / "toolkit" / "schemas" / f"{task_name}.json"
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_name": task_name,
                "task_family": "object_manipulation",
                "physics_timestep_seconds": 0.01,
                "action_dimension": 14,
                "probe_task_attributes": [],
                "tracked_actors": [
                    {
                        "id": "target",
                        "task_attribute": "target_actor",
                        "scene_name": "target",
                        "functional_points": [],
                        "contact_points": [],
                    }
                ],
                "contact_focus_actor_ids": ["target"],
                "semantic_fields": [
                    {
                        "name": "target_position",
                        "source": "actor_position",
                        "actor_id": "target",
                    }
                ],
                "semantic_roles": {"target": "target_position"},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = (
        root
        / "policy"
        / "ACT"
        / "act_ckpt"
        / f"act-{task_name}"
        / "demo_clean-50"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")
    (checkpoint / "policy_last.ckpt").write_bytes(b"weights")


class RuntimeTaskBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _write_unregistered_runtime_task(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_unregistered_source_schema_checkpoint_builds_binding(self):
        binding = build_runtime_policy_task_binding(
            self.root,
            "novel_task",
        )

        self.assertEqual(binding["task_name"], "novel_task")
        self.assertEqual(binding["task_module"], "envs.novel_task")
        self.assertEqual(
            binding["checkpoint"]["checkpoint_id"],
            "act-novel_task/demo_clean-50",
        )
        self.assertEqual(
            binding["hooks"]["official_success"]["method"],
            "check_success",
        )
        self.assertTrue(
            {
                "task_profile",
                "planner_kind",
                "aspects",
                "templates",
                "metrics",
            }.isdisjoint(binding)
        )

    def test_unregistered_binding_builds_catalog_free_open_world_target(self):
        target = build_runtime_open_world_evaluation_target(
            self.root,
            "novel_task",
            max_rounds=3,
        )

        self.assertEqual(target["max_rounds"], 3)
        self.assertEqual(
            target["policy_task_binding"]["task_name"],
            "novel_task",
        )
        self.assertEqual(
            set(target),
            {
                "schema_version",
                "binding_mode",
                "policy_task_binding",
                "max_rounds",
            },
        )
        session = OpenWorldPlanSession.from_target(target)
        self.assertEqual(session.task_name, "novel_task")
        self.assertEqual(
            session.control_template_id,
            "task_execution.official_baseline",
        )
        self.assertFalse(hasattr(session, "retrieval_aspects"))

    def test_missing_checkpoint_artifact_fails_closed(self):
        (
            self.root
            / "policy"
            / "ACT"
            / "act_ckpt"
            / "act-novel_task"
            / "demo_clean-50"
            / "policy_last.ckpt"
        ).unlink()

        with self.assertRaisesRegex(
            RuntimeTaskBindingError,
            "ACT policy weights is missing or empty",
        ):
            build_runtime_policy_task_binding(self.root, "novel_task")


if __name__ == "__main__":
    unittest.main()
