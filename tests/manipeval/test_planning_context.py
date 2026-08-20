import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from mea.planner.plan_agent_schema import project_open_query_capabilities
from mea.planner.context import (
    PlanningContextError,
    build_planning_context,
    validate_planning_context,
)
from mea.planner.policy_task_binding import build_policy_task_binding


def _task_schema(task_name: str) -> dict:
    return {
        "schema_version": 1,
        "task_name": task_name,
        "task_family": "manipulation",
        "trusted_tool_profile": "generic_success",
        "physics_timestep_seconds": 0.004,
        "action_dimension": 14,
        "probe_task_attributes": [],
        "tracked_actors": [
            {
                "id": "target",
                "task_attribute": "target",
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
        "semantic_roles": {"manipulated_object_position": "target_position"},
        "success_contract": {"type": "official_check_success"},
    }


def _runtime_target(root: Path, task_name: str) -> dict:
    schema = root / "mea/toolkit/schemas" / f"{task_name}.json"
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text(json.dumps(_task_schema(task_name)), encoding="utf-8")
    return {
        "schema_version": 3,
        "binding_mode": "single_task_single_checkpoint_open_world",
        "policy_task_binding": build_policy_task_binding(
            task_name=task_name,
            task_family="manipulation",
            policy={"name": "ACT", "language_conditioned": False},
            checkpoint={
                "checkpoint_id": f"act-{task_name}/demo_clean-50",
                "checkpoint_setting": "demo_clean",
                "expert_data_num": 50,
                "ready": True,
            },
        ),
        "max_rounds": 2,
    }


class PlanningContextTests(unittest.TestCase):
    def test_unregistered_runtime_binding_keeps_generic_generation_card(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_name = "runtime_schema_task"
            schema = root / "mea/toolkit/schemas" / f"{task_name}.json"
            schema.parent.mkdir(parents=True, exist_ok=True)
            schema.write_text(
                json.dumps(_task_schema(task_name)),
                encoding="utf-8",
            )
            source = root / "envs" / f"{task_name}.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(
                (
                    f"class {task_name}:\n"
                    "    def load_actors(self):\n"
                    "        pass\n\n"
                    "    def check_success(self):\n"
                    "        return False\n"
                ),
                encoding="utf-8",
            )
            target = _runtime_target(root, task_name)

            context = build_planning_context(root, target)
            projected = project_open_query_capabilities(context)

            self.assertEqual(
                set(context), {"schema_version", "policy_card", "simulator_card"}
            )
            self.assertEqual(
                projected["generation_card"]["backend_primitives"],
                {
                    "scene": True,
                    "checker": True,
                    "telemetry": True,
                    "rule": True,
                    "vqa": True,
                    "retrieve": True,
                    "generate": True,
                },
            )
            self.assertEqual(
                validate_planning_context(
                    context,
                    repo_root=root,
                    target=target,
                ),
                context,
            )

    def test_cards_are_schema_driven_and_cross_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contexts = {}
            for task_name in ("click_bell", "beat_block_hammer"):
                target = _runtime_target(root, task_name)
                context = build_planning_context(root, target)
                contexts[task_name] = context
                self.assertEqual(context["schema_version"], 1)
                self.assertEqual(
                    context["policy_card"]["checkpoint_id"],
                    f"act-{task_name}/demo_clean-50",
                )
                self.assertEqual(
                    context["simulator_card"]["action_dimension"], 14
                )
                self.assertNotIn("adapter_view", context)
                self.assertNotIn("available_aspect_ids", context["simulator_card"])
                self.assertNotIn(str(root), json.dumps(context))
            self.assertNotEqual(
                contexts["click_bell"]["policy_card"]["checkpoint_id"],
                contexts["beat_block_hammer"]["policy_card"]["checkpoint_id"],
            )

    def test_context_rejects_source_drift_and_extra_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = _runtime_target(root, "click_bell")
            context = build_planning_context(root, target)
            changed = deepcopy(context)
            changed["policy_card"]["policy_name"] = "another_policy"
            with self.assertRaisesRegex(
                PlanningContextError, "differs from trusted sources"
            ):
                validate_planning_context(
                    changed, repo_root=root, target=target
                )
            changed = deepcopy(context)
            changed["simulator_card"]["local_path"] = str(root)
            with self.assertRaisesRegex(PlanningContextError, "fields"):
                validate_planning_context(
                    changed, repo_root=root, target=target
                )
            changed = deepcopy(context)
            changed["adapter_view"] = {"templates": []}
            with self.assertRaisesRegex(PlanningContextError, "fields"):
                validate_planning_context(
                    changed,
                    repo_root=root,
                    target=target,
                )


if __name__ == "__main__":
    unittest.main()
