import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from mea.planner import BoundTaskPlanSession, build_act_catalog
from mea.planner.context import (
    PlanningContextError,
    build_planning_context,
    validate_planning_context,
)


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


def _ready_catalog(root: Path) -> dict:
    for task_name in ("beat_block_hammer", "click_bell"):
        schema = root / "mea/toolkit/schemas" / f"{task_name}.json"
        schema.parent.mkdir(parents=True, exist_ok=True)
        schema.write_text(
            json.dumps(_task_schema(task_name)), encoding="utf-8"
        )
        checkpoint = (
            root
            / "policy/ACT/act_ckpt"
            / f"act-{task_name}"
            / "demo_clean-50"
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")
        (checkpoint / "policy_last.ckpt").write_bytes(b"weights")
    return build_act_catalog(root)


class PlanningContextTests(unittest.TestCase):
    def test_cards_are_schema_driven_and_cross_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = _ready_catalog(root)
            contexts = {}
            for task_name in ("click_bell", "beat_block_hammer"):
                session = BoundTaskPlanSession.from_catalog(
                    catalog, task_name, max_rounds=1
                )
                context = session.planning_context(root)
                contexts[task_name] = context
                self.assertEqual(context["schema_version"], 1)
                self.assertEqual(
                    context["policy_card"]["checkpoint_id"],
                    f"act-{task_name}/demo_clean-50",
                )
                self.assertEqual(
                    context["simulator_card"]["action_dimension"], 14
                )
                self.assertEqual(
                    context["adapter_view"]["task_name"], task_name
                )
                self.assertTrue(context["adapter_view"]["templates"])
                self.assertNotIn(str(root), json.dumps(context))
            self.assertNotEqual(
                contexts["click_bell"]["adapter_view"]["templates"],
                contexts["beat_block_hammer"]["adapter_view"]["templates"],
            )

    def test_context_rejects_source_drift_and_extra_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = BoundTaskPlanSession.from_catalog(
                _ready_catalog(root), "click_bell", max_rounds=1
            )
            context = build_planning_context(root, session.target)
            changed = deepcopy(context)
            changed["policy_card"]["policy_name"] = "another_policy"
            with self.assertRaisesRegex(
                PlanningContextError, "differs from trusted sources"
            ):
                validate_planning_context(
                    changed, repo_root=root, target=session.target
                )
            changed = deepcopy(context)
            changed["simulator_card"]["local_path"] = str(root)
            with self.assertRaisesRegex(PlanningContextError, "fields"):
                validate_planning_context(
                    changed, repo_root=root, target=session.target
                )


if __name__ == "__main__":
    unittest.main()
