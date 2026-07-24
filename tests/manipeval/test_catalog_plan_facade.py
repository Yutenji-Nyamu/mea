import shutil
import tempfile
import unittest
from pathlib import Path

from mea.planner import (
    CATALOG_PLAN_TASKS,
    EXPERIMENT_ONLY_PLANNER_MODES,
    CatalogPlanAgent,
    CatalogPlanError,
    PlanMaterializer,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def copy_schema(root: Path, task_name: str) -> None:
    destination = root / "mea/toolkit/schemas"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        REPO_ROOT / f"mea/toolkit/schemas/{task_name}.json",
        destination / f"{task_name}.json",
    )


class CatalogPlanFacadeTests(unittest.TestCase):
    def test_materializer_covers_four_catalog_tasks(self):
        templates = {
            "beat_block_hammer": "safety.hammer_left_camera_contact.official",
            "click_bell": "performance.completion_time_stability.official",
            "adjust_bottle": "task_execution.official_baseline",
            "grab_roller": "task_execution.official_baseline",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task_name in CATALOG_PLAN_TASKS:
                copy_schema(root, task_name)
                round_plan = PlanMaterializer(
                    root,
                    task_name=task_name,
                    start_seed=11,
                    num_episodes=1,
                    execution_backend="act",
                ).materialize_plan_step(
                    templates[task_name],
                    1,
                    f"evaluate {task_name}",
                )
                self.assertEqual(round_plan["round_id"], "round_1")
                self.assertEqual(round_plan["task_name"], task_name)
                self.assertEqual(round_plan["template_id"], templates[task_name])
                self.assertEqual(round_plan["execution"]["backend"], "act")

    def test_plan_accepts_global_route_wrapper_for_official_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copy_schema(root, "adjust_bottle")
            wrapper = {
                "schema_version": 1,
                "task_name": "adjust_bottle",
                "task_profile": "official",
                "planner_kind": "deterministic_official_task",
                "proposal": {
                    "schema_version": 1,
                    "task_name": "adjust_bottle",
                    "evaluation_goal": "evaluate unchanged adjust_bottle",
                    "requested_aspect_ids": [
                        "task_execution.official_baseline"
                    ],
                    "first_aspect_id": "task_execution.official_baseline",
                },
            }
            manifest = CatalogPlanAgent(
                root,
                task_name="adjust_bottle",
                execution_backend="act",
                start_seed=19,
            ).plan(
                "evaluate adjust_bottle",
                evaluation_id="eval_catalog_adjust",
                validated_proposal=wrapper,
            )
            self.assertEqual(manifest["plan"]["task_name"], "adjust_bottle")
            self.assertEqual(
                manifest["plan"]["rounds"][0]["execution"]["backend"],
                "act",
            )
            self.assertFalse(manifest["planner"]["provider_called"])

    def test_plan_accepts_direct_claim_first_control_proposal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copy_schema(root, "click_bell")
            control = {
                "schema_version": 1,
                "task_name": "click_bell",
                "evaluation_goal": "establish control before attribution",
                "requested_aspect_ids": [
                    "performance.completion_time_stability"
                ],
                "first_aspect_id": "performance.completion_time_stability",
            }
            manifest = CatalogPlanAgent(
                root,
                task_name="click_bell",
                provider=None,
                start_seed=23,
                max_rounds=2,
            ).plan(
                "where does the policy first expose a weakness?",
                evaluation_id="eval_catalog_claim_first",
                validated_proposal=control,
            )
            self.assertEqual(
                manifest["plan"]["rounds"][0]["template_id"],
                "performance.completion_time_stability.official",
            )
            self.assertFalse(manifest["planner"]["provider_called"])

    def test_cross_task_proposal_is_rejected_at_facade_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copy_schema(root, "adjust_bottle")
            agent = CatalogPlanAgent(root, task_name="adjust_bottle")
            with self.assertRaisesRegex(
                CatalogPlanError, "cannot switch the bound task"
            ):
                agent.plan(
                    "evaluate adjust_bottle",
                    evaluation_id="eval_cross_task",
                    validated_proposal={
                        "schema_version": 1,
                        "task_name": "grab_roller",
                    },
                )

    def test_legacy_direct_modes_are_explicitly_experiments_only(self):
        self.assertEqual(
            set(EXPERIMENT_ONLY_PLANNER_MODES),
            {
                "click_bell_position_lr",
                "click_bell_adaptive_catalog",
                "click_bell_fixed_suite",
            },
        )


if __name__ == "__main__":
    unittest.main()
