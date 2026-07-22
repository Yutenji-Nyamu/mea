import json
import tempfile
import unittest
from pathlib import Path

from mea.planner import (
    GlobalQueryRouter,
    GlobalRouteError,
    build_act_catalog,
    build_global_route_prompt,
    route_to_bbh_proposal,
    route_to_click_proposal,
    route_to_planner_proposal,
    validate_route_selection,
)


def make_ready_repo(root: Path, *task_names: str) -> None:
    families = {
        "beat_block_hammer": "tool_use_contact",
        "click_bell": "press_contact",
    }
    for task_name in task_names:
        schema = root / f"mea/toolkit/schemas/{task_name}.json"
        schema.parent.mkdir(parents=True, exist_ok=True)
        schema.write_text(
            json.dumps({"task_name": task_name, "task_family": families[task_name]}),
            encoding="utf-8",
        )
        checkpoint = root / "policy/ACT/act_ckpt" / f"act-{task_name}" / "demo_clean-50"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")
        (checkpoint / "policy_last.ckpt").write_bytes(b"weights")


def click_route() -> dict:
    return {
        "schema_version": 2,
        "decision": "route",
        "task_name": "click_bell",
        "task_profile": "adaptive_properties",
        "evaluation_goal": "evaluate bell position and instance generalization",
        "requested_aspect_ids": ["object_position", "object_instance"],
        "first_aspect_id": "object_position",
        "unsupported_capabilities": [],
    }


def bbh_route() -> dict:
    return {
        "schema_version": 2,
        "decision": "route",
        "task_name": "beat_block_hammer",
        "task_profile": "generated",
        "evaluation_goal": "evaluate blue appearance and contact timing",
        "requested_aspect_ids": [
            "object_appearance.color",
            "performance.pickup_to_contact_timing",
        ],
        "first_aspect_id": "object_appearance.color",
        "unsupported_capabilities": [],
    }


class FakeProvider:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0
        self.last_metadata = {"provider": "fake"}

    def text(self, _prompt: str, **_kwargs) -> str:
        response = self.responses[self.calls]
        self.calls += 1
        return response


class GlobalQueryRouterTests(unittest.TestCase):
    def test_catalog_exposes_only_schema_and_checkpoint_ready_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_ready_repo(root, "beat_block_hammer")
            catalog = build_act_catalog(root)
            self.assertEqual(
                [task["task_name"] for task in catalog["tasks"]],
                ["beat_block_hammer"],
            )
            self.assertEqual(
                catalog["excluded_tasks"],
                [
                    {
                        "task_name": "click_bell",
                        "missing_requirements": [
                            "dataset_stats_missing",
                            "policy_weights_missing",
                            "task_schema_missing",
                        ],
                    }
                ],
            )
            with self.assertRaisesRegex(GlobalRouteError, "not ACT-ready"):
                validate_route_selection(click_route(), catalog)

    def test_strict_route_and_both_existing_proposal_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_ready_repo(root, "beat_block_hammer", "click_bell")
            catalog = build_act_catalog(root)

            click = route_to_click_proposal(click_route(), catalog)
            self.assertEqual(
                set(click),
                {
                    "schema_version",
                    "task_name",
                    "evaluation_goal",
                    "requested_aspect_ids",
                    "first_aspect_id",
                },
            )
            self.assertEqual(
                click["requested_aspect_ids"],
                [
                    "object_position",
                    "object_instance",
                ],
            )

            bbh = route_to_bbh_proposal(bbh_route(), catalog)
            self.assertEqual(bbh["schema_version"], 5)
            self.assertEqual(
                bbh["requested_template_ids"],
                [
                    "object_appearance.color_blue",
                    "performance.pickup_to_contact_timing",
                ],
            )
            self.assertEqual(bbh["first_template_id"], "object_appearance.color_blue")
            dispatched = route_to_planner_proposal(click_route(), catalog)
            self.assertEqual(dispatched["planner_kind"], "model_click_bell_adaptive_v1")

            click_task = next(
                task for task in catalog["tasks"] if task["task_name"] == "click_bell"
            )
            completion = next(
                aspect
                for aspect in click_task["aspects"]
                if aspect["aspect_id"] == "performance.completion_time_stability"
            )
            self.assertEqual(
                completion["template_ids"],
                ["performance.completion_time_stability.official"],
            )
            self.assertEqual(completion["taskgen_route"], "official")
            self.assertEqual(completion["default_metric"], "time_to_success")
            scene_route = {
                **click_route(),
                "evaluation_goal": "evaluate background and lighting shifts",
                "requested_aspect_ids": [
                    "scene_background_texture",
                    "scene_lighting",
                ],
                "first_aspect_id": "scene_background_texture",
            }
            scene = route_to_click_proposal(scene_route, catalog)
            self.assertEqual(
                scene["requested_aspect_ids"],
                ["scene_background_texture", "scene_lighting"],
            )

    def test_extra_fields_and_catalog_violations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_ready_repo(root, "beat_block_hammer", "click_bell")
            catalog = build_act_catalog(root)
            extra = {**click_route(), "seed": 7}
            with self.assertRaisesRegex(GlobalRouteError, "fields must be exactly"):
                validate_route_selection(extra, catalog)
            wrong_profile = {**click_route(), "task_profile": "official"}
            with self.assertRaisesRegex(GlobalRouteError, "not trusted"):
                validate_route_selection(wrong_profile, catalog)
            canonical_but_unavailable_aspect = {
                **click_route(),
                "requested_aspect_ids": ["camera_viewpoint"],
                "first_aspect_id": "camera_viewpoint",
            }
            with self.assertRaisesRegex(GlobalRouteError, "unsupported routed aspects"):
                validate_route_selection(canonical_but_unavailable_aspect, catalog)

    def test_unsupported_is_explicit_and_has_no_planner_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_ready_repo(root, "beat_block_hammer", "click_bell")
            catalog = build_act_catalog(root)
            unsupported = {
                "schema_version": 2,
                "decision": "unsupported",
                "task_name": None,
                "task_profile": None,
                "evaluation_goal": "evaluate camera-viewpoint robustness",
                "requested_aspect_ids": [],
                "first_aspect_id": None,
                "unsupported_capabilities": [
                    {"task_name": "click_bell", "aspect_id": "camera_viewpoint"}
                ],
            }
            self.assertEqual(
                validate_route_selection(unsupported, catalog)["decision"],
                "unsupported",
            )
            alias_gap = {
                **unsupported,
                "unsupported_capabilities": [
                    {
                        "task_name": "click_bell",
                        "aspect_id": "camera.viewpoint",
                    }
                ],
            }
            self.assertEqual(
                validate_route_selection(alias_gap, catalog)[
                    "unsupported_capabilities"
                ][0]["aspect_id"],
                "camera_viewpoint",
            )
            with self.assertRaisesRegex(GlobalRouteError, "no executable"):
                route_to_planner_proposal(unsupported, catalog)
            false_gap = {
                **unsupported,
                "unsupported_capabilities": [
                    {"task_name": "click_bell", "aspect_id": "object_position"}
                ],
            }
            with self.assertRaisesRegex(GlobalRouteError, "cannot be declared"):
                validate_route_selection(false_gap, catalog)
            task_qualified_gap = {
                **unsupported,
                "unsupported_capabilities": [
                    {
                        "task_name": "beat_block_hammer",
                        "aspect_id": "scene_background_texture",
                    }
                ],
            }
            self.assertEqual(
                validate_route_selection(task_qualified_gap, catalog)[
                    "unsupported_capabilities"
                ],
                task_qualified_gap["unsupported_capabilities"],
            )

    def test_router_retries_and_history_prompt_is_compact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_ready_repo(root, "beat_block_hammer", "click_bell")
            catalog = build_act_catalog(root)
            provider = FakeProvider(["{}", json.dumps(click_route())])
            planning_contexts = {
                "click_bell": {
                    "schema_version": 1,
                    "policy_card": {"task_name": "click_bell"},
                    "simulator_card": {
                        "task_name": "click_bell",
                        "simulator_name": "RoboTwin",
                    },
                    "adapter_view": {"task_name": "click_bell"},
                }
            }
            router = GlobalQueryRouter(
                provider,
                model="fake-model",
                catalog=catalog,
                planning_contexts=planning_contexts,
            )
            history = [
                {
                    "evaluation_id": "eval_prior",
                    "similarity": 0.9,
                    "user_request": "prior bell query",
                    "task_name": "click_bell",
                    "planning": {
                        "requested_template_ids": ["object_position.left_fixed"],
                        "first_template_id": "object_position.left_fixed",
                        "planning_state": "stopped_after_round_1",
                    },
                    "outcome": {"pipeline_passed": True},
                    "compatibility": {
                        "same_policy": True,
                        "same_checkpoint": True,
                    },
                    "trajectory": "must_not_enter_prompt",
                }
            ]
            result = router.route("test bell properties", history_context=history)
            self.assertEqual(result["selection"]["task_name"], "click_bell")
            self.assertEqual(
                result["resolved"]["checkpoint"]["checkpoint_id"],
                "act-click_bell/demo_clean-50",
            )
            self.assertEqual(
                [
                    item["taskgen_capability_id"]
                    for item in result["resolved"]["aspects"]
                ],
                ["object_position.fixed_xy", "object_instance.official_id"],
            )
            self.assertEqual(result["attempt_count"], 2)
            self.assertEqual(provider.calls, 2)
            self.assertNotIn("must_not_enter_prompt", router.last_prompt or "")
            self.assertNotIn("pipeline_passed", router.last_prompt or "")
            self.assertIn("eval_prior", router.last_prompt or "")
            self.assertIn(
                "TRUSTED POLICY / SIMULATOR / ADAPTER CONTEXT",
                router.last_prompt or "",
            )
            self.assertIn('"simulator_name": "RoboTwin"', router.last_prompt or "")
            prompt = build_global_route_prompt("same query", catalog, history)
            self.assertNotIn("trajectory", prompt)

    def test_parent_bound_aspects_are_exact_and_visible_to_child_router(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_ready_repo(root, "beat_block_hammer", "click_bell")
            catalog = build_act_catalog(root)
            position_only = {
                **click_route(),
                "evaluation_goal": "evaluate only position robustness",
                "requested_aspect_ids": ["object_position"],
                "first_aspect_id": "object_position",
            }
            provider = FakeProvider(
                [json.dumps(click_route()), json.dumps(position_only)]
            )
            router = GlobalQueryRouter(
                provider,
                model="fake-model",
                catalog=catalog,
                bound_task_name="click_bell",
                bound_requested_aspect_ids=["object_position"],
            )
            result = router.route("evaluate the parent-selected capability")
            self.assertEqual(
                result["selection"]["requested_aspect_ids"], ["object_position"]
            )
            self.assertEqual(result["attempt_count"], 2)
            self.assertIn(
                "fixed requested_aspect_ids to exactly ['object_position']",
                router.last_prompt or "",
            )

            with self.assertRaisesRegex(GlobalRouteError, "exactly match"):
                validate_route_selection(
                    click_route(),
                    catalog,
                    expected_task_name="click_bell",
                    expected_aspect_ids=["object_position"],
                )
            with self.assertRaisesRegex(GlobalRouteError, "require a bound task"):
                GlobalQueryRouter(
                    FakeProvider([]),
                    model="fake-model",
                    catalog=catalog,
                    bound_requested_aspect_ids=["object_position"],
                )


if __name__ == "__main__":
    unittest.main()
