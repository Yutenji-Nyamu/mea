from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mea.plan_agent_bootstrap import concern_candidate_domain_is_executable
from mea.planner.experiment_candidate import build_experiment_candidate
from mea.planner.query_interpretation import resolve_concern_candidate_domain
from mea.taskgen.round_materialization import materialize_open_world_round


REPO_ROOT = Path(__file__).resolve().parents[2]


class ProductionTaskIndependenceTests(unittest.TestCase):
    @staticmethod
    def _query_concern() -> dict[str, object]:
        return {
            "source_query": "Which change first exposes a policy weakness?",
            "sub_aspect": "target appearance under a bounded variation",
            "hypothesis": "The policy misses the changed target.",
            "requested_variation": "Change only the target appearance.",
            "measurement_need": "Observe target contact and task success.",
        }

    @staticmethod
    def _experiment_needs(*, taskgen: bool) -> dict[str, object]:
        return {
            "scene_need": {
                "required": taskgen,
                "description": (
                    "Change only the target appearance." if taskgen else None
                ),
            },
            "checker_need": {"required": False, "description": None},
            "rule_tool_need": {
                "required": True,
                "description": (
                    "Measure contact and success."
                    if taskgen
                    else "Reuse the official check_success result."
                ),
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
        }

    def test_typed_needs_go_directly_to_generic_proposal_work(self):
        concern = self._query_concern()
        needs = self._experiment_needs(taskgen=True)

        resolution = resolve_concern_candidate_domain(
            concern,
            experiment_needs=needs,
        )

        self.assertEqual(
            resolution,
            {
                "schema_version": 1,
                "decision": "proposal_reuse_or_generate",
                "resolution": "proposal_reuse_or_generate",
                "concern": concern,
                "experiment_needs": needs,
                "taskgen_required": True,
                "proposal_selection_required": True,
                "execution_authorized": True,
            },
        )
        self.assertTrue(
            concern_candidate_domain_is_executable(
                resolution,
                candidate_budget=1,
            )
        )
        self.assertFalse(
            concern_candidate_domain_is_executable(
                resolution,
                candidate_budget=0,
            )
        )

    def test_official_only_typed_needs_keep_the_direct_shortcut(self):
        resolution = resolve_concern_candidate_domain(
            self._query_concern(),
            experiment_needs=self._experiment_needs(taskgen=False),
        )

        self.assertEqual(
            resolution["resolution"],
            "official_execution_from_typed_needs",
        )
        self.assertEqual(resolution["decision"], "official_execution")
        self.assertFalse(resolution["taskgen_required"])
        self.assertNotIn("ranked_aspects", resolution)
        self.assertTrue(
            concern_candidate_domain_is_executable(
                resolution,
                candidate_budget=0,
            )
        )

    def test_unregistered_task_materializes_official_execution(self):
        candidate = {
            "schema_version": 2,
            "candidate_id": "press_stapler.official.success",
            "source_query": "Did the policy complete press_stapler?",
            "base_task": "press_stapler",
            "semantic_concern": "task_execution: official task success",
            "scene_need": None,
            "checker_need": None,
            "rule_tool_need": {
                "kind": "reuse",
                "description": "Reuse the official check_success result.",
                "reuse_first": True,
            },
            "vqa_tool_need": None,
        }
        with tempfile.TemporaryDirectory() as temporary:
            round_plan, tool_bundle = materialize_open_world_round(
                REPO_ROOT,
                Path(temporary),
                round_number=2,
                candidate=candidate,
                control_execution={
                    "backend": "act",
                    "seeds": [100000],
                    "num_episodes": 1,
                    "gates": [],
                },
                policy_backend="smolvla",
            )

        self.assertEqual(round_plan["task_name"], "press_stapler")
        self.assertEqual(round_plan["task_module"], "envs.press_stapler")
        self.assertEqual(round_plan["route"], "official")
        self.assertEqual(tool_bundle["source"], "official_checker_reuse")

    def test_unchanged_retry_reuses_official_route_and_frozen_seed_only(self):
        candidate = build_experiment_candidate(
            source_query="Can an unchanged retry establish the baseline?",
            base_task="press_stapler",
            semantic_concern="task_execution.official_retry",
            candidate_id="press_stapler.official.retry",
        )
        control_execution = {
            "backend": "act",
            "seeds": [314159],
            "num_episodes": 1,
            "gates": ["original_control_gate"],
        }

        with tempfile.TemporaryDirectory() as temporary:
            round_plan, tool_bundle = materialize_open_world_round(
                REPO_ROOT,
                Path(temporary),
                round_number=2,
                candidate=candidate,
                control_execution=control_execution,
                policy_backend="smolvla",
            )

        self.assertEqual(round_plan["route"], "official")
        self.assertEqual(round_plan["task_module"], "envs.press_stapler")
        self.assertEqual(round_plan["execution"]["seeds"], [314159])
        self.assertEqual(
            round_plan["execution"]["gates"],
            ["render", "smolvla", "toolkit", "aggregate"],
        )
        self.assertFalse(round_plan["open_tool_request_deferred"])
        self.assertEqual(
            tool_bundle["source"],
            "task_checker_evidence_no_new_tool_requested",
        )
        for need in ("task", "checker", "rule_tool", "vqa_tool"):
            self.assertFalse(
                round_plan["semantic_need_execution"][need]["requested"]
            )


if __name__ == "__main__":
    unittest.main()
