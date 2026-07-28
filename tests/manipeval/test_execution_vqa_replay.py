import json
import unittest

from experiments.paper.manipeval_execution_vqa_replay import (
    build_replay_query,
    extend_round_with_run_local_question,
)


class ExecutionVQAReplayTests(unittest.TestCase):
    def setUp(self):
        self.round_plan = {
            "round_id": "round_2",
            "task_name": "adjust_bottle",
            "template_id": None,
            "sub_aspect": "object_state.open_world_candidate",
            "tool_request": {
                "schema_version": 1,
                "task_name": "adjust_bottle",
                "metric": "runtime_generated_state_change",
                "question": "Did the tracked state change?",
            },
        }
        self.child_manifest = {"task_name": "adjust_bottle"}

    def test_cached_dynamic_round_uses_task_owned_bottle_question(self):
        query = build_replay_query(self.round_plan, self.child_manifest)
        self.assertEqual(
            query["phenomenon_ids"],
            ["bottle_visibly_repositioned"],
        )
        serialized = json.dumps(query).lower()
        self.assertNotIn("bell", serialized)
        self.assertNotIn("block", serialized)
        self.assertNotIn("hammer", serialized)

    def test_optional_run_local_question_extends_validated_default(self):
        question = {
            "id": "run_local.cached_object_progress",
            "question_type": "visible_state_change",
            "target_role": "task_target",
            "question": (
                "Does the cached rollout visibly show task-relevant progress?"
            ),
            "visual_scope": "rollout_change",
            "numeric_authority": "official_check_success_is_authoritative",
        }
        replay_plan = extend_round_with_run_local_question(
            self.round_plan,
            self.child_manifest,
            question,
        )
        query = build_replay_query(replay_plan, self.child_manifest)
        self.assertEqual(
            query["phenomenon_ids"],
            [
                "bottle_visibly_repositioned",
                "run_local.cached_object_progress",
            ],
        )
        self.assertEqual(query["questions"][1], question)


if __name__ == "__main__":
    unittest.main()
