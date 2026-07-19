import json
import tempfile
import unittest
from pathlib import Path

from mea.planner import BoundTaskPlanSession, build_act_catalog
from mea.proposal_agent import BoundedProposalAgent


class FakeProvider:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def text(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return json.dumps(self.value)


def _catalog(root: Path) -> dict:
    for task in ("beat_block_hammer", "click_bell"):
        schema = root / "mea/toolkit/schemas" / f"{task}.json"
        schema.parent.mkdir(parents=True, exist_ok=True)
        schema.write_text(
            json.dumps({"task_name": task, "task_family": "manipulation"}),
            encoding="utf-8",
        )
        checkpoint = root / "policy/ACT/act_ckpt" / f"act-{task}" / "demo_clean-50"
        checkpoint.mkdir(parents=True)
        (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")
        (checkpoint / "policy_last.ckpt").write_bytes(b"weights")
    return build_act_catalog(root)


class ProposalAgentTests(unittest.TestCase):
    def test_open_query_produces_unlisted_task_and_tool_proposals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = BoundTaskPlanSession.from_catalog(
                _catalog(root), "click_bell", max_rounds=1
            ).target
            value = {
                "schema_version": 1,
                "task_proposal": {
                    "schema_version": 1,
                    "proposal_id": "object_position.query_generated_midleft",
                    "task_name": "click_bell",
                    "aspect_id": "object_position",
                    "intent": "test an unseen safe left position",
                    "capability_id": "object_position.fixed_xy",
                    "reuse_first": True,
                    "changes": {
                        "bell": {
                            "position_mode": "fixed",
                            "xy": [-0.14, -0.12],
                        }
                    },
                    "preserve_success_semantics": True,
                },
                "tool_proposal": {
                    "schema_version": 1,
                    "proposal_id": "object_position.query_generated_midleft.tool",
                    "task_name": "click_bell",
                    "aspect_id": "object_position",
                    "evaluation_goal": "diagnose target reachability",
                    "metric": "bell_active_tcp_min_xy_error",
                    "question": "How close did the active TCP get to the bell?",
                    "vqa_phenomenon_ids": ["bell_visibly_pressed"],
                    "reuse_first": True,
                },
            }
            provider = FakeProvider(value)
            result = BoundedProposalAgent(provider, model="fake-model").propose(
                "How robust is this click_bell ACT policy to target position?",
                target=target,
                aspect_id="object_position",
            )
            self.assertEqual(
                result["task_proposal"]["changes"]["bell"]["xy"],
                [-0.14, -0.12],
            )
            self.assertEqual(
                result["tool_route_preview"]["resolved_route"], "force_codegen"
            )
            self.assertEqual(len(provider.calls), 1)
            self.assertIn("BOUND EVALUATION TARGET", provider.calls[0][0])

    def test_v2_tool_proposal_carries_a_bounded_run_local_visual_question(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = BoundTaskPlanSession.from_catalog(
                _catalog(root), "click_bell", max_rounds=1
            ).target
            value = {
                "schema_version": 1,
                "task_proposal": {
                    "schema_version": 1,
                    "proposal_id": "object_position.query_generated_midright",
                    "task_name": "click_bell",
                    "aspect_id": "object_position",
                    "intent": "test an unseen safe right position",
                    "capability_id": "object_position.fixed_xy",
                    "reuse_first": True,
                    "changes": {
                        "bell": {"position_mode": "fixed", "xy": [0.14, -0.12]}
                    },
                    "preserve_success_semantics": True,
                },
                "tool_proposal": {
                    "schema_version": 2,
                    "proposal_id": "object_position.query_generated_midright.tool",
                    "task_name": "click_bell",
                    "aspect_id": "object_position",
                    "evaluation_goal": "diagnose visible target interaction",
                    "metric": "bell_active_tcp_min_xy_error",
                    "question": "How close did the active TCP get to the bell?",
                    "vqa_phenomenon_ids": [
                        "bell_visibly_pressed",
                        "run_local.click_bell.midright_progress",
                    ],
                    "vqa_question_specs": [
                        {
                            "id": "run_local.click_bell.midright_progress",
                            "question_type": "visible_state_change",
                            "target_role": "task_target",
                            "question": (
                                "Does the robot visibly reach and interact with the "
                                "bell at the proposed position?"
                            ),
                            "visual_scope": "rollout_change",
                            "numeric_authority": (
                                "official_check_success_is_authoritative"
                            ),
                        }
                    ],
                    "reuse_first": True,
                },
            }
            result = BoundedProposalAgent(
                FakeProvider(value), model="fake-model"
            ).propose(
                "Can this click_bell ACT policy handle a new target position?",
                target=target,
                aspect_id="object_position",
            )
            self.assertEqual(result["tool_proposal"]["schema_version"], 2)
            self.assertEqual(
                result["tool_proposal"]["vqa_question_specs"][0]["id"],
                "run_local.click_bell.midright_progress",
            )


if __name__ == "__main__":
    unittest.main()
