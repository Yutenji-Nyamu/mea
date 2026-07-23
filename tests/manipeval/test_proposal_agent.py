import json
import tempfile
import unittest
from pathlib import Path

from mea.planner import BoundTaskPlanSession, build_act_catalog
from mea.proposal_agent import (
    BoundedProposalAgent,
    ProposalAgentError,
    build_proposal_prompt,
)
from mea.capability_adapter import resolve_capability_contract


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
    def test_v3_can_propose_a_new_typed_metric_inside_bound_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = BoundTaskPlanSession.from_catalog(
                _catalog(root), "beat_block_hammer", max_rounds=1
            ).target
            contract = resolve_capability_contract(
                "beat_block_hammer", "object_appearance.color_blue"
            )
            value = {
                "schema_version": 1,
                "task_proposal": {
                    "schema_version": 1,
                    "proposal_id": "object_appearance.color_blue.reuse",
                    "task_name": "beat_block_hammer",
                    "aspect_id": "object_appearance.color",
                    "intent": "reuse the registered blue appearance variant",
                    "capability_id": contract["taskgen"]["capability_id"],
                    "reuse_first": True,
                    "changes": contract["taskgen"]["changes"],
                    "preserve_success_semantics": True,
                },
                "tool_proposal": {
                    "schema_version": 3,
                    "proposal_id": "query_contact_count.tool",
                    "task_name": "beat_block_hammer",
                    "aspect_id": "object_appearance.color",
                    "evaluation_goal": "count strict task contacts",
                    "metric": "query_hammer_block_contact_count",
                    "question": "How many strict task contacts occurred?",
                    "vqa_phenomenon_ids": [
                        "block_visibly_displaced",
                        "run_local.bbh.contact_count",
                    ],
                    "vqa_question_specs": [
                        {
                            "id": "run_local.bbh.contact_count",
                            "question_type": "visible_state_change",
                            "target_role": "task_target",
                            "question": "Does the rollout visibly show task contact?",
                            "visual_scope": "rollout_change",
                            "numeric_authority": (
                                "official_check_success_is_authoritative"
                            ),
                        }
                    ],
                    "reuse_first": True,
                    "metric_spec": {
                        "schema_version": 1,
                        "operation": "event_count",
                        "event": {
                            "event_type": "contact_interval",
                            "actors": ["020_hammer", "box"],
                            "physical_only": True,
                        },
                        "unit": "count",
                        "null_semantics": "zero_if_absent",
                    },
                },
            }
            provider = FakeProvider(value)
            result = BoundedProposalAgent(provider, model="fake-model").propose(
                "How many contacts occur under an appearance shift?",
                target=target,
                aspect_id="object_appearance.color",
                base_template_id="object_appearance.color_blue",
                capability_mode="registered_reuse",
            )
            self.assertEqual(result["tool_proposal"]["schema_version"], 3)
            self.assertEqual(
                result["tool_route_preview"]["resolved_route"],
                "typed_metric_spec_compile",
            )
            self.assertIn("typed_metric_spec_v1", provider.calls[0][0])
            self.assertIn("null_if_no_finite_sample", provider.calls[0][0])
            self.assertIn("020_hammer", provider.calls[0][0])

            value["tool_proposal"]["metric_spec"]["event"]["actors"] = [
                "020_hammer",
                "fl_link1",
            ]
            with self.assertRaisesRegex(
                ProposalAgentError, "outside the bound actor pairs"
            ):
                BoundedProposalAgent(
                    FakeProvider(value), model="fake-model"
                ).propose(
                    "How many contacts occur under an appearance shift?",
                    target=target,
                    aspect_id="object_appearance.color",
                    base_template_id="object_appearance.color_blue",
                    capability_mode="registered_reuse",
                )

    def test_safety_prompt_uses_precise_left_camera_metric_example(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = BoundTaskPlanSession.from_catalog(
                _catalog(root), "beat_block_hammer", max_rounds=1
            ).target
            prompt = build_proposal_prompt(
                "Did the hammer collide with the left camera?",
                target,
                "safety.hammer_left_camera_contact",
                base_template_id="safety.hammer_left_camera_contact.official",
                capability_mode="registered_reuse",
            )
            self.assertIn("query_hammer_left_camera_contact_count", prompt)
            self.assertIn('"actors": [', prompt)
            self.assertIn('"left_camera"', prompt)
            self.assertIn('"null_semantics": "zero_if_absent"', prompt)

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

    def test_malformed_v2_vqa_binding_gets_one_audited_bounded_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = BoundTaskPlanSession.from_catalog(
                _catalog(root), "click_bell", max_rounds=1
            ).target
            value = {
                "schema_version": 1,
                "task_proposal": {
                    "schema_version": 1,
                    "proposal_id": "object_position.query_generated_repair",
                    "task_name": "click_bell",
                    "aspect_id": "object_position",
                    "intent": "test an unseen safe left position",
                    "capability_id": "object_position.fixed_xy",
                    "reuse_first": True,
                    "changes": {
                        "bell": {"position_mode": "fixed", "xy": [-0.14, -0.12]}
                    },
                    "preserve_success_semantics": True,
                },
                "tool_proposal": {
                    "schema_version": 2,
                    "proposal_id": "object_position.query_generated_repair.tool",
                    "task_name": "click_bell",
                    "aspect_id": "object_position",
                    "evaluation_goal": "diagnose visible target interaction",
                    "metric": "bell_active_tcp_min_xy_error",
                    "question": "How close did the active TCP get to the bell?",
                    "vqa_phenomenon_ids": [
                        "bell_visibly_pressed",
                        "run_local.click_bell.mismatched",
                    ],
                    "vqa_question_specs": [],
                    "reuse_first": True,
                },
            }
            provider = FakeProvider(value)
            agent = BoundedProposalAgent(provider, model="fake-model")
            result = agent.propose(
                "Can this policy handle a new target position?",
                target=target,
                aspect_id="object_position",
            )
            repaired_tool = result["tool_proposal"]
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(len(agent.last_repairs), 1)
            self.assertEqual(
                agent.last_repairs[0]["action"],
                "bind_card_reference_vqa_question",
            )
            question_id = repaired_tool["vqa_question_specs"][0]["id"]
            self.assertIn(question_id, repaired_tool["vqa_phenomenon_ids"])
            self.assertNotIn(
                "run_local.click_bell.mismatched",
                repaired_tool["vqa_phenomenon_ids"],
            )

    def test_non_novel_axis_uses_registered_changes_and_receives_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = BoundTaskPlanSession.from_catalog(
                _catalog(root), "beat_block_hammer", max_rounds=1
            ).target
            contract = resolve_capability_contract(
                "beat_block_hammer", "object_appearance.color_blue"
            )
            value = {
                "schema_version": 1,
                "task_proposal": {
                    "schema_version": 1,
                    "proposal_id": "object_appearance.color_blue.reuse",
                    "task_name": "beat_block_hammer",
                    "aspect_id": "object_appearance.color",
                    "intent": "reuse the registered blue appearance variant",
                    "capability_id": contract["taskgen"]["capability_id"],
                    "reuse_first": True,
                    "changes": contract["taskgen"]["changes"],
                    "preserve_success_semantics": True,
                },
                "tool_proposal": {
                    "schema_version": 1,
                    "proposal_id": "object_appearance.color_blue.reuse.tool",
                    "task_name": "beat_block_hammer",
                    "aspect_id": "object_appearance.color",
                    "evaluation_goal": "check task outcome under the blue variant",
                    "metric": contract["tool"]["metric"],
                    "question": "Did strict hammer-block contact occur?",
                    "vqa_phenomenon_ids": contract["vqa"]["phenomenon_ids"],
                    "reuse_first": True,
                },
            }
            provider = FakeProvider(value)
            result = BoundedProposalAgent(provider, model="fake-model").propose(
                "Does appearance affect this ACT checkpoint?",
                target=target,
                aspect_id="object_appearance.color",
                base_template_id="object_appearance.color_blue",
                capability_mode="registered_reuse",
                planning_context={"schema_version": 1, "source": "unit_test"},
            )
            self.assertEqual(
                result["task_proposal"]["changes"], contract["taskgen"]["changes"]
            )
            self.assertIn("TRUSTED POLICY/SIMULATOR/ADAPTER CONTEXT", provider.calls[0][0])


if __name__ == "__main__":
    unittest.main()
