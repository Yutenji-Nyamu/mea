import json
import unittest

from mea.planner.claim_first import (
    ClaimFirstOpenQueryAgent,
    ClaimFirstPlanError,
    project_open_query_capabilities,
    validate_open_query_capabilities,
    validate_open_query_evidence,
    validate_open_query_plan_proposal,
)


def _capabilities():
    return {
        "schema_version": 1,
        "policy_card": {
            "policy_name": "ACT",
            "task_name": "click_bell",
            "action_dimension": 14,
        },
        "simulator_card": {
            "simulator_name": "RoboTwin",
            "task_name": "click_bell",
            "tracked_actors": ["bell", "robot"],
        },
        "generation_card": {
            "taskgen_operations": [
                {
                    "operation": "scene_overlay",
                    "controlled_axis": "scene",
                    "generation_mode": "bounded_overlay",
                    "allowed_change_roots": ["scene"],
                }
            ],
            "toolgen": {
                "retrieve_first": True,
                "can_generate_rule_metric": True,
                "can_generate_vqa_question": True,
            },
        },
    }


def _evidence(outcome):
    return {
        "schema_version": 1,
        "round_id": "round_01",
        "tested_sub_aspect": "object_position.edge_offset",
        "tested_hypothesis": "The policy remains successful near the workspace edge.",
        "tested_perturbation": "Move the bell left within the reachable workspace.",
        "outcome": outcome,
        "evidence_summary": f"Cached typed evidence was {outcome}.",
        "limitations": ["one rollout"],
    }


def _proposal(
    sub_aspect,
    *,
    hypothesis,
    perturbation,
    task_required,
    tool_required,
):
    return {
        "schema_version": 1,
        "action": "continue",
        "sub_aspect": sub_aspect,
        "hypothesis": hypothesis,
        "requested_perturbation": {
            "description": perturbation,
            "controlled_changes": ["one diagnostic factor"],
            "preserve": ["task identity", "policy checkpoint"],
        },
        "task_need": {
            "required": task_required,
            "description": (
                "Generate the requested bounded scene."
                if task_required
                else None
            ),
        },
        "tool_need": {
            "required": tool_required,
            "description": (
                "Measure the requested diagnostic observable."
                if tool_required
                else None
            ),
            "reuse_first": True,
        },
        "rationale": "This next experiment best resolves the evidence-dependent uncertainty.",
    }


class _BranchingProvider:
    def __init__(self):
        self.prompts = []
        self.last_metadata = {"id": "fixture", "model": "fixture"}

    def text(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        if '"outcome": "failure"' in prompt:
            value = _proposal(
                "object_position.failure_boundary",
                hypothesis="Failure is caused by a reachable-workspace boundary.",
                perturbation="Bisect the failed offset toward the nominal position.",
                task_required=True,
                tool_required=False,
            )
        elif '"outcome": "ambiguous"' in prompt:
            value = _proposal(
                "observability.precontact_motion",
                hypothesis="The apparent outcome is confounded by unobserved precontact motion.",
                perturbation="Replay the same condition with precontact telemetry.",
                task_required=False,
                tool_required=True,
            )
        elif '"outcome": "success"' in prompt:
            value = _proposal(
                "robustness.visual_distractor",
                hypothesis="Position robustness may not transfer to target selection under clutter.",
                perturbation="Add one visually similar non-target distractor.",
                task_required=True,
                tool_required=True,
            )
        else:
            value = _proposal(
                "object_position.edge_offset",
                hypothesis="Workspace-edge position is a likely first generalization boundary.",
                perturbation="Move the bell left within the reachable workspace.",
                task_required=True,
                tool_required=False,
            )
        return json.dumps(value)


class _InvalidProvider:
    last_metadata = {}

    def __init__(self):
        self.calls = 0

    def text(self, *_args, **_kwargs):
        self.calls += 1
        return "{}"


class ClaimFirstOpenQueryTest(unittest.TestCase):
    def test_prompt_distinguishes_visible_axes_from_hidden_itinerary(self):
        prompt = ClaimFirstOpenQueryAgent._prompt(
            "Where is the first generalization weakness?",
            _capabilities(),
            [],
        )
        self.assertIn("candidate/template-ID itinerary", prompt)
        self.assertIn("may appear in the capability cards", prompt)
        self.assertIn("not a prescribed test order", prompt)
        self.assertNotIn("There is no candidate\naspect list", prompt)

    def test_capabilities_reject_predeclared_navigation(self):
        value = _capabilities()
        value["simulator_card"]["available_aspect_ids"] = ["object_position"]
        with self.assertRaisesRegex(
            ClaimFirstPlanError, "predeclared navigation"
        ):
            validate_open_query_capabilities(value)

    def test_runtime_projection_removes_aspect_and_template_itinerary(self):
        projected = project_open_query_capabilities(
            {
                "policy_card": {
                    "policy_name": "ACT",
                    "task_name": "click_bell",
                },
                "simulator_card": {
                    "simulator_name": "RoboTwin",
                    "task_name": "click_bell",
                    "available_aspect_ids": [
                        "object_position",
                        "object_instance",
                    ],
                },
                "adapter_view": {
                    "templates": [
                        {
                            "template_id": "object_position.left_fixed",
                            "aspect_id": "object_position",
                            "taskgen_operation": "scene_overlay",
                            "controlled_axis": "position",
                            "generation_mode": "bounded_overlay",
                            "allowed_change_roots": ["scene"],
                        }
                    ]
                },
            }
        )
        serialized = json.dumps(projected)
        self.assertNotIn("available_aspect_ids", serialized)
        self.assertNotIn("template_id", serialized)
        self.assertNotIn("aspect_id", serialized)
        self.assertEqual(
            projected["generation_card"]["taskgen_operations"][0]["operation"],
            "scene_overlay",
        )

    def test_success_failure_and_ambiguous_evidence_choose_different_next_tests(self):
        query = (
            "How does this ACT policy generalize across manipulated-object "
            "properties, and where does it first fail?"
        )
        selected = {}
        for outcome in ("success", "failure", "ambiguous"):
            provider = _BranchingProvider()
            bundle = ClaimFirstOpenQueryAgent(
                provider, model="fixture"
            ).propose(
                query,
                capabilities=_capabilities(),
                evidence_history=[_evidence(outcome)],
            )
            selected[outcome] = bundle["proposal"]["sub_aspect"]
            self.assertIn(query, provider.prompts[0])
            self.assertIn(f'"outcome": "{outcome}"', provider.prompts[0])
            self.assertNotIn('"aspect_id"', provider.prompts[0])
            self.assertNotIn('"template_id"', provider.prompts[0])
        self.assertEqual(len(set(selected.values())), 3)
        self.assertEqual(selected["failure"], "object_position.failure_boundary")
        self.assertEqual(selected["ambiguous"], "observability.precontact_motion")
        self.assertEqual(selected["success"], "robustness.visual_distractor")

    def test_first_proposal_has_no_hidden_route_or_fallback(self):
        provider = _BranchingProvider()
        result = ClaimFirstOpenQueryAgent(provider, model="fixture").propose(
            "Where is this policy's first object-generalization weakness?",
            capabilities=_capabilities(),
            evidence_history=[],
        )
        self.assertEqual(
            result["proposal"]["sub_aspect"], "object_position.edge_offset"
        )
        self.assertIn(
            "COMPLETED ROUND EVIDENCE (chronological; empty means first proposal):\n[]",
            provider.prompts[0],
        )
        self.assertNotIn("fallback_step", provider.prompts[0])

    def test_invalid_provider_does_not_restore_a_scripted_fallback(self):
        provider = _InvalidProvider()
        with self.assertRaisesRegex(
            ClaimFirstPlanError, "failed two open-Query proposal attempts"
        ):
            ClaimFirstOpenQueryAgent(provider, model="fixture").propose(
                "Where does it fail?",
                capabilities=_capabilities(),
                evidence_history=[],
            )
        self.assertEqual(provider.calls, 2)

    def test_stop_requires_completed_evidence_and_no_generation_need(self):
        stop = {
            "schema_version": 1,
            "action": "stop",
            "sub_aspect": None,
            "hypothesis": "The tested evidence answers the bounded Query.",
            "requested_perturbation": None,
            "task_need": {"required": False, "description": None},
            "tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "rationale": "The observed outcome directly resolves the requested claim.",
        }
        with self.assertRaisesRegex(ClaimFirstPlanError, "requires at least"):
            validate_open_query_plan_proposal(stop, has_evidence=False)
        self.assertEqual(
            validate_open_query_plan_proposal(stop, has_evidence=True)["action"],
            "stop",
        )

    def test_evidence_contract_rejects_duplicate_rounds(self):
        with self.assertRaisesRegex(ClaimFirstPlanError, "duplicate evidence"):
            validate_open_query_evidence(
                [_evidence("success"), _evidence("failure")]
            )


if __name__ == "__main__":
    unittest.main()
