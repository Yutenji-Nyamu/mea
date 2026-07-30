import json
import unittest

from mea.planner.claim_first import (
    PlanAgent,
    PlanAgentError,
    ClaimFirstOpenQueryAgent,
    ClaimFirstPlanError,
    build_open_query_planning_lineage,
    open_query_input_digest,
    project_open_query_capabilities,
    validate_open_query_capabilities,
    validate_open_query_evidence,
    validate_open_query_plan_proposal,
    validate_open_query_proposal_lineage,
)
from mea.planner.semantic_coverage import build_evaluation_intent


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


class _IntentRepairProvider:
    def __init__(self, *, allow_repair=True):
        self.prompts = []
        self.allow_repair = allow_repair
        self.last_metadata = {"id": "fixture", "model": "fixture"}

    def text(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        if (
            "PREVIOUS VALIDATION ERROR" not in prompt
            or not self.allow_repair
        ):
            value = {
                "schema_version": 2,
                "action": "continue",
                "sub_aspect": "robustness.visual_distractor",
                "hypothesis": (
                    "A lookalike distractor may cause target-selection "
                    "failure."
                ),
                "requested_perturbation": {
                    "description": "Add a visually similar distractor.",
                    "controlled_changes": ["distractor presence"],
                    "preserve": ["policy checkpoint"],
                },
                "scene_need": {
                    "required": True,
                    "description": "Add a lookalike distractor.",
                },
                "checker_need": {
                    "required": False,
                    "description": None,
                },
                "rule_tool_need": {
                    "required": True,
                    "description": "Measure target-selection accuracy.",
                    "reuse_first": True,
                },
                "vqa_tool_need": {
                    "required": False,
                    "description": None,
                    "reuse_first": True,
                },
                "rationale": "Probe a nearby robustness concern.",
            }
        else:
            value = {
                "schema_version": 2,
                "action": "continue",
                "sub_aspect": "motion.precontact_jitter",
                "hypothesis": (
                    "The policy exhibits visible jitter before contacting "
                    "the target."
                ),
                "requested_perturbation": {
                    "description": (
                        "Reuse the unchanged official scene and inspect only "
                        "the pre-contact trajectory."
                    ),
                    "controlled_changes": ["trajectory observation only"],
                    "preserve": ["official scene"],
                },
                "scene_need": {
                    "required": False,
                    "description": None,
                },
                "checker_need": {
                    "required": False,
                    "description": None,
                },
                "rule_tool_need": {
                    "required": True,
                    "description": (
                        "Measure pre-contact velocity oscillation and abrupt "
                        "acceleration."
                    ),
                    "reuse_first": True,
                },
                "vqa_tool_need": {
                    "required": False,
                    "description": None,
                    "reuse_first": True,
                },
                "rationale": "Directly measure the frozen motion concern.",
            }
        return json.dumps(value)


def _motion_intent():
    return build_evaluation_intent(
        source_query=(
            "In the unchanged official scene, does this ACT policy exhibit "
            "visible jitter before contacting the target?"
        ),
        original_concern="pre-contact motion smoothness",
        hypothesis=(
            "The policy exhibits visible jitter before contacting the target."
        ),
        requested_change=(
            "Reuse the unchanged official scene and inspect only the trajectory."
        ),
        preserved_conditions=["official scene"],
        required_observation=(
            "pre-contact velocity oscillation and abrupt acceleration"
        ),
    )


class ClaimFirstOpenQueryTest(unittest.TestCase):
    def test_plan_agent_is_canonical_and_claim_first_is_compatibility_alias(self):
        self.assertIs(ClaimFirstOpenQueryAgent, PlanAgent)
        self.assertIs(ClaimFirstPlanError, PlanAgentError)

    def test_typed_needs_are_independent_and_keep_legacy_views(self):
        proposal = validate_open_query_plan_proposal(
            {
                "schema_version": 2,
                "action": "continue",
                "sub_aspect": "motion.post_release_wobble",
                "hypothesis": "The bottle visibly wobbles after release.",
                "requested_perturbation": {
                    "description": "Reuse the official rollout.",
                    "controlled_changes": ["observation only"],
                    "preserve": ["scene", "checker"],
                },
                "scene_need": {"required": False, "description": None},
                "checker_need": {"required": False, "description": None},
                "rule_tool_need": {
                    "required": True,
                    "description": "Measure angular velocity.",
                    "reuse_first": True,
                },
                "vqa_tool_need": {
                    "required": True,
                    "description": "Check visible wobble.",
                    "reuse_first": True,
                },
                "rationale": "Numeric and visual evidence are complementary.",
            },
            has_evidence=False,
        )

        self.assertFalse(proposal["scene_need"]["required"])
        self.assertFalse(proposal["checker_need"]["required"])
        self.assertTrue(proposal["rule_tool_need"]["required"])
        self.assertTrue(proposal["vqa_tool_need"]["required"])
        self.assertFalse(proposal["task_need"]["required"])
        self.assertTrue(proposal["tool_need"]["required"])

    def test_legacy_task_need_does_not_force_checker_generation(self):
        proposal = validate_open_query_plan_proposal(
            _proposal(
                "object_position.edge_offset",
                hypothesis="Position changes may expose a failure.",
                perturbation="Move the target within the workspace.",
                task_required=True,
                tool_required=False,
            ),
            has_evidence=False,
        )

        self.assertTrue(proposal["scene_need"]["required"])
        self.assertFalse(proposal["checker_need"]["required"])

    def test_prompt_distinguishes_backend_primitives_from_hidden_itinerary(self):
        prompt = ClaimFirstOpenQueryAgent._prompt(
            "Where is the first generalization weakness?",
            _capabilities(),
            [],
        )
        self.assertIn("candidate/template-ID itinerary", prompt)
        self.assertIn("backend primitives", prompt)
        self.assertIn("not an operation menu", prompt)
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
        self.assertNotIn("scene_overlay", serialized)
        self.assertNotIn("controlled_axis", serialized)
        self.assertEqual(projected["schema_version"], 2)
        self.assertEqual(
            projected["generation_card"]["backend_primitives"],
            {
                "scene": True,
                "checker": True,
                "telemetry": False,
                "rule": True,
                "vqa": True,
                "retrieve": True,
                "generate": True,
            },
        )

    def test_runtime_projection_is_independent_of_query_bound_templates(self):
        context = {
            "policy_card": {"policy_name": "ACT", "task_name": "click_bell"},
            "simulator_card": {
                "simulator_name": "RoboTwin",
                "task_name": "click_bell",
                "available_aspect_ids": ["object_position", "robustness.distractor_avoidance"],
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
                    },
                    {
                        "template_id": "robustness.distractor_avoidance.lookalike_bell",
                        "aspect_id": "robustness.distractor_avoidance",
                        "taskgen_operation": "provider_scene_checker_codegen",
                        "controlled_axis": "robustness.distractor_avoidance",
                        "generation_mode": "provider_scene_checker_codegen",
                        "allowed_change_roots": ["distractor"],
                    },
                ]
            },
        }
        projected = project_open_query_capabilities(
            context,
            allowed_aspect_ids=["robustness.distractor_avoidance"],
        )
        unfiltered = project_open_query_capabilities(context)

        self.assertEqual(projected, unfiltered)
        self.assertEqual(
            set(projected["generation_card"]["backend_primitives"]),
            {
                "scene",
                "checker",
                "telemetry",
                "rule",
                "vqa",
                "retrieve",
                "generate",
            },
        )
        self.assertNotIn("aspect_id", json.dumps(projected))
        self.assertNotIn("distractor", json.dumps(projected))

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
            self.assertEqual(
                bundle["planning_lineage"]["decision_kind"],
                "evidence_conditioned_refinement",
            )
            self.assertTrue(
                bundle["planning_lineage"]["evidence_conditioned"]
            )
            self.assertEqual(
                bundle["planning_lineage"]["completed_round_ids"],
                ["round_01"],
            )
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
        result = PlanAgent(provider, model="fixture").propose(
            "Where is this policy's first object-generalization weakness?",
            capabilities=_capabilities(),
            evidence_history=[],
        )
        self.assertEqual(result["source"], "provider_plan_agent_open_query")
        self.assertIn(
            "You are the Plan Agent in ManipEvalAgent",
            provider.prompts[0],
        )
        self.assertNotIn("claim-first Plan Agent", provider.prompts[0])
        self.assertIn(
            "reuse_first MUST always be true",
            provider.prompts[0],
        )
        self.assertEqual(
            result["proposal"]["sub_aspect"], "object_position.edge_offset"
        )
        self.assertIn(
            "COMPLETED ROUND EVIDENCE (chronological; empty means first proposal):\n[]",
            provider.prompts[0],
        )
        self.assertNotIn("fallback_step", provider.prompts[0])
        self.assertEqual(
            result["planning_lineage"],
            build_open_query_planning_lineage(
                "Where is this policy's first object-generalization weakness?",
                _capabilities(),
                [],
            ),
        )
        self.assertEqual(
            result["planning_lineage"]["decision_kind"],
            "query_initial_candidate",
        )
        legacy_bundle = dict(result)
        legacy_bundle["source"] = "provider_claim_first_open_query"
        self.assertEqual(
            validate_open_query_proposal_lineage(
                legacy_bundle,
                user_query=(
                    "Where is this policy's first "
                    "object-generalization weakness?"
                ),
                capabilities=_capabilities(),
                evidence_history=[],
            ),
            result["planning_lineage"],
        )

    def test_explicit_success_definition_repairs_missing_checker_need(self):
        query = (
            "Is there a successful episode in which the target roller is "
            "lifted while the distractor roller remains on the table?"
        )
        missing_checker = _proposal(
            "target_selection.distractor",
            hypothesis="The policy may lift the wrong roller.",
            perturbation="Add one physical distractor roller.",
            task_required=True,
            tool_required=True,
        )
        corrected = {
            "schema_version": 2,
            "action": "continue",
            "sub_aspect": "target_selection.distractor",
            "hypothesis": "The policy may lift the wrong roller.",
            "requested_perturbation": {
                "description": "Add one physical distractor roller.",
                "controlled_changes": ["distractor roller"],
                "preserve": ["target roller pose", "policy checkpoint"],
            },
            "scene_need": {
                "required": True,
                "description": "Add one physical distractor roller.",
            },
            "checker_need": {
                "required": True,
                "description": (
                    "Success iff the target is lifted and the distractor "
                    "remains on the table."
                ),
            },
            "rule_tool_need": {
                "required": True,
                "description": "Measure both roller lift heights.",
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "rationale": "The experiment directly tests target selection.",
        }

        class Provider:
            last_metadata = {"model": "fixture"}

            def __init__(self):
                self.responses = [
                    json.dumps(missing_checker),
                    json.dumps(corrected),
                ]
                self.prompts = []

            def text(self, prompt, **_kwargs):
                self.prompts.append(prompt)
                return self.responses.pop(0)

        provider = Provider()
        result = PlanAgent(provider, model="fixture").propose(
            query,
            capabilities=_capabilities(),
            evidence_history=[_evidence("success")],
        )

        self.assertTrue(result["proposal"]["checker_need"]["required"])
        self.assertEqual(result["provider"]["attempt_count"], 2)
        self.assertIn(
            '"checker_need": {\n    "required": true',
            provider.prompts[0],
        )
        self.assertIn("checker_need.required", provider.prompts[1])

    def test_unadvertised_runtime_intervention_is_repaired_to_scene_change(self):
        unsupported = {
            "schema_version": 2,
            "action": "continue",
            "sub_aspect": "task_execution.gripper_precision",
            "hypothesis": "Reduced gripper precision may lift both objects.",
            "requested_perturbation": {
                "description": "Reduce the precision of the gripper.",
                "controlled_changes": ["gripper precision"],
                "preserve": ["task identity", "policy checkpoint"],
            },
            "scene_need": {
                "required": True,
                "description": "Generate a reduced-gripper-precision scene.",
            },
            "checker_need": {"required": False, "description": None},
            "rule_tool_need": {
                "required": True,
                "description": "Measure the target/non-target lift difference.",
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "rationale": "Probe a possible manipulation weakness.",
        }
        corrected = {
            **unsupported,
            "sub_aspect": "scene_geometry.distractor_spacing",
            "hypothesis": "Reduced target-distractor spacing may confuse selection.",
            "requested_perturbation": {
                "description": "Move the physical distractor closer to the target.",
                "controlled_changes": ["target-distractor spacing"],
                "preserve": ["task identity", "policy checkpoint"],
            },
            "scene_need": {
                "required": True,
                "description": "Change the distractor pose in load_actors.",
            },
        }

        class Provider:
            last_metadata = {"model": "fixture"}

            def __init__(self):
                self.responses = [json.dumps(unsupported), json.dumps(corrected)]
                self.prompts = []

            def text(self, prompt, **_kwargs):
                self.prompts.append(prompt)
                return self.responses.pop(0)

        provider = Provider()
        result = PlanAgent(provider, model="fixture").propose(
            "Where does target selection first become weak?",
            capabilities=_capabilities(),
            evidence_history=[_evidence("success")],
        )

        self.assertEqual(
            result["proposal"]["sub_aspect"],
            "scene_geometry.distractor_spacing",
        )
        self.assertEqual(result["provider"]["attempt_count"], 2)
        self.assertIn("runtime intervention", provider.prompts[1])

    def test_stale_bundle_cannot_be_relabelled_as_post_evidence_refinement(self):
        query = "Where is this policy's first object-generalization weakness?"
        bundle = ClaimFirstOpenQueryAgent(
            _BranchingProvider(), model="fixture"
        ).propose(
            query,
            capabilities=_capabilities(),
            evidence_history=[],
        )

        with self.assertRaisesRegex(
            ClaimFirstPlanError,
            "does not match the current completed",
        ):
            validate_open_query_proposal_lineage(
                bundle,
                user_query=query,
                capabilities=_capabilities(),
                evidence_history=[_evidence("success")],
            )

    def test_frozen_intent_repairs_silent_diagnostic_proxy(self):
        provider = _IntentRepairProvider()
        query = _motion_intent()["source_query"]
        intent = _motion_intent()
        result = ClaimFirstOpenQueryAgent(
            provider, model="fixture"
        ).propose(
            query,
            capabilities=_capabilities(),
            evidence_history=[],
            evaluation_intent=intent,
        )

        self.assertEqual(len(provider.prompts), 2)
        self.assertIn("FROZEN EVALUATION INTENT", provider.prompts[0])
        self.assertIn(intent["intent_id"], provider.prompts[0])
        self.assertIn(
            "silently pivots to a diagnostic proxy",
            provider.prompts[1],
        )
        self.assertEqual(
            result["proposal"]["sub_aspect"],
            "motion.precontact_jitter",
        )
        self.assertEqual(
            result["input_digest"],
            open_query_input_digest(
                query,
                _capabilities(),
                [],
                intent,
            ),
        )
        self.assertEqual(result["provider"]["attempt_count"], 2)

    def test_legacy_digest_excludes_optional_intent(self):
        query = _motion_intent()["source_query"]
        legacy = open_query_input_digest(query, _capabilities(), [])
        explicit_none = open_query_input_digest(
            query,
            _capabilities(),
            [],
            None,
        )
        with_intent = open_query_input_digest(
            query,
            _capabilities(),
            [],
            _motion_intent(),
        )

        self.assertEqual(legacy, explicit_none)
        self.assertNotEqual(legacy, with_intent)

    def test_frozen_intent_never_returns_an_unrepaired_proxy(self):
        provider = _IntentRepairProvider(allow_repair=False)
        intent = _motion_intent()

        with self.assertRaisesRegex(
            ClaimFirstPlanError,
            "failed two open-Query proposal attempts",
        ):
            ClaimFirstOpenQueryAgent(
                provider, model="fixture"
            ).propose(
                intent["source_query"],
                capabilities=_capabilities(),
                evidence_history=[],
                evaluation_intent=intent,
            )

        self.assertEqual(len(provider.prompts), 2)

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
