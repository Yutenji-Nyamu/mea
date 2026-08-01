import unittest

from mea.agent_acceptance import _RUNTIME_DISCOVERY_RESOLUTIONS
from mea.planner.claim_first import PlanAgent, build_open_query_planning_lineage
from mea.planner.claim_first_runtime import (
    PlanAgentSession,
    PlanAgentSessionError,
    ClaimFirstRuntimeController,
    ClaimFirstRuntimeError,
    build_claim_first_evidence_record,
    build_dynamic_experiment_candidate,
    build_initial_semantic_proposal_bundle,
    control_template_id,
    resolve_concern_candidate_domain,
    resolve_semantic_proposal,
)
from mea.planner.policy_task_binding import build_policy_task_binding
from mea.planner.query_contract import build_query_sufficiency_contract
from mea.planner.semantic_coverage import build_evaluation_intent


def target():
    return {
        "task_name": "click_bell",
        "max_rounds": 3,
        "policy": {"policy_name": "ACT"},
        "aspects": [
            {
                "aspect_id": "performance.completion_time_stability",
                "description": "Unchanged official scene control.",
                "template_ids": [
                    "performance.completion_time_stability.official"
                ],
            },
            {
                "aspect_id": "object_position",
                "description": "Generalization across left and right positions.",
                "template_ids": [
                    "object_position.left_fixed",
                    "object_position.right_fixed",
                ],
            },
            {
                "aspect_id": "object_instance",
                "description": "Generalization across supported bell instances.",
                "template_ids": [
                    "object_instance.base0",
                    "object_instance.base1",
                ],
            },
            {
                "aspect_id": "robustness.scene_clutter",
                "description": "Generalization with official table clutter.",
                "template_ids": [
                    "robustness.scene_clutter.official_table",
                ],
            },
            {
                "aspect_id": "robustness.distractor_avoidance",
                "description": (
                    "Robustness to one nearby physical look-alike bell."
                ),
                "template_ids": [
                    "robustness.distractor_avoidance.lookalike_bell",
                ],
            },
        ],
    }


def round_plan(round_number, template_id):
    aspect = (
        "performance.completion_time_stability"
        if template_id.endswith(".official")
        else "object_position"
    )
    return {
        "round_id": f"round_{round_number}",
        "template_id": template_id,
        "sub_aspect": aspect,
        "task_instruction": f"Evaluate {template_id}.",
        "execution": {"num_episodes": 1, "seeds": [1000 + round_number]},
        "tool_request": {"metric": "time_to_success"},
        "task_proposal": {
            "aspect_id": aspect,
            "intent": f"Test {template_id}.",
            "changes": (
                {}
                if template_id.endswith(".official")
                else {"bell": {"position_mode": "fixed"}}
            ),
        },
    }


def summary(
    plan,
    success_rate,
    *,
    pipeline_passed=True,
    policy_outcome=None,
    outcome_semantics=None,
):
    if policy_outcome is None:
        policy_outcome = {
            "metric": "official_check_success",
            "authority": "official_check_success",
            "binding": None,
            "value": success_rate,
            "official_equivalent": True,
            "execution_scope": "official_equivalent",
        }
    return {
        "round_id": plan["round_id"],
        "taskgen_run_id": f"claim_first_{plan['round_id']}",
        "execution_artifact_dir": (
            f"mea/evaluations/claim_first/execution/{plan['round_id']}"
        ),
        "pipeline_passed": pipeline_passed,
        "observations": {
            "policy_success": success_rate,
            "policy_outcome": policy_outcome,
            "outcome_semantics": (
                outcome_semantics
                if outcome_semantics is not None
                else {
                    "schema_version": 1,
                    "status": "official_only",
                    "evidence_conflict": False,
                    "official_equivalent": True,
                    "episodes": [],
                    "reason_codes": ["no_generated_checker_result"],
                }
            ),
            "aggregate": {
                "status": "passed",
                "input_issues": [],
                "metrics": [
                    {
                        "metric": "time_to_success",
                        "cohorts": [
                            {
                                "role": "policy_under_evaluation",
                                "summary": {
                                    "quality": {
                                        "valid": 1,
                                        "missing": 0,
                                        "invalid": 0,
                                    }
                                },
                            }
                        ],
                    }
                ],
            },
            "planned_tool": {
                "status": "passed",
                "route_decision": {"metric": "time_to_success"},
                "episodes": [],
            },
            "execution_vqa": {
                "status": "passed",
                "evidence_conflict": False,
                "artifacts": {
                    "result": (
                        "mea/evaluations/claim_first/execution/"
                        f"{plan['round_id']}/execution_vqa/execution_vqa.json"
                    )
                },
            },
        },
    }


def semantic_bundle(sub_aspect="object_position.left_fixed"):
    return {
        "schema_version": 1,
        "source": "cached_test_proposal",
        "proposal": {
            "schema_version": 1,
            "action": "continue",
            "sub_aspect": sub_aspect,
            "hypothesis": "The left fixed position may expose a weakness.",
            "requested_perturbation": {
                "description": "Place the bell at the safe left position.",
                "controlled_changes": ["left position"],
                "preserve": ["task identity", "checkpoint"],
            },
            "task_need": {
                "required": True,
                "description": "Materialize the left scene.",
            },
            "tool_need": {
                "required": True,
                "description": "Measure policy success and completion time.",
                "reuse_first": True,
            },
            "rationale": "A left sentinel is the first diagnostic candidate.",
        },
    }


def semantic_stop_bundle(query, capabilities, evidence_history):
    lineage = build_open_query_planning_lineage(
        query,
        capabilities,
        evidence_history,
    )
    return {
        "schema_version": 2,
        "source": "provider_plan_agent_open_query",
        "proposal": {
            "schema_version": 2,
            "action": "stop",
            "sub_aspect": None,
            "hypothesis": (
                "The completed evidence identifies the first bounded failure."
            ),
            "requested_perturbation": None,
            "scene_need": {"required": False, "description": None},
            "checker_need": {"required": False, "description": None},
            "rule_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "rationale": (
                "A definitive candidate failure answers the diagnostic Query."
            ),
        },
        "input_digest": lineage["input_digest"],
        "planning_lineage": lineage,
    }


def open_query_capabilities():
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
                    "operation": "retrieve_or_generate_scene_checker",
                    "controlled_axis": None,
                    "generation_mode": "generic_provider_scene_checker_codegen",
                    "allowed_change_roots": [
                        "load_actors",
                        "check_success",
                    ],
                }
            ],
            "toolgen": {
                "retrieve_first": True,
                "can_generate_rule_metric": True,
                "can_generate_vqa_question": True,
            },
        },
    }


class EvidenceConditionedPlanner:
    def __init__(self):
        self.histories = []

    def propose(
        self,
        user_query,
        *,
        capabilities,
        evidence_history,
        evaluation_intent=None,
    ):
        self.histories.append(list(evidence_history))
        latest = evidence_history[-1]["outcome"] if evidence_history else None
        sub_aspect = (
            "object_instance.base0"
            if len(evidence_history) >= 2 and latest == "success"
            else "object_position.left_fixed"
        )
        bundle = semantic_bundle(sub_aspect)
        bundle["source"] = "provider_claim_first_open_query"
        lineage = build_open_query_planning_lineage(
            user_query,
            capabilities,
            evidence_history,
            evaluation_intent,
        )
        bundle["input_digest"] = lineage["input_digest"]
        bundle["planning_lineage"] = lineage
        return bundle


class EvidenceConditionedStopPlanner:
    def __init__(self):
        self.histories = []

    def propose(
        self,
        user_query,
        *,
        capabilities,
        evidence_history,
        evaluation_intent=None,
    ):
        self.histories.append(list(evidence_history))
        return semantic_stop_bundle(
            user_query,
            capabilities,
            evidence_history,
        )


class ClaimFirstRuntimeTests(unittest.TestCase):
    def test_acceptance_allows_unregistered_runtime_generation(self):
        self.assertIn(
            "generation_required_no_registered_candidate",
            _RUNTIME_DISCOVERY_RESOLUTIONS,
        )

    def test_plan_agent_prompt_uses_authoritative_preservation_and_aligned_tool(
        self,
    ):
        prompt = PlanAgent._prompt(
            "Where does this policy first fail?",
            open_query_capabilities(),
            [],
        )
        compact_prompt = " ".join(prompt.split())

        self.assertIn("current preservation authority", compact_prompt)
        self.assertIn(
            "simulator card are measurement capabilities, not preservation",
            compact_prompt,
        )
        self.assertIn(
            '"task identity" and "policy checkpoint" as the default preserve set',
            compact_prompt,
        )
        self.assertIn(
            "actor identity, physics timestep, or object-to-target binding",
            compact_prompt,
        )
        self.assertIn(
            '"official core predicate as a required conjunct"',
            compact_prompt,
        )
        self.assertIn("terminal-state distance threshold", compact_prompt)
        self.assertIn("terminal value of that same", compact_prompt)
        self.assertIn("trajectory peak or maximum", compact_prompt)
        self.assertIn(
            "must not use its scale to relax, replace",
            compact_prompt,
        )
        self.assertIn(
            "`diagnostic_tool_measurements` value is supporting diagnosis only",
            compact_prompt,
        )
        self.assertIn(
            "`peak`/`maximum over the rollout` is not a terminal/current-state value",
            compact_prompt,
        )
        self.assertIn(
            '`outcome="success"` for a terminal checker',
            compact_prompt,
        )

    def test_plan_agent_session_is_canonical_with_legacy_aliases(self):
        self.assertIs(ClaimFirstRuntimeController, PlanAgentSession)
        self.assertIs(ClaimFirstRuntimeError, PlanAgentSessionError)

        import mea.planner as planner_api
        from mea.planner.open_world_session import (
            OpenWorldPlanSession,
            PlanAgentExecutionSession,
            _FrozenExecutionTransport,
        )

        self.assertIs(OpenWorldPlanSession, _FrozenExecutionTransport)
        self.assertIs(PlanAgentExecutionSession, _FrozenExecutionTransport)
        self.assertNotIn("OpenWorldPlanSession", planner_api.__all__)
        self.assertNotIn("PlanAgentExecutionSession", planner_api.__all__)

    def test_unregistered_runtime_task_uses_official_control_anchor(self):
        task_name = "runtime_schema_task"
        runtime_target = {
            "schema_version": 3,
            "binding_mode": "single_task_single_checkpoint_open_world",
            "policy_task_binding": build_policy_task_binding(
                task_name=task_name,
                task_family="runtime_discovered",
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

        self.assertEqual(
            control_template_id(runtime_target),
            "task_execution.official_baseline",
        )
        plan = round_plan(1, "task_execution.official_baseline")
        plan["task_name"] = task_name
        plan["task_proposal"]["changes"] = {}
        observed = summary(plan, 1.0)
        record = build_claim_first_evidence_record(plan, observed)
        self.assertEqual(
            record["open_query_evidence"]["tested_perturbation"],
            "unchanged official-scene control",
        )

    def test_plan_agent_session_owns_frozen_execution_transport(self):
        task_name = "runtime_schema_task"
        query = "Is there some generated condition that exposes a failure?"
        runtime_target = {
            "schema_version": 3,
            "binding_mode": "single_task_single_checkpoint_open_world",
            "policy_task_binding": build_policy_task_binding(
                task_name=task_name,
                task_family="runtime_discovered",
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
        contract = build_query_sufficiency_contract(
            query,
            candidate_universe=[],
            round_budget=1,
            claim_type="existential",
            candidate_universe_closed=False,
            existential_witness_outcome="fail",
        )
        control = round_plan(1, "task_execution.official_baseline")
        control["task_name"] = task_name
        control["task_module"] = f"envs.{task_name}"
        control["task_proposal"]["changes"] = {}
        session = PlanAgentSession(
            query,
            runtime_target,
            query_contract=contract,
            control_round=control,
        )
        binding = session.execution_binding
        plan = {
            "schema_version": 1,
            "task_name": task_name,
            "policy": binding["policy"],
            "checkpoint": binding["checkpoint"],
            "max_rounds": 2,
            "evaluation_goal": query,
            "rounds": [control],
            "round_decisions": [],
            "planning_state": "awaiting_round_1_observation",
            "query_contract": contract,
        }

        normalized = session.normalize_plan(plan)
        snapshot = session.snapshot(query, normalized)

        self.assertEqual(normalized["task_name"], task_name)
        self.assertEqual(snapshot["target"], runtime_target)
        self.assertEqual(
            session.execution_binding["task_name"],
            snapshot["target"]["policy_task_binding"]["task_name"],
        )

    def test_each_dynamic_candidate_keeps_its_own_preservation_contract(
        self,
    ):
        first_intent = build_evaluation_intent(
            source_query="Where does this policy first expose a weakness?",
            original_concern="bell size generalization",
            hypothesis="A larger bell may expose the first weakness.",
            requested_change="Increase the bell diameter by 50%.",
            preserved_conditions=["position", "orientation"],
            required_observation="Measure click success and contact error.",
        )
        first_bundle = semantic_bundle("object_size.scale")
        first_bundle["proposal"]["hypothesis"] = first_intent["hypothesis"]
        first = build_dynamic_experiment_candidate(
            user_query=first_intent["source_query"],
            task_name="click_bell",
            proposal=first_bundle["proposal"],
            evaluation_intent=first_intent,
        )

        second_bundle = semantic_bundle("object_position.edge")
        second_bundle["proposal"]["requested_perturbation"]["preserve"] = [
            "bell size",
            "policy checkpoint",
        ]
        second = build_dynamic_experiment_candidate(
            user_query=first_intent["source_query"],
            task_name="click_bell",
            proposal=second_bundle["proposal"],
        )

        self.assertEqual(
            first["evaluation_intent"]["preserved_conditions"],
            ["position", "orientation"],
        )
        self.assertEqual(
            second["evaluation_intent"]["preserved_conditions"],
            ["bell size", "policy checkpoint"],
        )
        self.assertNotEqual(
            first["evaluation_intent"]["intent_id"],
            second["evaluation_intent"]["intent_id"],
        )
        self.assertEqual(
            second["intent_alignment"]["relationship"], "direct"
        )

    def test_quantified_chinese_scene_delta_is_not_mistaken_for_unchanged(self):
        query = "这个策略最先会在哪种物体变化上暴露弱点？"
        requested_change = (
            "仅将roller的统一物体尺度设为原始尺度的0.85，"
            "保持位置、姿态、外观、光照和杂物不变。"
        )
        intent = build_evaluation_intent(
            source_query=query,
            original_concern="object_geometry.graspable_scale_reduction",
            hypothesis="缩小roller可能导致抓取失败。",
            requested_change=requested_change,
            preserved_conditions=["task identity", "policy checkpoint"],
            required_observation="记录官方成功和TCP到接触位置的最小距离。",
        )
        bundle = semantic_bundle(
            "object_geometry.graspable_scale_reduction"
        )
        bundle["proposal"]["hypothesis"] = intent["hypothesis"]
        bundle["proposal"]["requested_perturbation"] = {
            "description": requested_change,
            "controlled_changes": ["roller scale: 1.0 -> 0.85"],
            "preserve": list(intent["preserved_conditions"]),
        }
        bundle["proposal"]["task_need"]["description"] = requested_change
        bundle["proposal"]["tool_need"]["description"] = intent[
            "required_observation"
        ]

        candidate = build_dynamic_experiment_candidate(
            user_query=query,
            task_name="grab_roller",
            proposal=bundle["proposal"],
            evaluation_intent=intent,
        )

        self.assertEqual(
            candidate["intent_alignment"]["relationship"], "direct"
        )
        self.assertNotIn(
            "requested_change",
            candidate["intent_alignment"]["unmatched_intent_fields"],
        )

    def test_generic_official_tasks_have_claim_first_control_anchors(self):
        for task_name in (
            "adjust_bottle",
            "grab_roller",
            "place_phone_stand",
        ):
            generic_target = {
                "task_name": task_name,
                "max_rounds": 1,
                "policy": {"policy_name": "ACT"},
                "aspects": [
                    {
                        "aspect_id": "task_execution.official_baseline",
                        "description": "Unchanged official task.",
                        "template_ids": [
                            "task_execution.official_baseline"
                        ],
                    }
                ],
            }
            self.assertEqual(
                control_template_id(generic_target),
                "task_execution.official_baseline",
            )
    def test_routed_aspects_bound_query_candidate_universe(self):
        controller = ClaimFirstRuntimeController(
            "Can it succeed on at least one bell-property variation?",
            target(),
            candidate_aspect_ids=["object_position", "object_instance"],
        )

        self.assertEqual(
            set(controller.query_contract["candidate_universe"]),
            {
                "object_position.left_fixed",
                "object_position.right_fixed",
                "object_instance.base0",
                "object_instance.base1",
            },
        )
        self.assertNotIn(
            "robustness.scene_clutter.official_table",
            controller.query_contract["candidate_universe"],
        )

    def test_unbound_open_query_keeps_all_non_control_candidates(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )

        self.assertEqual(
            set(controller.query_contract["candidate_universe"]),
            {
                "object_position.left_fixed",
                "object_position.right_fixed",
                "object_instance.base0",
                "object_instance.base1",
                "robustness.scene_clutter.official_table",
                "robustness.distractor_avoidance.lookalike_bell",
            },
        )
        self.assertNotIn(
            "performance.completion_time_stability.official",
            controller.query_contract["candidate_universe"],
        )
        self.assertFalse(
            controller.query_contract["candidate_universe_closed"]
        )

    def test_no_control_query_can_freeze_first_typed_candidate(self):
        query = (
            "Find one bounded object change that exposes a weakness and define "
            "one additional success condition."
        )
        concern = {
            "schema_version": 1,
            "source_query": query,
            "sub_aspect": "Bell localization under reduced target size",
            "hypothesis": "A half-size bell increases click-position error.",
            "task_intent": "Locate and click the intended bell.",
            "requested_variation": (
                "Reduce the bell to 50% size while keeping its position, "
                "appearance identity, camera, and scene layout unchanged."
            ),
            "measurement_need": (
                "Measure success and click-position error to the bell center."
            ),
        }
        intent = build_evaluation_intent(
            source_query=query,
            original_concern=concern["sub_aspect"],
            hypothesis=concern["hypothesis"],
            requested_change=concern["requested_variation"],
            preserved_conditions=[
                "position",
                "appearance identity",
                "camera",
                "scene layout",
            ],
            required_observation=concern["measurement_need"],
        )
        bundle = build_initial_semantic_proposal_bundle(
            user_query=query,
            concern=concern,
            experiment_needs={
                "scene_need": {
                    "required": True,
                    "description": concern["requested_variation"],
                },
                "checker_need": {
                    "required": True,
                    "description": (
                        "Pass only if official success holds and the bell "
                        "remains inside the workspace."
                    ),
                },
                "rule_tool_need": {
                    "required": True,
                    "description": concern["measurement_need"],
                    "reuse_first": True,
                },
                "vqa_tool_need": {
                    "required": False,
                    "description": None,
                    "reuse_first": True,
                },
            },
            evaluation_intent=intent,
        )
        candidate = build_dynamic_experiment_candidate(
            user_query=query,
            task_name="click_bell",
            proposal=bundle["proposal"],
            evaluation_intent=intent,
        )

        self.assertEqual(
            bundle["source"],
            "provider_plan_agent_direct_materialization",
        )
        self.assertIn("50%", candidate["scene_need"]["description"])
        self.assertIn(
            "Pass only if official success",
            candidate["checker_need"]["description"],
        )
        self.assertIsNotNone(candidate["rule_tool_need"])
        self.assertIn(
            "Measure success and click-position error",
            candidate["rule_tool_need"]["description"],
        )
        self.assertNotEqual(
            candidate["checker_need"]["description"],
            candidate["rule_tool_need"]["description"],
        )
        self.assertEqual(
            candidate["intent_alignment"]["relationship"],
            "direct",
        )
        controller = PlanAgentSession(
            query,
            target(),
            require_control_anchor=False,
        )
        registered = controller.register_frozen_candidate(candidate)
        self.assertIn(
            registered["candidate_id"],
            controller.query_contract["candidate_universe"],
        )
        self.assertFalse(
            controller.query_contract["candidate_universe_closed"]
        )
        observation = controller.observe([], [])
        bound = controller.bind_frozen_candidate(
            bundle,
            registered,
            observation,
            executed_candidate_ids=[],
        )
        self.assertEqual(
            bound["resolution"]["resolution"],
            "pre_evidence_query_proposal",
        )
        self.assertEqual(
            bound["plan_step"]["candidate_id"],
            registered["candidate_id"],
        )
        self.assertEqual(
            bound["planning_lineage"]["decision_kind"],
            "pre_evidence_query_candidate",
        )
        self.assertFalse(
            bound["planning_lineage"]["evidence_conditioned"]
        )
        self.assertEqual(
            bound["planning_lineage"]["completed_round_ids"],
            [],
        )

    def test_query_derived_candidate_is_not_rejected_by_catalog_inventory(self):
        candidate_id = "dynamic.click.bell.precontact.motion.abc123"
        contract = build_query_sufficiency_contract(
            "Does the policy jitter before contact?",
            candidate_universe=[candidate_id],
            round_budget=1,
            claim_type="diagnostic",
            candidate_universe_closed=True,
            control_requirement="not_required",
        )

        controller = ClaimFirstRuntimeController(
            "Does the policy jitter before contact?",
            target(),
            query_contract=contract,
            require_control_anchor=False,
        )

        self.assertEqual(
            controller.query_contract["candidate_universe"],
            [candidate_id],
        )

    def test_online_concern_uniquely_binds_distractor_candidate(self):
        resolution = resolve_concern_candidate_domain(
            {
                "source_query": (
                    "Can it avoid a visually similar distractor bell?"
                ),
                "sub_aspect": (
                    "Target selection with visually similar objects nearby."
                ),
                "hypothesis": (
                    "The policy clicks the target without confusing a similar object."
                ),
                "requested_variation": (
                    "Place a visually similar object beside the target."
                ),
                "measurement_need": (
                    "Observe target success and any distractor contact."
                ),
            },
            target=target(),
        )

        self.assertEqual(
            resolution["candidate_aspect_ids"],
            ["robustness.distractor_avoidance"],
        )
        self.assertEqual(
            resolution["resolution"], "unique_query_supported_concern"
        )
        self.assertEqual(resolution["decision"], "bind_single_aspect")
        self.assertFalse(resolution["catalog_was_model_visible"])

    def test_concrete_catalog_external_concern_preserves_unresolved_needs(self):
        concern = {
            "schema_version": 1,
            "source_query": (
                "How robust is this policy when the target object's mass changes?"
            ),
            "sub_aspect": "Object mass sensitivity.",
            "hypothesis": (
                "The policy may fail to lift the target when its mass increases."
            ),
            "task_intent": "Press the intended bell.",
            "requested_variation": (
                "Increase the target object mass while preserving its geometry."
            ),
            "measurement_need": (
                "Measure success and unintended contact under the heavier mass."
            ),
        }

        resolution = resolve_concern_candidate_domain(
            concern,
            target=target(),
        )

        self.assertEqual(resolution["decision"], "catalog_external")
        self.assertEqual(
            resolution["resolution"],
            "unsupported_or_generation_required",
        )
        self.assertIsNone(resolution["candidate_aspect_ids"])
        self.assertIsNone(resolution["selected_aspect_id"])
        self.assertEqual(resolution["selected_template_ids"], [])
        self.assertEqual(resolution["concern"], concern)
        self.assertEqual(
            resolution["task_need"],
            {
                "required": True,
                "description": concern["requested_variation"],
            },
        )
        self.assertEqual(
            resolution["tool_need"],
            {
                "required": True,
                "description": concern["measurement_need"],
                "reuse_first": True,
            },
        )
        self.assertEqual(
            resolution["catalog_external_specificity"]["canonical_aspect_id"],
            "object_physics.mass",
        )
        self.assertFalse(resolution["execution_authorized"])
        self.assertNotIn("resolved_template_id", resolution)

    def test_official_only_task_routes_specific_concern_to_generation(self):
        official_only_target = {
            "task_name": "place_phone_stand",
            "max_rounds": 1,
            "policy": {"policy_name": "ACT"},
            "aspects": [
                {
                    "aspect_id": "task_execution.official_baseline",
                    "description": "Unchanged official task.",
                    "template_ids": ["task_execution.official_baseline"],
                }
            ],
        }
        concern = {
            "schema_version": 1,
            "source_query": (
                "Does phone-to-stand clearance expose a placement weakness?"
            ),
            "sub_aspect": "Phone-to-stand clearance sensitivity.",
            "hypothesis": "Low clearance causes unintended stand contact.",
            "task_intent": "Place the phone on its stand.",
            "requested_variation": (
                "Reduce phone-to-stand clearance while preserving task identity."
            ),
            "measurement_need": (
                "Measure minimum clearance and unintended contact."
            ),
        }

        resolution = resolve_concern_candidate_domain(
            concern, target=official_only_target
        )

        self.assertEqual(resolution["decision"], "catalog_external")
        self.assertEqual(
            resolution["resolution"],
            "generation_required_no_registered_candidate",
        )
        self.assertEqual(resolution["ranked_aspects"], [])
        self.assertTrue(resolution["task_need"]["required"])
        self.assertFalse(resolution["execution_authorized"])

    def test_official_only_task_admits_broad_concern_to_open_planner(self):
        official_only_target = {
            "task_name": "adjust_bottle",
            "max_rounds": 2,
            "policy": {"policy_name": "ACT"},
            "aspects": [
                {
                    "aspect_id": "task_execution.official_baseline",
                    "description": "Unchanged official task.",
                    "template_ids": ["task_execution.official_baseline"],
                }
            ],
        }
        concern = {
            "schema_version": 1,
            "source_query": (
                "How does this policy generalize, and where is its first weakness?"
            ),
            "sub_aspect": "General manipulation robustness.",
            "hypothesis": "A yet-undiscovered object property may expose a weakness.",
            "task_intent": "Adjust the bottle to the requested state.",
            "requested_variation": "Choose one informative bounded variation.",
            "measurement_need": "Measure task success and the causal failure signal.",
        }

        resolution = resolve_concern_candidate_domain(
            concern, target=official_only_target
        )

        self.assertEqual(resolution["decision"], "catalog_external")
        self.assertEqual(
            resolution["resolution"],
            "open_world_candidate_discovery_required",
        )
        self.assertIsNone(resolution["candidate_aspect_ids"])
        self.assertTrue(resolution["candidate_discovery_required"])
        self.assertFalse(resolution["execution_authorized"])
        self.assertNotIn("task_need", resolution)
        self.assertNotIn("tool_need", resolution)

    def test_provider_incidental_catalog_words_do_not_hide_external_mass_concern(self):
        resolution = resolve_concern_candidate_domain(
            {
                "source_query": (
                    "How robust is this ACT policy when the target object mass changes?"
                ),
                "sub_aspect": (
                    "Effect of target object mass variation on policy performance"
                ),
                "hypothesis": (
                    "The policy will click the bell regardless of target mass."
                ),
                "task_intent": "Click the bell.",
                "requested_variation": (
                    "Test a range from very light to very heavy while keeping "
                    "all other properties constant."
                ),
                "measurement_need": (
                    "Determine success across the range of target object masses."
                ),
            },
            target=target(),
        )

        self.assertEqual(resolution["decision"], "catalog_external")
        self.assertEqual(
            resolution["resolution"],
            "unsupported_or_generation_required",
        )
        self.assertTrue(resolution["task_need"]["required"])
        self.assertTrue(resolution["tool_need"]["required"])
        self.assertFalse(resolution["execution_authorized"])

    def test_catalog_external_detail_not_grounded_in_query_stays_discoverable(self):
        resolution = resolve_concern_candidate_domain(
            {
                "source_query": "How robust is this policy in general?",
                "sub_aspect": "Object mass sensitivity.",
                "hypothesis": "The policy may fail when object mass increases.",
                "requested_variation": "Increase the object mass.",
                "measurement_need": "Measure success under the heavier mass.",
            },
            target=target(),
        )

        self.assertEqual(resolution["decision"], "discover_candidates")
        self.assertEqual(resolution["resolution"], "broad_or_ambiguous")
        self.assertNotIn("task_need", resolution)
        self.assertNotIn("tool_need", resolution)
        self.assertTrue(resolution["candidate_discovery_required"])
        self.assertFalse(resolution["execution_authorized"])

    def test_tied_registered_concern_enters_candidate_discovery(self):
        resolution = resolve_concern_candidate_domain(
            {
                "source_query": (
                    "Does changing object_position or object_instance expose a failure?"
                ),
                "sub_aspect": "object_position and object_instance",
                "hypothesis": "Either supported variation may fail.",
                "requested_variation": (
                    "Change object_position or object_instance."
                ),
                "measurement_need": "Measure success for each variation.",
            },
            target=target(),
        )

        self.assertEqual(resolution["decision"], "discover_candidates")
        self.assertEqual(resolution["resolution"], "broad_or_ambiguous")
        self.assertGreaterEqual(len(resolution["candidate_aspect_ids"]), 2)
        self.assertEqual(resolution["selected_template_ids"], [])

    def test_broad_or_tied_concern_keeps_full_candidate_domain(self):
        resolution = resolve_concern_candidate_domain(
            {
                "source_query": "How robust is this policy in general?",
                "sub_aspect": "General task robustness.",
                "hypothesis": "The policy may expose a weakness.",
                "requested_variation": "Change an appropriate property.",
                "measurement_need": "Measure task success.",
            },
            target=target(),
        )

        self.assertEqual(
            set(resolution["candidate_aspect_ids"]),
            {
                "object_position",
                "object_instance",
                "robustness.scene_clutter",
                "robustness.distractor_avoidance",
            },
        )
        self.assertEqual(
            resolution["resolution"], "broad_or_ambiguous"
        )
        self.assertEqual(resolution["decision"], "discover_candidates")

    def test_tool_value_and_reuse_route_reach_next_planner_evidence(self):
        plan = round_plan(
            1,
            "performance.completion_time_stability.official",
        )
        observed = summary(plan, 1.0)
        observed["observations"]["planned_tool"] = {
            "status": "passed",
            "route": "run_local_reuse",
            "reference_tool": "precontact_jerk_peak",
            "tool_request": {
                "metric_spec": {
                    "description": (
                        "Maximum jerk over finite pre-contact trajectory samples."
                    ),
                },
            },
            "route_decision": {
                "resolved_route": "run_local_reuse",
                "provider_called": False,
            },
            "validation": {
                "semantic_review": {
                    "checks": {
                        "returns_diagnostic_not_success": True,
                    },
                },
            },
            "episodes": [
                {
                    "result": {
                        "tool": "precontact_jerk_peak",
                        "value": 0.031,
                        "unit": "m/s^3",
                        "passed": False,
                        "details": {"reason": None},
                    }
                }
            ],
        }

        record = build_claim_first_evidence_record(plan, observed)

        self.assertEqual(
            record["planned_tool_evidence"],
            [
                {
                    "metric": "precontact_jerk_peak",
                    "value": 0.031,
                    "unit": "m/s^3",
                    "passed": False,
                    "route": "run_local_reuse",
                    "provider_called": False,
                    "null_reason": None,
                    "description": (
                        "Maximum jerk over finite pre-contact trajectory samples."
                    ),
                    "returns_diagnostic_not_success": True,
                }
            ],
        )
        evidence_summary = record["open_query_evidence"][
            "evidence_summary"
        ]
        self.assertIn("precontact_jerk_peak", evidence_summary)
        self.assertIn('"value": 0.031', evidence_summary)
        self.assertIn('"provider_called": false', evidence_summary)
        self.assertIn(
            "Maximum jerk over finite pre-contact trajectory samples.",
            evidence_summary,
        )
        self.assertIn(
            "authoritative_candidate_outcome=success",
            evidence_summary,
        )
        self.assertIn(
            "diagnostic_tool_role=supporting_measurement_not_success_authority",
            evidence_summary,
        )

    def test_explicit_evidence_artifact_paths_override_shared_round_directory(self):
        plan = round_plan(
            1,
            "performance.completion_time_stability.official",
        )
        observed = summary(plan, 1.0)
        observed["evidence_artifact_paths"] = {
            "round_aggregate": "repair/aggregate_result.json",
            "tool_execution": "repair/first_query/tool_execution.json",
        }

        record = build_claim_first_evidence_record(plan, observed)

        self.assertIn(
            {
                "kind": "round_aggregate",
                "path": "repair/aggregate_result.json",
            },
            record["evidence_refs"],
        )
        self.assertIn(
            {
                "kind": "tool_execution",
                "path": "repair/first_query/tool_execution.json",
            },
            record["evidence_refs"],
        )

    def test_flat_compact_tool_value_reaches_next_planner_evidence(self):
        plan = round_plan(
            1,
            "performance.completion_time_stability.official",
        )
        observed = summary(plan, 1.0)
        observed["observations"]["planned_tool"] = {
            "status": "passed",
            "route": "generate",
            "reference_tool": "query_bottle_tcp_min_distance",
            "route_decision": {
                "resolved_route": "generate",
                "provider_called": True,
            },
            "episodes": [
                {
                    "metric": "query_bottle_tcp_min_distance",
                    "value": 0.017,
                    "unit": "m",
                    "passed": True,
                    "details": {},
                }
            ],
        }

        record = build_claim_first_evidence_record(plan, observed)

        self.assertEqual(
            record["planned_tool_evidence"][0]["metric"],
            "query_bottle_tcp_min_distance",
        )
        self.assertEqual(record["planned_tool_evidence"][0]["value"], 0.017)
        self.assertTrue(
            record["planned_tool_evidence"][0]["provider_called"]
        )
    def test_explicit_change_intent_outranks_preserved_scene_tokens(self):
        proposal = semantic_bundle("bell_property.object_instance_transfer")[
            "proposal"
        ]
        proposal["requested_perturbation"] = {
            "description": (
                "Replace the default bell with a supported non-default bell_id."
            ),
            "controlled_changes": ["bell object_instance (bell_id)"],
            "preserve": [
                "bell position",
                "scene clutter",
                "lighting and background conditions",
            ],
        }
        proposal["rationale"] = (
            "Preserve clutter while testing object-instance transfer."
        )

        resolved = resolve_semantic_proposal(
            proposal,
            target=target(),
            executed_template_ids=[
                "performance.completion_time_stability.official"
            ],
            control_template=(
                "performance.completion_time_stability.official"
            ),
        )

        self.assertEqual(
            resolved["resolved_aspect_id"], "object_instance"
        )
        self.assertEqual(
            resolved["resolved_template_id"], "object_instance.base0"
        )
        self.assertEqual(
            resolved["resolution"],
            "explicit_change_intent_aspect_runtime_order",
        )

    def test_control_pass_automatically_binds_evidence_and_semantic_step(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        observed = summary(control, 1.0)

        state = controller.observe([control], [observed])

        self.assertTrue(state["control_passed"])
        self.assertFalse(state["assessment"]["should_stop"])
        self.assertEqual(
            state["open_query_evidence_history"][0]["outcome"], "success"
        )
        refs = state["records"][0]["evidence_refs"]
        self.assertEqual(
            {item["kind"] for item in refs},
            {
                "child_manifest",
                "round_aggregate",
                "tool_execution",
                "execution_vqa_result",
            },
        )
        self.assertTrue(all("sha256" not in item for item in refs))
        bound = controller.bind_semantic_step(
            semantic_bundle(),
            state,
            executed_template_ids=[control["template_id"]],
        )
        self.assertEqual(
            bound["resolution"]["retrieval_template_id"],
            "object_position.left_fixed",
        )
        self.assertNotIn("template_id", bound["plan_step"])
        self.assertIn("proposal", bound["plan_step"])
        self.assertFalse(
            bound["resolution"]["catalog_was_model_visible"]
        )
        self.assertTrue(
            bound["semantic_needs"]["task_need"]["required"]
        )
        self.assertTrue(
            bound["semantic_needs"]["tool_need"]["required"]
        )

    def test_candidate_rejection_is_n_zero_evidence_and_preserves_budget(self):
        query = "Where does this policy first expose a weakness?"
        controller = ClaimFirstRuntimeController(query, target())
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        rejected = round_plan(2, "object_position.left_fixed")
        rejected_summary = summary(
            rejected,
            None,
            pipeline_passed=False,
        )
        rejected_summary["observations"]["planning_observation"] = {
            "schema_version": 1,
            "kind": "candidate_unexecutable",
            "candidate_id": rejected["template_id"],
            "sub_aspect": rejected["sub_aspect"],
            "reason_code": "taskgen_expert_gate_candidate_unexecutable",
            "diagnosis": "target_pose cannot be None",
            "policy_rollouts_started": 0,
            "policy_sample_count": 0,
            "taskgen_attempt_summary": (
                "mea/generated_tasks/rejected/validation/"
                "task_generation_attempt_summary.json"
            ),
        }

        state = controller.observe(
            [control, rejected],
            [summary(control, 1.0), rejected_summary],
        )

        self.assertEqual(state["assessment"]["completed_rounds"], 0)
        self.assertEqual(state["assessment"]["budget_remaining"], 2)
        self.assertEqual(state["assessment"]["observed_candidate_ids"], [])
        self.assertEqual(state["records"][1]["policy_sample_count"], 0)
        self.assertIn(
            "planning_observation=candidate_unexecutable",
            state["open_query_evidence_history"][1]["evidence_summary"],
        )

        class ReselectPlanner:
            def propose(
                self,
                user_query,
                *,
                capabilities,
                evidence_history,
                evaluation_intent=None,
            ):
                assert "target_pose cannot be None" in (
                    evidence_history[-1]["evidence_summary"]
                )
                bundle = semantic_bundle("object_instance.base0")
                lineage = build_open_query_planning_lineage(
                    user_query,
                    capabilities,
                    evidence_history,
                    evaluation_intent,
                )
                bundle["input_digest"] = lineage["input_digest"]
                bundle["planning_lineage"] = lineage
                return bundle

        bound = controller.propose_and_bind_semantic_step(
            ReselectPlanner(),
            state,
            capabilities=open_query_capabilities(),
            executed_candidate_ids=[
                control["template_id"],
                rejected["template_id"],
            ],
        )
        self.assertEqual(
            bound["resolution"]["retrieval_template_id"],
            "object_instance.base0",
        )
        self.assertNotEqual(
            bound["plan_step"]["candidate_id"],
            rejected["template_id"],
        )

    def test_control_evidence_is_read_before_round_two_concern_is_authored(self):
        query = "Where does this policy first expose a weakness?"
        controller = ClaimFirstRuntimeController(query, target())
        precontrol_bundle = semantic_bundle("object_instance.base0")
        precontrol_candidate = build_dynamic_experiment_candidate(
            user_query=query,
            task_name="click_bell",
            proposal=precontrol_bundle["proposal"],
        )
        with self.assertRaisesRegex(
            ClaimFirstRuntimeError,
            "cannot freeze a pre-evidence candidate",
        ):
            controller.register_frozen_candidate(precontrol_candidate)
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        state = controller.observe([control], [summary(control, 1.0)])
        planner = EvidenceConditionedPlanner()

        bound = controller.propose_and_bind_semantic_step(
            planner,
            state,
            capabilities=open_query_capabilities(),
            executed_candidate_ids=[control["template_id"]],
        )

        self.assertEqual(len(planner.histories), 1)
        self.assertEqual(
            [item["round_id"] for item in planner.histories[0]],
            ["round_1"],
        )
        self.assertEqual(
            bound["resolution"]["retrieval_template_id"],
            "object_position.left_fixed",
        )
        self.assertIn("proposal", bound["plan_step"])
        self.assertEqual(
            bound["planning_lineage"]["decision_kind"],
            "evidence_conditioned_refinement",
        )
        self.assertEqual(
            bound["planning_lineage"]["completed_round_ids"],
            ["round_1"],
        )
        self.assertNotEqual(
            bound["plan_step"]["candidate_id"],
            precontrol_candidate["candidate_id"],
        )
        self.assertEqual(
            bound["plan_step"]["planning_lineage"],
            bound["planning_lineage"],
        )

    def test_round_three_concern_is_derived_from_round_two_aggregate(self):
        query = "Where does this policy first expose a weakness?"
        controller = ClaimFirstRuntimeController(query, target())
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        first_candidate = round_plan(2, "object_position.left_fixed")
        state = controller.observe(
            [control, first_candidate],
            [summary(control, 1.0), summary(first_candidate, 1.0)],
        )
        self.assertFalse(state["assessment"]["should_stop"])
        planner = EvidenceConditionedPlanner()

        bound = controller.propose_and_bind_semantic_step(
            planner,
            state,
            capabilities=open_query_capabilities(),
            executed_candidate_ids=[
                control["template_id"],
                first_candidate["template_id"],
            ],
        )

        self.assertEqual(
            [item["round_id"] for item in planner.histories[0]],
            ["round_1", "round_2"],
        )
        self.assertEqual(
            planner.histories[0][-1]["outcome"],
            "success",
        )
        self.assertEqual(
            bound["resolution"]["retrieval_template_id"],
            "object_instance.base0",
        )
        self.assertIn("proposal", bound["plan_step"])
        self.assertEqual(
            bound["planning_lineage"]["completed_round_ids"],
            ["round_1", "round_2"],
        )

    def test_precomputed_bundle_cannot_pass_current_evidence_lineage_gate(self):
        query = "Where does this policy first expose a weakness?"
        controller = ClaimFirstRuntimeController(query, target())
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        state = controller.observe([control], [summary(control, 1.0)])
        stale_bundle = EvidenceConditionedPlanner().propose(
            query,
            capabilities=open_query_capabilities(),
            evidence_history=[],
        )

        with self.assertRaisesRegex(
            ClaimFirstRuntimeError,
            "does not match the current completed",
        ):
            controller.bind_evidence_conditioned_semantic_step(
                stale_bundle,
                state,
                capabilities=open_query_capabilities(),
                executed_candidate_ids=[control["template_id"]],
            )

    def test_auxiliary_vqa_conflict_does_not_override_official_control_success(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        observed = summary(control, 1.0)
        observed["observations"]["execution_vqa"]["evidence_conflict"] = True

        state = controller.observe([control], [observed])

        self.assertTrue(state["control_passed"])
        self.assertFalse(state["assessment"]["should_stop"])
        self.assertEqual(
            state["records"][0]["evidence_packet"]["evidence_strength"],
            "conflicting",
        )

    def test_candidate_vqa_conflict_stops_with_explicit_conflict_reason(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        candidate = round_plan(2, "object_position.left_fixed")
        candidate_summary = summary(candidate, 0.0)
        candidate_summary["observations"]["execution_vqa"][
            "evidence_conflict"
        ] = True

        state = controller.observe(
            [control, candidate],
            [summary(control, 1.0), candidate_summary],
        )

        self.assertTrue(state["assessment"]["should_stop"])
        self.assertFalse(state["assessment"]["evidence_sufficient"])
        self.assertEqual(
            state["assessment"]["stop_reason"],
            "evidence_conflict",
        )
        self.assertEqual(
            state["records"][1]["candidate_evidence"]["outcome"],
            "conflict",
        )
        self.assertEqual(
            state["assessment"]["conflict_candidate_ids"],
            ["object_position.left_fixed"],
        )

    def test_exact_retrieval_is_only_a_hint_for_typed_proposal(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        observed = summary(control, 1.0)
        state = controller.observe([control], [observed])

        first = controller.bind_semantic_step(
            semantic_bundle("object_position"),
            state,
            executed_template_ids=[control["template_id"]],
        )
        self.assertEqual(
            first["resolution"]["resolution"],
            "retrieval_hint_then_reuse_or_generate",
        )
        self.assertTrue(first["resolution"]["hidden"])
        self.assertEqual(
            first["resolution"]["retrieval_template_id"],
            "object_position.left_fixed",
        )
        self.assertIsNone(first["resolution"]["resolved_template_id"])
        self.assertNotIn("template_id", first["plan_step"])
        self.assertIn("proposal", first["plan_step"])

        with self.assertRaisesRegex(
            ClaimFirstRuntimeError,
            "dynamic candidate was already executed",
        ):
            controller.bind_semantic_step(
                semantic_bundle("object_position"),
                state,
                executed_template_ids=[
                    control["template_id"],
                    first["plan_step"]["candidate_id"],
                ],
            )

    def test_failed_control_stops_before_property_attribution(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        observed = summary(control, 0.0)

        state = controller.observe([control], [observed])

        self.assertTrue(state["assessment"]["should_stop"])
        self.assertEqual(
            state["assessment"]["stop_reason"],
            "control_baseline_policy_failed",
        )
        self.assertFalse(state["query_answer"]["answered"])
        with self.assertRaisesRegex(
            ClaimFirstRuntimeError, "after the query contract stopped"
        ):
            controller.bind_semantic_step(
                semantic_bundle(),
                state,
                executed_template_ids=[control["template_id"]],
            )

    def test_generated_checker_cannot_authorize_the_official_control(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        observed = summary(
            control,
            1.0,
            policy_outcome={
                "metric": "generated_check_success",
                "authority": "compiled_success_spec_experimental_bounded",
                "binding": {"success_spec_sha256": "a" * 64},
                "value": 1.0,
                "official_equivalent": False,
                "execution_scope": "experimental_bounded",
            },
            outcome_semantics={
                "schema_version": 1,
                "status": "expected_semantic_extension",
                "evidence_conflict": False,
                "official_equivalent": False,
                "episodes": [
                    {
                        "generated_checker_success": False,
                        "official_success": True,
                        "official_core_predicate_satisfied": True,
                    }
                ],
                "reason_codes": [
                    "generated_checker_adds_constraints_beyond_official_core"
                ],
            },
        )

        state = controller.observe([control], [observed])

        self.assertFalse(state["control_passed"])
        self.assertEqual(
            state["assessment"]["stop_reason"],
            "control_baseline_non_official_outcome",
        )
        self.assertFalse(state["query_answer"]["answered"])

    def test_generated_candidate_checker_stays_explicit_in_query_answer(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        candidate = round_plan(2, "object_position.left_fixed")
        control_summary = summary(control, 1.0)
        candidate_summary = summary(
            candidate,
            0.0,
            policy_outcome={
                "metric": "generated_check_success",
                "authority": "compiled_success_spec_experimental_bounded",
                "binding": {"success_spec_sha256": "b" * 64},
                "value": 0.0,
                "official_equivalent": False,
                "execution_scope": "experimental_bounded",
            },
            outcome_semantics={
                "schema_version": 1,
                "status": "expected_semantic_extension",
                "evidence_conflict": False,
                "official_equivalent": False,
                "episodes": [
                    {
                        "generated_checker_success": False,
                        "official_success": True,
                        "official_core_predicate_satisfied": True,
                    }
                ],
                "reason_codes": [
                    "generated_checker_adds_constraints_beyond_official_core"
                ],
            },
        )

        state = controller.observe(
            [control, candidate],
            [control_summary, candidate_summary],
        )

        self.assertTrue(state["query_answer"]["answered"])
        self.assertEqual(
            state["query_answer"]["evaluation_outcomes"][1]["metric"],
            "generated_check_success",
        )
        self.assertTrue(
            any(
                "must not be interpreted as official benchmark success"
                in item
                for item in state["query_answer"]["limitations"]
            )
        )

    def test_expected_semantic_extension_is_explicit_but_can_diagnose(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        candidate = round_plan(2, "object_position.left_fixed")
        control_summary = summary(control, 1.0)
        candidate_summary = summary(
            candidate,
            0.0,
            policy_outcome={
                "metric": "generated_check_success",
                "authority": "llm_generated_python_ast_validated",
                "binding": {"module_sha256": "b" * 64},
                "value": 0.0,
                "official_equivalent": False,
                "execution_scope": "experimental_bounded",
            },
            outcome_semantics={
                "schema_version": 1,
                "status": "expected_semantic_extension",
                "evidence_conflict": False,
                "official_equivalent": False,
                "episodes": [
                    {
                        "generated_checker_success": False,
                        "official_success": True,
                        "official_core_predicate_satisfied": True,
                    }
                ],
                "reason_codes": [
                    "generated_checker_adds_constraints_beyond_official_core"
                ],
            },
        )

        state = controller.observe(
            [control, candidate],
            [control_summary, candidate_summary],
        )

        self.assertTrue(state["assessment"]["evidence_sufficient"])
        self.assertFalse(state["query_answer"]["evidence_conflict"])
        self.assertEqual(
            state["query_answer"]["answer_scope"],
            "bounded_experimental_query_semantics",
        )
        self.assertFalse(state["query_answer"]["official_benchmark_answered"])
        self.assertEqual(
            state["query_answer"]["outcome_semantics"][1]["status"],
            "expected_semantic_extension",
        )
        self.assertTrue(
            any(
                "not been certified as official-equivalent" in item
                for item in state["query_answer"]["limitations"]
            )
        )

    def test_outcome_semantics_conflict_blocks_query_sufficiency(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        candidate = round_plan(2, "object_position.left_fixed")
        control_summary = summary(control, 1.0)
        candidate_summary = summary(
            candidate,
            0.0,
            outcome_semantics={
                "schema_version": 1,
                "status": "conflict",
                "evidence_conflict": True,
                "official_equivalent": False,
                "episodes": [
                    {
                        "generated_checker_success": True,
                        "official_success": False,
                        "official_core_predicate_satisfied": False,
                    }
                ],
                "reason_codes": [
                    "generated_success_without_official_core_predicate"
                ],
            },
        )

        state = controller.observe(
            [control, candidate],
            [control_summary, candidate_summary],
        )

        self.assertTrue(state["assessment"]["should_stop"])
        self.assertFalse(state["assessment"]["evidence_sufficient"])
        self.assertEqual(
            state["assessment"]["stop_reason"],
            "outcome_semantics_conflict",
        )
        self.assertEqual(
            state["records"][1]["candidate_evidence"]["outcome"],
            "conflict",
        )
        self.assertFalse(state["query_answer"]["answered"])
        self.assertTrue(state["query_answer"]["evidence_conflict"])

    def test_non_comparable_generated_checker_fails_closed(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        candidate = round_plan(2, "object_position.left_fixed")
        candidate_summary = summary(
            candidate,
            1.0,
            policy_outcome={
                "metric": "generated_check_success",
                "authority": "llm_generated_python_ast_validated",
                "binding": {"module_sha256": "b" * 64},
                "value": 1.0,
                "official_equivalent": False,
                "execution_scope": "experimental_bounded",
            },
            outcome_semantics={
                "schema_version": 1,
                "status": "non_comparable",
                "evidence_conflict": False,
                "official_equivalent": False,
                "episodes": [],
                "reason_codes": [
                    "non_equivalent_checker_has_no_official_core_projection"
                ],
            },
        )

        state = controller.observe(
            [control, candidate],
            [summary(control, 1.0), candidate_summary],
        )

        self.assertTrue(state["assessment"]["should_stop"])
        self.assertFalse(state["assessment"]["evidence_sufficient"])
        self.assertEqual(
            state["assessment"]["stop_reason"],
            "outcome_semantics_non_comparable",
        )
        self.assertEqual(
            state["records"][1]["candidate_evidence"]["outcome"],
            "unknown",
        )
        self.assertFalse(state["query_answer"]["answered"])

    def test_control_semantics_conflict_invalidates_baseline(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        state = controller.observe(
            [control],
            [
                summary(
                    control,
                    1.0,
                    outcome_semantics={
                        "schema_version": 1,
                        "status": "conflict",
                        "evidence_conflict": True,
                        "official_equivalent": True,
                        "episodes": [],
                        "reason_codes": [
                            "generated_and_official_equivalent_predicates_disagree"
                        ],
                    },
                )
            ],
        )
        self.assertFalse(state["control_passed"])
        self.assertEqual(
            state["assessment"]["stop_reason"],
            "control_baseline_semantics_conflict",
        )

    def test_diagnostic_failure_stops_by_sufficiency_not_hard_cap(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        candidate = round_plan(2, "object_position.left_fixed")
        control_summary = summary(control, 1.0)
        candidate_summary = summary(candidate, 0.0)

        state = controller.observe(
            [control, candidate],
            [control_summary, candidate_summary],
        )

        self.assertTrue(state["assessment"]["evidence_sufficient"])
        self.assertEqual(
            state["assessment"]["stop_reason"], "evidence_sufficient"
        )
        self.assertEqual(state["assessment"]["claim_verdict"], "diagnosed")
        self.assertTrue(state["query_answer"]["answered"])
        self.assertNotIn("hard", state["query_answer"]["stop_reason"])
        self.assertIn(
            "object_position.right_fixed",
            state["query_answer"]["untested_candidate_ids"],
        )
        self.assertGreaterEqual(
            len(state["query_answer"]["evidence_refs"]), 6
        )

    def test_plan_agent_stop_is_admitted_when_query_contract_is_sufficient(self):
        query = "Where does this policy first expose a weakness?"
        controller = ClaimFirstRuntimeController(query, target())
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        candidate = round_plan(2, "object_position.left_fixed")
        state = controller.observe(
            [control, candidate],
            [summary(control, 1.0), summary(candidate, 0.0)],
        )
        capabilities = open_query_capabilities()
        bundle = semantic_stop_bundle(
            query,
            capabilities,
            state["open_query_evidence_history"],
        )

        bound = controller.bind_evidence_conditioned_semantic_step(
            bundle,
            state,
            capabilities=capabilities,
            executed_candidate_ids=[
                control["template_id"],
                candidate["template_id"],
            ],
        )

        self.assertEqual(bound["plan_step"]["action"], "stop")
        self.assertTrue(bound["plan_step"]["answered_query"])
        self.assertEqual(
            bound["resolution"]["resolution"],
            "query_contract_validated_stop",
        )
        self.assertEqual(
            bound["query_assessment"]["stop_reason"],
            "evidence_sufficient",
        )
        self.assertEqual(
            bound["planning_lineage"]["decision_kind"],
            "evidence_conditioned_refinement",
        )

    def test_agent_path_allows_stop_decision_after_sufficient_evidence(self):
        query = "Where does this policy first expose a weakness?"
        controller = ClaimFirstRuntimeController(query, target())
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        candidate = round_plan(2, "object_position.left_fixed")
        state = controller.observe(
            [control, candidate],
            [summary(control, 1.0), summary(candidate, 0.0)],
        )
        planner = EvidenceConditionedStopPlanner()

        bound = controller.propose_and_bind_semantic_step(
            planner,
            state,
            capabilities=open_query_capabilities(),
            executed_candidate_ids=[
                control["template_id"],
                candidate["template_id"],
            ],
        )

        self.assertEqual(bound["plan_step"]["action"], "stop")
        self.assertEqual(len(planner.histories), 1)
        self.assertEqual(
            [item["round_id"] for item in planner.histories[0]],
            ["round_1", "round_2"],
        )

    def test_agent_path_allows_continue_after_sufficiency_with_budget(self):
        query = "Where does this policy first expose a weakness?"
        controller = ClaimFirstRuntimeController(query, target())
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        candidate = round_plan(2, "object_position.left_fixed")
        state = controller.observe(
            [control, candidate],
            [summary(control, 1.0), summary(candidate, 0.0)],
        )
        planner = EvidenceConditionedPlanner()

        self.assertTrue(state["assessment"]["evidence_sufficient"])
        self.assertGreater(state["assessment"]["budget_remaining"], 0)
        bound = controller.propose_and_bind_semantic_step(
            planner,
            state,
            capabilities=open_query_capabilities(),
            executed_candidate_ids=[
                control["template_id"],
                candidate["template_id"],
            ],
        )

        self.assertEqual(bound["plan_step"]["action"], "propose")
        self.assertEqual(len(planner.histories), 1)

    def test_agent_path_rejects_continue_when_budget_is_exhausted(self):
        query = "Where does this policy first expose a weakness?"
        limited_target = target()
        limited_target["max_rounds"] = 2
        controller = ClaimFirstRuntimeController(query, limited_target)
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        candidate = round_plan(2, "object_position.left_fixed")
        state = controller.observe(
            [control, candidate],
            [summary(control, 1.0), summary(candidate, 0.0)],
        )
        planner = EvidenceConditionedPlanner()

        self.assertTrue(state["assessment"]["evidence_sufficient"])
        self.assertEqual(state["assessment"]["budget_remaining"], 0)
        proposal_bundle = controller.propose_semantic_step(
            planner,
            state,
            capabilities=open_query_capabilities(),
        )
        self.assertEqual(
            proposal_bundle["proposal"]["action"],
            "continue",
        )
        with self.assertRaisesRegex(
            ClaimFirstRuntimeError,
            "after the query contract stopped",
        ):
            controller.bind_evidence_conditioned_semantic_step(
                proposal_bundle,
                state,
                capabilities=open_query_capabilities(),
                executed_candidate_ids=[
                    control["template_id"],
                    candidate["template_id"],
                ],
            )

        self.assertEqual(len(planner.histories), 1)

    def test_plan_agent_stop_is_rejected_before_query_contract_is_sufficient(self):
        query = "Where does this policy first expose a weakness?"
        controller = ClaimFirstRuntimeController(query, target())
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        state = controller.observe([control], [summary(control, 1.0)])
        capabilities = open_query_capabilities()
        bundle = semantic_stop_bundle(
            query,
            capabilities,
            state["open_query_evidence_history"],
        )

        with self.assertRaisesRegex(
            ClaimFirstRuntimeError,
            "stop rejected by QueryContract",
        ):
            controller.bind_evidence_conditioned_semantic_step(
                bundle,
                state,
                capabilities=capabilities,
                executed_candidate_ids=[control["template_id"]],
            )

    def test_chinese_failure_query_requires_a_failing_witness(self):
        controller = ClaimFirstRuntimeController(
            "\u662f\u5426\u5b58\u5728\u4e00\u4e2a\u4f1a\u8ba9\u7b56\u7565\u5931\u8d25\u7684\u53d8\u5316\uff1f",
            target(),
        )

        self.assertEqual(
            controller.query_contract["existential_witness_outcome"],
            "fail",
        )

    def test_official_only_task_bootstraps_dynamic_candidate_after_control(self):
        official_only_target = {
            "task_name": "place_phone_stand",
            "max_rounds": 1,
            "policy": {"policy_name": "ACT"},
            "aspects": [
                {
                    "aspect_id": "task_execution.official_baseline",
                    "description": "Unchanged official task.",
                    "template_ids": ["task_execution.official_baseline"],
                }
            ],
        }
        controller = ClaimFirstRuntimeController(
            "Where does pose-dependent clearance first expose a weakness?",
            official_only_target,
        )
        self.assertEqual(controller.query_contract["schema_version"], 3)
        self.assertEqual(controller.query_contract["candidate_universe"], [])
        self.assertFalse(
            controller.query_contract["candidate_universe_closed"]
        )

        control = round_plan(1, "task_execution.official_baseline")
        control["task_name"] = "place_phone_stand"
        state = controller.observe([control], [summary(control, 1.0)])
        self.assertTrue(state["assessment"]["candidate_discovery_required"])

        bundle = semantic_bundle("object_pose.receptacle_clearance")
        bundle["proposal"]["hypothesis"] = (
            "A rotated stand may cause phone-to-stand collision before placement."
        )
        bundle["proposal"]["requested_perturbation"] = {
            "description": "Rotate only the phone stand within the safe workspace.",
            "controlled_changes": ["phone stand yaw"],
            "preserve": ["phone identity", "policy checkpoint"],
        }
        bundle["proposal"]["task_need"] = {
            "required": True,
            "description": "Generate the rotated-stand scene and success checker.",
        }
        bundle["proposal"]["tool_need"] = {
            "required": True,
            "description": "Measure minimum phone-to-stand clearance.",
            "reuse_first": True,
        }
        bound = controller.bind_semantic_step(
            bundle,
            state,
            executed_template_ids=[control["template_id"]],
        )

        dynamic_step = bound["plan_step"]
        candidate = dynamic_step["proposal"]
        self.assertEqual(bound["schema_version"], 2)
        self.assertEqual(
            bound["resolution"]["resolution"],
            "proposal_reuse_or_generate",
        )
        self.assertNotIn("template_id", dynamic_step)
        self.assertEqual(candidate["base_task"], "place_phone_stand")
        self.assertEqual(dynamic_step["candidate_id"], candidate["candidate_id"])
        self.assertIn(
            candidate["candidate_id"],
            controller.query_contract["candidate_universe"],
        )

        executed_dynamic = {
            "round_id": "round_2",
            "template_id": None,
            "candidate_id": candidate["candidate_id"],
            "sub_aspect": dynamic_step["aspect_id"],
            "task_instruction": candidate["checker_need"],
            "execution": {"num_episodes": 1, "seeds": [1002]},
            "tool_request": {"metric": "time_to_success"},
            "task_proposal": {
                "aspect_id": dynamic_step["aspect_id"],
                "intent": bundle["proposal"]["hypothesis"],
                "changes": {"stand": {"yaw_mode": "bounded"}},
            },
        }
        completed = controller.observe(
            [control, executed_dynamic],
            [summary(control, 1.0), summary(executed_dynamic, 0.0)],
        )
        self.assertEqual(
            completed["records"][1]["candidate_id"],
            candidate["candidate_id"],
        )
        self.assertTrue(completed["assessment"]["evidence_sufficient"])
        self.assertEqual(
            completed["assessment"]["stop_reason"], "evidence_sufficient"
        )

    def test_unresolved_catalog_proposal_falls_back_to_dynamic_candidate(self):
        controller = ClaimFirstRuntimeController(
            "Where does target mass first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        state = controller.observe([control], [summary(control, 1.0)])
        bundle = semantic_bundle("object_physics.mass")
        bundle["proposal"]["hypothesis"] = (
            "A heavier bell may cause an incomplete press."
        )
        bundle["proposal"]["requested_perturbation"] = {
            "description": "Increase only target bell mass.",
            "controlled_changes": ["target mass"],
            "preserve": ["bell geometry", "policy checkpoint"],
        }
        bundle["proposal"]["task_need"] = {
            "required": True,
            "description": "Generate a bounded heavier-bell scene and checker.",
        }
        bundle["proposal"]["tool_need"] = {
            "required": True,
            "description": "Measure press depth and unintended contact.",
            "reuse_first": True,
        }
        bundle["proposal"]["rationale"] = (
            "Target mass is the queried controlled factor."
        )

        bound = controller.bind_semantic_step(
            bundle,
            state,
            executed_template_ids=[control["template_id"]],
        )

        self.assertEqual(
            bound["resolution"]["resolution"],
            "proposal_reuse_or_generate",
        )
        self.assertEqual(controller.query_contract["schema_version"], 3)
        self.assertFalse(
            controller.query_contract["candidate_universe_closed"]
        )
        self.assertIn(
            bound["plan_step"]["candidate_id"],
            controller.query_contract["candidate_universe"],
        )

    def test_query_contract_can_skip_control_for_tool_only_diagnostic(self):
        official_only_target = {
            "task_name": "adjust_bottle",
            "max_rounds": 1,
            "policy": {"policy_name": "ACT"},
            "aspects": [
                {
                    "aspect_id": "task_execution.official_baseline",
                    "description": "Unchanged official task.",
                    "template_ids": ["task_execution.official_baseline"],
                }
            ],
        }
        contract = build_query_sufficiency_contract(
            "Diagnose post-release bottle wobble.",
            candidate_universe=[],
            round_budget=1,
            claim_type="diagnostic",
            candidate_universe_closed=False,
            control_requirement="not_required",
        )
        controller = ClaimFirstRuntimeController(
            "Diagnose post-release bottle wobble.",
            official_only_target,
            query_contract=contract,
        )
        state = controller.observe([], [])
        self.assertFalse(state["control_required"])
        self.assertIsNone(state["control_passed"])
        self.assertTrue(state["assessment"]["candidate_discovery_required"])

        bundle = semantic_bundle("motion.post_release_wobble")
        bundle["proposal"]["hypothesis"] = (
            "The bottle oscillates after policy release."
        )
        bundle["proposal"]["task_need"] = {
            "required": False,
            "description": None,
        }
        bundle["proposal"]["tool_need"] = {
            "required": True,
            "description": "Measure post-release angular velocity.",
            "reuse_first": True,
        }
        bound = controller.bind_semantic_step(
            bundle,
            state,
            executed_template_ids=[],
        )
        self.assertEqual(
            bound["resolution"]["resolution"],
            "proposal_reuse_or_generate",
        )
        candidate = bound["plan_step"]["proposal"]
        self.assertIsNone(candidate["scene_need"])
        self.assertIsNone(candidate["checker_need"])
        self.assertIsNotNone(candidate["rule_tool_need"])
        self.assertIsNone(candidate["vqa_tool_need"])
        self.assertEqual(candidate["tool_need"]["kind"], "measure")

    def test_query_contract_can_skip_control_for_universal_trajectory_claim(self):
        contract = build_query_sufficiency_contract(
            "Do all observed trajectories stay below the jerk threshold?",
            candidate_universe=[],
            round_budget=3,
            claim_type="universal",
            candidate_universe_closed=False,
            control_requirement="not_required",
        )
        controller = ClaimFirstRuntimeController(
            "Do all observed trajectories stay below the jerk threshold?",
            target(),
            query_contract=contract,
        )

        self.assertFalse(controller.require_control_anchor)
        self.assertEqual(
            controller.query_contract["control_requirement"],
            "not_required",
        )

    def test_legacy_control_flag_cannot_override_query_contract(self):
        contract = build_query_sufficiency_contract(
            "Diagnose post-release wobble.",
            candidate_universe=[],
            round_budget=1,
            claim_type="diagnostic",
            candidate_universe_closed=False,
            control_requirement="not_required",
        )
        with self.assertRaisesRegex(
            ClaimFirstRuntimeError, "conflicts with QueryContract"
        ):
            ClaimFirstRuntimeController(
                "Diagnose post-release wobble.",
                target(),
                query_contract=contract,
                require_control_anchor=True,
            )


if __name__ == "__main__":
    unittest.main()
