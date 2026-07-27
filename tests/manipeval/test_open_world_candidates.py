import unittest

from mea.planner.experiment_candidate import (
    ExperimentCandidateError,
    build_experiment_candidate,
    validate_experiment_candidate,
)
from mea.planner.query_contract import (
    assess_query_sufficiency,
    build_query_sufficiency_contract,
    extend_query_candidate_universe,
    infer_control_requirement,
    validate_query_sufficiency_contract,
)


def evidence(candidate_id: str, outcome: str, *, score=None) -> dict:
    return {
        "candidate_id": candidate_id,
        "outcome": outcome,
        "score": score,
        "diagnosis": None,
    }


class ExperimentCandidateTests(unittest.TestCase):
    def test_runtime_concern_materializes_without_template_id(self):
        candidate = build_experiment_candidate(
            source_query="Where does reflective appearance first fail?",
            base_task="beat_block_hammer",
            semantic_concern=(
                "novel.reflective_surface_confusion: "
                "A reflective target causes a pre-contact miss."
            ),
            scene_need="make only the target surface reflective",
            checker_need=(
                "Generate an experimental check_success predicate that decides "
                "whether the reflective target causes a pre-contact miss."
            ),
            tool_need="target contact and pre-contact trajectory",
        )
        self.assertEqual(candidate["base_task"], "beat_block_hammer")
        self.assertTrue(candidate["candidate_id"].startswith("dynamic."))
        self.assertIn("reflective.surface.confusion", candidate["candidate_id"])
        self.assertNotIn("template_id", candidate)
        self.assertIn(
            "pre-contact miss", candidate["checker_need"]["description"]
        )
        self.assertEqual(
            candidate["tool_need"]["description"],
            "target contact and pre-contact trajectory",
        )
        self.assertEqual(candidate["schema_version"], 2)

    def test_tool_only_candidate_does_not_invent_scene_or_checker_work(self):
        candidate = build_experiment_candidate(
            source_query="Does the TCP jerk before contact?",
            base_task="beat_block_hammer",
            semantic_concern="observability.precontact_jerk",
            tool_need={
                "kind": "measure",
                "description": "Measure peak jerk before first target contact.",
                "reuse_first": True,
            },
        )

        self.assertIsNone(candidate["scene_need"])
        self.assertIsNone(candidate["checker_need"])
        self.assertEqual(candidate["tool_need"]["kind"], "measure")

    def test_candidate_requires_at_least_one_typed_need(self):
        with self.assertRaisesRegex(
            ExperimentCandidateError, "at least one"
        ):
            build_experiment_candidate(
                source_query="Inspect this policy.",
                base_task="beat_block_hammer",
                semantic_concern="unspecified",
            )

    def test_v1_candidate_normalizes_to_typed_v2(self):
        candidate = validate_experiment_candidate(
            {
                "schema_version": 1,
                "candidate_id": "legacy.candidate",
                "source_query": "Where does it fail?",
                "base_task": "adjust_bottle",
                "semantic_concern": "post-release stability",
                "scene_need": "Reuse the official scene.",
                "checker_need": "Check stable final pose.",
                "tool_need": "Measure angular velocity.",
            }
        )

        self.assertEqual(candidate["schema_version"], 2)
        self.assertEqual(candidate["scene_need"]["kind"], "adapt")
        self.assertEqual(candidate["checker_need"]["kind"], "generate")
        self.assertEqual(candidate["tool_need"]["kind"], "measure")

    def test_candidate_identity_tracks_experiment_not_measurement_wording(self):
        base = {
            "source_query": "Where does the policy fail?",
            "base_task": "adjust_bottle",
            "semantic_concern": "object_pose: rotation sensitivity",
            "scene_need": "Rotate the bottle by 15 degrees.",
            "checker_need": "Require upright placement.",
            "tool_need": "Measure minimum TCP distance.",
        }
        same_experiment_new_tool = build_experiment_candidate(
            **{**base, "tool_need": "Count contact intervals."}
        )
        refined_scene = build_experiment_candidate(
            **{**base, "scene_need": "Rotate the bottle by 30 degrees."}
        )
        first = build_experiment_candidate(**base)

        self.assertEqual(
            first["candidate_id"],
            same_experiment_new_tool["candidate_id"],
        )
        self.assertNotEqual(first["candidate_id"], refined_scene["candidate_id"])

    def test_direct_builder_is_generic_and_strict(self):
        candidate = build_experiment_candidate(
            source_query="Does the bottle wobble after placement?",
            base_task="adjust_bottle",
            semantic_concern="post-placement stability",
            scene_need="reuse the official scene",
            checker_need="check final pose and sustained stability",
            tool_need="measure post-release angular velocity",
        )
        self.assertEqual(candidate["base_task"], "adjust_bottle")
        invalid = {**candidate, "template_id": "predeclared.variant"}
        with self.assertRaisesRegex(
            ExperimentCandidateError, "fields must be exactly"
        ):
            validate_experiment_candidate(invalid)

class OpenWorldQueryContractTests(unittest.TestCase):
    def test_control_requirement_is_query_conditioned(self):
        self.assertEqual(
            infer_control_requirement(
                "接触前是否存在抖动或急动？",
                semantic_context={
                    "measurement_need": "precontact jerk telemetry",
                },
            ),
            "not_required",
        )
        self.assertEqual(
            infer_control_requirement(
                "这个策略对物体外观属性的泛化能力如何？"
            ),
            "required",
        )
        self.assertEqual(
            infer_control_requirement("Diagnose this policy."),
            "required",
        )

    def test_open_contract_can_start_before_first_candidate_is_discovered(self):
        contract = build_query_sufficiency_contract(
            "Diagnose the first unsupported concern.",
            candidate_universe=[],
            round_budget=1,
            claim_type="diagnostic",
            candidate_universe_closed=False,
        )
        self.assertEqual(
            contract["required_coverage"]["minimum_evaluated"], 0
        )
        initial = assess_query_sufficiency(contract, [])
        self.assertTrue(initial["candidate_discovery_required"])
        self.assertFalse(initial["should_stop"])

        extended = extend_query_candidate_universe(
            contract, ["dynamic.first.concern"]
        )
        self.assertEqual(
            extended["required_coverage"]["minimum_evaluated"], 1
        )

    def test_failure_seeking_existential_stops_on_counterexample(self):
        contract = build_query_sufficiency_contract(
            "Is there some condition that exposes a failure?",
            candidate_universe=["dynamic.reflective"],
            round_budget=2,
            claim_type="existential",
            candidate_universe_closed=False,
            existential_witness_outcome="fail",
        )
        result = assess_query_sufficiency(
            contract, [evidence("dynamic.reflective", "fail")]
        )
        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(result["claim_verdict"], "counterexample_found")
        self.assertEqual(result["stop_reason"], "evidence_sufficient")

    def test_open_existential_cannot_refute_without_domain_closure(self):
        contract = build_query_sufficiency_contract(
            "Is there some failing condition?",
            candidate_universe=["dynamic.reflective"],
            round_budget=2,
            claim_type="existential",
            candidate_universe_closed=False,
            existential_witness_outcome="fail",
        )
        result = assess_query_sufficiency(
            contract, [evidence("dynamic.reflective", "pass")]
        )
        self.assertFalse(result["evidence_sufficient"])
        self.assertEqual(result["claim_verdict"], "inconclusive")
        self.assertTrue(result["candidate_discovery_required"])

    def test_universal_remains_inconclusive_until_open_domain_is_closed(self):
        contract = build_query_sufficiency_contract(
            "Do all discovered appearance conditions pass?",
            candidate_universe=["dynamic.reflective"],
            round_budget=2,
            claim_type="universal",
            candidate_universe_closed=False,
        )
        open_result = assess_query_sufficiency(
            contract, [evidence("dynamic.reflective", "pass")]
        )
        self.assertFalse(open_result["evidence_sufficient"])
        self.assertTrue(open_result["candidate_discovery_required"])

        closed = extend_query_candidate_universe(
            contract, [], candidate_universe_closed=True
        )
        closed_result = assess_query_sufficiency(
            closed, [evidence("dynamic.reflective", "pass")]
        )
        self.assertTrue(closed_result["evidence_sufficient"])
        self.assertEqual(closed_result["claim_verdict"], "supported")

    def test_open_universal_failure_is_conservatively_inconclusive(self):
        contract = build_query_sufficiency_contract(
            "Do all possible conditions pass?",
            candidate_universe=["dynamic.reflective"],
            round_budget=2,
            claim_type="universal",
            candidate_universe_closed=False,
        )
        result = assess_query_sufficiency(
            contract, [evidence("dynamic.reflective", "fail")]
        )
        self.assertFalse(result["evidence_sufficient"])
        self.assertEqual(result["claim_verdict"], "inconclusive")

    def test_worst_case_requires_domain_closure(self):
        contract = build_query_sufficiency_contract(
            "What is the worst-case condition?",
            candidate_universe=["left", "right"],
            round_budget=3,
            claim_type="worst_case",
            candidate_universe_closed=False,
        )
        observations = [
            evidence("left", "pass", score=0.8),
            evidence("right", "fail", score=0.2),
        ]
        open_result = assess_query_sufficiency(contract, observations)
        self.assertFalse(open_result["evidence_sufficient"])
        closed = extend_query_candidate_universe(
            contract, [], candidate_universe_closed=True
        )
        closed_result = assess_query_sufficiency(closed, observations)
        self.assertTrue(closed_result["evidence_sufficient"])
        self.assertEqual(
            closed_result["statistics"]["worst_candidate_ids"], ["right"]
        )

    def test_dynamic_candidate_extends_required_universe(self):
        contract = build_query_sufficiency_contract(
            "Does every discovered concern pass?",
            candidate_universe=["official.control"],
            round_budget=3,
            claim_type="universal",
            candidate_universe_closed=False,
        )
        candidate = build_experiment_candidate(
            source_query="Does every discovered concern pass?",
            base_task="place_phone_stand",
            semantic_concern="pose-dependent receptacle clearance",
            scene_need="vary stand yaw within the existing workspace",
            checker_need="check stable placement without stand collision",
            tool_need="measure minimum phone-to-stand clearance",
        )
        extended = extend_query_candidate_universe(
            contract, [candidate["candidate_id"]]
        )
        self.assertEqual(
            extended["candidate_universe"],
            ["official.control", candidate["candidate_id"]],
        )
        self.assertEqual(
            extended["required_coverage"]["candidate_ids"],
            ["official.control", candidate["candidate_id"]],
        )
        self.assertEqual(
            extended["required_coverage"]["minimum_evaluated"], 2
        )

    def test_finite_contract_keeps_legacy_truth_semantics(self):
        contract = build_query_sufficiency_contract(
            "Does every registered condition pass?",
            candidate_universe=["registered.left", "registered.right"],
            round_budget=2,
            claim_type="universal",
        )
        self.assertEqual(contract["schema_version"], 3)
        self.assertEqual(contract["control_requirement"], "required")
        result = assess_query_sufficiency(
            contract, [evidence("registered.left", "fail")]
        )
        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(result["claim_verdict"], "refuted")

    def test_candidate_extension_preserves_control_requirement(self):
        contract = build_query_sufficiency_contract(
            "Diagnose pre-contact motion.",
            candidate_universe=[],
            round_budget=2,
            claim_type="diagnostic",
            candidate_universe_closed=False,
            control_requirement="not_required",
        )

        extended = extend_query_candidate_universe(contract, ["jerk"])

        self.assertEqual(extended["schema_version"], 3)
        self.assertEqual(
            extended["control_requirement"], "not_required"
        )

    def test_legacy_contracts_normalize_to_control_required_v3(self):
        finite = {
            "schema_version": 1,
            "claim_type": "diagnostic",
            "candidate_universe": ["official"],
            "required_coverage": {
                "candidate_ids": ["official"],
                "minimum_evaluated": 1,
                "minimum_per_group": None,
            },
            "round_budget": 1,
            "comparison_groups": None,
        }
        open_world = {
            **finite,
            "schema_version": 2,
            "candidate_universe_closed": False,
            "existential_witness_outcome": None,
        }

        for legacy in (finite, open_world):
            normalized = validate_query_sufficiency_contract(legacy)
            self.assertEqual(normalized["schema_version"], 3)
            self.assertEqual(
                normalized["control_requirement"], "required"
            )


if __name__ == "__main__":
    unittest.main()
