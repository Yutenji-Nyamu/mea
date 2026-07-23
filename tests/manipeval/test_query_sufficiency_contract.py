import unittest
import json
import tempfile
from pathlib import Path

from mea.planner import (
    BoundTaskPlanSession,
    PlanSessionError,
    QuerySufficiencyError,
    assess_query_sufficiency,
    build_act_catalog,
    build_query_sufficiency_contract,
    infer_claim_type,
)


def evidence(candidate_id, outcome, *, score=None, diagnosis=None):
    return {
        "candidate_id": candidate_id,
        "outcome": outcome,
        "score": score,
        "diagnosis": diagnosis,
    }


class QuerySufficiencyTruthTableTests(unittest.TestCase):
    CANDIDATES = ["left", "right"]

    def contract(self, claim_type, **kwargs):
        return build_query_sufficiency_contract(
            "bounded query",
            candidate_universe=self.CANDIDATES,
            required_candidate_ids=self.CANDIDATES,
            claim_type=claim_type,
            round_budget=kwargs.pop("round_budget", 2),
            **kwargs,
        )

    def test_claim_type_inference_is_conservative(self):
        self.assertEqual(infer_claim_type("Does every position pass?"), "universal")
        self.assertEqual(
            infer_claim_type("Is there at least one working position?"),
            "existential",
        )
        self.assertEqual(
            infer_claim_type("Compare the left and right positions."),
            "comparative",
        )
        self.assertEqual(
            infer_claim_type("How well does it generalize across properties?"),
            "diagnostic",
        )
        self.assertEqual(
            infer_claim_type("比较左侧与右侧位置的表现。"),
            "comparative",
        )
        self.assertEqual(
            infer_claim_type("所有候选条件是否都能成功？"),
            "universal",
        )
        self.assertEqual(
            infer_claim_type("是否至少有一个可行位置？"),
            "existential",
        )

    def test_universal_success_needs_all_but_one_failure_refutes(self):
        contract = self.contract("universal")
        first_pass = assess_query_sufficiency(
            contract, [evidence("left", "pass")]
        )
        self.assertFalse(first_pass["should_stop"])
        self.assertEqual(first_pass["stop_reason"], "continue")
        self.assertEqual(first_pass["recommended_candidate_ids"], ["right"])

        all_pass = assess_query_sufficiency(
            contract,
            [evidence("left", "pass"), evidence("right", "pass")],
        )
        self.assertTrue(all_pass["evidence_sufficient"])
        self.assertEqual(all_pass["claim_verdict"], "supported")
        self.assertEqual(all_pass["stop_reason"], "evidence_sufficient")

        one_failure = assess_query_sufficiency(
            contract, [evidence("left", "fail")]
        )
        self.assertTrue(one_failure["should_stop"])
        self.assertEqual(one_failure["claim_verdict"], "refuted")
        self.assertEqual(one_failure["recommended_candidate_ids"], [])
        self.assertEqual(
            one_failure["untested_required_candidate_ids"],
            ["right"],
        )

    def test_existential_success_is_a_witness_but_refutation_needs_all(self):
        contract = self.contract("existential")
        one_failure = assess_query_sufficiency(
            contract, [evidence("left", "fail")]
        )
        self.assertFalse(one_failure["should_stop"])
        witness = assess_query_sufficiency(
            contract, [evidence("left", "pass")]
        )
        self.assertTrue(witness["evidence_sufficient"])
        self.assertEqual(witness["claim_verdict"], "supported")
        all_failed = assess_query_sufficiency(
            contract,
            [evidence("left", "fail"), evidence("right", "fail")],
        )
        self.assertEqual(all_failed["claim_verdict"], "refuted")
        self.assertEqual(all_failed["stop_reason"], "evidence_sufficient")

    def test_budget_exhaustion_is_not_evidence_sufficiency(self):
        contract = self.contract("universal", round_budget=1)
        result = assess_query_sufficiency(
            contract, [evidence("left", "pass")]
        )
        self.assertTrue(result["should_stop"])
        self.assertFalse(result["evidence_sufficient"])
        self.assertEqual(result["claim_verdict"], "inconclusive")
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        self.assertEqual(result["recommended_candidate_ids"], [])
        self.assertEqual(
            result["untested_required_candidate_ids"],
            ["right"],
        )

    def test_comparison_requires_both_preregistered_groups(self):
        contract = self.contract(
            "comparative",
            minimum_evaluated=2,
            minimum_per_group=1,
            comparison_groups={"baseline": ["left"], "candidate": ["right"]},
        )
        first_only = assess_query_sufficiency(
            contract, [evidence("left", "pass", score=0.75)]
        )
        self.assertFalse(first_only["should_stop"])
        result = assess_query_sufficiency(
            contract,
            [
                evidence("left", "pass", score=0.75),
                evidence("right", "pass", score=1.0),
            ],
        )
        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(
            result["claim_verdict"], "candidate_higher_observed"
        )
        self.assertEqual(
            result["statistics"]["comparison_groups"]["candidate"]["evaluated"],
            1,
        )
        self.assertTrue(
            any("metric, unit, direction" in item for item in result["limitations"])
        )

    def test_diagnostic_needs_a_diagnosis_or_full_no_failure_coverage(self):
        contract = self.contract("diagnostic", minimum_evaluated=1)
        undiagnosed = assess_query_sufficiency(
            contract, [evidence("left", "fail")]
        )
        self.assertFalse(undiagnosed["should_stop"])
        self.assertEqual(undiagnosed["recommended_candidate_ids"], ["right", "left"])

        diagnosed = assess_query_sufficiency(
            contract,
            [
                evidence(
                    "left",
                    "fail",
                    diagnosis="bell was outside the reachable workspace",
                )
            ],
        )
        self.assertTrue(diagnosed["evidence_sufficient"])
        self.assertEqual(diagnosed["claim_verdict"], "diagnosed")
        self.assertTrue(
            any("validate causality" in item for item in diagnosed["limitations"])
        )

        no_failure = assess_query_sufficiency(
            contract,
            [evidence("left", "pass"), evidence("right", "pass")],
        )
        self.assertTrue(no_failure["evidence_sufficient"])
        self.assertEqual(no_failure["claim_verdict"], "no_failure_observed")

    def test_conflicting_repeats_fail_closed(self):
        contract = self.contract("existential")
        result = assess_query_sufficiency(
            contract,
            [evidence("left", "pass"), evidence("left", "fail")],
        )
        self.assertFalse(result["evidence_sufficient"])
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        self.assertEqual(result["conflict_candidate_ids"], ["left"])

    def test_contract_rejects_invalid_comparison_partition(self):
        with self.assertRaises(QuerySufficiencyError):
            self.contract(
                "comparative",
                comparison_groups={"a": ["left"], "b": ["left"]},
                minimum_per_group=1,
            )


class PlanSessionQueryContractTests(unittest.TestCase):
    def test_session_binds_candidate_universe_and_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            schema = root / "mea/toolkit/schemas/click_bell.json"
            schema.parent.mkdir(parents=True)
            schema.write_text(
                json.dumps(
                    {"task_name": "click_bell", "task_family": "manipulation"}
                ),
                encoding="utf-8",
            )
            checkpoint = (
                root
                / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50"
            )
            checkpoint.mkdir(parents=True)
            (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")
            (checkpoint / "policy_last.ckpt").write_bytes(b"weights")
            catalog = build_act_catalog(root)
            session = BoundTaskPlanSession.from_catalog(
                catalog, "click_bell", max_rounds=2
            )
            contract = build_query_sufficiency_contract(
                "Does any official instance work?",
                candidate_universe=[
                    "object_instance.base0",
                    "object_instance.base1",
                ],
                claim_type="existential",
                round_budget=2,
            )
            result = session.assess_query_sufficiency(
                contract,
                [evidence("object_instance.base0", "pass")],
            )
            self.assertEqual(result["stop_reason"], "evidence_sufficient")

            outside = build_query_sufficiency_contract(
                "Does any candidate work?",
                candidate_universe=["outside.template"],
                claim_type="existential",
                round_budget=1,
            )
            with self.assertRaisesRegex(PlanSessionError, "leaves the bound task"):
                session.assess_query_sufficiency(outside, [])


if __name__ == "__main__":
    unittest.main()
