import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from mea.matched_efficiency_protocol import (
    ADAPTIVE_STRATEGY,
    FIXED_STRATEGY,
    MatchedEfficiencyError,
    build_matched_preregistration,
    build_synthetic_demonstrations,
    compare_matched_results,
    validate_matched_preregistration,
)
from mea.planner.query_contract import build_query_sufficiency_contract


def make_preregistration():
    contract = build_query_sufficiency_contract(
        "Does at least one candidate work?",
        candidate_universe=["candidate_a", "candidate_b"],
        required_candidate_ids=["candidate_a", "candidate_b"],
        claim_type="existential",
        round_budget=2,
    )
    return build_matched_preregistration(
        study_id="matched_test",
        query="Does the policy answer the original Query?",
        checkpoint={
            "checkpoint_id": "act_test",
            "artifact_sha256": "c" * 64,
        },
        candidate_suite=["candidate_a", "candidate_b"],
        seeds=[7],
        max_budget={
            "act_episode_starts": 2,
            "expert_starts": 2,
            "probe_starts": 2,
            "provider_retries": 2,
            "wall_seconds": 30.0,
        },
        sufficiency_contract=contract,
    )


def make_result(preregistration, *, fixed, candidates, starts):
    strategy = FIXED_STRATEGY if fixed else ADAPTIVE_STRATEGY
    return {
        "strategy": strategy,
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "status": "completed",
        "evidence_source": "synthetic_fixture",
        "observations": [
            {
                "candidate_id": candidate,
                "seed": 7,
                "evidence": {"synthetic": True},
            }
            for candidate in candidates
        ],
        "resource_usage": {
            "act_episode_starts": starts,
            "completed_policy_trials": len(candidates),
            "policy_steps": 10 * len(candidates),
            "expert_starts": starts,
            "probe_starts": starts,
            "provider_logical_calls": starts,
            "provider_transport_attempts": starts + 1,
            "wall_seconds": 4.0 * starts,
        },
        "stopping": {
            "reason": "fixed_suite_complete" if fixed else "sufficiency_reached",
            "sufficiency_met": True,
        },
        "conclusion": {
            "query_sha256": preregistration["arms"]["fixed"]["query_sha256"],
            "conclusion_key": "same_query_answer",
            "verdict": "supported_in_tested_scope",
            "answer": "Only the frozen tested scope is supported.",
            "limitations": ["Synthetic fixture."],
        },
    }


class MatchedEfficiencyProtocolTests(unittest.TestCase):
    def test_synthetic_demo_contains_saving_and_zero_saving_pairs(self):
        demo = build_synthetic_demonstrations()
        saving = demo["scenarios"]["adaptive_one_vs_fixed_two"]
        zero = demo["scenarios"]["adaptive_two_vs_fixed_two_zero_savings"]
        self.assertEqual(saving["act_start_savings"], 1)
        self.assertTrue(saving["adaptive_used_fewer_act_episode_starts"])
        self.assertTrue(saving["original_query_conclusion"]["agrees"])
        self.assertFalse(saving["empirical_policy_claim_eligible"])
        self.assertEqual(zero["act_start_savings"], 0)
        self.assertTrue(zero["zero_act_savings"])
        self.assertFalse(zero["efficiency_pattern_passed"])
        self.assertEqual(
            saving["calls_started_by_comparison"],
            {
                "provider": 0,
                "simulator": 0,
                "expert": 0,
                "probe": 0,
                "act": 0,
            },
        )
        self.assertFalse(saving["paper_reference_configuration_met"])
        self.assertIsNone(saving["paper_reported_sample_count"])

    def test_all_scientific_identity_fields_must_match(self):
        base = make_preregistration()
        changes = {
            "query": "Different Query",
            "checkpoint": {
                "checkpoint_id": "different",
                "artifact_sha256": "d" * 64,
            },
            "candidate_suite": ["candidate_a", "candidate_c"],
            "seeds": [9],
            "max_budget": {
                **base["arms"]["adaptive"]["max_budget"],
                "wall_seconds": 31.0,
            },
            "sufficiency_contract": build_query_sufficiency_contract(
                "Does every candidate work?",
                candidate_universe=["candidate_a", "candidate_b"],
                required_candidate_ids=["candidate_a", "candidate_b"],
                claim_type="universal",
                round_budget=2,
            ),
        }
        for field, replacement in changes.items():
            with self.subTest(field=field):
                bad = deepcopy(base)
                bad["arms"]["adaptive"][field] = replacement
                if field == "candidate_suite":
                    bad["arms"]["adaptive"]["sufficiency_contract"] = (
                        build_query_sufficiency_contract(
                            "Does at least one changed candidate work?",
                            candidate_universe=replacement,
                            required_candidate_ids=replacement,
                            claim_type="existential",
                            round_budget=2,
                        )
                    )
                with self.assertRaisesRegex(
                    MatchedEfficiencyError, "matched arms differ"
                ):
                    validate_matched_preregistration(bad)

    def test_result_budget_and_fixed_coverage_fail_closed(self):
        prereg = make_preregistration()
        fixed = make_result(
            prereg, fixed=True, candidates=["candidate_a", "candidate_b"], starts=2
        )
        adaptive = make_result(
            prereg, fixed=False, candidates=["candidate_a"], starts=1
        )
        summary = compare_matched_results(prereg, fixed, adaptive)
        self.assertEqual(
            summary["resource_comparison"]["completed_policy_trials"],
            {"fixed": 2, "adaptive": 1, "fixed_minus_adaptive": 1},
        )
        self.assertEqual(
            summary["resource_comparison"]["policy_steps"]["fixed_minus_adaptive"],
            10,
        )
        self.assertEqual(
            summary["sufficiency_contract_sha256"],
            prereg["arms"]["fixed"]["sufficiency_contract_sha256"],
        )

        incomplete = deepcopy(fixed)
        incomplete["observations"].pop()
        with self.assertRaisesRegex(
            MatchedEfficiencyError, "complete frozen suite"
        ):
            compare_matched_results(prereg, incomplete, adaptive)

        over_budget = deepcopy(adaptive)
        over_budget["resource_usage"]["provider_transport_attempts"] = 10
        with self.assertRaisesRegex(
            MatchedEfficiencyError, "provider_retries exceeds"
        ):
            compare_matched_results(prereg, fixed, over_budget)

    def test_rejects_parallel_or_misaligned_sufficiency_schema(self):
        base = make_preregistration()
        legacy = deepcopy(base)
        legacy["arms"]["fixed"]["sufficiency_contract"] = {
            "schema_version": 1,
            "contract_id": "legacy_parallel_contract",
            "claim_type": "existential",
        }
        with self.assertRaisesRegex(
            MatchedEfficiencyError, "valid QuerySufficiencyContract"
        ):
            validate_matched_preregistration(legacy)

        outside = deepcopy(base)
        outside["arms"]["fixed"]["sufficiency_contract"] = (
            build_query_sufficiency_contract(
                "Does any other candidate work?",
                candidate_universe=["candidate_a", "candidate_c"],
                claim_type="existential",
                round_budget=2,
            )
        )
        with self.assertRaisesRegex(
            MatchedEfficiencyError, "exactly match candidate_suite"
        ):
            validate_matched_preregistration(outside)

    def test_conclusion_must_bind_original_query(self):
        prereg = make_preregistration()
        fixed = make_result(
            prereg, fixed=True, candidates=["candidate_a", "candidate_b"], starts=2
        )
        adaptive = make_result(
            prereg, fixed=False, candidates=["candidate_a"], starts=1
        )
        adaptive["conclusion"]["query_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            MatchedEfficiencyError, "original frozen Query"
        ):
            compare_matched_results(prereg, fixed, adaptive)

    def test_cli_writes_both_synthetic_scenarios(self):
        from scripts import manipeval_matched_efficiency as cli

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "summary.json"
            with patch.object(
                sys,
                "argv",
                [
                    "manipeval_matched_efficiency.py",
                    "--synthetic-demo",
                    "--output",
                    str(output),
                ],
            ), redirect_stdout(io.StringIO()):
                cli.main()
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                set(value["scenarios"]),
                {
                    "adaptive_one_vs_fixed_two",
                    "adaptive_two_vs_fixed_two_zero_savings",
                },
            )
            self.assertFalse(value["empirical_policy_claim_eligible"])


if __name__ == "__main__":
    unittest.main()
