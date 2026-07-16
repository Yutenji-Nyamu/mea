import copy
import unittest

from mea.paired import (
    PairedProtocolError,
    build_paired_summary,
    build_seed_manifest,
    validate_seed_manifest,
)


def measurement(
    seed,
    *,
    success=None,
    eligibility="passed",
    time_to_success=None,
):
    return {
        "seed": seed,
        "eligibility_status": eligibility,
        "policy_executed": success is not None,
        "policy_success": success,
        "time_to_success": time_to_success,
    }


def condition_run(rows):
    return {"seed_measurements": rows}


class SeedManifestTests(unittest.TestCase):
    def test_build_manifest_validates_and_preserves_seed_and_condition_order(self):
        manifest = build_seed_manifest(
            task_name="click_bell",
            seeds=[100403, 100401, 100402],
            conditions=(
                {"id": "easy", "task_config": "demo_clean"},
                {"id": "hard", "task_config": "demo_randomized"},
            ),
            checkpoint_setting="demo_clean",
            expert_data_num=25,
            policy_seed=7,
        )

        self.assertEqual(manifest["seeds"], [100403, 100401, 100402])
        self.assertEqual(
            [item["id"] for item in manifest["conditions"]],
            ["easy", "hard"],
        )
        self.assertEqual(manifest["expert_data_num"], 25)
        self.assertEqual(manifest["policy_seed"], 7)

    def test_manifest_rejects_duplicate_or_invalid_seeds(self):
        with self.assertRaisesRegex(PairedProtocolError, "duplicate seed: 3"):
            build_seed_manifest(task_name="click_bell", seeds=[3, 4, 3])
        for invalid in ([True], [-1], [1.5], []):
            with self.subTest(invalid=invalid):
                with self.assertRaises(PairedProtocolError):
                    build_seed_manifest(task_name="click_bell", seeds=invalid)

    def test_manifest_rejects_task_mismatch(self):
        manifest = build_seed_manifest(task_name="click_bell", seeds=[9])

        with self.assertRaisesRegex(
            PairedProtocolError,
            "does not match 'adjust_bottle'",
        ):
            validate_seed_manifest(
                manifest,
                expected_task_name="adjust_bottle",
            )


class PairedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.seeds = [11, 12, 13, 14]
        self.manifest = build_seed_manifest(
            task_name="click_bell",
            seeds=self.seeds,
        )

    def test_deterministic_tt_tf_ft_ff_statistics(self):
        runs = {
            "easy": condition_run(
                [
                    measurement(11, success=True, time_to_success=1.0),
                    measurement(12, success=True, time_to_success=2.0),
                    measurement(13, success=False),
                    measurement(14, success=False),
                ]
            ),
            "hard": condition_run(
                [
                    measurement(11, success=True, time_to_success=1.5),
                    measurement(12, success=False),
                    measurement(13, success=True),
                    measurement(14, success=False),
                ]
            ),
        }

        first = build_paired_summary(self.manifest, runs)
        second = build_paired_summary(
            self.manifest,
            {"hard": runs["hard"], "easy": runs["easy"]},
        )

        self.assertEqual(first, second)
        self.assertEqual(first["paired_eligible_count"], 4)
        self.assertEqual(first["paired_evaluated_count"], 4)
        self.assertEqual(first["paired_not_evaluated_count"], 0)
        self.assertTrue(first["valid_for_comparison"])
        self.assertEqual(first["coverage_rate"], 1.0)
        self.assertEqual(first["success"]["denominator"], 4)
        self.assertEqual(first["success"]["easy"], {"count": 2, "rate": 0.5})
        self.assertEqual(first["success"]["hard"], {"count": 2, "rate": 0.5})
        self.assertEqual(first["success"]["hard_minus_easy"], 0.0)
        self.assertEqual(first["success"]["easy_minus_hard"], 0.0)
        self.assertEqual(
            first["success"]["outcomes"],
            {
                "both_success": 1,
                "easy_only": 1,
                "hard_only": 1,
                "neither": 1,
            },
        )
        self.assertEqual(
            [pair["outcome"] for pair in first["pairs"]],
            ["both_success", "easy_only", "hard_only", "neither"],
        )
        self.assertEqual(first["time_to_success"]["paired_both_success_count"], 1)
        self.assertEqual(first["time_to_success"]["mean_easy"], 1.0)
        self.assertEqual(first["time_to_success"]["mean_hard"], 1.5)
        self.assertEqual(first["time_to_success"]["mean_hard_minus_easy"], 0.5)

    def test_ineligible_seed_is_excluded_from_paired_denominator(self):
        runs = {
            "easy": condition_run(
                [
                    measurement(11, success=True),
                    measurement(12, eligibility="unstable"),
                    measurement(13, success=False),
                    measurement(14, success=True),
                ]
            ),
            "hard": condition_run(
                [
                    measurement(11, success=True),
                    measurement(12, success=True),
                    measurement(13, success=False),
                    measurement(14, eligibility="expert_failed"),
                ]
            ),
        }

        summary = build_paired_summary(self.manifest, runs)

        self.assertEqual(summary["paired_eligible_count"], 2)
        self.assertEqual(summary["paired_evaluated_count"], 2)
        self.assertEqual(summary["paired_not_evaluated_count"], 2)
        self.assertTrue(summary["valid_for_comparison"])
        self.assertEqual(summary["coverage_rate"], 0.5)
        self.assertEqual(
            summary["eligibility_status_counts"]["easy"]["unstable"],
            1,
        )
        self.assertEqual(
            summary["eligibility_status_counts"]["hard"]["expert_failed"],
            1,
        )
        self.assertEqual(summary["success"]["denominator"], 2)
        self.assertEqual(summary["success"]["easy"]["rate"], 0.5)
        self.assertEqual(summary["success"]["hard"]["rate"], 0.5)
        self.assertFalse(summary["pairs"][1]["paired_evaluated"])
        self.assertFalse(summary["pairs"][3]["paired_evaluated"])
        self.assertIsNone(summary["pairs"][1]["outcome"])

    def test_rejects_missing_measurement(self):
        runs = self._all_success_runs()
        runs["hard"]["seed_measurements"].pop()

        with self.assertRaisesRegex(PairedProtocolError, "missing seeds: \[14\]"):
            build_paired_summary(self.manifest, runs)

    def test_rejects_duplicate_measurement(self):
        runs = self._all_success_runs()
        runs["easy"]["seed_measurements"][-1] = measurement(11, success=True)

        with self.assertRaisesRegex(PairedProtocolError, "duplicate seed 11"):
            build_paired_summary(self.manifest, runs)

    def test_rejects_unrequested_measurement(self):
        runs = self._all_success_runs()
        runs["easy"]["seed_measurements"][-1] = measurement(99, success=True)

        with self.assertRaisesRegex(PairedProtocolError, "unrequested seed 99"):
            build_paired_summary(self.manifest, runs)

    def test_rejects_executed_ineligible_or_failed_timing(self):
        runs = self._all_success_runs()
        runs["easy"]["seed_measurements"][0]["eligibility_status"] = "unstable"
        with self.assertRaisesRegex(PairedProtocolError, "cannot be counted"):
            build_paired_summary(self.manifest, runs)

        runs = self._all_success_runs()
        runs["easy"]["seed_measurements"][0].update(
            {"policy_success": False, "time_to_success": 1.0}
        )
        with self.assertRaisesRegex(PairedProtocolError, "successful policy"):
            build_paired_summary(self.manifest, runs)

    def test_protocol_violation_marks_summary_invalid(self):
        runs = self._all_success_runs()
        runs["hard"]["seed_measurements"][0] = measurement(
            11,
            eligibility="protocol_violation",
        )
        summary = build_paired_summary(self.manifest, runs)
        self.assertFalse(summary["valid_for_comparison"])
        self.assertEqual(summary["protocol_violation_measurement_count"], 1)

    def _all_success_runs(self):
        rows = [measurement(seed, success=True) for seed in self.seeds]
        return {
            "easy": condition_run(copy.deepcopy(rows)),
            "hard": condition_run(copy.deepcopy(rows)),
        }


if __name__ == "__main__":
    unittest.main()
