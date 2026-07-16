import json
import tempfile
import unittest
from pathlib import Path

from mea.toolkit.aggregate import (
    AggregateToolkitError,
    aggregate_tool_executions,
)


def result(tool, value, *, unit=None, passed=None, reason=None, step=None):
    details = {}
    if reason is not None:
        details["reason"] = reason
    payload = {
        "tool": tool,
        "value": value,
        "unit": unit,
        "evidence_steps": [] if step is None else [step],
        "details": details,
    }
    if passed is not None:
        payload["passed"] = passed
    return payload


def execution(metric, episodes):
    return {
        "schema_version": 1,
        "status": "passed",
        "route": "reuse",
        "tool_spec": {"metric": metric},
        "episodes": episodes,
    }


def episode(
    value,
    *,
    metric,
    seed,
    role="policy_under_evaluation",
    policy_name="ACT",
    unit=None,
    passed=None,
    reason=None,
    step=None,
):
    return {
        "episode_dir": f"telemetry/{role}/episode_seed_{seed}",
        "policy_name": policy_name,
        "seed": seed,
        "role": role,
        "result": result(
            metric,
            value,
            unit=unit,
            passed=passed,
            reason=reason,
            step=step,
        ),
    }


def cohort(aggregate, metric, role):
    metric_result = next(
        item for item in aggregate["metrics"] if item["metric"] == metric
    )
    return next(
        item for item in metric_result["cohorts"] if item["role"] == role
    )


class AggregateToolkitTests(unittest.TestCase):
    def test_boolean_rate_groups_and_policy_expert_isolation(self):
        metric = "hammer_block_contact_ever"
        policy = [
            episode(value, metric=metric, seed=seed, step=10 + seed)
            for seed, value in zip((1, 2, 3, 4), (False, True, True, False))
        ]
        expert = [
            episode(
                True,
                metric=metric,
                seed=1,
                role="expert_validation",
                policy_name="expert",
                step=99,
            )
        ]
        aggregate = aggregate_tool_executions(
            [
                {
                    "tool_execution": execution(metric, policy + expert),
                    "context": {
                        "round_id": "round_1",
                        "variant": "blue_block",
                        "source_artifact": "round_1/tool_execution.json",
                    },
                }
            ]
        )

        act = cohort(aggregate, metric, "policy_under_evaluation")
        validation = cohort(aggregate, metric, "expert_validation")
        self.assertEqual(
            act["summary"]["statistics"]["true_count"]["value"], 2
        )
        self.assertEqual(
            act["summary"]["statistics"]["true_rate"]["value"], 0.5
        )
        self.assertEqual(
            act["summary"]["statistics"]["true_rate"]["denominator"], 4
        )
        self.assertEqual(
            validation["summary"]["statistics"]["true_rate"]["value"], 1.0
        )
        self.assertEqual(len(act["groups"]["seed"]), 4)
        self.assertEqual(act["groups"]["round_id"][0]["value"], "round_1")
        self.assertEqual(act["groups"]["variant"][0]["value"], "blue_block")
        provenance = act["summary"]["statistics"]["true_rate"]["provenance"]
        self.assertEqual(len(provenance), 4)
        self.assertTrue(all(item["evidence_steps"] for item in provenance))
        self.assertTrue(
            all(item["role"] == "policy_under_evaluation" for item in provenance)
        )

    def test_numeric_statistics_exclude_missing_invalid_and_expert(self):
        metric = "pickup_to_first_contact_time"
        policy = [
            episode(1.0, metric=metric, seed=1, unit="s", step=10),
            episode(3.0, metric=metric, seed=2, unit="s", step=20),
            episode(
                None,
                metric=metric,
                seed=3,
                unit="s",
                reason="contact_not_observed_after_pickup",
                step=30,
            ),
            episode(
                None,
                metric=metric,
                seed=4,
                unit="s",
                reason="contact_precedes_pickup",
                step=40,
            ),
        ]
        expert = [
            episode(
                100.0,
                metric=metric,
                seed=1,
                unit="s",
                role="expert_validation",
                policy_name="expert",
                step=100,
            )
        ]
        aggregate = aggregate_tool_executions(
            [execution(metric, policy + expert)]
        )

        act = cohort(aggregate, metric, "policy_under_evaluation")["summary"]
        quality = act["quality"]
        self.assertEqual(quality["valid"]["value"], 2)
        self.assertEqual(quality["missing"]["value"], 1)
        self.assertEqual(quality["invalid"]["value"], 1)
        self.assertEqual(act["statistics"]["mean"]["value"], 2.0)
        self.assertEqual(act["statistics"]["median"]["value"], 2.0)
        self.assertEqual(act["statistics"]["min"]["value"], 1.0)
        self.assertEqual(act["statistics"]["max"]["value"], 3.0)
        self.assertEqual(
            act["statistics"]["population_stddev"]["value"], 1.0
        )
        self.assertEqual(len(act["statistics"]["mean"]["provenance"]), 2)
        invalid = quality["invalid"]["provenance"]
        self.assertEqual(invalid[0]["invalid_reason"], "contact_precedes_pickup")
        expert_summary = cohort(
            aggregate, metric, "expert_validation"
        )["summary"]
        self.assertEqual(expert_summary["statistics"]["mean"]["value"], 100.0)

    def test_passed_predicate_provides_pickup_rate(self):
        metric = "hammer_pickup_height"
        rows = [
            episode(0.01, metric=metric, seed=1, unit="m", passed=False),
            episode(0.06, metric=metric, seed=2, unit="m", passed=True),
            episode(None, metric=metric, seed=3, unit="m", passed=False),
        ]
        aggregate = aggregate_tool_executions([execution(metric, rows)])
        act = cohort(aggregate, metric, "policy_under_evaluation")

        self.assertEqual(act["summary"]["quality"]["missing"]["value"], 1)
        passed = act["passed_summary"]
        self.assertEqual(passed["quality"]["valid"]["value"], 3)
        self.assertEqual(passed["statistics"]["true_count"]["value"], 1)
        self.assertAlmostEqual(
            passed["statistics"]["true_rate"]["value"], 1 / 3
        )

    def test_official_success_exposes_success_aliases(self):
        metric = "official_check_success"
        rows = [
            episode(True, metric=metric, seed=1),
            episode(False, metric=metric, seed=2),
            episode(True, metric=metric, seed=3),
        ]
        aggregate = aggregate_tool_executions([execution(metric, rows)])
        stats = cohort(
            aggregate, metric, "policy_under_evaluation"
        )["summary"]["statistics"]
        self.assertEqual(stats["success_count"]["value"], 2)
        self.assertAlmostEqual(stats["success_rate"]["value"], 2 / 3)

    def test_trusted_tool_summary_shape_is_accepted(self):
        source = {
            "schema_version": 1,
            "episodes": [
                {
                    "episode_dir": "act/episode_000_seed_1",
                    "metadata": {
                        "policy_name": "ACT",
                        "seed": 1,
                    },
                    "tool_results": [
                        result("official_check_success", True, step=7),
                        result("hammer_block_contact_ever", False, step=5),
                    ],
                }
            ],
        }
        aggregate = aggregate_tool_executions([source])
        self.assertEqual(
            [item["metric"] for item in aggregate["metrics"]],
            ["hammer_block_contact_ever", "official_check_success"],
        )
        self.assertEqual(aggregate["episode_result_count"], 2)
        self.assertEqual(aggregate["unique_episode_count"], 1)

    def test_policy_name_prevents_role_spoofing(self):
        metric = "pickup_to_first_contact_time"
        mislabeled_expert = episode(
            99.0,
            metric=metric,
            seed=1,
            unit="s",
            role="policy_under_evaluation",
            policy_name="expert",
        )
        aggregate = aggregate_tool_executions(
            [execution(metric, [mislabeled_expert])]
        )
        metric_result = aggregate["metrics"][0]
        self.assertEqual(
            [item["role"] for item in metric_result["cohorts"]],
            ["expert_validation"],
        )
        summary = metric_result["cohorts"][0]["summary"]
        self.assertEqual(summary["quality"]["valid"]["value"], 0)
        self.assertEqual(summary["quality"]["invalid"]["value"], 1)
        self.assertEqual(
            summary["quality"]["invalid"]["provenance"][0][
                "invalid_reason"
            ],
            "policy_name_role_mismatch",
        )

    def test_artifact_is_deterministic_and_sorted(self):
        metric = "hammer_block_contact_ever"
        first = execution(
            metric,
            [episode(False, metric=metric, seed=2, step=20)],
        )
        second = execution(
            metric,
            [episode(True, metric=metric, seed=1, step=10)],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_path = root / "left" / "aggregate_result.json"
            right_path = root / "right"
            left = aggregate_tool_executions(
                [first, second], output_path=left_path
            )
            right = aggregate_tool_executions(
                [second, first], output_path=right_path
            )
            self.assertEqual(left, right)
            self.assertEqual(
                left_path.read_bytes(),
                (right_path / "aggregate_result.json").read_bytes(),
            )
            parsed = json.loads(left_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed, left)

    def test_rejects_empty_and_non_sequence_sources(self):
        with self.assertRaisesRegex(AggregateToolkitError, "sequence"):
            aggregate_tool_executions("tool_execution.json")
        with self.assertRaisesRegex(AggregateToolkitError, "no episode"):
            aggregate_tool_executions([])


if __name__ == "__main__":
    unittest.main()
