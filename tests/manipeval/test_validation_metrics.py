import json
import math
import tempfile
import unittest
from pathlib import Path

from mea.execution_vqa import build_execution_vqa_query
from mea.validation import (
    ValidationError,
    aggregate_planner_cases,
    binary_auroc,
    score_cached_suite,
    validate_suite,
)


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def vqa_artifact(*, observed, confidence) -> dict:
    query = build_execution_vqa_query(
        task_name="click_bell",
        template_id="task_execution.official_baseline",
        sub_aspect="task_execution.official_baseline",
        tool_contract={"metric": "official_check_success"},
    )
    return {
        "schema_version": 1,
        "selection": {"selected_frames": [{"frame_id": "initial"}]},
        "query": query,
        "observation": {
            "phenomena": [
                {
                    "id": "bell_visibly_pressed",
                    "observed": observed,
                    "description": "fixture",
                    "confidence": confidence,
                    "frame_ids": ["initial"],
                }
            ],
            "confidence": 0.8,
            "frame_ids": ["initial"],
            "numeric_consistency": "consistent",
            "conflicts": [],
            "evidence_conflict": False,
        },
    }


def planner_case(case_id: str, path: str, *, kind=None, model_generated=True) -> dict:
    prediction = {"path": path, "model_generated": model_generated}
    if kind is not None:
        prediction["planner_kind"] = kind
    return {
        "id": case_id,
        "gold": {
            "requested_template_ids": ["a", "b"],
            "acceptable_first_template_ids": ["a"],
        },
        "prediction": prediction,
    }


def vqa_case(case_id: str, path: str, gold: bool, source="human") -> dict:
    return {
        "id": case_id,
        "phenomenon_id": "bell_visibly_pressed",
        "gold": {"observed": gold, "label_source": source},
        "prediction": {"path": path},
    }


class ValidationMetricsTests(unittest.TestCase):
    def test_auroc_perfect_reversed_tied_single_class_and_invalid(self):
        self.assertEqual(binary_auroc([True, False], [0.9, 0.1])["value"], 1.0)
        self.assertEqual(binary_auroc([True, False], [0.1, 0.9])["value"], 0.0)
        self.assertEqual(binary_auroc([True, False], [0.5, 0.5])["value"], 0.5)
        self.assertEqual(
            binary_auroc([True], [0.9])["unavailable_reason"], "single_class"
        )
        with self.assertRaises(ValidationError):
            binary_auroc([True], [])
        with self.assertRaises(ValidationError):
            binary_auroc([1], [0.5])
        with self.assertRaises(ValidationError):
            binary_auroc([True], [math.nan])

    def test_cached_suite_scores_three_planner_and_vqa_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "artifacts/plan.json"
            write_json(
                plan_path,
                {
                    "planner": {"kind": "bounded_catalog_vlm"},
                    "plan": {
                        "requested_template_ids": ["a", "b"],
                        "rounds": [{"template_id": "a"}],
                    },
                },
            )
            positive_path = root / "artifacts/vqa_positive.json"
            negative_path = root / "artifacts/vqa_negative.json"
            write_json(positive_path, vqa_artifact(observed=True, confidence=0.9))
            write_json(negative_path, vqa_artifact(observed=False, confidence=0.8))
            plan_rel = str(plan_path.relative_to(root))
            positive_rel = str(positive_path.relative_to(root))
            negative_rel = str(negative_path.relative_to(root))
            suite = {
                "schema_version": 1,
                "suite_id": "fixture",
                "planner_cases": [
                    planner_case(f"p{index}", plan_rel) for index in range(1, 4)
                ],
                "vqa_cases": [
                    vqa_case("v1", positive_rel, True, "human"),
                    vqa_case("v2", negative_rel, False, "human"),
                    vqa_case("v3", positive_rel, True, "simulator_proxy"),
                ],
            }
            result = score_cached_suite(root, suite, budget=3, target="both")
            self.assertEqual(
                result["planner"]["metrics"]["template_micro_precision"], 1.0
            )
            self.assertEqual(result["vqa"]["metrics"]["accuracy_strict"], 1.0)
            self.assertEqual(result["vqa"]["metrics"]["auroc"]["value"], 1.0)
            self.assertAlmostEqual(
                result["vqa"]["cases"][1]["positive_score"], 0.2
            )
            self.assertEqual(
                result["vqa"]["metrics"]["label_source_counts"],
                {"human": 2, "simulator_proxy": 1},
            )
            self.assertFalse(result["provider_called"])

    def test_budget_one_reports_single_class_auroc_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "vqa.json", vqa_artifact(observed=True, confidence=0.7))
            result = score_cached_suite(
                root,
                {
                    "schema_version": 1,
                    "suite_id": "one",
                    "planner_cases": [],
                    "vqa_cases": [vqa_case("v", "vqa.json", True)],
                },
                budget=1,
                target="vqa",
            )
            auroc = result["vqa"]["metrics"]["auroc"]
            self.assertIsNone(auroc["value"])
            self.assertEqual(auroc["unavailable_reason"], "single_class")

    def test_invalid_vqa_bool_and_confidence_are_case_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, observed, confidence in (
                ("bad_observed", 1, 0.8),
                ("bad_confidence", True, True),
            ):
                write_json(
                    root / f"{name}.json",
                    vqa_artifact(observed=observed, confidence=confidence),
                )
                result = score_cached_suite(
                    root,
                    {
                        "schema_version": 1,
                        "suite_id": name,
                        "planner_cases": [],
                        "vqa_cases": [vqa_case(name, f"{name}.json", True)],
                    },
                    budget=1,
                    target="vqa",
                )
                self.assertFalse(result["vqa"]["cases"][0]["schema_valid"])
                self.assertEqual(result["vqa"]["metrics"]["coverage"], 0.0)

    def test_deterministic_planner_cannot_be_forced_into_model_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "plan.json",
                {
                    "planner": {"kind": "deterministic_official"},
                    "plan": {
                        "requested_template_ids": ["a", "b"],
                        "rounds": [{"template_id": "a"}],
                    },
                },
            )
            result = score_cached_suite(
                root,
                {
                    "schema_version": 1,
                    "suite_id": "deterministic",
                    "planner_cases": [planner_case("p", "plan.json")],
                    "vqa_cases": [],
                },
                budget=1,
                target="planner",
            )
            self.assertEqual(result["planner"]["metrics"]["case_count"], 0)
            self.assertEqual(
                result["planner"]["cases"][0]["exclusion_reason"],
                "deterministic_planner",
            )

    def test_suite_validation_and_budget_are_strict(self):
        with self.assertRaises(ValidationError):
            validate_suite({"schema_version": 1, "suite_id": "missing"})
        duplicate = {
            "schema_version": 1,
            "suite_id": "duplicate",
            "planner_cases": [planner_case("same", "x.json")],
            "vqa_cases": [vqa_case("same", "x.json", True)],
        }
        with self.assertRaises(ValidationError):
            validate_suite(duplicate)
        empty_first = {
            "schema_version": 1,
            "suite_id": "empty-first",
            "planner_cases": [planner_case("p", "x.json")],
            "vqa_cases": [],
        }
        empty_first["planner_cases"][0]["gold"]["acceptable_first_template_ids"] = []
        with self.assertRaises(ValidationError):
            validate_suite(empty_first)
        outside_requested = {
            "schema_version": 1,
            "suite_id": "outside-requested",
            "planner_cases": [planner_case("p", "x.json")],
            "vqa_cases": [],
        }
        outside_requested["planner_cases"][0]["gold"][
            "acceptable_first_template_ids"
        ] = ["c"]
        with self.assertRaises(ValidationError):
            validate_suite(outside_requested)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValidationError):
                score_cached_suite(
                    Path(temporary),
                    {
                        "schema_version": 1,
                        "suite_id": "short",
                        "planner_cases": [],
                        "vqa_cases": [vqa_case("v", "missing.json", True)],
                    },
                    budget=3,
                    target="vqa",
                )

    def test_planner_first_template_must_be_in_predicted_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "plan.json",
                {
                    "planner": {"kind": "bounded_catalog_vlm"},
                    "plan": {
                        "requested_template_ids": ["a", "b"],
                        "rounds": [{"template_id": "c"}],
                    },
                },
            )
            result = score_cached_suite(
                root,
                {
                    "schema_version": 1,
                    "suite_id": "first-not-requested",
                    "planner_cases": [planner_case("p", "plan.json")],
                    "vqa_cases": [],
                },
                budget=1,
                target="planner",
            )
            case = result["planner"]["cases"][0]
            self.assertFalse(case["schema_valid"])
            self.assertIn("must be in requested_template_ids", case["error"])

    def test_planner_f1_is_zero_when_precision_and_recall_are_zero(self):
        metrics = aggregate_planner_cases(
            [
                {
                    "eligible_for_model_metric": True,
                    "true_positive": 0,
                    "false_positive": 1,
                    "false_negative": 1,
                    "schema_valid": True,
                    "error": None,
                    "exact_set_match": False,
                    "first_template_match": False,
                }
            ]
        )
        self.assertEqual(metrics["template_micro_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
