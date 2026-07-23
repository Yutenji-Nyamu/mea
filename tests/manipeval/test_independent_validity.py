import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from mea.independent_validity import (
    IndependentValidityError,
    build_synthetic_validity_demonstration,
    summarize_independent_validity,
    validate_independent_validity_study,
)


class IndependentValidityTests(unittest.TestCase):
    def test_synthetic_multirater_and_vqa_controls_are_honest(self):
        demo = build_synthetic_validity_demonstration()
        summary = demo["summary"]
        self.assertEqual(summary["rater_count"], 3)
        self.assertEqual(summary["agreement"]["agreeing_pairs"], 10)
        self.assertEqual(summary["agreement"]["total_pairs"], 12)
        self.assertEqual(
            summary["agreement"]["disagreement_item_ids"],
            ["target_selected"],
        )
        self.assertEqual(summary["development_agent_annotation_count"], 4)
        self.assertFalse(
            summary["human_gold_status"]["available_by_declared_manifest"]
        )
        self.assertFalse(
            summary["human_gold_status"]["development_agent_is_human_gold"]
        )
        controls = summary["vqa_control_evaluation"]
        self.assertEqual(
            controls["by_polarity"]["positive_control"]["value"], 1.0
        )
        self.assertEqual(
            controls["by_polarity"]["negative_control"]["value"], 0.75
        )
        self.assertEqual(controls["overall"]["value"], 0.875)
        self.assertEqual(controls["auroc"]["value"], 0.875)
        self.assertEqual(
            set(controls["by_paper_condition"]),
            {"clean", "scene_clutter", "background_texture", "lighting"},
        )
        self.assertFalse(summary["paper_reference_configuration_met"])
        self.assertEqual(
            summary["paper_reference_configuration"]["sources"],
            {
                "vqa_perturbations_and_metrics": "paper_appendix_A.2.4",
                "four_annotators_and_tie_break": "paper_appendix_A.4.3",
            },
        )
        self.assertIn(
            "synthetic_fixture_cannot_meet_paper_annotation_target",
            summary["paper_reference_unmet"],
        )

    def test_four_humans_and_senior_tie_break_are_supported_but_self_attested(self):
        study = deepcopy(build_synthetic_validity_demonstration()["study"])
        study["study_id"] = "declared_human_import"
        study["evidence_source"] = "live_annotation"
        study["raters"] = [
            {
                "rater_id": f"human_{index}",
                "kind": "human",
                "role": "primary_annotator",
            }
            for index in range(4)
        ] + [
            {
                "rater_id": "senior",
                "kind": "human",
                "role": "senior_tiebreaker",
            }
        ]
        references = {
            "bell_pressed": True,
            "target_selected": False,
            "return_motion": True,
            "lighting_visibility": True,
        }
        annotations = []
        for item_id, reference in references.items():
            labels = [reference] * 4
            if item_id == "target_selected":
                labels = [False, False, True, True]
            for index, observed in enumerate(labels):
                annotations.append(
                    {
                        "item_id": item_id,
                        "rater_id": f"human_{index}",
                        "observed": observed,
                    }
                )
        annotations.append(
            {
                "item_id": "target_selected",
                "rater_id": "senior",
                "observed": False,
            }
        )
        study["annotations"] = annotations
        summary = summarize_independent_validity(study)
        target = next(
            row
            for row in summary["item_references"]
            if row["item_id"] == "target_selected"
        )
        self.assertEqual(
            target["reference_source"], "declared_human_senior_tiebreak"
        )
        self.assertFalse(target["reference_observed"])
        self.assertTrue(
            summary["human_gold_status"]["available_by_declared_manifest"]
        )
        self.assertFalse(
            summary["human_gold_status"][
                "independently_verified_by_this_aggregator"
            ]
        )
        self.assertTrue(summary["paper_reference_configuration_met"])
        self.assertEqual(summary["paper_reference_unmet"], [])
        self.assertFalse(summary["paper_table_eligible"])

    def test_duplicate_annotations_missing_controls_and_threshold_mismatch_fail(self):
        study = deepcopy(build_synthetic_validity_demonstration()["study"])
        duplicate = deepcopy(study)
        duplicate["annotations"].append(deepcopy(duplicate["annotations"][0]))
        with self.assertRaisesRegex(
            IndependentValidityError, "duplicate annotation"
        ):
            validate_independent_validity_study(duplicate)

        missing = deepcopy(study)
        missing["vqa_controls"].pop()
        with self.assertRaisesRegex(
            IndependentValidityError, "one positive and one negative"
        ):
            validate_independent_validity_study(missing)

        mismatch = deepcopy(study)
        mismatch["vqa_controls"][0]["positive_score"] = (
            0.1
            if mismatch["vqa_controls"][0]["predicted_observed"]
            else 0.9
        )
        with self.assertRaisesRegex(
            IndependentValidityError, "fixed_threshold"
        ):
            validate_independent_validity_study(mismatch)

    def test_cli_outputs_synthetic_study_and_summary(self):
        from scripts import manipeval_independent_validity as cli

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "validity.json"
            with patch.object(
                sys,
                "argv",
                [
                    "manipeval_independent_validity.py",
                    "--synthetic-demo",
                    "--output",
                    str(output),
                ],
            ), redirect_stdout(io.StringIO()):
                cli.main()
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                value["summary"]["calls_started_by_summary"],
                {"provider": 0, "simulator": 0, "act": 0},
            )
            self.assertFalse(
                value["summary"]["empirical_validity_claim_eligible"]
            )


if __name__ == "__main__":
    unittest.main()
