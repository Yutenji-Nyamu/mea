import unittest

from experiments.paper.manipeval_annotation_replacement import (
    AnnotationReplacementError,
    score_annotation_replacement,
    validate_annotation_packet,
)


def packet(label_kind, ids=("a", "b")):
    items = [
        {
            "id": item_id,
            "paper_category": "generalization",
            "condition": "clean" if index == 0 else "lighting",
        }
        for index, item_id in enumerate(ids)
    ]
    return {
        "schema_version": 1,
        "protocol": "human_replaceable_annotation_packet_v1",
        "packet_id": f"packet_{label_kind}",
        "label_kind": label_kind,
        "blinding": {
            "prediction_values_hidden": True,
            "reference_labels_hidden": True,
        },
        "aggregation": {
            "primary_human_rater_count": 4,
            "majority_rule": "strict_majority_3_of_4",
            "tie_break": "senior_robotics_annotator",
        },
        "human_rater_slots": [
            {
                "rater_id": f"human_{index}",
                "kind": "human",
                "role": "primary_annotator",
                "labels": None,
            }
            for index in range(1, 5)
        ],
        "senior_tiebreaker_slot": {
            "rater_id": "senior",
            "kind": "human",
            "role": "senior_tiebreaker",
            "labels": None,
        },
        "replacement_contract": {
            "replace_only_annotation_file": True,
        },
        "items": items,
    }


def predictions(label_kind):
    items = (
        [
            {
                "id": "a",
                "schema_valid": True,
                "predicted_sub_aspect_ids": ["x"],
            },
            {
                "id": "b",
                "schema_valid": True,
                "predicted_sub_aspect_ids": ["z"],
            },
        ]
        if label_kind == "plan_sub_aspect"
        else [
            {
                "id": "a",
                "predicted_observed": True,
                "positive_score": 0.9,
            },
            {
                "id": "b",
                "predicted_observed": None,
                "positive_score": 0.1,
            },
        ]
    )
    return {
        "schema_version": 1,
        "protocol": "frozen_annotation_predictions_v1",
        "packet_id": f"packet_{label_kind}",
        "label_kind": label_kind,
        "prediction_source": "fixture",
        "items": items,
    }


def proxy_annotations(label_kind):
    values = (
        {"a": ["x"], "b": ["y"]}
        if label_kind == "plan_sub_aspect"
        else {"a": True, "b": False}
    )
    return {
        "schema_version": 1,
        "protocol": "human_replaceable_annotations_v1",
        "packet_id": f"packet_{label_kind}",
        "label_kind": label_kind,
        "annotation_set_id": f"proxy_{label_kind}",
        "annotation_source": "codex_development_agent_proxy",
        "raters": [
            {
                "rater_id": "codex_development_agent_proxy",
                "kind": "development_agent_proxy",
                "role": "proxy",
                "labels": [
                    {"id": item_id, "value": value}
                    for item_id, value in values.items()
                ],
            }
        ],
    }


class AnnotationReplacementTests(unittest.TestCase):
    def test_blind_packet_requires_four_empty_human_slots(self):
        value = packet("plan_sub_aspect")
        self.assertEqual(
            validate_annotation_packet(value)["packet_id"],
            "packet_plan_sub_aspect",
        )
        value["human_rater_slots"][0]["labels"] = []
        with self.assertRaisesRegex(
            AnnotationReplacementError, "must remain empty"
        ):
            validate_annotation_packet(value)

    def test_plan_proxy_reports_table6_precision_without_human_claim(self):
        result = score_annotation_replacement(
            packet("plan_sub_aspect"),
            predictions("plan_sub_aspect"),
            proxy_annotations("plan_sub_aspect"),
        )
        self.assertEqual(result["human_reviewer_count"], 0)
        self.assertFalse(result["paper_eligible"])
        self.assertEqual(
            result["metrics"]["table6_sub_aspect_precision"], 0.5
        )
        self.assertEqual(result["metrics"]["sub_aspect_micro_recall"], 0.5)
        self.assertEqual(
            result["metrics"]["sub_aspect_exact_set_accuracy"], 0.5
        )

    def test_plan_schema_failure_can_have_no_predicted_aspects(self):
        value = predictions("plan_sub_aspect")
        value["items"][1]["schema_valid"] = False
        value["items"][1]["predicted_sub_aspect_ids"] = []
        result = score_annotation_replacement(
            packet("plan_sub_aspect"),
            value,
            proxy_annotations("plan_sub_aspect"),
        )
        self.assertEqual(result["metrics"]["table6_sub_aspect_precision"], 1.0)

    def test_vqa_proxy_reuses_null_prediction_as_incorrect_and_scores_auroc(self):
        result = score_annotation_replacement(
            packet("vqa_binary"),
            predictions("vqa_binary"),
            proxy_annotations("vqa_binary"),
        )
        self.assertEqual(result["metrics"]["count"], 2)
        self.assertEqual(result["metrics"]["covered"], 1)
        self.assertEqual(result["metrics"]["accuracy"], 0.5)
        self.assertEqual(result["metrics"]["auroc"]["value"], 1.0)
        self.assertEqual(result["calls_started_by_scorer"]["provider"], 0)

    def test_four_human_majority_and_senior_tie_replace_only_annotations(self):
        values = [
            {"a": ["x"], "b": ["z"]},
            {"a": ["x"], "b": ["z"]},
            {"a": ["x", "y"], "b": ["y"]},
            {"a": ["y"], "b": ["y"]},
        ]
        annotations = {
            "schema_version": 1,
            "protocol": "human_replaceable_annotations_v1",
            "packet_id": "packet_plan_sub_aspect",
            "label_kind": "plan_sub_aspect",
            "annotation_set_id": "human_fixture",
            "annotation_source": "independent_human_annotation",
            "raters": [
                {
                    "rater_id": f"human_{index + 1}",
                    "kind": "human",
                    "role": "primary_annotator",
                    "labels": [
                        {"id": item_id, "value": value}
                        for item_id, value in labels.items()
                    ],
                }
                for index, labels in enumerate(values)
            ]
            + [
                {
                    "rater_id": "senior",
                    "kind": "human",
                    "role": "senior_tiebreaker",
                    "labels": [
                        {"id": "a", "value": ["x"]},
                        {"id": "b", "value": ["z"]},
                    ],
                }
            ],
        }
        result = score_annotation_replacement(
            packet("plan_sub_aspect"),
            predictions("plan_sub_aspect"),
            annotations,
        )
        self.assertEqual(result["human_reviewer_count"], 4)
        self.assertEqual(result["metrics"]["table6_sub_aspect_precision"], 1.0)
        self.assertEqual(result["metrics"]["sub_aspect_micro_recall"], 1.0)

    def test_ids_cannot_change_when_annotations_are_replaced(self):
        annotations = proxy_annotations("vqa_binary")
        annotations["raters"][0]["labels"].reverse()
        with self.assertRaisesRegex(
            AnnotationReplacementError, "order/IDs differ"
        ):
            score_annotation_replacement(
                packet("vqa_binary"),
                predictions("vqa_binary"),
                annotations,
            )

    def test_non_object_rater_fails_with_protocol_error(self):
        annotations = proxy_annotations("vqa_binary")
        annotations["raters"] = ["not-an-object"]
        with self.assertRaisesRegex(
            AnnotationReplacementError, "must contain objects"
        ):
            score_annotation_replacement(
                packet("vqa_binary"),
                predictions("vqa_binary"),
                annotations,
            )


if __name__ == "__main__":
    unittest.main()
