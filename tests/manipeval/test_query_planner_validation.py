import unittest

from mea.query_dataset import summarize_query_dataset
from mea.query_planner_validation import (
    aggregate_live_query_cases,
    score_live_query_case,
    validate_capability_snapshot,
    validate_live_query_budget,
)


ANNOTATION = {
    "source": "development_agent_proxy",
    "review_status": "proxy_reviewed",
    "annotator_id": "codex_development_agent",
    "human_votes": [],
    "paper_eligible": False,
}


def case(index, *, supported=True, task="click_bell", aspects=None):
    aspects = aspects or ["object_position"]
    return {
        "id": f"q{index:03d}",
        "query": f"query {index}",
        "setting": "single_task",
        "task_name": task,
        "task_profile": "adaptive_properties",
        "paper_category": "generalization_object",
        "gold_sub_aspect_ids": aspects,
        "acceptable_first_sub_aspect_ids": aspects,
        "capability_status": "supported" if supported else "unsupported",
        "annotation": dict(ANNOTATION),
    }


def dataset():
    return {
        "schema_version": 1,
        "dataset_id": "proxy",
        "annotation_status": "development_agent_proxy_reviewed",
        "annotation_protocol": {
            "role": "development_agent_proxy",
            "tested_agent": "runtime_global_plan_agent",
            "human_reviewer_count": 0,
            "paper_eligible": False,
            "replacement_required": "independent_human_majority_annotation",
        },
        "cases": [case(index) for index in range(1, 21)],
    }


class QueryPlannerValidationTests(unittest.TestCase):
    def test_budget_is_agile_and_requires_proxy_review(self):
        self.assertEqual(
            summarize_query_dataset(dataset())["paper_category_counts"],
            {"generalization_object": 20},
        )
        selected = validate_live_query_budget(dataset(), 5)
        self.assertEqual(
            [item["id"] for item in selected],
            ["q001", "q004", "q006", "q013", "q020"],
        )
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_live_query_budget(dataset(), 2)
        value = dataset()
        value["annotation_status"] = "model_draft_unreviewed"
        value.pop("annotation_protocol")
        for item in value["cases"]:
            item.pop("paper_category")
            item["annotation"] = {
                "source": "model_draft",
                "review_status": "unreviewed",
                "human_votes": [],
            }
        with self.assertRaisesRegex(ValueError, "proxy labels"):
            validate_live_query_budget(value, 1)

    def test_capability_snapshot_prevents_silent_catalog_drift(self):
        selected = [case(1)]
        catalog = {
            "tasks": [
                {
                    "task_name": "click_bell",
                    "aspects": [{"aspect_id": "object_position"}],
                }
            ]
        }
        validate_capability_snapshot(selected, catalog)
        selected[0]["capability_status"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_capability_snapshot(selected, catalog)

    def test_scores_supported_and_task_qualified_unsupported(self):
        supported = score_live_query_case(
            case(1),
            {
                "attempt_count": 1,
                "selection": {
                    "decision": "route",
                    "task_name": "click_bell",
                    "requested_aspect_ids": ["object_position"],
                    "first_aspect_id": "object_position",
                    "unsupported_capabilities": [],
                },
            },
        )
        unsupported_case = case(
            2,
            supported=False,
            task="beat_block_hammer",
            aspects=["object_physics.mass"],
        )
        unsupported = score_live_query_case(
            unsupported_case,
            {
                "attempt_count": 1,
                "selection": {
                    "decision": "unsupported",
                    "task_name": None,
                    "unsupported_capabilities": [
                        {
                            "task_name": "beat_block_hammer",
                            "aspect_id": "object_physics.mass",
                        }
                    ],
                },
            },
        )
        metrics = aggregate_live_query_cases([supported, unsupported])
        self.assertEqual(metrics["capability_decision_accuracy"], 1.0)
        self.assertEqual(metrics["task_accuracy"], 1.0)
        self.assertEqual(metrics["aspect_micro_f1"], 1.0)
        self.assertEqual(metrics["task_qualified_gap_coverage"], 1.0)

    def test_flat_unsupported_ids_are_not_scored(self):
        scored = score_live_query_case(
            case(1, supported=False, aspects=["object_scale"]),
            {
                "selection": {
                    "decision": "unsupported",
                    "task_name": None,
                    "unsupported_aspect_ids": ["object_scale"],
                }
            },
        )
        self.assertFalse(scored["schema_valid"])
        self.assertEqual(scored["predicted_aspects"], [])
        self.assertIsNone(scored["task_match"])
        self.assertFalse(scored["task_qualified_gap_available"])

    def test_provider_failure_counts_as_wrong_and_keeps_run_alive(self):
        scored = score_live_query_case(case(1), None, error="timeout")
        metrics = aggregate_live_query_cases([scored])
        self.assertEqual(metrics["schema_valid_rate"], 0.0)
        self.assertEqual(metrics["capability_decision_accuracy"], 0.0)
        self.assertEqual(metrics["provider_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
