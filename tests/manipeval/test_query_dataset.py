import unittest

from mea.query_dataset import QueryDatasetError, summarize_query_dataset


def case(index, status="unsupported"):
    return {
        "id": f"q{index:03d}",
        "query": f"query {index}",
        "setting": "single_task",
        "task_name": "click_bell",
        "task_profile": "adaptive_properties",
        "gold_sub_aspect_ids": [f"aspect.{index}"],
        "acceptable_first_sub_aspect_ids": [f"aspect.{index}"],
        "capability_status": status,
        "annotation": {
            "source": "model_draft",
            "review_status": "unreviewed",
            "human_votes": [],
        },
    }


class QueryDatasetTests(unittest.TestCase):
    def test_twenty_unreviewed_cases_are_not_paper_eligible(self):
        value = {
            "schema_version": 1,
            "dataset_id": "draft",
            "annotation_status": "model_draft_unreviewed",
            "cases": [
                case(i, "supported" if i <= 5 else "unsupported") for i in range(1, 21)
            ],
        }
        summary = summarize_query_dataset(value)
        self.assertEqual(summary["case_count"], 20)
        self.assertEqual(summary["capability_status_counts"]["supported"], 5)
        self.assertIsNone(summary["human_agent_agreement"])
        self.assertFalse(summary["paper_table_eligible"])
        value["cases"][0]["annotation"]["source"] = "human"
        with self.assertRaises(QueryDatasetError):
            summarize_query_dataset(value)


if __name__ == "__main__":
    unittest.main()
