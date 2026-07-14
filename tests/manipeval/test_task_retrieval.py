import unittest
from pathlib import Path

from mea.retrieval import (
    TaskRetrievalError,
    discover_task_catalog,
    validate_task_selection,
)


class TaskRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.catalog = discover_task_catalog(cls.repo_root)

    def test_discovers_the_robotwin_task_library(self):
        names = {item["task_name"] for item in self.catalog}
        self.assertEqual(len(self.catalog), 50)
        self.assertIn("beat_block_hammer", names)
        self.assertIn("blocks_ranking_rgb", names)

    def test_accepts_canonical_task_first(self):
        selected = validate_task_selection(
            {
                "selected_tasks": [
                    "beat_block_hammer",
                    "blocks_ranking_rgb",
                ],
                "reasoning": "Behavior plus an RGB construction reference.",
            },
            canonical_task="beat_block_hammer",
            catalog=self.catalog,
        )
        self.assertEqual(selected["catalog_size"], 50)

    def test_rejects_unknown_or_reordered_canonical_task(self):
        with self.assertRaises(TaskRetrievalError):
            validate_task_selection(
                {
                    "selected_tasks": ["blocks_ranking_rgb"],
                    "reasoning": "Missing canonical task.",
                },
                canonical_task="beat_block_hammer",
                catalog=self.catalog,
            )
        with self.assertRaises(TaskRetrievalError):
            validate_task_selection(
                {
                    "selected_tasks": ["beat_block_hammer", "does_not_exist"],
                    "reasoning": "Unknown task.",
                },
                canonical_task="beat_block_hammer",
                catalog=self.catalog,
            )


if __name__ == "__main__":
    unittest.main()
