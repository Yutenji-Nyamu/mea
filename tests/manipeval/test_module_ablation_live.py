import json
import tempfile
import unittest
from pathlib import Path

from mea.module_ablation_live import (
    LiveModuleAblationError,
    generate_live_module_ablation,
    review_live_module_ablation,
)
from mea.module_ablation_protocol import prepare_module_ablation_schedule


class StubProvider:
    def __init__(self):
        self.calls = []
        self.last_metadata = {}

    def text(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        self.last_metadata = {"id": f"call-{len(self.calls)}"}
        return json.dumps({"candidate": "bounded", "limitations": ["not run"]})


def config():
    identity = {
        "base_commit": "a" * 40,
        "runner": "scripts/manipeval_taskgen.py",
        "runner_sha256": "b" * 64,
        "provider_model": "live-model",
        "config_sha256": "c" * 64,
        "seed": 1,
    }
    return {
        "schema_version": 1,
        "study_id": "live_micro",
        "artifact_root": "formal/live_micro",
        "components": {
            "taskgen": {
                "conditions": ["complete", "no_rag"],
                "cases": [{
                    "case_id": "task_case",
                    "input_identity": {
                        "query": "test generalization",
                        "retrieved_context": "trusted context marker",
                    },
                    "execution_identity": identity,
                }],
            }
        },
    }


class LiveModuleAblationTests(unittest.TestCase):
    def test_generation_and_review_are_separate_hash_bound_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = prepare_module_ablation_schedule(root, config())
            provider = StubProvider()
            generated = generate_live_module_ablation(
                root,
                schedule,
                output_dir="runs/live",
                provider=provider,
                model="live-model",
            )
            self.assertEqual(len(provider.calls), 2)
            self.assertIsNone(generated["success_rates"])
            self.assertIn("trusted context marker", provider.calls[0][0])
            self.assertNotIn("trusted context marker", provider.calls[1][0])
            labels = {
                "schema_version": 1,
                "reviewer": "development_agent_proxy",
                "labels": {
                    item["schedule_item_id"]: {
                        "success": item["condition"] == "complete",
                        "rationale": "proxy inspection of the saved candidate",
                    }
                    for item in generated["items"]
                },
            }
            reviewed = review_live_module_ablation(root / "runs/live", labels)
            self.assertEqual(
                reviewed["comparisons"][0]["proxy_absolute_success_difference"],
                1.0,
            )
            self.assertFalse(reviewed["paper_table_eligible"])
            with self.assertRaisesRegex(LiveModuleAblationError, "already exists"):
                review_live_module_ablation(root / "runs/live", labels)

    def test_model_and_exact_label_contract_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = prepare_module_ablation_schedule(root, config())
            with self.assertRaisesRegex(LiveModuleAblationError, "provider model"):
                generate_live_module_ablation(
                    root,
                    schedule,
                    output_dir="runs/wrong",
                    provider=StubProvider(),
                    model="wrong-model",
                )


if __name__ == "__main__":
    unittest.main()
