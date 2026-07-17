import json
import tempfile
import unittest
from pathlib import Path

from mea.micro_ablation import build_cached_micro_ablation


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class MicroAblationTests(unittest.TestCase):
    def test_cached_and_fault_rows_are_never_table3_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "tool"
            write_json(
                generated / "manifest.json",
                {"status": "passed", "successful_attempt": 1},
            )
            write_json(
                generated / "attempts/attempt_0/validation.json",
                {"valid": False},
            )
            (generated / "attempts/attempt_0/generated_tool.py").write_text(
                "def generated_tool(x): return x", encoding="utf-8"
            )
            write_json(
                generated / "attempts/attempt_1/validation.json",
                {"valid": True},
            )
            write_json(generated / "registration.json", {"status": "validated"})
            acceptance = {
                "checks": {
                    "scene_error_visual_reject_diagnose_repair": {
                        "passed": True,
                        "evidence": {
                            "static_pass": True,
                            "visual_reject": True,
                            "repair_installed": True,
                            "visual_pass": True,
                        },
                    },
                    "bbh_true_codegen_and_retrieval_provenance": {
                        "passed": True,
                        "evidence": {
                            "task_source_provenance_valid": True,
                            "knowledge_document_provenance_valid": True,
                        },
                    },
                }
            }
            result = build_cached_micro_ablation(
                root,
                taskgen_acceptance=acceptance,
                toolgen_dir="tool",
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["runtime"]["act_rollouts_started"], 0)
            self.assertIsNone(result["table3_success_rates"])
            self.assertFalse(result["paper_table_eligible"])
            self.assertEqual(len(result["rows"]), 5)
            self.assertEqual(
                result["functional_gate_checks"],
                {"passed": 4, "total": 4, "all_passed": True},
            )
            self.assertEqual(result["provenance_checks"]["passed"], 1)
            self.assertIsNone(
                result["provenance_checks"]["ablation_effect_estimate"]
            )


if __name__ == "__main__":
    unittest.main()
