import json
import tempfile
import unittest
from pathlib import Path

from mea.taskgen.acceptance import build_cached_taskgen_acceptance


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class TaskGenAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.generated = self.root / "mea/generated_tasks"
        self.run_ids = {
            "official": "run_official",
            "overlay": "run_overlay",
            "codegen": "run_codegen",
            "reflection": "run_reflection",
        }
        self._make_official()
        self._make_overlay()
        self._make_codegen()
        self._make_reflection()

    def tearDown(self):
        self.temp.cleanup()

    def run_dir(self, kind):
        return self.generated / self.run_ids[kind]

    def _make_official(self):
        root = self.run_dir("official")
        static = {
            "official_passthrough": {"valid": True},
            "code_generation": {"performed": False},
        }
        write_json(
            root / "manifest.json",
            {
                "mode": "official",
                "generation_kind": "official_passthrough",
                "task_module": "envs.click_bell",
                "provider": {"called": False},
            },
        )
        write_json(root / "validation/static.json", static)
        write_json(
            root / "generation/official_source.json", {"source": "envs/click_bell.py"}
        )

    def _make_overlay(self):
        root = self.run_dir("overlay")
        static = {
            "bounded_overlay": {"valid": True, "controlled_axis": "object_position"},
            "protected_diff": {"valid": True},
            "code_generation": {"performed": False},
        }
        write_json(
            root / "manifest.json",
            {
                "task_name": "click_bell",
                "mode": "reuse",
                "generation_kind": "bounded_variant_overlay",
                "variant_id": "object_position.left_fixed",
                "capability_id": "object_position.fixed_xy",
                "provider": {"called": False},
            },
        )
        write_json(
            root / "variant_spec.json",
            {
                "task_name": "click_bell",
                "generation_mode": "bounded_variant_overlay",
                "controlled_axis": "object_position",
            },
        )
        write_json(root / "validation/static.json", static)
        write_json(root / "generation/bounded_overlay.json", {"kind": "overlay"})
        (root / "overlay.yml").write_text("mea: {}\n", encoding="utf-8")

    def _make_codegen(self):
        root = self.run_dir("codegen")
        retrieval = {
            "selected_tasks": ["beat_block_hammer", "blocks_ranking_rgb"],
            "selected_sources": [
                "envs/beat_block_hammer.py",
                "envs/blocks_ranking_rgb.py",
            ],
        }
        knowledge = {
            "selected_ids": ["task.beat_block_hammer", "example.rgb"],
            "committed_index_current": True,
            "selected_documents": [
                {
                    "id": "task.beat_block_hammer",
                    "path": "mea/knowledge/tasks/beat_block_hammer.md",
                    "source_symbols": [
                        {
                            "path": "envs/beat_block_hammer.py",
                            "symbol": "beat_block_hammer.load_actors",
                            "sha256": "a" * 64,
                        }
                    ],
                }
            ],
            "selected_examples": [
                {
                    "id": "example.rgb",
                    "path": "envs/blocks_ranking_rgb.py",
                    "symbol": "blocks_ranking_rgb.load_actors",
                }
            ],
        }
        write_json(
            root / "manifest.json",
            {
                "task_name": "beat_block_hammer",
                "mode": "force_codegen",
                "task_module": "mea.generated_tasks.run_codegen.task",
                "base_commit": "b" * 40,
                "provider": {"calls": {"proposal": {}, "retrieval": {}, "codegen": {}}},
                "task_retrieval": retrieval,
                "knowledge_retrieval": knowledge,
            },
        )
        write_json(root / "variant_spec.json", {"task_name": "beat_block_hammer"})
        write_json(root / "generation/retrieval.json", retrieval)
        write_json(root / "generation/knowledge_retrieval.json", knowledge)
        (root / "generation/code_prompt.md").write_text("prompt\n", encoding="utf-8")
        (root / "generation/code_response.txt").write_text(
            "response\n", encoding="utf-8"
        )
        write_json(
            root / "validation/static.json",
            {
                "load_actors_ast": {
                    "valid": True,
                    "complete_method_generated": True,
                },
                "protected_diff": {"valid": True},
            },
        )
        (root / "task.py").write_text(
            "class beat_block_hammer: pass\n", encoding="utf-8"
        )

    def _make_reflection(self):
        root = self.run_dir("reflection")
        write_json(
            root / "manifest.json",
            {
                "task_name": "beat_block_hammer",
                "status": "completed_without_act",
                "provider": {"calls": {"proposal": {}, "repair": {}}},
            },
        )
        write_json(
            root / "reflection/fixture/fixture.json",
            {
                "fixture": "wrong_color",
                "test_only": True,
                "injected_method_structurally_valid": True,
                "injected_after_normal_static_gate": True,
            },
        )
        repair = {
            "installed": True,
            "method_sha256_before": "1" * 64,
            "method_sha256_after": "2" * 64,
            "static_validation": {
                "load_actors_ast": {"valid": True},
                "protected_diff": {"valid": True},
            },
        }
        summary = {
            "passed": True,
            "repairs_used": 1,
            "final_attempt": 1,
            "attempts": [
                {
                    "attempt_index": 0,
                    "observation": {
                        "passed": False,
                        "probe_passed": True,
                        "vision": {
                            "passed": False,
                            "diagnosis": "Expected blue but observed red.",
                            "unexpected_changes": ["wrong block color"],
                        },
                    },
                    "repair": repair,
                },
                {
                    "attempt_index": 1,
                    "observation": {
                        "passed": True,
                        "probe_passed": True,
                        "vision": {"passed": True},
                    },
                },
            ],
        }
        write_json(root / "reflection/summary.json", summary)
        write_json(
            root / "reflection/attempt_00/vision.json",
            summary["attempts"][0]["observation"]["vision"],
        )
        write_json(root / "reflection/attempt_00/repair.json", repair)
        write_json(root / "reflection/attempt_01/vision.json", {"passed": True})

    def build_report(self):
        return build_cached_taskgen_acceptance(
            self.root,
            official_run_id=self.run_ids["official"],
            overlay_run_id=self.run_ids["overlay"],
            codegen_run_id=self.run_ids["codegen"],
            reflection_run_id=self.run_ids["reflection"],
        )

    def test_cached_acceptance_covers_all_taskgen_routes_without_runtime(self):
        report = self.build_report()
        self.assertTrue(report["passed"])
        self.assertTrue(report["cached_artifact"])
        self.assertTrue(report["no_provider"])
        self.assertTrue(report["no_simulator"])
        self.assertTrue(report["no_ACT"])
        self.assertFalse(report["paper_table_eligible"])
        self.assertEqual(
            report["checks"]["scene_error_visual_reject_diagnose_repair"]["evidence"][
                "transition"
            ],
            [
                "static_pass",
                "visual_reject",
                "diagnosis",
                "repair_installed",
                "static_revalidate_pass",
                "visual_pass",
            ],
        )
        self.assertTrue(
            report["checks"]["bbh_true_codegen_and_retrieval_provenance"]["passed"]
        )

    def test_repair_artifact_with_act_or_missing_diagnosis_is_rejected(self):
        root = self.run_dir("reflection")
        write_json(root / "evaluation/act.json", {"passed": True})
        summary_path = root / "reflection/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["attempts"][0]["observation"]["vision"]["diagnosis"] = ""
        write_json(summary_path, summary)
        report = self.build_report()
        check = report["checks"]["scene_error_visual_reject_diagnose_repair"]
        self.assertFalse(report["passed"])
        self.assertFalse(check["passed"])
        self.assertTrue(check["evidence"]["original_act_artifact_present"])
        self.assertFalse(check["evidence"]["diagnosis_present"])

    def test_overlay_with_wrong_task_identity_is_rejected(self):
        root = self.run_dir("overlay")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["task_name"] = "beat_block_hammer"
        write_json(manifest_path, manifest)
        report = self.build_report()
        check = report["checks"]["click_overlay"]
        self.assertFalse(report["passed"])
        self.assertFalse(check["passed"])


if __name__ == "__main__":
    unittest.main()
