import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mea.taskgen.production_acceptance import (
    ProductionTaskAcceptanceError,
    _verify_reviewed_provenance,
    _validate_runtime_variant_spec,
    record_production_task_acceptance,
    require_production_task_acceptance,
    require_task_artifact_act_runtime_eligible,
)
from mea.taskgen.capabilities import build_variant_spec


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ProductionTaskAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "run_demo"
        self.run_dir.mkdir()
        (self.run_dir / "overlay.yml").write_text("{}\n", encoding="utf-8")
        write_json(self.run_dir / "variant_spec.json", {"task_name": "demo"})
        write_json(
            self.run_dir / "generation/task_artifact_bundle.json",
            {"task_name": "demo"},
        )
        self.manifest = {
            "run_id": "run_demo",
            "task_name": "demo",
            "task_module": "demo.task",
            "mode": "force_codegen",
            "generation_kind": "generated_scene_code",
            "provider": {"called": True, "calls": {"proposal": {}, "codegen": {}}},
        }
        self.scene = {
            "setup_success": True,
            "render_success": True,
            "rule_check": {"passed": True},
            "expert": {"passed": True},
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_reviewed_provenance_pins_inputs_but_allows_run_local_bindings(self):
        repo_root = Path(self.temporary.name) / "repo"
        run_dir = repo_root / "mea/generated_tasks/run_reviewed"
        copied_files = {
            "task.py": b"reviewed task\n",
            "variant_spec.json": b"{}\n",
            "overlay.yml": b"{}\n",
            "generation/load_actors.py.txt": b"def load_actors(self):\n    pass\n",
            "generation/task_artifact_bundle.json": b'{"source": true}\n',
            "generation/scene_check_spec.json": b'{"source": true}\n',
            "generation/success_spec.json": b"{}\n",
            "validation/static.json": b'{"reviewed": true}\n',
        }
        copied_hashes = {}
        for relative, payload in copied_files.items():
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            copied_hashes[relative] = hashlib.sha256(payload).hexdigest()

        runtime_hashes = {}
        from mea.taskgen.reviewed_registry import RUNTIME_DEPENDENCY_PATHS

        for relative in RUNTIME_DEPENDENCY_PATHS:
            path = repo_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {relative}\n", encoding="utf-8")
            runtime_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "generation_kind": "reviewed_generated_task_reuse",
            "reviewed_task_registration": {
                "copied_files": copied_hashes,
                "runtime_dependency_hashes": runtime_hashes,
            },
        }

        # These two bindings are intentionally rebuilt for the current run.
        write_json(
            run_dir / "generation/task_artifact_bundle.json", {"current": True}
        )
        write_json(run_dir / "generation/scene_check_spec.json", {"current": True})
        contract = _verify_reviewed_provenance(run_dir, manifest)
        self.assertIn("reviewed_immutable_artifacts_sha256", contract)
        self.assertIn("reviewed_runtime_dependencies_sha256", contract)

        (run_dir / "validation/static.json").write_text(
            '{"reviewed": false}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ProductionTaskAcceptanceError, "immutable artifact changed"
        ):
            _verify_reviewed_provenance(run_dir, manifest)

    def test_variant_validation_covers_official_and_generic_v2_routes(self):
        official = {
            "schema_version": 1,
            "task_name": "click_bell",
            "intent": "evaluate_official_task_unchanged",
            "generation_mode": "official",
            "changes": {},
            "preserve": ["official_task_source", "official_task_identity"],
        }
        self.assertEqual(
            _validate_runtime_variant_spec(
                official,
                {
                    "task_name": "click_bell",
                    "mode": "official",
                    "generation_kind": "official_passthrough",
                },
            ),
            official,
        )
        tampered = dict(official)
        tampered["changes"] = {"bell": {"xy": [0.0, 0.0]}}
        with self.assertRaisesRegex(
            ProductionTaskAcceptanceError, "official VariantSpec"
        ):
            _validate_runtime_variant_spec(
                tampered, {"task_name": "click_bell", "mode": "official"}
            )

        click = build_variant_spec(
            task_name="click_bell",
            variant_id="object_position.test",
            capability_id="object_position.fixed_xy",
            intent="test a bounded position",
            changes={
                "bell": {"position_mode": "fixed", "xy": [-0.14, -0.12]}
            },
            generation_mode="bounded_variant_overlay",
        )
        self.assertEqual(
            _validate_runtime_variant_spec(
                click, {"task_name": "click_bell", "mode": "reuse"}
            ),
            click,
        )

    @patch("mea.taskgen.production_acceptance._verify_bound_artifacts")
    @patch("mea.taskgen.production_acceptance.validate_task_artifact_bundle")
    @patch("mea.taskgen.production_acceptance.validate_variant_spec")
    def test_acceptance_is_append_only_and_precedes_act(
        self, validate_spec, validate_bundle, _verify_bound_artifacts
    ):
        validate_spec.return_value = {"task_name": "demo", "schema_version": 1}
        validate_bundle.return_value = {"task_name": "demo", "schema_version": 1}
        summary = record_production_task_acceptance(
            self.run_dir,
            self.manifest,
            scene=self.scene,
            position_samples={"passed": True},
            require_expert=True,
        )
        self.assertEqual(summary["status"], "accepted")
        self.assertEqual(summary["runtime"]["act_rollouts_started"], 0)
        self.assertEqual(summary["runtime"]["provider_calls"], 2)
        loaded = require_production_task_acceptance(self.run_dir, self.manifest)
        self.assertEqual(loaded["accepted_result"]["checks"]["expert_passed"], True)

        repeated = record_production_task_acceptance(
            self.run_dir,
            self.manifest,
            scene=self.scene,
            position_samples={"passed": True},
            require_expert=True,
        )
        self.assertEqual(repeated, summary)

        failed_current_scene = dict(self.scene)
        failed_current_scene["expert"] = {"passed": False}
        with self.assertRaisesRegex(
            ProductionTaskAcceptanceError, "current official expert gate"
        ):
            record_production_task_acceptance(
                self.run_dir,
                self.manifest,
                scene=failed_current_scene,
                position_samples={"passed": True},
                require_expert=True,
            )

        (self.run_dir / "overlay.yml").write_text(
            "mea:\n  enabled: true\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ProductionTaskAcceptanceError, "artifact contract changed"
        ):
            require_production_task_acceptance(self.run_dir, self.manifest)

    @patch("mea.taskgen.production_acceptance._validate_current_candidate")
    def test_final_bundle_forbids_act_despite_stale_or_missing_manifest_copy(
        self, validate_candidate
    ):
        validate_candidate.return_value = (
            {"task_name": "demo"},
            {
                "success_semantics": {
                    "act_runtime_eligible": False,
                    "runtime_blocker": "experimental success is probe-only",
                }
            },
            {},
        )
        manifests = (
            {},
            {"task_proposal": {"schema_version": 1}},
        )
        for manifest in manifests:
            with self.subTest(manifest=manifest), self.assertRaisesRegex(
                ProductionTaskAcceptanceError,
                "TaskArtifactBundle forbids ACT runtime execution",
            ):
                require_task_artifact_act_runtime_eligible(
                    self.run_dir,
                    manifest,
                )

    @patch("mea.taskgen.production_acceptance._verify_bound_artifacts")
    @patch("mea.taskgen.production_acceptance.validate_task_artifact_bundle")
    @patch("mea.taskgen.production_acceptance.validate_variant_spec")
    def test_failed_expert_gate_never_accepts(
        self, validate_spec, validate_bundle, _verify_bound_artifacts
    ):
        validate_spec.return_value = {"task_name": "demo"}
        validate_bundle.return_value = {"task_name": "demo"}
        scene = dict(self.scene)
        scene["expert"] = {"passed": False}
        with self.assertRaises(ProductionTaskAcceptanceError):
            record_production_task_acceptance(
                self.run_dir,
                self.manifest,
                scene=scene,
                position_samples={"passed": True},
                require_expert=True,
            )
        summary = json.loads(
            (
                self.run_dir
                / "validation/task_generation_attempts/task_generation_attempt_summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(
            summary["attempts"][0]["failure"]["stage"], "expert_gate"
        )

    @patch("mea.taskgen.production_acceptance._verify_bound_artifacts")
    @patch("mea.taskgen.production_acceptance.validate_task_artifact_bundle")
    @patch("mea.taskgen.production_acceptance.validate_variant_spec")
    def test_existing_act_evidence_fails_closed(
        self, validate_spec, validate_bundle, _verify_bound_artifacts
    ):
        validate_spec.return_value = {"task_name": "demo"}
        validate_bundle.return_value = {"task_name": "demo"}
        self.manifest["act_evaluation"] = {"actual_seeds": [7]}
        with self.assertRaises(ProductionTaskAcceptanceError):
            record_production_task_acceptance(
                self.run_dir,
                self.manifest,
                scene=self.scene,
                position_samples={"passed": True},
                require_expert=True,
            )


if __name__ == "__main__":
    unittest.main()
