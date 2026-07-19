import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mea.module_ablation_protocol import (
    ModuleAblationError,
    audit_module_ablation_artifacts,
    prepare_module_ablation_schedule,
)


def execution_identity():
    return {
        "base_commit": "a" * 40,
        "runner": "scripts/manipeval_taskgen.py",
        "runner_sha256": "b" * 64,
        "provider_model": "development-provider",
        "config_sha256": "c" * 64,
        "seed": 100401,
    }


def config(*, taskgen_conditions=None, toolgen=False):
    components = {
        "taskgen": {
            "conditions": taskgen_conditions or ["complete", "no_rag"],
            "cases": [
                {
                    "case_id": "case_001",
                    "input_identity": {
                        "query": "test object generalization",
                        "task_name": "click_bell",
                    },
                    "execution_identity": execution_identity(),
                }
            ],
        }
    }
    if toolgen:
        tool_execution = execution_identity()
        tool_execution["runner"] = "scripts/manipeval_toolgen.py"
        tool_execution["runner_sha256"] = "d" * 64
        components["toolgen"] = {
            "conditions": ["complete", "no_tool_validation"],
            "cases": [
                {
                    "case_id": "tool_001",
                    "input_identity": {
                        "tool_query": "distance to target",
                        "task_name": "click_bell",
                    },
                    "execution_identity": tool_execution,
                }
            ],
        }
    return {
        "schema_version": 1,
        "study_id": "table3_smoke",
        "artifact_root": "artifacts/table3_smoke",
        "components": components,
    }


def _outcome_evidence(schedule, item, measurement_kind, success):
    return {
        "schema_version": 1,
        "evidence_type": (
            "module_ablation_generation_outcome_v1"
            if measurement_kind == "generation_outcome"
            else "module_ablation_provenance_only_v1"
        ),
        "study_id": schedule["study_id"],
        "schedule_contract_sha256": schedule["schedule_contract_sha256"],
        "schedule_item_id": item["schedule_item_id"],
        "schedule_item_sha256": item["schedule_item_sha256"],
        "component": item["component"],
        "condition": item["condition"],
        "case_id": item["case_id"],
        "input_identity_sha256": item["input_identity_sha256"],
        "execution_identity_sha256": item["execution_identity_sha256"],
        "applied_module_switches": item["module_switches"],
        "measurement_kind": measurement_kind,
        "success": success,
    }


def write_completed(
    root: Path,
    schedule: dict,
    *,
    component: str,
    condition: str,
    success: bool | None,
    measurement_kind: str = "generation_outcome",
    act_rollouts: int = 0,
) -> tuple[Path, Path]:
    item = next(
        row
        for row in schedule["items"]
        if row["component"] == component and row["condition"] == condition
    )
    artifact_dir = root / item["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evidence = artifact_dir / "outcome.json"
    evidence.write_text(
        json.dumps(_outcome_evidence(schedule, item, measurement_kind, success)),
        encoding="utf-8",
    )
    evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "protocol": "table3_module_ablation_completed_artifact_v1",
        "status": "completed",
        "study_id": schedule["study_id"],
        "schedule_contract_sha256": schedule["schedule_contract_sha256"],
        "schedule_item_id": item["schedule_item_id"],
        "schedule_item_sha256": item["schedule_item_sha256"],
        "component": component,
        "condition": condition,
        "case_id": item["case_id"],
        "input_identity_sha256": item["input_identity_sha256"],
        "execution_identity": item["execution_identity"],
        "execution_identity_sha256": item["execution_identity_sha256"],
        "applied_module_switches": item["module_switches"],
        "runtime": {
            "provider_called": True,
            "simulator_called": component == "taskgen",
            "act_rollouts_started": act_rollouts,
        },
        "result": {
            "measurement_kind": measurement_kind,
            "success": success,
            "outcome_evidence_path": "outcome.json",
            "outcome_evidence_sha256": evidence_sha256,
        },
        "artifacts": [
            {
                "kind": "outcome_evidence",
                "path": "outcome.json",
                "sha256": evidence_sha256,
            }
        ],
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, evidence


def update_outcome_hash(manifest_path: Path, evidence_path: Path) -> None:
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["outcome_evidence_sha256"] = digest
    manifest["artifacts"][0]["sha256"] = digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class ModuleAblationProtocolTests(unittest.TestCase):
    def test_prepare_supports_the_exact_paper_table3_condition_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = config(
                taskgen_conditions=[
                    "complete",
                    "no_rag",
                    "no_visual_self_check",
                    "no_readme_agent",
                    "base",
                ],
                toolgen=True,
            )
            payload["components"]["toolgen"]["conditions"] = [
                "complete",
                "no_rag",
            ]
            schedule = prepare_module_ablation_schedule(root, payload)
            self.assertEqual(len(schedule["items"]), 7)
            switches = {
                (item["component"], item["condition"]): item["module_switches"]
                for item in schedule["items"]
            }
            self.assertEqual(
                switches[("taskgen", "no_readme_agent")],
                {
                    "rag": True,
                    "visual_self_check": True,
                    "readme_agent": False,
                },
            )
            self.assertEqual(switches[("toolgen", "no_rag")], {"rag": False})

    def test_prepare_is_bounded_rectangular_and_zero_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = prepare_module_ablation_schedule(
                root,
                config(
                    taskgen_conditions=["complete", "no_rag", "no_visual_gate"],
                    toolgen=True,
                ),
            )
            self.assertEqual(result["mode"], "prepare_only")
            self.assertEqual(
                result["runtime"],
                {
                    "provider_called": False,
                    "simulator_called": False,
                    "act_rollouts_started": 0,
                },
            )
            self.assertEqual(len(result["items"]), 5)
            task_items = [row for row in result["items"] if row["component"] == "taskgen"]
            self.assertEqual(len({row["input_identity_sha256"] for row in task_items}), 1)
            self.assertEqual(
                len({row["execution_identity_sha256"] for row in task_items}), 1
            )
            self.assertTrue(all(row["schedule_contract_sha256"] == result["schedule_contract_sha256"] for row in result["items"]))
            self.assertFalse(result["paper_table_eligible"])

    def test_incomplete_rectangle_or_cross_condition_identity_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = prepare_module_ablation_schedule(root, config())
            incomplete = copy.deepcopy(schedule)
            incomplete["items"].pop()
            with self.assertRaisesRegex(ModuleAblationError, "matrix is incomplete"):
                audit_module_ablation_artifacts(root, incomplete)

            mismatched = copy.deepcopy(schedule)
            mismatched["items"][1]["input_identity"]["query"] = "different input"
            with self.assertRaises(ModuleAblationError) as raised:
                audit_module_ablation_artifacts(root, mismatched)
            self.assertNotIsInstance(raised.exception, KeyError)

    def test_contract_hash_binds_root_contracts_matched_sets_and_item_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = prepare_module_ablation_schedule(root, config())
            mutations = []
            changed_root = copy.deepcopy(schedule)
            changed_root["artifact_root"] = "artifacts/changed"
            mutations.append(changed_root)
            changed_contract = copy.deepcopy(schedule)
            changed_contract["condition_contracts"]["taskgen"][0]["description"] = "changed"
            mutations.append(changed_contract)
            changed_match = copy.deepcopy(schedule)
            changed_match["matched_sets"][0]["input_identity"]["query"] = "changed"
            mutations.append(changed_match)
            changed_item_path = copy.deepcopy(schedule)
            changed_item_path["items"][0]["artifact_dir"] += "_changed"
            mutations.append(changed_item_path)
            for mutation in mutations:
                with self.subTest(mutation=mutations.index(mutation)):
                    with self.assertRaises(ModuleAblationError):
                        audit_module_ablation_artifacts(root, mutation)

    def test_exact_manifest_binds_switches_execution_schedule_and_item_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = prepare_module_ablation_schedule(root, config())
            manifest_path, _ = write_completed(
                root,
                schedule,
                component="taskgen",
                condition="complete",
                success=True,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["applied_module_switches"] = {"rag": False, "visual_gate": True}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ModuleAblationError, "applied switches mismatch"):
                audit_module_ablation_artifacts(root, schedule)

            manifest["applied_module_switches"] = schedule["items"][0]["module_switches"]
            manifest["execution_identity"]["seed"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ModuleAblationError, "execution identity mismatch"):
                audit_module_ablation_artifacts(root, schedule)

            manifest["execution_identity"] = schedule["items"][0]["execution_identity"]
            manifest["unexpected"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ModuleAblationError, "exact contract"):
                audit_module_ablation_artifacts(root, schedule)

    def test_typed_outcome_binds_success_condition_input_and_switches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = prepare_module_ablation_schedule(root, config())
            manifest_path, evidence_path = write_completed(
                root,
                schedule,
                component="taskgen",
                condition="complete",
                success=True,
            )
            for field, value in (
                ("success", False),
                ("condition", "no_rag"),
                ("input_identity_sha256", "0" * 64),
                ("applied_module_switches", {"rag": False, "visual_gate": True}),
            ):
                with self.subTest(field=field):
                    evidence = _outcome_evidence(
                        schedule, schedule["items"][0], "generation_outcome", True
                    )
                    evidence[field] = value
                    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
                    update_outcome_hash(manifest_path, evidence_path)
                    with self.assertRaisesRegex(ModuleAblationError, "typed outcome"):
                        audit_module_ablation_artifacts(root, schedule)

    def test_complete_matched_typed_outcomes_produce_functional_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = prepare_module_ablation_schedule(root, config())
            write_completed(
                root, schedule, component="taskgen", condition="complete", success=True
            )
            write_completed(
                root, schedule, component="taskgen", condition="no_rag", success=False
            )
            result = audit_module_ablation_artifacts(root, schedule)
            comparison = result["comparisons"][0]
            self.assertEqual(comparison["effect"]["absolute_success_rate_difference"], 1.0)
            self.assertTrue(result["all_effects_available"])
            self.assertEqual(
                result["historical_artifact_runtime"]["attestation"],
                "self_attested_by_completed_manifests_not_independently_observed",
            )
            self.assertIn("functional-only", result["claim_scope"])
            self.assertFalse(result["paper_table_eligible"])
            self.assertEqual(result["runtime"]["act_rollouts_started"], 0)

    def test_missing_or_provenance_only_pair_keeps_effect_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = prepare_module_ablation_schedule(root, config())
            write_completed(
                root,
                schedule,
                component="taskgen",
                condition="complete",
                success=None,
                measurement_kind="provenance_only",
            )
            result = audit_module_ablation_artifacts(root, schedule)
            self.assertIsNone(result["comparisons"][0]["effect"])
            self.assertEqual(result["artifact_audit"]["provenance_only"], 1)
            self.assertEqual(result["artifact_audit"]["missing_or_incomplete"], 1)
            self.assertIsNone(result["table3_success_rates"])

    def test_symlink_component_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real-artifacts"
            real.mkdir()
            link = root / "artifacts"
            try:
                os.symlink(real, link, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                # Windows without Developer Mode cannot create a symlink.  Keep
                # the regression deterministic by simulating the same Path
                # signal used by the component-by-component guard.
                with mock.patch.object(
                    type(root),
                    "is_symlink",
                    autospec=True,
                    side_effect=lambda value: value.name == "artifacts",
                ):
                    with self.assertRaisesRegex(
                        ModuleAblationError, "symlink component"
                    ):
                        prepare_module_ablation_schedule(root, config())
                self.assertIsInstance(exc, OSError)
                return
            with self.assertRaisesRegex(ModuleAblationError, "symlink component"):
                prepare_module_ablation_schedule(root, config())

    def test_cli_resolves_relative_output_from_repo_and_hides_tracebacks(self):
        script = Path(__file__).resolve().parents[2] / "scripts/manipeval_module_ablation.py"
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as cwd:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config()), encoding="utf-8")
            success = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--repo-root",
                    str(root),
                    "prepare",
                    "--config",
                    "config.json",
                    "--output-dir",
                    "relative-output",
                ],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertTrue((root / "relative-output/schedule.json").is_file())
            self.assertFalse((Path(cwd) / "relative-output").exists())

            bad = root / "bad.json"
            bad.write_text("{not-json", encoding="utf-8")
            failure = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--repo-root",
                    str(root),
                    "prepare",
                    "--config",
                    "bad.json",
                    "--output-dir",
                    "bad-output",
                ],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(failure.returncode, 0)
            self.assertNotIn("Traceback", failure.stderr)
            self.assertIn("error:", failure.stderr)

    def test_unregistered_condition_and_completed_act_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ModuleAblationError, "unsupported taskgen"):
                prepare_module_ablation_schedule(
                    root, config(taskgen_conditions=["complete", "disable_everything"])
                )
            schedule = prepare_module_ablation_schedule(root, config())
            write_completed(
                root,
                schedule,
                component="taskgen",
                condition="complete",
                success=True,
                act_rollouts=1,
            )
            with self.assertRaisesRegex(ModuleAblationError, "zero-ACT"):
                audit_module_ablation_artifacts(root, schedule)


if __name__ == "__main__":
    unittest.main()
