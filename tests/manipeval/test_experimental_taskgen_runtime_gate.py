import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mea.taskgen.production_acceptance import ProductionTaskAcceptanceError
from mea.taskgen.success_spec import experimental_bbh_success_spec_v2
from scripts.manipeval_taskgen import main


def proposal_v2() -> dict:
    return {
        "schema_version": 2,
        "proposal_id": "object_appearance.cli_runtime_gate",
        "task_name": "beat_block_hammer",
        "aspect_id": "object_appearance.color",
        "intent": "compile a bounded experimental fixture",
        "capability_id": "object_appearance.color",
        "reuse_first": True,
        "changes": {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 1.0,
                "color": [0.25, 0.25, 0.75],
            }
        },
        "preserve_success_semantics": False,
        "success_spec": experimental_bbh_success_spec_v2(),
    }


class ExperimentalTaskGenRuntimeGateTests(unittest.TestCase):
    def test_fresh_v2_run_act_fails_before_provider_or_simulator(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv = [
                "manipeval_taskgen.py",
                "--repo-root",
                str(Path(temporary)),
                "--request",
                "must not start ACT",
                "--task-name",
                "beat_block_hammer",
                "--mode",
                "force_codegen",
                "--task-proposal-json",
                json.dumps(proposal_v2()),
                "--run-act",
            ]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(
                SystemExit,
                "experimental TaskProposal v2 is 0-ACT only",
            ):
                main()

    def test_resumed_v2_run_act_also_fails_before_simulator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_v2_resume_gate"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "run_v2_resume_gate",
                        "status": "completed_without_act",
                        "task_name": "beat_block_hammer",
                        "mode": "force_codegen",
                        "task_proposal": proposal_v2(),
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "manipeval_taskgen.py",
                "--repo-root",
                str(root),
                "--resume-run",
                "run_v2_resume_gate",
                "--task-name",
                "beat_block_hammer",
                "--mode",
                "force_codegen",
                "--run-act",
            ]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(
                SystemExit,
                "experimental TaskProposal v2 is 0-ACT only",
            ):
                main()

    def test_resumed_act_uses_bundle_gate_when_manifest_copy_is_missing_or_stale(self):
        manifest_copies = (
            None,
            {"schema_version": 1},
        )
        for manifest_copy in manifest_copies:
            with (
                self.subTest(manifest_copy=manifest_copy),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                run_dir = root / "mea/generated_tasks/run_bundle_gate"
                run_dir.mkdir(parents=True)
                manifest = {
                    "schema_version": 1,
                    "run_id": "run_bundle_gate",
                    "status": "completed_without_act",
                    "task_name": "beat_block_hammer",
                    "mode": "force_codegen",
                }
                if manifest_copy is not None:
                    manifest["task_proposal"] = manifest_copy
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
                argv = [
                    "manipeval_taskgen.py",
                    "--repo-root",
                    str(root),
                    "--resume-run",
                    "run_bundle_gate",
                    "--task-name",
                    "beat_block_hammer",
                    "--mode",
                    "force_codegen",
                    "--run-act",
                ]
                blocker = ProductionTaskAcceptanceError(
                    "TaskArtifactBundle forbids ACT runtime execution: probe-only"
                )
                with (
                    patch.object(sys, "argv", argv),
                    patch(
                        "scripts.manipeval_taskgen."
                        "require_task_artifact_act_runtime_eligible",
                        side_effect=blocker,
                    ) as gate,
                    patch("scripts.manipeval_taskgen.run_probe") as simulator,
                    self.assertRaisesRegex(
                        SystemExit,
                        "TaskArtifactBundle forbids ACT runtime execution",
                    ),
                ):
                    main()
                gate.assert_called_once()
                simulator.assert_not_called()

    def test_task_only_acceptance_requires_expert_and_forbids_act(self):
        cases = (
            (
                ["--accept-task-only"],
                "--accept-task-only requires --expert",
            ),
            (
                ["--accept-task-only", "--expert", "--run-act"],
                "--accept-task-only cannot be combined with --run-act",
            ),
        )
        for flags, message in cases:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as temporary:
                argv = [
                    "manipeval_taskgen.py",
                    "--repo-root",
                    str(Path(temporary)),
                    "--request",
                    "validate flags only",
                    "--task-name",
                    "beat_block_hammer",
                    *flags,
                ]
                with patch.object(sys, "argv", argv), self.assertRaisesRegex(
                    SystemExit, message
                ):
                    main()


if __name__ == "__main__":
    unittest.main()
