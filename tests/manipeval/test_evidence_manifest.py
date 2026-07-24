import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mea.evidence_manifest import (
    EvidenceManifestError,
    prepare_evidence_manifest,
    validate_evidence_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def config(root: Path) -> dict:
    candidates = ["object_position.left_fixed", "object_position.right_fixed"]
    schedule = [
        {"strategy": strategy, "variant_id": variant, "seed": 100402}
        for strategy in ("fixed_predeclared_v1", "dynamic_evidence_v1")
        for variant in candidates
    ]
    return {
        "schema_version": 1,
        "registration_id": "click_bell_pair_n1",
        "claim_scope": "table1_efficiency_mechanism_plumbing_only",
        "task_name": "click_bell",
        "query": "How robust is ACT to bell position?",
        "base_commit": git_head(root),
        "candidate_suite": candidates,
        "checkpoint_setting": "demo_clean",
        "expert_data_num": 50,
        "checkpoint_files": [
            "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt",
            "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/dataset_stats.pkl",
        ],
        "telemetry_profile": "balanced_v1",
        "sample_schedule": schedule,
        "source_artifacts": ["configs/pair_source.json"],
    }


def materialize(root: Path) -> None:
    write(
        root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt",
        b"tiny-test-policy",
    )
    write(
        root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/dataset_stats.pkl",
        b"tiny-test-stats",
    )
    write(root / "configs/pair_source.json", b'{"schema_version":1}\n')
    write(
        root / "mea/toolkit/schemas/click_bell.json",
        b'{"task_name":"click_bell","task_family":"contact_trigger"}\n',
    )
    write(
        root / ".gitignore",
        b"policy/ACT/\nconfigs/pair_source.json\nmea/evidence_registrations/\nmea/validation_runs/\n",
    )
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "MEA Tests"], cwd=root, check=True
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=root,
        check=True,
        capture_output=True,
    )


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


class EvidenceManifestTests(unittest.TestCase):
    def test_wrong_task_checkpoint_and_arbitrary_template_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            wrong_task = config(root)
            wrong_task["task_name"] = "beat_block_hammer"
            with self.assertRaisesRegex(EvidenceManifestError, "task_name"):
                prepare_evidence_manifest(root, wrong_task)

            wrong_checkpoint = config(root)
            wrong_checkpoint["checkpoint_files"][-1] = (
                "policy/ACT/act_ckpt/act-click_bell/other/dataset_stats.pkl"
            )
            with self.assertRaisesRegex(EvidenceManifestError, "exactly equal"):
                prepare_evidence_manifest(root, wrong_checkpoint)

            arbitrary = config(root)
            arbitrary["candidate_suite"] = ["object_position.left_fixed"]
            with self.assertRaisesRegex(EvidenceManifestError, "complete ordered expansion"):
                prepare_evidence_manifest(root, arbitrary)

    def test_missing_git_and_dirty_tracked_worktree_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            valid = config(root)
            (root / "mea/toolkit/schemas/click_bell.json").write_text(
                '{"task_name":"click_bell","task_family":"changed"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvidenceManifestError, "tracked Git worktree"):
                prepare_evidence_manifest(root, valid)

            nongit = root / "not_a_repo"
            nongit.mkdir()
            with self.assertRaisesRegex(EvidenceManifestError, "Git worktree"):
                prepare_evidence_manifest(nongit, valid)

    def test_prepare_and_validate_pin_all_required_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            manifest = prepare_evidence_manifest(root, config(root))
            result = validate_evidence_manifest(root, manifest)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["checkpoint_file_count"], 2)
            self.assertEqual(result["source_artifact_count"], 1)
            self.assertEqual(result["act_rollouts_started"], 0)
            self.assertEqual(
                manifest["telemetry"]["profile_id"], "balanced_v1"
            )
            self.assertEqual(len(manifest["sample_schedule"]["entries"]), 4)

    def test_changed_checkpoint_and_source_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            manifest = prepare_evidence_manifest(root, config(root))
            write(
                root
                / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt",
                b"changed",
            )
            with self.assertRaisesRegex(EvidenceManifestError, "checkpoint artifact changed"):
                validate_evidence_manifest(root, manifest)

            write(
                root
                / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt",
                b"tiny-test-policy",
            )
            write(root / "configs/pair_source.json", b'{"changed":true}\n')
            with self.assertRaisesRegex(
                EvidenceManifestError, "source_artifacts artifact changed"
            ):
                validate_evidence_manifest(root, manifest)

    def test_manifest_payload_tampering_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            manifest = prepare_evidence_manifest(root, config(root))
            tampered = copy.deepcopy(manifest)
            tampered["claim_scope"] = "paper_result"
            with self.assertRaisesRegex(EvidenceManifestError, "payload hash mismatch"):
                validate_evidence_manifest(root, tampered)

            escaped = config(root)
            escaped["source_artifacts"] = ["../outside.json"]
            with self.assertRaisesRegex(EvidenceManifestError, "stay inside"):
                prepare_evidence_manifest(root, escaped)

    def test_schedule_must_be_matched_n1_candidate_universe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            bad = config(root)
            bad["sample_schedule"].pop()
            with self.assertRaisesRegex(EvidenceManifestError, "exactly one sample"):
                prepare_evidence_manifest(root, bad)
            bad = config(root)
            bad["sample_schedule"][-1]["seed"] = 2
            with self.assertRaisesRegex(EvidenceManifestError, "identities must match"):
                prepare_evidence_manifest(root, bad)

    def test_symlinked_registered_artifact_is_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            target = root / "configs/real_source.json"
            write(target, b"{}\n")
            link = root / "configs/source_link.json"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlink creation is not available on this host")
            bad = config(root)
            bad["source_artifacts"] = ["configs/source_link.json"]
            with self.assertRaisesRegex(EvidenceManifestError, "symlink"):
                prepare_evidence_manifest(root, bad)

    def test_allowlisted_checkpoint_symlink_is_bound_and_retarget_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mea"
            root.mkdir()
            materialize(root)
            first = root.parent / "RoboTwinA"
            second = root.parent / "RoboTwinB"
            shutil.copytree(root / "policy", first / "policy")
            shutil.copytree(root / "policy", second / "policy")
            shutil.rmtree(root / "policy")
            try:
                (root / "policy").symlink_to(first / "policy", target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation is not available on this host")

            manifest = prepare_evidence_manifest(root, config(root))
            checkpoint = manifest["checkpoint"]["files"][0]
            self.assertEqual(checkpoint["symlink_chain"][0]["path"], "policy")
            self.assertEqual(
                checkpoint["resolved_path"],
                (
                    first
                    / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt"
                ).as_posix(),
            )

            (root / "policy").unlink()
            (root / "policy").symlink_to(second / "policy", target_is_directory=True)
            with self.assertRaisesRegex(
                EvidenceManifestError, "checkpoint artifact changed"
            ):
                validate_evidence_manifest(root, manifest)

    def test_prepare_and_validate_cli_execute_zero_act(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            config_path = root / "configs/preregister.json"
            config_path.write_text(json.dumps(config(root)), encoding="utf-8")
            output = root / "mea/validation_runs/preregister/manifest.json"
            command = [
                sys.executable,
                str(
                    REPO_ROOT
                    / "experiments/paper/manipeval_evidence_manifest.py"
                ),
                "--repo-root",
                str(root),
                "prepare",
                "--config",
                str(config_path),
                "--output",
                str(output),
            ]
            prepared = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            self.assertEqual(json.loads(prepared.stdout)["act_rollouts_started"], 0)
            validated = subprocess.run(
                [
                    sys.executable,
                    str(
                        REPO_ROOT
                        / "experiments/paper/manipeval_evidence_manifest.py"
                    ),
                    "--repo-root",
                    str(root),
                    "validate",
                    "--manifest",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            self.assertEqual(json.loads(validated.stdout)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
