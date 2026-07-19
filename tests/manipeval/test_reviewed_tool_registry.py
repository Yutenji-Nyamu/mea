import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from mea.toolgen import (
    ReviewedRegistryError,
    build_review_manifest_template,
    execute_tool_request,
    install_reviewed_registration,
    load_registry,
    load_reviewed_registry,
    pickup_to_contact_tool_request,
)
from tests.manipeval.test_tool_orchestration import (
    FakeProvider,
    NeverCalledProvider,
    generated_pickup_to_contact_source,
    write_episode,
)


class ReviewedToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.repo_root = Path(__file__).resolve().parents[2]
        self.child_run = self.root / "run_blue"
        self.act_episode = (
            self.child_run
            / "evaluation/telemetry/act/episode_000_seed_100000"
        )
        self.expert_episode = (
            self.child_run
            / "evaluation/telemetry/expert/episode_000_seed_100000"
        )
        write_episode(
            self.act_episode, policy_name="ACT", physical_contact=False
        )
        write_episode(
            self.expert_episode, policy_name="expert", physical_contact=True
        )
        self.source_evaluation = self.root / "eval_source"
        self.source_registry = self.source_evaluation / "tool_registry"
        self.reviewed_registry = self.root / "reviewed_registry"
        self.source = generated_pickup_to_contact_source()

    def tearDown(self):
        self._temporary.cleanup()

    def _generate_source(self):
        provider = FakeProvider(f"```python\n{self.source}```")
        output = (
            self.source_evaluation
            / "execution/round_1/planned_tool"
        )
        result = execute_tool_request(
            self.repo_root,
            self.child_run,
            output,
            pickup_to_contact_tool_request(),
            provider=provider,
            model="fake-toolgen",
        )
        self.assertEqual(provider.calls, 1)
        entry = load_registry(self.source_registry)["entries"][0]
        return result, entry["registration_id"]

    def _approved_manifest(self, registration_id):
        manifest = build_review_manifest_template(
            self.source_registry, registration_id
        )
        manifest.update(
            {
                "decision": "approved",
                "reviewer": {
                    "id": "unit-test-development-review",
                    "kind": "development_agent",
                },
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "notes": "Test-only explicit source and evidence review.",
            }
        )
        manifest["checks"] = {
            key: True for key in manifest["checks"]
        }
        return manifest

    def _install(self):
        _, registration_id = self._generate_source()
        manifest = self._approved_manifest(registration_id)
        match = install_reviewed_registration(
            self.source_registry,
            registration_id,
            manifest,
            self.reviewed_registry,
        )
        return match, registration_id, manifest

    def test_pending_or_hash_mismatched_review_never_installs(self):
        _, registration_id = self._generate_source()
        pending = build_review_manifest_template(
            self.source_registry, registration_id
        )
        with self.assertRaisesRegex(
            ReviewedRegistryError, "decision must be approved"
        ):
            install_reviewed_registration(
                self.source_registry,
                registration_id,
                pending,
                self.reviewed_registry,
            )
        self.assertFalse(self.reviewed_registry.exists())

        approved = self._approved_manifest(registration_id)
        approved["code_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ReviewedRegistryError, "does not match source registration"
        ):
            install_reviewed_registration(
                self.source_registry,
                registration_id,
                approved,
                self.reviewed_registry,
            )
        self.assertFalse(self.reviewed_registry.exists())

    @unittest.skipIf(os.name == "nt", "symlink creation is not portable on Windows")
    def test_install_rejects_symlinked_entries_directory(self):
        _, registration_id = self._generate_source()
        approved = self._approved_manifest(registration_id)
        external = self.root / "external_entries"
        external.mkdir()
        self.reviewed_registry.mkdir()
        (self.reviewed_registry / "entries").symlink_to(
            external, target_is_directory=True
        )
        with self.assertRaisesRegex(ReviewedRegistryError, "must not be a symlink"):
            install_reviewed_registration(
                self.source_registry,
                registration_id,
                approved,
                self.reviewed_registry,
            )
        self.assertEqual(list(external.iterdir()), [])

    def test_explicit_review_installs_minimal_immutable_entry(self):
        match, registration_id, manifest = self._install()
        index = load_reviewed_registry(self.reviewed_registry)
        self.assertEqual(index["scope"], "reviewed_persistent")
        self.assertEqual(len(index["entries"]), 1)
        entry = index["entries"][0]
        self.assertEqual(entry["status"], "approved")
        self.assertEqual(
            match["registration"]["source_registration_id"], registration_id
        )
        self.assertEqual(match["review_manifest"], manifest)
        self.assertEqual(
            sorted(path.name for path in match["source_path"].parent.iterdir()),
            [
                "generated_tool.py",
                "registration.json",
                "reviewed_manifest.json",
                "tool_spec.json",
            ],
        )

    def test_new_process_reuses_without_any_provider(self):
        self._install()
        output = self.root / "eval_new_process/execution/round_1/planned_tool"
        program = """
import json
import sys
from pathlib import Path
from mea.toolgen import execute_tool_request, pickup_to_contact_tool_request

result = execute_tool_request(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    pickup_to_contact_tool_request(),
    reviewed_registry_dir=Path(sys.argv[4]),
)
print(json.dumps({
    "route": result["route"],
    "provider_called": result["validation"]["provider_called"],
    "matched_registry": result["route_decision"]["matched_registry"],
    "all_gates_passed": result["validation"]["all_gates_passed"],
}))
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                program,
                str(self.repo_root),
                str(self.child_run),
                str(output),
                str(self.reviewed_registry),
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["route"], "reviewed_persistent_reuse")
        self.assertEqual(payload["matched_registry"], "reviewed_tool_registry")
        self.assertFalse(payload["provider_called"])
        self.assertTrue(payload["all_gates_passed"])

    def test_tampered_persistent_source_is_not_executed(self):
        match, _, _ = self._install()
        match["source_path"].write_text(
            match["source_path"].read_text(encoding="utf-8") + "\n# tampered\n",
            encoding="utf-8",
        )
        provider = FakeProvider(f"```python\n{self.source}```")
        output = self.root / "eval_tampered/execution/round_1/planned_tool"
        result = execute_tool_request(
            self.repo_root,
            self.child_run,
            output,
            pickup_to_contact_tool_request(),
            provider=provider,
            model="fake-toolgen",
            reviewed_registry_dir=self.reviewed_registry,
        )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(result["route"], "force_codegen")
        self.assertEqual(
            result["route_decision"]["reviewed_lookup"]["status"],
            "invalid_registry",
        )

    def test_paraphrased_question_reuses_reviewed_executable_contract(self):
        self._install()
        request = pickup_to_contact_tool_request()
        request["question"] = "The same executable metric, phrased differently."
        provider = NeverCalledProvider()
        output = self.root / "eval_changed/execution/round_1/planned_tool"
        result = execute_tool_request(
            self.repo_root,
            self.child_run,
            output,
            request,
            provider=provider,
            model="fake-toolgen",
            reviewed_registry_dir=self.reviewed_registry,
        )
        self.assertEqual(provider.calls, 0)
        self.assertEqual(result["route"], "reviewed_persistent_reuse")
        self.assertEqual(
            result["route_decision"]["matched_registry"],
            "reviewed_tool_registry",
        )


if __name__ == "__main__":
    unittest.main()
