import json
import tempfile
import unittest
from pathlib import Path

from mea.toolgen import (
    execute_tool_request,
    load_registry,
    pickup_to_contact_tool_request,
    request_candidate_promotion,
)
from tests.manipeval.test_tool_orchestration import (
    FakeProvider,
    NeverCalledProvider,
    generated_pickup_to_contact_source,
    write_episode,
)


class RunLocalRegistryTests(unittest.TestCase):
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
        self.evaluation_dir = self.root / "eval_local_reuse"
        self.source = generated_pickup_to_contact_source()

    def tearDown(self):
        self._temporary.cleanup()

    def _output(self, round_number):
        return (
            self.evaluation_dir
            / "execution"
            / f"round_{round_number}"
            / "planned_tool"
        )

    def _generate_first(self):
        provider = FakeProvider(f"```python\n{self.source}```")
        result = execute_tool_request(
            self.repo_root,
            self.child_run,
            self._output(1),
            pickup_to_contact_tool_request(),
            provider=provider,
            model="fake-toolgen",
        )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(result["route"], "force_codegen")
        return result

    def test_second_identical_contract_reuses_evaluation_local_tool(self):
        first = self._generate_first()
        provider = NeverCalledProvider()
        second = execute_tool_request(
            self.repo_root,
            self.child_run,
            self._output(2),
            pickup_to_contact_tool_request(),
            provider=provider,
            model="must-not-be-used",
        )

        self.assertEqual(provider.calls, 0)
        self.assertEqual(second["route"], "run_local_reuse")
        self.assertEqual(
            second["route_decision"]["matched_registry"],
            "evaluation_local_tool_registry",
        )
        self.assertFalse(second["route_decision"]["provider_called"])
        self.assertEqual(
            [item["result"] for item in second["episodes"]],
            [item["result"] for item in first["episodes"]],
        )
        self.assertEqual(second["source"]["scope"], "run_local_registry")
        self.assertTrue(second["validation"]["all_gates_passed"])

        registry_dir = self.evaluation_dir / "tool_registry"
        index = load_registry(registry_dir)
        self.assertEqual(len(index["entries"]), 1)
        entry = index["entries"][0]
        registration = json.loads(
            (registry_dir / entry["registration_artifact"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(registration["scope"], "run_local")
        self.assertEqual(registration["version"], 1)
        self.assertRegex(registration["code_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(registration["contract_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(
            registration["telemetry_schema_compatibility"][
                "compatibility_sha256"
            ],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            registration["required_signals"],
            registration["tool_contract"]["required_signals"],
        )
        self.assertIsNotNone(registration["generation"]["prompt_sha256"])
        self.assertTrue(registration["generation"]["source_examples"])
        self.assertEqual(
            registration["promotion"],
            {
                "current_scope": "run_local",
                "candidate": {"status": "not_requested"},
                "trusted": {"status": "not_requested"},
            },
        )

    def test_paraphrased_question_reuses_same_executable_contract(self):
        self._generate_first()
        request = pickup_to_contact_tool_request()
        request["question"] = "Measure the same executable metric in other words."
        provider = NeverCalledProvider()
        result = execute_tool_request(
            self.repo_root,
            self.child_run,
            self._output(2),
            request,
            provider=provider,
            model="fake-toolgen",
        )
        self.assertEqual(provider.calls, 0)
        self.assertEqual(result["route"], "run_local_reuse")
        self.assertEqual(
            result["route_decision"]["matched_registry"],
            "evaluation_local_tool_registry",
        )

    def test_changed_telemetry_schema_falls_back_to_codegen(self):
        self._generate_first()
        for episode in (self.act_episode, self.expert_episode):
            path = episode / "schema.json"
            schema = json.loads(path.read_text(encoding="utf-8"))
            schema["balanced_profile"] = "changed"
            path.write_text(json.dumps(schema), encoding="utf-8")
        provider = FakeProvider(f"```python\n{self.source}```")
        result = execute_tool_request(
            self.repo_root,
            self.child_run,
            self._output(2),
            pickup_to_contact_tool_request(),
            provider=provider,
            model="fake-toolgen",
        )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(result["route"], "force_codegen")

    def test_modified_registered_source_is_never_reused(self):
        self._generate_first()
        registry_dir = self.evaluation_dir / "tool_registry"
        index = load_registry(registry_dir)
        source_path = registry_dir / index["entries"][0]["source_artifact"]
        source_path.write_text(
            source_path.read_text(encoding="utf-8") + "\n# modified\n",
            encoding="utf-8",
        )
        provider = FakeProvider(f"```python\n{self.source}```")
        result = execute_tool_request(
            self.repo_root,
            self.child_run,
            self._output(2),
            pickup_to_contact_tool_request(),
            provider=provider,
            model="fake-toolgen",
        )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(result["route"], "force_codegen")
        self.assertEqual(len(load_registry(registry_dir)["entries"]), 2)

    def _promotion_evidence(self):
        return {
            "positive_examples": ["property_positive", "expert_rollout"],
            "negative_examples": ["no_pickup", "contact_before_pickup"],
            "determinism_passed": True,
            "oracle_agreement_passed": True,
            "real_rollouts": ["act/episode_000_seed_100000"],
        }

    def test_explicit_candidate_promotion_requires_bounded_evidence(self):
        self._generate_first()
        registry_dir = self.evaluation_dir / "tool_registry"
        registration_id = load_registry(registry_dir)["entries"][0][
            "registration_id"
        ]

        rejected = request_candidate_promotion(
            registry_dir,
            registration_id,
            {"positive_examples": [], "negative_examples": []},
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn(
            "at_least_one_real_rollout_required", rejected["reasons"]
        )
        unchanged = load_registry(registry_dir)["entries"][0]["promotion"]
        self.assertEqual(unchanged["candidate"]["status"], "not_requested")

        eligible = request_candidate_promotion(
            registry_dir,
            registration_id,
            self._promotion_evidence(),
        )
        self.assertEqual(eligible["status"], "eligible")
        promoted = load_registry(registry_dir)["entries"][0]["promotion"]
        self.assertEqual(promoted["candidate"]["status"], "eligible")
        self.assertEqual(
            promoted["trusted"]["status"],
            "requires_code_review_and_tests",
        )
        self.assertEqual(promoted["current_scope"], "run_local")

    def test_candidate_promotion_rejects_modified_code(self):
        self._generate_first()
        registry_dir = self.evaluation_dir / "tool_registry"
        entry = load_registry(registry_dir)["entries"][0]
        source_path = registry_dir / entry["source_artifact"]
        source_path.write_text(
            source_path.read_text(encoding="utf-8") + "\n# tampered\n",
            encoding="utf-8",
        )
        decision = request_candidate_promotion(
            registry_dir,
            entry["registration_id"],
            self._promotion_evidence(),
        )
        self.assertEqual(decision["status"], "rejected")
        self.assertIn("registered_code_integrity_failed", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
