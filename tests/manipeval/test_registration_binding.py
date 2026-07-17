import json
import tempfile
import unittest
from pathlib import Path

from scripts.manipeval_agent import build_taskgen_command, update_manifest as update_parent
from scripts.manipeval_taskgen import (
    bind_registration_to_episode_metadata,
    update_manifest as update_child,
    validate_registration_identity,
)


def registration() -> dict:
    return {
        "schema_version": 1,
        "registration_id": "pair_demo",
        "evidence_manifest_payload_sha256": "1" * 64,
        "command_plan_sha256": "2" * 64,
        "registered_route_sha256": "3" * 64,
        "checkpoint_file_set_sha256": "4" * 64,
        "source_artifact_file_set_sha256": "5" * 64,
        "base_commit": "a" * 40,
        "candidate_suite_sha256": "6" * 64,
        "trusted_catalog_sha256": "7" * 64,
        "trusted_template_contract_sha256": "8" * 64,
        "strategy": "fixed_predeclared_v1",
        "expected_evaluation_id": "eval_pair_demo_fixed",
        "expected_child_run_prefix": "run_pair_demo_fixed_",
    }


class RegistrationBindingTests(unittest.TestCase):
    def test_parent_child_episode_and_taskgen_argv_share_exact_identity(self):
        identity = validate_registration_identity(registration())
        round_plan = {
            "round_id": "round_1",
            "task_name": "click_bell",
            "route": "reuse",
            "task_instruction": "registered test",
            "template_id": "object_position.left_fixed",
            "variant_hint": {
                "bell": {"position_mode": "fixed", "xy": [-0.2, -0.08]}
            },
            "execution": {
                "backend": "act",
                "seeds": [100402],
                "num_episodes": 1,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, run_id = build_taskgen_command(
                root,
                "eval_pair_demo_fixed",
                round_plan,
                text_model="text",
                vision_model="vision",
                base_url=None,
                gpu=0,
                max_reflections=0,
                registration_identity=identity,
            )
            self.assertEqual(run_id, "run_pair_demo_fixed_round_1")
            value = json.loads(
                command[command.index("--registration-identity-json") + 1]
            )
            self.assertEqual(value, identity)
            validate_registration_identity(value, run_id=run_id)

            parent = root / "mea/evaluation_runs/eval_pair_demo_fixed"
            child = root / f"mea/generated_tasks/{run_id}"
            episode = child / "evaluation/telemetry/act/episode_1/episode.json"
            parent.mkdir(parents=True)
            episode.parent.mkdir(parents=True)
            (parent / "manifest.json").write_text("{}", encoding="utf-8")
            (child / "manifest.json").write_text("{}", encoding="utf-8")
            episode.write_text('{"policy_name":"ACT"}', encoding="utf-8")

            update_parent(parent, registration_identity=identity)
            update_child(child, registration_identity=identity)
            bind_registration_to_episode_metadata(child, identity)
            self.assertEqual(
                json.loads((parent / "manifest.json").read_text(encoding="utf-8"))[
                    "registration_identity"
                ],
                identity,
            )
            self.assertEqual(
                json.loads((child / "manifest.json").read_text(encoding="utf-8"))[
                    "registration_identity"
                ],
                identity,
            )
            self.assertEqual(
                json.loads(episode.read_text(encoding="utf-8"))[
                    "registration_identity"
                ],
                identity,
            )

    def test_wrong_child_prefix_and_registration_hash_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "registered parent"):
            validate_registration_identity(
                registration(), run_id="run_different_round_1"
            )
        value = registration()
        value["checkpoint_file_set_sha256"] = "short"
        with self.assertRaisesRegex(ValueError, "registration hash"):
            validate_registration_identity(value)


if __name__ == "__main__":
    unittest.main()
