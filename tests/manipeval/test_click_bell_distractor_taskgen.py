from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mea.taskgen.click_bell_distractor import (
    ClickBellDistractorTaskGenError,
    build_click_bell_distractor_module,
    click_bell_distractor_rollout_execution,
    default_click_bell_distractor_proposal,
    materialize_click_bell_distractor_candidate,
    reference_click_bell_distractor_methods,
    validate_click_bell_distractor_methods,
)
from mea.capability_adapter import resolve_capability_contract
from mea.proposals import task_proposal_from_contract
from scripts.manipeval_taskgen import main as taskgen_main
from mea.toolkit.recorder import (
    RecorderError,
    _schema_with_task_tracked_actors,
)


class _Provider:
    def __init__(self, responses: list[dict[str, str]]) -> None:
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []
        self.last_metadata: dict[str, object] = {}

    def text(self, prompt: str, **_kwargs: object) -> str:
        self.prompts.append(prompt)
        response = self.responses[self.calls]
        self.calls += 1
        self.last_metadata = {"call": self.calls}
        return json.dumps(response)


def _write_official_click_bell(root: Path) -> None:
    path = root / "envs/click_bell.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
class click_bell:
    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        self.bell_id = np.random.choice([0, 1], 1)[0]
        self.bell = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )
        self.add_prohibit_area(self.bell, padding=0.07)
        self.check_arm_function = self.is_left_gripper_close

    def check_success(self):
        if self.stage_success_tag:
            return True
        if not self.check_arm_function():
            return False
        bell_pose = self.bell.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("050_bell")
        eps = [0.025, 0.025]
        for position in positions:
            if np.all(np.abs(position[:2] - bell_pose[:2]) < eps):
                self.stage_success_tag = True
                return True
        return False
""".lstrip(),
        encoding="utf-8",
    )


def _write_click_episode(
    root: Path,
    *,
    task_module: str,
    success: bool,
) -> Path:
    episode = root / "episode_000_seed_100405"
    episode.mkdir(parents=True)
    count = 3
    zero = np.zeros((count, 3), dtype=np.float32)
    np.savez(
        episode / "semantic_trace.npz",
        physics_step=np.arange(count, dtype=np.int64),
        policy_step=np.arange(count, dtype=np.int64),
        simulation_time_seconds=np.arange(count, dtype=float) * 0.004,
        success=np.asarray([False, False, success], dtype=bool),
        bell_position=zero,
        bell_contact_position=zero,
        left_tcp_position=zero,
        right_tcp_position=zero,
    )
    (episode / "episode.json").write_text(
        json.dumps(
            {
                "task_name": "click_bell",
                "task_module": task_module,
                "policy_name": "ACT",
                "seed": 100405,
                "success": success,
                "physics_steps": count - 1,
                "semantic_trace_rows": count,
            }
        ),
        encoding="utf-8",
    )
    (episode / "schema.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_name": "click_bell",
            }
        ),
        encoding="utf-8",
    )
    (episode / "states.csv").write_text(
        "policy_step\n0\n1\n2\n",
        encoding="utf-8",
    )
    events = (
        [
            {
                "type": "success_transition",
                "physics_step": 2,
                "policy_step": 2,
                "simulation_time_seconds": 0.008,
            }
        ]
        if success
        else []
    )
    (episode / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events),
        encoding="utf-8",
    )
    return episode


class ClickBellDistractorTaskGenTests(unittest.TestCase):
    def test_reference_scene_checker_compiles_and_passes_six_fixtures(self) -> None:
        proposal = default_click_bell_distractor_proposal()
        methods = reference_click_bell_distractor_methods(proposal)
        report = validate_click_bell_distractor_methods(methods, proposal)
        module = build_click_bell_distractor_module(methods)
        compile(
            module,
            "<click-bell-distractor>",
            "exec",
        )
        self.assertIn("mea_telemetry_tracked_actors", module)
        self.assertIn('"scene_name": "distractor_bell"', module)
        self.assertTrue(report["scene_fixture"]["passed"])
        self.assertEqual(report["checker_fixture_count"], 6)
        self.assertTrue(
            all(item["passed"] for item in report["checker_fixtures"])
        )
        self.assertTrue(report["model_written_python"])
        self.assertFalse(report["restricted_success_spec_compiler_used"])

    def test_generated_distractor_extends_future_recorder_schema(self) -> None:
        base = {
            "tracked_actors": [
                {
                    "id": "bell",
                    "task_attribute": "bell",
                    "scene_name": "050_bell",
                    "functional_points": [],
                    "contact_points": [0],
                }
            ],
            "contact_focus_actor_ids": ["bell"],
        }
        extension = [
            {
                "id": "distractor",
                "task_attribute": "distractor",
                "scene_name": "distractor_bell",
                "functional_points": (),
                "contact_points": (0,),
                "contact_focus": True,
            }
        ]
        task = SimpleNamespace(
            bell=object(),
            distractor=object(),
            mea_telemetry_tracked_actors=extension,
        )
        schema = _schema_with_task_tracked_actors(base, task)
        self.assertEqual(
            [item["id"] for item in schema["tracked_actors"]],
            ["bell", "distractor"],
        )
        self.assertEqual(
            schema["contact_focus_actor_ids"],
            ["bell", "distractor"],
        )
        duplicate = SimpleNamespace(
            bell=object(),
            mea_telemetry_tracked_actors=[
                {**extension[0], "id": "bell", "task_attribute": "bell"}
            ],
        )
        with self.assertRaisesRegex(RecorderError, "duplicates"):
            _schema_with_task_tracked_actors(base, duplicate)

    def test_ast_boundary_rejects_non_dialect_calls(self) -> None:
        proposal = default_click_bell_distractor_proposal()
        methods = reference_click_bell_distractor_methods(proposal)
        methods["check_success"] = (
            "def check_success(self):\n"
            "    return open('/tmp/result').read()\n"
        )
        with self.assertRaisesRegex(
            ClickBellDistractorTaskGenError, "forbidden|not allowed"
        ):
            validate_click_bell_distractor_methods(methods, proposal)

    def test_shared_attempt_controller_regenerates_at_most_once(self) -> None:
        proposal = default_click_bell_distractor_proposal()
        methods = reference_click_bell_distractor_methods(proposal)
        invalid = dict(methods)
        invalid["check_success"] = "def check_success(self):\n    return True\n"
        provider = _Provider([invalid, methods])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_official_click_bell(root)
            readme = root / "mea/taskgen/README.Agent.md"
            readme.parent.mkdir(parents=True)
            readme.write_text("Generate only the requested methods.\n")
            manifest = materialize_click_bell_distractor_candidate(
                repo_root=root,
                run_id="run_click_distractor_fixture",
                proposal=proposal,
                provider=provider,
                model="fixture-model",
            )
            run_dir = (
                root
                / "mea/generated_tasks/run_click_distractor_fixture"
            )
            attempt_summary = json.loads(
                (run_dir / "provider_attempts.json").read_text()
            )
            first_attempt = (
                root
                / "mea/generated_task_attempts/run_click_distractor_fixture"
                / "attempt_01"
            )
            first_prompt = (
                first_attempt / "provider_prompt.md"
            ).read_text(encoding="utf-8")
            proposal_saved = (first_attempt / "proposal.json").is_file()
        self.assertEqual(provider.calls, 2)
        self.assertEqual(attempt_summary["regenerations_used"], 1)
        self.assertEqual(manifest["task_name"], "click_bell")
        self.assertEqual(
            manifest["checker_contract"]["fixture_pass_count"], 6
        )
        self.assertEqual(
            manifest["codegen_provenance"]["local_regeneration_limit"], 1
        )
        self.assertIn("RETRIEVED OFFICIAL CLICK_BELL METHODS", first_prompt)
        self.assertIn("self.bell = create_actor(", first_prompt)
        self.assertIn(
            'self.get_gripper_actor_contact_position("050_bell")',
            first_prompt,
        )
        self.assertIn("SUPPORTED DELTA API", first_prompt)
        self.assertNotIn("_load_official_bell", first_prompt)
        self.assertEqual(first_prompt.count("def load_actors(self):"), 1)
        self.assertEqual(first_prompt.count("def check_success(self):"), 1)
        self.assertNotIn(
            "self._mea_distractor_contact_seen = bool(",
            first_prompt,
        )
        self.assertTrue(proposal_saved)

    def test_invalid_second_response_is_terminal_without_third_call(self) -> None:
        proposal = default_click_bell_distractor_proposal()
        invalid = {
            "load_actors": "def load_actors(self):\n    return None\n",
            "check_success": "def check_success(self):\n    return True\n",
        }
        provider = _Provider([invalid, invalid])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_official_click_bell(root)
            readme = root / "mea/taskgen/README.Agent.md"
            readme.parent.mkdir(parents=True)
            readme.write_text("Generate only the requested methods.\n")
            with self.assertRaises(ClickBellDistractorTaskGenError):
                materialize_click_bell_distractor_candidate(
                    repo_root=root,
                    run_id="run_click_distractor_failure",
                    proposal=proposal,
                    provider=provider,
                    model="fixture-model",
                )
        self.assertEqual(provider.calls, 2)

    def test_missing_official_rag_source_fails_before_provider(self) -> None:
        proposal = default_click_bell_distractor_proposal()
        provider = _Provider([reference_click_bell_distractor_methods(proposal)])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ClickBellDistractorTaskGenError,
                "RAG source is unavailable",
            ):
                materialize_click_bell_distractor_candidate(
                    repo_root=Path(directory),
                    run_id="run_must_not_call_provider",
                    proposal=proposal,
                    provider=provider,
                    model="fixture-model",
                )
        self.assertEqual(provider.calls, 0)

    def test_shared_run_id_gate_rejects_path_before_provider(self) -> None:
        proposal = default_click_bell_distractor_proposal()
        provider = _Provider(
            [reference_click_bell_distractor_methods(proposal)]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ClickBellDistractorTaskGenError, "run_id"
            ):
                materialize_click_bell_distractor_candidate(
                    repo_root=Path(directory),
                    run_id="../run_escape",
                    proposal=proposal,
                    provider=provider,
                    model="fixture-model",
                )
        self.assertEqual(provider.calls, 0)

    def test_production_cli_dispatches_click_provider_dialect(self) -> None:
        contract = resolve_capability_contract(
            "click_bell",
            "robustness.distractor_avoidance.lookalike_bell",
        )
        public = task_proposal_from_contract(
            contract,
            intent="test target selection with a lookalike bell",
        )
        bounded = default_click_bell_distractor_proposal()
        provider = _Provider(
            [reference_click_bell_distractor_methods(bounded)]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_official_click_bell(root)
            for relative in (
                "envs/beat_block_hammer.py",
                "policy/ACT/eval.sh",
                "script/eval_policy.py",
            ):
                protected = root / relative
                protected.parent.mkdir(parents=True, exist_ok=True)
                protected.write_text("# protected fixture\n", encoding="utf-8")
            readme = root / "mea/taskgen/README.Agent.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "Generate only the requested methods.\n",
                encoding="utf-8",
            )
            argv = [
                "manipeval_taskgen.py",
                "--repo-root",
                str(root),
                "--request",
                "Can the policy avoid a lookalike bell?",
                "--task-name",
                "click_bell",
                "--mode",
                "provider_scene_checker_codegen",
                "--run-id",
                "run_click_provider_cli",
                "--variant-id",
                public["proposal_id"],
                "--capability-contract-json",
                json.dumps(contract),
                "--task-proposal-json",
                json.dumps(public),
            ]
            with patch("sys.argv", argv), patch(
                "scripts.manipeval_taskgen.OpenAICompatibleProvider",
                return_value=provider,
            ):
                taskgen_main()
            manifest = json.loads(
                (
                    root
                    / "mea/generated_tasks/run_click_provider_cli/manifest.json"
                ).read_text(encoding="utf-8")
            )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(manifest["task_name"], "click_bell")
        self.assertEqual(
            manifest["generation_kind"],
            "provider_scene_checker_codegen",
        )
        self.assertEqual(
            manifest["provider_proposal_artifact"],
            "generation/click_bell_distractor_proposal.json",
        )

    def test_rollout_bridge_binds_generated_checker_and_latch_scope(self) -> None:
        proposal = default_click_bell_distractor_proposal()
        methods = reference_click_bell_distractor_methods(proposal)
        provider = _Provider([methods])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_official_click_bell(root)
            readme = root / "mea/taskgen/README.Agent.md"
            readme.parent.mkdir(parents=True)
            readme.write_text("Generate the requested methods.\n")
            manifest = materialize_click_bell_distractor_candidate(
                repo_root=root,
                run_id="run_click_rollout_bridge",
                proposal=proposal,
                provider=provider,
                model="fixture-model",
            )
            episode = _write_click_episode(
                root / "telemetry",
                task_module=manifest["task_module"],
                success=True,
            )
            execution = click_bell_distractor_rollout_execution(
                episode_dir=episode,
                candidate_dir=(
                    root
                    / "mea/generated_tasks/run_click_rollout_bridge"
                ),
            )
        result = execution["episodes"][0]["result"]
        self.assertTrue(result["value"])
        self.assertTrue(
            result["details"]["official_core_predicate_satisfied"]
        )
        self.assertFalse(
            result["details"]["distractor_contact_latched"]
        )
        self.assertEqual(
            result["details"]["distractor_latch_authority"],
            "logical_implication_of_validated_checker_success",
        )
        self.assertEqual(
            result["details"]["distractor_trace_coverage"],
            "not_registered_in_current_click_bell_task_schema",
        )


if __name__ == "__main__":
    unittest.main()
