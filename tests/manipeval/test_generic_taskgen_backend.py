from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from mea.planner.experiment_candidate import build_experiment_candidate
from mea.taskgen.generic_backend import (
    GenericRoboTwinTaskAdapter,
    GenericRoboTwinTaskGenBackend,
    GenericTaskGenError,
    GenericTaskGenHooks,
    build_generic_task_subclass_module,
    generic_task_semantic_key,
    load_generic_robotwin_task_adapter,
    validate_generic_task_methods,
)
from mea.taskgen.provider_scene_checker import validate_method_ast


class _Provider:
    def __init__(self, responses: list[dict[str, str]]) -> None:
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []
        self.last_metadata: dict[str, Any] = {}

    def text(self, prompt: str, **_kwargs: Any) -> str:
        self.prompts.append(prompt)
        response = self.responses[self.calls]
        self.calls += 1
        self.last_metadata = {"call": self.calls}
        return json.dumps(response)


def _write_cold_task_repo(root: Path) -> None:
    source = root / "envs/cold_unseen_task.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
class cold_unseen_task:
    def load_actors(self):
        self.target = "official"

    def check_success(self):
        return self.target == "official"
""".lstrip(),
        encoding="utf-8",
    )
    docs = root / "description/cold_unseen_task.md"
    docs.parent.mkdir(parents=True)
    docs.write_text(
        "The target string is the observable scene state.\n",
        encoding="utf-8",
    )
    asset = root / "assets/cold_target.asset"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"cold-target")
    readme = root / "mea/taskgen/README.Agent.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(
        "Generate bounded RoboTwin task methods.\n",
        encoding="utf-8",
    )


def _write_discoverable_task_repo(root: Path, task_name: str) -> None:
    source = root / "envs" / f"{task_name}.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        f"""
class {task_name}:
    def load_actors(self):
        self.target = create_actor(modelname="900_novel_target")

    def check_success(self):
        return self.target.is_ready()
""".lstrip(),
        encoding="utf-8",
    )
    schema = root / "mea/toolkit/schemas" / f"{task_name}.json"
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_name": task_name,
                "task_family": "runtime_discovered",
                "trusted_tool_profile": "generic_success",
                "physics_timestep_seconds": 0.004,
                "action_dimension": 14,
                "probe_task_attributes": ["target"],
                "tracked_actors": [
                    {
                        "id": "target",
                        "task_attribute": "target",
                        "scene_name": "900_novel_target",
                        "functional_points": [],
                        "contact_points": [],
                    }
                ],
                "contact_focus_actor_ids": ["target"],
                "semantic_fields": [
                    {
                        "name": "target_position",
                        "source": "actor_position",
                        "actor_id": "target",
                    }
                ],
                "semantic_roles": {
                    "manipulated_object_position": "target_position"
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    instruction = (
        root / "description/task_instruction" / f"{task_name}.json"
    )
    instruction.parent.mkdir(parents=True, exist_ok=True)
    instruction.write_text(
        json.dumps({"full_description": "operate the novel target"}) + "\n",
        encoding="utf-8",
    )
    asset = (
        root
        / "description/objects_description/900_novel_target/base0.json"
    )
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text(
        json.dumps({"description": "novel target asset"}) + "\n",
        encoding="utf-8",
    )
    readme = root / "mea/taskgen/README.Agent.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "Generate bounded RoboTwin task methods.\n",
        encoding="utf-8",
    )


def _candidate() -> dict[str, Any]:
    return build_experiment_candidate(
        candidate_id="dynamic.cold_unseen_task.contact_stability",
        source_query="Does this policy remain stable before target contact?",
        base_task="cold_unseen_task",
        semantic_concern="contact stability under a shifted target",
        scene_need="Move the existing target while preserving its identity.",
        checker_need="Require target completion without pre-contact failure.",
        tool_need="Measure pre-contact motion stability.",
    )


def _validate_methods(
    methods: Mapping[str, str],
    _candidate_value: Mapping[str, Any],
) -> dict[str, Any]:
    parsed = {
        name: validate_method_ast(
            source,
            name,
            safe_direct_calls=set(),
            safe_module_calls=set(),
            safe_method_calls=set(),
            allowed_private_attributes=set(),
            error_type=GenericTaskGenError,
        )
        for name, source in methods.items()
    }
    namespace: dict[str, Any] = {}
    exec(compile(methods["load_actors"], "<scene-fixture>", "exec"), namespace)
    exec(compile(methods["check_success"], "<checker-fixture>", "exec"), namespace)
    task = SimpleNamespace()
    namespace["load_actors"](task)
    positive = bool(namespace["check_success"](task))
    task.target = ""
    negative = bool(namespace["check_success"](task))
    fixtures = [
        {
            "fixture": "generated_scene_succeeds",
            "observed": positive,
            "expected": True,
            "passed": positive is True,
        },
        {
            "fixture": "missing_target_fails",
            "observed": negative,
            "expected": False,
            "passed": negative is False,
        },
    ]
    return {
        "valid": all(item["passed"] for item in fixtures),
        "policy": "cold_task_shared_safe_ast_fixture_v1",
        "scene_ast_nodes": sum(
            1 for _ in ast.walk(parsed["load_actors"])
        ),
        "success_ast_nodes": sum(
            1 for _ in ast.walk(parsed["check_success"])
        ),
        "checker_fixtures": fixtures,
    }


def _build_module(
    methods: Mapping[str, str],
    _candidate_value: Mapping[str, Any],
) -> str:
    scene = "\n".join(
        "    " + line for line in methods["load_actors"].splitlines()
    )
    checker = "\n".join(
        "    " + line for line in methods["check_success"].splitlines()
    )
    return (
        "class cold_unseen_task:\n"
        + scene
        + "\n\n"
        + checker
        + "\n"
    )


def _preflight(
    attempt_dir: Path,
    module_source: str,
    _candidate_value: Mapping[str, Any],
) -> dict[str, Any]:
    render_passed = "render_bad" not in module_source
    (attempt_dir / "render_probe.json").write_text(
        json.dumps({"passed": render_passed}) + "\n",
        encoding="utf-8",
    )
    return {
        "render_passed": render_passed,
        "expert_passed": render_passed,
        "scene_change_passed": render_passed,
        "simulator_probes": 1,
        "expert_probes": 1,
    }


def _adapter() -> GenericRoboTwinTaskAdapter:
    return GenericRoboTwinTaskAdapter(
        task_name="cold_unseen_task",
        official_source="envs/cold_unseen_task.py",
        official_class="cold_unseen_task",
        task_schema={
            "tracked_actors": ["target"],
            "signals": ["target_position", "success"],
        },
        documentation_paths=("description/cold_unseen_task.md",),
        asset_paths=("assets/cold_target.asset",),
        hooks=GenericTaskGenHooks(
            validate_methods=_validate_methods,
            build_module=_build_module,
            preflight_candidate=_preflight,
            resolve_metric=lambda candidate: candidate["tool_need"],
            resolve_checker_contract=lambda candidate: {
                "semantic_concern": candidate["semantic_concern"],
                "experimental_success": True,
            },
            prompt_constraints=(
                "Use only self.target from the retrieved official task."
            ),
        ),
    )


class GenericTaskGenBackendTests(unittest.TestCase):
    def test_loader_discovers_unknown_task_without_a_core_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_name = "runtime_novel_task"
            _write_discoverable_task_repo(root, task_name)

            def fixtures(
                methods: Mapping[str, str],
                _candidate_value: Mapping[str, Any],
            ) -> list[Mapping[str, Any]]:
                namespace: dict[str, Any] = {}
                exec(methods["check_success"], namespace)
                positive = SimpleNamespace(target=SimpleNamespace())
                positive.target.is_ready = lambda: True
                negative = SimpleNamespace(target=SimpleNamespace())
                negative.target.is_ready = lambda: False
                checker = namespace["check_success"]
                return [
                    {
                        "fixture": "ready",
                        "passed": checker(positive) is True,
                    },
                    {
                        "fixture": "not_ready",
                        "passed": checker(negative) is False,
                    },
                ]

            adapter = load_generic_robotwin_task_adapter(
                root,
                task_name,
                checker_fixtures=fixtures,
                preflight_candidate=lambda _path, _source, _candidate_value: {
                    "render_passed": True,
                    "expert_passed": True,
                    "scene_change_passed": True,
                },
                resolve_metric=lambda _candidate_value: (
                    "runtime_novel_completion"
                ),
                resolve_checker_contract=lambda candidate_value: {
                    "semantic_concern": candidate_value[
                        "semantic_concern"
                    ]
                },
            )

            self.assertEqual(adapter.task_name, task_name)
            self.assertEqual(
                adapter.official_source,
                f"envs/{task_name}.py",
            )
            self.assertIn(
                f"description/task_instruction/{task_name}.json",
                adapter.documentation_paths,
            )
            self.assertIn(
                (
                    "description/objects_description/"
                    "900_novel_target/base0.json"
                ),
                adapter.asset_paths,
            )
            self.assertNotIn(
                "template",
                json.dumps(
                    {
                        "task": adapter.task_name,
                        "docs": adapter.documentation_paths,
                        "assets": adapter.asset_paths,
                    }
                ),
            )

            candidate = build_experiment_candidate(
                source_query="Can it operate an unseen shifted target?",
                base_task=task_name,
                semantic_concern="target shift response",
                scene_need="Move the target to a new valid pose.",
                checker_need="Require the generated target to be ready.",
                tool_need="Measure target completion.",
            )
            provider = _Provider(
                [
                    {
                        "load_actors": (
                            "def load_actors(self):\n"
                            "    self.target = create_actor("
                            'modelname="900_novel_target")\n'
                        ),
                        "check_success": (
                            "def check_success(self):\n"
                            "    return self.target.is_ready()\n"
                        ),
                    }
                ]
            )
            result = GenericRoboTwinTaskGenBackend(
                root, provider, model="fixture-model"
            ).materialize(
                candidate,
                adapter,
                run_id="run_runtime_novel_task",
            )
            self.assertEqual(result["status"], "generated")
            module = (
                Path(result["run_dir"]) / "task.py"
            ).read_text(encoding="utf-8")
            self.assertIn(
                f"from envs.{task_name} import *",
                module,
            )
            self.assertIn(
                f"class {task_name}("
                f"_official_task_module.{task_name}):",
                module,
            )

    def test_loader_requires_real_preflight_hook(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_discoverable_task_repo(root, "runtime_novel_task")
            with self.assertRaisesRegex(
                GenericTaskGenError, "preflight_candidate"
            ):
                load_generic_robotwin_task_adapter(
                    root,
                    "runtime_novel_task",
                    checker_fixtures=lambda _methods, _candidate: [],
                    preflight_candidate=None,  # type: ignore[arg-type]
                    resolve_metric=lambda _candidate: "metric",
                    resolve_checker_contract=lambda _candidate: {},
                )

    def test_common_ast_and_subclass_builder_are_data_derived(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_name = "runtime_novel_task"
            _write_discoverable_task_repo(root, task_name)
            methods = {
                "load_actors": (
                    "def load_actors(self):\n"
                    "    self.target = create_actor("
                    'modelname="900_novel_target")\n'
                ),
                "check_success": (
                    "def check_success(self):\n"
                    "    return self.target.is_ready()\n"
                ),
            }
            report = validate_generic_task_methods(
                methods,
                official_source=root / f"envs/{task_name}.py",
                official_class=task_name,
            )
            self.assertTrue(report["valid"])
            self.assertTrue(
                report["policy"].startswith(
                    "generic_official_api_ast_v1:"
                )
            )
            module = build_generic_task_subclass_module(
                methods,
                official_module=f"envs.{task_name}",
                official_class=task_name,
            )
            compile(module, "<runtime-novel-task>", "exec")
            self.assertIn("def mea_official_check_success(self):", module)
            self.assertIn(
                (
                    f"return _official_task_module.{task_name}."
                    "check_success(self)"
                ),
                module,
            )
            with self.assertRaisesRegex(
                GenericTaskGenError, "forbidden AST node Import"
            ):
                validate_generic_task_methods(
                    {
                        **methods,
                        "check_success": (
                            "def check_success(self):\n"
                            "    import os\n"
                            "    return True\n"
                        ),
                    },
                    official_source=root / f"envs/{task_name}.py",
                    official_class=task_name,
                )

    def test_semantic_reuse_ignores_query_wording_and_candidate_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            first = _candidate()
            second = {
                **first,
                "candidate_id": "dynamic.cold_unseen_task.rephrased",
                "source_query": (
                    "Rephrased: is motion stable before contact?"
                ),
                "tool_need": "Measure the same scene with a different metric.",
            }
            self.assertEqual(
                generic_task_semantic_key(
                    first, _adapter(), repo_root=root
                ),
                generic_task_semantic_key(
                    second, _adapter(), repo_root=root
                ),
            )

    def test_exact_semantic_reuse_skips_provider_and_has_no_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            provider = _Provider([])
            observed_queries: list[dict[str, Any]] = []

            def find_exact(query: Mapping[str, Any]) -> Mapping[str, Any]:
                observed_queries.append(dict(query))
                return {
                    "schema_version": 1,
                    "status": "approved",
                    "artifact_id": "task_artifact_cold_exact",
                    "semantic_key": query["semantic_key"],
                    "semantic_key_sha256": query["semantic_key_sha256"],
                }

            backend = GenericRoboTwinTaskGenBackend(
                root,
                provider,
                model="fixture-model",
                find_exact=find_exact,
            )
            result = backend.materialize(
                _candidate(), _adapter(), run_id="run_cold_exact"
            )

            self.assertEqual(result["status"], "reused")
            self.assertEqual(result["provider_call_count"], 0)
            self.assertEqual(provider.calls, 0)
            self.assertEqual(len(observed_queries), 1)
            serialized = json.dumps(observed_queries[0], sort_keys=True)
            self.assertNotIn("template", serialized)
            self.assertNotIn("aspect_id", serialized)
            self.assertFalse(
                (root / "mea/generated_tasks/run_cold_exact").exists()
            )

    def test_ablation_condition_never_reuses_complete_task_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            provider = _Provider(
                [
                    {
                        "load_actors": (
                            "def load_actors(self):\n"
                            '    self.target = "generated"\n'
                        ),
                        "check_success": (
                            "def check_success(self):\n"
                            '    return self.target == "generated"\n'
                        ),
                    }
                ]
            )
            lookup_calls = 0

            def find_exact(_query: Mapping[str, Any]) -> Mapping[str, Any]:
                nonlocal lookup_calls
                lookup_calls += 1
                raise AssertionError("ablation must not reuse an artifact")

            result = GenericRoboTwinTaskGenBackend(
                root,
                provider,
                model="fixture-model",
                find_exact=find_exact,
            ).materialize(
                _candidate(),
                _adapter(),
                run_id="run_cold_ablation",
                ablation_switches={
                    "rag": False,
                    "visual_self_check": True,
                    "readme_agent": True,
                },
            )

            self.assertEqual(result["status"], "generated")
            self.assertEqual(lookup_calls, 0)
            self.assertEqual(provider.calls, 1)

    def test_unseen_task_generates_after_one_render_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            provider = _Provider(
                [
                    {
                        "load_actors": (
                            "def load_actors(self):\n"
                            '    self.target = "render_bad"\n'
                        ),
                        "check_success": (
                            "def check_success(self):\n"
                            '    return self.target != ""\n'
                        ),
                    },
                    {
                        "load_actors": (
                            "def load_actors(self):\n"
                            '    self.target = "generated"\n'
                        ),
                        "check_success": (
                            "def check_success(self):\n"
                            '    return self.target != ""\n'
                        ),
                    },
                ]
            )
            backend = GenericRoboTwinTaskGenBackend(
                root, provider, model="fixture-model"
            )

            result = backend.materialize(
                _candidate(),
                _adapter(),
                run_id="run_cold_generated",
                max_regenerations=1,
            )

            self.assertEqual(result["status"], "generated")
            self.assertEqual(result["provider_call_count"], 2)
            self.assertEqual(result["local_regeneration_count"], 1)
            self.assertEqual(provider.calls, 2)
            self.assertTrue(result["validation"]["preflight"]["render_passed"])
            self.assertTrue(result["validation"]["preflight"]["expert_passed"])
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "task.py").is_file())
            self.assertTrue(
                (run_dir / "generic_taskgen_resolution.json").is_file()
            )
            self.assertEqual(
                result["candidate_manifest"]["task_name"],
                "cold_unseen_task",
            )
            self.assertTrue(
                result["candidate_manifest"]["codegen_provenance"][
                    "generated_by_model"
                ]
            )
            attempt_summary = json.loads(
                (
                    root
                    / "mea/generated_task_attempts/run_cold_generated"
                    / "task_generation_attempt_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(attempt_summary["attempt_count"], 2)
            self.assertEqual(
                attempt_summary["attempts"][0]["recovery_action"],
                "regenerate_candidate",
            )
            prompt = provider.prompts[0]
            self.assertIn("cold_unseen_task", prompt)
            self.assertIn("TASK TELEMETRY/EXECUTION SCHEMA", prompt)
            self.assertIn("description/cold_unseen_task.md", prompt)
            self.assertIn("assets/cold_target.asset", prompt)
            self.assertNotIn("template_id", prompt)
            self.assertNotIn("aspect_id", prompt)

    def test_candidate_must_bind_the_adapter_base_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_cold_task_repo(root)
            candidate = _candidate()
            candidate["base_task"] = "another_task"
            backend = GenericRoboTwinTaskGenBackend(
                root, _Provider([]), model="fixture-model"
            )
            with self.assertRaisesRegex(
                GenericTaskGenError, "base_task differs"
            ):
                backend.materialize(
                    candidate, _adapter(), run_id="run_wrong_task"
                )


if __name__ == "__main__":
    unittest.main()
