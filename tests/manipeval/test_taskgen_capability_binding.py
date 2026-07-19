import json
import tempfile
import unittest
from pathlib import Path

from mea.capability_adapter import (
    build_contract_tool_request,
    resolve_capability_contract,
    taskgen_route,
)
from mea.taskgen import build_variant_spec
from scripts.manipeval_agent import build_taskgen_command
from scripts.manipeval_taskgen import (
    prepare_planner_capability_binding,
    validate_planner_capability_binding,
)


class TaskGenCapabilityBindingTests(unittest.TestCase):
    def _run_dir(
        self, root: Path, spec: dict, manifest_updates: dict | None = None
    ) -> Path:
        run_dir = root / "mea/generated_tasks/run_binding"
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": "run_binding",
            "status": "generated",
            "task_name": spec.get("task_name"),
            "mode": spec.get("generation_mode"),
        }
        manifest.update(manifest_updates or {})
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        (run_dir / "variant_spec.json").write_text(
            json.dumps(spec), encoding="utf-8"
        )
        return run_dir

    def test_bbh_contract_rejects_model_changes_before_runtime(self):
        contract = resolve_capability_contract(
            "beat_block_hammer", "object_appearance.color_blue"
        )
        red = build_variant_spec(
            task_name="beat_block_hammer",
            variant_id="object_appearance.color_blue",
            capability_id="object_appearance.color",
            intent="model proposed a different color",
            changes={
                "block": {
                    "position_mode": "official_random",
                    "yaw_mode": "official_random",
                    "scale": 1.0,
                    "color": [1.0, 0.0, 0.0],
                }
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._run_dir(Path(temporary), red)
            with self.assertRaisesRegex(RuntimeError, "differs from planner"):
                validate_planner_capability_binding(
                    contract,
                    task_name="beat_block_hammer",
                    mode="force_codegen",
                    variant_id="object_appearance.color_blue",
                    run_dir=run_dir,
                )
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertNotIn("capability_contract_validation", manifest)

    def test_exact_codegen_and_reused_bbh_contracts_are_bound(self):
        cases = (
            ("object_appearance.color_blue", "force_codegen"),
            ("object_position.official_random", "reuse"),
        )
        for template_id, mode in cases:
            contract = resolve_capability_contract("beat_block_hammer", template_id)
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                spec = build_variant_spec(
                    task_name="beat_block_hammer",
                    variant_id=contract["taskgen"]["task_variant_id"],
                    capability_id=contract["taskgen"]["capability_id"],
                    intent="exact trusted blue variant",
                    changes=contract["taskgen"]["changes"],
                    generation_mode=contract["taskgen"]["generation_mode"],
                )
                run_dir = self._run_dir(
                    Path(temporary),
                    spec,
                    {
                        "task_module": (
                            "mea.generated_tasks.run_binding.task"
                            if mode == "force_codegen"
                            else "mea.tasks.beat_block_hammer"
                        ),
                        "variant_spec_authority": "planner_capability_contract",
                    },
                )
                result = validate_planner_capability_binding(
                    contract,
                    task_name="beat_block_hammer",
                    mode=mode,
                    variant_id="object_appearance.color_blue",
                    run_dir=run_dir,
                )
                self.assertEqual(result["status"], "passed")
                self.assertEqual(
                    result["variant_spec_authority"],
                    "planner_capability_contract",
                )
                manifest = json.loads((run_dir / "manifest.json").read_text())
                self.assertEqual(manifest["capability_contract"], contract)

        force_contract = resolve_capability_contract(
            "beat_block_hammer", "object_appearance.color_blue"
        )
        with self.assertRaisesRegex(RuntimeError, "conflicts with capability route"):
            prepare_planner_capability_binding(
                force_contract,
                task_name="beat_block_hammer",
                mode="reuse",
                variant_id="object_appearance.color_blue",
            )

    def test_agent_command_sends_contract_and_separates_template_variant(self):
        contract = resolve_capability_contract(
            "beat_block_hammer", "object_position.official_random"
        )
        plan = {
            "round_id": "round_2",
            "task_name": "beat_block_hammer",
            "task_instruction": "test position",
            "template_id": "object_position.official_random",
            "capability_id": contract["taskgen"]["capability_id"],
            "task_variant_id": contract["taskgen"]["task_variant_id"],
            "capability_contract": contract,
            "sub_aspect": contract["aspect"]["aspect_id"],
            "route": taskgen_route(contract),
            "variant_hint": contract["taskgen"]["changes"],
            "execution": {
                "backend": "act",
                "seeds": [7],
                "num_episodes": 1,
                "gates": contract["required_gates"],
            },
            "tool_request": build_contract_tool_request(contract),
            "vqa_phenomenon_ids": contract["vqa"]["phenomenon_ids"],
        }
        command, _ = build_taskgen_command(
            Path("/repo"),
            "eval_binding",
            plan,
            text_model="text",
            vision_model="vision",
            base_url=None,
            gpu=0,
            max_reflections=1,
        )
        self.assertEqual(
            command[command.index("--variant-id") + 1],
            "object_appearance.color_blue",
        )
        encoded = command[command.index("--capability-contract-json") + 1]
        self.assertEqual(json.loads(encoded), contract)

        for field, replacement in (
            ("tool_request", {"metric": "wrong"}),
            ("vqa_phenomenon_ids", ["wrong"]),
        ):
            tampered = json.loads(json.dumps(plan))
            tampered[field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "differ from capability contract"
            ):
                build_taskgen_command(
                    Path("/repo"),
                    "eval_binding",
                    tampered,
                    text_model="text",
                    vision_model="vision",
                    base_url=None,
                    gpu=0,
                    max_reflections=1,
                )

    def test_official_contract_binds_passthrough_identity(self):
        contract = resolve_capability_contract(
            "click_bell", "task_execution.official_baseline"
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "mea/generated_tasks/run_official"
            run_dir.mkdir(parents=True)
            spec = {
                "schema_version": 1,
                "task_name": "click_bell",
                "intent": "evaluate_official_task_unchanged",
                "generation_mode": "official",
                "changes": {},
                "preserve": ["official_task_source", "official_task_identity"],
            }
            manifest = {
                "run_id": "run_official",
                "status": "generated",
                "task_name": "click_bell",
                "task_module": "envs.click_bell",
                "mode": "official",
                "generation_kind": "official_passthrough",
                "static_validation": {
                    "official_passthrough": {
                        "valid": True,
                        "task_module": "envs.click_bell",
                    }
                },
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            (run_dir / "variant_spec.json").write_text(json.dumps(spec))
            (run_dir / "overlay.yml").write_text("{}\n")
            result = validate_planner_capability_binding(
                contract,
                task_name="click_bell",
                mode="official",
                variant_id=None,
                run_dir=run_dir,
            )
            self.assertEqual(result["variant_spec_authority"], "official_passthrough")

            tampered = json.loads((run_dir / "manifest.json").read_text())
            tampered["task_module"] = "envs.beat_block_hammer"
            (run_dir / "manifest.json").write_text(json.dumps(tampered))
            with self.assertRaisesRegex(RuntimeError, "official TaskGen artifact"):
                validate_planner_capability_binding(
                    contract,
                    task_name="click_bell",
                    mode="official",
                    variant_id=None,
                    run_dir=run_dir,
                )


if __name__ == "__main__":
    unittest.main()
