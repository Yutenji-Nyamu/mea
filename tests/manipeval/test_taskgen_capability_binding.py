import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mea.capability_adapter import (
    build_contract_tool_request,
    resolve_capability_contract,
    taskgen_route,
)
from mea.taskgen import TaskArtifactBundleError, build_variant_spec
from mea.proposals import materialize_round_proposals, task_proposal_from_contract
from scripts.manipeval_agent import (
    build_taskgen_command,
    validate_round_capability_contract,
)
from scripts.manipeval_taskgen import (
    prepare_planner_capability_binding,
    task_artifact_summary,
    validate_planner_capability_binding,
)


class TaskGenCapabilityBindingTests(unittest.TestCase):
    def test_task_artifact_summary_fails_with_typed_error(self):
        with self.assertRaisesRegex(
            TaskArtifactBundleError, "success semantics are missing"
        ):
            task_artifact_summary({"success_semantics": None})

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

    def test_reviewed_reuse_accepts_only_hash_pinned_variant_alias(self):
        contract = resolve_capability_contract(
            "beat_block_hammer", "object_scale.bounded_1_2"
        )
        proposal = task_proposal_from_contract(
            contract, intent="evaluate a semantically identical reviewed task"
        )
        spec = build_variant_spec(
            task_name="beat_block_hammer",
            variant_id="object_scale.legacy_reviewed_alias",
            capability_id=contract["taskgen"]["capability_id"],
            intent="legacy reviewed artifact metadata",
            changes=contract["taskgen"]["changes"],
            generation_mode="force_codegen",
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._run_dir(
                Path(temporary),
                spec,
                {
                    "task_module": "mea.generated_tasks.run_binding.task",
                    "generation_kind": "reviewed_generated_task_reuse",
                    "variant_spec_authority": "reviewed_task_registry",
                },
            )
            variant_path = run_dir / "variant_spec.json"
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["reviewed_task_registration"] = {
                "copied_files": {
                    "variant_spec.json": hashlib.sha256(
                        variant_path.read_bytes()
                    ).hexdigest()
                }
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = validate_planner_capability_binding(
                contract,
                task_name="beat_block_hammer",
                mode="force_codegen",
                variant_id=proposal["proposal_id"],
                run_dir=run_dir,
                task_proposal=proposal,
            )
            self.assertEqual(result["variant_spec_authority"], "reviewed_task_registry")
            self.assertEqual(
                result["materialized_task_variant_id"],
                "object_scale.legacy_reviewed_alias",
            )

            tampered = json.loads(variant_path.read_text())
            tampered["variant_id"] = "object_scale.unreviewed_alias"
            variant_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "registry-pinned"):
                validate_planner_capability_binding(
                    contract,
                    task_name="beat_block_hammer",
                    mode="force_codegen",
                    variant_id=proposal["proposal_id"],
                    run_dir=run_dir,
                    task_proposal=proposal,
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
            reviewed_task_registry=Path("/repo/mea/task_registry/reviewed"),
        )
        self.assertEqual(
            command[command.index("--variant-id") + 1],
            "object_appearance.color_blue",
        )
        encoded = command[command.index("--capability-contract-json") + 1]
        self.assertEqual(json.loads(encoded), contract)
        self.assertEqual(
            command[command.index("--reviewed-task-registry") + 1],
            str(Path("/repo/mea/task_registry/reviewed")),
        )

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

    def test_novel_task_proposal_uses_contract_as_envelope(self):
        contract = resolve_capability_contract(
            "click_bell", "object_position.left_fixed"
        )
        base_round = {
            "round_id": "round_1",
            "task_name": "click_bell",
            "task_instruction": "test a query-generated target position",
            "template_id": contract["template_id"],
            "capability_id": contract["taskgen"]["capability_id"],
            "task_variant_id": contract["taskgen"]["task_variant_id"],
            "capability_contract": contract,
            "sub_aspect": contract["aspect"]["aspect_id"],
            "aspect_id": contract["aspect"]["aspect_id"],
            "task_module": "mea.tasks.click_bell",
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
        task_proposal = {
            "schema_version": 1,
            "proposal_id": "object_position.run_local_midleft",
            "task_name": "click_bell",
            "aspect_id": "object_position",
            "intent": "test an unseen safe mid-left target position",
            "capability_id": "object_position.fixed_xy",
            "reuse_first": True,
            "changes": {
                "bell": {"position_mode": "fixed", "xy": [-0.14, -0.12]}
            },
            "preserve_success_semantics": True,
        }
        tool_proposal = {
            "schema_version": 1,
            "proposal_id": "object_position.run_local_midleft.tool",
            "task_name": "click_bell",
            "aspect_id": "object_position",
            "evaluation_goal": "measure reachability",
            "metric": "bell_active_tcp_min_xy_error",
            "question": "How close did the active TCP get to the bell?",
            "vqa_phenomenon_ids": ["bell_visibly_pressed"],
            "reuse_first": True,
        }
        round_plan = materialize_round_proposals(
            base_round, task_proposal, tool_proposal
        )
        validate_round_capability_contract(round_plan)
        self.assertEqual(
            round_plan["capability_contract"]["taskgen"]["changes"],
            contract["taskgen"]["changes"],
        )
        self.assertEqual(round_plan["variant_hint"], task_proposal["changes"])
        command, _ = build_taskgen_command(
            Path("/repo"),
            "eval_novel",
            round_plan,
            text_model="text",
            vision_model="vision",
            base_url=None,
            gpu=0,
            max_reflections=1,
        )
        self.assertEqual(
            command[command.index("--variant-id") + 1], task_proposal["proposal_id"]
        )
        encoded = command[command.index("--task-proposal-json") + 1]
        self.assertEqual(json.loads(encoded), task_proposal)
        _, trusted_spec = prepare_planner_capability_binding(
            contract,
            task_name="click_bell",
            mode="reuse",
            variant_id=task_proposal["proposal_id"],
            task_proposal=task_proposal,
        )
        self.assertEqual(trusted_spec["changes"], task_proposal["changes"])
        self.assertEqual(trusted_spec["variant_id"], task_proposal["proposal_id"])

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
