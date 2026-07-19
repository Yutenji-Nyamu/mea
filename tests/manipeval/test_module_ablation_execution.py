import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mea.module_ablation_execution import (
    FunctionalSwitchExecutionError,
    execute_module_ablation_schedule,
    resolve_condition_switches,
)
from mea.module_ablation_protocol import (
    _canonical_sha256,
    _contract_payload,
    _item_payload,
)


TASKGEN_CONDITIONS = {
    "complete": {
        "rag": True,
        "visual_self_check": True,
        "readme_agent": True,
    },
    "no_rag": {
        "rag": False,
        "visual_self_check": True,
        "readme_agent": True,
    },
    "no_visual_self_check": {
        "rag": True,
        "visual_self_check": False,
        "readme_agent": True,
    },
    "no_readme_agent": {
        "rag": True,
        "visual_self_check": True,
        "readme_agent": False,
    },
    "base": {
        "rag": False,
        "visual_self_check": False,
        "readme_agent": False,
    },
}

TOOLGEN_CONDITIONS = {
    "complete": {"rag": True},
    "no_rag": {"rag": False},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schedule(component: str, conditions: dict[str, dict[str, bool]]) -> dict:
    case_id = f"{component}_case"
    input_identity = {
        "request": f"deterministic {component} functional smoke",
        "task_name": "beat_block_hammer",
    }
    execution_identity = {
        "base_commit": "a" * 40,
        "runner": "mea/module_ablation_execution.py",
        "runner_sha256": "b" * 64,
        "provider_model": None,
        "config_sha256": "c" * 64,
        "seed": 100401,
    }
    condition_contracts = {
        component: [
            {
                "condition": condition,
                "description": f"test {condition}",
                "module_switches": switches,
            }
            for condition, switches in conditions.items()
        ]
    }
    matched_sets = [
        {
            "component": component,
            "case_id": case_id,
            "conditions": list(conditions),
            "schedule_item_ids": {
                condition: f"{component}.{condition}.{case_id}"
                for condition in conditions
            },
            "input_identity": input_identity,
            "input_identity_sha256": _canonical_sha256(input_identity),
            "execution_identity": execution_identity,
            "execution_identity_sha256": _canonical_sha256(execution_identity),
        }
    ]
    holder = {
        "study_id": f"{component}_functional_smoke",
        "artifact_root": f"artifacts/{component}_functional_smoke",
        "condition_contracts": condition_contracts,
        "matched_sets": matched_sets,
    }
    contract_hash = _canonical_sha256(_contract_payload(holder))
    items = []
    for condition, switches in conditions.items():
        artifact_dir = f"{holder['artifact_root']}/{component}/{condition}/{case_id}"
        item = {
            "schedule_item_id": f"{component}.{condition}.{case_id}",
            "schedule_contract_sha256": contract_hash,
            "status": "scheduled",
            "component": component,
            "condition": condition,
            "case_id": case_id,
            "input_identity": input_identity,
            "input_identity_sha256": _canonical_sha256(input_identity),
            "execution_identity": execution_identity,
            "execution_identity_sha256": _canonical_sha256(execution_identity),
            "module_switches": switches,
            "artifact_dir": artifact_dir,
            "expected_manifest": f"{artifact_dir}/manifest.json",
        }
        item["schedule_item_sha256"] = _canonical_sha256(_item_payload(item))
        items.append(item)
    return {
        "schema_version": 1,
        "protocol": "table3_module_ablation_schedule_v2",
        "status": "prepared",
        "mode": "prepare_only",
        "study_id": holder["study_id"],
        "artifact_root": holder["artifact_root"],
        "schedule_contract_sha256": contract_hash,
        "claim_scope": "prepare-only",
        "paper_table_eligible": False,
        "runtime": {
            "provider_called": False,
            "simulator_called": False,
            "act_rollouts_started": 0,
        },
        "condition_contracts": condition_contracts,
        "items": items,
        "matched_sets": matched_sets,
    }


class ModuleAblationExecutionTests(unittest.TestCase):
    def test_taskgen_paper_switches_are_actually_applied_and_traced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = _schedule("taskgen", TASKGEN_CONDITIONS)
            result = execute_module_ablation_schedule(root, schedule)
            self.assertEqual(result["selected_item_count"], 5)
            self.assertFalse(result["paper_table_eligible"])
            self.assertEqual(
                result["runtime"],
                {
                    "provider_called": False,
                    "simulator_called": False,
                    "act_rollouts_started": 0,
                },
            )
            by_condition = {row["condition"]: row for row in result["items"]}
            self.assertTrue(by_condition["complete"]["success"])
            self.assertFalse(by_condition["no_rag"]["success"])
            self.assertFalse(by_condition["no_visual_self_check"]["success"])
            self.assertFalse(by_condition["no_readme_agent"]["success"])
            self.assertFalse(by_condition["base"]["success"])
            self.assertEqual(by_condition["complete"]["call_counts"]["rag"], 1)
            self.assertEqual(by_condition["no_rag"]["call_counts"]["rag"], 0)
            self.assertEqual(
                by_condition["no_visual_self_check"]["call_counts"][
                    "visual_self_check"
                ],
                0,
            )
            self.assertEqual(
                by_condition["no_readme_agent"]["call_counts"]["readme_agent"],
                0,
            )

            complete_dir = root / result["items"][0]["manifest"]
            complete_dir = complete_dir.parent
            manifest = json.loads((complete_dir / "manifest.json").read_text())
            trace = json.loads((complete_dir / "execution_trace.json").read_text())
            outcome = json.loads((complete_dir / "outcome.json").read_text())
            self.assertEqual(manifest["claim_scope"], result["claim_scope"])
            self.assertFalse(manifest["paper_table_eligible"])
            self.assertNotEqual(
                result["development_artifact_root"], schedule["artifact_root"]
            )
            self.assertFalse((root / schedule["artifact_root"]).exists())
            self.assertTrue(trace["paper_table3_condition"])
            self.assertFalse(trace["paper_table_eligible"])
            self.assertEqual(trace["call_counts"], by_condition["complete"]["call_counts"])
            self.assertEqual(
                outcome["execution_trace_sha256"],
                _sha256(complete_dir / "execution_trace.json"),
            )
            self.assertEqual(
                manifest["result"]["outcome_evidence_sha256"],
                _sha256(complete_dir / "outcome.json"),
            )

    def test_toolgen_complete_and_no_rag_take_different_real_branches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = _schedule("toolgen", TOOLGEN_CONDITIONS)
            result = execute_module_ablation_schedule(root, schedule)
            by_condition = {row["condition"]: row for row in result["items"]}
            self.assertTrue(by_condition["complete"]["success"])
            self.assertFalse(by_condition["no_rag"]["success"])
            self.assertEqual(by_condition["complete"]["call_counts"]["rag"], 1)
            self.assertEqual(by_condition["no_rag"]["call_counts"]["rag"], 0)
            self.assertEqual(
                by_condition["complete"]["call_counts"]["independent_judge"], 1
            )
            self.assertEqual(
                by_condition["no_rag"]["call_counts"]["independent_judge"], 1
            )

    def test_legacy_switch_shape_is_explicitly_non_paper(self):
        resolved = resolve_condition_switches(
            "taskgen", "no_visual_gate", {"rag": True, "visual_gate": False}
        )
        self.assertFalse(resolved.paper_table3_condition)
        self.assertEqual(resolved.contract, "legacy_non_paper_compatibility")
        self.assertFalse(resolved.normalized["visual_self_check"])

    def test_append_only_and_switch_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = _schedule("toolgen", TOOLGEN_CONDITIONS)
            item_id = schedule["items"][0]["schedule_item_id"]
            execute_module_ablation_schedule(
                root, schedule, schedule_item_ids=[item_id]
            )
            with self.assertRaisesRegex(
                FunctionalSwitchExecutionError, "already exists"
            ):
                execute_module_ablation_schedule(
                    root, schedule, schedule_item_ids=[item_id]
                )

            changed = _schedule("toolgen", TOOLGEN_CONDITIONS)
            changed["items"][0]["module_switches"] = {"rag": False}
            with self.assertRaisesRegex(
                FunctionalSwitchExecutionError, "item hash mismatch"
            ):
                execute_module_ablation_schedule(Path(temporary) / "other", changed)

    def test_development_root_cannot_contain_formal_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = _schedule("toolgen", TOOLGEN_CONDITIONS)
            with self.assertRaisesRegex(
                FunctionalSwitchExecutionError, "must be disjoint"
            ):
                execute_module_ablation_schedule(
                    root,
                    schedule,
                    development_artifact_root="artifacts",
                )


if __name__ == "__main__":
    unittest.main()
