import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mea.planner import build_query_sufficiency_contract


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _evidence(candidate_id, outcome):
    return {
        "candidate_id": candidate_id,
        "outcome": outcome,
        "score": None,
        "diagnosis": None,
    }


def _ready_root(root: Path) -> None:
    schema = root / "mea/toolkit/schemas/click_bell.json"
    _write(schema, {"task_name": "click_bell", "task_family": "manipulation"})
    checkpoint = root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50"
    checkpoint.mkdir(parents=True)
    (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")
    (checkpoint / "policy_last.ckpt").write_bytes(b"weights")


class QuerySufficiencyCLITests(unittest.TestCase):
    SCRIPT = (
        Path(__file__).resolve().parents[2]
        / "scripts/manipeval_query_sufficiency.py"
    )

    def _run(self, root: Path, *, contract, evidence):
        _write(root / "inputs/contract.json", contract)
        _write(root / "inputs/evidence.json", evidence)
        return subprocess.run(
            [
                sys.executable,
                str(self.SCRIPT),
                "--repo-root",
                str(root),
                "--task-name",
                "click_bell",
                "--contract-json",
                "inputs/contract.json",
                "--candidate-evidence-json",
                "inputs/evidence.json",
                "--output-json",
                "outputs/assessment.json",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_compact_cached_fixture_runs_zero_act_assessment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _ready_root(root)
            contract = build_query_sufficiency_contract(
                "Does every position pass?",
                candidate_universe=[
                    "object_position.left_fixed",
                    "object_position.right_fixed",
                ],
                claim_type="universal",
                round_budget=2,
            )
            completed = self._run(
                root,
                contract=contract,
                evidence={
                    "schema_version": 1,
                    "source_kind": "cached_compact_evidence",
                    "candidate_evidence": [
                        _evidence("object_position.left_fixed", "fail")
                    ],
                    "completed_rounds": 1,
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(
                (root / "outputs/assessment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                output["execution_mode"], "cached_offline_0_act"
            )
            self.assertFalse(output["live_planner_changed"])
            self.assertEqual(output["provider_calls_started"], 0)
            self.assertEqual(output["simulator_calls_started"], 0)
            self.assertEqual(output["act_rollouts_started"], 0)
            self.assertEqual(
                output["assessment"]["stop_reason"], "evidence_sufficient"
            )
            self.assertEqual(output["assessment"]["claim_verdict"], "refuted")
            self.assertEqual(
                output["candidate_evidence_source_kind"],
                "cached_compact_evidence",
            )

    def test_bare_explicit_evidence_list_is_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _ready_root(root)
            contract = build_query_sufficiency_contract(
                "Does any official instance pass?",
                candidate_universe=[
                    "object_instance.base0",
                    "object_instance.base1",
                ],
                claim_type="existential",
                round_budget=2,
            )
            completed = self._run(
                root,
                contract=contract,
                evidence=[_evidence("object_instance.base0", "pass")],
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(
                output["candidate_evidence_source_kind"],
                "explicit_candidate_evidence_list",
            )
            self.assertEqual(output["assessment"]["claim_verdict"], "supported")

    def test_candidate_outside_bound_task_fails_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _ready_root(root)
            contract = build_query_sufficiency_contract(
                "Does any candidate pass?",
                candidate_universe=["outside.template"],
                claim_type="existential",
                round_budget=1,
            )
            completed = self._run(
                root,
                contract=contract,
                evidence=[],
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("leaves the bound task", completed.stderr)
            self.assertFalse((root / "outputs/assessment.json").exists())


if __name__ == "__main__":
    unittest.main()
