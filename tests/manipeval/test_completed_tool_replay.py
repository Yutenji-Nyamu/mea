import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paper.manipeval_replay_completed_tool import (
    CompletedToolReplayError,
    _evolved_query_contract,
    replay_completed_round_tool,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class CompletedToolReplayTests(unittest.TestCase):
    def test_replay_uses_contract_that_admitted_dynamic_round(self):
        initial = {
            "schema_version": 3,
            "candidate_universe": [],
            "required_coverage": {"candidate_ids": []},
        }
        dynamic = {
            "schema_version": 3,
            "candidate_universe": ["dynamic.one"],
            "required_coverage": {"candidate_ids": ["dynamic.one"]},
        }
        contract, source = _evolved_query_contract(
            {
                "round_decisions": [
                    {
                        "query_assessment": {"contract": dynamic},
                        "next_round": {"round_id": "round_2"},
                    }
                ]
            },
            round_id="round_2",
            initial_contract=initial,
        )
        self.assertEqual(contract, dynamic)
        self.assertEqual(
            source,
            "round_decision.query_assessment.contract",
        )

    def make_source(self, root: Path) -> tuple[Path, Path]:
        evaluation = root / "mea/evaluation_runs/eval_source"
        round_plan = {
            "round_id": "round_1",
            "template_id": "candidate.one",
            "sub_aspect": "trajectory",
            "task_instruction": "measure trajectory",
            "route": "official",
            "execution": {"backend": "act", "seeds": [1], "num_episodes": 1},
            "tool_request": {
                "schema_version": 1,
                "task_name": "adjust_bottle",
                "metric": "official_check_success",
                "question": "success?",
            },
        }
        write_json(
            evaluation / "manifest.json",
            {
                "user_request": "Does the motion remain smooth?",
                "evaluation_target": {"task_name": "adjust_bottle"},
            },
        )
        write_json(
            evaluation / "plan/evaluation_plan.json",
            {"rounds": [round_plan]},
        )
        write_json(
            evaluation / "plan/query_sufficiency_contract.json",
            {"schema_version": 2},
        )
        write_json(
            evaluation / "summary/round_1.json",
            {
                "round_id": "round_1",
                "taskgen_run_id": "child_1",
                "taskgen_returncode": 0,
                "observations": {"execution_vqa": {"status": "passed"}},
            },
        )
        write_json(
            evaluation
            / "execution/round_1/open_tool_request/tool_request_bundle.json",
            {
                "tool_request": {
                    "schema_version": 1,
                    "task_name": "adjust_bottle",
                    "metric": "invalid_original_metric",
                    "question": "Original request",
                }
            },
        )
        child = root / "mea/generated_tasks/child_1"
        write_json(child / "manifest.json", {"status": "completed"})
        write_json(
            child / "evaluation/telemetry/act/episode_000_seed_1/episode.json",
            {"seed": 1},
        )
        request_bundle = root / "request_bundle.json"
        write_json(
            request_bundle,
            {
                "tool_request": {
                    "schema_version": 1,
                    "task_name": "adjust_bottle",
                    "metric": "query_smoothness",
                    "question": "Was motion smooth?",
                }
            },
        )
        return evaluation, request_bundle

    def test_replay_is_append_only_and_records_exact_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation, request_bundle = self.make_source(root)
            source_summary = evaluation / "summary/round_1.json"
            original = source_summary.read_bytes()
            first = {
                "status": "passed",
                "route": "typed_metric_spec_compile",
                "validation": {"provider_called": False},
                "episodes": [{"result": {"value": 0.25}}],
                "artifacts": {"tool_execution": "first/tool_execution.json"},
            }
            second = {
                "status": "passed",
                "route": "run_local_reuse",
                "validation": {"provider_called": False},
                "episodes": [{"result": {"value": 0.25}}],
                "artifacts": {"tool_execution": "second/tool_execution.json"},
            }
            repaired_summary = {
                "round_id": "round_1",
                "pipeline_passed": True,
                "observations": {},
            }
            with (
                patch(
                    "experiments.paper.manipeval_replay_completed_tool."
                    "execute_tool_request",
                    side_effect=[first, second],
                ) as execute,
                patch(
                    "experiments.paper.manipeval_replay_completed_tool."
                    "aggregate_round_results",
                    return_value={"status": "passed", "metrics": []},
                ),
                patch(
                    "experiments.paper.manipeval_replay_completed_tool."
                    "summarize_round",
                    return_value=repaired_summary,
                ),
                patch(
                    "experiments.paper.manipeval_replay_completed_tool."
                    "build_claim_first_evidence_record",
                    return_value={
                        "evidence_packet": {"evidence_strength": "sufficient"}
                    },
                ),
                patch(
                    "experiments.paper.manipeval_replay_completed_tool."
                    "ClaimFirstRuntimeController"
                ) as controller,
            ):
                controller.return_value.observe.return_value = {
                    "assessment": {"evidence_sufficient": True}
                }
                result = replay_completed_round_tool(
                    root,
                    evaluation_id="eval_source",
                    round_id="round_1",
                    repair_id="repair_1",
                    tool_request_path=request_bundle,
                )
            self.assertEqual(result["act_rollouts_started"], 0)
            self.assertEqual(result["exact_reuse_route"], "run_local_reuse")
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(
                execute.call_args_list[0].args[3],
                execute.call_args_list[1].args[3],
            )
            self.assertEqual(
                execute.call_args_list[0].kwargs["run_local_registry_dir"],
                execute.call_args_list[1].kwargs["run_local_registry_dir"],
            )
            replayed_plans = controller.return_value.observe.call_args.args[0]
            self.assertEqual(
                replayed_plans[0]["tool_request"]["metric"],
                "query_smoothness",
            )
            self.assertEqual(source_summary.read_bytes(), original)
            provenance = json.loads(
                (
                    evaluation / "repairs/repair_1/repair_provenance.json"
                ).read_text(encoding="utf-8")
            )
            repaired_summary = json.loads(
                (
                    evaluation / "repairs/repair_1/repaired_round_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(
                repaired_summary["evidence_artifact_paths"][
                    "round_aggregate"
                ].endswith("repairs/repair_1/aggregate_result.json")
            )
            self.assertEqual(
                repaired_summary["evidence_artifact_paths"][
                    "tool_execution"
                ],
                "first/tool_execution.json",
            )
            self.assertEqual(provenance["status"], "completed")
            self.assertTrue(provenance["source_artifacts_immutable"])
            self.assertEqual(
                provenance["original_tool_request"]["status"],
                "recorded",
            )
            self.assertEqual(
                provenance["repaired_tool_request"]["request"]["metric"],
                "query_smoothness",
            )
            self.assertNotEqual(
                provenance["original_tool_request"]["sha256"],
                provenance["repaired_tool_request"]["sha256"],
            )
            with self.assertRaises(CompletedToolReplayError):
                replay_completed_round_tool(
                    root,
                    evaluation_id="eval_source",
                    round_id="round_1",
                    repair_id="repair_1",
                    tool_request_path=request_bundle,
                )

    def test_replay_can_compose_append_only_vqa_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _evaluation, request_bundle = self.make_source(root)
            replacement_vqa_path = root / "vqa_replay_manifest.json"
            replacement_vqa = {
                "status": "passed",
                "evidence_conflict": True,
                "observation": {"evidence_conflict": True},
            }
            write_json(
                replacement_vqa_path,
                {
                    "evidence_kind": "cached_rollout_dynamic_vqa_replay",
                    "result": replacement_vqa,
                },
            )
            first = {
                "status": "passed",
                "route": "typed_metric_spec_compile",
                "validation": {"provider_called": False},
                "episodes": [{"result": {"value": 0.25}}],
                "artifacts": {"tool_execution": "first/tool_execution.json"},
            }
            second = {
                "status": "passed",
                "route": "run_local_reuse",
                "validation": {"provider_called": False},
                "episodes": [{"result": {"value": 0.25}}],
                "artifacts": {"tool_execution": "second/tool_execution.json"},
            }
            with (
                patch(
                    "experiments.paper.manipeval_replay_completed_tool."
                    "execute_tool_request",
                    side_effect=[first, second],
                ),
                patch(
                    "experiments.paper.manipeval_replay_completed_tool."
                    "aggregate_round_results",
                    return_value={"status": "passed", "metrics": []},
                ),
                patch(
                    "experiments.paper.manipeval_replay_completed_tool."
                    "summarize_round",
                    return_value={
                        "round_id": "round_1",
                        "pipeline_passed": True,
                        "observations": {},
                    },
                ) as summarize,
                patch(
                    "experiments.paper.manipeval_replay_completed_tool."
                    "build_claim_first_evidence_record",
                    return_value={
                        "evidence_packet": {"evidence_strength": "conflicting"}
                    },
                ),
                patch(
                    "experiments.paper.manipeval_replay_completed_tool."
                    "ClaimFirstRuntimeController"
                ) as controller,
            ):
                controller.return_value.observe.return_value = {
                    "assessment": {"evidence_sufficient": False}
                }
                result = replay_completed_round_tool(
                    root,
                    evaluation_id="eval_source",
                    round_id="round_1",
                    repair_id="repair_with_vqa",
                    tool_request_path=request_bundle,
                    execution_vqa_path=replacement_vqa_path,
                )
            self.assertEqual(summarize.call_args.args[5], replacement_vqa)
            self.assertEqual(
                result["execution_vqa_source"]["source"],
                "append_only_replay",
            )


if __name__ == "__main__":
    unittest.main()
