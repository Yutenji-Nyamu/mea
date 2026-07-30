import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paper.manipeval_replay_completed_tool import (
    CompletedToolReplayError,
    _evolved_query_contract,
    _exact_reuse_kind,
    _source_context,
    replay_completed_round_tool,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def run_local_execution_pair() -> tuple[dict, dict]:
    request = {
        "schema_version": 1,
        "task_name": "adjust_bottle",
        "metric": "query_smoothness",
        "question": "Was motion smooth?",
    }
    registration_id = "runlocal_1234567890abcdef"
    code_sha256 = "c" * 64
    first = {
        "status": "passed",
        "route": "provider_python_codegen",
        "tool_request": request,
        "source": {
            "scope": "run_local_generated",
            "tool": request["metric"],
            "registration_id": registration_id,
        },
        "route_decision": {
            "status": "resolved",
            "resolved_route": "provider_python_codegen",
            "task_name": request["task_name"],
            "metric": request["metric"],
        },
        "validation": {"provider_called": True},
        "episodes": [
            {
                "result": {
                    "tool": request["metric"],
                    "tool_sha256": code_sha256,
                    "value": 0.25,
                }
            }
        ],
        "artifacts": {"tool_execution": "first/tool_execution.json"},
    }
    replay = json.loads(json.dumps(first))
    replay.update(
        {
            "route": "run_local_reuse",
            "source": {
                "scope": "run_local_registry",
                "tool": request["metric"],
                "registration_id": registration_id,
                "tool_sha256": code_sha256,
            },
            "route_decision": {
                "status": "resolved",
                "resolved_route": "run_local_reuse",
                "task_name": request["task_name"],
                "metric": request["metric"],
            },
            "validation": {"provider_called": False},
            "artifacts": {
                "tool_execution": "second/tool_execution.json"
            },
        }
    )
    return first, replay


class CompletedToolReplayTests(unittest.TestCase):
    def test_trusted_catalog_identity_is_exact_reuse(self):
        request = {
            "schema_version": 1,
            "task_name": "grab_roller",
            "metric": "official_check_success",
            "question": "Did the official success predicate pass?",
        }
        first = {
            "status": "passed",
            "route": "reuse",
            "tool_request": request,
            "source": {
                "scope": "trusted_catalog",
                "tool": "official_check_success",
                "tool_sha256": "a" * 64,
            },
            "route_decision": {
                "status": "resolved",
                "resolved_route": "reuse",
                "task_name": request["task_name"],
                "metric": request["metric"],
                "exact_match": True,
            },
            "validation": {"provider_called": False},
        }
        replay = json.loads(json.dumps(first))

        self.assertEqual(
            _exact_reuse_kind(first, replay),
            "trusted_catalog",
        )
        replay["tool_request"]["metric"] = "different_metric"
        self.assertIsNone(_exact_reuse_kind(first, replay))
        replay = json.loads(json.dumps(first))
        replay["source"].pop("tool_sha256")
        self.assertIsNone(_exact_reuse_kind(first, replay))

    def test_run_local_identity_must_be_stable_and_provider_free(self):
        first, replay = run_local_execution_pair()
        self.assertEqual(
            _exact_reuse_kind(first, replay),
            "run_local_registry",
        )

        replay["source"]["registration_id"] = "runlocal_different"
        self.assertIsNone(_exact_reuse_kind(first, replay))
        _, replay = run_local_execution_pair()
        replay["validation"]["provider_called"] = True
        self.assertIsNone(_exact_reuse_kind(first, replay))

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
            first, second = run_local_execution_pair()
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

    def test_failed_parent_can_recover_completed_child_without_round_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation, _request_bundle = self.make_source(root)
            summary_path = evaluation / "summary/round_1.json"
            summary_path.unlink()
            write_json(
                evaluation / "manifest.json",
                {
                    "status": "failed",
                    "failure_stage": "round_1_execution",
                    "user_request": "Does the motion remain smooth?",
                    "evaluation_target": {"task_name": "adjust_bottle"},
                },
            )
            write_json(
                evaluation / "execution/round_1/child_run.json",
                {
                    "run_id": "child_1",
                    "returncode": 0,
                    "status": "completed",
                },
            )

            context = _source_context(root, "eval_source", "round_1")

            self.assertFalse(context["summary_available"])
            self.assertIsNone(context["summary_path"])
            self.assertEqual(
                context["summary"]["source"],
                "synthesized_from_completed_child_run",
            )
            self.assertEqual(
                context["summary"]["taskgen_run_id"],
                "child_1",
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
            first, second = run_local_execution_pair()
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
