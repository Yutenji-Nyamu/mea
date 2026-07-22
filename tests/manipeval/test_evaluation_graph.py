import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mea.evaluation_graph import (
    EvaluationGraphError,
    EvaluationGraphPlanner,
    EvaluationGraphSession,
    build_child_command_plan,
    child_outcome_from_evaluation,
    validate_evaluation_graph,
)
from mea.planner.catalog import build_act_catalog
from scripts.manipeval_evaluation_graph import _require_plan_identity


def make_catalog_root(root: Path) -> dict:
    for task, family in (
        ("click_bell", "button_interaction"),
        ("beat_block_hammer", "tool_use"),
    ):
        schema = root / f"mea/toolkit/schemas/{task}.json"
        schema.parent.mkdir(parents=True, exist_ok=True)
        schema.write_text(
            json.dumps({"task_name": task, "task_family": family}),
            encoding="utf-8",
        )
        checkpoint = root / f"policy/ACT/act_ckpt/act-{task}/demo_clean-50"
        checkpoint.mkdir(parents=True)
        (checkpoint / "policy_last.ckpt").write_bytes(b"weights")
        (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")
    return build_act_catalog(root)


def graph_value() -> dict:
    return {
        "schema_version": 1,
        "graph_id": "graph_test",
        "user_query": "How broadly does ACT generalize across object properties?",
        "evaluation_goal": "test two relevant task families",
        "max_children": 2,
        "nodes": [
            {
                "node_id": "node_click",
                "task_name": "click_bell",
                "requested_aspect_ids": ["object_position"],
                "activation": "initial",
                "rationale": "test spatial generalization first",
            },
            {
                "node_id": "node_bbh",
                "task_name": "beat_block_hammer",
                "requested_aspect_ids": ["object_appearance.color"],
                "activation": "if_previous_failed_or_uncertain",
                "rationale": "diagnose another object family when needed",
            },
        ],
    }


def outcome(*, success: float, answered: bool = True) -> dict:
    return {
        "schema_version": 1,
        "node_id": "node_click",
        "task_name": "click_bell",
        "evaluation_id": "eval_click",
        "pipeline_passed": True,
        "evidence_strength": "sufficient",
        "policy_success": success,
        "answered_query": answered,
        "summary": f"click success={success}",
    }


def child_binding() -> dict:
    return {
        "manifest_user_request": graph_value()["user_query"],
        "evidence_user_request": graph_value()["user_query"],
        "bound_task_name": "click_bell",
        "bound_requested_aspect_ids": ["object_position"],
        "planned_requested_aspect_ids": ["object_position"],
        "executed_aspect_ids": ["object_position"],
        "max_agent_rounds": 1,
        "plan_max_rounds": 1,
        "executed_rounds": 1,
    }


class FakeProvider:
    def __init__(self, value: dict):
        self.values = value if isinstance(value, list) else [value]
        self.calls = 0

    def text(self, *args, **kwargs):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return json.dumps(value)


class EvaluationGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = make_catalog_root(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_provider_plan_is_bounded_by_catalog(self):
        value = graph_value()
        planner = EvaluationGraphPlanner(FakeProvider(value), model="test-model")
        self.assertEqual(
            planner.plan(value["user_query"], self.catalog, graph_id="graph_test"),
            value,
        )
        commands = build_child_command_plan(
            value, self.catalog, repo_root="/repo"
        )
        self.assertEqual(len(commands), 2)
        self.assertIn("bounded_each_round", commands[0]["argv"])
        self.assertIn("--bound-requested-aspect-id", commands[0]["argv"])
        self.assertIn("object_position", commands[0]["argv"])
        self.assertNotIn("--execution-backend", commands[1]["argv"])
        self.assertEqual(
            commands[0]["argv"][commands[0]["argv"].index("--max-agent-rounds") + 1],
            "1",
        )
        self.assertEqual(commands[1]["execution_state"], "inert_until_parent_activates")

    def test_sufficient_success_stops_conditional_second_child(self):
        session = EvaluationGraphSession(graph_value(), self.catalog)
        snapshot = session.record(outcome(success=1.0))
        self.assertEqual(snapshot["status"], "completed")
        self.assertIsNone(snapshot["next_node"])
        self.assertIn("click success=1.0", snapshot["synthesis"]["strengths"])

    def test_provider_retries_one_out_of_budget_graph(self):
        invalid = graph_value()
        invalid["nodes"][0]["requested_aspect_ids"] = [
            "object_position",
            "object_instance",
        ]
        provider = FakeProvider([invalid, graph_value()])
        planner = EvaluationGraphPlanner(provider, model="test-model")
        result = planner.plan(
            graph_value()["user_query"], self.catalog, graph_id="graph_test"
        )
        self.assertEqual(result, graph_value())
        self.assertEqual(provider.calls, 2)
        self.assertEqual(len(planner.validation_errors), 1)

    def test_failure_activates_second_task_family(self):
        session = EvaluationGraphSession(graph_value(), self.catalog)
        snapshot = session.record(outcome(success=0.0))
        self.assertEqual(snapshot["status"], "awaiting_child")
        self.assertEqual(snapshot["next_node"]["task_name"], "beat_block_hammer")

    def test_rejects_unknown_aspect_and_wrong_outcome_task(self):
        value = graph_value()
        value["nodes"][0]["requested_aspect_ids"] = ["safety.unintended_contact"]
        with self.assertRaises(EvaluationGraphError):
            validate_evaluation_graph(value, self.catalog)
        value = graph_value()
        value["nodes"][0]["requested_aspect_ids"] = [
            "object_position",
            "object_instance",
        ]
        with self.assertRaisesRegex(EvaluationGraphError, "exactly one"):
            validate_evaluation_graph(value, self.catalog)
        session = EvaluationGraphSession(graph_value(), self.catalog)
        invalid = outcome(success=0.0)
        invalid["task_name"] = "beat_block_hammer"
        with self.assertRaises(EvaluationGraphError):
            session.record(invalid)

    def test_schema_is_strict_and_writes_back_canonical_identifiers(self):
        value = graph_value()
        value["schema_version"] = True
        with self.assertRaisesRegex(EvaluationGraphError, "schema_version"):
            validate_evaluation_graph(value, self.catalog)
        value = graph_value()
        value["max_children"] = True
        with self.assertRaisesRegex(EvaluationGraphError, "integer"):
            validate_evaluation_graph(value, self.catalog)

        value = graph_value()
        value["graph_id"] = " graph_test "
        value["nodes"][0]["node_id"] = " node_click "
        value["nodes"][0]["task_name"] = " click_bell "
        value["nodes"][0]["requested_aspect_ids"] = [" object_position "]
        normalized = validate_evaluation_graph(value, self.catalog)
        self.assertEqual(normalized["graph_id"], "graph_test")
        self.assertEqual(normalized["nodes"][0]["node_id"], "node_click")
        self.assertEqual(normalized["nodes"][0]["task_name"], "click_bell")
        self.assertEqual(
            normalized["nodes"][0]["requested_aspect_ids"], ["object_position"]
        )

    def test_proposal_json_identity_must_match_cli(self):
        value = graph_value()
        _require_plan_identity(
            value,
            graph_id=value["graph_id"],
            query=value["user_query"],
        )
        with self.assertRaisesRegex(EvaluationGraphError, "command-line identity"):
            _require_plan_identity(
                value,
                graph_id="graph_other",
                query=value["user_query"],
            )
        with self.assertRaisesRegex(EvaluationGraphError, "command-line identity"):
            _require_plan_identity(
                value,
                graph_id=value["graph_id"],
                query="another query",
            )

    def test_verified_child_adapter_preserves_policy_and_pipeline_distinction(self):
        node = graph_value()["nodes"][0]
        with patch(
            "mea.portfolio.load_child_evaluation",
            return_value={
                "task_name": "click_bell",
                "evaluation_id": "eval_test_node_click",
                "pipeline_passed": False,
                "policy_success": 1.0,
                "act_rollouts_started": 1,
                "evaluation_binding": child_binding(),
            },
        ):
            converted = child_outcome_from_evaluation(
                self.root,
                graph_value(),
                self.catalog,
                node_id=node["node_id"],
                evaluation_id="eval_test_node_click",
            )
        self.assertFalse(converted["answered_query"])
        self.assertEqual(converted["evidence_strength"], "pipeline_invalid")
        self.assertEqual(converted["policy_success"], 1.0)

    def test_child_adapter_rejects_unrelated_or_unbound_evidence(self):
        value = graph_value()
        with self.assertRaisesRegex(EvaluationGraphError, "graph-derived"):
            child_outcome_from_evaluation(
                self.root,
                value,
                self.catalog,
                node_id="node_click",
                evaluation_id="eval_unrelated_click",
            )

        valid_child = {
            "task_name": "click_bell",
            "evaluation_id": "eval_test_node_click",
            "pipeline_passed": True,
            "policy_success": 1.0,
            "act_rollouts_started": 1,
            "evaluation_binding": child_binding(),
        }
        for field, invalid in (
            ("manifest_user_request", "unrelated query"),
            ("bound_requested_aspect_ids", ["object_instance"]),
            ("executed_aspect_ids", ["object_instance"]),
            ("max_agent_rounds", 2),
            ("executed_rounds", 2),
        ):
            child = json.loads(json.dumps(valid_child))
            child["evaluation_binding"][field] = invalid
            with self.subTest(field=field), self.assertRaisesRegex(
                EvaluationGraphError, field
            ), patch("mea.portfolio.load_child_evaluation", return_value=child):
                child_outcome_from_evaluation(
                    self.root,
                    value,
                    self.catalog,
                    node_id="node_click",
                    evaluation_id="eval_test_node_click",
                )

        invalid_count = json.loads(json.dumps(valid_child))
        invalid_count["act_rollouts_started"] = 2
        with patch(
            "mea.portfolio.load_child_evaluation", return_value=invalid_count
        ), self.assertRaisesRegex(EvaluationGraphError, "exactly one ACT"):
            child_outcome_from_evaluation(
                self.root,
                value,
                self.catalog,
                node_id="node_click",
                evaluation_id="eval_test_node_click",
            )


if __name__ == "__main__":
    unittest.main()
