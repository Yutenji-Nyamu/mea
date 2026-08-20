from __future__ import annotations

import json
import unittest
from pathlib import Path

from mea.toolgen.open_request import (
    OpenToolRequestAgent,
    OpenToolRequestError,
    validate_open_tool_request,
)
from mea.toolgen.router import route_tool_request


class _Provider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.last_metadata = {"provider": "fixture"}

    def text(self, *_args, **_kwargs):
        self.calls += 1
        return json.dumps(self.payload)


class _RetryProvider:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.prompts = []
        self.last_metadata = {"provider": "fixture"}

    def text(self, prompt, *_args, **_kwargs):
        self.prompts.append(prompt)
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return json.dumps(payload)


def _typed(task_name: str = "runtime_task"):
    return {
        "schema_version": 2,
        "task_name": task_name,
        "metric": "runtime_xy_gap",
        "question": "How close did the two actors get?",
        "metric_spec": {
            "schema_version": 1,
            "operation": "minimum_distance",
            "left_signal": "left_position",
            "right_signal": "right_position",
            "dimensions": ["x", "y"],
            "unit": "m",
            "null_semantics": "null_if_no_finite_sample",
        },
    }


def _terminal_difference(task_name: str = "adjust_bottle"):
    return {
        "schema_version": 2,
        "task_name": task_name,
        "metric": "terminal_bottle_to_right_tcp_height_difference",
        "question": (
            "What was the absolute terminal height difference between the "
            "bottle functional position and right TCP position?"
        ),
        "metric_spec": {
            "schema_version": 1,
            "operation": "terminal_signal_difference",
            "left_signal": "bottle_functional_position",
            "right_signal": "right_tcp_position",
            "component": "z",
            "absolute": True,
            "unit": "m",
            "null_semantics": "null_if_terminal_not_finite",
        },
    }


def _derived(task_name: str = "beat_block_hammer"):
    return {
        "schema_version": 2,
        "task_name": task_name,
        "metric": "query_peak_hammer_motion",
        "question": "What was the peak per-step hammer motion?",
        "metric_spec": {
            "schema_version": 2,
            "operation": "derived_observable",
            "observable_id": "query_peak_hammer_motion",
            "description": (
                "Maximum Euclidean displacement per positive physics step "
                "between consecutive hammer-position samples."
            ),
            "required_signals": ["hammer_position"],
            "unit": "m_per_step",
            "null_semantics": "null_if_no_finite_sample",
        },
    }


class OpenToolRequestTest(unittest.TestCase):
    def test_novel_metric_requires_typed_spec(self):
        with self.assertRaisesRegex(OpenToolRequestError, "typed MetricSpec"):
            validate_open_tool_request(
                {
                    "schema_version": 1,
                    "task_name": "runtime_task",
                    "metric": "unknown_metric",
                    "question": "Unknown?",
                },
                task_name="runtime_task",
            )

    def test_typed_request_is_task_bound(self):
        self.assertEqual(
            validate_open_tool_request(
                _typed(), task_name="runtime_task"
            )["metric"],
            "runtime_xy_gap",
        )
        with self.assertRaisesRegex(OpenToolRequestError, "changed the bound task"):
            validate_open_tool_request(
                _typed("other_task"), task_name="runtime_task"
            )

    def test_agent_can_propose_a_new_compositional_observable(self):
        root = Path(__file__).resolve().parents[2]
        provider = _Provider(_derived())
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")
        bundle = agent.propose(
            source_query="Where does the trajectory first become jerky?",
            semantic_concern="pre-contact motion quality",
            tool_need="Measure the peak per-step hammer motion.",
            task_name="beat_block_hammer",
            derived_observable_oracle_available=True,
        )

        spec = bundle["tool_request"]["metric_spec"]
        self.assertEqual(spec["schema_version"], 2)
        self.assertEqual(spec["operation"], "derived_observable")
        self.assertEqual(spec["required_signals"], ["hammer_position"])
        self.assertIn("directly implement this description", agent.last_prompt)
        self.assertNotIn("semantic review", agent.last_prompt)
        self.assertEqual(
            route_tool_request(bundle["tool_request"])["route_decision"][
                "resolved_route"
            ],
            "typed_metric_spec_execute",
        )

    def test_derived_observable_is_advertised_with_toolgen_validation(self):
        root = Path(__file__).resolve().parents[2]
        provider = _Provider(_derived())
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")
        bundle = agent.propose(
            source_query="Where does the trajectory first become jerky?",
            semantic_concern="pre-contact motion quality",
            tool_need="Measure the peak per-step hammer motion.",
            task_name="beat_block_hammer",
        )
        self.assertEqual(
            bundle["tool_request"]["metric_spec"]["operation"],
            "derived_observable",
        )
        self.assertIn(
            'metric_spec.operation exactly "derived_observable"',
            agent.last_prompt,
        )
        self.assertIn(
            "description MUST contain 1-240 characters",
            agent.last_prompt,
        )
        self.assertIn(
            "null_semantics MUST be exactly null_if_no_finite_sample",
            agent.last_prompt,
        )

    def test_derived_observable_repair_repeats_complete_contract(self):
        root = Path(__file__).resolve().parents[2]
        invalid = _derived()
        invalid["metric_spec"]["description"] = "x" * 241
        provider = _RetryProvider([invalid, _derived()])
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")

        bundle = agent.propose(
            source_query="Where does the trajectory first become jerky?",
            semantic_concern="pre-contact motion quality",
            tool_need="Measure the peak per-step hammer motion.",
            task_name="beat_block_hammer",
            derived_observable_oracle_available=True,
        )

        self.assertEqual(
            bundle["tool_request"]["metric_spec"]["operation"],
            "derived_observable",
        )
        self.assertEqual(provider.calls, 2)
        self.assertIn(
            "description must contain 1-240 characters",
            provider.prompts[1],
        )
        self.assertIn(
            "null_semantics must be exactly null_if_no_finite_sample",
            provider.prompts[1],
        )
        self.assertIn("PREVIOUS COMPLETE TOOL REQUEST JSON", provider.prompts[1])
        self.assertIn('"description": "' + ("x" * 241), provider.prompts[1])

    def test_derived_observable_identity_matches_the_tool_metric(self):
        request = _derived()
        request["metric_spec"]["observable_id"] = "different_observable"
        with self.assertRaisesRegex(
            OpenToolRequestError,
            "observable_id must equal the Tool metric",
        ):
            validate_open_tool_request(
                request,
                task_name="beat_block_hammer",
                derived_observable_oracle_available=True,
            )

    def test_agent_uses_schema_not_task_catalog(self):
        root = Path(__file__).resolve().parents[2]
        payload = _typed("adjust_bottle")
        payload["metric_spec"]["left_signal"] = "left_tcp_position"
        payload["metric_spec"]["right_signal"] = "bottle_position"
        provider = _Provider(payload)
        agent = OpenToolRequestAgent(
            root, provider, model="fixture-model"
        )
        bundle = agent.propose(
            source_query="Which condition fails?",
            semantic_concern="runtime gap",
            tool_need="measure actor proximity",
            task_name="adjust_bottle",
            reusable_tool_requests=[
                {
                    "registration_id": "runlocal_fixture",
                    "request": payload,
                    "validation": {"scope": "evaluation_local"},
                }
            ],
        )
        self.assertEqual(
            bundle["tool_request"]["metric"], "runtime_xy_gap"
        )
        self.assertEqual(provider.calls, 1)
        self.assertNotIn(
            "template",
            json.dumps(bundle["context"], ensure_ascii=False),
        )
        self.assertEqual(
            bundle["context"]["validated_generated_tools"][0][
                "registration_id"
            ],
            "runlocal_fixture",
        )

    def test_agent_rejects_missing_schema_signal(self):
        root = Path(__file__).resolve().parents[2]
        payload = _typed("adjust_bottle")
        provider = _Provider(payload)
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")
        with self.assertRaisesRegex(
            OpenToolRequestError, "unavailable telemetry signals"
        ):
            agent.propose(
                source_query="Which condition fails?",
                semantic_concern="runtime gap",
                tool_need="measure actor proximity",
                task_name="adjust_bottle",
            )

    def test_second_query_exact_generated_difference_request_remains_valid(self):
        root = Path(__file__).resolve().parents[2]
        request = _terminal_difference()
        provider = _Provider(request)
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")
        registration = {
            "registration_id": "runlocal_terminal_difference",
            "request": request,
            "validation": {"scope": "evaluation_local", "status": "validated"},
        }

        bundle = agent.propose(
            source_query=(
                "At the terminal state, compare object and gripper heights."
            ),
            semantic_concern="terminal relative object state",
            tool_need=(
                "At the terminal state, reuse the absolute height difference "
                "between bottle functional position and right TCP position."
            ),
            task_name="adjust_bottle",
            reusable_tool_requests=[registration],
        )

        self.assertEqual(bundle["tool_request"], request)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(
            bundle["context"]["validated_generated_tools"],
            [registration],
        )

    def test_invalid_metric_id_is_repaired_inside_provider_loop(self):
        root = Path(__file__).resolve().parents[2]
        invalid = _typed("adjust_bottle")
        invalid["metric"] = "Bottle Gap"
        invalid["metric_spec"]["left_signal"] = "left_tcp_position"
        invalid["metric_spec"]["right_signal"] = "bottle_position"
        repaired = json.loads(json.dumps(invalid))
        repaired["metric"] = "bottle_gap"
        provider = _RetryProvider([invalid, repaired])
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")

        bundle = agent.propose(
            source_query="Which condition fails?",
            semantic_concern="runtime gap",
            tool_need="measure actor proximity",
            task_name="adjust_bottle",
        )

        self.assertEqual(bundle["tool_request"]["metric"], "bottle_gap")
        self.assertEqual(provider.calls, 2)
        self.assertIn("lower_snake_case", bundle["provider"]["errors"][0])

    def test_bound_task_and_question_are_filled_for_metric_only_selection(self):
        root = Path(__file__).resolve().parents[2]
        provider = _Provider(
            {
                "schema_version": 1,
                "metric": "bell_active_tcp_min_xy_error",
            }
        )
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")

        bundle = agent.propose(
            source_query="Where does bell recoloring fail?",
            semantic_concern="bounded bell recoloring",
            tool_need="Measure click-point accuracy with the active TCP.",
            task_name="click_bell",
        )

        self.assertEqual(bundle["tool_request"]["task_name"], "click_bell")
        self.assertEqual(
            bundle["tool_request"]["question"],
            "Measure click-point accuracy with the active TCP.",
        )
        self.assertEqual(
            bundle["provider"]["bound_fields_filled"],
            ["task_name", "question"],
        )

    def test_generated_checker_context_forbids_official_success_alias(self):
        root = Path(__file__).resolve().parents[2]
        forbidden = {
            "schema_version": 1,
            "task_name": "adjust_bottle",
            "metric": "official_check_success",
            "question": "Did the task succeed?",
        }
        repaired = {
            "schema_version": 1,
            "task_name": "adjust_bottle",
            "metric": "generated_check_success",
            "question": "Did the generated checker pass?",
        }
        provider = _RetryProvider([forbidden, repaired])
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")

        bundle = agent.propose(
            source_query="Which condition fails?",
            semantic_concern="query-derived experimental scene",
            tool_need="measure the generated checker outcome",
            task_name="adjust_bottle",
            generated_checker_semantics=True,
        )

        self.assertEqual(
            bundle["tool_request"]["metric"],
            "generated_check_success",
        )
        self.assertEqual(provider.calls, 2)
        registry_names = {
            item["name"]
            for item in bundle["context"]["tool_registry"]["trusted_tools"]
        }
        self.assertNotIn("official_check_success", registry_names)
        self.assertNotIn("time_to_success", registry_names)

    def test_query_induced_tool_does_not_duplicate_already_measured_metrics(self):
        root = Path(__file__).resolve().parents[2]
        duplicate = {
            "schema_version": 1,
            "task_name": "click_bell",
            "metric": "official_check_success",
            "question": "Did the task succeed?",
        }
        additional = {
            "schema_version": 2,
            "task_name": "click_bell",
            "metric": "recolored_bell_right_tcp_min_xy_error",
            "question": "How accurately did the right TCP reach the bell?",
            "metric_spec": {
                "schema_version": 1,
                "operation": "minimum_distance",
                "left_signal": "right_tcp_position",
                "right_signal": "bell_contact_position",
                "dimensions": ["x", "y"],
                "unit": "m",
                "null_semantics": "null_if_no_finite_sample",
            },
        }
        provider = _RetryProvider([duplicate, additional])
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")

        bundle = agent.propose(
            source_query="Where does color generalization fail?",
            semantic_concern="bounded bell recoloring",
            tool_need="Compare success and click-point accuracy.",
            task_name="click_bell",
            forbidden_metric_ids={
                "official_check_success",
                "time_to_success",
            },
        )

        self.assertEqual(bundle["tool_request"]["metric"], additional["metric"])
        self.assertEqual(provider.calls, 2)
        self.assertEqual(
            bundle["context"]["forbidden_metric_ids"],
            ["official_check_success", "time_to_success"],
        )
        registry_names = {
            item["name"]
            for item in bundle["context"]["tool_registry"]["trusted_tools"]
        }
        self.assertNotIn("official_check_success", registry_names)
        self.assertIn("already measured", bundle["provider"]["errors"][0])

    def test_active_arm_need_retries_to_registered_composite_target(self):
        root = Path(__file__).resolve().parents[2]
        invalid = _typed("click_bell")
        invalid["metric_spec"]["left_signal"] = "left_tcp_position"
        invalid["metric_spec"]["right_signal"] = "bell_contact_position"
        repaired = {
            "schema_version": 1,
            "task_name": "click_bell",
            "metric": "bell_active_tcp_min_xy_error",
            "question": (
                "What was the minimum XY distance between the official "
                "active-arm TCP and the bell contact point?"
            ),
        }
        provider = _RetryProvider([invalid, repaired])
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")

        bundle = agent.propose(
            source_query="Where does bell-position generalization fail?",
            semantic_concern="bounded bell translation",
            tool_need=(
                "Measure target-contact distance with the active gripper."
            ),
            task_name="click_bell",
        )

        self.assertEqual(
            bundle["tool_request"]["metric"],
            "bell_active_tcp_min_xy_error",
        )
        self.assertEqual(provider.calls, 2)
        self.assertIn(
            "fixed-side minimum_distance",
            bundle["provider"]["errors"][0],
        )


if __name__ == "__main__":
    unittest.main()
