"""Query-driven parent planning across fixed single-task ACT evaluations.

Each child remains bound to one RoboTwin task and one ACT checkpoint.  The
parent graph only decides which child should run next and whether evidence from
the previous child is sufficient to stop.  It never changes a checkpoint
inside an evaluation.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.planner.catalog import validate_act_catalog


class EvaluationGraphError(ValueError):
    """Raised when a parent graph or child outcome exceeds trusted bounds."""


_PLAN_KEYS = {
    "schema_version",
    "graph_id",
    "user_query",
    "evaluation_goal",
    "max_children",
    "nodes",
}
_NODE_KEYS = {
    "node_id",
    "task_name",
    "requested_aspect_ids",
    "activation",
    "rationale",
}
_OUTCOME_KEYS = {
    "schema_version",
    "node_id",
    "task_name",
    "evaluation_id",
    "pipeline_passed",
    "evidence_strength",
    "policy_success",
    "answered_query",
    "summary",
}
_ACTIVATIONS = {"initial", "always", "if_previous_failed_or_uncertain"}
_STRENGTHS = {"sufficient", "uncertain", "conflicting", "pipeline_invalid"}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationGraphError(f"{field} must be a non-empty string")
    return value.strip()


def _graph_identifier(value: Any) -> str:
    graph_id = _text(value, "graph_id")
    if re.fullmatch(r"graph_[A-Za-z0-9_]+", graph_id) is None:
        raise EvaluationGraphError("graph_id must match graph_[A-Za-z0-9_]+")
    return graph_id


def _extract_json(response: str) -> dict[str, Any]:
    text = _text(response, "provider response")
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.I | re.S)
    for candidate in [*fenced, text]:
        try:
            value = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise EvaluationGraphError("provider response has no JSON object")


def validate_evaluation_graph(
    value: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a model-authored graph against the checkpoint-ready catalog."""

    trusted_catalog = validate_act_catalog(catalog)
    if not isinstance(value, Mapping) or set(value) != _PLAN_KEYS:
        raise EvaluationGraphError(
            f"EvaluationGraph fields must be exactly {sorted(_PLAN_KEYS)}"
        )
    plan = deepcopy(dict(value))
    schema_version = plan.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise EvaluationGraphError("schema_version must be 1")
    graph_id = _graph_identifier(plan.get("graph_id"))
    plan["graph_id"] = graph_id
    plan["user_query"] = _text(plan.get("user_query"), "user_query")
    plan["evaluation_goal"] = _text(
        plan.get("evaluation_goal"), "evaluation_goal"
    )
    max_children = plan.get("max_children")
    if isinstance(max_children, bool) or not isinstance(max_children, int):
        raise EvaluationGraphError("max_children must be an integer")
    if max_children not in {1, 2}:
        raise EvaluationGraphError("max_children must be 1 or 2")
    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= plan["max_children"]:
        raise EvaluationGraphError("nodes must fit the declared 1-2 child budget")

    ready = {task["task_name"]: task for task in trusted_catalog["tasks"]}
    seen_nodes: set[str] = set()
    seen_tasks: set[str] = set()
    normalized_nodes: list[dict[str, Any]] = []
    for index, raw in enumerate(nodes):
        if not isinstance(raw, Mapping) or set(raw) != _NODE_KEYS:
            raise EvaluationGraphError(
                f"nodes[{index}] fields must be exactly {sorted(_NODE_KEYS)}"
            )
        node = deepcopy(dict(raw))
        node_id = _text(node.get("node_id"), f"nodes[{index}].node_id")
        if re.fullmatch(r"node_[A-Za-z0-9_]+", node_id) is None:
            raise EvaluationGraphError("node_id must match node_[A-Za-z0-9_]+")
        if node_id in seen_nodes:
            raise EvaluationGraphError("node_id values must be unique")
        seen_nodes.add(node_id)
        node["node_id"] = node_id
        task_name = _text(node.get("task_name"), f"nodes[{index}].task_name")
        if task_name not in ready:
            raise EvaluationGraphError(f"task is not checkpoint-ready: {task_name}")
        if task_name in seen_tasks:
            raise EvaluationGraphError("each child task may appear at most once")
        seen_tasks.add(task_name)
        node["task_name"] = task_name
        available_aspects = {
            aspect["aspect_id"] for aspect in ready[task_name]["aspects"]
        }
        aspects = node.get("requested_aspect_ids")
        if not isinstance(aspects, list) or len(aspects) != 1:
            raise EvaluationGraphError(
                f"nodes[{index}] must select exactly one supported aspect for "
                f"{task_name}"
            )
        normalized_aspects = [
            _text(item, f"nodes[{index}].requested_aspect_ids") for item in aspects
        ]
        if (
            len(normalized_aspects) != len(set(normalized_aspects))
            or not set(normalized_aspects) <= available_aspects
        ):
            raise EvaluationGraphError(
                f"nodes[{index}] must select exactly one supported aspect for "
                f"{task_name}"
            )
        node["requested_aspect_ids"] = normalized_aspects
        activation = node.get("activation")
        if activation not in _ACTIVATIONS:
            raise EvaluationGraphError(f"unsupported activation: {activation!r}")
        if (index == 0) != (activation == "initial"):
            raise EvaluationGraphError("only the first node may use initial activation")
        node["rationale"] = _text(node.get("rationale"), f"nodes[{index}].rationale")
        normalized_nodes.append(node)
    plan["nodes"] = normalized_nodes
    return plan


def build_evaluation_graph_prompt(
    user_query: str, catalog: Mapping[str, Any], *, graph_id: str
) -> str:
    trusted = validate_act_catalog(catalog)
    query = _text(user_query, "user_query")
    graph_id = _graph_identifier(graph_id)
    example_tasks = trusted["tasks"][:2]
    example_nodes = []
    for index, task in enumerate(example_tasks):
        example_nodes.append(
            {
                "node_id": f"node_{index + 1}",
                "task_name": task["task_name"],
                "requested_aspect_ids": [task["aspects"][0]["aspect_id"]],
                "activation": (
                    "initial" if index == 0 else "if_previous_failed_or_uncertain"
                ),
                "rationale": "query-relevant bounded child evaluation",
            }
        )
    example = {
        "schema_version": 1,
        "graph_id": graph_id,
        "user_query": query,
        "evaluation_goal": "answer the original query with bounded ACT evidence",
        "max_children": min(2, max(1, len(example_nodes))),
        "nodes": example_nodes,
    }
    return f"""You are the parent Plan Agent for MEA.
Select one or at most two checkpoint-ready child evaluations that directly
answer the user's query.  Each child must select exactly one supported aspect
and is permanently bound to one task, one ACT checkpoint, and one ACT round.
Do not invent tasks, aspects, checkpoints, seeds, code, tools, or execution
settings.

For a second child choose activation `always` only when both task families are
required to answer the query.  Otherwise choose
`if_previous_failed_or_uncertain`, so sufficient successful evidence may stop
the graph early.

USER QUERY:
{query}

TRUSTED CATALOG:
{json.dumps(trusted, ensure_ascii=False, indent=2)}

Return strict JSON with exactly this shape:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""


class EvaluationGraphPlanner:
    """Ask a provider for a bounded graph and validate every child choice."""

    def __init__(self, provider: Any, *, model: str):
        self.provider = provider
        self.model = _text(model, "model")
        self.last_prompt: str | None = None
        self.last_response: str | None = None
        self.last_responses: list[str] = []
        self.validation_errors: list[str] = []

    def plan(
        self,
        user_query: str,
        catalog: Mapping[str, Any],
        *,
        graph_id: str,
    ) -> dict[str, Any]:
        graph_id = _graph_identifier(graph_id)
        prompt = build_evaluation_graph_prompt(user_query, catalog, graph_id=graph_id)
        self.last_prompt = prompt
        self.last_responses = []
        self.validation_errors = []
        for attempt in range(2):
            retry = (
                "\n\nYour previous response was rejected: "
                + self.validation_errors[-1]
                + "\nReturn a corrected object within the exact contract."
                if attempt
                else ""
            )
            response = self.provider.text(
                prompt + retry,
                model=self.model,
                system="Return only strict EvaluationGraph JSON.",
                max_tokens=900,
                temperature=0.0,
            )
            self.last_response = str(response)
            self.last_responses.append(str(response))
            try:
                plan = validate_evaluation_graph(_extract_json(str(response)), catalog)
                if (
                    plan["graph_id"] != graph_id
                    or plan["user_query"] != user_query.strip()
                ):
                    raise EvaluationGraphError(
                        "provider changed graph identity or user query"
                    )
                return plan
            except EvaluationGraphError as exc:
                self.validation_errors.append(str(exc))
        raise EvaluationGraphError(
            f"provider graph failed two bounded attempts: {self.validation_errors}"
        )


def validate_child_outcome(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _OUTCOME_KEYS:
        raise EvaluationGraphError(
            f"ChildOutcome fields must be exactly {sorted(_OUTCOME_KEYS)}"
        )
    outcome = deepcopy(dict(value))
    schema_version = outcome.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise EvaluationGraphError("ChildOutcome schema_version must be 1")
    for field in ("node_id", "task_name", "evaluation_id", "summary"):
        outcome[field] = _text(outcome.get(field), field)
    if not isinstance(outcome.get("pipeline_passed"), bool):
        raise EvaluationGraphError("pipeline_passed must be bool")
    if outcome.get("evidence_strength") not in _STRENGTHS:
        raise EvaluationGraphError("invalid evidence_strength")
    success = outcome.get("policy_success")
    if success is not None:
        if isinstance(success, bool) or not isinstance(success, (int, float)):
            raise EvaluationGraphError("policy_success must be numeric or null")
        success = float(success)
        if not math.isfinite(success) or not 0.0 <= success <= 1.0:
            raise EvaluationGraphError("policy_success must be finite in [0, 1]")
        outcome["policy_success"] = success
    if not isinstance(outcome.get("answered_query"), bool):
        raise EvaluationGraphError("answered_query must be bool")
    return outcome


def _child_evaluation_id(plan: Mapping[str, Any], node: Mapping[str, Any]) -> str:
    return (
        f"eval_{str(plan['graph_id']).removeprefix('graph_')}_"
        f"{str(node['node_id'])}"
    )


def child_outcome_from_evaluation(
    repo_root: str | Path,
    plan: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    node_id: str,
    evaluation_id: str,
) -> dict[str, Any]:
    """Convert one verified completed ACT child into the parent outcome type."""

    # Import lazily so graph planning remains usable without touching any
    # evaluation artifacts.
    from mea.portfolio import PortfolioError, load_child_evaluation

    trusted_plan = validate_evaluation_graph(plan, catalog)
    normalized_node_id = _text(node_id, "node_id")
    node = next(
        (
            item
            for item in trusted_plan["nodes"]
            if item["node_id"] == normalized_node_id
        ),
        None,
    )
    if node is None:
        raise EvaluationGraphError(f"unknown graph node: {normalized_node_id}")
    expected_evaluation_id = _child_evaluation_id(trusted_plan, node)
    if evaluation_id != expected_evaluation_id:
        raise EvaluationGraphError(
            "child evaluation_id does not match the graph-derived command identity"
        )
    task_name = node["task_name"]
    try:
        child = load_child_evaluation(repo_root, task_name, evaluation_id)
    except PortfolioError as exc:
        raise EvaluationGraphError(f"child evidence is invalid: {exc}") from exc
    binding = child.get("evaluation_binding")
    if not isinstance(binding, Mapping):
        raise EvaluationGraphError("child evidence has no graph execution binding")
    expected_aspects = list(node["requested_aspect_ids"])
    expected_binding = {
        "manifest_user_request": trusted_plan["user_query"],
        "evidence_user_request": trusted_plan["user_query"],
        "bound_task_name": task_name,
        "bound_requested_aspect_ids": expected_aspects,
        "planned_requested_aspect_ids": expected_aspects,
        "executed_aspect_ids": expected_aspects,
        "max_agent_rounds": 1,
        "plan_max_rounds": 1,
        "executed_rounds": 1,
    }
    for field, expected in expected_binding.items():
        actual = binding.get(field)
        invalid_integer = field in {
            "max_agent_rounds",
            "plan_max_rounds",
            "executed_rounds",
        } and (
            isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual != expected
        )
        if invalid_integer or (
            field
            not in {"max_agent_rounds", "plan_max_rounds", "executed_rounds"}
            and actual != expected
        ):
            raise EvaluationGraphError(
                f"child binding {field} does not match the graph command"
            )
    act_starts = child.get("act_rollouts_started")
    if (
        isinstance(act_starts, bool)
        or not isinstance(act_starts, int)
        or act_starts != 1
    ):
        raise EvaluationGraphError(
            "child must have exactly one ACT rollout start for this graph node"
        )
    pipeline = bool(child["pipeline_passed"])
    success = child.get("policy_success")
    if not pipeline:
        strength = "pipeline_invalid"
    elif success is None:
        strength = "uncertain"
    else:
        strength = "sufficient"
    answered = pipeline and success is not None
    return validate_child_outcome(
        {
            "schema_version": 1,
            "node_id": normalized_node_id,
            "task_name": task_name,
            "evaluation_id": str(child["evaluation_id"]),
            "pipeline_passed": pipeline,
            "evidence_strength": strength,
            "policy_success": success,
            "answered_query": answered,
            "summary": (
                f"{task_name}: ACT policy_success={success}; "
                f"pipeline_passed={str(pipeline).lower()}"
            ),
        }
    )


class EvaluationGraphSession:
    """Deterministically advance a validated graph from typed child evidence."""

    def __init__(self, plan: Mapping[str, Any], catalog: Mapping[str, Any]):
        self.catalog = validate_act_catalog(catalog)
        self.plan = validate_evaluation_graph(plan, self.catalog)
        self.outcomes: list[dict[str, Any]] = []

    def next_node(self) -> dict[str, Any] | None:
        index = len(self.outcomes)
        if index >= len(self.plan["nodes"]):
            return None
        node = self.plan["nodes"][index]
        if index == 0 or node["activation"] == "always":
            return deepcopy(node)
        previous = self.outcomes[-1]
        unresolved = (
            not previous["pipeline_passed"]
            or previous["evidence_strength"] != "sufficient"
            or previous["policy_success"] is None
            or previous["policy_success"] < 1.0
            or not previous["answered_query"]
        )
        return deepcopy(node) if unresolved else None

    def record(self, value: Mapping[str, Any]) -> dict[str, Any]:
        expected = self.next_node()
        if expected is None:
            raise EvaluationGraphError("graph has no active child to record")
        outcome = validate_child_outcome(value)
        if (
            outcome["node_id"] != expected["node_id"]
            or outcome["task_name"] != expected["task_name"]
        ):
            raise EvaluationGraphError("ChildOutcome does not match the active node")
        self.outcomes.append(outcome)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        next_node = self.next_node()
        status = "awaiting_child" if next_node is not None else "completed"
        return {
            "schema_version": 1,
            "graph": deepcopy(self.plan),
            "status": status,
            "outcomes": deepcopy(self.outcomes),
            "next_node": next_node,
            "synthesis": self.synthesize() if status == "completed" else None,
        }

    def synthesize(self) -> dict[str, Any]:
        strengths = [
            item["summary"]
            for item in self.outcomes
            if item["pipeline_passed"] and item["policy_success"] == 1.0
        ]
        weaknesses = [
            item["summary"]
            for item in self.outcomes
            if (
                not item["pipeline_passed"]
                or item["policy_success"] is None
                or item["policy_success"] < 1.0
            )
        ]
        limitations = [
            "Each child is a fixed single-task ACT checkpoint evaluation.",
            "N=1 child evidence is functional evidence, not a statistical result.",
        ]
        if len(self.outcomes) < len(self.plan["nodes"]):
            limitations.append("The graph stopped before all conditional children ran.")
        recommendation = (
            "Investigate the failed or uncertain child before increasing repetitions."
            if weaknesses
            else "Repeat the same bounded graph with N=3 only if a stable estimate is needed."
        )
        return {
            "answer": (
                f"Completed {len(self.outcomes)} bounded child evaluation(s) for: "
                f"{self.plan['user_query']}"
            ),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": [recommendation],
            "limitations": limitations,
        }


def build_child_command_plan(
    plan: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    repo_root: str,
    python_executable: str = "python",
    start_seed: int = 100500,
    model_profile: str = "economy",
    gpu: int = 0,
) -> list[dict[str, Any]]:
    if (
        isinstance(start_seed, bool)
        or not isinstance(start_seed, int)
        or start_seed < 0
    ):
        raise EvaluationGraphError("start_seed must be a non-negative integer")
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise EvaluationGraphError("gpu must be a non-negative integer")
    python_executable = _text(python_executable, "python_executable")
    model_profile = _text(model_profile, "model_profile")
    trusted = validate_evaluation_graph(plan, catalog)
    commands: list[dict[str, Any]] = []
    for index, node in enumerate(trusted["nodes"]):
        evaluation_id = _child_evaluation_id(trusted, node)
        argv = [
            python_executable,
            "scripts/manipeval_agent.py",
            "--repo-root",
            repo_root,
            "--request",
            trusted["user_query"],
            "--evaluation-id",
            evaluation_id,
            "--auto-route",
            "--bound-task-name",
            node["task_name"],
        ]
        for aspect_id in node["requested_aspect_ids"]:
            argv.extend(["--bound-requested-aspect-id", aspect_id])
        argv.extend(
            [
                "--proposal-mode",
                "bounded_each_round",
                "--max-agent-rounds",
                "1",
                "--num-episodes",
                "1",
                "--start-seed",
                str(start_seed + index),
                "--model-profile",
                model_profile,
                "--gpu",
                str(gpu),
            ]
        )
        commands.append(
            {
                "node_id": node["node_id"],
                "task_name": node["task_name"],
                "activation": node["activation"],
                "evaluation_id": evaluation_id,
                "argv": argv,
                "execution_state": "inert_until_parent_activates",
            }
        )
    return commands


__all__ = [
    "EvaluationGraphError",
    "EvaluationGraphPlanner",
    "EvaluationGraphSession",
    "build_child_command_plan",
    "build_evaluation_graph_prompt",
    "child_outcome_from_evaluation",
    "validate_child_outcome",
    "validate_evaluation_graph",
]
