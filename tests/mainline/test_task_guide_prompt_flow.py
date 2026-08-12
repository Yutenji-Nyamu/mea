from __future__ import annotations

import json
from pathlib import Path

from mea.planner.plan_agent_provider import PlanAgent
from mea.task_guide import load_task_guide
from mea.taskgen.generic_backend import _discover_task_documents
from mea.toolgen.metric_codegen import _provider_codegen_prompt
from mea.toolgen.open_request_agent import OpenToolRequestAgent
from mea.toolgen.open_request_context import tool_generation_context
from mea.toolkit.tools import TrajectoryView
from tests.mainline.test_tool_orchestration import write_episode


MARKER = "MEA_TASK_GUIDE: grab_roller"


def _capabilities() -> dict:
    return {
        "schema_version": 1,
        "policy_card": {"policy_name": "SmolVLA", "task_name": "grab_roller"},
        "simulator_card": {
            "simulator_name": "RoboTwin",
            "task_name": "grab_roller",
            "tracked_actors": ["roller"],
        },
        "generation_card": {
            "taskgen_operations": [],
            "toolgen": {
                "retrieve_first": True,
                "can_generate_rule_metric": True,
                "can_generate_vqa_question": True,
            },
        },
    }


def test_one_bound_task_guide_reaches_plan_taskgen_and_tool_prompts(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    guide = load_task_guide(root, "grab_roller")
    assert MARKER in guide

    plan_prompt = PlanAgent._prompt(
        "Where does this policy first fail?",
        _capabilities(),
        [],
        task_guide=guide,
    )
    assert MARKER in plan_prompt

    documents = _discover_task_documents(root, task_name="grab_roller")
    assert "mea/knowledge/tasks/grab_roller.md" in documents
    assert not any("task_instruction" in item for item in documents)

    schema = json.loads(
        (root / "mea/toolkit/schemas/grab_roller.json").read_text(
            encoding="utf-8"
        )
    )
    context = tool_generation_context(
        root,
        task_name="grab_roller",
        runtime_schema=schema,
        proposal={
            "schema_version": 2,
            "candidate_id": "candidate.grab_roller.fixture",
            "source_query": "Where does this policy first fail?",
            "base_task": "grab_roller",
            "semantic_concern": "terminal contact geometry",
            "scene_need": None,
            "checker_need": None,
            "rule_tool_need": {
                "kind": "measure",
                "description": "Measure terminal TCP to contact-point distance.",
                "reuse_first": True,
            },
            "vqa_tool_need": None,
        },
    )
    request_prompt = OpenToolRequestAgent._prompt(
        source_query="Where does this policy first fail?",
        semantic_concern="terminal contact geometry",
        tool_need="Measure terminal TCP to contact-point distance.",
        context=context,
    )
    assert MARKER in request_prompt

    episode = tmp_path / "episode"
    write_episode(episode, policy_name="fixture", physical_contact=True)
    trajectory = TrajectoryView(episode)
    code_prompt = _provider_codegen_prompt(
        repo_root=root,
        metric="terminal_contact_distance",
        question="What is the terminal contact distance?",
        metric_spec={
            "schema_version": 1,
            "operation": "terminal_signal_difference",
            "left_signal": "roller_left_contact_position",
            "right_signal": "left_tcp_position",
            "component": "z",
            "absolute": True,
            "unit": "m",
            "null_semantics": "null_if_terminal_not_finite",
        },
        trajectory=trajectory,
        task_code_context={
            "task_name": "grab_roller",
            "task_implementation_guide": guide,
        },
        previous_error=None,
    )
    assert MARKER in code_prompt
