"""Bounded multi-round Plan Agent for the MEA prototype."""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from mea.taskgen import extract_json_response
from mea.toolgen import (
    PICKUP_TO_CONTACT_METRIC,
    ToolOrchestrationError,
    contact_tool_spec,
    pickup_to_contact_tool_spec,
    validate_tool_spec,
)


class PlanAgentError(RuntimeError):
    """Raised when an outer-agent decision violates the prototype contract."""


BLUE_TASK_INSTRUCTION = (
    "把 beat_block_hammer 任务中的红色方块改成蓝色，其他行为保持不变。"
)
POSITION_TASK_INSTRUCTION = (
    "保持 beat_block_hammer 的方块为蓝色和其他任务行为不变，使用官方位置与朝向随机化，"
    "在两个通过 expert gate 的 evaluation seed 上评估 2 个 episode。"
)
REQUIRED_GATES = ["ast", "render", "rule", "vision", "expert", "act"]
REQUIRED_OBSERVATIONS = [
    "scene_alignment",
    "observed_color",
    "expert_solvable",
    "act_pipeline_status",
    "policy_success",
]
EXPECTED_POLICY = {
    "name": "ACT",
    "checkpoint_setting": "demo_clean",
    "expert_data_num": 50,
    "language_conditioned": False,
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_head(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def make_evaluation_id() -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"eval_{timestamp}_{uuid.uuid4().hex[:8]}"


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanAgentError(f"{field} 必须是非空字符串")
    return value.strip()


def _normalized_blue(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise PlanAgentError("variant_hint.block.color 必须是三个通道的 list")
    color = [float(channel) for channel in value]
    if any(channel < 0.0 or channel > 1.0 for channel in color):
        raise PlanAgentError("variant_hint.block.color 通道必须在 [0, 1]")
    expected = [0.0, 0.2, 1.0]
    if any(abs(actual - target) > 1e-6 for actual, target in zip(color, expected)):
        raise PlanAgentError(f"当前原型只允许已验证的蓝色 {expected}")
    return color


def _validate_round(
    round_plan: dict[str, Any],
    *,
    round_id: str,
    sub_aspect: str,
    route: str,
    tool_route: str,
    tool_metric: str,
    instruction: str,
    seeds: list[int],
) -> dict[str, Any]:
    if not isinstance(round_plan, dict):
        raise PlanAgentError(f"{round_id} 必须是 object")
    if round_plan.get("round_id") != round_id:
        raise PlanAgentError(f"round_id 必须是 {round_id}")
    if round_plan.get("sub_aspect") != sub_aspect:
        raise PlanAgentError(f"{round_id}.sub_aspect 必须是 {sub_aspect}")
    if round_plan.get("route") != route:
        raise PlanAgentError(f"{round_id}.route 必须是 {route}")
    task_instruction = _require_string(
        round_plan.get("task_instruction"), f"{round_id}.task_instruction"
    )
    if task_instruction != instruction:
        raise PlanAgentError(f"{round_id} 必须输出受支持的规范 task instruction")

    variant_hint = round_plan.get("variant_hint")
    if not isinstance(variant_hint, dict) or not isinstance(
        variant_hint.get("block"), dict
    ):
        raise PlanAgentError(f"{round_id}.variant_hint.block 必须是 object")
    block = variant_hint["block"]
    if block.get("position_mode") != "official_random":
        raise PlanAgentError(f"{round_id} 必须使用 official position sampling")
    if block.get("yaw_mode") != "official_random":
        raise PlanAgentError(f"{round_id} 必须使用 official yaw sampling")
    if float(block.get("scale", 0.0)) != 1.0:
        raise PlanAgentError(f"{round_id} 必须保持 scale=1.0")
    color = _normalized_blue(block.get("color"))

    execution = round_plan.get("execution")
    if not isinstance(execution, dict):
        raise PlanAgentError(f"{round_id}.execution 必须是 object")
    if execution.get("seeds") != seeds:
        raise PlanAgentError(f"{round_id}.execution.seeds 必须是 {seeds}")
    if execution.get("num_episodes") != len(seeds):
        raise PlanAgentError(
            f"{round_id}.execution.num_episodes 必须是 {len(seeds)}"
        )
    if execution.get("gates") != REQUIRED_GATES:
        raise PlanAgentError(f"gates 必须按顺序为 {REQUIRED_GATES}")
    if round_plan.get("observations") != REQUIRED_OBSERVATIONS:
        raise PlanAgentError(
            f"observations 必须按顺序为 {REQUIRED_OBSERVATIONS}"
        )
    try:
        tool_spec = validate_tool_spec(
            round_plan.get("tool_spec"),
            expected_route=tool_route,
            expected_metric=tool_metric,
        )
    except ToolOrchestrationError as exc:
        raise PlanAgentError(f"{round_id}.tool_spec 无效: {exc}") from exc

    return {
        "round_id": round_id,
        "sub_aspect": sub_aspect,
        "rationale": _require_string(round_plan.get("rationale"), f"{round_id}.rationale"),
        "task_instruction": task_instruction,
        "route": route,
        "variant_hint": {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 1.0,
                "color": color,
            }
        },
        "execution": {
            "seeds": list(seeds),
            "num_episodes": len(seeds),
            "gates": list(REQUIRED_GATES),
        },
        "observations": list(REQUIRED_OBSERVATIONS),
        "tool_spec": tool_spec,
    }


def validate_evaluation_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate the initial plan, which deliberately proposes Round 1 only."""

    if not isinstance(plan, dict):
        raise PlanAgentError("EvaluationPlan 必须是 JSON object")
    if plan.get("task_name") != "beat_block_hammer":
        raise PlanAgentError("当前原型只支持 task_name=beat_block_hammer")
    if plan.get("policy") != EXPECTED_POLICY:
        raise PlanAgentError(f"policy metadata 必须为 {EXPECTED_POLICY}")
    rounds = plan.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 1:
        raise PlanAgentError("初始 Plan 必须且只能提出 Round 1")
    if plan.get("max_rounds") != 2:
        raise PlanAgentError("当前原型 max_rounds 必须是 2")

    round_1 = _validate_round(
        rounds[0],
        round_id="round_1",
        sub_aspect="object_appearance.color",
        route="force_codegen",
        tool_route="force_codegen",
        tool_metric=PICKUP_TO_CONTACT_METRIC,
        instruction=BLUE_TASK_INSTRUCTION,
        seeds=[100000],
    )
    return {
        "schema_version": 4,
        "task_name": "beat_block_hammer",
        "policy": dict(EXPECTED_POLICY),
        "evaluation_goal": _require_string(
            plan.get("evaluation_goal"), "evaluation_goal"
        ),
        "rounds": [round_1],
        "max_rounds": 2,
        "planning_state": "awaiting_round_1_observation",
    }


def validate_next_round_decision(
    decision: dict[str, Any],
    round_1_observation: dict[str, Any],
) -> dict[str, Any]:
    """Validate a Plan Agent decision grounded in the completed first round."""

    if not isinstance(decision, dict):
        raise PlanAgentError("NextRoundDecision 必须是 JSON object")
    pipeline_passed = bool(round_1_observation.get("pipeline_passed"))
    action = decision.get("action")
    if not pipeline_passed:
        if action != "stop":
            raise PlanAgentError("Round 1 流水线失败时必须停止并报告")
        return {
            "schema_version": 1,
            "action": "stop",
            "observation_summary": _require_string(
                decision.get("observation_summary"), "observation_summary"
            ),
            "decision_reason": _require_string(
                decision.get("decision_reason"), "decision_reason"
            ),
            "next_round": None,
        }

    if action != "continue":
        raise PlanAgentError("Round 1 流水线通过后必须继续位置变化评估")
    next_round = _validate_round(
        decision.get("next_round"),
        round_id="round_2",
        sub_aspect="object_position",
        route="reuse",
        tool_route="reuse",
        tool_metric="hammer_block_contact_ever",
        instruction=POSITION_TASK_INSTRUCTION,
        seeds=[100002, 100003],
    )
    return {
        "schema_version": 1,
        "action": "continue",
        "observation_summary": _require_string(
            decision.get("observation_summary"), "observation_summary"
        ),
        "decision_reason": _require_string(
            decision.get("decision_reason"), "decision_reason"
        ),
        "next_round": next_round,
    }


def _initial_plan_prompt(repo_root: Path, user_request: str) -> str:
    agent_readme = (repo_root / "mea/planner/README.Agent.md").read_text(
        encoding="utf-8"
    )
    example = {
        "schema_version": 4,
        "task_name": "beat_block_hammer",
        "policy": EXPECTED_POLICY,
        "evaluation_goal": "evaluate_blue_block_and_position_variation",
        "rounds": [
            {
                "round_id": "round_1",
                "sub_aspect": "object_appearance.color",
                "rationale": "先隔离用户指定的蓝色外观变化。",
                "task_instruction": BLUE_TASK_INSTRUCTION,
                "route": "force_codegen",
                "variant_hint": {
                    "block": {
                        "position_mode": "official_random",
                        "yaw_mode": "official_random",
                        "scale": 1.0,
                        "color": [0.0, 0.2, 1.0],
                    }
                },
                "execution": {
                    "seeds": [100000],
                    "num_episodes": 1,
                    "gates": REQUIRED_GATES,
                },
                "observations": REQUIRED_OBSERVATIONS,
                "tool_spec": pickup_to_contact_tool_spec("force_codegen"),
            }
        ],
        "max_rounds": 2,
    }
    return f"""你是 MEA 的外层 Plan Agent。你负责规划、观察和调整评估，不生成 Python。

USER QUERY:
{user_request}

POLICY / SIMULATOR CAPABILITIES AND VALIDATED EXAMPLE:
{agent_readme}

用户要求评估蓝色方块和位置变化。此时只提出第一个 sub-aspect；不要预先生成 Round 2，
必须等真实 Round 1 observations 返回后再决定。输出严格 JSON，不要 Markdown，内容必须为：
{json.dumps(example, ensure_ascii=False, indent=2)}
"""


def _next_round_prompt(
    user_request: str,
    current_plan: dict[str, Any],
    round_1_observation: dict[str, Any],
) -> str:
    example = {
        "schema_version": 1,
        "action": "continue",
        "observation_summary": "Round 1 场景与评估流水线通过；记录 policy result 后继续位置维度。",
        "decision_reason": "用户同时要求位置变化，Round 1 已提供可解释观察，因此继续 Round 2。",
        "next_round": {
            "round_id": "round_2",
            "sub_aspect": "object_position",
            "rationale": "在保持蓝色和任务逻辑不变时，用两个 seed 检查官方位置采样。",
            "task_instruction": POSITION_TASK_INSTRUCTION,
            "route": "reuse",
            "variant_hint": {
                "block": {
                    "position_mode": "official_random",
                    "yaw_mode": "official_random",
                    "scale": 1.0,
                    "color": [0.0, 0.2, 1.0],
                }
            },
            "execution": {
                "seeds": [100002, 100003],
                "num_episodes": 2,
                "gates": REQUIRED_GATES,
            },
            "observations": REQUIRED_OBSERVATIONS,
            "tool_spec": contact_tool_spec("reuse"),
        },
    }
    return f"""你是 MEA 的外层 Plan Agent。根据真实的上一轮观察，决定继续还是停止。

USER QUERY:
{user_request}

CURRENT PLAN:
{json.dumps(current_plan, ensure_ascii=False, indent=2)}

ROUND 1 OBSERVATION:
{json.dumps(round_1_observation, ensure_ascii=False, indent=2)}

规则：pipeline_passed=true 表示生成与执行链可用，即使 policy_success=0 也应继续收集用户要求的
位置证据；pipeline_passed=false 才输出 action=stop。当前流水线通过时，输出严格 JSON：
{json.dumps(example, ensure_ascii=False, indent=2)}
"""


class PlanAgentPrototype:
    """Generate Round 1, then adapt once using actual execution observations."""

    def __init__(self, repo_root: str | Path, provider: Any, *, model: str):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.provider = provider
        self.model = model

    def plan(
        self,
        user_request: str,
        *,
        evaluation_id: str | None = None,
    ) -> dict[str, Any]:
        request = _require_string(user_request, "user_request")
        evaluation_id = evaluation_id or make_evaluation_id()
        if not re.fullmatch(r"eval_[A-Za-z0-9_]+", evaluation_id):
            raise PlanAgentError(
                "evaluation_id 必须是合法目录名并以 eval_ 开头"
            )

        evaluation_dir = self.repo_root / "mea/evaluation_runs" / evaluation_id
        if evaluation_dir.exists():
            raise PlanAgentError(f"evaluation directory 已存在: {evaluation_dir}")
        for child in ("plan", "execution", "summary"):
            (evaluation_dir / child).mkdir(parents=True, exist_ok=False)

        manifest: dict[str, Any] = {
            "schema_version": 4,
            "evaluation_id": evaluation_id,
            "status": "planning_round_1",
            "created_at": datetime.now().astimezone().isoformat(),
            "user_request": request,
            "base_commit": _git_head(self.repo_root),
            "planner": {"model_requested": self.model},
        }
        _write_json(evaluation_dir / "request.json", {"user_request": request})
        _write_json(evaluation_dir / "manifest.json", manifest)

        prompt = _initial_plan_prompt(self.repo_root, request)
        (evaluation_dir / "plan/round_1_prompt.md").write_text(
            prompt, encoding="utf-8"
        )
        response = self.provider.text(
            prompt,
            model=self.model,
            system="只输出满足 EvaluationPlan schema 的 JSON object。",
            max_tokens=1800,
            temperature=0.0,
        )
        (evaluation_dir / "plan/round_1_response.txt").write_text(
            response + "\n", encoding="utf-8"
        )
        plan = validate_evaluation_plan(extract_json_response(response))
        _write_json(evaluation_dir / "plan/evaluation_plan.json", plan)

        manifest.update(
            {
                "status": "planned_round_1",
                "plan_path": "plan/evaluation_plan.json",
                "plan": plan,
                "planner": {
                    "model_requested": self.model,
                    "round_1_metadata": dict(
                        getattr(self.provider, "last_metadata", {})
                    ),
                },
            }
        )
        _write_json(evaluation_dir / "manifest.json", manifest)
        return manifest

    def decide_next_round(
        self,
        *,
        evaluation_id: str,
        user_request: str,
        current_plan: dict[str, Any],
        round_1_observation: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Feed Round 1 evidence back to the planner and append a validated decision."""

        evaluation_dir = self.repo_root / "mea/evaluation_runs" / evaluation_id
        if not evaluation_dir.is_dir():
            raise PlanAgentError(f"evaluation directory 不存在: {evaluation_dir}")
        prompt = _next_round_prompt(
            _require_string(user_request, "user_request"),
            current_plan,
            round_1_observation,
        )
        (evaluation_dir / "plan/round_2_prompt.md").write_text(
            prompt, encoding="utf-8"
        )

        errors: list[str] = []
        decision = None
        for attempt in range(2):
            attempt_prompt = prompt
            if errors:
                attempt_prompt += (
                    "\n\nPREVIOUS VALIDATION ERROR:\n"
                    + errors[-1]
                    + "\n请重新输出完整严格 JSON。\n"
                )
            response = self.provider.text(
                attempt_prompt,
                model=self.model,
                system="只基于 observations 决策，并输出严格 NextRoundDecision JSON。",
                max_tokens=1800,
                temperature=0.0,
            )
            suffix = "" if attempt == 0 else f"_retry_{attempt}"
            (evaluation_dir / f"plan/round_2_response{suffix}.txt").write_text(
                response + "\n", encoding="utf-8"
            )
            try:
                decision = validate_next_round_decision(
                    extract_json_response(response), round_1_observation
                )
                break
            except PlanAgentError as exc:
                errors.append(str(exc))
        if decision is None:
            raise PlanAgentError(f"NextRoundDecision 两次均未通过: {errors}")

        updated_plan = deepcopy(current_plan)
        updated_plan.setdefault("round_decisions", []).append(decision)
        if decision["action"] == "continue":
            updated_plan["rounds"].append(decision["next_round"])
            updated_plan["planning_state"] = "round_2_planned"
        else:
            updated_plan["planning_state"] = "stopped_after_round_1"

        _write_json(evaluation_dir / "plan/round_2_decision.json", decision)
        _write_json(evaluation_dir / "plan/evaluation_plan.json", updated_plan)
        manifest_path = evaluation_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": updated_plan["planning_state"],
                "plan": updated_plan,
                "round_2_decision_path": "plan/round_2_decision.json",
            }
        )
        manifest.setdefault("planner", {})["round_2_metadata"] = dict(
            getattr(self.provider, "last_metadata", {})
        )
        manifest["planner"]["round_2_validation_errors"] = errors
        _write_json(manifest_path, manifest)
        return updated_plan, decision
