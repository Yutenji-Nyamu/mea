"""Single-round outer Plan Agent for the first MEA orchestration prototype."""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mea.taskgen import extract_json_response


class PlanAgentError(RuntimeError):
    """Raised when an outer-agent evaluation plan violates the prototype contract."""


BLUE_TASK_INSTRUCTION = (
    "把 beat_block_hammer 任务中的红色方块改成蓝色，其他行为保持不变。"
)
REQUIRED_GATES = ["ast", "render", "rule", "vision", "expert", "act"]
REQUIRED_OBSERVATIONS = [
    "scene_alignment",
    "observed_color",
    "expert_solvable",
    "act_pipeline_status",
    "policy_success",
]


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


def _normalized_color(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise PlanAgentError("variant_hint.block.color 必须是三个通道的 list")
    color = [float(channel) for channel in value]
    if any(channel < 0.0 or channel > 1.0 for channel in color):
        raise PlanAgentError("variant_hint.block.color 通道必须在 [0, 1]")
    if any(abs(actual - expected) > 1e-6 for actual, expected in zip(color, [0.0, 0.2, 1.0])):
        raise PlanAgentError("第一版 Plan Agent 只允许已验证的蓝色 [0.0, 0.2, 1.0]")
    return color


def validate_evaluation_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the deliberately narrow first EvaluationPlan."""

    if not isinstance(plan, dict):
        raise PlanAgentError("EvaluationPlan 必须是 JSON object")
    if plan.get("task_name") != "beat_block_hammer":
        raise PlanAgentError("第一版只支持 task_name=beat_block_hammer")

    policy = plan.get("policy")
    if not isinstance(policy, dict):
        raise PlanAgentError("policy 必须是 object")
    expected_policy = {
        "name": "ACT",
        "checkpoint_setting": "demo_clean",
        "expert_data_num": 50,
        "language_conditioned": False,
    }
    if policy != expected_policy:
        raise PlanAgentError(f"policy metadata 必须为 {expected_policy}")

    rounds = plan.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 1:
        raise PlanAgentError("第一版必须且只能规划一个 round")
    round_plan = rounds[0]
    if not isinstance(round_plan, dict):
        raise PlanAgentError("round_1 必须是 object")
    if round_plan.get("round_id") != "round_1":
        raise PlanAgentError("唯一 round_id 必须是 round_1")
    if round_plan.get("sub_aspect") != "object_appearance.color":
        raise PlanAgentError("第一版 sub_aspect 必须是 object_appearance.color")
    if round_plan.get("route") != "force_codegen":
        raise PlanAgentError("第一版必须通过 force_codegen 驱动内层 TaskGen")

    task_instruction = _require_string(
        round_plan.get("task_instruction"),
        "rounds[0].task_instruction",
    )
    if task_instruction != BLUE_TASK_INSTRUCTION:
        raise PlanAgentError("第一版必须输出已验证的规范蓝色方块指令")

    variant_hint = round_plan.get("variant_hint")
    if not isinstance(variant_hint, dict) or not isinstance(
        variant_hint.get("block"), dict
    ):
        raise PlanAgentError("variant_hint.block 必须是 object")
    block = variant_hint["block"]
    if block.get("position_mode") != "official_random":
        raise PlanAgentError("蓝色 smoke test 必须保持 official position sampling")
    if block.get("yaw_mode") != "official_random":
        raise PlanAgentError("蓝色 smoke test 必须保持 official yaw sampling")
    if float(block.get("scale", 0.0)) != 1.0:
        raise PlanAgentError("蓝色 smoke test 必须保持 scale=1.0")
    color = _normalized_color(block.get("color"))

    execution = round_plan.get("execution")
    if not isinstance(execution, dict):
        raise PlanAgentError("execution 必须是 object")
    if execution.get("seeds") != [100000]:
        raise PlanAgentError("第一版必须使用 seeds=[100000]")
    if execution.get("num_episodes") != 1:
        raise PlanAgentError("第一版必须使用 num_episodes=1")
    if execution.get("gates") != REQUIRED_GATES:
        raise PlanAgentError(f"gates 必须按顺序为 {REQUIRED_GATES}")
    if round_plan.get("observations") != REQUIRED_OBSERVATIONS:
        raise PlanAgentError(
            f"observations 必须按顺序为 {REQUIRED_OBSERVATIONS}"
        )
    if plan.get("stop_after_round") != 1:
        raise PlanAgentError("第一版 stop_after_round 必须为 1")

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "policy": expected_policy,
        "evaluation_goal": _require_string(
            plan.get("evaluation_goal"), "evaluation_goal"
        ),
        "rounds": [
            {
                "round_id": "round_1",
                "sub_aspect": "object_appearance.color",
                "rationale": _require_string(
                    round_plan.get("rationale"), "rounds[0].rationale"
                ),
                "task_instruction": task_instruction,
                "route": "force_codegen",
                "variant_hint": {
                    "block": {
                        "position_mode": "official_random",
                        "yaw_mode": "official_random",
                        "scale": 1.0,
                        "color": color,
                    }
                },
                "execution": {
                    "seeds": [100000],
                    "num_episodes": 1,
                    "gates": list(REQUIRED_GATES),
                },
                "observations": list(REQUIRED_OBSERVATIONS),
            }
        ],
        "stop_after_round": 1,
    }


def _plan_prompt(repo_root: Path, user_request: str) -> str:
    agent_readme = (repo_root / "mea/planner/README.Agent.md").read_text(
        encoding="utf-8"
    )
    return f"""你是 MEA 的外层 Plan Agent。你只负责决定本轮评估什么，不生成 Python。

USER QUERY:
{user_request}

POLICY / SIMULATOR CAPABILITIES AND VALIDATED EXAMPLE:
{agent_readme}

请输出严格 JSON，不要输出 Markdown。第一版只能规划一个蓝色方块 smoke-test round，格式必须为：
{{
  "schema_version": 1,
  "task_name": "beat_block_hammer",
  "policy": {{
    "name": "ACT",
    "checkpoint_setting": "demo_clean",
    "expert_data_num": 50,
    "language_conditioned": false
  }},
  "evaluation_goal": "evaluate_act_with_blue_block",
  "rounds": [
    {{
      "round_id": "round_1",
      "sub_aspect": "object_appearance.color",
      "rationale": "用户明确要求评估蓝色方块，因此本轮隔离物体颜色，保持其他变量不变。",
      "task_instruction": "{BLUE_TASK_INSTRUCTION}",
      "route": "force_codegen",
      "variant_hint": {{
        "block": {{
          "position_mode": "official_random",
          "yaw_mode": "official_random",
          "scale": 1.0,
          "color": [0.0, 0.2, 1.0]
        }}
      }},
      "execution": {{
        "seeds": [100000],
        "num_episodes": 1,
        "gates": ["ast", "render", "rule", "vision", "expert", "act"]
      }},
      "observations": [
        "scene_alignment",
        "observed_color",
        "expert_solvable",
        "act_pipeline_status",
        "policy_success"
      ]
    }}
  ],
  "stop_after_round": 1
}}
"""


class PlanAgentPrototype:
    """Generate one validated single-round EvaluationPlan and its manifest."""

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
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "status": "planning",
            "created_at": datetime.now().astimezone().isoformat(),
            "user_request": request,
            "base_commit": _git_head(self.repo_root),
            "planner": {"model_requested": self.model},
        }
        _write_json(evaluation_dir / "request.json", {"user_request": request})
        _write_json(evaluation_dir / "manifest.json", manifest)

        prompt = _plan_prompt(self.repo_root, request)
        (evaluation_dir / "plan/prompt.md").write_text(prompt, encoding="utf-8")
        response = self.provider.text(
            prompt,
            model=self.model,
            system="只输出满足 EvaluationPlan schema 的 JSON object。",
            max_tokens=1800,
            temperature=0.0,
        )
        (evaluation_dir / "plan/response.txt").write_text(
            response + "\n", encoding="utf-8"
        )
        plan = validate_evaluation_plan(extract_json_response(response))
        _write_json(evaluation_dir / "plan/evaluation_plan.json", plan)

        manifest.update(
            {
                "status": "planned",
                "plan_path": "plan/evaluation_plan.json",
                "plan": plan,
                "planner": {
                    "model_requested": self.model,
                    "last_metadata": dict(
                        getattr(self.provider, "last_metadata", {})
                    ),
                },
            }
        )
        _write_json(evaluation_dir / "manifest.json", manifest)
        return manifest
