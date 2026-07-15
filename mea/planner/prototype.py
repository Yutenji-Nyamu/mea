"""Bounded, catalog-backed multi-round Plan Agent for MEA."""

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
    contact_tool_request,
    pickup_to_contact_tool_request,
    validate_tool_request,
)


class PlanAgentError(RuntimeError):
    """Raised when an outer-agent proposal violates the bounded contract."""


BLUE_TASK_INSTRUCTION = (
    "把 beat_block_hammer 任务中的红色方块改成蓝色，其他行为保持不变。"
)
POSITION_TASK_INSTRUCTION = (
    "保持 beat_block_hammer 的方块为蓝色和其他任务行为不变，使用官方位置与朝向随机化，"
    "在两个通过 expert gate 的 evaluation seed 上评估 2 个 episode。"
)
TIMING_TASK_INSTRUCTION = (
    "保持 beat_block_hammer 的方块为蓝色、官方位置与朝向随机化以及其他任务行为不变，"
    "评估 1 个 episode，并分析从锤子首次抬升到首次严格物理接触方块的时间。"
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
MAX_ROUNDS = 3


def _blue_variant() -> dict[str, Any]:
    return {
        "block": {
            "position_mode": "official_random",
            "yaw_mode": "official_random",
            "scale": 1.0,
            "color": [0.0, 0.2, 1.0],
        }
    }


# This is trusted runtime configuration, not model output.  The model selects a
# template id; the system injects every executable detail below.
SUB_ASPECT_CATALOG: dict[str, dict[str, Any]] = {
    "object_appearance.color_blue": {
        "sub_aspect": "object_appearance.color",
        "rationale": "先隔离用户指定的蓝色外观变化。",
        "task_instruction": BLUE_TASK_INSTRUCTION,
        "route": "force_codegen",
        "variant_hint": _blue_variant(),
        "seeds": [100000],
        "tool_metric": "hammer_block_contact_ever",
        "tool_request_factory": contact_tool_request,
    },
    "object_position.official_random": {
        "sub_aspect": "object_position",
        "rationale": "保持蓝色与任务逻辑不变，检查官方位置和朝向采样。",
        "task_instruction": POSITION_TASK_INSTRUCTION,
        "route": "reuse",
        "variant_hint": _blue_variant(),
        "seeds": [100002, 100003],
        "tool_metric": "hammer_block_contact_ever",
        "tool_request_factory": contact_tool_request,
    },
    "performance.pickup_to_contact_timing": {
        "sub_aspect": "performance.pickup_to_contact_timing",
        "rationale": "量化锤子首次抬升到首次严格物理接触的经过时间。",
        "task_instruction": TIMING_TASK_INSTRUCTION,
        "route": "reuse",
        "variant_hint": _blue_variant(),
        # seed 100000 already passed the expert gate and provides a stable
        # ACT/expert contrast for the bounded timing prototype.
        "seeds": [100000],
        "tool_metric": "pickup_to_first_contact_time",
        "tool_request_factory": pickup_to_contact_tool_request,
    },
}

INITIAL_PROPOSAL_KEYS = {
    "schema_version",
    "task_name",
    "policy",
    "evaluation_goal",
    "requested_template_ids",
    "first_template_id",
    "max_rounds",
}
DECISION_KEYS = {
    "schema_version",
    "action",
    "observation_summary",
    "decision_reason",
    "next_template_id",
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


def _require_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PlanAgentError(f"{name} fields 不匹配，missing={missing}, extra={extra}")


def _validate_template_ids(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PlanAgentError("requested_template_ids 必须是非空 list")
    if len(value) > MAX_ROUNDS:
        raise PlanAgentError(f"最多只能请求 {MAX_ROUNDS} 个受限 template")
    if any(not isinstance(item, str) for item in value):
        raise PlanAgentError("requested_template_ids 只能包含字符串")
    if len(set(value)) != len(value):
        raise PlanAgentError("requested_template_ids 不得重复")
    unknown = [item for item in value if item not in SUB_ASPECT_CATALOG]
    if unknown:
        raise PlanAgentError(f"未注册的 sub-aspect template: {unknown}")
    return list(value)


def _materialize_round(template_id: str, round_number: int) -> dict[str, Any]:
    if template_id not in SUB_ASPECT_CATALOG:
        raise PlanAgentError(f"未注册的 sub-aspect template: {template_id}")
    if round_number < 1 or round_number > MAX_ROUNDS:
        raise PlanAgentError(f"round_number 必须在 [1, {MAX_ROUNDS}]")
    template = SUB_ASPECT_CATALOG[template_id]
    try:
        tool_request = validate_tool_request(
            template["tool_request_factory"](),
            expected_metric=template["tool_metric"],
        )
    except RuntimeError as exc:
        raise PlanAgentError(f"catalog tool_request 无效: {exc}") from exc
    seeds = list(template["seeds"])
    return {
        "round_id": f"round_{round_number}",
        "template_id": template_id,
        "sub_aspect": template["sub_aspect"],
        "rationale": template["rationale"],
        "task_instruction": template["task_instruction"],
        "route": template["route"],
        "variant_hint": deepcopy(template["variant_hint"]),
        "execution": {
            "seeds": seeds,
            "num_episodes": len(seeds),
            "gates": list(REQUIRED_GATES),
        },
        "observations": list(REQUIRED_OBSERVATIONS),
        "tool_request": tool_request,
    }


def _validate_current_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise PlanAgentError("current_plan 必须是 JSON object")
    if plan.get("schema_version") != 5:
        raise PlanAgentError("current_plan.schema_version 必须是 5")
    if plan.get("task_name") != "beat_block_hammer":
        raise PlanAgentError("当前原型只支持 task_name=beat_block_hammer")
    if plan.get("policy") != EXPECTED_POLICY:
        raise PlanAgentError(f"policy metadata 必须为 {EXPECTED_POLICY}")
    if plan.get("max_rounds") != MAX_ROUNDS:
        raise PlanAgentError(f"max_rounds 必须是 {MAX_ROUNDS}")
    requested = _validate_template_ids(plan.get("requested_template_ids"))
    rounds = plan.get("rounds")
    if not isinstance(rounds, list) or not rounds or len(rounds) > MAX_ROUNDS:
        raise PlanAgentError("current_plan.rounds 数量必须在 [1, 3]")
    executed: list[str] = []
    for number, round_plan in enumerate(rounds, start=1):
        if not isinstance(round_plan, dict):
            raise PlanAgentError(f"round_{number} 必须是 object")
        template_id = round_plan.get("template_id")
        if template_id not in requested:
            raise PlanAgentError("round template 必须来自 requested_template_ids")
        if template_id in executed:
            raise PlanAgentError("同一 template 不得重复执行")
        if round_plan != _materialize_round(template_id, number):
            raise PlanAgentError(f"round_{number} 与 trusted catalog 不一致")
        executed.append(template_id)
    return plan


def validate_evaluation_plan(proposal: dict[str, Any]) -> dict[str, Any]:
    """Validate a small model proposal and inject a trusted first round."""

    if not isinstance(proposal, dict):
        raise PlanAgentError("EvaluationProposal 必须是 JSON object")
    _require_exact_keys(proposal, INITIAL_PROPOSAL_KEYS, "EvaluationProposal")
    if proposal.get("schema_version") != 5:
        raise PlanAgentError("EvaluationProposal.schema_version 必须是 5")
    if proposal.get("task_name") != "beat_block_hammer":
        raise PlanAgentError("当前原型只支持 task_name=beat_block_hammer")
    if proposal.get("policy") != EXPECTED_POLICY:
        raise PlanAgentError(f"policy metadata 必须为 {EXPECTED_POLICY}")
    if proposal.get("max_rounds") != MAX_ROUNDS:
        raise PlanAgentError(f"max_rounds 必须是 {MAX_ROUNDS}")
    requested = _validate_template_ids(proposal.get("requested_template_ids"))
    first_template = proposal.get("first_template_id")
    if first_template not in requested:
        raise PlanAgentError("first_template_id 必须来自 requested_template_ids")

    return {
        "schema_version": 5,
        "task_name": "beat_block_hammer",
        "policy": dict(EXPECTED_POLICY),
        "evaluation_goal": _require_string(
            proposal.get("evaluation_goal"), "evaluation_goal"
        ),
        "requested_template_ids": requested,
        "rounds": [_materialize_round(first_template, 1)],
        "round_decisions": [],
        "max_rounds": MAX_ROUNDS,
        "planning_state": "awaiting_round_1_observation",
    }


def _validate_observation_history(
    current_plan: dict[str, Any], observation_history: Any
) -> list[dict[str, Any]]:
    if not isinstance(observation_history, list) or not observation_history:
        raise PlanAgentError("observation_history 必须是非空 list")
    rounds = current_plan["rounds"]
    if len(observation_history) != len(rounds):
        raise PlanAgentError("每个已规划 round 必须恰好有一条 observation")
    normalized: list[dict[str, Any]] = []
    for round_plan, observation in zip(rounds, observation_history):
        if not isinstance(observation, dict):
            raise PlanAgentError("每条 observation 必须是 object")
        if observation.get("round_id") != round_plan["round_id"]:
            raise PlanAgentError("observation.round_id 与 plan 不一致")
        if not isinstance(observation.get("pipeline_passed"), bool):
            raise PlanAgentError("observation.pipeline_passed 必须是 boolean")
        normalized.append(deepcopy(observation))
    return normalized


def validate_next_round_decision(
    decision: dict[str, Any],
    current_plan: dict[str, Any],
    observation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate one generic bounded decision and materialize its next round."""

    current = _validate_current_plan(current_plan)
    history = _validate_observation_history(current, observation_history)
    if not isinstance(decision, dict):
        raise PlanAgentError("NextRoundDecision 必须是 JSON object")
    _require_exact_keys(decision, DECISION_KEYS, "NextRoundDecision")
    if decision.get("schema_version") != 2:
        raise PlanAgentError("NextRoundDecision.schema_version 必须是 2")

    action = decision.get("action")
    if action not in {"continue", "stop"}:
        raise PlanAgentError("action 只允许 continue 或 stop")
    summary = _require_string(
        decision.get("observation_summary"), "observation_summary"
    )
    reason = _require_string(decision.get("decision_reason"), "decision_reason")
    executed = [item["template_id"] for item in current["rounds"]]
    remaining = [
        item for item in current["requested_template_ids"] if item not in executed
    ]
    latest_pipeline_passed = history[-1]["pipeline_passed"]
    budget_exhausted = len(current["rounds"]) >= current["max_rounds"]
    stop_is_forced = (
        not latest_pipeline_passed or budget_exhausted or not remaining
    )
    next_template_id = decision.get("next_template_id")

    if stop_is_forced and action != "stop":
        raise PlanAgentError(
            "pipeline failure、round budget exhausted 或无剩余 template 时必须 stop"
        )
    if not stop_is_forced and action != "continue":
        raise PlanAgentError(
            "pipeline 通过且仍有用户请求的 template 时必须继续收集证据"
        )
    if action == "stop":
        if next_template_id is not None:
            raise PlanAgentError("stop decision 的 next_template_id 必须为 null")
        next_round = None
    else:
        if next_template_id not in remaining:
            raise PlanAgentError(
                "continue 只能选择尚未执行且由用户请求的 template"
            )
        next_round = _materialize_round(next_template_id, len(current["rounds"]) + 1)

    return {
        "schema_version": 2,
        "action": action,
        "observation_summary": summary,
        "decision_reason": reason,
        "next_template_id": next_template_id,
        "remaining_template_ids_before_decision": remaining,
        "round_budget_before_decision": current["max_rounds"] - len(current["rounds"]),
        "next_round": next_round,
    }


def _catalog_for_prompt() -> dict[str, Any]:
    return {
        template_id: {
            "sub_aspect": value["sub_aspect"],
            "description": value["rationale"],
            "episodes": len(value["seeds"]),
            "tool_metric": value["tool_metric"],
        }
        for template_id, value in SUB_ASPECT_CATALOG.items()
    }


def _initial_plan_prompt(repo_root: Path, user_request: str) -> str:
    agent_readme = (repo_root / "mea/planner/README.Agent.md").read_text(
        encoding="utf-8"
    )
    example = {
        "schema_version": 5,
        "task_name": "beat_block_hammer",
        "policy": EXPECTED_POLICY,
        "evaluation_goal": "evaluate_requested_blue_block_aspects",
        "requested_template_ids": list(SUB_ASPECT_CATALOG),
        "first_template_id": "object_appearance.color_blue",
        "max_rounds": MAX_ROUNDS,
    }
    return f"""你是 MEA 的外层 Plan Agent。你只选择受限 sub-aspect template，不生成 Python，
不写 seed、gate、TaskGen route 或 Tool route；系统会从 trusted catalog 注入这些执行细节。

USER QUERY:
{user_request}

POLICY / SIMULATOR CAPABILITIES:
{agent_readme}

TRUSTED SUB-ASPECT CATALOG:
{json.dumps(_catalog_for_prompt(), ensure_ascii=False, indent=2)}

只选择用户明确要求的 template。初始阶段只输出 requested_template_ids 和第一个 template，
不要输出 rounds 或任何执行字段。输出严格 JSON，不要 Markdown。示例：
{json.dumps(example, ensure_ascii=False, indent=2)}
"""


def _decision_prompt(
    user_request: str,
    current_plan: dict[str, Any],
    observation_history: list[dict[str, Any]],
) -> str:
    executed = [item["template_id"] for item in current_plan["rounds"]]
    remaining = [
        item
        for item in current_plan["requested_template_ids"]
        if item not in executed
    ]
    forced_stop = (
        not observation_history[-1]["pipeline_passed"]
        or len(current_plan["rounds"]) >= current_plan["max_rounds"]
        or not remaining
    )
    example = {
        "schema_version": 2,
        "action": "stop" if forced_stop else "continue",
        "observation_summary": "概括全部已有 observation，并区分 pipeline 与 policy 结果。",
        "decision_reason": (
            "流水线失败、预算耗尽或用户要求已经覆盖。"
            if forced_stop
            else "继续收集尚未覆盖的用户请求证据。"
        ),
        "next_template_id": None if forced_stop else remaining[0],
    }
    return f"""你是 MEA 的外层 Plan Agent。根据完整 observation history 决定继续或停止。
模型只选择 action 和 template id；系统负责注入下一轮的所有执行字段。

USER QUERY:
{user_request}

CURRENT PLAN:
{json.dumps(current_plan, ensure_ascii=False, indent=2)}

OBSERVATION HISTORY:
{json.dumps(observation_history, ensure_ascii=False, indent=2)}

EXECUTED TEMPLATE IDS:
{json.dumps(executed, ensure_ascii=False)}

REMAINING REQUESTED TEMPLATE IDS:
{json.dumps(remaining, ensure_ascii=False)}

规则：最新 pipeline_passed=false、max_rounds={MAX_ROUNDS} 已用尽或没有剩余 template 时必须 stop。
否则必须 continue 到一个 remaining template。可根据 observations 决定剩余项目的顺序，
但不得跳过用户已请求的证据、重复 template 或选择 catalog 外项目。
只输出以下严格 JSON 结构，不要输出 next_round、seed、gate 或 route：
{json.dumps(example, ensure_ascii=False, indent=2)}
"""


class PlanAgentPrototype:
    """Select catalog templates and adapt for at most three rounds."""

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
            raise PlanAgentError("evaluation_id 必须是合法目录名并以 eval_ 开头")

        evaluation_dir = self.repo_root / "mea/evaluation_runs" / evaluation_id
        if evaluation_dir.exists():
            raise PlanAgentError(f"evaluation directory 已存在: {evaluation_dir}")
        for child in ("plan", "execution", "summary"):
            (evaluation_dir / child).mkdir(parents=True, exist_ok=False)

        manifest: dict[str, Any] = {
            "schema_version": 5,
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
        errors: list[str] = []
        plan = None
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
                system="只输出满足 EvaluationProposal schema 的 JSON object。",
                max_tokens=1200,
                temperature=0.0,
            )
            suffix = "" if attempt == 0 else f"_retry_{attempt}"
            (evaluation_dir / f"plan/round_1_response{suffix}.txt").write_text(
                response + "\n", encoding="utf-8"
            )
            try:
                plan = validate_evaluation_plan(extract_json_response(response))
                break
            except PlanAgentError as exc:
                errors.append(str(exc))
        if plan is None:
            raise PlanAgentError(f"EvaluationProposal 两次均未通过: {errors}")
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
                    "round_1_validation_errors": errors,
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
        observation_history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Use all observations to select a remaining template or stop."""

        evaluation_dir = self.repo_root / "mea/evaluation_runs" / evaluation_id
        if not evaluation_dir.is_dir():
            raise PlanAgentError(f"evaluation directory 不存在: {evaluation_dir}")
        current = _validate_current_plan(current_plan)
        history = _validate_observation_history(current, observation_history)
        completed_round = len(current["rounds"])
        prompt = _decision_prompt(
            _require_string(user_request, "user_request"), current, history
        )
        stem = f"decision_after_round_{completed_round}"
        (evaluation_dir / f"plan/{stem}_prompt.md").write_text(
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
                system="只基于 observations 选择 action/template，并输出严格 JSON。",
                max_tokens=900,
                temperature=0.0,
            )
            suffix = "" if attempt == 0 else f"_retry_{attempt}"
            (evaluation_dir / f"plan/{stem}_response{suffix}.txt").write_text(
                response + "\n", encoding="utf-8"
            )
            try:
                decision = validate_next_round_decision(
                    extract_json_response(response), current, history
                )
                break
            except PlanAgentError as exc:
                errors.append(str(exc))
        if decision is None:
            raise PlanAgentError(f"NextRoundDecision 两次均未通过: {errors}")

        updated_plan = deepcopy(current)
        updated_plan.setdefault("round_decisions", []).append(decision)
        if decision["action"] == "continue":
            updated_plan["rounds"].append(decision["next_round"])
            next_number = len(updated_plan["rounds"])
            updated_plan["planning_state"] = (
                f"awaiting_round_{next_number}_observation"
            )
        else:
            updated_plan["planning_state"] = f"stopped_after_round_{completed_round}"

        decision_path = evaluation_dir / f"plan/{stem}.json"
        _write_json(decision_path, decision)
        _write_json(evaluation_dir / "plan/evaluation_plan.json", updated_plan)
        manifest_path = evaluation_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": updated_plan["planning_state"],
                "plan": updated_plan,
                f"{stem}_path": f"plan/{stem}.json",
            }
        )
        planner = manifest.setdefault("planner", {})
        planner[f"{stem}_metadata"] = dict(
            getattr(self.provider, "last_metadata", {})
        )
        planner[f"{stem}_validation_errors"] = errors
        _write_json(manifest_path, manifest)
        return updated_plan, decision
