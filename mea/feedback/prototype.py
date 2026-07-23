"""Evidence-grounded feedback generation and unified report rendering."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mea.taskgen import extract_json_response

from .answer_scope import (
    build_answer_scope,
    project_answer_scope,
    validate_answer_scope_projection,
)


class FeedbackAgentError(RuntimeError):
    """Raised when final feedback violates the structured output contract."""


FALSE_POLICY_SUCCESS_PATTERNS = (
    r"任务成功完成",
    r"成功完成任务",
    r"策略执行成功",
    r"policy\s*(?:执行)?成功",
    r"ACT.*任务.*成功",
    r"表现符合任务要求",
)


def _deterministic_aggregate(evidence: dict[str, Any]) -> dict[str, Any] | None:
    """Return the evaluation-level deterministic Aggregate result, if present."""

    observations = evidence.get("observations")
    if isinstance(observations, dict) and isinstance(
        observations.get("aggregate"), dict
    ):
        return observations["aggregate"]
    aggregate = evidence.get("aggregate")
    return aggregate if isinstance(aggregate, dict) else None


def _authoritative_policy_success(evidence: dict[str, Any]) -> float | None:
    """Read policy success from a precomputed aggregate without recomputing it."""

    aggregate = _deterministic_aggregate(evidence)
    for metric in (aggregate or {}).get("metrics", []):
        if metric.get("metric") != "official_check_success":
            continue
        for cohort in metric.get("cohorts", []):
            if cohort.get("role") != "policy_under_evaluation":
                continue
            statistics = cohort.get("summary", {}).get("statistics", {})
            for statistic_name in ("success_rate", "true_rate"):
                statistic = statistics.get(statistic_name)
                value = (
                    statistic.get("value")
                    if isinstance(statistic, dict)
                    else None
                )
                if value is not None:
                    return float(value)
    policy_success = evidence.get("observations", {}).get("policy_success")
    return float(policy_success) if policy_success is not None else None


def _execution_vqa_entries(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect round-level Execution VQA observations for reporting."""

    entries: list[dict[str, Any]] = []
    direct = evidence.get("execution_vqa")
    if isinstance(direct, dict):
        entries.append(direct)
    for round_evidence in evidence.get("rounds", []):
        if not isinstance(round_evidence, dict):
            continue
        item = round_evidence.get("execution_vqa")
        if isinstance(item, dict):
            entries.append(
                {
                    "round_id": round_evidence.get("round_id"),
                    **item,
                }
            )
    return entries


def _has_execution_vqa_conflict(evidence: dict[str, Any]) -> bool:
    observations = evidence.get("observations")
    if isinstance(observations, dict) and observations.get(
        "execution_vqa_conflict"
    ) is True:
        return True
    return any(
        bool(item.get("evidence_conflict"))
        for item in _execution_vqa_entries(evidence)
    )


def _claims_policy_success(text: str) -> bool:
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in FALSE_POLICY_SUCCESS_PATTERNS
    )


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeedbackAgentError(f"{field} 必须是非空字符串")
    return value.strip()


def _require_text_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise FeedbackAgentError(f"{field} 必须是非空字符串 list")
    return [_require_text(item, f"{field}[]") for item in value]


def validate_feedback(
    value: dict[str, Any],
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FeedbackAgentError("Feedback 必须是 JSON object")
    feedback = {
        "answer": _require_text(value.get("answer"), "answer"),
        "evaluation_scope": _require_text(
            value.get("evaluation_scope"), "evaluation_scope"
        ),
        "findings": _require_text_list(value.get("findings"), "findings"),
        "limitations": _require_text_list(value.get("limitations"), "limitations"),
        "recommended_next_step": _require_text(
            value.get("recommended_next_step"), "recommended_next_step"
        ),
    }
    policy_success = _authoritative_policy_success(evidence or {})
    if policy_success is not None and float(policy_success) <= 0.0:
        conclusion_text = "\n".join(
            [feedback["answer"], *feedback["findings"]]
        )
        if _claims_policy_success(conclusion_text):
            raise FeedbackAgentError(
                "answer/findings 声称任务成功，但 evidence 中 policy_success <= 0"
            )
    return feedback


def apply_deterministic_consistency_guard(
    value: dict[str, Any],
    evidence: dict[str, Any],
    *,
    validation_errors: list[str] | None = None,
    attempts_used: int = 1,
) -> dict[str, Any]:
    """Force policy-success wording to agree with the numeric evidence."""

    feedback = validate_feedback(value)
    policy_success = _authoritative_policy_success(evidence)
    deterministic_correction = False
    if policy_success is not None and float(policy_success) <= 0.0:
        conclusion_text = "\n".join(
            [feedback["answer"], *feedback["findings"]]
        )
        if _claims_policy_success(conclusion_text):
            feedback["answer"] = (
                "场景生成和评估流水线通过，但 ACT policy 在本次 episode "
                f"未完成任务（policy_success={float(policy_success):.1f}）。"
            )
            feedback["findings"] = [
                item
                for item in feedback["findings"]
                if not _claims_policy_success(item)
            ]
            feedback["findings"].extend(
                [
                    "场景生成、视觉对齐和评估流水线已完成。",
                    (
                        "ACT policy 在本次 episode 未完成任务，"
                        f"policy_success={float(policy_success):.1f}。"
                    ),
                ]
            )
            deterministic_correction = True
    feedback["consistency_validation"] = {
        "passed": True,
        "attempts_used": attempts_used,
        "rejected_responses": len(validation_errors or []),
        "errors": list(validation_errors or []),
        "deterministic_correction": deterministic_correction,
    }
    validate_feedback(feedback, evidence)
    scope = build_answer_scope(evidence)
    feedback = project_answer_scope(feedback, scope)
    validate_answer_scope_projection(feedback, scope)
    return feedback


def _feedback_prompt(repo_root: Path, evidence: dict[str, Any]) -> str:
    instructions = (repo_root / "mea/feedback/README.Agent.md").read_text(
        encoding="utf-8"
    )
    answer_scope = build_answer_scope(evidence)
    return f"""你是 MEA 的最终 Feedback Agent。请基于证据回答用户，不要补充未经测试的结论。

EVIDENCE INTERPRETATION CONTRACT:
1. `observations.aggregate` 是 deterministic Aggregate Toolkit 已经计算好的结果。
   直接引用其中 count/rate/mean/median/min/max/stddev 与 quality；禁止从 episode、
   ToolResult 或若干 JSON 自行做数学运算。
2. `policy_under_evaluation` 与 `expert_validation` 是不同 cohort，禁止合并。
3. simulator numeric Tool 是距离、接触、时间、冲量、成功等数值事实的权威来源。
   Execution VQA 只补充颜色、可见抬起、可见位移等视觉现象，不能覆盖数值 Tool。
4. 若 Execution VQA 含 `evidence_conflict=true`，明确报告冲突和对应 frame，保留
   simulator Tool 结论，并建议复查或追加测试；不要替视觉或数值一方消除冲突。
5. `history_retrieval` 只用于保持 planning decomposition 一致。历史 policy outcome
   不是本次 evaluation evidence，禁止与本次 Aggregate 合并或据此声称本次成功。
6. `ANSWER SCOPE` 是 deterministic validator 从证据投影的硬边界。回答必须与其
   N/seeds、未测试候选、unsupported capability、冲突和停止原因一致。

AGENT RULES:
{instructions}

EVIDENCE BUNDLE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

ANSWER SCOPE:
{json.dumps(answer_scope, ensure_ascii=False, indent=2)}

返回严格 JSON，不要输出 Markdown：
{{
  "answer": "面向用户的简洁回答",
  "evaluation_scope": "本次实际测试范围",
  "findings": ["证据支持的发现"],
  "limitations": ["一个 episode 等限制"],
  "recommended_next_step": "下一项最有价值的评估"
}}
"""


def feedback_markdown(feedback: dict[str, Any]) -> str:
    findings = "\n".join(f"- {item}" for item in feedback["findings"])
    limitations = "\n".join(f"- {item}" for item in feedback["limitations"])
    return (
        "# Evaluation Feedback\n\n"
        f"{feedback['answer']}\n\n"
        "## Evaluation scope\n\n"
        f"{feedback['evaluation_scope']}\n\n"
        "## Findings\n\n"
        f"{findings}\n\n"
        "## Limitations\n\n"
        f"{limitations}\n\n"
        "## Recommended next step\n\n"
        f"{feedback['recommended_next_step']}\n"
    )


class FeedbackAgent:
    """Summarize a completed evaluation using a separate LLM call."""

    def __init__(self, repo_root: str | Path, provider: Any, *, model: str):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.provider = provider
        self.model = model

    def generate(
        self,
        evidence: dict[str, Any],
        *,
        output_dir: Path,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt = _feedback_prompt(self.repo_root, evidence)
        (output_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        feedback = None
        last_structured_feedback = None
        validation_errors: list[str] = []
        for attempt_index in range(2):
            attempt_prompt = prompt
            if validation_errors:
                attempt_prompt += f"""

PREVIOUS RESPONSE VALIDATION ERROR:
{validation_errors[-1]}

Regenerate the entire strict JSON. Pipeline completion never means policy task
success. If policy_success is 0.0, explicitly state that the policy did not
complete the task.
"""
                (output_dir / "retry_prompt.md").write_text(
                    attempt_prompt, encoding="utf-8"
                )
            response = self.provider.text(
                attempt_prompt,
                model=self.model,
                system="Use only the evidence and return strict Feedback JSON.",
                max_tokens=1200,
                temperature=0.0,
            )
            response_name = (
                "response.txt" if attempt_index == 0 else "retry_response.txt"
            )
            (output_dir / response_name).write_text(
                response + "\n", encoding="utf-8"
            )
            try:
                parsed = extract_json_response(response)
                last_structured_feedback = validate_feedback(parsed)
                feedback = validate_feedback(
                    parsed,
                    evidence,
                )
                break
            except FeedbackAgentError as exc:
                validation_errors.append(str(exc))
        deterministic_correction = False
        if feedback is None:
            policy_success = _authoritative_policy_success(evidence)
            if last_structured_feedback is None or policy_success is None:
                raise FeedbackAgentError(
                    "Feedback 两次响应均未通过，且没有可校正的 structured output: "
                    f"{validation_errors}"
                )
            feedback = apply_deterministic_consistency_guard(
                last_structured_feedback,
                evidence,
                validation_errors=validation_errors,
                attempts_used=2,
            )
            deterministic_correction = bool(
                feedback["consistency_validation"]["deterministic_correction"]
            )
        if not deterministic_correction:
            feedback["consistency_validation"] = {
                "passed": True,
                "attempts_used": len(validation_errors) + 1,
                "rejected_responses": len(validation_errors),
                "errors": validation_errors,
                "deterministic_correction": False,
            }
        aggregate = _deterministic_aggregate(evidence)
        feedback["evidence_policy"] = {
            "aggregate_source": (
                "deterministic_aggregate" if aggregate is not None else None
            ),
            "aggregate_status": (
                aggregate.get("status") if aggregate is not None else None
            ),
            "episode_math_by_feedback_agent": False,
            "numeric_simulator_tools_authoritative": True,
            "execution_vqa_is_visual_only": True,
            "evidence_conflict": _has_execution_vqa_conflict(evidence),
        }
        feedback["provider_metadata"] = dict(
            getattr(self.provider, "last_metadata", {})
        )
        answer_scope = build_answer_scope(evidence)
        feedback = project_answer_scope(feedback, answer_scope)
        validate_answer_scope_projection(feedback, answer_scope)
        (output_dir / "feedback.json").write_text(
            json.dumps(feedback, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "feedback.md").write_text(
            feedback_markdown(feedback), encoding="utf-8"
        )
        return feedback


def render_evaluation_report(
    evidence: dict[str, Any],
    feedback: dict[str, Any],
) -> str:
    """Render the one-file human entry point for single- or multi-round runs."""

    def report_value(value: Any, *, na_reason: str) -> str:
        return f"N/A ({na_reason})" if value is None else str(value)

    aggregate = _deterministic_aggregate(evidence)
    aggregate_markdown = (
        "以下数值直接来自 deterministic Aggregate Toolkit；Feedback Agent "
        "没有重新计算 episode 统计量。\n\n"
        "```json\n"
        + json.dumps(aggregate, ensure_ascii=False, indent=2)
        + "\n```"
        if aggregate is not None
        else "No deterministic cross-episode aggregate was available."
    )
    execution_vqa_entries = _execution_vqa_entries(evidence)
    execution_vqa_markdown = (
        "Execution VQA 只提供视觉补充证据。Simulator numeric Tool 保持权威；"
        "不一致会原样保留为 `evidence_conflict`。\n\n```json\n"
        + json.dumps(execution_vqa_entries, ensure_ascii=False, indent=2)
        + "\n```"
        if execution_vqa_entries
        else "No Execution VQA observation was available."
    )
    history_retrieval = evidence.get("history_retrieval") or {}
    history_markdown = (
        "以下历史只作为 planning prior，不属于本次 policy evidence。\n\n"
        "```json\n"
        + json.dumps(history_retrieval, ensure_ascii=False, indent=2)
        + "\n```"
        if history_retrieval
        else "No historical planning context was available."
    )

    if evidence.get("rounds"):
        observations = evidence["observations"]
        round_sections = []
        for item in evidence["rounds"]:
            round_observations = item["observations"]
            positions = round_observations.get("position_samples", [])
            position_rows = []
            for sample in positions:
                if "bell_position" in sample:
                    position_name = "bell_position"
                elif "block_position" in sample:
                    position_name = "block_position"
                else:
                    position_name = "position"
                matched = (
                    f", matched={sample['position_matched']}"
                    if "position_matched" in sample
                    else ""
                )
                position_rows.append(
                    "  - episode {episode}, seed {seed}: {name}={position}{matched}".format(
                        episode=sample.get("episode_index", "unknown"),
                        seed=sample.get("seed", "unknown"),
                        name=position_name,
                        position=sample.get(position_name),
                        matched=matched,
                    )
                )
            position_lines = "\n".join(position_rows) or "  - none"
            selected = ", ".join(
                f"`{name}`"
                for name in item.get("task_retrieval", {}).get(
                    "selected_tasks", []
                )
            ) or "none (reuse route)"
            tool_evaluation = item.get("tool_evaluation") or {}
            planned_reference = tool_evaluation.get("reference_tool")
            planned_tool_lines = []
            for episode in tool_evaluation.get("episodes", []):
                result = episode.get("result", {})
                planned_tool_lines.append(
                    "  - {policy} ({role}) seed {seed}: value={value}, "
                    "evidence_steps={steps}".format(
                        policy=episode.get("policy_name"),
                        role=episode.get("role"),
                        seed=episode.get("seed"),
                        value=result.get("value"),
                        steps=result.get("evidence_steps", []),
                    )
                )
            planned_tool_result_lines = (
                "\n".join(planned_tool_lines) or "  - none"
            )
            validation = tool_evaluation.get("validation", {})
            validation_summary = {
                key: validation[key]
                for key in (
                    "provider_called",
                    "successful_attempt",
                    "all_gates_passed",
                    "catalog_tool_found",
                    "episode_count",
                )
                if key in validation
            }
            planned_tool_markdown = (
                "- planned Tool requested route: `{requested_route}`\n"
                "- planned Tool resolved route: `{route}`\n"
                "- planned Tool source: `{scope}`\n"
                "- planned Tool: `{tool}`\n"
                "- planned Tool validation: `{validation}`\n"
                "- planned Tool results:\n{results}".format(
                    requested_route=tool_evaluation.get(
                        "requested_route", "explicit"
                    ),
                    route=tool_evaluation.get("route"),
                    scope=tool_evaluation.get("source", {}).get("scope"),
                    tool=tool_evaluation.get("source", {}).get("tool"),
                    validation=validation_summary,
                    results=planned_tool_result_lines,
                )
                if tool_evaluation
                else ""
            )
            tool_lines = []
            for episode in item.get("trusted_tool_evaluation", {}).get(
                "episodes", []
            ):
                result_text = ", ".join(
                    "{tool}={value}{unit}".format(
                        tool=result.get("tool"),
                        value=result.get("value"),
                        unit=(
                            f" {result.get('unit')}"
                            if result.get("unit")
                            else ""
                        ),
                    )
                    for result in episode.get("results", [])
                    if result.get("tool") != planned_reference
                )
                tool_lines.append(
                    "  - {policy} seed {seed}: {results}".format(
                        policy=episode.get("policy_name"),
                        seed=episode.get("seed"),
                        results=result_text or "none",
                    )
                )
            trusted_tool_lines = "\n".join(tool_lines) or "  - none"
            execution_backend = str(
                round_observations.get("execution_backend") or "ACT"
            )
            act_pipeline_display = report_value(
                round_observations.get("act_pipeline_status"),
                na_reason=f"{execution_backend} backend",
            )
            policy_success_display = report_value(
                round_observations.get("policy_success"),
                na_reason=f"{execution_backend} backend",
            )
            round_sections.append(
                f"""### {item['round_id']}: `{item['sub_aspect']}`

- TaskGen route: `{item['route']}`
- instruction: {item['task_instruction']}
- seeds: `{item['seeds']}`
- episodes: `{item['num_episodes']}`
- selected retrieval tasks: {selected}
- observed color: `{round_observations.get('observed_color')}`
- expert solvable: `{round_observations.get('expert_solvable')}`
- execution backend: `{execution_backend}`
- ACT pipeline status: `{act_pipeline_display}`
- policy success: `{policy_success_display}`
- pipeline passed: `{round_observations.get('pipeline_passed')}`
{planned_tool_markdown}
- position samples:
{position_lines}
- trusted Tool results:
{trusted_tool_lines}
"""
            )
        rounds_markdown = "\n".join(round_sections)
        execution_backends = observations.get("execution_backends") or ["ACT"]
        aggregate_backend_label = ", ".join(str(item) for item in execution_backends)
        aggregate_act_display = report_value(
            observations.get("act_pipeline_status"),
            na_reason=f"{aggregate_backend_label} backend",
        )
        aggregate_policy_display = report_value(
            observations.get("policy_success"),
            na_reason=f"{aggregate_backend_label} backend",
        )
        findings = "\n".join(f"- {item}" for item in feedback["findings"])
        limitations = "\n".join(f"- {item}" for item in feedback["limitations"])
        artifacts = evidence["artifacts"]
        decision_artifacts = artifacts.get("plan_decisions")
        if decision_artifacts is None:
            legacy_decision = artifacts.get("round_2_decision")
            decision_artifacts = [legacy_decision] if legacy_decision else []
        decision_artifact_lines = "\n".join(
            f"- Plan decision: `{path}`" for path in decision_artifacts
        ) or "- Plan decision: `none`"
        assessment_artifact_lines = "\n".join(
            f"- Evidence assessment: `{path}`"
            for path in artifacts.get("evidence_assessments", [])
        ) or "- Evidence assessment: `none`"
        round_artifact_lines = []
        for round_index, round_artifacts in enumerate(
            artifacts.get("round_artifacts", []), start=1
        ):
            for name, path in round_artifacts.items():
                if path is None:
                    continue
                paths = path if isinstance(path, list) else [path]
                round_artifact_lines.extend(
                    f"- round {round_index} `{name}`: `{item}`"
                    for item in paths
                )
        round_artifact_markdown = (
            "\n".join(round_artifact_lines)
            or "- round artifacts: `none`"
        )
        return f"""# MEA Multi-Round Evaluation Report

## Identity

- evaluation id: `{evidence['evaluation_id']}`
- user query: {evidence['user_request']}
- executed rounds: `{evidence['plan']['executed_rounds']}`
- total episodes: `{evidence['total_episodes']}`

## Plan Agent decisions

```json
{json.dumps(evidence['plan']['round_decisions'], ensure_ascii=False, indent=2)}
```

## Historical planning retrieval

{history_markdown}

## Round evidence

{rounds_markdown}

## Aggregate observations

- scene alignment: `{observations['scene_alignment']}`
- observed color by round: `{observations['observed_color_by_round']}`
- expert solvable: `{observations['expert_solvable']}`
- execution backends: `{execution_backends}`
- ACT pipeline status: `{aggregate_act_display}`
- weighted policy success: `{aggregate_policy_display}`
- policy success by round: `{observations['policy_success_by_round']}`
- position varied: `{observations['position_varied']}`
- position metrics: `{observations['position_metrics']}`
- pipeline passed: `{observations['pipeline_passed']}`

## Deterministic Aggregate Toolkit

{aggregate_markdown}

## Execution VQA

{execution_vqa_markdown}

## Feedback Agent answer

{feedback['answer']}

### Findings

{findings}

### Limitations

{limitations}

### Recommended next step

{feedback['recommended_next_step']}

## Artifact index

- evaluation plan: `{artifacts['evaluation_plan']}`
- history retrieval: `{artifacts.get('history_retrieval')}`
{decision_artifact_lines}
{assessment_artifact_lines}
- machine-readable summary: `{artifacts['summary']}`
- deterministic aggregate: `{artifacts.get('aggregate')}`
{round_artifact_markdown}
"""

    retrieval = evidence["task_retrieval"]
    observations = evidence["observations"]
    reflection = evidence.get("visual_self_reflection", {})
    artifacts = evidence["artifacts"]
    selected = ", ".join(f"`{name}`" for name in retrieval["selected_tasks"])
    artifact_lines = "\n".join(
        f"- `{name}`: `{path}`" for name, path in artifacts.items()
    )
    findings = "\n".join(f"- {item}" for item in feedback["findings"])
    limitations = "\n".join(f"- {item}" for item in feedback["limitations"])
    return f"""# MEA Evaluation Report

## Identity

- evaluation id: `{evidence['evaluation_id']}`
- child TaskGen run: `{evidence['child_run_id']}`
- user query: {evidence['user_request']}
- sub-aspect: `{evidence['sub_aspect']}`

## Plan

- task instruction: {evidence['task_instruction']}
- route: `{evidence['route']}`
- seed: `{evidence['seed']}`
- episodes: `{evidence['num_episodes']}`

## Task retrieval

- catalog size: `{retrieval['catalog_size']}`
- selected tasks: {selected}
- reasoning: {retrieval['reasoning']}

## Execution observations

- scene alignment: `{observations['scene_alignment']}`
- observed color: `{observations['observed_color']}`
- expert solvable: `{observations['expert_solvable']}`
- ACT pipeline status: `{observations['act_pipeline_status']}`
- policy success: `{observations['policy_success']}`
- pipeline passed: `{observations['pipeline_passed']}`
- visual reflection passed: `{reflection.get('passed')}`
- visual repairs used: `{reflection.get('repairs_used')}`
- visual attempts: `{reflection.get('attempt_count')}`

## Deterministic Aggregate Toolkit

{aggregate_markdown}

## Execution VQA

{execution_vqa_markdown}

## Feedback Agent answer

{feedback['answer']}

### Findings

{findings}

### Limitations

{limitations}

### Recommended next step

{feedback['recommended_next_step']}

## Artifact index

{artifact_lines}
"""
