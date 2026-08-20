"""Evidence-grounded Plan Agent final-answer generation and report rendering."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mea.providers.json_response import extract_json_response

from .answer_evidence import project_answer_evidence
from .answer_scope import (
    build_answer_scope,
    project_answer_scope,
    validate_answer_scope_projection,
)


class PlanAgentFinalSummaryError(RuntimeError):
    """Raised when the Plan Agent final summary violates its output contract."""


FALSE_POLICY_SUCCESS_PATTERNS = (
    r"任务成功完成",
    r"成功完成任务",
    r"策略执行成功",
    r"policy\s*(?:执行)?成功",
    r"ACT.*任务.*成功",
    r"表现符合任务要求",
)


def _authoritative_policy_success(evidence: dict[str, Any]) -> float | None:
    """Read the seed-weighted success fact from typed RoundEvidence records."""

    weighted = 0.0
    samples = 0
    for item in evidence.get("rounds", []):
        if not isinstance(item, dict):
            continue
        policy = item.get("policy")
        if not isinstance(policy, dict):
            continue
        rate = policy.get("success_rate")
        seeds = policy.get("seeds")
        if rate is None or not isinstance(seeds, list) or not seeds:
            continue
        weighted += float(rate) * len(seeds)
        samples += len(seeds)
    return weighted / samples if samples else None


def _execution_vqa_entries(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect round-level Execution VQA observations for reporting."""

    entries: list[dict[str, Any]] = []
    for round_evidence in evidence.get("rounds", []):
        if not isinstance(round_evidence, dict):
            continue
        item = round_evidence.get("vqa")
        if isinstance(item, dict):
            entries.append(
                {
                    "round_id": round_evidence.get("round_id"),
                    **item,
                }
            )
    return entries


def _has_execution_vqa_conflict(evidence: dict[str, Any]) -> bool:
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
        raise PlanAgentFinalSummaryError(f"{field} 必须是非空字符串")
    return value.strip()


def _require_text_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PlanAgentFinalSummaryError(f"{field} 必须是非空字符串 list")
    return [_require_text(item, f"{field}[]") for item in value]


def validate_feedback(
    value: dict[str, Any],
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanAgentFinalSummaryError("Plan Agent answer 必须是 JSON object")
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
            raise PlanAgentFinalSummaryError(
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
                "场景生成和评估流水线通过，但被评估策略在本次 episode "
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
                        "被评估策略在本次 episode 未完成任务，"
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
    answer_evidence = project_answer_evidence(evidence)
    return f"""你是 MEA Plan Agent 的最终总结阶段。请基于证据回答原始 Query，不要补充未经测试的结论。

EVIDENCE INTERPRETATION CONTRACT:
1. `ANSWER SCOPE` 是 deterministic validator 从 RoundEvidence 投影的硬边界。
   回答必须与其 N/seeds、未测试候选、冲突和停止原因一致。
2. `generated_check_success` 在 `expected_semantic_extension` 且无 evidence
   conflict 时，是该 Query 所定义的有界实验条件的结果；它不是官方 benchmark
   success，但不能仅因 `official_equivalent=false` 就被说成“无法判断实验条件”。
3. Tool 的时间语义不可改写：trajectory peak/max 是诊断量，不是 terminal/current
   value。不得用较大的 peak 把成功的 terminal checker 改写成 terminal failure。

AGENT RULES:
{instructions}

ANSWER EVIDENCE PROJECTION:
{json.dumps(answer_evidence, ensure_ascii=False, indent=2)}

ANSWER SCOPE:
{json.dumps(answer_scope, ensure_ascii=False, indent=2)}

返回严格 JSON，不要输出 Markdown：
{{
  "answer": "面向用户的简洁回答",
  "evaluation_scope": "本次实际测试范围",
  "findings": ["证据支持的发现"],
  "limitations": ["本次证据范围的限制"],
  "recommended_next_step": "下一项最有价值的评估"
}}
"""


def answer_markdown(feedback: dict[str, Any]) -> str:
    findings = "\n".join(f"- {item}" for item in feedback["findings"])
    limitations = "\n".join(f"- {item}" for item in feedback["limitations"])
    return (
        "# Plan Agent answer\n\n"
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


class PlanAgentFinalSummary:
    """Answer the original Query from the completed evidence bundle."""

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
                system=(
                    "You are the Plan Agent final-summary stage. Use only the "
                    "evidence and return strict answer JSON."
                ),
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
            except PlanAgentFinalSummaryError as exc:
                validation_errors.append(str(exc))
        deterministic_correction = False
        if feedback is None:
            policy_success = _authoritative_policy_success(evidence)
            if last_structured_feedback is None or policy_success is None:
                raise PlanAgentFinalSummaryError(
                    "Plan Agent final summary 两次响应均未通过，且没有可校正的 structured output: "
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
        feedback["evidence_policy"] = {
            "source": "RoundEvidence",
            "episode_math_by_plan_agent_summary": False,
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
        (output_dir / "answer.json").write_text(
            json.dumps(feedback, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "answer.md").write_text(
            answer_markdown(feedback), encoding="utf-8"
        )
        return feedback


# Historical import compatibility. New writers and production callers use the
# paper-aligned Plan Agent final-summary terminology.
FeedbackAgent = PlanAgentFinalSummary
FeedbackAgentError = PlanAgentFinalSummaryError
feedback_markdown = answer_markdown


def render_evaluation_report(
    evidence: dict[str, Any],
    feedback: dict[str, Any],
) -> str:
    """Render one compact human view of the typed final evidence bundle."""

    round_sections: list[str] = []
    for item in evidence.get("rounds", []):
        if not isinstance(item, dict):
            continue
        pipeline = item.get("pipeline") or {}
        policy = item.get("policy") or {}
        rule = item.get("rule") or {}
        vqa = item.get("vqa") or {}
        semantics = item.get("outcome_semantics") or {}
        round_sections.append(
            f"""### {item.get('round_id')}: {item.get('candidate_id')}

- pipeline passed: {pipeline.get('passed')}
- failure stage: {pipeline.get('failure_stage')}
- policy metric: {policy.get('metric')}
- policy authority: {policy.get('authority')}
- policy success rate: {policy.get('success_rate')}
- policy seeds: {policy.get('seeds')}
- Rule requested/status/metric: {rule.get('requested')} / {rule.get('status')} / {rule.get('metric')}
- Rule results: {json.dumps(rule.get('results') or [], ensure_ascii=False)}
- VQA required/status/conflict: {vqa.get('required')} / {vqa.get('status')} / {vqa.get('evidence_conflict')}
- VQA observation: {json.dumps(vqa.get('observation'), ensure_ascii=False)}
- outcome semantics: {semantics.get('status')}
- scene change: {json.dumps(item.get('scene_change'), ensure_ascii=False)}
"""
        )
    rounds_markdown = "\n".join(round_sections) or "No RoundEvidence was recorded."
    findings = "\n".join(f"- {item}" for item in feedback["findings"])
    limitations = "\n".join(f"- {item}" for item in feedback["limitations"])
    artifacts = evidence.get("artifacts") or {}
    artifact_lines: list[str] = []
    for name, value in artifacts.items():
        values = value if isinstance(value, list) else [value]
        artifact_lines.extend(
            f"- {name}: {path}" for path in values if path is not None
        )
    artifact_markdown = "\n".join(artifact_lines) or "- none"
    plan = evidence.get("plan") or {}
    return f"""# MEA Evaluation Report

## Identity

- evaluation id: {evidence.get('evaluation_id')}
- user query: {evidence.get('query')}
- executed rounds: {plan.get('executed_rounds')}
- total policy episodes: {evidence.get('total_policy_episodes')}

## Plan state

- planning state: {plan.get('planning_state')}
- maximum rounds: {plan.get('max_rounds')}
- remaining round budget: {plan.get('round_budget_remaining')}

## RoundEvidence

{rounds_markdown}

## Plan Agent answer

{feedback['answer']}

### Findings

{findings}

### Limitations

{limitations}

### Recommended next step

{feedback['recommended_next_step']}

## Artifact index

{artifact_markdown}
"""
