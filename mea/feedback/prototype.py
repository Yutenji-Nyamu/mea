"""Evidence-grounded feedback generation and unified report rendering."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mea.taskgen import extract_json_response


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
    policy_success = (evidence or {}).get("observations", {}).get(
        "policy_success"
    )
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
    policy_success = evidence.get("observations", {}).get("policy_success")
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
    return feedback


def _feedback_prompt(repo_root: Path, evidence: dict[str, Any]) -> str:
    instructions = (repo_root / "mea/feedback/README.Agent.md").read_text(
        encoding="utf-8"
    )
    return f"""你是 MEA 的最终 Feedback Agent。请基于证据回答用户，不要补充未经测试的结论。

AGENT RULES:
{instructions}

EVIDENCE BUNDLE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

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
            policy_success = evidence.get("observations", {}).get(
                "policy_success"
            )
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
        feedback["provider_metadata"] = dict(
            getattr(self.provider, "last_metadata", {})
        )
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

    if evidence.get("rounds"):
        observations = evidence["observations"]
        round_sections = []
        for item in evidence["rounds"]:
            round_observations = item["observations"]
            positions = round_observations.get("position_samples", [])
            position_lines = "\n".join(
                "  - episode {episode_index}, seed {seed}: {block_position}".format(
                    **sample
                )
                for sample in positions
            ) or "  - none"
            selected = ", ".join(
                f"`{name}`"
                for name in item.get("task_retrieval", {}).get(
                    "selected_tasks", []
                )
            ) or "none (reuse route)"
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
                )
                tool_lines.append(
                    "  - {policy} seed {seed}: {results}".format(
                        policy=episode.get("policy_name"),
                        seed=episode.get("seed"),
                        results=result_text or "none",
                    )
                )
            trusted_tool_lines = "\n".join(tool_lines) or "  - none"
            round_sections.append(
                f"""### {item['round_id']}: `{item['sub_aspect']}`

- route: `{item['route']}`
- instruction: {item['task_instruction']}
- seeds: `{item['seeds']}`
- episodes: `{item['num_episodes']}`
- selected retrieval tasks: {selected}
- observed color: `{round_observations.get('observed_color')}`
- expert solvable: `{round_observations.get('expert_solvable')}`
- ACT pipeline status: `{round_observations.get('act_pipeline_status')}`
- policy success: `{round_observations.get('policy_success')}`
- pipeline passed: `{round_observations.get('pipeline_passed')}`
- position samples:
{position_lines}
- trusted Tool results:
{trusted_tool_lines}
"""
            )
        rounds_markdown = "\n".join(round_sections)
        findings = "\n".join(f"- {item}" for item in feedback["findings"])
        limitations = "\n".join(f"- {item}" for item in feedback["limitations"])
        artifacts = evidence["artifacts"]
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

## Round evidence

{rounds_markdown}

## Aggregate observations

- scene alignment: `{observations['scene_alignment']}`
- observed color by round: `{observations['observed_color_by_round']}`
- expert solvable: `{observations['expert_solvable']}`
- ACT pipeline status: `{observations['act_pipeline_status']}`
- weighted policy success: `{observations['policy_success']}`
- policy success by round: `{observations['policy_success_by_round']}`
- position varied: `{observations['position_varied']}`
- position metrics: `{observations['position_metrics']}`
- pipeline passed: `{observations['pipeline_passed']}`

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
- Round 2 decision: `{artifacts['round_2_decision']}`
- machine-readable summary: `{artifacts['summary']}`
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
