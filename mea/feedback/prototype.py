"""Evidence-grounded feedback generation and unified report rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mea.taskgen import extract_json_response


class FeedbackAgentError(RuntimeError):
    """Raised when final feedback violates the structured output contract."""


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeedbackAgentError(f"{field} 必须是非空字符串")
    return value.strip()


def _require_text_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise FeedbackAgentError(f"{field} 必须是非空字符串 list")
    return [_require_text(item, f"{field}[]") for item in value]


def validate_feedback(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FeedbackAgentError("Feedback 必须是 JSON object")
    return {
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
        response = self.provider.text(
            prompt,
            model=self.model,
            system="Use only the evidence and return strict Feedback JSON.",
            max_tokens=1200,
            temperature=0.0,
        )
        (output_dir / "response.txt").write_text(response + "\n", encoding="utf-8")
        feedback = validate_feedback(extract_json_response(response))
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
    """Render the one-file human entry point for a completed evaluation."""

    retrieval = evidence["task_retrieval"]
    observations = evidence["observations"]
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
