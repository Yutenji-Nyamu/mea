"""Validation and rendering for the Plan Agent's final answer."""

from __future__ import annotations

import json
from typing import Any, Mapping


class FinalAnswerError(RuntimeError):
    """Raised when a Plan Agent answer has an invalid output shape."""


def require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalAnswerError(f"{field} must be a non-empty string")
    return value.strip()


def _require_text_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise FinalAnswerError(f"{field} must be a non-empty string list")
    return [require_text(item, f"{field}[]") for item in value]


def validate_final_answer(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the presentation shape; Plan owns the conclusion."""

    if not isinstance(value, Mapping):
        raise FinalAnswerError("final answer must be an object")
    return {
        "answer": require_text(value.get("answer"), "answer"),
        "evaluation_scope": require_text(
            value.get("evaluation_scope"), "evaluation_scope"
        ),
        "findings": _require_text_list(value.get("findings"), "findings"),
        "limitations": _require_text_list(
            value.get("limitations"), "limitations"
        ),
        "recommended_next_step": require_text(
            value.get("recommended_next_step"), "recommended_next_step"
        ),
    }


def answer_markdown(answer: Mapping[str, Any]) -> str:
    findings = "\n".join(f"- {item}" for item in answer["findings"])
    limitations = "\n".join(
        f"- {item}" for item in answer["limitations"]
    )
    return (
        "# Plan Agent answer\n\n"
        f"{answer['answer']}\n\n"
        "## Evaluation scope\n\n"
        f"{answer['evaluation_scope']}\n\n"
        "## Findings\n\n"
        f"{findings}\n\n"
        "## Limitations\n\n"
        f"{limitations}\n\n"
        "## Recommended next step\n\n"
        f"{answer['recommended_next_step']}\n"
    )


def render_evaluation_report(
    evidence: Mapping[str, Any],
    answer: Mapping[str, Any],
) -> str:
    """Render one compact human view of the typed final evidence bundle."""

    round_sections: list[str] = []
    for item in evidence.get("rounds", []):
        if not isinstance(item, Mapping):
            continue
        policy = item.get("policy") or {}
        rule = item.get("rule") or {}
        vqa = item.get("vqa") or {}
        semantics = item.get("outcome_semantics") or {}
        round_sections.append(
            f"""### {item.get('round_id')}: {item.get('candidate_id')}

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
    rounds_markdown = (
        "\n".join(round_sections)
        or "No RoundEvidence was recorded."
    )
    findings = "\n".join(f"- {item}" for item in answer["findings"])
    limitations = "\n".join(
        f"- {item}" for item in answer["limitations"]
    )
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

{answer['answer']}

### Findings

{findings}

### Limitations

{limitations}

### Recommended next step

{answer['recommended_next_step']}

## Artifact index

{artifact_markdown}
"""


__all__ = [
    "FinalAnswerError",
    "answer_markdown",
    "render_evaluation_report",
    "require_text",
    "validate_final_answer",
]
