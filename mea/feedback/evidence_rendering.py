"""Markdown rendering for the compact MEA evidence report."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def relative_link(path: Path, report_path: Path) -> str:
    return Path(os.path.relpath(path, report_path.parent)).as_posix()


def quote(value: Any) -> list[str]:
    text = str(value or "N/A").strip() or "N/A"
    return [f"> {line}" if line else ">" for line in text.splitlines()]


def render_report_header(
    *,
    evaluation_id: Any,
    evaluation_name: str,
    query: Any,
    target: Mapping[str, Any],
    session: Mapping[str, Any],
    plan: Mapping[str, Any],
    rounds: list[dict[str, Any]],
    episode_counts: list[int],
    semantic_dir: Path,
    report_path: Path,
) -> list[str]:
    lines = [
        f"# MEA method evidence: {evaluation_id or evaluation_name}",
        "",
        "> Compact, movable view of one real method run. Complete raw telemetry "
        "and Aggregate payloads remain in the server evaluation directory.",
        "",
        "## 1. Query and execution scope",
        "",
        *quote(query),
        "",
        f"- Task: `{target.get('task_name')}`",
        f"- Policy: `{(target.get('policy') or {}).get('name')}`",
        f"- Checkpoint: `{(target.get('checkpoint') or {}).get('checkpoint_id')}`",
        "- Round budget / evidence episodes per round: "
        f"`{session.get('round_budget') or plan.get('max_rounds')}` / "
        f"`{episode_counts}`",
        "",
        "## 2. Paper-level data flow",
        "",
        "```mermaid",
        "flowchart LR",
        '  Q["Open Query"] --> P["Plan Agent / sub-aspect"]',
        '  P --> T["TaskGen: reuse or generate"]',
        '  T --> I["Render / visual reflection"]',
        '  I --> E["Policy rollout"]',
        '  E --> V["Rule Tool + dynamic VQA"]',
        '  V --> A["Aggregate"]',
        '  A -->|"evidence"| P',
        '  A --> R["Final answer"]',
        "```",
        "",
        "## 3. Plan Agent trace",
        "",
        f"- Goal: {plan.get('evaluation_goal')}",
        f"- Initial round: `{rounds[0].get('round_id') if rounds else None}`",
        f"- Final planning state: `{plan.get('planning_state')}`",
    ]
    interpretation_prompt = semantic_dir / "plan/query_interpretation_prompt.md"
    interpretation_responses = sorted(
        (semantic_dir / "plan").glob("query_interpretation_response_*.txt")
    )
    if interpretation_prompt.is_file() and interpretation_responses:
        response_links = " / ".join(
            f"[response {index}]({relative_link(path, report_path)})"
            for index, path in enumerate(interpretation_responses, start=1)
        )
        lines.extend(
            [
                "- Query interpretation trace: "
                f"[prompt]({relative_link(interpretation_prompt, report_path)})"
                f" / {response_links}",
                "",
            ]
        )
    return lines


def render_round(
    *,
    index: int,
    round_id: str,
    round_plan: Mapping[str, Any],
    target: Mapping[str, Any],
    compact: Mapping[str, Any],
    proposal_copy: Path | None,
    taskgen_destination: Path,
    code_copy: Path | None,
    variant_copy: Path | None,
    scene_copy: Path | None,
    video_copy: Path | None,
    video_source: Path | None,
    tool_code_copy: Path | None,
    montage_copy: Path | None,
    report_path: Path,
) -> list[str]:
    lines = [
        f"## 4.{index}. `{round_id}` — {compact['aspect_id']}",
        "",
        "### Plan → TaskGen",
        "",
        f"- Task: `{round_plan.get('task_name') or target.get('task_name')}`",
        f"- Route/materialization: `{compact['taskgen_route']}` / "
        f"`{compact['taskgen_kind'] or 'not recorded'}`",
        "- Gates: "
        + json.dumps(compact["taskgen_gates"], ensure_ascii=False),
    ]
    if proposal_copy:
        lines.append(
            f"- Proposal: [{proposal_copy.name}]"
            f"({relative_link(proposal_copy, report_path)})"
        )
    taskgen_prompt = taskgen_destination / "generation/code_prompt.md"
    taskgen_response = taskgen_destination / "generation/provider_response.txt"
    if taskgen_prompt.is_file() and taskgen_response.is_file():
        lines.append(
            "- Provider trace: "
            f"[prompt]({relative_link(taskgen_prompt, report_path)}) / "
            f"[response]({relative_link(taskgen_response, report_path)})"
        )
    if code_copy and code_copy.stat().st_size > 3:
        lines.append(
            f"- Task artifact: [{code_copy.name}]"
            f"({relative_link(code_copy, report_path)})"
        )
    elif code_copy:
        lines.append(
            f"- Official passthrough marker: [{code_copy.name}]"
            f"({relative_link(code_copy, report_path)})"
        )
    else:
        lines.append("- Generated/reused source: N/A (artifact was not present)")
    if variant_copy:
        lines.append(
            f"- VariantSpec: [{variant_copy.name}]"
            f"({relative_link(variant_copy, report_path)})"
        )
    if scene_copy:
        lines.extend(
            [
                "",
                "### Render / scene check",
                "",
                f"![{round_id} initial scene]"
                f"({relative_link(scene_copy, report_path)})",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "### Render / scene check",
                "",
                "N/A - no real scene image was found.",
            ]
        )
    lines.extend(
        [
            "",
            "### Rollout",
            "",
            f"- Backend/seeds: `{compact['execution_backend']}` / "
            f"`{compact['seeds']}`",
            f"- Policy success: `{compact['policy_success']}`",
        ]
    )
    if video_copy:
        link = relative_link(video_copy, report_path)
        lines.extend(
            [
                "",
                f"[Open policy video]({link})",
                "",
                f'<video src="{link}" controls width="720"></video>',
            ]
        )
    elif video_source is not None:
        lines.append(
            "\nVideo exists in the raw run but exceeded the publish size limit."
        )
    else:
        lines.append("\nN/A - no evaluated-policy video was found.")
    lines.extend(
        [
            "",
            "### Tool / VQA",
            "",
            f"- Tool: `{compact['tool_route']}` → `{compact['tool_metric']}`",
            "- Values: "
            + json.dumps(compact["tool_rows"], ensure_ascii=False),
        ]
    )
    if tool_code_copy:
        lines.append(
            "- [Open generated/reused Tool source]"
            f"({relative_link(tool_code_copy, report_path)})"
        )
    lines.append(
        f"- VQA status: `{compact['vqa'].get('status')}`; "
        f"conflict: `{compact['vqa'].get('evidence_conflict')}`"
    )
    if montage_copy:
        lines.extend(
            [
                "",
                f"![{round_id} VQA keyframes]"
                f"({relative_link(montage_copy, report_path)})",
            ]
        )
    lines.extend(
        [
            "",
            "### Aggregate -> next decision",
            "",
            "- Aggregate: "
            + json.dumps(compact["aggregate"], ensure_ascii=False),
            "- Decision: "
            + json.dumps(compact["next_decision"], ensure_ascii=False),
            "",
        ]
    )
    return lines


def render_final(
    *,
    final_payload: Mapping[str, Any],
    alignment_reaudit_summary: Mapping[str, Any] | None,
    run_summary_path: Path,
    repair_result: Mapping[str, Any] | None,
    reuse_summary: Mapping[str, Any] | None,
    report_path: Path,
    evaluation: Path,
    root: Path,
) -> list[str]:
    lines = [
        "## 5. Final answer to the original Query",
        "",
        *quote(final_payload.get("answer")),
        "",
        *[
            f"- Finding: {item}"
            for item in final_payload.get("findings") or []
        ],
        f"- Next: {final_payload.get('recommended_next_step')}",
        *[
            f"- Limitation: {item}"
            for item in final_payload.get("limitations") or []
        ],
        *(
            [
                "",
                "### Post-run 0-ACT semantic alignment re-audit",
                "",
                json.dumps(alignment_reaudit_summary, ensure_ascii=False),
                "The source evaluation and Answer remain immutable; this "
                "cached recomputation adds no policy-performance evidence.",
            ]
            if alignment_reaudit_summary is not None
            else []
        ),
        "## 6. Boundaries",
        "",
        "- Task, policy, Rule, and VQA facts are reported separately.",
        "- Expert evidence, when present, is a solvability/instrumentation "
        "gate, not evaluated-policy performance.",
        "- Few-shot N=1 rounds demonstrate method wiring, not benchmark-level "
        "generalization.",
        "- Missing artifacts are shown as N/A; this report never substitutes "
        "proxy images or invented values.",
        "",
        "## 7. Artifact index",
        "",
        f"- [Compact machine summary]({relative_link(run_summary_path, report_path)})",
        "- [Published artifact index with paths and byte sizes]"
        "(artifact_index.json)",
        "- Complete raw source remains server-side at "
        f"`{str(evaluation.relative_to(root)).replace(chr(92), '/')}`.",
    ]
    if repair_result is not None:
        lines.extend(
            [
                "",
                "### Completed-round Tool reuse audit",
                "",
                json.dumps(reuse_summary, ensure_ascii=False),
                "This independent follow-up Query reuses completed policy "
                "telemetry and starts no simulator or policy rollout. It "
                "proves exact reuse within this evaluation's registry, not "
                "cross-evaluation reuse.",
                "",
            ]
        )
    return lines


__all__ = [
    "quote",
    "relative_link",
    "render_final",
    "render_report_header",
    "render_round",
]
