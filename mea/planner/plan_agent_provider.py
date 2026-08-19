"""Provider prompt and bounded generation for the Plan Agent."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from mea.planner.open_task_resolver import (
    EXPERIMENTAL_SUCCESS_CHECKER_GUIDANCE,
)
from mea.planner.proposal_execution import (
    ProposalExecutionError,
    validate_plan_agent_proposal_execution,
)
from mea.planner.semantic_coverage import (
    SemanticCoverageError,
    build_candidate_intent_alignment,
    validate_evaluation_intent,
)
from mea.providers.json_response import extract_json_response
from mea.task_guide import task_guide_from_capabilities

from .plan_agent_schema import (
    PlanAgentError,
    _text,
    validate_open_query_capabilities,
    validate_open_query_evidence,
    validate_open_query_plan_proposal,
)

class PlanAgent:
    """Ask a provider to discover the next sub-aspect from evidence."""

    def __init__(
        self,
        provider: Any,
        *,
        model: str,
        repo_root: str | Path | None = None,
    ):
        self.provider = provider
        self.model = _text(model, "model")
        self.repo_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_errors: list[str] = []

    @staticmethod
    def _shared_contract() -> str:
        return Path(__file__).with_name("README.Agent.md").read_text(
            encoding="utf-8"
        ).strip()

    @staticmethod
    def _prompt(
        user_query: str,
        capabilities: Mapping[str, Any],
        evidence_history: Sequence[Mapping[str, Any]],
        evaluation_intent: Mapping[str, Any] | None = None,
        task_guide: str = "",
        decision_context: Mapping[str, Any] | None = None,
    ) -> str:
        continue_example = {
            "schema_version": 2,
            "action": "continue",
            "sub_aspect": "semantic.sub_aspect_discovered_now",
            "hypothesis": "A falsifiable statement this one round will test.",
            "requested_perturbation": {
                "description": (
                    "Set one advertised factor from its baseline to one "
                    "bounded diagnostic value."
                ),
                "controlled_changes": ["factor: baseline -> diagnostic value"],
                "preserve": [
                    {
                        "actor": None,
                        "property": "task_identity",
                        "axis": None,
                        "relation": "preserve",
                    },
                    {
                        "actor": None,
                        "property": "policy_checkpoint",
                        "axis": None,
                        "relation": "preserve",
                    },
                ],
            },
            "scene_need": {
                "required": True,
                "description": "Scene construction or adaptation needed.",
            },
            "checker_need": {
                "required": False,
                "description": None,
            },
            "rule_tool_need": {
                "required": True,
                "description": "Numeric or symbolic Rule Tool observable needed.",
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "rationale": "Why this is the most informative next test for the Query.",
            "answer": None,
            "claim_verdict": None,
            "evidence_sufficient": False,
        }
        stop_example = {
            "schema_version": 2,
            "action": "stop",
            "sub_aspect": None,
            "hypothesis": (
                "The bounded completed evidence supports this stop decision."
            ),
            "requested_perturbation": None,
            "scene_need": {"required": False, "description": None},
            "checker_need": {"required": False, "description": None},
            "rule_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "rationale": (
                "Answer the tested scope or explain why further supported "
                "experiments are saturated or unsupported."
            ),
            "answer": "A concise answer to the original Query, or null when inconclusive.",
            "claim_verdict": "supported",
            "evidence_sufficient": True,
        }
        intent_section = ""
        if evaluation_intent is not None:
            intent_section = f"""
FROZEN EVALUATION INTENT (must be implemented directly):
{json.dumps(validate_evaluation_intent(evaluation_intent), ensure_ascii=False, indent=2)}

Every action=continue proposal must directly implement this frozen intent.
Do not silently replace it with a nearby diagnostic proxy.  Preserve its
requested change, preserved conditions, hypothesis, and required observation.
"""
        checker_contract = (
            "checker_need is optional. Request it only when your Proposal "
            "needs success semantics beyond the official task."
        )
        guide_section = (
            "\nBOUND TASK IMPLEMENTATION GUIDE "
            "(source-backed execution knowledge, not a concern menu):\n"
            + task_guide.strip()
            + "\n"
            if task_guide.strip()
            else ""
        )
        last_feedback = ""
        if evidence_history:
            latest = evidence_history[-1]
            last_feedback = (
                "\nLAST ROUND FEEDBACK (read before proposing; do not repeat "
                "the same failed or saturated test without new observability):\n"
                + json.dumps(
                    {
                        key: latest.get(key)
                        for key in (
                            "round_id",
                            "tested_sub_aspect",
                            "tested_hypothesis",
                            "tested_perturbation",
                            "outcome",
                            "evidence_summary",
                            "limitations",
                        )
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
        stop_state = ""
        if decision_context is not None:
            compact_stop_state = {
                key: decision_context.get(key)
                for key in (
                    "budget_remaining",
                    "limitations",
                )
                if decision_context.get(key) is not None
            }
            stop_state = (
                "\nCURRENT RUNTIME LIMITS (not a stop verdict):\n"
                + json.dumps(
                    compact_stop_state,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\nJudge continue or stop from the completed evidence and "
                "original Query. These limits contain no precomputed stop, "
                "sufficiency, or verdict decision.\n"
            )
        shared_contract = PlanAgent._shared_contract()
        return f"""You are the Plan Agent in ManipEvalAgent.
PLAN AGENT CONTRACT:
{shared_contract}

Apply the contract above to choose exactly one next action. The capability card
is an execution boundary, not an experiment menu. For action=continue, fill one
falsifiable sub-aspect and only the independent Task/Tool needs required for
that experiment; keep both reuse_first fields true. State one concrete bounded
delta and preserve only conditions backed by an advertised authority. For
every preserve entry, return exactly {{actor, property, axis, relation}}; use
actor=null for task-wide facts and axis=null unless property is position. For
action=stop, clear the perturbation and all artifact needs. Set answer,
claim_verdict, and evidence_sufficient from completed evidence: an answered
stop needs supported/refuted plus a concise answer; an inconclusive stop uses
answer=null, claim_verdict=inconclusive, evidence_sufficient=false. When completed
evidence is non-empty, rationale must name the outcome, Tool value, abstention,
or failure that changed this decision.

{checker_contract}
{EXPERIMENTAL_SUCCESS_CHECKER_GUIDANCE}

ORIGINAL QUERY:
{user_query}
{intent_section}

POLICY AND SIMULATOR CAPABILITIES:
{json.dumps(capabilities, ensure_ascii=False, indent=2)}
{guide_section}

COMPLETED ROUND EVIDENCE (chronological; empty means first proposal):
{json.dumps(evidence_history, ensure_ascii=False, indent=2)}
{last_feedback}
{stop_state}

Return strict JSON with exactly these fields. Use the first shape for continue
and the second shape for stop; do not combine them.

CONTINUE EXAMPLE:
{json.dumps(continue_example, ensure_ascii=False, indent=2)}

STOP EXAMPLE:
{json.dumps(stop_example, ensure_ascii=False, indent=2)}
"""

    def propose(
        self,
        user_query: str,
        *,
        capabilities: Mapping[str, Any],
        evidence_history: Sequence[Mapping[str, Any]],
        evaluation_intent: Mapping[str, Any] | None = None,
        decision_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        query = _text(user_query, "user_query")
        trusted_capabilities = validate_open_query_capabilities(capabilities)
        trusted_evidence = validate_open_query_evidence(evidence_history)
        trusted_intent = (
            validate_evaluation_intent(evaluation_intent)
            if evaluation_intent is not None
            else None
        )
        task_guide = task_guide_from_capabilities(
            self.repo_root,
            trusted_capabilities,
        )
        prompt = self._prompt(
            query,
            trusted_capabilities,
            trusted_evidence,
            trusted_intent,
            task_guide,
            decision_context,
        )
        self.last_prompt = prompt
        self.last_responses = []
        self.last_errors = []

        proposal: dict[str, Any] | None = None
        for _attempt in range(2):
            attempt_prompt = prompt
            if self.last_errors:
                attempt_prompt += (
                    "\nPREVIOUS VALIDATION ERROR:\n"
                    + self.last_errors[-1]
                    + (
                        "\nPREVIOUS COMPLETE PROPOSAL JSON:\n"
                        + self.last_responses[-1]
                        if self.last_responses
                        else ""
                    )
                    + "\nCorrect the field named by the error and return one "
                    "complete JSON object. Preserve valid fields unless the "
                    "error requires changing them. If action=stop, copy the "
                    "STOP EXAMPLE shape exactly: sub_aspect and "
                    "requested_perturbation are null, and every artifact need "
                    "has required=false and description=null.\n"
                )
            try:
                response = self.provider.text(
                    attempt_prompt,
                    model=self.model,
                    system="Return only strict OpenQueryPlanProposal JSON.",
                    max_tokens=900,
                    temperature=0.0,
                )
                self.last_responses.append(response)
                proposal = validate_open_query_plan_proposal(
                    extract_json_response(response),
                    has_evidence=bool(trusted_evidence),
                )
                try:
                    proposal = validate_plan_agent_proposal_execution(
                        proposal,
                        capabilities=trusted_capabilities,
                    )
                except ProposalExecutionError as exc:
                    raise PlanAgentError(str(exc)) from exc
                if trusted_intent is not None and proposal["action"] == "continue":
                    scene_need = proposal["scene_need"]
                    checker_need = proposal["checker_need"]
                    rule_tool_need = proposal["rule_tool_need"]
                    vqa_tool_need = proposal["vqa_tool_need"]
                    observation_contract = (
                        trusted_intent["required_observation"]
                        + "\n"
                        + trusted_intent["hypothesis"]
                    )
                    alignment = build_candidate_intent_alignment(
                        trusted_intent,
                        semantic_concern=(
                            trusted_intent["original_concern"]
                            + "\n"
                            + trusted_intent["hypothesis"]
                        ),
                        scene_need=(
                            {
                                "description": trusted_intent[
                                    "requested_change"
                                ]
                            }
                            if scene_need["required"]
                            else None
                        ),
                        checker_need=(
                            {"description": observation_contract}
                            if checker_need["required"]
                            else None
                        ),
                        rule_tool_need=(
                            {"description": observation_contract}
                            if rule_tool_need["required"]
                            else None
                        ),
                        vqa_tool_need=(
                            {"description": observation_contract}
                            if vqa_tool_need["required"]
                            else None
                        ),
                    )
                    if alignment["relationship"] != "direct":
                        raise PlanAgentError(
                            "proposal silently pivots to a diagnostic proxy; "
                            "it must directly implement the frozen "
                            "EvaluationIntent. Unmatched intent fields: "
                            + ", ".join(
                                alignment["unmatched_intent_fields"]
                            )
                        )
                break
            except SemanticCoverageError as exc:
                proposal = None
                self.last_errors.append(
                    f"{type(exc).__name__}: {exc}"
                )
            except Exception as exc:
                proposal = None
                self.last_errors.append(f"{type(exc).__name__}: {exc}")
        if proposal is None:
            raise PlanAgentError(
                "provider failed two open-Query proposal attempts: "
                + " | ".join(self.last_errors)
            )
        return {
            "schema_version": 1,
            "source": "provider_plan_agent_open_query",
            "proposal": proposal,
            "provider": {
                "model_requested": self.model,
                "called": True,
                "attempt_count": len(self.last_responses),
                "errors": list(self.last_errors),
                "last_metadata": deepcopy(
                    dict(getattr(self.provider, "last_metadata", {}))
                ),
            },
        }


__all__ = ["PlanAgent"]
