"""Provider-authored, catalog-bounded next-step proposals for Plan(psi, Y_1:t).

The first MEA prototype selected every sub-aspect before the first rollout and
then asked a model to explain a deterministic transition.  The paper instead
defines a small dynamic set of sub-aspects discovered during evaluation.  This
module gives the model that bounded decision right while keeping task,
checkpoint, executable templates, and the round budget runtime-owned.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

from mea.taskgen import extract_json_response


class AdaptiveStepError(ValueError):
    """Raised when a PlanStepProposal leaves its trusted navigation options."""


_STEP_KEYS = {
    "schema_version",
    "action",
    "aspect_id",
    "template_id",
    "rationale",
    "answered_query",
}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdaptiveStepError(f"{field} must be a non-empty string")
    return value.strip()


def _candidate_pairs(options: Mapping[str, Any], action: str) -> set[tuple[str, str]]:
    candidates = options.get("available_steps", {}).get(action)
    if not isinstance(candidates, list):
        raise AdaptiveStepError(f"navigation options omit {action!r} candidates")
    result: set[tuple[str, str]] = set()
    for item in candidates:
        if not isinstance(item, Mapping):
            raise AdaptiveStepError("navigation candidate must be an object")
        aspect_id = item.get("aspect_id")
        templates = item.get("template_ids")
        if not isinstance(aspect_id, str) or not isinstance(templates, list):
            raise AdaptiveStepError("navigation candidate is malformed")
        result.update((aspect_id, str(template_id)) for template_id in templates)
    return result


def validate_plan_step_proposal(
    value: Mapping[str, Any], navigation_options: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one provider choice against evidence-derived legal options."""

    if not isinstance(value, Mapping) or set(value) != _STEP_KEYS:
        raise AdaptiveStepError(
            f"PlanStepProposal fields must be exactly {sorted(_STEP_KEYS)}"
        )
    proposal = deepcopy(dict(value))
    if proposal.get("schema_version") != 1:
        raise AdaptiveStepError("PlanStepProposal.schema_version must be 1")
    action = proposal.get("action")
    if action not in {"propose", "refine", "stop"}:
        raise AdaptiveStepError("PlanStepProposal.action must be propose, refine, or stop")
    answered = proposal.get("answered_query")
    if not isinstance(answered, bool):
        raise AdaptiveStepError("PlanStepProposal.answered_query must be bool")
    proposal["rationale"] = _text(proposal.get("rationale"), "rationale")

    if action == "stop":
        if proposal.get("aspect_id") is not None or proposal.get("template_id") is not None:
            raise AdaptiveStepError("stop must not select an aspect or template")
        if navigation_options.get("available_steps", {}).get("stop") is not True:
            raise AdaptiveStepError("current evidence does not allow stop")
        stop_requires_answer = bool(navigation_options.get("stop_requires_answered_query"))
        if stop_requires_answer and not answered:
            raise AdaptiveStepError("voluntary stop requires answered_query=true")
        return proposal

    if answered:
        raise AdaptiveStepError("a continuing PlanStepProposal cannot answer the query")
    aspect_id = _text(proposal.get("aspect_id"), "aspect_id")
    template_id = _text(proposal.get("template_id"), "template_id")
    if (aspect_id, template_id) not in _candidate_pairs(navigation_options, action):
        raise AdaptiveStepError(
            f"{action} target {(aspect_id, template_id)!r} is outside navigation options"
        )
    proposal.update({"aspect_id": aspect_id, "template_id": template_id})
    return proposal


class AdaptivePlanStepAgent:
    """Ask a model for the next bounded sub-aspect after real typed evidence."""

    def __init__(self, provider: Any, *, model: str):
        self.provider = provider
        self.model = str(model)
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_errors: list[str] = []

    @staticmethod
    def _prompt(
        user_query: str,
        navigation_options: Mapping[str, Any],
        planning_context: Mapping[str, Any],
    ) -> str:
        example = deepcopy(dict(navigation_options["fallback_step"]))
        example["rationale"] = (
            "Explain how the Rule result, VQA result, and original query justify "
            "this next evaluation step."
        )
        return f"""You are the Plan Agent in a bounded manipulation-policy evaluation.
After every rollout you may discover a new supported sub-aspect, refine the
current sub-aspect with a counterfactual template, or stop when the original
query is answered.  The runtime has already frozen one RoboTwin task, one ACT
checkpoint, the executable capability catalog, and the remaining round budget.
Do not invent a task, checkpoint, aspect, template, metric, seed, or path.

Use all typed evidence, including the Rule metric, policy success, VQA answer,
and evidence-conflict flag.  Treat evidence_packet.policy.success_rate as the
evaluated policy outcome.  evidence_packet.rule.aggregate_status only says
whether Rule aggregation produced valid evidence; "passed" there does not mean
the manipulation policy succeeded.  A pipeline failure is not policy failure.
Choose:
- propose: a new aspect from available_steps.propose;
- refine: a same-aspect counterfactual from available_steps.refine;
- stop: only when available_steps.stop is true.  If stop is voluntary, set
  answered_query=true; otherwise use false for a forced budget/pipeline stop.

ORIGINAL QUERY:
{user_query}

TRUSTED PLANNING CONTEXT:
{json.dumps(planning_context, ensure_ascii=False, indent=2)}

EVIDENCE-DERIVED NAVIGATION OPTIONS:
{json.dumps(navigation_options, ensure_ascii=False, indent=2)}

Return strict JSON with exactly this shape:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""

    def propose(
        self,
        user_query: str,
        *,
        navigation_options: Mapping[str, Any],
        planning_context: Mapping[str, Any],
    ) -> dict[str, Any]:
        query = _text(user_query, "user_query")
        prompt = self._prompt(query, navigation_options, planning_context)
        self.last_prompt = prompt
        self.last_responses = []
        self.last_errors = []
        proposal = None
        for _attempt in range(2):
            attempt_prompt = prompt
            if self.last_errors:
                attempt_prompt += (
                    "\nPREVIOUS VALIDATION ERROR:\n"
                    + self.last_errors[-1]
                    + "\nReturn one complete corrected JSON object.\n"
                )
            try:
                response = self.provider.text(
                    attempt_prompt,
                    model=self.model,
                    system="Return only strict PlanStepProposal JSON.",
                    max_tokens=500,
                    temperature=0.0,
                )
                self.last_responses.append(response)
                proposal = validate_plan_step_proposal(
                    extract_json_response(response), navigation_options
                )
                break
            except Exception as exc:
                self.last_errors.append(f"{type(exc).__name__}: {exc}")
        source = "provider"
        if proposal is None:
            proposal = validate_plan_step_proposal(
                navigation_options["fallback_step"], navigation_options
            )
            source = "deterministic_fallback_after_provider_failure"
        return {
            "schema_version": 1,
            "source": source,
            "proposal": proposal,
            "navigation_options": deepcopy(dict(navigation_options)),
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


__all__ = [
    "AdaptivePlanStepAgent",
    "AdaptiveStepError",
    "validate_plan_step_proposal",
]
