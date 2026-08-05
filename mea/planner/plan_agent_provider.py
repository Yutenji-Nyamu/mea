"""Provider prompt and bounded generation for the Plan Agent."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.planner.open_task_resolver import (
    EXPERIMENTAL_SUCCESS_CHECKER_GUIDANCE,
    query_requires_experimental_checker,
)
from mea.planner.proposal_execution import ProposalExecutionError
from mea.planner.semantic_coverage import (
    SemanticCoverageError,
    build_candidate_intent_alignment,
    validate_evaluation_intent,
)
from mea.providers.json_response import extract_json_response

from .plan_agent_schema import (
    ClaimFirstPlanError,
    PlanAgentError,
    _text,
    build_open_query_planning_lineage,
    validate_open_query_capabilities,
    validate_open_query_evidence,
    validate_open_query_plan_proposal,
)

class PlanAgent:
    """Ask a provider to discover the next sub-aspect from evidence."""

    def __init__(self, provider: Any, *, model: str):
        self.provider = provider
        self.model = _text(model, "model")
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_errors: list[str] = []

    @staticmethod
    def _prompt(
        user_query: str,
        capabilities: Mapping[str, Any],
        evidence_history: Sequence[Mapping[str, Any]],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> str:
        checker_required = query_requires_experimental_checker(user_query)
        example = {
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
                "preserve": ["task identity", "policy checkpoint"],
            },
            "scene_need": {
                "required": True,
                "description": "Scene construction or adaptation needed.",
            },
            "checker_need": {
                "required": checker_required,
                "description": (
                    "The additional experimental success predicate."
                    if checker_required
                    else None
                ),
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
            "CURRENT QUERY CONTRACT: every action=continue Proposal MUST set "
            "checker_need.required=true and describe the directly observable "
            "experimental predicate; false is invalid."
            if checker_required
            else (
                "CURRENT QUERY CONTRACT: checker_need is optional and must be "
                "requested only when the proposed hypothesis needs a new "
                "experimental success predicate."
            )
        )
        return f"""You are the Plan Agent in ManipEvalAgent.
Discover a small set of evaluation sub-aspects online.  There is no predeclared
candidate/template-ID itinerary, success-then-switch script, or fallback route.
The capability card exposes only backend primitives such as scene/checker
generation, telemetry, Rule/VQA Tools, and artifact retrieval.  It is an
execution boundary, not an operation menu or prescribed test order.  Choose
only the single most informative next experiment for the original Query, using
the policy/simulator capabilities and completed evidence below.

{checker_contract}

For action=continue, invent a precise semantic sub_aspect identifier and one
falsifiable hypothesis.  Request a bounded perturbation supported by the
capability cards.  Independently state whether the scene, success checker,
Rule Tool, and VQA Tool must be retrieved, created, or altered.  Do not request
a scene or checker merely because a Tool is needed, and do not couple scene
and checker needs.  A new Tool need may be named even when it is not in an
existing metric/question list.  Avoid repeating a tested perturbation unless
ambiguous evidence requires a more observable version.
Each Rule/VQA need must name one primary scalar or boolean observation for this
round.  Leave independent measurements for a later evidence-conditioned round
instead of bundling them into one Tool request.
For both rule_tool_need and vqa_tool_need, reuse_first MUST always be true,
including when required=false: retrieve-first is the ToolGen method contract,
not a choice to bypass reuse.
State the intentional delta in requested_perturbation.description and
controlled_changes with an explicit operation and concrete value or direction;
put unchanged conditions only in preserve.  When scene_need.required is true,
repeat that same explicit delta in scene_need.description.  Preserve only the
isolation-critical factors supported by a current preservation authority.
Fields merely listed as observable in the simulator card are measurement
capabilities, not preservation authorities.  Use "task identity" and "policy
checkpoint" as the default preserve set; add another condition only when the
current input identifies an authority that can compare it.  Do not add actor
identity, physics timestep, or object-to-target binding merely because those
fields appear in simulator metadata.  When an additional experimental checker
must retain the official goal, add exactly "official core predicate as a
required conjunct" to preserve.  Do not call the extended checker "official
success semantics" or claim full equivalence.
Request a generated checker only when every added relation is directly
observable from the advertised current-state simulator API.  Gripper closure
is not target contact, sequential events are not simultaneous events, and
height is not placement.  A declared actor contact point is a geometric
reference, not a PhysX contact-event identity: do not request that "point i is
physically contacted" unless the runtime explicitly binds collision contacts
to that point ID.  Prefer a directly observable point/TCP distance condition
or an entity-pair contact condition with exactly the semantics the API
supports.  If the exact relation is unavailable, choose a
scene-only experiment with a Rule/VQA observation, or another informative
sub-aspect, instead of asking TaskGen to implement a correlated proxy.
The generated checker is an experimental success criterion, not a way to
encode the predicted policy failure.  It must remain satisfiable by the expert
on the proposed scene.  In particular, do not request an added relation that
the controlled scene change itself makes deterministically false for both the
expert and the policy; any weakness must be established by rollout evidence.
If the original Query explicitly requires an experimental checker for every
generated round, scene-only is not a valid fallback: choose another directly
observable relation or stop with the unsupported limitation stated plainly.
TaskGen may retrieve or generate scene and checker code; ToolGen may retrieve
or generate Rule/VQA Tools.  These artifact primitives do not authorize policy
or controller intervention: do not reduce gripper precision, inject action
noise or latency, or change policy weights.  After successful evidence, refine
to another executable scene/checker/tool concern instead of relabelling a scene
change as an unavailable policy intervention.
{EXPERIMENTAL_SUCCESS_CHECKER_GUIDANCE}

Use success to probe the most consequential remaining uncertainty; use failure
to discriminate a causal failure hypothesis; use ambiguous evidence to improve
observability or isolate the confound.  When completed evidence is non-empty,
the rationale must cite a concrete observed outcome or limitation and explain
why it changed the priority of this sub-aspect.  Do not present a candidate
that was already frozen before seeing that evidence as evidence-conditioned
refinement.  If completed evidence contains a finite scalar, bracket the next
intervention or falsifiable threshold around that observed scale.  Never put a
numeric boundary into a generated success checker unless that exact boundary
comes from the original Query or from completed finite scalar/state evidence.
A successful control alone is not numeric calibration.  After a checker
fixture fails, use its expert-terminal actor/TCP coordinates to derive or
bracket a new observable boundary; do not repeat the same arbitrary threshold
with only an actor or robot-side relabel.  When no grounded boundary exists,
choose an exact discrete relation supported by the current-state API, request
scene-only diagnostic evidence, or report the need unsupported.  For a broad
robustness Query, for example, a successful control
can justify selecting the highest-risk supported perturbation, while a failed
control should redirect to baseline reliability or failure diagnosis.
For a pre-policy TaskGen failure, inspect `bounded_repair_evidence` as well as
the terminal diagnosis.  If an earlier expert fixture gives concrete terminal
state showing that the requested boolean relation is false, do not repeat that
same relation merely because the local repair later violated the Proposal.
Use the simulator state to correct the Proposal itself or switch concern.
Failure example from prior runs: after a successful generated test whose live
scalar shows a comfortable margin, merely increasing the same scene factor is
not a new sub-aspect unless that value brackets a clear boundary.  When the
scalar instead weakens the current hypothesis, switch to the most informative
orthogonal concern that the capability card can execute.  State explicitly
which observed Tool value or outcome caused the switch.  Conversely, do not
manufacture another concern
after the completed evidence already satisfies the Query contract: propose
action=stop so the contract can validate the answer.

Interpret completed evidence by its declared role.  The top-level `outcome`
is the authoritative verdict for the tested hypothesis.  A
`diagnostic_tool_measurements` value is supporting diagnosis only and never
rewrites that verdict.  Preserve the Tool's temporal semantics exactly:
`peak`/`maximum over the rollout` is not a terminal/current-state value.
Failure example: if `outcome="success"` for a terminal checker while a
trajectory-peak distance is large, do not call that a terminal failure or a
failing existential witness.  The correct next step may diagnose the large
transient or choose a stronger scene challenge, but it must retain the
successful terminal-checker result.

Stop only when the completed evidence already answers the original Query.  For
action=stop set sub_aspect and
requested_perturbation to null, all four needs to
required=false/description=null, and express the evidence-supported conclusion
in hypothesis.

ORIGINAL QUERY:
{user_query}
{intent_section}

POLICY AND SIMULATOR CAPABILITIES:
{json.dumps(capabilities, ensure_ascii=False, indent=2)}

COMPLETED ROUND EVIDENCE (chronological; empty means first proposal):
{json.dumps(evidence_history, ensure_ascii=False, indent=2)}

Return strict JSON with exactly these fields:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""

    def propose(
        self,
        user_query: str,
        *,
        capabilities: Mapping[str, Any],
        evidence_history: Sequence[Mapping[str, Any]],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        query = _text(user_query, "user_query")
        trusted_capabilities = validate_open_query_capabilities(capabilities)
        trusted_evidence = validate_open_query_evidence(evidence_history)
        trusted_intent = (
            validate_evaluation_intent(evaluation_intent)
            if evaluation_intent is not None
            else None
        )
        prompt = self._prompt(
            query,
            trusted_capabilities,
            trusted_evidence,
            trusted_intent,
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
                    + "\nReturn one complete corrected JSON object.\n"
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
                if (
                    proposal["action"] == "continue"
                    and query_requires_experimental_checker(query)
                    and proposal["checker_need"]["required"] is not True
                ):
                    raise PlanAgentError(
                        "the original Query explicitly defines experimental "
                        "success semantics, so checker_need.required must be true"
                    )
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
                                "description": (
                                    trusted_intent["requested_change"]
                                    + "\nPreserve: "
                                    + "; ".join(
                                        trusted_intent[
                                            "preserved_conditions"
                                        ]
                                    )
                                )
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
                        raise ClaimFirstPlanError(
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
            raise ClaimFirstPlanError(
                "provider failed two open-Query proposal attempts: "
                + " | ".join(self.last_errors)
            )
        planning_lineage = build_open_query_planning_lineage(
            query,
            trusted_capabilities,
            trusted_evidence,
            trusted_intent,
        )
        return {
            "schema_version": 1,
            "source": "provider_plan_agent_open_query",
            "input_digest": planning_lineage["input_digest"],
            "planning_lineage": planning_lineage,
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
