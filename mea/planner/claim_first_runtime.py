"""Runtime bridge for Plan Agent open-Query planning.

``PlanAgent`` deliberately emits a semantic experiment rather
than an executable catalog step.  This module connects that semantic proposal
to the existing bounded ACT runtime without letting a language model invent
execution details or decide when evidence is sufficient.

The bridge has four explicit responsibilities:

* run one unchanged official-scene control before property attribution
  (diagnostic-only protocols may explicitly opt out);
* derive OpenQueryEvidence and finite-domain candidate evidence directly from
  the runtime-owned EvidencePacket and lightweight artifact paths;
* apply the query-sufficiency contract before accepting a model-authored stop;
* reuse a matching trusted template when one exists, otherwise materialize a
  Query-derived ``Proposal`` for TaskGen/ToolGen.

The legacy finite catalog path remains supported.  Dynamic candidates use an
open-world sufficiency contract and never make the hidden executable catalog
part of the model prompt.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from mea.aspects import public_aspect_ontology
from mea.capability_adapter import (
    resolve_task_retrieval_index,
)

from .claim_first import (
    ClaimFirstPlanError,
    validate_open_query_capabilities,
    validate_open_query_evidence,
    validate_open_query_plan_proposal,
    validate_open_query_proposal_lineage,
)
from .evidence_policy import build_evidence_packet, validate_evidence_packet
from .experiment_candidate import (
    build_experiment_candidate,
    validate_experiment_candidate,
)
from .open_task_resolver import (
    validate_free_concern,
    validate_free_concern_experiment_needs,
)
from .open_world_session import _FrozenExecutionTransport
from .policy_task_binding import (
    PolicyTaskBindingError,
    policy_task_binding_from_target,
)
from .semantic_coverage import (
    SemanticCoverageError,
    build_evaluation_intent,
    validate_evaluation_intent,
)
from .query_contract import (
    assess_query_sufficiency,
    build_query_sufficiency_contract,
    extend_query_candidate_universe,
    infer_claim_type,
    query_is_official_only,
    validate_query_sufficiency_contract,
)


class PlanAgentSessionError(ValueError):
    """Raised when semantic planning cannot be bound to trusted evidence."""


# Compatibility name for existing callers and historical exception handlers.
ClaimFirstRuntimeError = PlanAgentSessionError


_SEMANTIC_STOPWORDS = {
    "a",
    "act",
    "an",
    "and",
    "bell",
    "can",
    "click",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "under",
    "with",
    "while",
    "without",
    "semantic",
    "sub",
    "aspect",
    "test",
    "task",
    "policy",
}

_ASPECT_GENERIC_TOKENS = {
    "object",
    "scene",
    "performance",
    "robustness",
}


def build_initial_semantic_proposal_bundle(
    *,
    user_query: str,
    concern: Mapping[str, Any],
    experiment_needs: Mapping[str, Any],
    evaluation_intent: Mapping[str, Any],
    provider_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize a Query-only candidate before any runtime evidence exists.

    Query interpretation already chose the first sub-aspect, hypothesis,
    perturbation, observation, and independent Task/Tool needs before catalog
    retrieval.  This adapter preserves that Query-authored seed for a no-control
    first round or a legacy protocol.  It is explicitly marked
    ``pre_evidence_query_candidate`` and must never be reported as a Fig. 5
    evidence-conditioned refinement.
    """

    query = _nonempty_text(user_query, "user_query")
    trusted_concern = validate_free_concern(
        concern,
        expected_query=query,
    )
    trusted_needs = validate_free_concern_experiment_needs(
        experiment_needs
    )
    intent = validate_evaluation_intent(evaluation_intent)
    if intent["source_query"] != query:
        raise ClaimFirstRuntimeError(
            "EvaluationIntent source_query differs from the original Query"
        )
    slug = re.sub(
        r"[^a-z0-9]+",
        ".",
        trusted_concern["sub_aspect"].casefold(),
    ).strip(".")
    if not slug:
        slug = intent["intent_id"].removeprefix("intent.")
    proposal = validate_open_query_plan_proposal(
        {
            "schema_version": 2,
            "action": "continue",
            "sub_aspect": f"query_interpretation.{slug[:96]}",
            "hypothesis": intent["hypothesis"],
            "requested_perturbation": {
                "description": intent["requested_change"],
                "controlled_changes": [intent["requested_change"]],
                "preserve": list(intent["preserved_conditions"]),
            },
            **deepcopy(trusted_needs),
            "rationale": (
                "Directly execute the catalog-free first concern selected for "
                "the original Query; no second Planner may replace it before "
                "the control evidence is observed."
            ),
        },
        has_evidence=False,
    )
    return {
        "schema_version": 1,
        "source": "provider_plan_agent_direct_materialization",
        "input_intent_id": intent["intent_id"],
        "planning_lineage": {
            "schema_version": 1,
            "decision_kind": "pre_evidence_query_candidate",
            "evidence_conditioned": False,
            "completed_round_ids": [],
            "completed_round_count": 0,
            "input_digest": None,
        },
        "proposal": proposal,
        "provider": deepcopy(dict(provider_record or {})),
    }


def build_dynamic_experiment_candidate(
    *,
    user_query: str,
    task_name: str,
    proposal: Mapping[str, Any],
    evaluation_intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one semantic proposal without rewriting its first-candidate intent.

    The provider owns the four independent generation/tool requirements.  For
    the first Query-derived candidate, the Query interpretation owns
    the exact change, hypothesis, preserved conditions, and observation text.
    Later evidence-driven pivots receive a fresh per-candidate intent derived
    directly from their proposal, so every round carries the same traceable
    experiment contract.
    """

    trusted = validate_open_query_plan_proposal(proposal, has_evidence=True)
    if trusted["action"] != "continue":
        raise ClaimFirstRuntimeError(
            "only a continue decision can become an executable Proposal"
        )
    perturbation = trusted["requested_perturbation"]
    scene_need = trusted["scene_need"]
    checker_need = trusted["checker_need"]
    rule_tool_need = trusted["rule_tool_need"]
    vqa_tool_need = trusted["vqa_tool_need"]
    observation_parts = [
        str(need.get("description") or "").strip()
        for need in (checker_need, rule_tool_need, vqa_tool_need)
        if need["required"]
    ]
    proposal_intent = build_evaluation_intent(
        source_query=user_query,
        original_concern=trusted["sub_aspect"],
        hypothesis=trusted["hypothesis"],
        requested_change=perturbation["description"],
        preserved_conditions=perturbation["preserve"],
        required_observation=(
            " ".join(part for part in observation_parts if part)
            or "Observe the policy outcome needed to decide: "
            + trusted["hypothesis"]
        ),
    )
    intent = (
        validate_evaluation_intent(evaluation_intent)
        if evaluation_intent is not None
        else proposal_intent
    )
    exact_hypothesis = intent["hypothesis"]
    semantic_concern = (
        f"{intent['original_concern']}: {exact_hypothesis}"
    )
    scene_description = (
        str(scene_need["description"]).strip()
        if scene_need["required"]
        else ""
    )
    missing_preserved = [
        condition
        for condition in intent["preserved_conditions"]
        if condition.casefold() not in scene_description.casefold()
    ]
    if missing_preserved:
        scene_description += (
            " Preserve unchanged: " + "; ".join(missing_preserved) + "."
        )
    return build_experiment_candidate(
        source_query=_nonempty_text(user_query, "user_query"),
        base_task=_nonempty_text(task_name, "task_name"),
        semantic_concern=semantic_concern,
        scene_need=(
            {
                "kind": "adapt",
                "description": scene_description,
                "reuse_first": True,
            }
            if scene_need["required"]
            else None
        ),
        checker_need=(
            {
                "kind": "generate",
                "description": str(checker_need["description"]).strip(),
                "reuse_first": True,
            }
            if checker_need["required"]
            else None
        ),
        rule_tool_need=(
            {
                "kind": (
                    "reuse"
                    if query_is_official_only(user_query)
                    else "measure"
                ),
                "description": str(rule_tool_need["description"]).strip(),
                "reuse_first": True,
            }
            if rule_tool_need["required"]
            else None
        ),
        vqa_tool_need=(
            {
                "kind": "vqa",
                "description": str(vqa_tool_need["description"]).strip(),
                "reuse_first": True,
            }
            if vqa_tool_need["required"]
            else None
        ),
        evaluation_intent=intent,
    )

_CONCERN_GENERIC_TOKENS = _ASPECT_GENERIC_TOKENS | {
    "ability",
    "appropriate",
    "change",
    "correct",
    "evaluation",
    "expose",
    "general",
    "generalization",
    "measure",
    "may",
    "observe",
    "property",
    "reliability",
    "robustness",
    "success",
    "target",
    "variation",
    "weakness",
}

_SEMANTIC_ALIASES = {
    "distractor": (
        "distractor",
        "look alike",
        "look-alike",
        "lookalike",
        "similar object",
        "similar objects",
        "visually similar",
        "wrong target",
        "confusion",
        "干扰",
        "相似",
        "类似",
        "混淆",
    ),
    "clutter": ("clutter", "cluttered", "tabletop objects", "杂物", "杂乱"),
    "instance": ("instance", "identity", "appearance variant", "实例", "身份"),
    "position": (
        "position",
        "location",
        "left",
        "right",
        "workspace",
        "位置",
        "左侧",
        "右侧",
    ),
    "lighting": ("lighting", "illumination", "light color", "光照", "照明"),
    "texture": ("texture", "background", "wall appearance", "纹理", "背景"),
}


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClaimFirstRuntimeError(f"{field} must be a non-empty string")
    return value.strip()


def _round_candidate_id(round_plan: Mapping[str, Any]) -> str:
    """Return the semantic candidate identity for catalog or dynamic rounds."""

    return _nonempty_text(
        round_plan.get("candidate_id") or round_plan.get("template_id"),
        "round_plan.candidate_id",
    )


def _failure_seeking_existential(user_query: str) -> bool:
    """Recognize an existential whose witness is a policy counterexample."""

    query = user_query.casefold()
    return bool(
        re.search(
            r"\b(?:fail(?:s|ed|ing|ure)?|weakness|counterexample|breaks?)\b",
            query,
        )
        or any(
            term in query
            for term in (
                "\u5931\u8d25",
                "\u5931\u6548",
                "\u5f31\u70b9",
                "\u53cd\u4f8b",
                "\u5d29\u6e83",
            )
        )
    )


def _target_task_name(target: Mapping[str, Any]) -> str:
    if "policy_task_binding" in target:
        try:
            return policy_task_binding_from_target(target)["task_name"]
        except (PolicyTaskBindingError, TypeError) as exc:
            raise ClaimFirstRuntimeError(str(exc)) from exc
    return _nonempty_text(target.get("task_name"), "target.task_name")


def _adapter_retrieval_aspects(task_name: str) -> list[dict[str, Any]]:
    """Project registered artifacts without adding them to the binding."""

    retrieval_index = resolve_task_retrieval_index(
        task_name,
        allow_unregistered=True,
    )
    grouped: dict[str, dict[str, Any]] = {}
    for contract in retrieval_index["entries"]:
        aspect = contract["aspect"]
        aspect_id = str(aspect["aspect_id"])
        entry = grouped.setdefault(
            aspect_id,
            {
                "aspect_id": aspect_id,
                "description": (
                    "Retrieval hint for "
                    f"{aspect['semantic_scope']} / {aspect['target_role']}."
                ),
                "template_ids": [],
            },
        )
        entry["template_ids"].append(str(contract["template_id"]))
    return [deepcopy(grouped[key]) for key in sorted(grouped)]


def control_template_id(target: Mapping[str, Any]) -> str:
    """Return the trusted official-scene control for a bound task."""

    task_name = _target_task_name(target)
    retrieval_index = resolve_task_retrieval_index(
        task_name,
        allow_unregistered=True,
    )
    template_id = retrieval_index["control_template_id"]
    if "policy_task_binding" in target:
        return template_id
    available = {
        str(item)
        for aspect in target.get("aspects", [])
        if isinstance(aspect, Mapping)
        for item in aspect.get("template_ids", [])
    }
    if template_id not in available:
        # Cached plans and older test fixtures may predate the neutral
        # official-baseline contract.  Keep them readable by accepting only an
        # already-bound unchanged-scene passthrough; newly built targets always
        # expose ``task_execution.official_baseline``.
        legacy_controls = [
            contract["template_id"]
            for contract in retrieval_index["entries"]
            if contract["template_id"] in available
            and contract["taskgen"]["operation"] == "official_passthrough"
        ]
        if len(legacy_controls) == 1:
            return legacy_controls[0]
        raise ClaimFirstRuntimeError(
            f"control template {template_id!r} is outside the bound task"
        )
    return template_id


def _uses_task_control_template(round_plan: Mapping[str, Any]) -> bool:
    task_name = round_plan.get("task_name")
    template_id = round_plan.get("template_id")
    if (
        not isinstance(task_name, str)
        or not task_name.strip()
        or not isinstance(template_id, str)
        or not template_id.strip()
    ):
        return False
    try:
        retrieval_index = resolve_task_retrieval_index(
            task_name.strip(),
            allow_unregistered=True,
        )
    except ValueError:
        return False
    return template_id.strip() == retrieval_index["control_template_id"]


def _template_aspect(target: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(template_id): str(aspect["aspect_id"])
        for aspect in target.get("aspects", [])
        if isinstance(aspect, Mapping)
        for template_id in aspect.get("template_ids", [])
    }


def _semantic_tokens(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return set().union(
            *(_semantic_tokens(key) | _semantic_tokens(item) for key, item in value.items())
        ) if value else set()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return set().union(*(_semantic_tokens(item) for item in value)) if value else set()
    if not isinstance(value, str):
        return set()
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", value.casefold())
        if token not in _SEMANTIC_STOPWORDS and len(token) > 1
    }


def _expanded_semantic_tokens(value: Any) -> set[str]:
    tokens = _semantic_tokens(value)
    text = json.dumps(value, ensure_ascii=False).casefold().replace("_", " ")
    for canonical, aliases in _SEMANTIC_ALIASES.items():
        if any(alias in text for alias in aliases):
            tokens.add(canonical)
    return tokens


def _normalized_identifier_phrase(value: str) -> str:
    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            value.casefold().replace("_", " ").replace(".", " "),
        )
    )


def _catalog_external_specificity(
    semantic_fields: Mapping[str, str],
    *,
    source_query: str,
) -> dict[str, Any]:
    """Describe a concrete, Query-grounded concern without inventing a route.

    A concern is specific only when a non-generic semantic anchor occurs in at
    least two Query-interpretation fields *and* in the original Query. This
    broad Query such as "test general robustness" ambiguous instead of letting
    a model-authored detail silently create an executable itinerary.

    The public aspect ontology is used only to name an exact catalog-external
    semantic concept.  It does not authorize a TaskGen operation or template.
    """

    field_tokens = {
        field: _expanded_semantic_tokens(value) - _CONCERN_GENERIC_TOKENS
        for field, value in semantic_fields.items()
    }
    counts: dict[str, int] = {}
    for tokens in field_tokens.values():
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    repeated = {token for token, count in counts.items() if count >= 2}
    query_tokens = (
        _expanded_semantic_tokens(source_query) - _CONCERN_GENERIC_TOKENS
    )
    grounded = sorted(repeated & query_tokens)

    primary_text = " ".join(
        [
            source_query,
            semantic_fields["sub_aspect"],
            semantic_fields["requested_variation"],
        ]
    )
    normalized_primary = _normalized_identifier_phrase(primary_text)
    ontology_matches: list[str] = []
    for item in public_aspect_ontology():
        identifiers = [item["aspect_id"], *item.get("aliases", [])]
        if any(
            f" {_normalized_identifier_phrase(identifier)} "
            in f" {normalized_primary} "
            for identifier in identifiers
        ):
            ontology_matches.append(str(item["aspect_id"]))

    return {
        "specific": bool(grounded),
        "grounded_anchor_tokens": grounded,
        "repeated_anchor_tokens": sorted(repeated),
        "ontology_matches": sorted(set(ontology_matches)),
        "canonical_aspect_id": (
            ontology_matches[0] if len(set(ontology_matches)) == 1 else None
        ),
    }


def resolve_concern_candidate_domain(
    concern: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an online Query interpretation to a finite capability domain.

    Query interpretation is authored before task/capability retrieval and never sees the
    executable catalog.  The runtime may therefore use its semantic fields to
    narrow a trusted capability inventory, but only on an exact or unique
    lexical match.  Broad or tied concerns keep the complete non-control
    domain and explicitly ask the Planner to discover the most informative
    first candidate.  Ambiguity is therefore a planning state, not an
    admission failure.
    """

    if not isinstance(concern, Mapping):
        raise ClaimFirstRuntimeError("Query interpretation must be an object")
    semantic_fields = {
        field: _nonempty_text(
            concern.get(field), f"QueryInterpretation.{field}"
        )
        for field in (
            "sub_aspect",
            "hypothesis",
            "requested_variation",
            "measurement_need",
        )
    }
    source_query = _nonempty_text(
        concern.get("source_query"), "QueryInterpretation.source_query"
    )
    control_template = control_template_id(target)
    concern_text = " ".join(semantic_fields.values()).casefold()
    concern_tokens = (
        _expanded_semantic_tokens(semantic_fields) - _CONCERN_GENERIC_TOKENS
    )
    source_query_tokens = (
        _expanded_semantic_tokens(source_query) - _CONCERN_GENERIC_TOKENS
    )
    candidates: list[dict[str, Any]] = []
    for raw_aspect in target.get("aspects", []):
        if not isinstance(raw_aspect, Mapping):
            continue
        aspect_id = _nonempty_text(
            raw_aspect.get("aspect_id"), "target.aspect.aspect_id"
        )
        template_ids = [
            str(item)
            for item in raw_aspect.get("template_ids", [])
            if str(item) != control_template
        ]
        if not template_ids:
            continue
        exact = bool(
            aspect_id.casefold() in concern_text
            or any(template_id.casefold() in concern_text for template_id in template_ids)
        )
        aspect_tokens = (
            _expanded_semantic_tokens(
                {
                    "aspect_id": aspect_id,
                    "description": raw_aspect.get("description"),
                    "template_ids": template_ids,
                }
            )
            - _CONCERN_GENERIC_TOKENS
        )
        matched = sorted(concern_tokens & aspect_tokens)
        query_matched = sorted(source_query_tokens & aspect_tokens)
        candidates.append(
            {
                "aspect_id": aspect_id,
                "template_ids": template_ids,
                "exact": exact,
                "score": len(matched),
                "matched_tokens": matched,
                "source_query_matched_tokens": query_matched,
            }
        )
    external_specificity = _catalog_external_specificity(
        semantic_fields,
        source_query=source_query,
    )
    if not candidates:
        if not external_specificity["specific"]:
            return {
                "schema_version": 1,
                # There is no registered non-control domain to discover
                # inside.  Admit the Query to the open-world Planner instead
                # of treating an empty catalog as an unsupported request.
                # The later Proposal, not this broad Query interpretation,
                # decides which scene/checker/tool needs materialization.
                "decision": "catalog_external",
                "resolution": "open_world_candidate_discovery_required",
                "candidate_aspect_ids": None,
                "selected_aspect_id": None,
                "selected_template_ids": [],
                "ranked_aspects": [],
                "concern_created_before_catalog": True,
                "catalog_was_model_visible": False,
                "concern": deepcopy(dict(concern)),
                "candidate_discovery_required": True,
                "execution_authorized": False,
            }
        return {
            "schema_version": 1,
            "decision": "catalog_external",
            "resolution": "generation_required_no_registered_candidate",
            "candidate_aspect_ids": None,
            "selected_aspect_id": None,
            "selected_template_ids": [],
            "ranked_aspects": [],
            "concern_created_before_catalog": True,
            "catalog_was_model_visible": False,
            "concern": deepcopy(dict(concern)),
            # Read-only compatibility projection for legacy callers.
            "task_need": {
                "required": True,
                "description": semantic_fields["requested_variation"],
            },
            "tool_need": {
                "required": True,
                "description": semantic_fields["measurement_need"],
                "reuse_first": True,
            },
            "catalog_external_specificity": external_specificity,
            # Generation, validation, and registration must occur before an
            # execution backend treats this semantic request as runnable.
            "execution_authorized": False,
        }

    exact_matches = [item for item in candidates if item["exact"]]
    selected: dict[str, Any] | None = None
    resolution = "broad_or_ambiguous"
    top_score = max(int(item["score"]) for item in candidates)
    if len(exact_matches) == 1:
        exact_candidate = exact_matches[0]
        if exact_candidate["source_query_matched_tokens"]:
            selected = exact_candidate
            resolution = "exact_query_supported_concern"
    elif not exact_matches:
        top = [item for item in candidates if int(item["score"]) == top_score]
        second_score = max(
            (
                int(item["score"])
                for item in candidates
                if item is not top[0]
            ),
            default=0,
        ) if len(top) == 1 else top_score
        if (
            top_score >= 2
            and top_score - second_score >= 1
            and len(top) == 1
            and top[0]["source_query_matched_tokens"]
        ):
            selected = top[0]
            resolution = "unique_query_supported_concern"

    catalog_external = bool(
        selected is None
        and not exact_matches
        and all(
            not item["source_query_matched_tokens"]
            for item in candidates
        )
        and external_specificity["specific"]
    )
    ranked_aspects = sorted(
        candidates,
        key=lambda item: (
            not bool(item["exact"]),
            -int(item["score"]),
            str(item["aspect_id"]),
        ),
    )
    result = {
        "schema_version": 1,
        "decision": (
            "bind_single_aspect"
            if selected is not None
            else "catalog_external"
            if catalog_external
            else "discover_candidates"
        ),
        "resolution": (
            "unsupported_or_generation_required"
            if catalog_external
            else resolution
        ),
        "candidate_aspect_ids": (
            [str(selected["aspect_id"])]
            if selected is not None
            else None
            if catalog_external
            else [str(item["aspect_id"]) for item in ranked_aspects]
        ),
        "selected_aspect_id": (
            str(selected["aspect_id"]) if selected is not None else None
        ),
        "selected_template_ids": (
            list(selected["template_ids"]) if selected is not None else []
        ),
        "ranked_aspects": ranked_aspects,
        "concern_created_before_catalog": True,
        "catalog_was_model_visible": False,
    }
    if catalog_external:
        result.update(
            {
                # Preserve the provider-authored semantic request for a later
                # TaskAdapter.  Nothing here names or authorizes an executable
                # template, operation, route, or success checker.
                "concern": deepcopy(dict(concern)),
                # Read-only compatibility projection for legacy callers.
                "task_need": {
                    "required": True,
                    "description": semantic_fields["requested_variation"],
                },
                "tool_need": {
                    "required": True,
                    "description": semantic_fields["measurement_need"],
                    "reuse_first": True,
                },
                "catalog_external_specificity": external_specificity,
                "execution_authorized": False,
            }
        )
    elif selected is None:
        result.update(
            {
                "concern": deepcopy(dict(concern)),
                "candidate_discovery_required": True,
                "execution_authorized": False,
            }
        )
    return result


def resolve_semantic_proposal(
    proposal: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    executed_template_ids: Sequence[str],
    control_template: str,
) -> dict[str, Any]:
    """Resolve one semantic proposal to an unexecuted trusted template.

    The Plan Agent chooses a semantic sub-aspect, not a hidden left/right or
    instance id.  An exact aspect therefore materializes the first remaining
    template in the preregistered runtime order.  Lexical ambiguity *across*
    aspects still fails closed.
    """

    normalized = validate_open_query_plan_proposal(proposal, has_evidence=True)
    if normalized["action"] != "continue":
        raise ClaimFirstRuntimeError(
            "the query contract, not the model, owns Plan Agent stopping"
        )
    executed = {str(item) for item in executed_template_ids}
    proposal_aspect = str(normalized["sub_aspect"])
    proposal_tokens = _semantic_tokens(normalized)
    perturbation = normalized.get("requested_perturbation")
    if not isinstance(perturbation, Mapping):
        perturbation = {}
    # Resolve what the proposal explicitly asks to *change* before looking at
    # the full prose.  In particular, tokens in ``preserve`` must not turn an
    # object-instance proposal into a clutter or lighting experiment.
    change_intent_tokens = _semantic_tokens(
        {
            "sub_aspect": proposal_aspect,
            "description": perturbation.get("description"),
            "controlled_changes": perturbation.get("controlled_changes", []),
        }
    )
    eligible_aspects: list[dict[str, Any]] = []
    for aspect in target.get("aspects", []):
        if not isinstance(aspect, Mapping):
            continue
        aspect_id = str(aspect.get("aspect_id") or "")
        templates = [
            str(raw_template)
            for raw_template in aspect.get("template_ids", [])
            if str(raw_template) != control_template
            and str(raw_template) not in executed
        ]
        if not templates:
            continue
        aspect_tokens = _semantic_tokens(
            {
                "aspect_id": aspect_id,
                "description": aspect.get("description"),
                "templates": templates,
            }
        )
        eligible_aspects.append(
            {
                "aspect_id": aspect_id,
                "template_ids": templates,
                "score": len(proposal_tokens & aspect_tokens),
                "matched_tokens": sorted(proposal_tokens & aspect_tokens),
                "change_intent_tokens": sorted(
                    change_intent_tokens
                    & (
                        _semantic_tokens(aspect_id)
                        - _ASPECT_GENERIC_TOKENS
                    )
                ),
            }
        )
    if not eligible_aspects:
        raise ClaimFirstRuntimeError(
            "no unexecuted non-control template remains in the bound task"
        )

    for aspect in eligible_aspects:
        if proposal_aspect in aspect["template_ids"]:
            chosen = proposal_aspect
            resolution = "exact_template"
            break
    else:
        exact_aspects = [
            aspect
            for aspect in eligible_aspects
            if proposal_aspect == aspect["aspect_id"]
        ]
        if exact_aspects:
            aspect = exact_aspects[0]
            chosen = aspect["template_ids"][0]
            resolution = "exact_aspect_runtime_order"
        else:
            best_change_score = max(
                len(aspect["change_intent_tokens"])
                for aspect in eligible_aspects
            )
            change_tied = [
                aspect
                for aspect in eligible_aspects
                if len(aspect["change_intent_tokens"]) == best_change_score
            ]
            if best_change_score > 0 and len(change_tied) == 1:
                aspect = change_tied[0]
                chosen = aspect["template_ids"][0]
                resolution = "explicit_change_intent_aspect_runtime_order"
            else:
                best_score = max(
                    int(aspect["score"]) for aspect in eligible_aspects
                )
                tied = [
                    aspect
                    for aspect in eligible_aspects
                    if int(aspect["score"]) == best_score
                ]
                if best_score <= 0 or len(tied) != 1:
                    raise ClaimFirstRuntimeError(
                        "semantic proposal does not resolve uniquely across "
                        "trusted aspects; top candidates="
                        f"{[(item['aspect_id'], item['score']) for item in tied]}"
                    )
                aspect = tied[0]
                chosen = aspect["template_ids"][0]
                resolution = "unique_lexical_aspect_runtime_order"
    selected = next(
        aspect
        for aspect in eligible_aspects
        if chosen in aspect["template_ids"]
    )
    if not chosen:
        raise ClaimFirstRuntimeError(
            "semantic proposal did not select a remaining trusted template"
        )
    return {
        "schema_version": 1,
        "semantic_sub_aspect": proposal_aspect,
        "resolved_aspect_id": selected["aspect_id"],
        "resolved_template_id": chosen,
        "resolution": resolution,
        "hidden": True,
        "matched_tokens": selected["matched_tokens"],
        "catalog_was_model_visible": False,
    }


def _round_artifact_refs(
    round_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project the small set of artifacts needed to inspect one round.

    These are navigation pointers, not integrity claims.  Experimental
    preregistration may hash a bundle separately without making the normal
    Plan Agent runtime depend on a provenance subsystem.
    """

    refs: list[dict[str, Any]] = []
    child_run_id = str(round_summary.get("taskgen_run_id") or "").strip()
    if child_run_id:
        refs.append(
            {
                "kind": "child_manifest",
                "path": f"mea/generated_tasks/{child_run_id}/manifest.json",
            }
        )

    execution_dir = str(
        round_summary.get("execution_artifact_dir") or ""
    ).strip().rstrip("/\\")
    explicit_artifacts = round_summary.get("evidence_artifact_paths")
    explicit_artifacts = (
        explicit_artifacts
        if isinstance(explicit_artifacts, Mapping)
        else {}
    )
    observations = round_summary.get("observations")
    observations = observations if isinstance(observations, Mapping) else {}
    evidence_aggregate_path = str(
        explicit_artifacts.get("evidence_aggregate") or ""
    ).strip()
    if not evidence_aggregate_path and execution_dir:
        evidence_aggregate_path = f"{execution_dir}/evidence_aggregate.json"
    if evidence_aggregate_path and isinstance(
        observations.get("evidence_aggregate"), Mapping
    ):
        refs.append(
            {
                "kind": "evidence_aggregate",
                "path": evidence_aggregate_path,
            }
        )
    aggregate_path = str(
        explicit_artifacts.get("round_aggregate") or ""
    ).strip()
    if not aggregate_path and execution_dir:
        aggregate_path = f"{execution_dir}/aggregate_result.json"
    if aggregate_path and isinstance(observations.get("aggregate"), Mapping):
        refs.append(
            {
                "kind": "round_aggregate",
                "path": aggregate_path,
            }
        )
    planned_tool = observations.get("planned_tool")
    tool_path = str(
        explicit_artifacts.get("tool_execution") or ""
    ).strip()
    if not tool_path and execution_dir:
        tool_path = f"{execution_dir}/planned_tool/tool_execution.json"
    if (
        tool_path
        and isinstance(planned_tool, Mapping)
        and planned_tool.get("status") != "skipped"
    ):
        refs.append(
            {
                "kind": "tool_execution",
                "path": tool_path,
            }
        )
    execution_vqa = observations.get("execution_vqa")
    if isinstance(execution_vqa, Mapping):
        artifacts = execution_vqa.get("artifacts")
        result_path = (
            str(artifacts.get("result") or "").strip()
            if isinstance(artifacts, Mapping)
            else ""
        )
        if result_path:
            refs.append(
                {
                    "kind": "execution_vqa_result",
                    "path": result_path,
                }
            )
    return refs


def _compact_planned_tool_evidence(
    observations: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expose measured Tool values to the next semantic planning step.

    The typed Tool execution remains authoritative. This projection is only a
    compact prompt-facing observation; it never replaces official success or
    changes the deterministic QueryContract outcome by itself.
    """

    planned = observations.get("planned_tool")
    if not isinstance(planned, Mapping):
        return []
    route_decision = planned.get("route_decision")
    route_decision = (
        route_decision if isinstance(route_decision, Mapping) else {}
    )
    validation = planned.get("validation")
    validation = validation if isinstance(validation, Mapping) else {}
    route = (
        planned.get("route")
        or route_decision.get("resolved_route")
        or route_decision.get("route")
    )
    provider_called = route_decision.get(
        "provider_called",
        validation.get("provider_called"),
    )
    tool_request = planned.get("tool_request")
    tool_request = (
        tool_request if isinstance(tool_request, Mapping) else {}
    )
    metric_spec = tool_request.get("metric_spec")
    metric_spec = metric_spec if isinstance(metric_spec, Mapping) else {}
    metric_description = str(metric_spec.get("description") or "").strip()
    semantic_review = validation.get("semantic_review")
    semantic_review = (
        semantic_review if isinstance(semantic_review, Mapping) else {}
    )
    semantic_checks = semantic_review.get("checks")
    semantic_checks = (
        semantic_checks if isinstance(semantic_checks, Mapping) else {}
    )
    compact: list[dict[str, Any]] = []
    episodes = planned.get("episodes")
    if not isinstance(episodes, list):
        return compact
    for episode in episodes:
        if not isinstance(episode, Mapping):
            continue
        result = episode.get("result")
        result = result if isinstance(result, Mapping) else episode
        details = result.get("details")
        details = details if isinstance(details, Mapping) else {}
        item = {
            "metric": str(
                result.get("tool")
                or result.get("metric")
                or planned.get("reference_tool")
                or ""
            ),
            "value": result.get("value"),
            "unit": result.get("unit"),
            "passed": result.get("passed"),
            "route": route,
            "provider_called": provider_called,
            "null_reason": details.get("reason"),
        }
        if metric_description:
            item["description"] = metric_description
        if semantic_checks.get("returns_diagnostic_not_success") is True:
            item["returns_diagnostic_not_success"] = True
        compact.append(item)
    return compact


def build_claim_first_evidence_record(
    round_plan: Mapping[str, Any],
    round_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive compact semantic/query evidence from one completed runtime round."""

    if round_plan.get("round_id") != round_summary.get("round_id"):
        raise ClaimFirstRuntimeError("round plan and summary ids disagree")
    candidate_id = _round_candidate_id(round_plan)
    evidence_round = deepcopy(dict(round_plan))
    # EvidencePacket v1 calls this execution identity ``template_id``.  A
    # dynamic plan has no catalog template, so project its candidate id into
    # that legacy transport field without mutating the runtime plan.
    if not evidence_round.get("template_id"):
        evidence_round["template_id"] = candidate_id
    packet = validate_evidence_packet(
        build_evidence_packet(
            {"rounds": [evidence_round], "max_rounds": 1},
            [deepcopy(dict(round_summary))],
        )
    )
    refs = _round_artifact_refs(round_summary)
    observations = round_summary.get("observations")
    observations = observations if isinstance(observations, Mapping) else {}
    policy_outcome = (
        observations.get("policy_outcome")
        if isinstance(observations.get("policy_outcome"), Mapping)
        else {
            "metric": "official_check_success",
            "authority": "official_check_success",
            "binding": None,
            "value": None,
            "official_equivalent": True,
            "execution_scope": "legacy_unspecified_official",
        }
    )
    outcome_semantics = observations.get("outcome_semantics")
    if not isinstance(outcome_semantics, Mapping):
        outcome_semantics = policy_outcome.get("outcome_semantics")
    outcome_semantics = (
        deepcopy(dict(outcome_semantics))
        if isinstance(outcome_semantics, Mapping)
        else {
            "schema_version": 1,
            "status": "non_comparable",
            "evidence_conflict": False,
            "official_equivalent": policy_outcome.get("official_equivalent"),
            "episodes": [],
            "reason_codes": ["outcome_semantics_not_recorded"],
        }
    )
    outcome_semantics_status = str(
        outcome_semantics.get("status") or "non_comparable"
    )
    strength = packet["evidence_strength"]
    success_rate = packet["policy"]["success_rate"]
    generated_metric = (
        policy_outcome.get("metric") == "generated_check_success"
    )
    if outcome_semantics_status == "conflict":
        semantic_outcome = "ambiguous"
        candidate_outcome = "conflict"
    elif generated_metric and outcome_semantics_status not in {
        "equivalent_agreement",
        "expected_semantic_extension",
    }:
        semantic_outcome = "ambiguous"
        candidate_outcome = "unknown"
    elif strength == "conflicting":
        semantic_outcome = "ambiguous"
        candidate_outcome = "conflict"
    elif strength != "sufficient" or success_rate is None:
        semantic_outcome = "ambiguous"
        candidate_outcome = "unknown"
    elif float(success_rate) >= 1.0:
        semantic_outcome = "success"
        candidate_outcome = "pass"
    else:
        semantic_outcome = "failure"
        candidate_outcome = "fail"

    task_proposal = round_plan.get("task_proposal") or {}
    sub_aspect = str(
        task_proposal.get("aspect_id")
        or round_plan.get("sub_aspect")
        or round_plan.get("aspect_id")
        or "unknown"
    )
    hypothesis = str(
        task_proposal.get("intent")
        or round_plan.get("task_instruction")
        or f"Evaluate {sub_aspect}."
    ).strip()
    changes = task_proposal.get("changes")
    perturbation = (
        json.dumps(changes, ensure_ascii=False, sort_keys=True)
        if isinstance(changes, Mapping) and changes
        else "unchanged official-scene control"
        if _uses_task_control_template(round_plan)
        else candidate_id
    )
    limitations = [
        "One bounded runtime round is not a statistical generalization estimate."
    ]
    if strength != "sufficient":
        limitations.append(
            "The typed Rule/VQA/pipeline evidence is not sufficient: "
            + ", ".join(packet["reason_codes"] or [strength])
        )
    if success_rate is None:
        limitations.append("Policy success was not reported for this round.")
    if policy_outcome.get("official_equivalent") is False:
        limitations.append(
            "This round is judged by the bounded generated_check_success "
            "predicate and is not an official RoboTwin success result."
        )
    if outcome_semantics_status == "expected_semantic_extension":
        limitations.append(
            "The generated checker has not been certified as equivalent to "
            "the official core predicate; its verdict must be treated as "
            "experimental."
        )
    elif outcome_semantics_status == "conflict":
        limitations.append(
            "Generated and official/core success semantics conflict; this "
            "round cannot satisfy the Query sufficiency contract."
        )
    planned_tool_evidence = _compact_planned_tool_evidence(observations)
    tool_summary = (
        json.dumps(
            planned_tool_evidence,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        if planned_tool_evidence
        else "[]"
    )
    summary_text = (
        f"EvidencePacket strength={strength}; "
        f"authoritative_candidate_outcome={semantic_outcome}; "
        f"success_predicate_metric={policy_outcome.get('metric')}; "
        f"success_predicate_value={policy_outcome.get('value')}; "
        f"success_predicate_authority={policy_outcome.get('authority')}; "
        f"success_predicate_semantics={outcome_semantics_status}; "
        f"policy_success_rate={success_rate}; "
        f"Rule metric={packet['rule']['metric']}; "
        f"VQA status={packet['vqa']['status']}; "
        "diagnostic_tool_role=supporting_measurement_not_success_authority; "
        f"diagnostic_tool_measurements={tool_summary}."
    )
    open_query = validate_open_query_evidence(
        [
            {
                "schema_version": 1,
                "round_id": str(round_plan["round_id"]),
                "tested_sub_aspect": sub_aspect,
                "tested_hypothesis": hypothesis,
                "tested_perturbation": perturbation,
                "outcome": semantic_outcome,
                "evidence_summary": summary_text,
                "limitations": limitations,
            }
        ]
    )[0]
    diagnosis = None
    if candidate_outcome == "fail":
        diagnosis = (
            f"Observed policy success_rate={float(success_rate):.6g} for "
            f"{candidate_id} with complete Rule metric "
            f"{packet['rule']['metric']}; this localizes an observed weakness "
            "but does not establish a causal mechanism."
        )
    candidate = {
        "candidate_id": candidate_id,
        "outcome": candidate_outcome,
        "score": (
            float(success_rate) if success_rate is not None else None
        ),
        "diagnosis": diagnosis,
    }
    return {
        "schema_version": 1,
        "round_id": str(round_plan["round_id"]),
        "template_id": str(round_plan.get("template_id") or ""),
        "candidate_id": candidate_id,
        "open_query_evidence": open_query,
        "candidate_evidence": candidate,
        "evaluation_outcome": deepcopy(dict(policy_outcome)),
        "outcome_semantics": outcome_semantics,
        "planned_tool_evidence": planned_tool_evidence,
        "evidence_packet": packet,
        "evidence_refs": refs,
    }


def render_query_answer(
    user_query: str,
    assessment: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_valid: bool,
    baseline_stop_reason: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic query answer/limitation projection."""

    query = _nonempty_text(user_query, "user_query")
    contract = assessment.get("contract")
    contract = contract if isinstance(contract, Mapping) else {}
    dynamic_domain = contract.get("schema_version") == 2
    if not baseline_valid:
        answered = False
        stop_reason = baseline_stop_reason or "control_baseline_invalid"
        verdict = "inconclusive"
        answer = (
            "The original Query cannot be attributed to a tested property "
            "because the required unchanged-scene control did not produce "
            "complete successful policy evidence."
        )
        untested = list(
            assessment.get("contract", {}).get("candidate_universe", [])
        )
        limitations = [
            "No property attribution is allowed without a passing control.",
            "The observed control result may reflect policy, simulator, or pipeline effects.",
        ]
    else:
        answered = bool(assessment.get("evidence_sufficient"))
        stop_reason = str(assessment.get("stop_reason") or "continue")
        verdict = str(assessment.get("claim_verdict") or "inconclusive")
        if answered:
            if dynamic_domain:
                answer = (
                    "For the currently discovered candidate domain, the Query "
                    f"verdict is {verdict}."
                )
            else:
                answer = (
                    "For the finite registered candidate domain, the Query "
                    f"verdict is {verdict}."
                )
        else:
            answer = (
                "The bounded evidence does not yet satisfy the truth conditions "
                "needed to answer the original Query."
            )
        untested = list(assessment.get("untested_candidate_ids") or [])
        limitations = list(assessment.get("limitations") or [])
    if untested:
        limitations.append(
            "Untested candidates: " + ", ".join(untested)
        )
    limitations.extend(
        [
            "This answer is limited to the bound task, checkpoint, variants, and recorded seeds.",
            "A finite-domain N-small result is not a broad generalization guarantee.",
        ]
    )
    refs = [
        deepcopy(ref)
        for record in records
        for ref in record.get("evidence_refs", [])
        if isinstance(ref, Mapping)
    ]
    outcome_authorities = [
        deepcopy(record["evaluation_outcome"])
        for record in records
        if isinstance(record.get("evaluation_outcome"), Mapping)
    ]
    non_official = [
        item
        for item in outcome_authorities
        if item.get("official_equivalent") is False
    ]
    if non_official:
        limitations.append(
            "At least one candidate verdict uses generated_check_success; "
            "it must not be interpreted as official benchmark success."
        )
    outcome_semantics = [
        deepcopy(record["outcome_semantics"])
        for record in records
        if isinstance(record.get("outcome_semantics"), Mapping)
    ]
    semantic_conflicts = [
        item for item in outcome_semantics if item.get("status") == "conflict"
    ]
    semantic_extensions = [
        item
        for item in outcome_semantics
        if item.get("status") == "expected_semantic_extension"
    ]
    if semantic_conflicts:
        limitations.append(
            "At least one round has conflicting generated versus "
            "official/core success semantics."
        )
    if semantic_extensions:
        limitations.append(
            "At least one generated checker has not been certified as "
            "official-equivalent; its verdict must be treated as experimental."
        )
    answer_scope = (
        "bounded_experimental_query_semantics"
        if non_official
        else "official_equivalent"
    )
    return {
        "schema_version": 1,
        "original_query": query,
        "answered": answered,
        "answer_scope": answer_scope,
        "official_benchmark_answered": bool(answered and not non_official),
        "stop_reason": stop_reason,
        "claim_type": assessment.get("contract", {}).get("claim_type"),
        "claim_verdict": verdict,
        "answer": answer,
        "tested_candidate_ids": list(
            assessment.get("observed_candidate_ids") or []
        ),
        "untested_candidate_ids": untested,
        "limitations": list(dict.fromkeys(limitations)),
        "evidence_refs": refs,
        "evaluation_outcomes": outcome_authorities,
        "outcome_semantics": outcome_semantics,
        "evidence_conflict": bool(semantic_conflicts),
    }


def _current_planning_evidence(
    observation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return evidence whose round lineage agrees with runtime records."""

    if not isinstance(observation, Mapping):
        raise ClaimFirstRuntimeError("Plan Agent observation must be an object")
    raw_history = observation.get("open_query_evidence_history")
    if not isinstance(raw_history, list):
        raise ClaimFirstRuntimeError(
            "Plan Agent observation has no open_query_evidence_history"
        )
    try:
        history = validate_open_query_evidence(raw_history)
    except ClaimFirstPlanError as exc:
        raise ClaimFirstRuntimeError(str(exc)) from exc
    records = observation.get("records")
    if not isinstance(records, list):
        raise ClaimFirstRuntimeError("Plan Agent observation has no records")
    record_round_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ClaimFirstRuntimeError(
                f"Plan Agent observation record {index} must be an object"
            )
        record_round_ids.append(
            _nonempty_text(
                record.get("round_id"),
                f"observation.records[{index}].round_id",
            )
        )
    evidence_round_ids = [item["round_id"] for item in history]
    if evidence_round_ids != record_round_ids:
        raise ClaimFirstRuntimeError(
            "open_query_evidence_history does not align with completed "
            "runtime records"
        )
    return history


def _attach_planning_lineage(
    bound_step: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist semantic-decision lineage at both bundle and plan-step levels."""

    result = deepcopy(dict(bound_step))
    trusted_lineage = deepcopy(dict(lineage))
    result["planning_lineage"] = trusted_lineage
    plan_step = result.get("plan_step")
    if not isinstance(plan_step, Mapping):
        raise ClaimFirstRuntimeError("bound semantic step has no plan_step")
    result["plan_step"] = {
        **deepcopy(dict(plan_step)),
        "planning_lineage": deepcopy(trusted_lineage),
    }
    return result


class PlanAgentSession:
    """Own one complete Plan Agent session.

    The public session owns both semantic decisions and the frozen execution
    transport.  Legacy semantic-only fixtures may still pass a catalog-shaped
    target, but production targets construct their execution transport here so
    callers cannot keep two independently evolving session states.
    """

    def __init__(
        self,
        user_query: str,
        target: Mapping[str, Any],
        *,
        query_contract: Mapping[str, Any] | None = None,
        candidate_aspect_ids: Sequence[str] | None = None,
        require_control_anchor: bool | None = None,
        retrieval_aspects: Sequence[Mapping[str, Any]] | None = None,
        control_round: Mapping[str, Any] | None = None,
    ):
        self.user_query = _nonempty_text(user_query, "user_query")
        self.target = deepcopy(dict(target))
        self.task_name = _target_task_name(self.target)
        if retrieval_aspects is None:
            raw_retrieval_aspects = self.target.get("aspects")
            if not isinstance(raw_retrieval_aspects, list):
                raw_retrieval_aspects = _adapter_retrieval_aspects(
                    self.task_name
                )
        else:
            raw_retrieval_aspects = list(retrieval_aspects)
        if any(
            not isinstance(aspect, Mapping)
            for aspect in raw_retrieval_aspects
        ):
            raise ClaimFirstRuntimeError(
                "retrieval_aspects must contain only artifact-hint objects"
            )
        if (
            not raw_retrieval_aspects
            and "policy_task_binding" not in self.target
        ):
            raise ClaimFirstRuntimeError(
                "legacy retrieval_aspects must contain artifact hints"
            )
        self.retrieval_aspects = [
            deepcopy(dict(aspect)) for aspect in raw_retrieval_aspects
        ]
        if require_control_anchor is not None and not isinstance(
            require_control_anchor, bool
        ):
            raise ClaimFirstRuntimeError(
                "require_control_anchor must be bool or None"
            )
        self.control_template = control_template_id(self.target)
        if candidate_aspect_ids is not None:
            allowed_aspects = {
                _nonempty_text(item, "candidate_aspect_ids[]")
                for item in candidate_aspect_ids
            }
            known_aspects = {
                str(aspect.get("aspect_id") or "")
                for aspect in self.retrieval_aspects
                if isinstance(aspect, Mapping)
            }
            unknown_aspects = allowed_aspects - known_aspects
            if unknown_aspects:
                raise ClaimFirstRuntimeError(
                    "routed candidate aspects leave the bound task catalog: "
                    f"{sorted(unknown_aspects)}"
                )
            self.retrieval_aspects = [
                deepcopy(dict(aspect))
                for aspect in self.retrieval_aspects
                if isinstance(aspect, Mapping)
                and (
                    str(aspect.get("aspect_id") or "") in allowed_aspects
                    or self.control_template
                    in {str(item) for item in aspect.get("template_ids", [])}
                )
            ]
        retrieval_target = {"aspects": self.retrieval_aspects}
        self.template_to_aspect = _template_aspect(retrieval_target)
        candidates = [
            template_id
            for template_id in self.template_to_aspect
            if template_id != self.control_template
        ]
        supplied_contract = (
            validate_query_sufficiency_contract(query_contract)
            if query_contract is not None
            else None
        )
        contract_requires_control = (
            supplied_contract["control_requirement"] == "required"
            if supplied_contract is not None
            else (
                True
                if require_control_anchor is None
                else require_control_anchor
            )
        )
        if (
            supplied_contract is not None
            and require_control_anchor is not None
            and require_control_anchor != contract_requires_control
        ):
            raise ClaimFirstRuntimeError(
                "require_control_anchor conflicts with QueryContract "
                "control_requirement"
            )
        self.require_control_anchor = contract_requires_control
        round_budget = int(self.target.get("max_rounds") or 0) - (
            1 if self.require_control_anchor else 0
        )
        # Legacy official-only adapters use max_rounds=1 because their catalog
        # contains only the control.  That catalog size must not prohibit one
        # Query-derived generation round.
        if not candidates and round_budget < 1:
            round_budget = 1
        if round_budget < 1:
            raise ClaimFirstRuntimeError(
                "Plan Agent runtime needs budget for at least one candidate round"
            )
        if query_contract is None:
            claim_type = infer_claim_type(self.user_query)
            if claim_type == "comparative":
                raise ClaimFirstRuntimeError(
                    "comparative Query requires an explicit preregistered "
                    "query-sufficiency contract with two groups"
                )
            failure_witness = bool(
                claim_type == "existential"
                and _failure_seeking_existential(self.user_query)
            )
            contract = build_query_sufficiency_contract(
                self.user_query,
                candidate_universe=candidates,
                round_budget=round_budget,
                claim_type=claim_type,
                # Bound-task templates are retrieval seeds, never proof that an
                # open Query's semantic candidate universe is exhaustive.
                candidate_universe_closed=False,
                existential_witness_outcome=(
                    "fail" if failure_witness else None
                ),
                control_requirement=(
                    "required"
                    if self.require_control_anchor
                    else "not_required"
                ),
            )
        else:
            assert supplied_contract is not None
            contract = supplied_contract
            # The task catalog is a retrieval index, not an authorization
            # boundary. A Query-derived candidate may therefore be absent from
            # the registered template inventory; its executable Task/Tool needs
            # are validated later by the materialization and evidence gates.
            if int(contract["round_budget"]) > round_budget:
                raise ClaimFirstRuntimeError(
                    "query contract spends rounds reserved for the control anchor"
                )
        self.query_contract = contract
        self.dynamic_candidates: dict[str, dict[str, Any]] = {}
        self._frozen_execution = (
            _FrozenExecutionTransport.from_target(
                self.target,
                control_round=control_round,
                query_contract=self.query_contract,
            )
            if "policy_task_binding" in self.target
            else None
        )
        if self._frozen_execution is not None:
            self.target = deepcopy(self._frozen_execution.target)

    def _require_frozen_execution(self) -> _FrozenExecutionTransport:
        transport = self._frozen_execution
        if transport is None:
            raise ClaimFirstRuntimeError(
                "this legacy Plan Agent fixture has no frozen execution binding"
            )
        return transport

    @property
    def execution_binding(self) -> dict[str, Any]:
        """Return the frozen policy/task binding owned by this session."""

        return deepcopy(self._require_frozen_execution().binding)

    def normalize_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize the executable plan through this session's transport."""

        return self._require_frozen_execution().normalize_plan(plan)

    def planning_context(self, repo_root: str | Path) -> dict[str, Any]:
        """Project trusted runtime capabilities for retrieval and generation."""

        return self._require_frozen_execution().planning_context(repo_root)

    def apply_plan_step(
        self,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]],
        proposal: Mapping[str, Any],
        *,
        materialized_round: Mapping[str, Any] | None = None,
        source: str = "provider",
        query_contract: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Apply one semantic decision to the frozen executable plan."""

        contract = (
            query_contract
            if query_contract is not None
            else self.query_contract
        )
        result = self._require_frozen_execution().apply_plan_step(
            plan,
            observation_history,
            proposal,
            materialized_round=materialized_round,
            source=source,
            query_contract=contract,
        )
        normalized_contract = result[0].get("query_contract")
        if isinstance(normalized_contract, Mapping):
            self.query_contract = deepcopy(dict(normalized_contract))
        return result

    def snapshot(
        self,
        user_query: str,
        plan: Mapping[str, Any],
        observation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return one combined semantic/execution session snapshot."""

        return self._require_frozen_execution().snapshot(
            user_query,
            plan,
            observation_history,
        )

    def _register_dynamic_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        require_direct: bool,
    ) -> dict[str, Any]:
        trusted = validate_experiment_candidate(candidate)
        expected_task = _nonempty_text(
            self.task_name,
            "target.task_name",
        )
        if trusted["base_task"] != expected_task:
            raise ClaimFirstRuntimeError(
                "frozen candidate base_task differs from the bound policy task"
            )
        if (
            require_direct
            and trusted.get("intent_alignment", {}).get("relationship")
            != "direct"
        ):
            raise ClaimFirstRuntimeError(
                "frozen candidate must directly implement its EvaluationIntent"
            )
        candidate_id = trusted["candidate_id"]
        existing = self.dynamic_candidates.get(candidate_id)
        if existing is not None and existing != trusted:
            raise ClaimFirstRuntimeError(
                f"dynamic candidate id collision: {candidate_id}"
            )
        self.dynamic_candidates[candidate_id] = trusted
        self.query_contract = extend_query_candidate_universe(
            self.query_contract,
            [candidate_id],
            candidate_universe_closed=False,
        )
        return deepcopy(trusted)

    def register_frozen_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Register a Query-only first candidate for a no-control session.

        A control-required session must leave its first semantic sub-aspect
        unfrozen until the control Aggregate has been observed.  Otherwise a
        Query-interpreter candidate authored before execution could be relabelled
        as Fig. 5 evidence-conditioned planning after the control merely passes.
        """

        if self.require_control_anchor:
            raise ClaimFirstRuntimeError(
                "control-required Plan Agent cannot freeze a pre-evidence "
                "candidate; author the next sub-aspect from observed control "
                "evidence"
            )

        return self._register_dynamic_candidate(
            candidate,
            require_direct=True,
        )

    def _bind_dynamic_candidate(
        self,
        *,
        proposal_bundle: Mapping[str, Any],
        proposal: Mapping[str, Any],
        candidate: Mapping[str, Any],
        executed_candidate_ids: Sequence[str],
        resolution: str,
        catalog_resolution_error: str | None,
        retrieval_hint: Mapping[str, Any] | None = None,
        require_direct: bool = False,
    ) -> dict[str, Any]:
        trusted = self._register_dynamic_candidate(
            candidate,
            require_direct=require_direct,
        )
        candidate_id = trusted["candidate_id"]
        if candidate_id in {str(item) for item in executed_candidate_ids}:
            raise ClaimFirstRuntimeError(
                f"dynamic candidate was already executed: {candidate_id}"
            )
        hint = dict(retrieval_hint or {})
        dynamic_resolution = {
            "schema_version": 1,
            "semantic_sub_aspect": proposal["sub_aspect"],
            "resolved_aspect_id": hint.get("resolved_aspect_id"),
            "resolved_template_id": None,
            "resolved_candidate_id": candidate_id,
            "resolution": resolution,
            "hidden": bool(hint.get("hidden", False)),
            "matched_tokens": list(hint.get("matched_tokens") or []),
            "catalog_was_model_visible": False,
            "catalog_resolution_error": catalog_resolution_error,
            "retrieval_aspect_id": hint.get("resolved_aspect_id"),
            "retrieval_template_id": hint.get("resolved_template_id"),
            "retrieval_resolution": hint.get("resolution"),
        }
        return {
            "schema_version": 2,
            "semantic_proposal_bundle": deepcopy(dict(proposal_bundle)),
            "semantic_needs": {
                "scene_need": {
                    "required": trusted["scene_need"] is not None,
                    "description": (
                        trusted["scene_need"]["description"]
                        if trusted["scene_need"] is not None
                        else None
                    ),
                },
                "checker_need": {
                    "required": trusted["checker_need"] is not None,
                    "description": (
                        trusted["checker_need"]["description"]
                        if trusted["checker_need"] is not None
                        else None
                    ),
                },
                "rule_tool_need": {
                    "required": trusted["rule_tool_need"] is not None,
                    "description": (
                        trusted["rule_tool_need"]["description"]
                        if trusted["rule_tool_need"] is not None
                        else None
                    ),
                    "reuse_first": True,
                },
                "vqa_tool_need": {
                    "required": trusted["vqa_tool_need"] is not None,
                    "description": (
                        trusted["vqa_tool_need"]["description"]
                        if trusted["vqa_tool_need"] is not None
                        else None
                    ),
                    "reuse_first": True,
                },
                "task_need": deepcopy(proposal["task_need"]),
                "tool_need": deepcopy(proposal["tool_need"]),
            },
            "resolution": dynamic_resolution,
            "query_contract": deepcopy(self.query_contract),
            "plan_step": {
                "schema_version": 2,
                "action": "propose",
                "aspect_id": proposal["sub_aspect"],
                "candidate_id": candidate_id,
                "execution_mode": "reuse_or_generate",
                "proposal": trusted,
                "rationale": proposal["rationale"],
                "answered_query": False,
            },
        }

    def bind_frozen_candidate(
        self,
        proposal_bundle: Mapping[str, Any],
        candidate: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        executed_candidate_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Authorize a no-control Query-only candidate without claiming refinement.

        Its semantic choice predates rollout evidence.  The returned lineage
        therefore remains ``pre_evidence`` and must not be counted as Fig. 5
        evidence-conditioned sub-aspect selection.
        """

        if self.require_control_anchor:
            raise ClaimFirstRuntimeError(
                "control-required Plan Agent cannot bind a pre-evidence "
                "candidate; author the next sub-aspect from observed control "
                "evidence"
            )
        assessment = observation.get("assessment")
        if not isinstance(assessment, Mapping):
            raise ClaimFirstRuntimeError(
                "Plan Agent observation has no assessment"
            )
        if assessment.get("should_stop"):
            raise ClaimFirstRuntimeError(
                "cannot bind a frozen candidate after the query contract stopped"
            )
        if (
            self.require_control_anchor
            and observation.get("control_passed") is not True
        ):
            raise ClaimFirstRuntimeError(
                "cannot execute a frozen property candidate before control passes"
            )
        raw_proposal = proposal_bundle.get("proposal")
        if not isinstance(raw_proposal, Mapping):
            raise ClaimFirstRuntimeError(
                "frozen proposal bundle has no proposal object"
            )
        proposal = validate_open_query_plan_proposal(
            raw_proposal,
            has_evidence=bool(observation.get("records")),
        )
        if proposal["action"] != "continue":
            raise ClaimFirstRuntimeError(
                "a frozen first candidate must be an action=continue proposal"
            )
        raw_lineage = proposal_bundle.get("planning_lineage")
        if not isinstance(raw_lineage, Mapping):
            raw_lineage = {
                "schema_version": 1,
                "decision_kind": "pre_evidence_query_candidate",
                "evidence_conditioned": False,
                "completed_round_ids": [],
                "completed_round_count": 0,
                "input_digest": None,
            }
        lineage = deepcopy(dict(raw_lineage))
        if (
            lineage.get("decision_kind")
            != "pre_evidence_query_candidate"
            or lineage.get("evidence_conditioned") is not False
            or lineage.get("completed_round_ids") != []
            or lineage.get("completed_round_count") != 0
        ):
            raise ClaimFirstRuntimeError(
                "bind_frozen_candidate accepts only an explicitly pre-evidence "
                "Query candidate"
            )
        bound = self._bind_dynamic_candidate(
            proposal_bundle=proposal_bundle,
            proposal=proposal,
            candidate=candidate,
            executed_candidate_ids=executed_candidate_ids,
            resolution="pre_evidence_query_proposal",
            catalog_resolution_error=None,
            require_direct=True,
        )
        return _attach_planning_lineage(bound, lineage)

    def observe(
        self,
        round_plans: Sequence[Mapping[str, Any]],
        round_summaries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Normalize all completed rounds and decide whether execution stops."""

        if len(round_plans) != len(round_summaries):
            raise ClaimFirstRuntimeError(
                "completed plans and summaries must be aligned"
            )
        if self.require_control_anchor and not round_plans:
            raise ClaimFirstRuntimeError(
                "control-first observation requires one completed control round"
            )
        records = [
            build_claim_first_evidence_record(plan, summary)
            for plan, summary in zip(round_plans, round_summaries)
        ]
        control_semantics: Mapping[str, Any] = {}
        control_authority_valid = True
        control_pipeline_valid = True
        baseline_valid = True
        if self.require_control_anchor:
            if records[0]["template_id"] != self.control_template:
                raise ClaimFirstRuntimeError(
                    "Plan Agent property attribution requires the control "
                    "template first"
                )
            control_packet = records[0]["evidence_packet"]
            control_outcome = records[0]["evaluation_outcome"]
            control_semantics = records[0].get("outcome_semantics") or {}
            control_authority_valid = bool(
                control_outcome.get("metric") == "official_check_success"
                and control_outcome.get("official_equivalent") is not False
                and control_semantics.get("status") != "conflict"
            )
            control_pipeline_valid = bool(
                control_packet["pipeline"]["passed"]
                and control_packet["policy"]["reported"]
                and control_packet["policy"]["success_rate"] is not None
            )
            baseline_valid = bool(
                control_authority_valid
                and control_pipeline_valid
                and float(control_packet["policy"]["success_rate"]) >= 1.0
            )
        candidate_records = (
            records[1:] if self.require_control_anchor else records
        )
        outside_candidate_ids = [
            record["candidate_id"]
            for record in candidate_records
            if record["candidate_id"]
            not in self.query_contract["candidate_universe"]
        ]
        if outside_candidate_ids:
            raise ClaimFirstRuntimeError(
                "completed candidate evidence is outside the active "
                "QueryContract universe: "
                f"{list(dict.fromkeys(outside_candidate_ids))}"
            )
        candidate_evidence = [
            deepcopy(record["candidate_evidence"])
            for record in candidate_records
        ]
        assessment = assess_query_sufficiency(
            self.query_contract,
            candidate_evidence,
            completed_rounds=len(candidate_records),
        )
        transport_conflict_ids = [
            record["candidate_id"]
            for record in candidate_records
            if (
                record["candidate_id"]
                in self.query_contract["candidate_universe"]
                and (
                    record.get("candidate_evidence", {}).get("outcome")
                    == "conflict"
                    or record.get("evidence_packet", {}).get(
                        "evidence_strength"
                    )
                    == "conflicting"
                )
            )
        ]
        if transport_conflict_ids:
            assessment = {
                **assessment,
                "should_stop": True,
                "stop_reason": "evidence_conflict",
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": (
                    "Rule, VQA, or execution evidence conflicts for a "
                    "completed candidate; the Query cannot be answered from "
                    "this evidence."
                ),
                "conflict_candidate_ids": list(
                    dict.fromkeys(
                        list(assessment.get("conflict_candidate_ids") or [])
                        + transport_conflict_ids
                    )
                ),
                "recommended_candidate_ids": [],
            }
        semantic_conflict_ids = [
            record["candidate_id"]
            for record in candidate_records
            if (
                record["candidate_id"]
                in self.query_contract["candidate_universe"]
                and record.get("outcome_semantics", {}).get("status")
                == "conflict"
            )
        ]
        if semantic_conflict_ids:
            assessment = {
                **assessment,
                "should_stop": True,
                "stop_reason": "outcome_semantics_conflict",
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": (
                    "Generated and official/core success semantics disagree for "
                    "a completed candidate; the Query cannot be answered from "
                    "this evidence."
                ),
                "conflict_candidate_ids": list(
                    dict.fromkeys(
                        list(assessment.get("conflict_candidate_ids") or [])
                        + semantic_conflict_ids
                    )
                ),
                "recommended_candidate_ids": [],
            }
        semantic_non_comparable_ids = [
            record["candidate_id"]
            for record in candidate_records
            if (
                record["candidate_id"]
                in self.query_contract["candidate_universe"]
                and record.get("evaluation_outcome", {}).get("metric")
                == "generated_check_success"
                and record.get("outcome_semantics", {}).get("status")
                == "non_comparable"
            )
        ]
        if semantic_non_comparable_ids:
            assessment = {
                **assessment,
                "should_stop": True,
                "stop_reason": "outcome_semantics_non_comparable",
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": (
                    "The generated checker lacks a comparable official/core "
                    "projection, so this Query cannot be answered from the "
                    "completed evidence."
                ),
                "non_comparable_candidate_ids": list(
                    dict.fromkeys(semantic_non_comparable_ids)
                ),
                "recommended_candidate_ids": [],
            }
        if self.require_control_anchor and not baseline_valid:
            reason = (
                "control_baseline_semantics_conflict"
                if control_semantics.get("status") == "conflict"
                else
                "control_baseline_non_official_outcome"
                if not control_authority_valid
                else
                "control_baseline_pipeline_invalid"
                if not control_pipeline_valid
                else "control_baseline_policy_failed"
            )
            assessment = {
                **assessment,
                "should_stop": True,
                "stop_reason": reason,
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": (
                    "The unchanged-scene control must pass before property "
                    "attribution; no candidate experiment is authorized."
                ),
                "recommended_candidate_ids": [],
            }
        answer = (
            render_query_answer(
                self.user_query,
                assessment,
                records,
                baseline_valid=baseline_valid,
                baseline_stop_reason=assessment["stop_reason"],
            )
            if assessment["should_stop"]
            else None
        )
        return {
            "schema_version": 1,
            "control_template_id": self.control_template,
            "control_required": self.require_control_anchor,
            "control_passed": (
                baseline_valid if self.require_control_anchor else None
            ),
            "query_contract": deepcopy(self.query_contract),
            "assessment": assessment,
            "records": records,
            "open_query_evidence_history": validate_open_query_evidence(
                [record["open_query_evidence"] for record in records]
            ),
            "query_answer": answer,
        }

    def bind_evidence_conditioned_semantic_step(
        self,
        proposal_bundle: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        capabilities: Mapping[str, Any],
        executed_candidate_ids: Sequence[str],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Bind a proposal only if it consumed the current evidence history.

        This is the auditable boundary for cached/provider proposals.  A bundle
        authored before the latest completed round fails its input digest or
        completed-round lineage check instead of being released as the next
        sub-aspect.
        """

        try:
            trusted_capabilities = validate_open_query_capabilities(
                capabilities
            )
            history = _current_planning_evidence(observation)
            if self.require_control_anchor and not history:
                raise ClaimFirstRuntimeError(
                    "control-required Plan Agent needs observed control evidence "
                    "before binding the next sub-aspect"
                )
            trusted_intent = (
                validate_evaluation_intent(evaluation_intent)
                if evaluation_intent is not None
                else None
            )
            lineage = validate_open_query_proposal_lineage(
                proposal_bundle,
                user_query=self.user_query,
                capabilities=trusted_capabilities,
                evidence_history=history,
                evaluation_intent=trusted_intent,
            )
        except (ClaimFirstPlanError, SemanticCoverageError) as exc:
            raise ClaimFirstRuntimeError(str(exc)) from exc
        bound = self.bind_semantic_step(
            proposal_bundle,
            observation,
            executed_template_ids=executed_candidate_ids,
            evaluation_intent=trusted_intent,
        )
        return _attach_planning_lineage(bound, lineage)

    def propose_semantic_step(
        self,
        planner: Any,
        observation: Mapping[str, Any],
        *,
        capabilities: Mapping[str, Any],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Read completed evidence, then author one validated next decision.

        Keeping the provider call inside the session makes the temporal order
        explicit: round evidence is validated first; only then may the Plan
        Agent choose stop or a next semantic concern.  Binding a continuing
        Proposal remains a separate step so an outer execution cap can reject
        it without pre-empting an evidence-backed stop decision.
        """

        assessment = observation.get("assessment")
        if not isinstance(assessment, Mapping):
            raise ClaimFirstRuntimeError(
                "Plan Agent observation has no assessment"
            )
        if (
            assessment.get("should_stop")
            and assessment.get("evidence_sufficient") is not True
        ):
            raise ClaimFirstRuntimeError(
                "cannot propose a semantic step after the query contract stopped"
            )
        if (
            self.require_control_anchor
            and observation.get("control_passed") is not True
        ):
            raise ClaimFirstRuntimeError(
                "cannot propose a property experiment before the control passes"
            )
        history = _current_planning_evidence(observation)
        if self.require_control_anchor and not history:
            raise ClaimFirstRuntimeError(
                "control-required Plan Agent needs observed control evidence "
                "before authoring the next sub-aspect"
            )
        try:
            trusted_capabilities = validate_open_query_capabilities(
                capabilities
            )
            trusted_intent = (
                validate_evaluation_intent(evaluation_intent)
                if evaluation_intent is not None
                else None
            )
        except (ClaimFirstPlanError, SemanticCoverageError) as exc:
            raise ClaimFirstRuntimeError(str(exc)) from exc
        propose = getattr(planner, "propose", None)
        if not callable(propose):
            raise ClaimFirstRuntimeError(
                "Plan Agent must expose a callable propose()"
            )
        try:
            proposal_bundle = propose(
                self.user_query,
                capabilities=trusted_capabilities,
                evidence_history=history,
                evaluation_intent=trusted_intent,
            )
        except Exception as exc:
            if isinstance(exc, ClaimFirstRuntimeError):
                raise
            raise ClaimFirstRuntimeError(
                "evidence-conditioned Plan Agent failed after completed "
                f"rounds {[item['round_id'] for item in history]}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(proposal_bundle, Mapping):
            raise ClaimFirstRuntimeError(
                "Plan Agent returned no proposal bundle"
            )
        try:
            validate_open_query_proposal_lineage(
                proposal_bundle,
                user_query=self.user_query,
                capabilities=trusted_capabilities,
                evidence_history=history,
                evaluation_intent=trusted_intent,
            )
        except ClaimFirstPlanError as exc:
            raise ClaimFirstRuntimeError(str(exc)) from exc
        return deepcopy(dict(proposal_bundle))

    def propose_and_bind_semantic_step(
        self,
        planner: Any,
        observation: Mapping[str, Any],
        *,
        capabilities: Mapping[str, Any],
        executed_candidate_ids: Sequence[str],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Author from completed evidence, then bind exactly one next step."""

        proposal_bundle = self.propose_semantic_step(
            planner,
            observation,
            capabilities=capabilities,
            evaluation_intent=evaluation_intent,
        )
        return self.bind_evidence_conditioned_semantic_step(
            proposal_bundle,
            observation,
            capabilities=capabilities,
            executed_candidate_ids=executed_candidate_ids,
            evaluation_intent=evaluation_intent,
        )

    def bind_semantic_step(
        self,
        proposal_bundle: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        executed_template_ids: Sequence[str],
        evaluation_intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate and bind one semantic Proposal to the generic runtime.

        Retrieval may identify an exact historical artifact, but it is only a
        reuse hint.  It never changes the Proposal into a catalog-authorized
        execution step.  Consequently every production Plan Agent round has
        the same typed Proposal boundary and the generic TaskGen/ToolGen
        materializer decides whether to reuse or generate each requested
        artifact.
        """

        assessment = observation.get("assessment")
        if not isinstance(assessment, Mapping):
            raise ClaimFirstRuntimeError("Plan Agent observation has no assessment")
        raw_proposal = proposal_bundle.get("proposal")
        if not isinstance(raw_proposal, Mapping):
            raise ClaimFirstRuntimeError(
                "Plan Agent proposal bundle has no proposal object"
            )
        try:
            proposal = validate_open_query_plan_proposal(
                raw_proposal,
                has_evidence=bool(observation.get("records")),
            )
        except ClaimFirstPlanError as exc:
            raise ClaimFirstRuntimeError(str(exc)) from exc

        if proposal["action"] == "stop":
            query_answer = observation.get("query_answer")
            if not (
                assessment.get("should_stop") is True
                and assessment.get("evidence_sufficient") is True
                and assessment.get("stop_reason") == "evidence_sufficient"
                and isinstance(query_answer, Mapping)
                and query_answer.get("answered") is True
            ):
                raise ClaimFirstRuntimeError(
                    "Plan Agent stop rejected by QueryContract: completed "
                    "evidence does not yet support an answer to the original "
                    "Query"
                )
            return {
                "schema_version": 2,
                "semantic_proposal_bundle": deepcopy(dict(proposal_bundle)),
                "semantic_needs": {
                    "scene_need": deepcopy(proposal["scene_need"]),
                    "checker_need": deepcopy(proposal["checker_need"]),
                    "rule_tool_need": deepcopy(proposal["rule_tool_need"]),
                    "vqa_tool_need": deepcopy(proposal["vqa_tool_need"]),
                    "task_need": deepcopy(proposal["task_need"]),
                    "tool_need": deepcopy(proposal["tool_need"]),
                },
                "resolution": {
                    "schema_version": 1,
                    "semantic_sub_aspect": None,
                    "resolved_aspect_id": None,
                    "resolved_template_id": None,
                    "resolved_candidate_id": None,
                    "resolution": "query_contract_validated_stop",
                    "hidden": False,
                    "matched_tokens": [],
                    "catalog_was_model_visible": False,
                    "catalog_resolution_error": None,
                    "retrieval_aspect_id": None,
                    "retrieval_template_id": None,
                    "retrieval_resolution": None,
                },
                "query_contract": deepcopy(self.query_contract),
                "query_assessment": deepcopy(dict(assessment)),
                "query_answer": deepcopy(dict(query_answer)),
                "plan_step": {
                    "schema_version": 2,
                    "action": "stop",
                    "aspect_id": None,
                    "candidate_id": None,
                    "execution_mode": "none",
                    "proposal": None,
                    "rationale": proposal["rationale"],
                    "hypothesis": proposal["hypothesis"],
                    "answered_query": True,
                    "claim_verdict": assessment.get("claim_verdict"),
                    "stop_reason": assessment.get("stop_reason"),
                    "next_round": None,
                },
            }

        budget_remaining = assessment.get("budget_remaining")
        may_continue_after_sufficiency = bool(
            assessment.get("should_stop") is True
            and assessment.get("evidence_sufficient") is True
            and assessment.get("stop_reason") == "evidence_sufficient"
            and isinstance(budget_remaining, int)
            and not isinstance(budget_remaining, bool)
            and budget_remaining > 0
        )
        if assessment.get("should_stop") and not may_continue_after_sufficiency:
            raise ClaimFirstRuntimeError(
                "cannot bind a semantic step after the query contract stopped"
            )
        if (
            self.require_control_anchor
            and observation.get("control_passed") is not True
        ):
            raise ClaimFirstRuntimeError(
                "cannot attribute a property before the control passes"
            )
        retrieval_hint: dict[str, Any] | None = None
        retrieval_error: str | None = None
        try:
            retrieval_hint = resolve_semantic_proposal(
                proposal,
                target={"aspects": self.retrieval_aspects},
                executed_template_ids=executed_template_ids,
                control_template=self.control_template,
            )
        except ClaimFirstRuntimeError as catalog_error:
            retrieval_error = str(catalog_error)
        candidate = build_dynamic_experiment_candidate(
            user_query=self.user_query,
            task_name=self.task_name,
            proposal=proposal,
            evaluation_intent=evaluation_intent,
        )
        return self._bind_dynamic_candidate(
            proposal_bundle=proposal_bundle,
            proposal=proposal,
            candidate=candidate,
            executed_candidate_ids=executed_template_ids,
            resolution=(
                "retrieval_hint_then_reuse_or_generate"
                if retrieval_hint is not None
                else "proposal_reuse_or_generate"
            ),
            catalog_resolution_error=retrieval_error,
            retrieval_hint=retrieval_hint,
        )


# Compatibility class name; new callers should use ``PlanAgentSession``.
ClaimFirstRuntimeController = PlanAgentSession


__all__ = [
    "PlanAgentSession",
    "PlanAgentSessionError",
    "ClaimFirstRuntimeController",
    "ClaimFirstRuntimeError",
    "build_claim_first_evidence_record",
    "build_dynamic_experiment_candidate",
    "build_initial_semantic_proposal_bundle",
    "control_template_id",
    "render_query_answer",
    "resolve_concern_candidate_domain",
    "resolve_semantic_proposal",
]
