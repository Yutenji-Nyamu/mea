"""Query interpretation and Proposal binding for the Plan Agent.

This module owns the semantic boundary between an open user Query and a typed
Proposal. Retrieval is advisory: it may identify reusable artifacts, but it
never restricts what the Plan Agent is allowed to propose.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.aspects import public_aspect_ontology
from mea.artifact_retrieval_index import resolve_task_retrieval_index

from .claim_first import validate_open_query_plan_proposal
from .experiment_candidate import build_experiment_candidate
from .open_task_resolver import (
    validate_free_concern,
    validate_free_concern_experiment_needs,
)
from .plan_agent_errors import ClaimFirstRuntimeError
from .policy_task_binding import (
    PolicyTaskBindingError,
    policy_task_binding_from_target,
)
from .semantic_coverage import build_evaluation_intent, validate_evaluation_intent

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
    official_success_reuse: bool = False,
) -> dict[str, Any]:
    """Bind one semantic proposal without rewriting its first-candidate intent.

    The provider owns the four independent generation/tool requirements.  For
    the first Query-derived candidate, the Query interpretation owns
    the exact change, hypothesis, preserved conditions, and observation text.
    Later evidence-driven pivots receive a fresh per-candidate intent derived
    directly from their proposal, so every round carries the same traceable
    experiment contract.
    """

    if not isinstance(official_success_reuse, bool):
        raise ClaimFirstRuntimeError("official_success_reuse must be bool")
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
                    if official_success_reuse
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
    experiment_needs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind an online Query interpretation to a finite capability domain.

    Query interpretation is authored before task/capability retrieval and never sees the
    executable catalog.  Its typed experiment needs take precedence over
    prose matching when they say the unchanged official execution already
    supplies the required evidence.  Otherwise the runtime may use semantic
    fields to narrow a trusted capability inventory, but only on an exact or
    unique lexical match.  Broad or tied concerns keep the complete
    non-control domain and explicitly ask the Planner to discover the most
    informative first candidate.  Ambiguity is therefore a planning state,
    not an admission failure.
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
    trusted_needs = (
        validate_free_concern_experiment_needs(experiment_needs)
        if experiment_needs is not None
        else None
    )
    if (
        trusted_needs is not None
        and trusted_needs["scene_need"]["required"] is False
        and trusted_needs["checker_need"]["required"] is False
    ):
        # Typed work needs are the Plan Agent's machine-readable artifact
        # contract.  When neither scene nor checker work is requested, the
        # executable experiment is the unchanged official task plus the
        # requested Rule/VQA observation.  Do not reinterpret prose tokens as
        # a request for a generated catalog-external scene.
        # Runtime-bound schema-less tasks deliberately have no ``aspects``
        # menu. ``control_template_id`` is already the trusted task binding's
        # official execution identity, so it is also the neutral semantic
        # aspect for this no-TaskGen experiment.
        rule_description = str(
            trusted_needs["rule_tool_need"].get("description") or ""
        ).casefold()
        official_success_reuse = bool(
            trusted_needs["rule_tool_need"]["required"]
            and "official" in rule_description
            and "check_success" in rule_description
        )
        return {
            "schema_version": 1,
            "decision": "bind_single_aspect",
            "resolution": "official_execution_from_typed_needs",
            "candidate_aspect_ids": [control_template],
            "selected_aspect_id": control_template,
            "selected_template_ids": [control_template],
            "ranked_aspects": [],
            "concern_created_before_catalog": True,
            "catalog_was_model_visible": False,
            "concern": deepcopy(dict(concern)),
            "experiment_needs": deepcopy(trusted_needs),
            "taskgen_required": False,
            "official_success_reuse": official_success_reuse,
            "execution_authorized": True,
        }
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

__all__ = [
    "build_dynamic_experiment_candidate",
    "build_initial_semantic_proposal_bundle",
    "control_template_id",
    "resolve_concern_candidate_domain",
    "resolve_semantic_proposal",
]
