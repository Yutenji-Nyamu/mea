"""Runtime bridge for claim-first open-Query planning.

``ClaimFirstOpenQueryAgent`` deliberately emits a semantic experiment rather
than an executable catalog step.  This module connects that semantic proposal
to the existing bounded ACT runtime without letting a language model invent
execution details or decide when evidence is sufficient.

The bridge has four explicit responsibilities:

* run one unchanged official-scene control before property attribution;
* derive OpenQueryEvidence and finite-domain candidate evidence directly from
  the runtime-owned EvidencePacket and lightweight artifact paths;
* apply the query-sufficiency contract before accepting a model-authored stop;
* resolve a semantic sub-aspect to one still-legal trusted template only after
  the model has made its claim-first proposal.

This remains a bounded finite-domain protocol.  It is not a statistical
generalization guarantee and does not make the hidden executable catalog part
of the model prompt.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from mea.aspects import public_aspect_ontology
from mea.capability_adapter import resolve_task_adapter

from .claim_first import (
    ClaimFirstPlanError,
    validate_open_query_evidence,
    validate_open_query_plan_proposal,
)
from .evidence_policy import build_evidence_packet, validate_evidence_packet
from .query_contract import (
    assess_query_sufficiency,
    build_query_sufficiency_contract,
    infer_claim_type,
    validate_query_sufficiency_contract,
)


class ClaimFirstRuntimeError(ValueError):
    """Raised when semantic planning cannot be bound to trusted evidence."""


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


def control_template_id(target: Mapping[str, Any]) -> str:
    """Return the trusted official-scene control for a bound task."""

    task_name = _nonempty_text(target.get("task_name"), "target.task_name")
    try:
        adapter = resolve_task_adapter(task_name)
    except ValueError as exc:
        raise ClaimFirstRuntimeError(
            f"claim-first control anchor is not defined for {task_name!r}"
        ) from exc
    template_id = adapter["control_template_id"]
    available = {
        str(item)
        for aspect in target.get("aspects", [])
        if isinstance(aspect, Mapping)
        for item in aspect.get("template_ids", [])
    }
    if template_id not in available:
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
        adapter = resolve_task_adapter(task_name)
    except ValueError:
        return False
    return template_id.strip() == adapter["control_template_id"]


def build_control_anchor_proposal(
    target: Mapping[str, Any],
    user_query: str,
) -> dict[str, Any]:
    """Build the cached first-round proposal consumed by legacy materializers.

    No provider call is needed to choose a control: it is a protocol
    prerequisite rather than an answer to the open Query.
    """

    query = _nonempty_text(user_query, "user_query")
    task_name = _nonempty_text(target.get("task_name"), "target.task_name")
    template_id = control_template_id(target)
    adapter = resolve_task_adapter(task_name)
    planner_kind = adapter["planner_kind"]
    if planner_kind == "model_click_bell_adaptive_v1":
        return {
            "schema_version": 1,
            "task_name": "click_bell",
            "evaluation_goal": (
                "establish_clean_control_before_claim_first_attribution: "
                + query
            ),
            "requested_aspect_ids": [
                "performance.completion_time_stability"
            ],
            "first_aspect_id": "performance.completion_time_stability",
        }
    if planner_kind == "bounded_bbh_v1":
        return {
            "schema_version": 5,
            "task_name": "beat_block_hammer",
            "policy": deepcopy(dict(target["policy"])),
            "evaluation_goal": (
                "establish_clean_control_before_claim_first_attribution: "
                + query
            ),
            "requested_template_ids": [template_id],
            "first_template_id": template_id,
            "max_rounds": int(target["max_rounds"]),
        }
    if planner_kind == "deterministic_official_task":
        return {
            "schema_version": 1,
            "task_name": task_name,
            "evaluation_goal": (
                "establish_clean_control_before_claim_first_attribution: "
                + query
            ),
            "requested_aspect_ids": ["task_execution.official_baseline"],
            "first_aspect_id": "task_execution.official_baseline",
        }
    raise ClaimFirstRuntimeError(
        f"claim-first control proposal is not supported for {task_name!r}"
    )


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
    least two FreeConcern fields *and* in the original Query.  This keeps a
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
    """Bind an online FreeConcern to a finite capability domain.

    FreeConcern is authored before task/capability retrieval and never sees the
    executable catalog.  The runtime may therefore use its semantic fields to
    narrow a trusted capability inventory, but only on an exact or unique
    lexical match.  Broad or tied concerns keep the complete non-control
    domain; they are never silently turned into a one-candidate itinerary.
    """

    if not isinstance(concern, Mapping):
        raise ClaimFirstRuntimeError("FreeConcern must be an object")
    semantic_fields = {
        field: _nonempty_text(concern.get(field), f"FreeConcern.{field}")
        for field in (
            "sub_aspect",
            "hypothesis",
            "requested_variation",
            "measurement_need",
        )
    }
    source_query = _nonempty_text(
        concern.get("source_query"), "FreeConcern.source_query"
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
    if not candidates:
        raise ClaimFirstRuntimeError(
            "bound task has no non-control capability for the FreeConcern"
        )

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

    external_specificity = _catalog_external_specificity(
        semantic_fields,
        source_query=source_query,
    )
    catalog_external = bool(
        selected is None
        and not exact_matches
        and all(
            not item["source_query_matched_tokens"]
            for item in candidates
        )
        and external_specificity["specific"]
    )
    result = {
        "schema_version": 1,
        "decision": (
            "bind_single_aspect"
            if selected is not None
            else "catalog_external"
            if catalog_external
            else "ambiguous"
        ),
        "resolution": (
            "unsupported_or_generation_required"
            if catalog_external
            else resolution
        ),
        "candidate_aspect_ids": (
            [str(selected["aspect_id"])] if selected is not None else None
        ),
        "selected_aspect_id": (
            str(selected["aspect_id"]) if selected is not None else None
        ),
        "selected_template_ids": (
            list(selected["template_ids"]) if selected is not None else []
        ),
        "ranked_aspects": sorted(
            candidates,
            key=lambda item: (
                not bool(item["exact"]),
                -int(item["score"]),
                str(item["aspect_id"]),
            ),
        ),
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
            "the query contract, not the model, owns claim-first stopping"
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
    ClaimFirst runtime depend on a provenance subsystem.
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
    observations = round_summary.get("observations")
    observations = observations if isinstance(observations, Mapping) else {}
    if execution_dir and isinstance(observations.get("aggregate"), Mapping):
        refs.append(
            {
                "kind": "round_aggregate",
                "path": f"{execution_dir}/aggregate_result.json",
            }
        )
    planned_tool = observations.get("planned_tool")
    if (
        execution_dir
        and isinstance(planned_tool, Mapping)
        and planned_tool.get("status") != "skipped"
    ):
        refs.append(
            {
                "kind": "tool_execution",
                "path": f"{execution_dir}/planned_tool/tool_execution.json",
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
    compact: list[dict[str, Any]] = []
    episodes = planned.get("episodes")
    if not isinstance(episodes, list):
        return compact
    for episode in episodes:
        if not isinstance(episode, Mapping):
            continue
        result = episode.get("result")
        if not isinstance(result, Mapping):
            continue
        details = result.get("details")
        details = details if isinstance(details, Mapping) else {}
        compact.append(
            {
                "metric": str(
                    result.get("tool")
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
        )
    return compact


def build_claim_first_evidence_record(
    round_plan: Mapping[str, Any],
    round_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive compact semantic/query evidence from one completed runtime round."""

    if round_plan.get("round_id") != round_summary.get("round_id"):
        raise ClaimFirstRuntimeError("round plan and summary ids disagree")
    packet = validate_evidence_packet(
        build_evidence_packet(
            {"rounds": [deepcopy(dict(round_plan))], "max_rounds": 1},
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
    if outcome_semantics_status == "conflict":
        semantic_outcome = "ambiguous"
        candidate_outcome = "conflict"
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
        else str(round_plan.get("template_id") or sub_aspect)
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
            "The generated checker adds experimental constraints beyond the "
            "official core predicate; its verdict is not official-equivalent."
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
        f"EvidencePacket strength={strength}; policy_success_rate="
        f"{success_rate}; Rule metric={packet['rule']['metric']}; "
        f"outcome_metric={policy_outcome.get('metric')}; "
        f"outcome_authority={policy_outcome.get('authority')}; "
        f"outcome_semantics={outcome_semantics_status}; "
        f"VQA status={packet['vqa']['status']}; "
        f"planned_tool_measurements={tool_summary}."
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
            f"{round_plan.get('template_id')} with complete Rule metric "
            f"{packet['rule']['metric']}; this localizes an observed weakness "
            "but does not establish a causal mechanism."
        )
    candidate = {
        "candidate_id": str(round_plan.get("template_id") or ""),
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
            answer = (
                f"For the finite registered candidate domain, the Query verdict "
                f"is {verdict}."
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
            "Untested finite-domain candidates: " + ", ".join(untested)
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
            "At least one generated checker is an expected semantic extension "
            "of the official core predicate, not an official-equivalent result."
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


class ClaimFirstRuntimeController:
    """Own control gating, query sufficiency, and semantic catalog resolution."""

    def __init__(
        self,
        user_query: str,
        target: Mapping[str, Any],
        *,
        query_contract: Mapping[str, Any] | None = None,
        candidate_aspect_ids: Sequence[str] | None = None,
    ):
        self.user_query = _nonempty_text(user_query, "user_query")
        self.target = deepcopy(dict(target))
        self.control_template = control_template_id(self.target)
        if candidate_aspect_ids is not None:
            allowed_aspects = {
                _nonempty_text(item, "candidate_aspect_ids[]")
                for item in candidate_aspect_ids
            }
            known_aspects = {
                str(aspect.get("aspect_id") or "")
                for aspect in self.target.get("aspects", [])
                if isinstance(aspect, Mapping)
            }
            unknown_aspects = allowed_aspects - known_aspects
            if unknown_aspects:
                raise ClaimFirstRuntimeError(
                    "routed candidate aspects leave the bound task catalog: "
                    f"{sorted(unknown_aspects)}"
                )
            self.target["aspects"] = [
                deepcopy(dict(aspect))
                for aspect in self.target.get("aspects", [])
                if isinstance(aspect, Mapping)
                and (
                    str(aspect.get("aspect_id") or "") in allowed_aspects
                    or self.control_template
                    in {str(item) for item in aspect.get("template_ids", [])}
                )
            ]
        self.template_to_aspect = _template_aspect(self.target)
        candidates = [
            template_id
            for template_id in self.template_to_aspect
            if template_id != self.control_template
        ]
        round_budget = int(self.target.get("max_rounds") or 0) - 1
        if not candidates or round_budget < 1:
            raise ClaimFirstRuntimeError(
                "claim-first runtime needs one control and at least one candidate round"
            )
        if query_contract is None:
            claim_type = infer_claim_type(self.user_query)
            if claim_type == "comparative":
                raise ClaimFirstRuntimeError(
                    "comparative Query requires an explicit preregistered "
                    "query-sufficiency contract with two groups"
                )
            contract = build_query_sufficiency_contract(
                self.user_query,
                candidate_universe=candidates,
                round_budget=round_budget,
                claim_type=claim_type,
            )
        else:
            contract = validate_query_sufficiency_contract(query_contract)
            if set(contract["candidate_universe"]) - set(candidates):
                raise ClaimFirstRuntimeError(
                    "query contract leaves the non-control bound candidate domain"
                )
            if int(contract["round_budget"]) > round_budget:
                raise ClaimFirstRuntimeError(
                    "query contract spends rounds reserved for the control anchor"
                )
        self.query_contract = contract

    def observe(
        self,
        round_plans: Sequence[Mapping[str, Any]],
        round_summaries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Normalize all completed rounds and decide whether execution stops."""

        if not (len(round_plans) == len(round_summaries) and round_plans):
            raise ClaimFirstRuntimeError(
                "completed plans and summaries must be non-empty and aligned"
            )
        records = [
            build_claim_first_evidence_record(plan, summary)
            for plan, summary in zip(round_plans, round_summaries)
        ]
        if records[0]["template_id"] != self.control_template:
            raise ClaimFirstRuntimeError(
                "claim-first property attribution requires the control template first"
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
        candidate_records = records[1:]
        candidate_evidence = [
            deepcopy(record["candidate_evidence"])
            for record in candidate_records
            if record["template_id"] in self.query_contract["candidate_universe"]
        ]
        assessment = assess_query_sufficiency(
            self.query_contract,
            candidate_evidence,
            completed_rounds=len(candidate_records),
        )
        semantic_conflict_ids = [
            record["template_id"]
            for record in candidate_records
            if (
                record["template_id"]
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
        if not baseline_valid:
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
            "control_passed": baseline_valid,
            "query_contract": deepcopy(self.query_contract),
            "assessment": assessment,
            "records": records,
            "open_query_evidence_history": validate_open_query_evidence(
                [record["open_query_evidence"] for record in records]
            ),
            "query_answer": answer,
        }

    def bind_semantic_step(
        self,
        proposal_bundle: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        executed_template_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Validate and resolve a provider/cached semantic next-step bundle."""

        assessment = observation.get("assessment")
        if not isinstance(assessment, Mapping):
            raise ClaimFirstRuntimeError("claim-first observation has no assessment")
        if assessment.get("should_stop"):
            raise ClaimFirstRuntimeError(
                "cannot bind a semantic step after the query contract stopped"
            )
        if observation.get("control_passed") is not True:
            raise ClaimFirstRuntimeError(
                "cannot attribute a property before the control passes"
            )
        raw_proposal = proposal_bundle.get("proposal")
        if not isinstance(raw_proposal, Mapping):
            raise ClaimFirstRuntimeError(
                "claim-first proposal bundle has no proposal object"
            )
        try:
            proposal = validate_open_query_plan_proposal(
                raw_proposal, has_evidence=True
            )
        except ClaimFirstPlanError as exc:
            raise ClaimFirstRuntimeError(str(exc)) from exc
        resolution = resolve_semantic_proposal(
            proposal,
            target=self.target,
            executed_template_ids=executed_template_ids,
            control_template=self.control_template,
        )
        current_aspect = self.template_to_aspect.get(
            str(executed_template_ids[-1])
        )
        return {
            "schema_version": 1,
            "semantic_proposal_bundle": deepcopy(dict(proposal_bundle)),
            "semantic_needs": {
                "task_need": deepcopy(proposal["task_need"]),
                "tool_need": deepcopy(proposal["tool_need"]),
            },
            "resolution": resolution,
            "plan_step": {
                "schema_version": 1,
                "action": (
                    "refine"
                    if resolution["resolved_aspect_id"] == current_aspect
                    else "propose"
                ),
                "aspect_id": resolution["resolved_aspect_id"],
                "template_id": resolution["resolved_template_id"],
                "rationale": proposal["rationale"],
                "answered_query": False,
            },
        }


__all__ = [
    "ClaimFirstRuntimeController",
    "ClaimFirstRuntimeError",
    "build_claim_first_evidence_record",
    "build_control_anchor_proposal",
    "control_template_id",
    "render_query_answer",
    "resolve_concern_candidate_domain",
    "resolve_semantic_proposal",
]
