"""Pure projections used to audit a compact online flagship evaluation."""

from __future__ import annotations

from typing import Any, Mapping


_RUNTIME_DISCOVERY_RESOLUTIONS = frozenset(
    {
        "broad_or_ambiguous",
        "open_world_candidate_discovery_required",
        "generation_required_no_registered_candidate",
    }
)


def _episode_tool_results(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize legacy and current episode Tool-result envelopes."""

    raw_results = episode.get("tool_results")
    if not isinstance(raw_results, list):
        direct_result = episode.get("result")
        raw_results = [direct_result] if isinstance(direct_result, dict) else []
    return [result for result in raw_results if isinstance(result, dict)]


def _valid_query_interpretation_provider_trace(
    provider_trace: Mapping[str, Any],
) -> bool:
    """Accept a direct response or exactly one schema-guided repair."""

    attempt_count = provider_trace.get("attempt_count")
    errors = provider_trace.get("errors")
    return bool(
        provider_trace.get("called") is True
        and isinstance(attempt_count, int)
        and not isinstance(attempt_count, bool)
        and 1 <= attempt_count <= 2
        and isinstance(errors, list)
        and len(errors) == attempt_count - 1
        and all(isinstance(item, str) and item.strip() for item in errors)
    )


def build_compact_flagship_acceptance(
    round_runs: list[dict[str, Any]],
    *,
    global_route_result: Mapping[str, Any] | None,
    claim_first_runtime_state: Mapping[str, Any] | None,
    claim_first_query_answer: Mapping[str, Any] | None = None,
    free_concern_bundle: Mapping[str, Any] | None,
    open_task_resolution: Mapping[str, Any] | None,
    concern_candidate_resolution: Mapping[str, Any] | None,
    history_disabled: bool,
    cli_candidate_hint_used: bool = False,
) -> dict[str, Any]:
    """Project a strict, scoped acceptance for one online 2-3 round run."""

    policy_rollouts = 0
    round_routes: list[str] = []
    semantics_statuses: list[str] = []
    runtime_candidate_ids: list[str] = []
    typed_candidate_completion: dict[str, bool] = {}
    same_bundle_bound_checker_reuse = False
    bound_checker_metric: str | None = None
    bound_checker_module_sha256: str | None = None
    for run in round_runs:
        summary = run.get("round_summary")
        summary = summary if isinstance(summary, Mapping) else {}
        round_routes.append(str(summary.get("route") or ""))
        semantic_execution = summary.get("semantic_need_execution")
        if isinstance(semantic_execution, Mapping) and isinstance(
            semantic_execution.get("candidate_id"), str
        ):
            runtime_candidate_ids.append(
                str(semantic_execution["candidate_id"])
            )
        observations = summary.get("observations")
        observations = observations if isinstance(observations, Mapping) else {}
        implementation_trace = observations.get("implementation_trace")
        if (
            isinstance(semantic_execution, Mapping)
            and isinstance(semantic_execution.get("candidate_id"), str)
            and isinstance(implementation_trace, Mapping)
        ):
            typed_candidate_completion[
                str(semantic_execution["candidate_id"])
            ] = bool(
                implementation_trace.get("relationship") == "direct"
                and implementation_trace.get("coverage_status") == "complete"
            )
        actual_seeds = observations.get("actual_seeds")
        if (
            isinstance(actual_seeds, list)
            and observations.get("execution_backend") != "expert"
        ):
            policy_rollouts += len(actual_seeds)
        semantics = observations.get("outcome_semantics")
        if isinstance(semantics, Mapping) and isinstance(
            semantics.get("status"), str
        ):
            semantics_statuses.append(str(semantics["status"]))

        tool_evaluation = run.get("tool_evaluation")
        if not isinstance(tool_evaluation, Mapping):
            continue
        route_decision = tool_evaluation.get("route_decision")
        route_decision = (
            route_decision if isinstance(route_decision, Mapping) else {}
        )
        route = str(
            tool_evaluation.get("route")
            or route_decision.get("resolved_route")
            or ""
        )
        if route != "bound_child_trusted_checker":
            continue
        metric = route_decision.get("metric")
        validation = tool_evaluation.get("validation")
        validation = validation if isinstance(validation, Mapping) else {}
        source = tool_evaluation.get("source")
        source = source if isinstance(source, Mapping) else {}
        bound_episodes = tool_evaluation.get("episodes")
        bound_episodes = (
            bound_episodes if isinstance(bound_episodes, list) else []
        )
        episode_bindings: list[tuple[str, str]] = []
        episodes_valid = bool(bound_episodes)
        for episode in bound_episodes:
            if (
                not isinstance(episode, Mapping)
                or episode.get("role") != "policy_under_evaluation"
            ):
                episodes_valid = False
                continue
            results = _episode_tool_results(episode)
            if len(results) != 1:
                episodes_valid = False
                continue
            result = results[0]
            details = result.get("details")
            if (
                not isinstance(metric, str)
                or result.get("tool") != metric
                or not isinstance(result.get("value"), bool)
                or not isinstance(details, Mapping)
                or details.get("authority")
                != "llm_generated_python_ast_validated"
                or not isinstance(details.get("module_sha256"), str)
                or len(details["module_sha256"]) != 64
            ):
                episodes_valid = False
                continue
            episode_bindings.append(
                (metric, str(details["module_sha256"]))
            )
        one_binding = set(episode_bindings)
        same_bundle_bound_checker_reuse = bool(
            route_decision.get("provider_called") is False
            and route_decision.get("exact_match") is True
            and validation.get("status") == "passed"
            and validation.get("exact_metric_match") is True
            and source.get("authority")
            == "llm_generated_python_ast_validated"
            and episodes_valid
            and len(one_binding) == 1
        )
        if same_bundle_bound_checker_reuse:
            bound_checker_metric, bound_checker_module_sha256 = next(
                iter(one_binding)
            )

    assessment = (
        claim_first_runtime_state.get("assessment")
        if isinstance(claim_first_runtime_state, Mapping)
        else None
    )
    assessment = assessment if isinstance(assessment, Mapping) else {}
    global_router_provider_calls = (
        global_route_result.get("global_router_provider_calls")
        if isinstance(global_route_result, Mapping)
        else None
    )
    free_provider = (
        free_concern_bundle.get("provider")
        if isinstance(free_concern_bundle, Mapping)
        else None
    )
    free_provider = free_provider if isinstance(free_provider, Mapping) else {}
    online_query_interpretation = bool(
        isinstance(free_concern_bundle, Mapping)
        and free_concern_bundle.get("source")
        in {
            "provider_plan_agent_query_interpretation",
            "provider_catalog_free_concern",
        }
        and _valid_query_interpretation_provider_trace(free_provider)
        and isinstance(open_task_resolution, Mapping)
        and open_task_resolution.get("decision") == "retrieve_and_adapt"
    )
    runtime_bound_route = bool(
        isinstance(global_route_result, Mapping)
        and global_route_result.get("route_source")
        in {
            "runtime_task_checkpoint_binding",
            "runtime_bound_control_handoff",
        }
        and global_route_result.get("provider_called") is False
    )
    exact_catalog_candidate_binding = bool(
        isinstance(concern_candidate_resolution, Mapping)
        and concern_candidate_resolution.get("decision")
        == "bind_single_aspect"
        and concern_candidate_resolution.get("resolution")
        in {
            "exact_query_supported_concern",
            "unique_query_supported_concern",
        }
        and isinstance(
            concern_candidate_resolution.get("candidate_aspect_ids"), list
        )
        and len(concern_candidate_resolution["candidate_aspect_ids"]) == 1
        and concern_candidate_resolution.get("concern_created_before_catalog")
        is True
        and concern_candidate_resolution.get("catalog_was_model_visible")
        is False
        and isinstance(
            concern_candidate_resolution.get("selected_template_ids"), list
        )
        and len(concern_candidate_resolution["selected_template_ids"]) == 1
    )
    runtime_candidate_discovery = bool(
        isinstance(concern_candidate_resolution, Mapping)
        and concern_candidate_resolution.get("decision")
        in {"discover_candidates", "catalog_external"}
        and concern_candidate_resolution.get("resolution")
        in _RUNTIME_DISCOVERY_RESOLUTIONS
        and concern_candidate_resolution.get("concern_created_before_catalog")
        is True
        and concern_candidate_resolution.get("catalog_was_model_visible")
        is False
        and concern_candidate_resolution.get("selected_template_ids") == []
        and len(set(runtime_candidate_ids)) >= 1
    )
    online_query_candidate_binding = bool(
        exact_catalog_candidate_binding or runtime_candidate_discovery
    )
    query_contract = (
        claim_first_runtime_state.get("query_contract")
        if isinstance(claim_first_runtime_state, Mapping)
        else None
    )
    query_contract = query_contract if isinstance(query_contract, Mapping) else {}
    bound_candidate_templates = query_contract.get("candidate_universe")
    observed_candidate_ids = assessment.get("observed_candidate_ids")
    singleton_query_candidate = bool(
        isinstance(observed_candidate_ids, list)
        and len(observed_candidate_ids) == 1
        and (
            (
                exact_catalog_candidate_binding
                and isinstance(bound_candidate_templates, list)
                and observed_candidate_ids[0] in bound_candidate_templates
            )
            or (
                runtime_candidate_discovery
                and observed_candidate_ids[0] in runtime_candidate_ids
            )
        )
    )
    query_candidates_bound = bool(
        isinstance(observed_candidate_ids, list)
        and observed_candidate_ids
        and (
            (
                exact_catalog_candidate_binding
                and isinstance(bound_candidate_templates, list)
                and all(
                    candidate_id in bound_candidate_templates
                    for candidate_id in observed_candidate_ids
                )
            )
            or (
                runtime_candidate_discovery
                and all(
                    candidate_id in runtime_candidate_ids
                    for candidate_id in observed_candidate_ids
                )
            )
        )
    )
    answer = (
        claim_first_query_answer
        if isinstance(claim_first_query_answer, Mapping)
        else {}
    )
    evidence_sufficient = bool(
        assessment.get("evidence_sufficient") is True
        and (not answer or answer.get("answered") is True)
    )
    no_outcome_conflict = "conflict" not in semantics_statuses
    candidate_semantics_scoped = bool(
        len(semantics_statuses) >= 2
        and all(
            status
            in {
                "official_only",
                "expected_semantic_extension",
                "equivalent_agreement",
            }
            for status in semantics_statuses
        )
    )
    control_requirement = query_contract.get("control_requirement")
    # Historical contracts omitted the field and always used an official
    # control.  New production contracts state ``not_required`` explicitly.
    control_required = control_requirement != "not_required"
    method_round_sequence = bool(
        2 <= len(round_routes) <= 3
        and policy_rollouts == len(round_routes)
        and any(route != "official" for route in round_routes)
        and (
            (
                round_routes[0] == "official"
                and all(route != "official" for route in round_routes[1:])
            )
            if control_required
            else all(route != "official" for route in round_routes)
        )
    )
    answer_scope = answer.get("answer_scope")
    answer_semantics_scoped = bool(
        answer_scope == "official_equivalent"
        or (
            "expected_semantic_extension" in semantics_statuses
            and answer_scope == "bounded_experimental_query_semantics"
        )
    )
    decisive_candidate_ids = assessment.get("decisive_candidate_ids")
    decisive_candidate_ids = (
        decisive_candidate_ids
        if isinstance(decisive_candidate_ids, list)
        else []
    )
    typed_execution_complete = bool(
        decisive_candidate_ids
        and all(
            typed_candidate_completion.get(str(candidate_id)) is True
            for candidate_id in decisive_candidate_ids
        )
    )
    candidate_execution_accepted = bool(
        same_bundle_bound_checker_reuse or typed_execution_complete
    )
    accepted = bool(
        online_query_interpretation
        and online_query_candidate_binding
        and query_candidates_bound
        and not cli_candidate_hint_used
        and history_disabled
        and runtime_bound_route
        and global_router_provider_calls == 0
        and method_round_sequence
        and evidence_sufficient
        and no_outcome_conflict
        and candidate_semantics_scoped
        and answer_semantics_scoped
        and candidate_execution_accepted
    )
    return {
        "schema_version": 1,
        "accepted": accepted,
        "execution_entrypoint": "scripts/manipeval_agent.py",
        "history_replay_disabled": history_disabled,
        "online_query_interpretation": online_query_interpretation,
        "query_interpretation_attempt_count": free_provider.get("attempt_count"),
        "query_interpretation_bounded_repair_used": bool(
            online_query_interpretation
            and free_provider.get("attempt_count") == 2
        ),
        "online_query_candidate_binding": online_query_candidate_binding,
        "candidate_binding_mode": (
            "exact_catalog_retrieval"
            if exact_catalog_candidate_binding
            else "online_runtime_discovery"
            if runtime_candidate_discovery
            else "unresolved"
        ),
        "cli_candidate_hint_used": cli_candidate_hint_used,
        "candidate_domain_resolution": (
            concern_candidate_resolution.get("resolution")
            if isinstance(concern_candidate_resolution, Mapping)
            else None
        ),
        "candidate_aspect_ids": (
            concern_candidate_resolution.get("candidate_aspect_ids")
            if isinstance(concern_candidate_resolution, Mapping)
            else None
        ),
        "bound_candidate_templates": bound_candidate_templates,
        "singleton_query_candidate": singleton_query_candidate,
        "query_candidates_bound": query_candidates_bound,
        "runtime_bound_route": runtime_bound_route,
        "global_router_provider_calls": global_router_provider_calls,
        # Retained as a compatibility projection for historical readers.
        "act_rollouts": policy_rollouts,
        "required_act_rollouts": 2,
        "accepted_act_rollout_range": [2, 3],
        "policy_rollouts": policy_rollouts,
        "accepted_policy_rollout_range": [2, 3],
        "control_requirement": control_requirement,
        "control_requirement_satisfied": method_round_sequence,
        "round_routes": round_routes,
        "stop_reason": answer.get("stop_reason") or assessment.get("stop_reason"),
        "evidence_sufficient": evidence_sufficient,
        "outcome_semantics_statuses": list(dict.fromkeys(semantics_statuses)),
        "answer_scope": answer_scope,
        "same_bundle_bound_checker_reuse": same_bundle_bound_checker_reuse,
        "typed_execution_complete": typed_execution_complete,
        "candidate_execution_accepted": candidate_execution_accepted,
        "bound_checker_metric": bound_checker_metric,
        "bound_checker_module_sha256": bound_checker_module_sha256,
        "cross_query_registry_reuse_established": False,
    }


# Historical tests and immutable evidence readers may retain the old helper
# name, but new acceptance artifacts use Query interpretation terminology.
_valid_free_concern_provider_trace = (
    _valid_query_interpretation_provider_trace
)
