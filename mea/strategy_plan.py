"""Zero-ACT command planning for a matched fixed/dynamic micro-pilot.

This layer converts a validated evidence preregistration into two auditable
``manipeval_agent.py`` argv vectors and a post-hoc ``strategy_comparison``
configuration.  It never invokes those commands.  The existing strict
artifact comparator remains the authority after both evaluations finish.
"""

from __future__ import annotations

import re
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

from mea.evidence_manifest import (
    EvidenceManifestError,
    canonical_sha256,
    read_repo_json,
    validate_evidence_manifest,
)
from mea.providers import available_model_profiles
from mea.strategy_comparison import StrategyComparisonError, compare_fixed_dynamic
from mea.planner import (
    build_act_catalog,
    catalog_task,
    route_to_planner_proposal,
    validate_route_selection,
)


class StrategyPlanError(RuntimeError):
    """Raised when a matched command plan cannot be built safely."""


_PLAN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]{0,95}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise StrategyPlanError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StrategyPlanError(f"{label} must be a non-empty trimmed string")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise StrategyPlanError(f"{label} must be a single safe command argument")
    return value


def _optional_registry_dir(
    value: Any, *, label: str, registered_sources: set[str]
) -> str | None:
    if value is None:
        return None
    path = _text(value, label=label)
    if "\\" in path or ":" in path:
        raise StrategyPlanError(f"{label} must be a POSIX repo-relative directory")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise StrategyPlanError(f"{label} must stay inside the repository")
    canonical = pure.as_posix()
    if canonical != path:
        raise StrategyPlanError(f"{label} is not a canonical path")
    index = f"{canonical}/index.json"
    if index not in registered_sources:
        raise StrategyPlanError(
            f"{label}/index.json is not hash-pinned as a source artifact"
        )
    return canonical


def _agent_argv(
    *,
    python_executable: str,
    query: str,
    evaluation_id: str,
    planning_policy: str,
    start_seed: int,
    generated_rounds: int,
    telemetry_profile: str,
    model_profile: str,
    gpu: int,
    reviewed_tool_registry: str | None,
    reviewed_vqa_registry: str | None,
    evidence_manifest: str,
    command_plan: str,
    registered_route: str,
) -> list[str]:
    task_profile = (
        "fixed_suite"
        if planning_policy == "fixed_predeclared_v1"
        else "adaptive_properties"
    )
    command = [
        python_executable,
        "scripts/manipeval_agent.py",
        "--repo-root",
        ".",
        "--request",
        query,
        "--evaluation-id",
        evaluation_id,
        "--task-name",
        "click_bell",
        "--task-profile",
        task_profile,
        "--planning-policy",
        planning_policy,
        "--generated-rounds",
        str(generated_rounds),
        "--execution-backend",
        "act",
        "--start-seed",
        str(start_seed),
        "--num-episodes",
        "1",
        "--telemetry-profile",
        telemetry_profile,
        "--model-profile",
        model_profile,
        "--gpu",
        str(gpu),
        "--tool-recovery-max-restarts",
        "0",
        "--round-recovery-max-restarts",
        "0",
        "--no-history",
        "--evidence-manifest",
        evidence_manifest,
        "--command-plan",
        command_plan,
        "--registered-route",
        registered_route,
        "--registered-strategy",
        planning_policy,
    ]
    if reviewed_tool_registry is not None:
        command.extend(["--reviewed-tool-registry", reviewed_tool_registry])
    if reviewed_vqa_registry is not None:
        command.extend(["--reviewed-vqa-registry", reviewed_vqa_registry])
    return command


def _build_registered_route(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    catalog = build_act_catalog(root)
    aspects = list(manifest["candidate_suite"]["aspect_ids"])
    selection = validate_route_selection(
        {
            "schema_version": 2,
            "decision": "route",
            "task_name": "click_bell",
            "task_profile": "adaptive_properties",
            "evaluation_goal": manifest["query"]["text"],
            "requested_aspect_ids": aspects,
            "first_aspect_id": aspects[0],
            "unsupported_capabilities": [],
        },
        catalog,
    )
    routed = route_to_planner_proposal(selection, catalog)
    route: dict[str, Any] = {
        "schema_version": 1,
        "route_type": "deterministic_preregistered_click_bell_v1",
        "provider_called": False,
        "registration_id": manifest["registration_id"],
        "evidence_manifest_payload_sha256": manifest["integrity"][
            "canonical_payload_sha256"
        ],
        "trusted_contract": dict(manifest["trusted_contract"]),
        "query_sha256": manifest["query"]["sha256"],
        "candidate_suite": list(manifest["candidate_suite"]["template_ids"]),
        "candidate_suite_sha256": manifest["candidate_suite"]["sha256"],
        "selection": selection,
        "routed_planner": routed,
    }
    route["integrity"] = {
        "algorithm": "sha256",
        "canonical_payload_sha256": canonical_sha256(route),
    }
    validate_registered_route(root, manifest, route)
    return route


def validate_registered_route(
    repo_root: str | Path,
    manifest: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    if not isinstance(route, Mapping):
        raise StrategyPlanError("registered route must be an object")
    expected_keys = {
        "schema_version",
        "route_type",
        "provider_called",
        "registration_id",
        "evidence_manifest_payload_sha256",
        "trusted_contract",
        "query_sha256",
        "candidate_suite",
        "candidate_suite_sha256",
        "selection",
        "routed_planner",
        "integrity",
    }
    _exact_keys(route, expected_keys, "registered route")
    if (
        route["schema_version"] != 1
        or route["route_type"] != "deterministic_preregistered_click_bell_v1"
        or route["provider_called"] is not False
    ):
        raise StrategyPlanError("unsupported registered route contract")
    payload = dict(route)
    integrity = payload.pop("integrity")
    if not isinstance(integrity, Mapping) or set(integrity) != {
        "algorithm",
        "canonical_payload_sha256",
    }:
        raise StrategyPlanError("invalid registered route integrity")
    if (
        integrity.get("algorithm") != "sha256"
        or integrity.get("canonical_payload_sha256") != canonical_sha256(payload)
    ):
        raise StrategyPlanError("registered route hash mismatch")
    if (
        route["registration_id"] != manifest["registration_id"]
        or route["evidence_manifest_payload_sha256"]
        != manifest["integrity"]["canonical_payload_sha256"]
        or route["trusted_contract"] != manifest["trusted_contract"]
        or route["query_sha256"] != manifest["query"]["sha256"]
        or route["candidate_suite"]
        != manifest["candidate_suite"]["template_ids"]
        or route["candidate_suite_sha256"]
        != manifest["candidate_suite"]["sha256"]
    ):
        raise StrategyPlanError("registered route differs from evidence manifest")
    catalog = build_act_catalog(root)
    if catalog["catalog_sha256"] != manifest["trusted_contract"]["catalog_sha256"]:
        raise StrategyPlanError("registered route trusted catalog changed")
    selection = validate_route_selection(route["selection"], catalog)
    if (
        selection["task_name"] != "click_bell"
        or selection["task_profile"] != "adaptive_properties"
        or selection["evaluation_goal"] != manifest["query"]["text"]
        or selection["requested_aspect_ids"]
        != manifest["candidate_suite"]["aspect_ids"]
        or selection["first_aspect_id"]
        != manifest["candidate_suite"]["first_aspect_id"]
        or selection["unsupported_capabilities"] != []
    ):
        raise StrategyPlanError("registered route selection drifted from preregistration")
    routed = route_to_planner_proposal(selection, catalog)
    if route["routed_planner"] != routed:
        raise StrategyPlanError("registered routed planner proposal changed")
    task = catalog_task(catalog, "click_bell")
    aspect_map = {
        item["aspect_id"]: list(item["template_ids"])
        for item in task["aspects"]
    }
    expanded = [
        template
        for aspect in selection["requested_aspect_ids"]
        for template in aspect_map[aspect]
    ]
    if expanded != manifest["candidate_suite"]["template_ids"]:
        raise StrategyPlanError("registered route aspect expansion/order changed")
    return dict(route)


def build_matched_strategy_plan(
    repo_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return an auditable command plan while executing no provider or policy."""

    root = Path(repo_root).expanduser().resolve()
    if not isinstance(config, Mapping):
        raise StrategyPlanError("strategy plan config must be an object")
    _exact_keys(
        config,
        {
            "schema_version",
            "plan_id",
            "evidence_manifest",
            "task_name",
            "model_profile",
            "python_executable",
            "gpu",
            "reviewed_tool_registry",
            "reviewed_vqa_registry",
        },
        "strategy plan config",
    )
    if config["schema_version"] != 1:
        raise StrategyPlanError("strategy plan config schema_version must be 1")
    plan_id = _text(config["plan_id"], label="plan_id")
    if not _PLAN_ID.fullmatch(plan_id):
        raise StrategyPlanError("plan_id must use only letters, digits, and underscores")
    if config["task_name"] != "click_bell":
        raise StrategyPlanError("the matched strategy micro-pilot currently supports click_bell only")
    model_profile = _text(config["model_profile"], label="model_profile")
    if model_profile not in available_model_profiles():
        raise StrategyPlanError(f"unknown model_profile: {model_profile}")
    python_executable = _text(config["python_executable"], label="python_executable")
    gpu = config["gpu"]
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise StrategyPlanError("gpu must be a non-negative integer")

    evidence_path = _text(config["evidence_manifest"], label="evidence_manifest")
    try:
        manifest = read_repo_json(root, evidence_path, label="evidence manifest")
        validation = validate_evidence_manifest(root, manifest)
    except EvidenceManifestError as exc:
        raise StrategyPlanError(str(exc)) from exc
    registered_sources = {
        item["path"] for item in manifest["source_artifacts"]["files"]
    }
    reviewed_tool_registry = _optional_registry_dir(
        config["reviewed_tool_registry"],
        label="reviewed_tool_registry",
        registered_sources=registered_sources,
    )
    reviewed_vqa_registry = _optional_registry_dir(
        config["reviewed_vqa_registry"],
        label="reviewed_vqa_registry",
        registered_sources=registered_sources,
    )

    candidates = list(manifest["candidate_suite"]["template_ids"])
    if len(candidates) > 5:
        raise StrategyPlanError("candidate suite exceeds the Agent round budget")
    schedule = list(manifest["sample_schedule"]["entries"])
    seeds = {item["seed"] for item in schedule}
    if len(seeds) != 1:
        raise StrategyPlanError(
            "the current Agent CLI requires the same N=1 start seed for every candidate"
        )
    start_seed = next(iter(seeds))
    query = manifest["query"]["text"]
    telemetry_profile = manifest["telemetry"]["profile_id"]
    fixed_id = f"eval_{plan_id}_fixed"
    dynamic_id = f"eval_{plan_id}_dynamic"
    command_plan_path = f"mea/validation_runs/{plan_id}/command_plan.json"
    registered_route_path = f"mea/validation_runs/{plan_id}/registered_route.json"
    registered_route = _build_registered_route(root, manifest)

    fixed_argv = _agent_argv(
        python_executable=python_executable,
        query=query,
        evaluation_id=fixed_id,
        planning_policy="fixed_predeclared_v1",
        start_seed=start_seed,
        generated_rounds=len(candidates),
        telemetry_profile=telemetry_profile,
        model_profile=model_profile,
        gpu=gpu,
        reviewed_tool_registry=reviewed_tool_registry,
        reviewed_vqa_registry=reviewed_vqa_registry,
        evidence_manifest=evidence_path,
        command_plan=command_plan_path,
        registered_route=registered_route_path,
    )
    dynamic_argv = _agent_argv(
        python_executable=python_executable,
        query=query,
        evaluation_id=dynamic_id,
        planning_policy="dynamic_evidence_v1",
        start_seed=start_seed,
        generated_rounds=len(candidates),
        telemetry_profile=telemetry_profile,
        model_profile=model_profile,
        gpu=gpu,
        reviewed_tool_registry=reviewed_tool_registry,
        reviewed_vqa_registry=reviewed_vqa_registry,
        evidence_manifest=evidence_path,
        command_plan=command_plan_path,
        registered_route=registered_route_path,
    )
    comparison_config = {
        "schema_version": 1,
        "fixed_evaluation_dir": f"mea/evaluation_runs/{fixed_id}",
        "dynamic_evaluation_dir": f"mea/evaluation_runs/{dynamic_id}",
    }
    comparison_argv = [
        python_executable,
        "scripts/manipeval_compare_registered_strategies.py",
        "--repo-root",
        ".",
        "--command-plan",
        f"mea/validation_runs/{plan_id}/command_plan.json",
        "--output-dir",
        f"mea/validation_runs/{plan_id}/comparison",
    ]

    plan: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "matched_act_fixed_dynamic_command_plan_v1",
        "plan_id": plan_id,
        "execution_status": "planned_not_started",
        "act_rollouts_started": 0,
        "provider_calls_started": 0,
        "paper_table_eligible": False,
        "claim_scope": "table1_efficiency_mechanism_plumbing_only",
        "table2_consistency": None,
        "table2_unavailable_reason": "n1_has_no_trial_distribution",
        "evidence": {
            "manifest_path": evidence_path,
            "manifest_payload_sha256": validation["manifest_payload_sha256"],
            "serialized_manifest_sha256": canonical_sha256(manifest),
            "registration_id": manifest["registration_id"],
        },
        "registered_route": {
            "path": registered_route_path,
            "serialized_sha256": canonical_sha256(registered_route),
            "payload": registered_route,
        },
        "identity": {
            "task_name": "click_bell",
            "query_sha256": manifest["query"]["sha256"],
            "candidate_suite": candidates,
            "candidate_suite_sha256": manifest["candidate_suite"]["sha256"],
            "base_commit": manifest["base_commit"],
            "checkpoint_setting": manifest["checkpoint"]["setting"],
            "expert_data_num": manifest["expert_data_num"],
            "checkpoint_file_set_sha256": manifest["checkpoint"]["file_set_sha256"],
            "telemetry_profile": telemetry_profile,
            "telemetry_profile_sha256": manifest["telemetry"]["profile_sha256"],
            "start_seed": start_seed,
            "num_episodes_per_candidate": 1,
        },
        "schedule": {
            "registered_sample_universe": schedule,
            "fixed_execution": "all registered candidates in frozen order",
            "dynamic_execution": "registered candidates are an upper bound; evidence may stop early",
            "fixed_max_act_rollouts": len(candidates),
            "dynamic_max_act_rollouts": len(candidates),
            "pair_max_act_rollouts": len(candidates) * 2,
        },
        "strategies": {
            "fixed_predeclared_v1": {
                "evaluation_id": fixed_id,
                "argv": fixed_argv,
                "argv_sha256": canonical_sha256(fixed_argv),
            },
            "dynamic_evidence_v1": {
                "evaluation_id": dynamic_id,
                "argv": dynamic_argv,
                "argv_sha256": canonical_sha256(dynamic_argv),
            },
        },
        "posthoc": {
            "comparison_module": "mea.strategy_comparison.compare_fixed_dynamic",
            "comparison_config": comparison_config,
            "comparison_argv": comparison_argv,
            "required_checks": [
                "validate the evidence manifest immediately before and after execution",
                "require identical query, route, candidate suite, base commit, ACT policy, and telemetry",
                "reject dynamic samples outside the fixed candidate universe",
            ],
        },
        "limitations": [
            "Generating this plan executes neither command and starts zero ACT rollouts.",
            "Both commands consume one frozen validated route; task-specific dynamic decisions may occur only after rollout evidence.",
            "This tests a Table 1 efficiency mechanism, not the paper's full standard-benchmark result.",
            "N=1 cannot produce the paper Table 2 consistency statistic.",
        ],
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def _validate_command_plan_hash(command_plan: Mapping[str, Any]) -> str:
    recorded = command_plan.get("plan_sha256")
    payload = dict(command_plan)
    payload.pop("plan_sha256", None)
    if recorded != canonical_sha256(payload):
        raise StrategyPlanError("command plan hash mismatch")
    if (
        command_plan.get("schema_version") != 1
        or command_plan.get("protocol")
        != "matched_act_fixed_dynamic_command_plan_v1"
        or command_plan.get("act_rollouts_started") != 0
    ):
        raise StrategyPlanError("unsupported or already-executed command plan")
    return str(recorded)


def _registration_identity(
    manifest: Mapping[str, Any],
    command_plan: Mapping[str, Any],
    route: Mapping[str, Any],
    strategy: str,
) -> dict[str, Any]:
    strategies = command_plan.get("strategies")
    if not isinstance(strategies, Mapping) or strategy not in strategies:
        raise StrategyPlanError(f"strategy is not registered: {strategy}")
    entry = strategies[strategy]
    if not isinstance(entry, Mapping):
        raise StrategyPlanError("registered strategy entry must be an object")
    evaluation_id = entry.get("evaluation_id")
    if not isinstance(evaluation_id, str):
        raise StrategyPlanError("registered strategy has no evaluation id")
    return {
        "schema_version": 1,
        "registration_id": manifest["registration_id"],
        "evidence_manifest_payload_sha256": manifest["integrity"][
            "canonical_payload_sha256"
        ],
        "command_plan_sha256": command_plan["plan_sha256"],
        "registered_route_sha256": route["integrity"][
            "canonical_payload_sha256"
        ],
        "checkpoint_file_set_sha256": manifest["checkpoint"][
            "file_set_sha256"
        ],
        "source_artifact_file_set_sha256": manifest["source_artifacts"][
            "file_set_sha256"
        ],
        "base_commit": manifest["base_commit"],
        "candidate_suite_sha256": manifest["candidate_suite"]["sha256"],
        "trusted_catalog_sha256": manifest["trusted_contract"]["catalog_sha256"],
        "trusted_template_contract_sha256": manifest["trusted_contract"][
            "template_contract_sha256"
        ],
        "strategy": strategy,
        "expected_evaluation_id": evaluation_id,
        "expected_child_run_prefix": f"run_{evaluation_id.removeprefix('eval_')}_",
    }


def load_registered_execution(
    repo_root: str | Path,
    *,
    evidence_manifest_path: str,
    command_plan_path: str,
    registered_route_path: str,
    strategy: str,
    evaluation_id: str,
    observed_argv: list[str] | None = None,
) -> dict[str, Any]:
    """Fail-closed registered Agent preflight, before provider/simulator work."""

    root = Path(repo_root).expanduser().resolve()
    try:
        manifest = read_repo_json(
            root, evidence_manifest_path, label="evidence manifest"
        )
        validation = validate_evidence_manifest(root, manifest)
        command_plan = read_repo_json(root, command_plan_path, label="command plan")
        plan_hash = _validate_command_plan_hash(command_plan)
        route = read_repo_json(root, registered_route_path, label="registered route")
    except EvidenceManifestError as exc:
        raise StrategyPlanError(str(exc)) from exc
    validate_registered_route(root, manifest, route)

    evidence = command_plan.get("evidence")
    registered = command_plan.get("registered_route")
    if not isinstance(evidence, Mapping) or not isinstance(registered, Mapping):
        raise StrategyPlanError("command plan is missing evidence/route binding")
    if (
        evidence.get("manifest_path") != evidence_manifest_path
        or evidence.get("manifest_payload_sha256")
        != validation["manifest_payload_sha256"]
        or evidence.get("serialized_manifest_sha256") != canonical_sha256(manifest)
        or registered.get("path") != registered_route_path
        or registered.get("serialized_sha256") != canonical_sha256(route)
        or registered.get("payload") != route
    ):
        raise StrategyPlanError("command plan evidence/registered route binding changed")
    if command_plan_path != f"mea/validation_runs/{command_plan.get('plan_id')}/command_plan.json":
        raise StrategyPlanError("command plan path differs from registered path")
    strategies = command_plan.get("strategies")
    if not isinstance(strategies, Mapping) or strategy not in strategies:
        raise StrategyPlanError("unknown registered strategy")
    entry = strategies[strategy]
    if not isinstance(entry, Mapping) or entry.get("evaluation_id") != evaluation_id:
        raise StrategyPlanError("evaluation id differs from registered strategy")
    argv = entry.get("argv")
    if (
        not isinstance(argv, list)
        or entry.get("argv_sha256") != canonical_sha256(argv)
        or len(argv) < 2
    ):
        raise StrategyPlanError("registered strategy argv hash mismatch")
    if observed_argv is not None and list(observed_argv) != argv[1:]:
        raise StrategyPlanError("Agent argv differs from exact registered argv")
    identity = _registration_identity(manifest, command_plan, route, strategy)
    if identity["command_plan_sha256"] != plan_hash:
        raise StrategyPlanError("registration command plan hash mismatch")
    return {
        "manifest": manifest,
        "command_plan": command_plan,
        "route": route,
        "registration_identity": identity,
        "validated_proposal": route["routed_planner"]["proposal"],
        "expected_candidate_suite": list(
            manifest["candidate_suite"]["template_ids"]
        ),
    }


def compare_registered_strategies(
    repo_root: str | Path, command_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the existing artifact comparator and bind its output to preregistration."""

    root = Path(repo_root).expanduser().resolve()
    if not isinstance(command_plan, Mapping):
        raise StrategyPlanError("command plan must be an object")
    _validate_command_plan_hash(command_plan)

    evidence = command_plan.get("evidence")
    posthoc = command_plan.get("posthoc")
    if not isinstance(evidence, Mapping) or not isinstance(posthoc, Mapping):
        raise StrategyPlanError("command plan is missing evidence/posthoc identity")
    try:
        manifest = read_repo_json(
            root, str(evidence.get("manifest_path") or ""), label="evidence manifest"
        )
        validation = validate_evidence_manifest(root, manifest)
    except EvidenceManifestError as exc:
        raise StrategyPlanError(str(exc)) from exc
    if (
        evidence.get("manifest_payload_sha256")
        != validation["manifest_payload_sha256"]
        or evidence.get("serialized_manifest_sha256") != canonical_sha256(manifest)
    ):
        raise StrategyPlanError("command plan points to a different evidence manifest")

    registered_route = command_plan.get("registered_route")
    if not isinstance(registered_route, Mapping):
        raise StrategyPlanError("command plan has no registered route")
    try:
        route = read_repo_json(
            root, str(registered_route.get("path") or ""), label="registered route"
        )
    except EvidenceManifestError as exc:
        raise StrategyPlanError(str(exc)) from exc
    validate_registered_route(root, manifest, route)
    if (
        registered_route.get("serialized_sha256") != canonical_sha256(route)
        or registered_route.get("payload") != route
    ):
        raise StrategyPlanError("command plan registered route binding changed")

    comparison_config = posthoc.get("comparison_config")
    if not isinstance(comparison_config, Mapping):
        raise StrategyPlanError("command plan has no comparison config")
    strategies = command_plan.get("strategies")
    if not isinstance(strategies, Mapping):
        raise StrategyPlanError("command plan has no strategies")
    expected_dirs = {
        "fixed_evaluation_dir": (
            "mea/evaluation_runs/"
            + str(strategies["fixed_predeclared_v1"]["evaluation_id"])
        ),
        "dynamic_evaluation_dir": (
            "mea/evaluation_runs/"
            + str(strategies["dynamic_evidence_v1"]["evaluation_id"])
        ),
    }
    if dict(comparison_config) != {"schema_version": 1, **expected_dirs}:
        raise StrategyPlanError("comparison directories differ from registered evaluation ids")
    try:
        comparison = compare_fixed_dynamic(root, comparison_config)
    except StrategyComparisonError as exc:
        raise StrategyPlanError(str(exc)) from exc

    identity = comparison["identity"]
    policy = identity.get("policy")
    expected_identity = {
        "task_name": "click_bell",
        "base_commit": manifest["base_commit"],
        "candidate_suite": manifest["candidate_suite"]["template_ids"],
        "candidate_suite_sha256": manifest["candidate_suite"]["sha256"],
        "user_request_sha256": manifest["query"]["sha256"],
        "telemetry_profile": manifest["telemetry"]["profile_id"],
    }
    mismatches = [
        field
        for field, expected in expected_identity.items()
        if identity.get(field) != expected
    ]
    if mismatches:
        raise StrategyPlanError(
            f"comparison differs from preregistered identity: {mismatches}"
        )
    if (
        not isinstance(policy, Mapping)
        or str(policy.get("name", "")).casefold() != "act"
        or policy.get("checkpoint_setting") != manifest["checkpoint"]["setting"]
    ):
        raise StrategyPlanError("comparison ACT checkpoint setting differs from preregistration")

    for strategy in ("fixed_predeclared_v1", "dynamic_evidence_v1"):
        expected_registration = _registration_identity(
            manifest, command_plan, route, strategy
        )
        actual_registration = comparison["strategies"][strategy].get(
            "registration_identity"
        )
        if actual_registration != expected_registration:
            raise StrategyPlanError(
                f"{strategy} run registration identity differs from command plan"
            )

    schedule = manifest["sample_schedule"]["entries"]
    expected_by_strategy = {
        strategy: [
            (entry["variant_id"], entry["seed"])
            for entry in schedule
            if entry["strategy"] == strategy
        ]
        for strategy in ("fixed_predeclared_v1", "dynamic_evidence_v1")
    }
    fixed_samples = [
        (item["variant_id"], item["seed"])
        for item in comparison["strategies"]["fixed_predeclared_v1"]["samples"]
    ]
    dynamic_samples = [
        (item["variant_id"], item["seed"])
        for item in comparison["strategies"]["dynamic_evidence_v1"]["samples"]
    ]
    if fixed_samples != expected_by_strategy["fixed_predeclared_v1"]:
        raise StrategyPlanError("fixed samples differ from preregistered N=1 schedule")
    expected_dynamic = expected_by_strategy["dynamic_evidence_v1"]
    if dynamic_samples != expected_dynamic[: len(dynamic_samples)]:
        raise StrategyPlanError(
            "dynamic samples must be an ordered prefix of the preregistered N=1 schedule"
        )

    return {
        "schema_version": 1,
        "protocol": "registered_" + comparison["protocol"],
        "status": "passed",
        "paper_table_eligible": False,
        "claim_scope": "table1_efficiency_mechanism_facing_n1_micro_pilot",
        "table2_consistency": None,
        "table2_unavailable_reason": "n1_has_no_trial_distribution",
        "registered_identity_match": True,
        "evidence": {
            "registration_id": manifest["registration_id"],
            "manifest_payload_sha256": validation["manifest_payload_sha256"],
            "checkpoint_file_set_sha256": manifest["checkpoint"]["file_set_sha256"],
            "source_artifact_file_set_sha256": manifest["source_artifacts"]["file_set_sha256"],
        },
        "comparison": comparison,
        "limitations": [
            "The evidence manifest was validated against current files before comparison.",
            "Checkpoint bytes are preregistered and revalidated, but legacy episode artifacts do not embed their content hash.",
            "N=1 cannot produce the paper Table 2 consistency statistic.",
        ],
    }


__all__ = [
    "StrategyPlanError",
    "build_matched_strategy_plan",
    "compare_registered_strategies",
    "load_registered_execution",
    "validate_registered_route",
]
