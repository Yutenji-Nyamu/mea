"""Minimal cross-task portfolio planning and cached evidence synthesis.

The existing global query router intentionally selects one task.  This module
adds a thin parent layer without weakening that contract: one parent query is
bound to exactly the two currently trusted ACT tasks, while each child remains
an ordinary ``manipeval_agent.py`` evaluation.  The parent either emits inert,
exact child argv plans or audits explicitly named completed children.

Cached synthesis never calls a provider, simulator, or policy.  In particular,
``pipeline_passed`` is reported as evidence-chain health and is never promoted
to an ACT policy outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from mea.planner.catalog import (
    ACTCatalogError,
    build_act_catalog,
    validate_act_catalog,
)
from mea.providers.model_profiles import available_model_profiles


PROTOCOL = "mea_cross_task_portfolio_v1"
TRUSTED_TASKS = ("click_bell", "beat_block_hammer")
_PORTFOLIO_ID = re.compile(r"portfolio_[A-Za-z0-9_]+")
_EVALUATION_ID = re.compile(r"eval_[A-Za-z0-9_]+")


class PortfolioError(RuntimeError):
    """Raised when a portfolio plan or cached child cannot be trusted."""


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PortfolioError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise PortfolioError(f"{field} must be a non-empty string")
    return value.strip()


def _portfolio_id(value: Any) -> str:
    identifier = _text(value, field="portfolio_id")
    if _PORTFOLIO_ID.fullmatch(identifier) is None:
        raise PortfolioError(
            "portfolio_id must start with portfolio_ and contain only letters, "
            "digits, and underscores"
        )
    return identifier


def _evaluation_id(value: Any, *, field: str) -> str:
    identifier = _text(value, field=field)
    if _EVALUATION_ID.fullmatch(identifier) is None:
        raise PortfolioError(f"{field} must be a canonical eval_ identifier")
    return identifier


def _repo_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise PortfolioError(f"repo_root is not a regular directory: {root}")
    return root


def _assert_no_symlink(root: Path, path: Path, *, field: str) -> None:
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as exc:
        raise PortfolioError(f"{field} escapes repo_root") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PortfolioError(f"{field} contains a symlink component")


def _read_regular_file(root: Path, path: Path, *, field: str) -> tuple[Path, bytes]:
    _assert_no_symlink(root, path, field=field)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PortfolioError(f"{field} is missing: {path}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise PortfolioError(f"{field} is not a regular repo file")
    data = resolved.read_bytes()
    if not data:
        raise PortfolioError(f"{field} is empty")
    return resolved, data


def _json_object(data: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortfolioError(f"{field} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PortfolioError(f"{field} must contain a JSON object")
    return value


def _ref(root: Path, path: Path, data: bytes) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _file_sha256(data),
        "size_bytes": len(data),
    }


def _artifact_path(
    root: Path,
    evaluation_dir: Path,
    value: Any,
    *,
    default: str,
    field: str,
) -> Path:
    raw = default if value is None else value
    if (
        not isinstance(raw, str)
        or not raw
        or "\\" in raw
        or PurePosixPath(raw).is_absolute()
        or any(part in {"", ".", ".."} for part in PurePosixPath(raw).parts)
    ):
        raise PortfolioError(f"{field} must be a canonical relative POSIX path")
    posix = PurePosixPath(raw)
    if posix.parts[:2] == ("mea", "evaluation_runs"):
        candidate = root.joinpath(*posix.parts)
    else:
        candidate = evaluation_dir.joinpath(*posix.parts)
    _assert_no_symlink(root, candidate, field=field)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PortfolioError(f"{field} is missing: {raw}") from exc
    if not resolved.is_relative_to(evaluation_dir):
        raise PortfolioError(f"{field} is outside its child evaluation")
    return resolved


def _policy_success(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PortfolioError(f"{field} must be numeric or null")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise PortfolioError(f"{field} must be finite in [0, 1]")
    return normalized


def _historical_provider_called(
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    feedback: Mapping[str, Any],
) -> bool:
    planner = manifest.get("planner")
    if isinstance(planner, Mapping) and planner.get("provider_called") is True:
        return True
    route = evidence.get("global_query_route")
    if isinstance(route, Mapping) and route.get("provider_called") is True:
        return True
    if isinstance(feedback.get("provider_metadata"), Mapping) and bool(
        feedback["provider_metadata"]
    ):
        return True
    for raw_round in evidence.get("rounds") or []:
        if not isinstance(raw_round, Mapping):
            continue
        tool = raw_round.get("tool_evaluation")
        route_decision = tool.get("route_decision") if isinstance(tool, Mapping) else None
        if isinstance(route_decision, Mapping) and route_decision.get(
            "provider_called"
        ) is True:
            return True
    return False


def _same_rate(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _validate_outcomes(
    evidence: Mapping[str, Any], *, evaluation_id: str
) -> dict[str, Any]:
    """Recompute aggregate outcomes from the per-round episode evidence."""

    rounds = evidence.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise PortfolioError(f"{evaluation_id} evidence has no rounds")
    completed = 0
    weighted_success = 0.0
    measured_episodes = 0
    round_pipeline: list[bool] = []
    round_success: list[float | None] = []
    for index, raw_round in enumerate(rounds):
        if not isinstance(raw_round, Mapping):
            raise PortfolioError(f"{evaluation_id} round {index} is invalid")
        observations = raw_round.get("observations")
        if not isinstance(observations, Mapping):
            raise PortfolioError(f"{evaluation_id} round {index} has no observations")
        backend = str(observations.get("execution_backend") or "").casefold()
        if backend not in {"act", "act+expert"}:
            raise PortfolioError(
                f"{evaluation_id} round {index} is not an ACT policy evaluation"
            )
        top_level_seeds = raw_round.get("seeds")
        observed_seeds = observations.get("actual_seeds")
        if top_level_seeds is not None and observed_seeds is not None:
            if (
                not isinstance(top_level_seeds, list)
                or not isinstance(observed_seeds, list)
                or top_level_seeds != observed_seeds
            ):
                raise PortfolioError(
                    f"{evaluation_id} round {index} seed evidence is inconsistent"
                )
        seeds = top_level_seeds if top_level_seeds is not None else observed_seeds
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        ):
            raise PortfolioError(
                f"{evaluation_id} round {index} has no exact ACT seed list"
            )
        episodes = raw_round.get("num_episodes", len(seeds))
        if (
            isinstance(episodes, bool)
            or not isinstance(episodes, int)
            or episodes < 1
            or episodes != len(seeds)
        ):
            raise PortfolioError(
                f"{evaluation_id} round {index} episode count conflicts with seeds"
            )
        pipeline = observations.get("pipeline_passed")
        if not isinstance(pipeline, bool):
            raise PortfolioError(
                f"{evaluation_id} round {index} has no pipeline status"
            )
        success = _policy_success(
            observations.get("policy_success"),
            field=f"{evaluation_id}.rounds[{index}].observations.policy_success",
        )
        completed += episodes
        round_pipeline.append(pipeline)
        round_success.append(success)
        if success is not None:
            weighted_success += success * episodes
            measured_episodes += episodes
    declared = evidence.get("total_episodes")
    if (
        isinstance(declared, bool)
        or not isinstance(declared, int)
        or declared != completed
    ):
        raise PortfolioError(
            f"{evaluation_id} total_episodes does not match ACT seed evidence"
        )
    aggregate = evidence.get("observations")
    if not isinstance(aggregate, Mapping):
        raise PortfolioError(f"{evaluation_id} has no aggregate observations")
    pipeline_passed = aggregate.get("pipeline_passed")
    expected_pipeline = all(round_pipeline)
    if not isinstance(pipeline_passed, bool) or pipeline_passed != expected_pipeline:
        raise PortfolioError(
            f"{evaluation_id} aggregate pipeline status conflicts with its rounds"
        )
    by_round = aggregate.get("policy_success_by_round")
    if not isinstance(by_round, list) or len(by_round) != len(round_success):
        raise PortfolioError(
            f"{evaluation_id} policy_success_by_round does not match its rounds"
        )
    normalized_by_round = [
        _policy_success(
            value,
            field=f"{evaluation_id}.observations.policy_success_by_round[{index}]",
        )
        for index, value in enumerate(by_round)
    ]
    if any(
        not _same_rate(actual, expected)
        for actual, expected in zip(normalized_by_round, round_success)
    ):
        raise PortfolioError(
            f"{evaluation_id} policy_success_by_round conflicts with round evidence"
        )
    policy_success = _policy_success(
        aggregate.get("policy_success"),
        field=f"{evaluation_id}.observations.policy_success",
    )
    expected_success = (
        weighted_success / measured_episodes if measured_episodes else None
    )
    if not _same_rate(policy_success, expected_success):
        raise PortfolioError(
            f"{evaluation_id} aggregate policy_success conflicts with weighted rounds"
        )
    return {
        "pipeline_passed": pipeline_passed,
        "policy_success": policy_success,
        "completed_act_episodes": completed,
    }


def _validated_runtime(
    root: Path,
    evaluation_dir: Path,
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    feedback: Mapping[str, Any],
    *,
    evaluation_id: str,
    completed_act_episodes: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    del root, evaluation_dir, evaluation_id
    declared = manifest.get("run_manifest")
    if isinstance(declared, Mapping):
        raw_act_starts = declared.get("act_rollouts_started")
        raw_provider_calls = declared.get("provider_calls_started")
        act_starts = (
            int(raw_act_starts)
            if isinstance(raw_act_starts, int)
            and not isinstance(raw_act_starts, bool)
            and raw_act_starts >= completed_act_episodes
            else completed_act_episodes
        )
        provider_calls = (
            int(raw_provider_calls)
            if isinstance(raw_provider_calls, int)
            and not isinstance(raw_provider_calls, bool)
            and raw_provider_calls >= 0
            else None
        )
        return (
            {
                "accounting_mode": "run_manifest_declaration",
                "provider_called": (
                    provider_calls > 0
                    if provider_calls is not None
                    else _historical_provider_called(manifest, evidence, feedback)
                ),
                "provider_calls_started": provider_calls,
                "provider_transport_attempts_started": None,
                "act_rollouts_started": act_starts,
                "completed_act_episodes": completed_act_episodes,
                "started_count_exact": False,
                "limitation": (
                    "run manifest counters are descriptive and are not backed by "
                    "per-call ledger files"
                ),
            },
            {},
        )
    return (
        {
            "accounting_mode": "completed_episode_fallback",
            "provider_called": _historical_provider_called(
                manifest, evidence, feedback
            ),
            "provider_calls_started": None,
            "provider_transport_attempts_started": None,
            "act_rollouts_started": completed_act_episodes,
            "completed_act_episodes": completed_act_episodes,
            "started_count_exact": False,
            "limitation": (
                "No run manifest counters are available; ACT starts fall back "
                "to completed episode seeds and can undercount interrupted runs."
            ),
        },
        {},
    )


def _checkpoint_contract(
    manifest: Mapping[str, Any], *, task_name: str, evaluation_id: str
) -> dict[str, Any]:
    plan = manifest.get("plan")
    policy = plan.get("policy") if isinstance(plan, Mapping) else None
    if not isinstance(policy, Mapping):
        return {
            "status": "unsupported_missing_plan_policy",
            "logical_checkpoint_id": None,
            "checkpoint_bytes_hash_bound": False,
            "limitation": (
                "child manifest lacks a plan.policy checkpoint contract; portfolio "
                "cannot establish checkpoint identity"
            ),
        }
    expected = {
        "name": "ACT",
        "checkpoint_setting": "demo_clean",
        "expert_data_num": 50,
    }
    for field, value in expected.items():
        if policy.get(field) != value:
            raise PortfolioError(
                f"{evaluation_id} plan.policy.{field} is not the trusted ACT contract"
            )
    return {
        "status": "validated_logical_contract",
        "logical_checkpoint_id": f"act-{task_name}/demo_clean-50",
        "policy_name": "ACT",
        "checkpoint_setting": "demo_clean",
        "expert_data_num": 50,
        "checkpoint_bytes_hash_bound": False,
        "limitation": (
            "logical policy/checkpoint settings are validated, but legacy child "
            "evidence does not hash-bind checkpoint bytes"
        ),
    }


def _load_child(root: Path, task_name: str, evaluation_id: str) -> dict[str, Any]:
    evaluation_dir = root / "mea" / "evaluation_runs" / evaluation_id
    _assert_no_symlink(root, evaluation_dir, field=f"{evaluation_id} directory")
    if not evaluation_dir.is_dir():
        raise PortfolioError(f"child evaluation is missing: {evaluation_id}")

    manifest_path, manifest_bytes = _read_regular_file(
        root,
        evaluation_dir / "manifest.json",
        field=f"{evaluation_id} manifest",
    )
    manifest = _json_object(manifest_bytes, field=f"{evaluation_id} manifest")
    if (
        manifest.get("evaluation_id") != evaluation_id
        or manifest.get("task_name") != task_name
        or manifest.get("lifecycle_status") != "completed"
        or manifest.get("status")
        not in {"completed", "completed_with_pipeline_failure"}
    ):
        raise PortfolioError(
            f"{evaluation_id} is not a completed {task_name} Agent evaluation"
        )

    paths = {
        "evidence": _artifact_path(
            root,
            evaluation_dir,
            manifest.get("evidence_path"),
            default="summary/evidence_bundle.json",
            field=f"{evaluation_id}.evidence_path",
        ),
        "feedback": _artifact_path(
            root,
            evaluation_dir,
            manifest.get("feedback_path"),
            default="feedback/feedback.json",
            field=f"{evaluation_id}.feedback_path",
        ),
        "report": _artifact_path(
            root,
            evaluation_dir,
            manifest.get("report_path"),
            default="evaluation_report.md",
            field=f"{evaluation_id}.report_path",
        ),
    }
    loaded: dict[str, tuple[Path, bytes]] = {
        "manifest": (manifest_path, manifest_bytes)
    }
    for name, path in paths.items():
        loaded[name] = _read_regular_file(
            root, path, field=f"{evaluation_id} {name}"
        )
    evidence = _json_object(loaded["evidence"][1], field=f"{evaluation_id} evidence")
    feedback = _json_object(loaded["feedback"][1], field=f"{evaluation_id} feedback")
    try:
        report_text = loaded["report"][1].decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise PortfolioError(f"{evaluation_id} report is not UTF-8") from exc
    if not report_text:
        raise PortfolioError(f"{evaluation_id} report is empty")
    if evidence.get("schema_version") != 2 or evidence.get(
        "evaluation_id"
    ) != evaluation_id:
        raise PortfolioError(f"{evaluation_id} evidence identity is invalid")
    if isinstance(manifest.get("feedback"), Mapping) and dict(
        manifest["feedback"]
    ) != feedback:
        raise PortfolioError(f"{evaluation_id} feedback differs from parent manifest")

    outcomes = _validate_outcomes(evidence, evaluation_id=evaluation_id)
    pipeline_passed = bool(outcomes["pipeline_passed"])
    expected_status = "completed" if pipeline_passed else "completed_with_pipeline_failure"
    if manifest.get("status") != expected_status:
        raise PortfolioError(
            f"{evaluation_id} manifest status conflicts with pipeline evidence"
        )
    runtime, runtime_refs = _validated_runtime(
        root,
        evaluation_dir,
        manifest,
        evidence,
        feedback,
        evaluation_id=evaluation_id,
        completed_act_episodes=int(outcomes["completed_act_episodes"]),
    )
    checkpoint = _checkpoint_contract(
        manifest, task_name=task_name, evaluation_id=evaluation_id
    )
    artifacts = {
        name: _ref(root, path, data) for name, (path, data) in loaded.items()
    }
    artifacts.update(runtime_refs)
    manifest_plan = manifest.get("plan")
    planned_aspects = (
        deepcopy(manifest_plan.get("requested_aspect_ids"))
        if isinstance(manifest_plan, Mapping)
        else None
    )
    executed_aspects = [
        raw_round.get("sub_aspect")
        for raw_round in evidence.get("rounds", [])
        if isinstance(raw_round, Mapping)
    ]
    return {
        "task_name": task_name,
        "evaluation_id": evaluation_id,
        "source_user_request": evidence.get("user_request"),
        "pipeline_passed": pipeline_passed,
        "policy_success": outcomes["policy_success"],
        "act_rollouts_started": runtime["act_rollouts_started"],
        "completed_act_episodes": runtime["completed_act_episodes"],
        "provider_called": runtime["provider_called"],
        "provider_call_count": runtime["provider_calls_started"],
        "provider_transport_attempt_count": runtime[
            "provider_transport_attempts_started"
        ],
        "runtime_accounting": runtime,
        "checkpoint_contract": checkpoint,
        "evaluation_binding": {
            "manifest_user_request": manifest.get("user_request"),
            "evidence_user_request": evidence.get("user_request"),
            "bound_task_name": manifest.get("bound_task_name"),
            "bound_requested_aspect_ids": deepcopy(
                manifest.get("bound_requested_aspect_ids")
            ),
            "planned_requested_aspect_ids": planned_aspects,
            "executed_aspect_ids": executed_aspects,
            "max_agent_rounds": manifest.get("max_agent_rounds"),
            "plan_max_rounds": (
                manifest_plan.get("max_rounds")
                if isinstance(manifest_plan, Mapping)
                else None
            ),
            "executed_rounds": len(evidence.get("rounds", [])),
        },
        "artifacts": artifacts,
    }


def load_child_evaluation(
    repo_root: str | Path, task_name: str, evaluation_id: str
) -> dict[str, Any]:
    """Load one completed ACT child through the portfolio evidence checks.

    This small public adapter lets other parent orchestration layers consume
    the same recomputed policy outcome instead of interpreting report prose or
    treating pipeline completion as policy success.
    """

    root = _repo_root(repo_root)
    normalized_task = _text(task_name, field="task_name")
    if normalized_task not in TRUSTED_TASKS:
        raise PortfolioError(f"task is not a trusted ACT child: {normalized_task}")
    normalized_evaluation = _evaluation_id(evaluation_id, field="evaluation_id")
    return _load_child(root, normalized_task, normalized_evaluation)


def _synthesize(children: list[Mapping[str, Any]], *, mode: str) -> dict[str, Any]:
    strengths: list[str] = []
    weaknesses: list[str] = []
    recommendations: list[str] = []
    measured_successes = 0
    measured_failures = 0
    unavailable = 0
    for child in children:
        task = str(child["task_name"])
        success = child.get("policy_success")
        pipeline = bool(child.get("pipeline_passed"))
        if success is None:
            unavailable += 1
            weaknesses.append(
                f"{task}: no authoritative ACT policy_success was available; "
                f"pipeline_passed={str(pipeline).lower()} is not a substitute."
            )
            recommendations.append(
                f"Run one completed ACT episode for {task} before making a policy claim."
            )
        elif float(success) <= 0.0:
            measured_failures += 1
            weaknesses.append(
                f"{task}: the observed ACT policy_success was {float(success):.3f}."
            )
            recommendations.append(
                f"Diagnose {task} with another preregistered seed before broadening scope."
            )
        elif float(success) >= 1.0:
            measured_successes += 1
            strengths.append(
                f"{task}: the bound child evidence reports ACT policy_success=1.000."
            )
        else:
            measured_successes += 1
            measured_failures += 1
            strengths.append(
                f"{task}: at least one bound ACT episode succeeded "
                f"(policy_success={float(success):.3f})."
            )
            weaknesses.append(
                f"{task}: success was partial rather than complete "
                f"(policy_success={float(success):.3f})."
            )
            recommendations.append(
                f"Add 3 seeds for {task} to localize the partial-failure regime."
            )
        if not pipeline:
            weaknesses.append(
                f"{task}: the evidence pipeline did not pass, independently of policy outcome."
            )

    if not strengths:
        strengths.append(
            "The parent hash-binds both child evidence chains; this is provenance "
            "strength, not evidence that either policy succeeded."
        )
    if not weaknesses:
        weaknesses.append(
            "Each task is represented by only the supplied small child evaluation; "
            "generalization is not established."
        )
    if not recommendations:
        recommendations.append(
            "Repeat each task on 3 independent seeds while preserving the same ACT checkpoint."
        )
    limitations = [
        "Pipeline completion and ACT policy success are separate fields and claims.",
        "This two-task development portfolio is not a paper-scale benchmark.",
        (
            "Reused children were generated by earlier task-specific queries; the current "
            "query binds their evidence but did not causally execute them."
            if mode == "reused_completed_children"
            else "Commands are inert plans until the two child Agent processes complete."
        ),
    ]
    return {
        "answer": (
            f"The query is bound to {len(children)} trusted ACT tasks. "
            f"Observed policy outcomes: {measured_successes} with nonzero success, "
            f"{measured_failures} with failure evidence, {unavailable} unavailable. "
            "Pipeline status was not used as policy success."
        ),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
        "limitations": limitations,
    }


def _plan_synthesis() -> dict[str, Any]:
    return {
        "answer": (
            "The open query is bound to hard-capped one-round child command plans for "
            "click_bell and beat_block_hammer; no child has been executed yet."
        ),
        "strengths": [
            "Both tasks are present in the checkpoint-ready trusted ACT catalog."
        ],
        "weaknesses": [
            "There is no policy outcome until both child commands finish and pass postflight."
        ],
        "recommendations": [
            "Execute each argv exactly once, then rebuild the portfolio in reuse mode."
        ],
        "limitations": [
            "This command plan starts no provider, simulator, or ACT call.",
            "The child Agent enforces --max-agent-rounds=1 independently of planner output.",
            "A two-task N=1 smoke is not a paper-scale benchmark.",
        ],
    }


def build_portfolio_command_plan(
    repo_root: str | Path,
    *,
    portfolio_id: str,
    user_query: str,
    start_seed: int = 100403,
    gpu: int = 0,
    model_profile: str = "economy",
    python_executable: str = "python",
    catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build two inert, exact child Agent argv plans with a two-ACT ceiling."""

    root = _repo_root(repo_root)
    identifier = _portfolio_id(portfolio_id)
    query = _text(user_query, field="user_query")
    if isinstance(start_seed, bool) or not isinstance(start_seed, int) or start_seed < 0:
        raise PortfolioError("start_seed must be a non-negative integer")
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise PortfolioError("gpu must be a non-negative integer")
    executable = _text(python_executable, field="python_executable")
    profile = _text(model_profile, field="model_profile")
    if profile not in available_model_profiles():
        raise PortfolioError(
            f"model_profile must be one of {list(available_model_profiles())}"
        )
    try:
        trusted_catalog = (
            validate_act_catalog(catalog)
            if catalog is not None
            else build_act_catalog(root)
        )
    except ACTCatalogError as exc:
        raise PortfolioError(f"trusted ACT catalog is invalid: {exc}") from exc
    ready = {task["task_name"] for task in trusted_catalog["tasks"]}
    if ready.intersection(TRUSTED_TASKS) != set(TRUSTED_TASKS):
        missing = sorted(set(TRUSTED_TASKS) - ready)
        raise PortfolioError(f"trusted ACT catalog is missing portfolio tasks: {missing}")
    runner_path, runner_bytes = _read_regular_file(
        root,
        root / "scripts/manipeval_agent.py",
        field="child Agent runner",
    )
    runner_ref = _ref(root, runner_path, runner_bytes)

    suffix = identifier.removeprefix("portfolio_")
    children: list[dict[str, Any]] = []
    for task_name in TRUSTED_TASKS:
        short = "click_bell" if task_name == "click_bell" else "bbh"
        evaluation_id = f"eval_{suffix}_{short}"
        child_request = (
            query
            + "\n\nPortfolio slice: evaluate exactly one "
            + (
                "click_bell position-generalization round"
                if task_name == "click_bell"
                else "beat_block_hammer object-appearance round"
            )
            + " with one ACT episode, then stop."
        )
        argv = [
            executable,
            "scripts/manipeval_agent.py",
            "--repo-root",
            ".",
            "--request",
            child_request,
            "--evaluation-id",
            evaluation_id,
            "--task-name",
            task_name,
            "--start-seed",
            str(start_seed),
            "--num-episodes",
            "1",
            "--generated-rounds",
            "1",
            "--max-agent-rounds",
            "1",
            "--telemetry-profile",
            "balanced_v1",
            "--model-profile",
            profile,
            "--gpu",
            str(gpu),
            "--no-history",
        ]
        if task_name == "click_bell":
            argv.extend(["--task-profile", "adaptive_properties"])
        children.append(
            {
                "task_name": task_name,
                "evaluation_id": evaluation_id,
                "parent_query_sha256": _canonical_sha256(query),
                "derived_request": child_request,
                "derived_request_sha256": _canonical_sha256(child_request),
                "runner": deepcopy(runner_ref),
                "argv": argv,
                "argv_sha256": _canonical_sha256(argv),
                "expected_postconditions": {
                    "executed_rounds": 1,
                    "hard_agent_round_limit": 1,
                    "act_rollouts_started": 1,
                    "execution_backend": "ACT",
                    "task_name": task_name,
                },
            }
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "portfolio_id": identifier,
        "mode": "live_command_plan",
        "status": "planned",
        "user_query": query,
        "user_query_sha256": _canonical_sha256(query),
        "task_bindings": list(TRUSTED_TASKS),
        "catalog_sha256": trusted_catalog["catalog_sha256"],
        "runner": runner_ref,
        "children": children,
        "runtime": {
            "provider_calls_started": 0,
            "simulator_calls_started": 0,
            "act_rollouts_started": 0,
        },
        "planned_runtime": {
            "child_count": 2,
            "rounds": 2,
            "max_act_rollouts": 2,
            "provider_call_count": None,
            "provider_call_count_unavailable_reason": (
                "child Agents own planner, generation, VQA, and feedback calls"
            ),
        },
        "synthesis": _plan_synthesis(),
        "claim_scope": "two-task one-query functional command plan only",
        "paper_table_eligible": False,
    }
    result["portfolio_sha256"] = _canonical_sha256(result)
    return result


def build_reused_portfolio(
    repo_root: str | Path,
    *,
    portfolio_id: str,
    user_query: str,
    child_evaluation_ids: Mapping[str, str],
) -> dict[str, Any]:
    """Hash-bind and synthesize two explicitly selected completed children."""

    root = _repo_root(repo_root)
    identifier = _portfolio_id(portfolio_id)
    query = _text(user_query, field="user_query")
    if not isinstance(child_evaluation_ids, Mapping) or set(
        child_evaluation_ids
    ) != set(TRUSTED_TASKS):
        raise PortfolioError(
            f"child_evaluation_ids must contain exactly {list(TRUSTED_TASKS)}"
        )
    normalized_ids = {
        task: _evaluation_id(
            child_evaluation_ids[task], field=f"child_evaluation_ids.{task}"
        )
        for task in TRUSTED_TASKS
    }
    if len(set(normalized_ids.values())) != len(TRUSTED_TASKS):
        raise PortfolioError("the two trusted tasks must use unique child evaluations")
    children = [
        _load_child(root, task, normalized_ids[task]) for task in TRUSTED_TASKS
    ]
    historical_act = sum(int(child["act_rollouts_started"]) for child in children)
    historical_completed = sum(
        int(child["completed_act_episodes"]) for child in children
    )
    provider_counts = [child["provider_call_count"] for child in children]
    transport_counts = [
        child["provider_transport_attempt_count"] for child in children
    ]
    exact_provider_counts = all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in provider_counts
    )
    exact_transport_counts = all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in transport_counts
    )
    runtime_limitations = [
        child["runtime_accounting"]["limitation"]
        for child in children
        if child["runtime_accounting"].get("limitation")
    ]
    checkpoint_limitations = [
        f"{child['task_name']}: {child['checkpoint_contract']['limitation']}"
        for child in children
        if child["checkpoint_contract"].get("limitation")
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "portfolio_id": identifier,
        "mode": "reused_completed_children",
        "status": "completed",
        "user_query": query,
        "user_query_sha256": _canonical_sha256(query),
        "task_bindings": list(TRUSTED_TASKS),
        "children": children,
        "runtime": {
            "provider_calls_started": 0,
            "simulator_calls_started": 0,
            "act_rollouts_started": 0,
        },
        "historical_child_runtime": {
            "provider_called": any(child["provider_called"] for child in children),
            "provider_calls_started": (
                sum(provider_counts) if exact_provider_counts else None
            ),
            "provider_transport_attempts_started": (
                sum(transport_counts) if exact_transport_counts else None
            ),
            "act_rollouts_started": historical_act,
            "completed_act_episodes": historical_completed,
            "started_count_exact": all(
                child["runtime_accounting"]["started_count_exact"]
                for child in children
            ),
            "accounting_by_child": {
                child["task_name"]: child["runtime_accounting"]["accounting_mode"]
                for child in children
            },
        },
        "synthesis": _synthesize(children, mode="reused_completed_children"),
        "claim_scope": (
            "hash-bound synthesis of explicitly selected completed child evaluations"
        ),
        "paper_table_eligible": False,
        "limitations": [
            "The parent made no new runtime calls.",
            "Reused evidence is not causal evidence that this query launched the children.",
            *runtime_limitations,
            *checkpoint_limitations,
        ],
    }
    result["portfolio_sha256"] = _canonical_sha256(result)
    return result


def render_portfolio_report(value: Mapping[str, Any]) -> str:
    """Render a compact human-readable entry point for either portfolio mode."""

    synthesis = value.get("synthesis")
    if not isinstance(synthesis, Mapping):
        raise PortfolioError("portfolio has no synthesis")
    lines = [
        "# MEA Cross-task Portfolio",
        "",
        str(synthesis.get("answer") or ""),
        "",
        f"- mode: `{value.get('mode')}`",
        f"- tasks: `{', '.join(value.get('task_bindings') or [])}`",
        f"- paper-table eligible: `{str(value.get('paper_table_eligible')).lower()}`",
        f"- new ACT rollouts: `{(value.get('runtime') or {}).get('act_rollouts_started')}`",
        "",
    ]
    historical = value.get("historical_child_runtime")
    if isinstance(historical, Mapping):
        lines.extend(
            [
                "## Historical child runtime",
                "",
                f"- provider calls started: `{historical.get('provider_calls_started')}`",
                f"- provider transport attempts started: `{historical.get('provider_transport_attempts_started')}`",
                f"- ACT rollouts started: `{historical.get('act_rollouts_started')}`",
                f"- ACT episodes completed: `{historical.get('completed_act_episodes')}`",
                f"- started counts exact: `{str(historical.get('started_count_exact')).lower()}`",
                "",
            ]
        )
    for heading, key in (
        ("Strengths", "strengths"),
        ("Weaknesses", "weaknesses"),
        ("Recommendations", "recommendations"),
        ("Limitations", "limitations"),
    ):
        lines.extend([f"## {heading}", ""])
        items = list(synthesis.get(key) or [])
        if key == "limitations":
            items.extend(value.get("limitations") or [])
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    if value.get("mode") == "live_command_plan":
        lines.extend(["## Exact child argv", ""])
        for child in value.get("children") or []:
            lines.extend(
                [
                    f"### {child['task_name']}",
                    "",
                    "```json",
                    json.dumps(child["argv"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "PROTOCOL",
    "TRUSTED_TASKS",
    "PortfolioError",
    "build_portfolio_command_plan",
    "build_reused_portfolio",
    "load_child_evaluation",
    "render_portfolio_report",
]
