"""Resumable, task-discovered RoboTwin breadth harness.

Each phase answers one question and keeps its cost explicit:

``preflight`` (no provider/policy), ``plan`` (provider, no simulator),
``materialize`` (provider + TaskGen gates, no learned policy), and ``official``
(one SmolVLA rollout).  Tasks come only from the repository's official task
library; this file contains no task-name branches or second artifact registry.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mea.method_runtime import (  # noqa: E402
    BackendBindingRequest,
    CandidateRequest,
    MethodRuntime,
    RolloutRequest,
)
from mea.plan_agent_bootstrap import (  # noqa: E402
    build_pending_task_binding_policy_card,
)
from mea.planner import (  # noqa: E402
    build_dynamic_experiment_candidate,
    build_initial_semantic_proposal_bundle,
    evaluation_intent_from_query_interpretation,
)
from mea.planner.open_task_resolver import PlanAgentQueryInterpreter  # noqa: E402
from mea.planner.policy_task_binding import policy_task_binding_from_target  # noqa: E402
from mea.planner.runtime_task_binding import (  # noqa: E402
    RuntimePolicySpec,
    build_runtime_open_world_evaluation_target,
    build_smolvla_policy_spec,
)
from mea.providers import OpenAICompatibleProvider, ProviderError  # noqa: E402
from mea.robotwin.runtime import RoboTwinMethodBackend  # noqa: E402
from mea.robotwin.smolvla_rollout import SmolVLARobotwinRolloutRunner  # noqa: E402
from mea.robotwin.task_identity import (  # noqa: E402
    RoboTwinTaskIdentity,
    discover_robotwin_official_tasks,
)
from mea.robotwin_task_context import probe_official_robotwin_task_context  # noqa: E402
from mea.taskgen.runtime import create_generic_provider_taskgen_run  # noqa: E402


PHASES = ("preflight", "plan", "materialize", "official")
FAILURE_TAXONOMY = {
    "task_context": "source, binding, or policy-free reset",
    "plan_agent": "Query interpretation or typed Proposal",
    "taskgen": "scene/checker generation or pre-rollout gates",
    "policy": "completed official episode with a negative outcome",
    "provider": "text or vision provider",
    "simulator": "official simulator execution",
    "infrastructure": "checkpoint, process, network, or unknown runtime",
}


@dataclass(frozen=True)
class HarnessConfig:
    repo_root: Path
    output_dir: Path
    checkpoint: Path
    phase: str
    query: str
    tasks: tuple[str, ...]
    seed: int = 1000
    policy_server_port: int = 18771
    text_model: str = "gpt-4o-2024-11-20"
    vision_model: str = "gpt-4o-2024-11-20"
    telemetry_profile: str = "balanced_v1"
    resume: bool = True


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _git_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() or None


def _relative(path: str | Path, root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def _result_path(config: HarnessConfig, task: str) -> Path:
    return config.output_dir / "tasks" / task / f"{config.phase}.json"


def _target(
    config: HarnessConfig, task: str, policy: RuntimePolicySpec
) -> dict[str, Any]:
    return build_runtime_open_world_evaluation_target(
        config.repo_root, task, max_rounds=2, policy_spec=policy
    )


def _provider(config: HarnessConfig) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        text_model=config.text_model,
        vision_model=config.vision_model,
        timeout=180.0,
        max_retries=0,
    )


def _runtime(
    config: HarnessConfig,
    target: Mapping[str, Any],
    provider: OpenAICompatibleProvider | None,
) -> tuple[MethodRuntime, Any, RoboTwinMethodBackend]:
    binding_contract = policy_task_binding_from_target(target)
    backend = RoboTwinMethodBackend(
        repo_root=config.repo_root,
        rollout_runner=SmolVLARobotwinRolloutRunner(
            port=config.policy_server_port,
            repo_root=config.repo_root,
            telemetry_profile=config.telemetry_profile,
        ),
        accepted_taskgen_materializer=(
            create_generic_provider_taskgen_run if provider is not None else None
        ),
        taskgen_provider=provider,
        taskgen_text_model=config.text_model if provider is not None else "",
        taskgen_vision_model=config.vision_model if provider is not None else "",
        taskgen_telemetry_profile=config.telemetry_profile,
    )
    runtime = MethodRuntime(backend)
    binding = runtime.bind_task(
        BackendBindingRequest(
            task_reference={
                "task_name": binding_contract["task_name"],
                "binding_id": (
                    f"{binding_contract['task_name']}/"
                    f"{binding_contract['policy']['name']}"
                ),
                "policy": binding_contract["policy"],
            },
            artifacts={"checkpoint": str(config.checkpoint)},
            metadata={
                "checkpoint_id": binding_contract["checkpoint"]["checkpoint_id"]
            },
        )
    )
    return runtime, binding, backend


def _proposal(
    config: HarnessConfig,
    identity: RoboTwinTaskIdentity,
    policy: RuntimePolicySpec,
    provider: OpenAICompatibleProvider,
) -> tuple[dict[str, Any], dict[str, Any]]:
    interpreter = PlanAgentQueryInterpreter(
        provider, model=config.text_model, max_attempts=2
    )
    bundle = interpreter.propose(
        config.query,
        policy_card=build_pending_task_binding_policy_card(policy),
    )
    needs = bundle.get("experiment_needs")
    if not isinstance(needs, Mapping):
        raise ValueError("Plan Agent returned no typed experiment needs")
    intent = evaluation_intent_from_query_interpretation(bundle["concern"])
    semantic = build_initial_semantic_proposal_bundle(
        user_query=config.query,
        concern=bundle["concern"],
        experiment_needs=needs,
        evaluation_intent=intent,
        provider_record=bundle.get("provider"),
    )
    candidate = build_dynamic_experiment_candidate(
        user_query=config.query,
        task_name=identity.task_name,
        proposal=semantic["proposal"],
        evaluation_intent=intent,
    )
    task_dir = config.output_dir / "tasks" / identity.task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    if interpreter.last_prompt:
        (task_dir / "plan_prompt.md").write_text(
            interpreter.last_prompt, encoding="utf-8"
        )
    for index, response in enumerate(interpreter.last_responses, start=1):
        (task_dir / f"plan_response_{index}.txt").write_text(
            response + "\n", encoding="utf-8"
        )
    return bundle, candidate


def _preflight(
    config: HarnessConfig,
    identity: RoboTwinTaskIdentity,
    policy: RuntimePolicySpec,
) -> dict[str, Any]:
    target = _target(config, identity.task_name, policy)
    probe = probe_official_robotwin_task_context(
        repo_root=config.repo_root,
        task_name=identity.task_name,
        seed=config.seed,
        action_dimension=int(policy.metadata["action_dimension"]),
    )
    return {
        "status": "passed",
        "task_name": identity.task_name,
        "official_source": identity.official_source,
        "task_schema_available": identity.task_schema_available,
        "actor_count": len(probe["actors"]),
        "observables": probe.get("observables"),
        "checkpoint_id": policy_task_binding_from_target(target)["checkpoint"][
            "checkpoint_id"
        ],
        "policy_rollouts": 0,
    }


def _plan(
    config: HarnessConfig,
    identity: RoboTwinTaskIdentity,
    policy: RuntimePolicySpec,
) -> dict[str, Any]:
    bundle, candidate = _proposal(
        config, identity, policy, _provider(config)
    )
    return {
        "status": "passed",
        "task_name": identity.task_name,
        "query_interpretation": bundle,
        "proposal": candidate,
        "policy_rollouts": 0,
    }


def _materialize(
    config: HarnessConfig,
    identity: RoboTwinTaskIdentity,
    policy: RuntimePolicySpec,
) -> dict[str, Any]:
    target = _target(config, identity.task_name, policy)
    provider = _provider(config)
    bundle, candidate = _proposal(config, identity, policy, provider)
    runtime, binding, _backend = _runtime(config, target, provider)
    run_id = re.sub(
        r"[^a-z0-9_]+", "_",
        f"breadth_{config.output_dir.name}_{identity.task_name}".casefold(),
    )
    materialized = runtime.materialize_candidate(
        binding,
        CandidateRequest(
            candidate_id=candidate["candidate_id"],
            source_query=candidate["source_query"],
            proposal_bundle=candidate,
            output_dir=config.output_dir / "tasks" / identity.task_name / "candidate",
            seed=config.seed,
            context={"taskgen_run_id": run_id},
        ),
    )
    return {
        "status": "passed",
        "task_name": identity.task_name,
        "query_interpretation": bundle,
        "proposal": candidate,
        "taskgen_route": materialized.metadata.get("taskgen_route"),
        "validation": materialized.validation,
        "artifacts": {
            key: _relative(value, config.repo_root)
            for key, value in materialized.artifacts.items()
            if key in {"manifest", "task_source", "task_context", "overlay"}
        },
        "policy_rollouts": 0,
    }


def _official(
    config: HarnessConfig,
    identity: RoboTwinTaskIdentity,
    policy: RuntimePolicySpec,
) -> dict[str, Any]:
    target = _target(config, identity.task_name, policy)
    runtime, binding, backend = _runtime(config, target, None)
    candidate = backend.official_candidate(
        binding, source_query=config.query, seed=config.seed
    )
    rollout = runtime.rollout(
        candidate,
        RolloutRequest(
            round_id="round_1",
            seed=config.seed,
            output_dir=(
                config.output_dir / "tasks" / identity.task_name / "official_episode"
            ),
            provenance={"harness": "robotwin_breadth_v1"},
        ),
    )
    return {
        "status": "passed" if rollout.success else "policy_negative",
        "task_name": identity.task_name,
        "failure_kind": None if rollout.success else "policy",
        "policy_success": rollout.success,
        "official_check_success": rollout.episode.get("official_check_success"),
        "actions_executed": rollout.episode.get("actions_executed"),
        "chunk_count": rollout.episode.get("chunk_count"),
        "artifacts": {
            key: _relative(value, config.output_dir)
            for key, value in rollout.artifacts.items()
        },
        "policy_rollouts": 1,
    }


def _failure_kind(phase: str, exc: Exception) -> str:
    if isinstance(exc, ProviderError):
        return "provider"
    return {
        "preflight": "task_context",
        "plan": "plan_agent",
        "materialize": "taskgen",
        "official": "simulator",
    }.get(phase, "infrastructure")


def _summary(config: HarnessConfig) -> dict[str, Any]:
    records = [
        record
        for task in config.tasks
        for record in [_read_json(_result_path(config, task))]
        if record is not None
    ]
    statuses = Counter(str(item.get("status") or "invalid") for item in records)
    failures = Counter(
        str(item["failure_kind"]) for item in records if item.get("failure_kind")
    )
    return {
        "schema_version": 1,
        "harness": "robotwin_breadth_v1",
        "phase": config.phase,
        "task_count": len(config.tasks),
        "completed_task_count": len(records),
        "status_counts": dict(sorted(statuses.items())),
        "failure_kind_counts": dict(sorted(failures.items())),
        "policy_rollout_count": sum(int(item.get("policy_rollouts") or 0) for item in records),
        "tasks": [
            {key: item.get(key) for key in ("task_name", "status", "failure_kind", "wall_seconds")}
            for item in records
        ],
        "claim_boundary": (
            "Breadth characterization only. N=1 policy outcomes are separate "
            "from method/system failures and are not benchmark estimates."
        ),
    }


def run_harness(config: HarnessConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "harness": "robotwin_breadth_v1",
        "source_commit": _git_head(config.repo_root),
        "phase": config.phase,
        "query": config.query,
        "tasks": list(config.tasks),
        "seed": config.seed,
        "checkpoint": str(config.checkpoint),
        "failure_taxonomy": FAILURE_TAXONOMY,
    }
    config_path = config.output_dir / "run_config.json"
    previous = _read_json(config_path)
    if previous is not None:
        _validate_resume_contract(previous, contract)
    else:
        _write_json(config_path, contract)
    identities = {item.task_name: item for item in discover_robotwin_official_tasks(config.repo_root)}
    policy = build_smolvla_policy_spec(config.checkpoint)
    runners = {
        "preflight": _preflight,
        "plan": _plan,
        "materialize": _materialize,
        "official": _official,
    }
    for task in config.tasks:
        path = _result_path(config, task)
        if config.resume and _read_json(path) is not None:
            continue
        started = time.perf_counter()
        try:
            result = runners[config.phase](config, identities[task], policy)
        except Exception as exc:  # one task must not erase the resumable batch
            kind = _failure_kind(config.phase, exc)
            result = {
                "status": "failed",
                "task_name": task,
                "failure_kind": kind,
                "failure_taxonomy": FAILURE_TAXONOMY[kind],
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "policy_rollouts": 0,
            }
        result.update(
            phase=config.phase,
            recorded_at=datetime.now().astimezone().isoformat(),
            wall_seconds=round(time.perf_counter() - started, 6),
        )
        _write_json(path, result)
        _write_json(config.output_dir / "summary.json", _summary(config))
        print(json.dumps({key: result.get(key) for key in ("task_name", "status", "failure_kind")}), flush=True)
    summary = _summary(config)
    _write_json(config.output_dir / "summary.json", summary)
    return summary


def _validate_resume_contract(
    previous: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    if dict(previous) != dict(contract):
        raise ValueError(
            "resume run_config or source commit differs; use a new output "
            "directory"
        )


def _selected_tasks(
    identities: Sequence[RoboTwinTaskIdentity], requested: Sequence[str]
) -> tuple[RoboTwinTaskIdentity, ...]:
    available = {item.task_name: item for item in identities}
    names = [name.strip() for item in requested for name in item.split(",") if name.strip()]
    if not names:
        return tuple(identities)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"tasks are outside discovered inventory: {unknown}")
    return tuple(available[name] for name in dict.fromkeys(names))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin"))
    parser.add_argument("--query", default="What bounded executable variation is most likely to expose this policy's first weakness? Choose without an aspect or template menu.")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--policy-server-port", type=int, default=18771)
    parser.add_argument("--text-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--vision-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--telemetry-profile", default="balanced_v1")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    root = args.repo_root.expanduser().resolve()
    selected = _selected_tasks(discover_robotwin_official_tasks(root), args.task)
    if args.limit is not None:
        selected = selected[: args.limit]
    summary = run_harness(
        HarnessConfig(
            repo_root=root,
            output_dir=args.output_dir.expanduser().resolve(),
            checkpoint=args.checkpoint.expanduser().resolve(),
            phase=args.phase,
            query=args.query.strip(),
            tasks=tuple(item.task_name for item in selected),
            seed=args.seed,
            policy_server_port=args.policy_server_port,
            text_model=args.text_model,
            vision_model=args.vision_model,
            telemetry_profile=args.telemetry_profile,
            resume=args.resume,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if summary["status_counts"].get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
