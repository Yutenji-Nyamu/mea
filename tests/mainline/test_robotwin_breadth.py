from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.paper.robotwin_breadth import (
    HarnessConfig,
    _selected_tasks,
    _summary,
    _validate_resume_contract,
)
from mea.robotwin.task_identity import discover_robotwin_official_tasks


def _write_task(root: Path, name: str) -> None:
    source = root / "envs" / f"{name}.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f"class {name}:\n    pass\n", encoding="utf-8")
    instruction = root / "description/task_instruction" / f"{name}.json"
    instruction.parent.mkdir(parents=True, exist_ok=True)
    instruction.write_text(
        json.dumps({"full_description": f"execute {name}"}),
        encoding="utf-8",
    )


def _config(root: Path, output: Path, tasks: tuple[str, ...]) -> HarnessConfig:
    return HarnessConfig(
        repo_root=root,
        output_dir=output,
        checkpoint=root / "checkpoint",
        phase="preflight",
        query="which bounded variation exposes a weakness?",
        tasks=tasks,
        seed=1000,
        policy_server_port=18771,
        text_model="fixture-text",
        vision_model="fixture-vision",
        telemetry_profile="balanced_v1",
        resume=True,
    )


def test_breadth_task_selection_is_discovery_driven(tmp_path: Path) -> None:
    _write_task(tmp_path, "alpha_task")
    _write_task(tmp_path, "beta_task")
    identities = discover_robotwin_official_tasks(tmp_path)

    assert [item.task_name for item in _selected_tasks(identities, [])] == [
        "alpha_task",
        "beta_task",
    ]
    assert [
        item.task_name
        for item in _selected_tasks(identities, ["beta_task,alpha_task"])
    ] == ["beta_task", "alpha_task"]


def test_breadth_summary_separates_policy_negative_and_system_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "batch"
    config = _config(tmp_path, output, ("alpha_task", "beta_task"))
    for task, record in {
        "alpha_task": {
            "task_name": "alpha_task",
            "status": "policy_negative",
            "failure_kind": "policy",
            "policy_rollouts": 1,
            "wall_seconds": 1.0,
        },
        "beta_task": {
            "task_name": "beta_task",
            "status": "failed",
            "failure_kind": "task_context",
            "policy_rollouts": 0,
            "wall_seconds": 0.2,
        },
    }.items():
        path = output / "tasks" / task / "preflight.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")

    summary = _summary(config)

    assert summary["status_counts"] == {
        "failed": 1,
        "policy_negative": 1,
    }
    assert summary["failure_kind_counts"] == {
        "policy": 1,
        "task_context": 1,
    }
    assert summary["policy_rollout_count"] == 1


def test_breadth_resume_rejects_a_different_source_commit() -> None:
    contract = {
        "schema_version": 1,
        "harness": "robotwin_breadth_v1",
        "source_commit": "commit-a",
        "phase": "preflight",
    }
    previous = {**contract, "source_commit": "commit-b"}

    with pytest.raises(ValueError, match="source commit differs"):
        _validate_resume_contract(previous, contract)
