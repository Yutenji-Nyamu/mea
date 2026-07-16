"""Deterministic one-round planner for unchanged schema-backed tasks."""

from __future__ import annotations

import json
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from mea.toolgen import official_success_tool_request
from mea.toolkit import load_task_schema

from .prototype import PlanAgentError, make_evaluation_id


OFFICIAL_TEMPLATE_ID = "task_execution.official_baseline"
OFFICIAL_GATES = ["render", "rule"]
OFFICIAL_POST_EXECUTION_GATES = ["toolkit", "planned_tool", "aggregate"]
EXECUTION_BACKENDS = {"expert", "act", "both"}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_head(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


class OfficialTaskPlanAgent:
    """Plan exactly one schema-backed official-task execution.

    This route intentionally does not ask an LLM to invent a sub-aspect or
    source change.  It proves cross-task execution independently from the
    BeatBlockHammer-specific natural-language TaskGen prototype.
    """

    def __init__(
        self,
        repo_root: str | Path,
        *,
        task_name: str,
        task_module: str | None = None,
        start_seed: int = 100000,
        num_episodes: int = 1,
        telemetry_profile: str = "balanced_v1",
        execution_backend: str = "expert",
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.task_name = task_name
        self.task_module = task_module or f"envs.{task_name}"
        self.start_seed = int(start_seed)
        self.num_episodes = int(num_episodes)
        self.telemetry_profile = telemetry_profile
        self.execution_backend = str(execution_backend).casefold()
        if self.num_episodes <= 0:
            raise PlanAgentError("num_episodes must be positive")
        if self.execution_backend not in EXECUTION_BACKENDS:
            raise PlanAgentError(
                "execution_backend must be one of expert, act, both"
            )
        load_task_schema(self.repo_root, task_name)

    def _round(self, user_request: str) -> dict[str, Any]:
        seeds = [self.start_seed + index for index in range(self.num_episodes)]
        return {
            "round_id": "round_1",
            "template_id": OFFICIAL_TEMPLATE_ID,
            "sub_aspect": "task_execution.official_baseline",
            "rationale": (
                "Run the unchanged schema-backed task through the requested "
                f"{self.execution_backend} execution backend."
            ),
            "task_instruction": user_request,
            "task_name": self.task_name,
            "task_module": self.task_module,
            "telemetry_profile": self.telemetry_profile,
            "route": "official",
            "variant_hint": {},
            "execution": {
                "backend": self.execution_backend,
                "seeds": seeds,
                "num_episodes": len(seeds),
                "gates": (
                    list(OFFICIAL_GATES)
                    + (
                        ["expert"]
                        if self.execution_backend in {"expert", "both"}
                        else []
                    )
                    + (
                        ["act"]
                        if self.execution_backend in {"act", "both"}
                        else []
                    )
                    + list(OFFICIAL_POST_EXECUTION_GATES)
                ),
            },
            "observations": [
                "scene_alignment",
                "expert_solvable",
                "trusted_tools",
                "aggregate",
            ],
            "tool_request": official_success_tool_request(self.task_name),
        }

    def plan(
        self,
        user_request: str,
        *,
        evaluation_id: str | None = None,
        history_context: list[dict[str, Any]] | None = None,
        history_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = str(user_request).strip()
        if not request:
            raise PlanAgentError("user_request must be non-empty")
        resolved_id = evaluation_id or make_evaluation_id()
        if not re.fullmatch(r"eval_[A-Za-z0-9_]+", resolved_id):
            raise PlanAgentError("evaluation_id must begin with 'eval_'")
        evaluation_dir = self.repo_root / "mea/evaluation_runs" / resolved_id
        if evaluation_dir.exists():
            raise PlanAgentError(f"evaluation directory already exists: {evaluation_dir}")
        for child in ("plan", "execution", "summary"):
            (evaluation_dir / child).mkdir(parents=True, exist_ok=False)

        compact_history = [
            {
                key: item.get(key)
                for key in ("evaluation_id", "user_request", "task_name", "similarity")
            }
            for item in (history_context or [])
            if isinstance(item, dict)
        ]
        history_retrieval = {
            "schema_version": 1,
            "status": "passed" if compact_history else "empty",
            "match_count": len(compact_history),
            "matches": compact_history,
            **deepcopy(history_metadata or {}),
        }
        plan = {
            "schema_version": 5,
            "task_name": self.task_name,
            "policy": {
                "name": "expert" if self.execution_backend == "expert" else "ACT",
                "checkpoint_setting": "demo_clean",
                "expert_data_num": (
                    None if self.execution_backend == "expert" else 50
                ),
                "language_conditioned": False,
            },
            "evaluation_goal": "validate_schema_backed_official_task_execution",
            "requested_template_ids": [OFFICIAL_TEMPLATE_ID],
            "rounds": [self._round(request)],
            "round_decisions": [],
            "max_rounds": 1,
            "planning_state": "awaiting_round_1_observation",
        }
        manifest = {
            "schema_version": 5,
            "evaluation_id": resolved_id,
            "status": "planned_round_1",
            "created_at": datetime.now().astimezone().isoformat(),
            "user_request": request,
            "base_commit": _git_head(self.repo_root),
            "planner": {
                "kind": "deterministic_official_task",
                "model_requested": None,
                "provider_called": False,
            },
            "plan_path": "plan/evaluation_plan.json",
            "history_retrieval_path": "plan/history_retrieval.json",
            "history_retrieval": history_retrieval,
            "plan": plan,
        }
        _write_json(evaluation_dir / "request.json", {"user_request": request})
        _write_json(evaluation_dir / "plan/history_retrieval.json", history_retrieval)
        _write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
        _write_json(evaluation_dir / "manifest.json", manifest)
        return manifest

    def decide_next_round(
        self,
        *,
        evaluation_id: str,
        user_request: str,
        current_plan: dict[str, Any],
        observation_history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if len(observation_history) != 1 or len(current_plan.get("rounds", [])) != 1:
            raise PlanAgentError("official task planner expects exactly one completed round")
        evaluation_dir = self.repo_root / "mea/evaluation_runs" / evaluation_id
        assessment = {
            "schema_version": 1,
            "state": "official_baseline_complete",
            "required_action": "stop",
            "pipeline_passed": bool(observation_history[0].get("pipeline_passed")),
            "reason": "The bounded official-task vertical slice contains one round.",
        }
        decision = {
            "schema_version": 2,
            "action": "stop",
            "observation_summary": (
                "The official schema-backed task round completed; inspect its "
                "pipeline status and aggregate evidence."
            ),
            "decision_reason": assessment["reason"],
            "next_template_id": None,
            "remaining_template_ids_before_decision": [],
            "round_budget_before_decision": 0,
            "evidence_assessment": assessment,
            "next_round": None,
        }
        updated = deepcopy(current_plan)
        updated.setdefault("round_decisions", []).append(decision)
        updated["planning_state"] = "stopped_after_round_1"
        _write_json(evaluation_dir / "plan/evidence_after_round_1.json", assessment)
        _write_json(evaluation_dir / "plan/decision_after_round_1.json", decision)
        _write_json(evaluation_dir / "plan/evaluation_plan.json", updated)
        manifest_path = evaluation_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": updated["planning_state"],
                "plan": updated,
                "decision_after_round_1_path": "plan/decision_after_round_1.json",
                "evidence_after_round_1_path": "plan/evidence_after_round_1.json",
            }
        )
        _write_json(manifest_path, manifest)
        return updated, decision
