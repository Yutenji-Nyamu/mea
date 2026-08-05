"""Evidence persistence and next-step execution for Plan Agent applications."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from mea.plan_artifacts import (
    PLAN_AGENT_SESSION,
    PLAN_AGENT_STEPS,
    PROPOSAL_MATERIALIZATION,
)
from mea.taskgen.round_materialization import materialize_open_world_round

from .plan_agent_persistence import (
    _write_json,
    apply_external_hard_round_cap,
    refresh_plan_agent_capabilities_from_runtime_context,
    update_manifest,
)


class PlanAgentRuntimeDecisionMixin:
    def _persist_session(
        self,
        plan: Mapping[str, Any],
        round_runs: list[dict[str, Any]],
    ) -> None:
        _write_json(
            self.evaluation_dir / "plan/bound_task_session.json",
            self.session.snapshot(
                self.user_request,
                plan,
                [item["round_summary"] for item in round_runs],
            ),
        )

    def _observe(
        self,
        round_runs: list[dict[str, Any]],
        *,
        executed_rounds: int,
    ) -> dict[str, Any]:
        state = self.session.observe(
            [item["round_plan"] for item in round_runs],
            [item["round_summary"] for item in round_runs],
        )
        contract_candidate_ids = {
            str(item)
            for item in state["query_contract"].get("candidate_universe", [])
        }
        records_by_round = {
            str(record["round_id"]): record
            for record in state["records"]
        }
        if len(records_by_round) != len(round_runs):
            raise RuntimeError(
                "Plan Agent records are not one-to-one with completed runtime "
                "rounds"
            )
        for completed_run in round_runs:
            round_id = str(completed_run["round_plan"]["round_id"])
            record = records_by_round.get(round_id)
            if record is None:
                raise RuntimeError(
                    "Plan Agent record is missing for completed round "
                    f"{round_id!r}"
                )
            if record["candidate_id"] in contract_candidate_ids:
                completed_run["round_summary"]["candidate_evidence"] = deepcopy(
                    record["candidate_evidence"]
                )
        _write_json(
            self.evaluation_dir
            / PLAN_AGENT_SESSION
            / f"evidence_after_round_{executed_rounds:02d}.json",
            state,
        )
        return state

    def _persist_contract_stop(
        self,
        *,
        plan: dict[str, Any],
        round_plan: Mapping[str, Any],
        round_runs: list[dict[str, Any]],
        runtime_state: Mapping[str, Any],
        executed_rounds: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assessment = runtime_state["assessment"]
        query_answer = deepcopy(dict(runtime_state["query_answer"]))
        decision = {
            "schema_version": 3,
            "action": "stop",
            "transition": "stop",
            "next_aspect_id": None,
            "next_template_id": None,
            "observation_summary": assessment["rationale"],
            "decision_reason": "external_query_contract_stop",
            "answered_query": bool(query_answer["answered"]),
            "plan_step_source": "external_query_contract_stop",
            "round_budget_before_decision": assessment["budget_remaining"],
            "evidence_assessment": assessment,
            "semantic_stop_step": None,
            "next_round": None,
        }
        plan.setdefault("round_decisions", []).append(decision)
        plan["planning_state"] = (
            f"stopped_after_round_{executed_rounds}_"
            f"{assessment['stop_reason']}"
        )
        _write_json(
            self.evaluation_dir
            / f"plan/decision_after_{round_plan['round_id']}.json",
            decision,
        )
        _write_json(
            self.evaluation_dir / PLAN_AGENT_SESSION / "query_answer.json",
            query_answer,
        )
        _write_json(
            self.evaluation_dir / "plan/evaluation_plan.json",
            plan,
        )
        self._persist_session(plan, round_runs)
        update_manifest(
            self.evaluation_dir,
            status=plan["planning_state"],
            plan=plan,
            plan_agent_stop={
                "stop_reason": assessment["stop_reason"],
                "evidence_sufficient": assessment["evidence_sufficient"],
                "answered_query": query_answer["answered"],
                "answer_path": (
                    (PLAN_AGENT_SESSION / "query_answer.json").as_posix()
                ),
                "plan_agent_stop_proposed": False,
            },
        )
        return decision, query_answer

    def _decide_next_step(
        self,
        *,
        plan: dict[str, Any],
        round_plan: Mapping[str, Any],
        round_runs: list[dict[str, Any]],
        runtime_state: Mapping[str, Any],
        executed_rounds: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        observation_history = [
            item["round_summary"] for item in round_runs
        ]
        executed_candidate_ids = [
            str(
                item["round_plan"].get("candidate_id")
                or item["round_plan"].get("template_id")
            )
            for item in round_runs
        ]
        step_dir = (
            self.evaluation_dir
            / PLAN_AGENT_STEPS
            / f"after_round_{executed_rounds:02d}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)
        self.capabilities = (
            refresh_plan_agent_capabilities_from_runtime_context(
                self.capabilities,
                round_runs[-1]["child_manifest"],
            )
        )
        _write_json(
            step_dir / "runtime_capabilities.json",
            self.capabilities,
        )
        semantic_bundle = self.session.propose_semantic_step(
            self.agent,
            runtime_state,
            capabilities=self.capabilities,
            evaluation_intent=None,
        )
        (step_dir / "prompt.md").write_text(
            self.agent.last_prompt or "",
            encoding="utf-8",
        )
        for index, response in enumerate(
            self.agent.last_responses,
            start=1,
        ):
            (step_dir / f"response_{index}.txt").write_text(
                response + "\n",
                encoding="utf-8",
            )
        _write_json(
            step_dir / "semantic_proposal_bundle.json",
            semantic_bundle,
        )
        raw_proposal = semantic_bundle.get("proposal")
        if not isinstance(raw_proposal, Mapping):
            raise RuntimeError(
                "Plan Agent decision artifact has no semantic proposal"
            )

        if (
            raw_proposal.get("action") != "stop"
            and self.max_agent_rounds is not None
            and executed_rounds >= self.max_agent_rounds
        ):
            cap_artifact = {
                "schema_version": 1,
                "status": "continue_rejected_by_external_hard_cap",
                "semantic_proposal_bundle": deepcopy(semantic_bundle),
                "planning_lineage": deepcopy(
                    semantic_bundle.get("planning_lineage")
                ),
                "query_contract": deepcopy(self.session.query_contract),
                "plan_step": {
                    "action": "continue",
                    "sub_aspect": raw_proposal.get("sub_aspect"),
                    "next_round": None,
                },
            }
            _write_json(step_dir / "bound_semantic_step.json", cap_artifact)
            artifact_path = (
                f"{PLAN_AGENT_STEPS.as_posix()}/"
                f"after_round_{executed_rounds:02d}/bound_semantic_step.json"
            )
            update_manifest(
                self.evaluation_dir,
                last_plan_agent_step={
                    "status": "continue_rejected_by_external_hard_cap",
                    "after_round": executed_rounds,
                    "action": "continue",
                    "answered_query": False,
                    "semantic_sub_aspect": raw_proposal.get("sub_aspect"),
                    "resolved_template_id": None,
                    "resolved_candidate_id": None,
                    "evidence_conditioned": bool(
                        semantic_bundle.get("planning_lineage", {}).get(
                            "evidence_conditioned"
                        )
                    ),
                    "planning_lineage": deepcopy(
                        semantic_bundle.get("planning_lineage")
                    ),
                    "artifact_path": artifact_path,
                },
            )
            capped_plan, decision, _ = apply_external_hard_round_cap(
                evaluation_dir=self.evaluation_dir,
                plan=plan,
                round_runs=round_runs,
                executed_rounds=executed_rounds,
                max_agent_rounds=self.max_agent_rounds,
                user_request=self.user_request,
                bound_plan_session=self.session,
                plan_agent_proposal=raw_proposal,
                plan_agent_artifact_path=artifact_path,
            )
            return capped_plan, decision, None

        bound_step = self.session.bind_evidence_conditioned_semantic_step(
            semantic_bundle,
            runtime_state,
            capabilities=self.capabilities,
            executed_candidate_ids=executed_candidate_ids,
            evaluation_intent=None,
        )
        _write_json(step_dir / "bound_semantic_step.json", bound_step)
        plan_step = bound_step["plan_step"]
        query_answer: dict[str, Any] | None = None
        materialized_round = None
        if plan_step["action"] == "stop":
            raw_query_answer = bound_step.get("query_answer")
            if not isinstance(raw_query_answer, Mapping):
                raise RuntimeError(
                    "validated Plan Agent stop has no Query answer"
                )
            query_answer = deepcopy(dict(raw_query_answer))
            _write_json(
                self.evaluation_dir
                / PLAN_AGENT_SESSION
                / "query_answer.json",
                query_answer,
            )
        else:
            dynamic_candidate = (
                plan_step.get("proposal")
                or plan_step.get("experiment_candidate")
            )
            if not isinstance(dynamic_candidate, Mapping):
                raise RuntimeError(
                    "Plan Agent must bind every continue decision to a typed "
                    "Proposal before execution"
                )
            next_round_number = len(plan["rounds"]) + 1
            materialized_round, open_tool_bundle = materialize_open_world_round(
                self.repo_root,
                evaluation_dir=self.evaluation_dir,
                round_number=next_round_number,
                candidate=dynamic_candidate,
                control_execution=plan["rounds"][0]["execution"],
                policy_backend=self.policy_backend,
            )
            bound_step["execution_binding"] = {
                "schema_version": 2,
                "candidate_id": dynamic_candidate["candidate_id"],
                "materialization_path": (
                    f"{PROPOSAL_MATERIALIZATION.as_posix()}/"
                    f"round_{next_round_number:02d}"
                ),
                "taskgen_route": materialized_round["route"],
                "toolgen_route": open_tool_bundle["source"],
                "catalog_template_used": False,
                "retrieval_template_hint": bound_step["resolution"].get(
                    "retrieval_template_id"
                ),
            }
            _write_json(step_dir / "bound_semantic_step.json", bound_step)

        next_plan, decision, runtime_directive = self.session.apply_plan_step(
            plan,
            observation_history,
            plan_step,
            materialized_round=materialized_round,
            source=str(
                semantic_bundle.get("source")
                or "provider_plan_agent_open_query"
            ),
            query_contract=bound_step.get("query_contract"),
        )
        decision["semantic_proposal"] = deepcopy(semantic_bundle["proposal"])
        decision["semantic_resolution"] = deepcopy(bound_step["resolution"])
        _write_json(
            self.evaluation_dir
            / f"plan/runtime_directive_after_{round_plan['round_id']}.json",
            {
                "schema_version": 1,
                "owner": type(self.session).__name__,
                "adapter_role": (
                    "plan_agent_stop_validated_by_query_contract"
                    if plan_step["action"] == "stop"
                    else "plan_agent_retrieve_or_generate_and_adjudicate"
                ),
                **runtime_directive,
            },
        )
        artifact_path = (
            f"{PLAN_AGENT_STEPS.as_posix()}/"
            f"after_round_{executed_rounds:02d}/bound_semantic_step.json"
        )
        update_manifest(
            self.evaluation_dir,
            last_plan_agent_step={
                "status": "transition_applied",
                "after_round": executed_rounds,
                "action": plan_step["action"],
                "answered_query": bool(
                    plan_step.get("answered_query", False)
                ),
                "semantic_sub_aspect": semantic_bundle["proposal"].get(
                    "sub_aspect"
                ),
                "resolved_template_id": plan_step.get("template_id"),
                "resolved_candidate_id": plan_step.get("candidate_id"),
                "evidence_conditioned": bool(
                    bound_step.get("planning_lineage", {}).get(
                        "evidence_conditioned"
                    )
                ),
                "planning_lineage": deepcopy(
                    bound_step.get("planning_lineage")
                ),
                "artifact_path": artifact_path,
            },
        )
        if plan_step["action"] == "stop":
            update_manifest(
                self.evaluation_dir,
                plan_agent_stop={
                    "stop_reason": plan_step.get("stop_reason"),
                    "evidence_sufficient": True,
                    "answered_query": True,
                    "answer_path": (
                        (PLAN_AGENT_SESSION / "query_answer.json").as_posix()
                    ),
                    "plan_agent_stop_proposed": True,
                    "artifact_path": artifact_path,
                },
            )
        next_plan = self.session.normalize_plan(next_plan)
        if decision.get("next_round") is not None:
            decision["next_round"] = next_plan["rounds"][-1]
        _write_json(
            self.evaluation_dir / "plan/evaluation_plan.json",
            next_plan,
        )
        _write_json(
            self.evaluation_dir
            / f"plan/decision_after_{round_plan['round_id']}.json",
            decision,
        )
        update_manifest(
            self.evaluation_dir,
            status=next_plan.get("planning_state"),
            plan=next_plan,
        )
        self._persist_session(next_plan, round_runs)
        return next_plan, decision, query_answer



__all__ = ["PlanAgentRuntimeDecisionMixin"]

