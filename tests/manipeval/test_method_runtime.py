from __future__ import annotations

from pathlib import Path

import pytest

from mea.method_runtime import (
    CandidateRequest,
    EvidenceRequest,
    MaterializedCandidate,
    MethodRuntime,
    RolloutObservation,
    RolloutRequest,
    BackendBindingRequest,
    BackendTaskBinding,
    build_round_evidence,
)


class _FixtureBackend:
    benchmark = "fixture"

    def bind_task(
        self, _request: BackendBindingRequest
    ) -> BackendTaskBinding:
        return BackendTaskBinding(
            benchmark=self.benchmark,
            binding_id="fixture/task0",
            task_contract={"task": 0},
            native_task=object(),
        )

    def materialize_candidate(
        self,
        binding: BackendTaskBinding,
        request: CandidateRequest,
    ) -> MaterializedCandidate:
        return MaterializedCandidate(
            benchmark=self.benchmark,
            candidate_id=request.candidate_id,
            binding_id=binding.binding_id,
            source_query=request.source_query,
            task_contract={"task": 0, "generated": True},
            native_task=object(),
        )

    def rollout(
        self,
        candidate: MaterializedCandidate,
        request: RolloutRequest,
    ) -> RolloutObservation:
        return RolloutObservation(
            benchmark=self.benchmark,
            round_id=request.round_id,
            candidate_id=candidate.candidate_id,
            seed=request.seed,
            success=True,
            episode={"success": True},
            native_episode=object(),
        )

    def evidence(
        self,
        rollout: RolloutObservation,
        request: EvidenceRequest,
    ):
        return build_round_evidence(rollout, request)


def test_method_runtime_preserves_identity_across_outer_loop(tmp_path: Path) -> None:
    runtime = MethodRuntime(_FixtureBackend())
    binding = runtime.bind_task(
        BackendBindingRequest(task_reference={"task": 0})
    )
    candidate = runtime.materialize_candidate(
        binding,
        CandidateRequest(
            candidate_id="planner-owned-candidate",
            source_query="Can this policy handle the proposed variation?",
            proposal_bundle={"proposal": {"action": "continue"}},
            output_dir=tmp_path / "taskgen",
            seed=7,
        ),
    )
    rollout = runtime.rollout(
        candidate,
        RolloutRequest(
            round_id="round_01",
            seed=7,
            output_dir=tmp_path / "episode",
        ),
    )
    evidence = runtime.evidence(
        rollout,
        EvidenceRequest(
            sub_aspect="object identity",
            hypothesis="The variation remains solvable.",
            perturbation="replace the target object",
            summary="The fixture rollout succeeded.",
            limitations=("N=1",),
        ),
    )

    assert binding.binding_id == candidate.binding_id
    assert candidate.candidate_id == rollout.candidate_id
    assert evidence.candidate_id == candidate.candidate_id
    assert evidence.outcome == "success"
    assert set(evidence.to_planner_dict()) == {
        "schema_version",
        "round_id",
        "tested_sub_aspect",
        "tested_hypothesis",
        "tested_perturbation",
        "outcome",
        "evidence_summary",
        "limitations",
    }


def test_method_runtime_rejects_backend_candidate_identity_rewrite(
    tmp_path: Path,
) -> None:
    class RewritingBackend(_FixtureBackend):
        def materialize_candidate(
            self,
            binding: BackendTaskBinding,
            request: CandidateRequest,
        ) -> MaterializedCandidate:
            candidate = super().materialize_candidate(binding, request)
            return MaterializedCandidate(
                benchmark=candidate.benchmark,
                candidate_id="backend-rewrite",
                binding_id=candidate.binding_id,
                source_query=candidate.source_query,
                task_contract=candidate.task_contract,
                native_task=candidate.native_task,
            )

    runtime = MethodRuntime(RewritingBackend())
    with pytest.raises(ValueError, match="candidate_id"):
        runtime.materialize_candidate(
            runtime.bind_task(
                BackendBindingRequest(task_reference={"task": 0})
            ),
            CandidateRequest(
                candidate_id="planner-owned-candidate",
                source_query="test query",
                proposal_bundle={},
                output_dir=tmp_path,
                seed=1,
            ),
        )
