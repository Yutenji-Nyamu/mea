"""Simulator-neutral outer runtime contract for the MEA method loop.

The paper-level loop is independent of whether a candidate is materialized as
RoboTwin Python or a LIBERO BDDL problem:

``bind task -> materialize candidate -> rollout -> evidence``.

Simulator backends own native task objects, environment factories, and policy
calls.  The outer runtime owns only the typed hand-off between those phases.
Keeping this boundary explicit prevents a benchmark adapter from growing a
second Plan/TaskGen/Aggregate implementation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


@dataclass(frozen=True)
class BackendBindingRequest:
    """Executable task identity supplied after semantic planning."""

    task_reference: Mapping[str, Any]
    artifacts: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_reference", _mapping(self.task_reference))
        object.__setattr__(self, "artifacts", dict(self.artifacts))
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True)
class BackendTaskBinding:
    """One executable policy/task binding, before an experiment is proposed."""

    benchmark: str
    binding_id: str
    task_contract: Mapping[str, Any]
    native_task: Any = field(repr=False, compare=False)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", _required_text(self.benchmark, "benchmark"))
        object.__setattr__(
            self, "binding_id", _required_text(self.binding_id, "binding_id")
        )
        object.__setattr__(self, "task_contract", _mapping(self.task_contract))
        object.__setattr__(self, "artifacts", dict(self.artifacts))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": self.benchmark,
            "binding_id": self.binding_id,
            "task_contract": _mapping(self.task_contract),
            "artifacts": dict(self.artifacts),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class CandidateRequest:
    """Paper-level Proposal handed to a simulator-specific TaskGen backend."""

    candidate_id: str
    source_query: str
    proposal_bundle: Mapping[str, Any]
    output_dir: Path
    seed: int
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _required_text(self.candidate_id, "candidate_id")
        )
        object.__setattr__(
            self, "source_query", _required_text(self.source_query, "source_query")
        )
        object.__setattr__(self, "proposal_bundle", _mapping(self.proposal_bundle))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "context", _mapping(self.context))


@dataclass(frozen=True)
class MaterializedCandidate:
    """A validated candidate task ready for one policy rollout."""

    benchmark: str
    candidate_id: str
    binding_id: str
    source_query: str
    task_contract: Mapping[str, Any]
    native_task: Any = field(repr=False, compare=False)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("benchmark", "candidate_id", "binding_id", "source_query"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "task_contract", _mapping(self.task_contract))
        object.__setattr__(self, "artifacts", dict(self.artifacts))
        object.__setattr__(self, "validation", _mapping(self.validation))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": self.benchmark,
            "candidate_id": self.candidate_id,
            "binding_id": self.binding_id,
            "source_query": self.source_query,
            "task_contract": _mapping(self.task_contract),
            "artifacts": dict(self.artifacts),
            "validation": _mapping(self.validation),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class RolloutRequest:
    round_id: str
    seed: int
    output_dir: Path
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_id", _required_text(self.round_id, "round_id"))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "provenance", _mapping(self.provenance))


@dataclass(frozen=True)
class RolloutObservation:
    """Compact outer-loop projection of one native simulator episode."""

    benchmark: str
    round_id: str
    candidate_id: str
    seed: int
    success: bool
    episode: Mapping[str, Any]
    native_episode: Any = field(repr=False, compare=False)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("benchmark", "round_id", "candidate_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "episode", _mapping(self.episode))
        object.__setattr__(self, "artifacts", dict(self.artifacts))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": self.benchmark,
            "round_id": self.round_id,
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "success": self.success,
            "episode": _mapping(self.episode),
            "artifacts": dict(self.artifacts),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class EvidenceRequest:
    sub_aspect: str
    hypothesis: str
    perturbation: str
    summary: str
    limitations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("sub_aspect", "hypothesis", "perturbation", "summary"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "limitations",
            tuple(str(item).strip() for item in self.limitations if str(item).strip()),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))


@dataclass(frozen=True)
class RoundEvidence:
    benchmark: str
    round_id: str
    candidate_id: str
    tested_sub_aspect: str
    tested_hypothesis: str
    tested_perturbation: str
    outcome: str
    evidence_summary: str
    limitations: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": self.benchmark,
            "round_id": self.round_id,
            "candidate_id": self.candidate_id,
            "tested_sub_aspect": self.tested_sub_aspect,
            "tested_hypothesis": self.tested_hypothesis,
            "tested_perturbation": self.tested_perturbation,
            "outcome": self.outcome,
            "evidence_summary": self.evidence_summary,
            "limitations": list(self.limitations),
            "metadata": _mapping(self.metadata),
        }

    def to_planner_dict(self) -> dict[str, Any]:
        """Project onto the strict public ClaimFirst evidence schema."""

        return {
            "schema_version": 1,
            "round_id": self.round_id,
            "tested_sub_aspect": self.tested_sub_aspect,
            "tested_hypothesis": self.tested_hypothesis,
            "tested_perturbation": self.tested_perturbation,
            "outcome": self.outcome,
            "evidence_summary": self.evidence_summary,
            "limitations": list(self.limitations),
        }


@runtime_checkable
class MethodBackend(Protocol):
    """The only benchmark-specific interface required by the outer loop."""

    benchmark: str

    def bind_task(self, request: BackendBindingRequest) -> BackendTaskBinding:
        ...

    def materialize_candidate(
        self,
        binding: BackendTaskBinding,
        request: CandidateRequest,
    ) -> MaterializedCandidate:
        ...

    def rollout(
        self,
        candidate: MaterializedCandidate,
        request: RolloutRequest,
    ) -> RolloutObservation:
        ...

    def evidence(
        self,
        rollout: RolloutObservation,
        request: EvidenceRequest,
    ) -> RoundEvidence:
        ...


class MethodRuntime:
    """Thin invariant-preserving delegate shared by simulator backends."""

    def __init__(self, backend: MethodBackend) -> None:
        self.backend = backend

    def bind_task(
        self, request: BackendBindingRequest
    ) -> BackendTaskBinding:
        binding = self.backend.bind_task(request)
        self._same_benchmark(binding.benchmark)
        return binding

    def materialize_candidate(
        self,
        binding: BackendTaskBinding,
        request: CandidateRequest,
    ) -> MaterializedCandidate:
        self._same_benchmark(binding.benchmark)
        candidate = self.backend.materialize_candidate(binding, request)
        self._same_benchmark(candidate.benchmark)
        if candidate.binding_id != binding.binding_id:
            raise ValueError("materialized candidate changed the bound policy/task")
        if candidate.candidate_id != request.candidate_id:
            raise ValueError("backend changed the planner-owned candidate_id")
        return candidate

    def rollout(
        self,
        candidate: MaterializedCandidate,
        request: RolloutRequest,
    ) -> RolloutObservation:
        self._same_benchmark(candidate.benchmark)
        observation = self.backend.rollout(candidate, request)
        self._same_benchmark(observation.benchmark)
        if observation.candidate_id != candidate.candidate_id:
            raise ValueError("rollout evidence changed candidate identity")
        if observation.round_id != request.round_id:
            raise ValueError("rollout evidence changed round identity")
        return observation

    def evidence(
        self,
        rollout: RolloutObservation,
        request: EvidenceRequest,
    ) -> RoundEvidence:
        self._same_benchmark(rollout.benchmark)
        evidence = self.backend.evidence(rollout, request)
        self._same_benchmark(evidence.benchmark)
        if evidence.round_id != rollout.round_id:
            raise ValueError("evidence changed round identity")
        if evidence.candidate_id != rollout.candidate_id:
            raise ValueError("evidence changed candidate identity")
        return evidence

    def _same_benchmark(self, value: str) -> None:
        if value != self.backend.benchmark:
            raise ValueError(
                f"backend={self.backend.benchmark!r} cannot handle benchmark={value!r}"
            )


def build_round_evidence(
    rollout: RolloutObservation,
    request: EvidenceRequest,
) -> RoundEvidence:
    """Default evidence projection usable by both simulator families."""

    return RoundEvidence(
        benchmark=rollout.benchmark,
        round_id=rollout.round_id,
        candidate_id=rollout.candidate_id,
        tested_sub_aspect=request.sub_aspect,
        tested_hypothesis=request.hypothesis,
        tested_perturbation=request.perturbation,
        outcome="success" if rollout.success else "failure",
        evidence_summary=request.summary,
        limitations=request.limitations,
        metadata=request.metadata,
    )


__all__ = [
    "CandidateRequest",
    "EvidenceRequest",
    "MaterializedCandidate",
    "MethodBackend",
    "MethodRuntime",
    "RolloutObservation",
    "RolloutRequest",
    "RoundEvidence",
    "BackendBindingRequest",
    "BackendTaskBinding",
    "build_round_evidence",
]
