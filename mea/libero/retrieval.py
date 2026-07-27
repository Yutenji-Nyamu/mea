"""Compatibility-first retrieval for open-query LIBERO evaluation.

The query/concern is open text. Retrieval is intentionally *not* permission:
a semantically close BDDL is considered only after an artifact-backed profile
or an explicit run binding authorizes it. A binding authorizes a control
attempt; it does not claim checkpoint training support. The selected source
then receives a separate controlled-change authorization before TaskGen.
"""

from __future__ import annotations

import re
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .benchmark import LiberoContractError, TaskContract


_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _TOKEN_RE.findall(value)}


@dataclass(frozen=True)
class BDDLTaskRecord:
    suite: str
    task_id: int
    problem_name: str
    language: str
    bddl_path: str
    init_state_path: str
    objects: tuple[str, ...] = ()
    goal_predicates: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def from_task_contract(cls, contract: TaskContract) -> "BDDLTaskRecord":
        return cls(
            suite=contract.suite,
            task_id=contract.official_task_id,
            problem_name=contract.problem_name,
            language=contract.language,
            bddl_path=contract.bddl_path,
            init_state_path=contract.initial_state_source,
            objects=tuple(
                sorted(item for values in contract.objects.values() for item in values)
            ),
            goal_predicates=tuple(
                tuple(str(part) for part in predicate)
                for predicate in contract.goal_predicates
            ),
        )

    def semantic_text(self) -> str:
        goals = " ".join(" ".join(item) for item in self.goal_predicates)
        return " ".join(
            (self.suite, self.problem_name, self.language, " ".join(self.objects), goals)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyTaskCompatibility:
    """Artifact-backed support plus any explicit run binding.

    An explicit binding authorizes an official control; it does not claim that
    the checkpoint was trained on the bound task.
    """

    policy_name: str
    checkpoint: str
    declared_scope: str
    authorized_task_ids: Mapping[str, tuple[int, ...]]
    authorization_source: str
    artifact_evidence: Mapping[str, Any]
    authorized_problem_names: tuple[str, ...] = ()

    def authorizes(self, task: BDDLTaskRecord) -> bool:
        if task.task_id not in self.authorized_task_ids.get(task.suite, ()):
            return False
        return not self.authorized_problem_names or (
            task.problem_name in self.authorized_problem_names
        )

    def verdict(self, task: BDDLTaskRecord) -> str:
        if self.authorizes(task):
            return (
                "artifact_manifest_supported"
                if self.authorization_source == "artifact_manifest"
                else "explicit_run_binding_only"
            )
        return (
            "not_authorized_scope_unknown"
            if self.declared_scope == "unknown"
            else "not_supported_by_artifact_manifest"
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["authorized_task_ids"] = {
            suite: list(ids) for suite, ids in self.authorized_task_ids.items()
        }
        return value


@dataclass(frozen=True)
class BDDLRetrieval:
    query_concern: str
    selected: BDDLTaskRecord
    score: float
    authorized_candidate_count: int
    not_authorized_candidate_count: int
    selection_authorization: str
    route: str = "policy_authorized_nearest_bddl"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["selected"] = self.selected.to_dict()
        return value


@dataclass(frozen=True)
class ControlledChangeContract:
    """Authorization between retrieval and TaskGen.

    ``pending`` is used by plan-only before an evidence-conditioned Planner has
    proposed a concrete change.  TaskGen accepts only ``authorized``.
    """

    source_suite: str
    source_task_id: int
    source_problem_name: str
    query_concern: str
    requested_change_roots: tuple[str, ...]
    allowed_change_roots: tuple[str, ...]
    preserved_roots: tuple[str, ...]
    status: str
    reason: str

    @property
    def authorized(self) -> bool:
        return self.status == "authorized"

    def require_authorized(self) -> None:
        if not self.authorized:
            raise LiberoContractError(
                "LIBERO controlled-change contract is not authorized: " + self.reason
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["authorized"] = self.authorized
        return value


class BDDLTaskIndex:
    """Small in-memory index; callers may populate it from any LIBERO suite."""

    def __init__(self, tasks: Iterable[BDDLTaskRecord]):
        self.tasks = tuple(tasks)
        if not self.tasks:
            raise LiberoContractError("BDDL task index cannot be empty")

    @classmethod
    def from_contracts(cls, contracts: Iterable[TaskContract]) -> "BDDLTaskIndex":
        return cls(BDDLTaskRecord.from_task_contract(item) for item in contracts)

    @classmethod
    def from_libero_suite(cls, suite_name: str) -> "BDDLTaskIndex":
        """Index upstream BDDLs without assuming the current policy supports them."""

        from libero.libero import benchmark, get_libero_path

        from .benchmark import parse_bddl

        try:
            suite = benchmark.get_benchmark_dict()[suite_name]()
        except KeyError as exc:
            raise LiberoContractError(f"unknown LIBERO suite: {suite_name}") from exc
        bddl_root = Path(get_libero_path("bddl_files"))
        init_root = Path(get_libero_path("init_states"))
        records: list[BDDLTaskRecord] = []
        for task_id in range(suite.get_num_tasks()):
            task = suite.get_task(task_id)
            bddl_path = (bddl_root / task.problem_folder / task.bddl_file).resolve()
            init_path = (
                init_root / task.problem_folder / Path(task.init_states_file).name
            ).resolve()
            parsed = parse_bddl(bddl_path)
            records.append(
                BDDLTaskRecord(
                    suite=suite_name,
                    task_id=task_id,
                    problem_name=str(parsed["problem_name"]),
                    language=" ".join(
                        str(item) for item in parsed["language_instruction"]
                    ),
                    bddl_path=str(bddl_path),
                    init_state_path=str(init_path),
                    objects=tuple(
                        sorted(
                            item
                            for values in parsed.get("objects", {}).values()
                            for item in values
                        )
                    ),
                    goal_predicates=tuple(
                        tuple(str(part) for part in predicate)
                        for predicate in parsed.get("goal_state", [])
                    ),
                )
            )
        return cls(records)

    def retrieve_nearest(
        self,
        query_concern: str,
        *,
        compatibility: PolicyTaskCompatibility,
    ) -> BDDLRetrieval:
        concern = query_concern.strip()
        if not concern:
            raise LiberoContractError("open-query concern cannot be empty")
        authorized = [item for item in self.tasks if compatibility.authorizes(item)]
        if not authorized:
            raise LiberoContractError(
                "no BDDL candidate is authorized by an artifact profile or explicit binding"
            )
        concern_tokens = _tokens(concern)

        def score(item: BDDLTaskRecord) -> float:
            task_tokens = _tokens(item.semantic_text())
            union = concern_tokens | task_tokens
            return len(concern_tokens & task_tokens) / len(union) if union else 0.0

        ranked = sorted(
            authorized,
            key=lambda item: (-score(item), item.suite, item.task_id, item.problem_name),
        )
        selected = ranked[0]
        return BDDLRetrieval(
            query_concern=concern,
            selected=selected,
            score=round(score(selected), 6),
            authorized_candidate_count=len(authorized),
            not_authorized_candidate_count=len(self.tasks) - len(authorized),
            selection_authorization=compatibility.verdict(selected),
        )


_ALLOWED_ROOTS = ("language", "obj_of_interest", "goal")
_PRESERVED_ROOTS = (
    "suite",
    "task_id",
    "problem_name",
    "domain",
    "fixtures",
    "regions",
    "objects",
    "initial_state",
    "camera",
    "workspace",
    "action_mode",
    "horizon",
)


def pending_controlled_change(retrieval: BDDLRetrieval) -> ControlledChangeContract:
    return ControlledChangeContract(
        source_suite=retrieval.selected.suite,
        source_task_id=retrieval.selected.task_id,
        source_problem_name=retrieval.selected.problem_name,
        query_concern=retrieval.query_concern,
        requested_change_roots=(),
        allowed_change_roots=_ALLOWED_ROOTS,
        preserved_roots=_PRESERVED_ROOTS,
        status="pending",
        reason="awaiting an evidence-conditioned Planner proposal",
    )


def authorize_controlled_change(
    retrieval: BDDLRetrieval,
    proposal_bundle: Mapping[str, Any],
) -> ControlledChangeContract:
    proposal = proposal_bundle.get("proposal", proposal_bundle)
    perturbation = proposal.get("requested_perturbation", {})
    raw_changes = (
        perturbation.get("controlled_changes", ())
        if isinstance(perturbation, Mapping)
        else ()
    )
    requested = tuple(
        str(item).strip() for item in raw_changes if str(item).strip()
    )
    normalized = " ".join(requested).casefold()
    identity_change = re.search(
        r"\b(?:goal\s+object|object\s+identity|goal\s+identity)\b",
        normalized,
    )
    preserve_only = re.search(
        r"\b(?:preserve|keep|retain|unchanged|do\s+not\s+change)\b.{0,32}"
        r"\b(?:goal|object)\b",
        normalized,
    )
    represents_goal_identity = bool(identity_change and not preserve_only)
    status = "authorized" if represents_goal_identity else "unsupported"
    reason = (
        "existing-object goal identity can be expressed by the Phase-1 BDDL contract"
        if represents_goal_identity
        else (
            "the free-form concern is valid, but this TaskGen backend cannot express "
            "it without crossing the registered problem/policy compatibility boundary"
        )
    )
    return ControlledChangeContract(
        source_suite=retrieval.selected.suite,
        source_task_id=retrieval.selected.task_id,
        source_problem_name=retrieval.selected.problem_name,
        query_concern=retrieval.query_concern,
        requested_change_roots=requested,
        allowed_change_roots=_ALLOWED_ROOTS,
        preserved_roots=_PRESERVED_ROOTS,
        status=status,
        reason=reason,
    )


def smolvla_policy_compatibility(
    *,
    checkpoint: str | Path,
    explicit_task_binding: BDDLTaskRecord,
) -> PolicyTaskCompatibility:
    """Read a task manifest when available, otherwise expose unknown scope.

    The current public SmolVLA artifact says ``datasets: unknown`` and its
    config carries no task manifest.  In that case an explicit run binding
    authorizes exactly one *control attempt*, while training support remains
    unknown.  A future checkpoint can add ``policy_task_manifest.json`` with
    ``{"suite_tasks": {"libero_object": [0, ...]}}``.
    """

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    model_card = checkpoint_path / "README.md"
    config_path = checkpoint_path / "config.json"
    manifest_path = checkpoint_path / "policy_task_manifest.json"
    dataset_declaration = "missing"
    if model_card.is_file():
        match = re.search(
            r"(?m)^datasets:\s*(.+?)\s*$",
            model_card.read_text(encoding="utf-8"),
        )
        if match:
            dataset_declaration = match.group(1).strip()
    repo_id: Any = None
    if config_path.is_file():
        repo_id = json.loads(config_path.read_text(encoding="utf-8")).get("repo_id")
    evidence: dict[str, Any] = {
        "model_card": str(model_card),
        "model_card_datasets": dataset_declaration,
        "config": str(config_path),
        "config_repo_id": repo_id,
        "task_manifest": str(manifest_path) if manifest_path.is_file() else None,
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        suite_tasks = manifest.get("suite_tasks")
        if not isinstance(suite_tasks, Mapping) or not suite_tasks:
            raise LiberoContractError("policy_task_manifest.json has no suite_tasks")
        authorized = {
            str(suite): tuple(int(task_id) for task_id in task_ids)
            for suite, task_ids in suite_tasks.items()
        }
        count = sum(len(ids) for ids in authorized.values())
        return PolicyTaskCompatibility(
            policy_name="SmolVLA",
            checkpoint=str(checkpoint_path),
            declared_scope="single_task" if count == 1 else "multi_task",
            authorized_task_ids=authorized,
            authorization_source="artifact_manifest",
            artifact_evidence=evidence,
        )
    return PolicyTaskCompatibility(
        policy_name="SmolVLA",
        checkpoint=str(checkpoint_path),
        declared_scope="unknown",
        authorized_task_ids={
            explicit_task_binding.suite: (explicit_task_binding.task_id,)
        },
        authorized_problem_names=(explicit_task_binding.problem_name,),
        authorization_source="explicit_run_binding",
        artifact_evidence=evidence,
    )
