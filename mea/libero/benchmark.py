"""LIBERO benchmark contract and custom-BDDL environment factory.

Imports from LIBERO/LeRobot stay lazy so the regular RoboTwin CLI remains
usable in its existing environment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


class LiberoContractError(ValueError):
    """Raised when a generated task crosses the Phase-1 compatibility boundary."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class TaskContract:
    """Benchmark-specific extension of the existing task artifact envelope."""

    schema_version: int
    benchmark: str
    suite: str
    official_task_id: int
    bddl_path: str
    bddl_sha256: str
    problem_name: str
    domain: str
    language: str
    objects: Mapping[str, list[str]]
    regions: list[str]
    initial_state_sha256: str
    goal_predicates: list[list[str]]
    python_problem_impl: str
    initial_state_source: str
    control_mode: str = "relative"
    horizon_steps: int = 100
    source_query: str | None = None
    proposal_artifact: str | None = None
    validation: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeRecord:
    """Compact LIBERO projection into the current episode/telemetry envelope."""

    schema_version: int
    benchmark: str
    policy_name: str
    checkpoint: str
    suite: str
    task_id: str
    seed: int
    horizon_steps: int
    executed_steps: int
    success: bool
    reward_sum: float
    goal_predicate_satisfied: bool
    elapsed_seconds: float
    bddl_path: str
    video_path: str | None
    task_contract_path: str
    actions_path: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_bddl(path: str | Path) -> dict[str, Any]:
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    parsed = robosuite_parse_problem(str(Path(path).expanduser().resolve()))
    if not isinstance(parsed, dict):
        raise LiberoContractError("LIBERO parser did not return an object")
    return parsed


def _problem_registry() -> Mapping[str, type]:
    # Importing the problem package performs the upstream class registration.
    import libero.libero.envs.problems  # noqa: F401
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING

    return TASK_MAPPING


def validate_phase1_bddl(
    *,
    base_path: str | Path,
    candidate_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Allow only language/interest/goal edits over a registered problem class."""

    base = parse_bddl(base_path)
    candidate = parse_bddl(candidate_path)
    registry = _problem_registry()
    checks = {
        "problem_class_registered": candidate.get("problem_name") in registry,
        "same_problem_name": candidate.get("problem_name") == base.get("problem_name"),
        "same_fixtures": candidate.get("fixtures") == base.get("fixtures"),
        "same_regions": candidate.get("regions") == base.get("regions"),
        "same_objects": candidate.get("objects") == base.get("objects"),
        "same_initial_state": candidate.get("initial_state") == base.get("initial_state"),
        "language_changed": candidate.get("language_instruction")
        != base.get("language_instruction"),
        "goal_changed": candidate.get("goal_state") != base.get("goal_state"),
    }
    goal = candidate.get("goal_state")
    all_objects = {
        item for values in candidate.get("objects", {}).values() for item in values
    }
    checks["single_in_goal"] = bool(
        isinstance(goal, list)
        and len(goal) == 1
        and isinstance(goal[0], list)
        and len(goal[0]) == 3
        and str(goal[0][0]).casefold() == "in"
        and goal[0][1] in all_objects
        and goal[0][2] in candidate.get("regions", {})
    )
    interest = candidate.get("obj_of_interest")
    checks["interest_matches_goal"] = bool(
        checks["single_in_goal"]
        and isinstance(interest, list)
        and set(interest) == {goal[0][1], "basket_1"}
    )
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise LiberoContractError("Phase-1 BDDL validation failed: " + ", ".join(failed))
    return base, candidate, checks


def build_task_contract(
    *,
    candidate_path: str | Path,
    candidate: Mapping[str, Any],
    checks: Mapping[str, Any],
    official_init_state_path: str | Path,
    source_query: str,
    proposal_artifact: str,
) -> TaskContract:
    path = Path(candidate_path).expanduser().resolve()
    language = " ".join(str(item) for item in candidate["language_instruction"])
    problem_name = str(candidate["problem_name"])
    registry = _problem_registry()
    impl = registry[problem_name]
    return TaskContract(
        schema_version=1,
        benchmark="libero",
        suite="libero_object",
        official_task_id=0,
        bddl_path=str(path),
        bddl_sha256=_sha256_bytes(path.read_bytes()),
        problem_name=problem_name,
        domain="robosuite",
        language=language,
        objects=dict(candidate["objects"]),
        regions=sorted(candidate["regions"]),
        initial_state_sha256=_sha256_bytes(
            _canonical(candidate["initial_state"]).encode("utf-8")
        ),
        goal_predicates=[list(item) for item in candidate["goal_state"]],
        python_problem_impl=f"{impl.__module__}.{impl.__name__}",
        initial_state_source=str(Path(official_init_state_path).expanduser().resolve()),
        source_query=source_query,
        proposal_artifact=proposal_artifact,
        validation=dict(checks),
    )


def build_official_task_contract() -> TaskContract:
    bddl_path, init_path = LiberoBenchmarkAdapter.official_paths()
    parsed = parse_bddl(bddl_path)
    problem_name = str(parsed["problem_name"])
    impl = _problem_registry()[problem_name]
    return TaskContract(
        schema_version=1,
        benchmark="libero",
        suite="libero_object",
        official_task_id=0,
        bddl_path=str(bddl_path),
        bddl_sha256=_sha256_bytes(bddl_path.read_bytes()),
        problem_name=problem_name,
        domain="robosuite",
        language=" ".join(str(item) for item in parsed["language_instruction"]),
        objects=dict(parsed["objects"]),
        regions=sorted(parsed["regions"]),
        initial_state_sha256=_sha256_bytes(
            _canonical(parsed["initial_state"]).encode("utf-8")
        ),
        goal_predicates=[list(item) for item in parsed["goal_state"]],
        python_problem_impl=f"{impl.__module__}.{impl.__name__}",
        initial_state_source=str(init_path),
        source_query=None,
        proposal_artifact=None,
        validation={"official_upstream_task": True},
    )


class LiberoBenchmarkAdapter:
    """Create official or custom task-0 environments without stock task-id spoofing."""

    def __init__(self, *, episode_length: int = 100):
        if episode_length != 100:
            raise LiberoContractError("batch24 protocol requires an explicit 100-step horizon")
        self.episode_length = episode_length

    @staticmethod
    def suite_and_task() -> tuple[Any, Any]:
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()["libero_object"]()
        return suite, suite.get_task(0)

    @classmethod
    def official_paths(cls) -> tuple[Path, Path]:
        from libero.libero import get_libero_path

        suite, task = cls.suite_and_task()
        bddl = (
            Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        init_state = (
            Path(get_libero_path("init_states"))
            / task.problem_folder
            / Path(task.init_states_file).name
        )
        return bddl.resolve(), init_state.resolve()

    def make_official_env(self):
        from lerobot.envs.libero import LiberoEnv

        suite, _task = self.suite_and_task()
        return LiberoEnv(
            task_suite=suite,
            task_id=0,
            task_suite_name="libero_object",
            episode_length=self.episode_length,
            control_mode="relative",
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
        )

    def make_custom_env(self, contract: TaskContract):
        from lerobot.envs.libero import LiberoEnv

        suite, _task = self.suite_and_task()
        env = LiberoEnv(
            task_suite=suite,
            task_id=0,
            task_suite_name="libero_object",
            episode_length=self.episode_length,
            control_mode="relative",
            obs_type="pixels_agent_pos",
            observation_height=256,
            observation_width=256,
        )
        if env._env is not None:
            raise LiberoContractError("custom BDDL must be bound before first reset")
        env._task_bddl_file = contract.bddl_path
        env.task = f"mea_custom_{Path(contract.bddl_path).stem}"
        env.task_description = contract.language
        return env

    @staticmethod
    def render_and_init_probe(env: Any, *, seed: int, output_png: str | Path) -> dict[str, Any]:
        """Verify registered-class construction, official init compatibility and render."""

        from PIL import Image

        output = Path(output_png).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        observation, _info = env.reset(seed=seed)
        image = env.render()
        Image.fromarray(image).save(output)
        result = {
            "reset": isinstance(observation, dict),
            "robot_state_present": bool(
                isinstance(observation, dict) and "robot_state" in observation
            ),
            "official_init_state_applied": bool(env.init_states and env._init_states is not None),
            "render_nonempty": bool(getattr(image, "size", 0)),
            "render_shape": list(image.shape),
            "render_path": str(output),
            "underlying_problem": env._env.problem_name,
        }
        env.close()
        if not all(
            result[key]
            for key in (
                "reset",
                "robot_state_present",
                "official_init_state_applied",
                "render_nonempty",
            )
        ):
            raise LiberoContractError("custom environment compatibility probe failed")
        return result
