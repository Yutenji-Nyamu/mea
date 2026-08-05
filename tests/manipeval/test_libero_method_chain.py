from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest

from mea.libero.benchmark import (
    BATCH23_PARITY_ACTION_STEPS,
    BATCH23_PARITY_HORIZON_STEPS,
    BATCH23_PARITY_OBSERVATION_SIZE,
    EpisodeRecord,
    LiberoBenchmarkAdapter,
    LiberoContractError,
    TaskContract,
    build_official_task_contract,
)
from mea.libero.chain import (
    _capabilities,
    _method_chain_is_valid,
    _planner_taskgen_misaligned_result,
    parse_bound_libero_task,
)
from mea.libero.policy import LeRobotPolicyAdapter
from mea.libero.retrieval import (
    BDDLRetrieval,
    BDDLTaskRecord,
    BDDLTaskIndex,
    ControlledChangeContract,
    PolicyTaskCompatibility,
    authorize_controlled_change,
    smolvla_policy_compatibility,
)
from mea.libero.taskgen import LiberoTaskGenBackend
from mea.libero.tool import LiberoPredicateToolBackend
from mea.method_runtime import BackendTaskBinding
from mea.planner.claim_first import validate_open_query_capabilities
from mea.planner.plan_agent_session import PlanAgentSession
from mea.planner.query_contract import build_query_sufficiency_contract
from mea.toolkit.aggregate import aggregate_tool_executions


class _FakeProvider:
    last_metadata = {"model": "fixture"}

    def text(self, prompt: str, **_kwargs) -> str:
        marker = "OFFICIAL BDDL:\n"
        base = prompt.split(marker, 1)[1].split("\n\nReturn strict JSON", 1)[0]
        candidate = base.replace(
            "Pick the alphabet soup and place it in the basket",
            "Pick the milk and place it in the basket",
        )
        candidate = candidate.replace(
            "(:obj_of_interest\n    alphabet_soup_1\n    basket_1",
            "(:obj_of_interest\n    milk_1\n    basket_1",
        )
        candidate = candidate.replace(
            "(In alphabet_soup_1 basket_1_contain_region)",
            "(In milk_1 basket_1_contain_region)",
        )
        return json.dumps(
            {
                "bddl_text": candidate,
                "selected_object": "milk_1",
                "rationale": "fixture",
            }
        )


def test_libero_capabilities_match_public_plan_agent_schema() -> None:
    official = TaskContract(
        schema_version=1,
        benchmark="libero",
        suite="libero_object",
        official_task_id=0,
        bddl_path="/tmp/official.bddl",
        bddl_sha256="fixture",
        problem_name="fixture",
        domain="libero",
        language="fixture",
        objects={"fixture": ["fixture_1"]},
        regions=["fixture_region"],
        initial_state_sha256="fixture",
        goal_predicates=[["In", "fixture_1", "fixture_region"]],
        python_problem_impl="fixture",
        initial_state_source="/tmp/fixture.pruned_init",
    )

    validated = validate_open_query_capabilities(
        _capabilities(Path("/checkpoint"), official)
    )

    operation = validated["generation_card"]["taskgen_operations"][0]
    assert set(operation) == {
        "operation",
        "controlled_axis",
        "generation_mode",
        "allowed_change_roots",
    }
    assert validated["generation_card"]["toolgen"] == {
        "retrieve_first": True,
        "can_generate_rule_metric": True,
        "can_generate_vqa_question": False,
    }


def test_taskgen_state_compatible_and_tool_exact_reuse(tmp_path: Path) -> None:
    proposal = {
        "schema_version": 1,
        "proposal": {
            "sub_aspect": "existing_object_identity",
            "requested_perturbation": {
                "description": "change only the goal object",
                "controlled_changes": ["goal_object_identity"],
                "preserve": ["initial state"],
            },
            "tool_need": {"description": "goal predicate"},
        },
    }
    official = build_official_task_contract()
    index = BDDLTaskIndex.from_contracts([official])
    compatibility = smolvla_policy_compatibility(
        checkpoint="/checkpoint",
        explicit_task_binding=index.tasks[0],
    )
    retrieval = index.retrieve_nearest(
        "test object identity",
        compatibility=compatibility,
    )
    change_contract = authorize_controlled_change(retrieval, proposal)
    contract, result = LiberoTaskGenBackend(
        _FakeProvider(), model="fixture"
    ).generate(
        user_query="test object identity",
        proposal_bundle=proposal,
        output_dir=tmp_path / "taskgen",
        seed=100800,
        retrieval=retrieval,
        change_contract=change_contract,
    )
    assert result["status"] == "passed"
    assert result["experiment_seed"] == 100800
    assert result["selected_object"] == "milk_1"
    assert contract.goal_predicates == [["in", "milk_1", "basket_1_contain_region"]]
    assert contract.validation["same_initial_state"] is True

    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    actions = episode_dir / "actions.npy"
    actions.write_bytes(b"fixture")
    record = EpisodeRecord(
        schema_version=1,
        benchmark="libero",
        policy_name="smolvla",
        checkpoint="/checkpoint",
        suite="libero_object",
        task_id="libero_object/task0/mea_custom",
        seed=100800,
        horizon_steps=BATCH23_PARITY_HORIZON_STEPS,
        executed_steps=BATCH23_PARITY_HORIZON_STEPS,
        success=False,
        reward_sum=0.0,
        goal_predicate_satisfied=False,
        elapsed_seconds=1.0,
        bddl_path=contract.bddl_path,
        video_path=None,
        task_contract_path=result["artifacts"]["task_contract"],
        actions_path=str(actions),
    )
    backend = LiberoPredicateToolBackend(registry_dir=tmp_path / "registry")
    generated, execution = backend.compile_validate_register(
        output_dir=tmp_path / "tool",
        episode_record=record,
        goal_predicates=contract.goal_predicates,
        source_query="did the goal predicate hold",
    )
    assert generated["live_value_non_null"] is True
    assert generated["artifact_kind"] == "deterministic_predicate_metric_adapter"
    assert generated["model_generated"] is False
    assert generated["route"] == "predicate_metric_compile_validate_register"
    assert generated["tool_execution"]["episodes"][0]["result"]["value"] is False
    aggregate = aggregate_tool_executions([execution])
    assert aggregate["status"] == "passed"
    assert aggregate["metrics"][0]["cohorts"][0]["role"] == "policy_under_evaluation"

    reused = backend.exact_reuse(
        output_dir=tmp_path / "reuse",
        episode_record=record,
        goal_predicates=contract.goal_predicates,
        source_query="reuse the same observable",
    )
    assert reused["route"] == "exact_registry_reuse"
    assert reused["additional_rollouts"] == 0


def test_custom_factory_is_not_stock_task_id() -> None:
    adapter = LiberoBenchmarkAdapter(
        episode_length=BATCH23_PARITY_HORIZON_STEPS,
        observation_size=BATCH23_PARITY_OBSERVATION_SIZE,
    )
    official = adapter.make_official_env()
    official_path = official._task_bddl_file
    official.close()
    assert "libero_object" in official_path


def test_batch23_parity_protocol_defaults_are_shared() -> None:
    adapter = LiberoBenchmarkAdapter()
    policy = LeRobotPolicyAdapter(checkpoint="/checkpoint")
    assert adapter.episode_length == BATCH23_PARITY_HORIZON_STEPS == 280
    assert adapter.observation_size == BATCH23_PARITY_OBSERVATION_SIZE == 360
    assert policy.horizon_steps == BATCH23_PARITY_HORIZON_STEPS
    assert policy.observation_size == BATCH23_PARITY_OBSERVATION_SIZE
    assert policy.n_action_steps == BATCH23_PARITY_ACTION_STEPS == 10
    assert policy.suite_name == "libero_object"
    assert policy.task_id == 0


def test_policy_load_matches_stock_seed_order_and_supports_bound_task(
    monkeypatch,
) -> None:
    events: list[tuple] = []
    captured_rng_state = {"python": "fixture", "torch": "fixture"}

    class FakeVectorEnv:
        def close(self) -> None:
            events.append(("env_close",))

    class FakeEnvConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakePolicyConfig:
        @classmethod
        def from_pretrained(cls, checkpoint):
            events.append(("policy_config", str(checkpoint)))
            return SimpleNamespace()

    class FakeParameter:
        def numel(self) -> int:
            return 7

    class FakePolicy:
        def eval(self) -> None:
            events.append(("policy_eval",))

        def parameters(self):
            return [FakeParameter()]

    vector_env = FakeVectorEnv()

    def fake_set_seed(seed: int) -> None:
        events.append(("set_seed", seed))

    def fake_make_env(env_config, *, n_envs: int, use_async_envs: bool):
        events.append(
            (
                "make_env",
                env_config.kwargs["task"],
                tuple(env_config.kwargs["task_ids"]),
                n_envs,
                use_async_envs,
            )
        )
        return {"libero_goal": {3: vector_env}}

    def fake_make_policy(*, cfg, env_cfg, rename_map):
        events.append(
            (
                "make_policy",
                cfg.n_action_steps,
                env_cfg.kwargs["task"],
                tuple(env_cfg.kwargs["task_ids"]),
                rename_map,
            )
        )
        return FakePolicy()

    def fake_get_rng_state():
        events.append(("capture_rng",))
        return captured_rng_state

    def fake_set_rng_state(state):
        events.append(("restore_rng", state))

    modules = {
        "lerobot": types.ModuleType("lerobot"),
        "lerobot.configs": types.ModuleType("lerobot.configs"),
        "lerobot.envs": types.ModuleType("lerobot.envs"),
        "lerobot.envs.configs": types.ModuleType("lerobot.envs.configs"),
        "lerobot.envs.factory": types.ModuleType("lerobot.envs.factory"),
        "lerobot.policies": types.ModuleType("lerobot.policies"),
        "lerobot.utils": types.ModuleType("lerobot.utils"),
        "lerobot.utils.random_utils": types.ModuleType(
            "lerobot.utils.random_utils"
        ),
    }
    modules["lerobot.configs"].PreTrainedConfig = FakePolicyConfig
    modules["lerobot.envs.configs"].LiberoEnv = FakeEnvConfig
    modules["lerobot.envs.factory"].make_env = fake_make_env
    modules["lerobot.envs.factory"].make_env_pre_post_processors = (
        lambda **_kwargs: ("env_pre", "env_post")
    )
    modules["lerobot.policies"].make_policy = fake_make_policy
    modules["lerobot.policies"].make_pre_post_processors = (
        lambda **_kwargs: ("policy_pre", "policy_post")
    )
    modules["lerobot.utils.random_utils"].get_rng_state = fake_get_rng_state
    modules["lerobot.utils.random_utils"].set_seed = fake_set_seed
    modules["lerobot.utils.random_utils"].set_rng_state = fake_set_rng_state
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    policy = LeRobotPolicyAdapter(
        checkpoint="/checkpoint",
        suite_name="libero_goal",
        task_id=3,
    )
    loaded = policy.load(seed=100800)

    assert [event[0] for event in events[:4]] == [
        "set_seed",
        "make_env",
        "policy_config",
        "make_policy",
    ]
    assert events[1] == ("make_env", "libero_goal", (3,), 1, True)
    assert events[3][-1] == {}
    assert loaded["suite"] == "libero_goal"
    assert loaded["task_id"] == 3
    assert loaded["seed_contract"]["seed_scope"] == (
        "once_before_env_and_policy_construction"
    )
    assert loaded["seed_contract"]["first_rollout_integer_reseed"] is False
    assert loaded["seed_contract"]["paired_round_rng_restore"] is True
    assert policy.make_stock_official_vector_env() is vector_env

    first_index, first_mode = policy.prepare_policy_rng_for_rollout()
    second_index, second_mode = policy.prepare_policy_rng_for_rollout()
    assert (first_index, first_mode) == (1, "stock_first_rollout_continuation")
    assert (second_index, second_mode) == (
        2,
        "restored_pre_first_rollout_state",
    )
    assert [event for event in events if event[0] == "set_seed"] == [
        ("set_seed", 100800)
    ]
    assert [event for event in events if event[0] == "restore_rng"] == [
        ("restore_rng", captured_rng_state)
    ]


def test_method_chain_valid_is_mechanism_not_custom_outcome() -> None:
    complete = dict(
        official_success=True,
        rollouts_executed=2,
        planner_taskgen_alignment=True,
        compatibility_probe_passed=True,
        aggregate_status="passed",
        exact_reuse=True,
        validated_final_decision_present=True,
        episode_protocol_matches=True,
    )
    assert _method_chain_is_valid(**complete) is True
    for key in complete:
        broken = dict(complete)
        broken[key] = (
            1
            if key == "rollouts_executed"
            else "failed"
            if key == "aggregate_status"
            else False
        )
        assert _method_chain_is_valid(**broken) is False


def test_libero_method_evidence_uses_shared_query_contract_and_answer() -> None:
    binding = BackendTaskBinding(
        benchmark="libero",
        binding_id="libero_object/task0",
        task_contract={"suite": "libero_object", "task_id": 0},
        native_task=object(),
        metadata={"task_name": "libero_object_task0"},
    )
    contract = build_query_sufficiency_contract(
        "Does any generated object variation fail?",
        candidate_universe=["variant_milk"],
        round_budget=1,
        claim_type="existential",
        candidate_universe_closed=False,
        existential_witness_outcome="fail",
        control_requirement="required",
    )
    session = PlanAgentSession(
        "Does any generated object variation fail?",
        method_binding=binding,
        method_max_rounds=2,
        query_contract=contract,
        require_control_anchor=True,
    )
    evidence = [
        {
            "schema_version": 1,
            "round_id": "round_01_official_control",
            "tested_sub_aspect": "official_control",
            "tested_hypothesis": "The unchanged task succeeds.",
            "tested_perturbation": "none",
            "outcome": "success",
            "evidence_summary": "The official control succeeded.",
            "limitations": ["N=1"],
        },
        {
            "schema_version": 1,
            "round_id": "round_02_custom_bddl",
            "tested_sub_aspect": "object_identity",
            "tested_hypothesis": "The generated variation succeeds.",
            "tested_perturbation": "change the goal object",
            "outcome": "failure",
            "evidence_summary": "The generated predicate failed.",
            "limitations": ["N=1"],
        },
    ]
    state = session.observe_method_evidence(
        evidence,
        candidate_evidence=[
            {
                "candidate_id": "variant_milk",
                "outcome": "fail",
                "score": 0.0,
                "diagnosis": "generated predicate not satisfied",
            }
        ],
        baseline_valid=True,
    )
    assert state["assessment"]["evidence_sufficient"] is True
    assert state["assessment"]["stop_reason"] == "evidence_sufficient"
    assert state["query_answer"]["answered"] is True
    assert state["query_answer"]["claim_verdict"] == "counterexample_found"


def test_planner_taskgen_misalignment_is_structured_and_rollout_bounded(
    tmp_path: Path,
) -> None:
    task = BDDLTaskRecord(
        suite="libero_object",
        task_id=0,
        problem_name="fixture_problem",
        language="pick object",
        bddl_path="/fixture/task0.bddl",
        init_state_path="/fixture/task0.pruned_init",
        objects=("object_1", "basket_1"),
        goal_predicates=(("in", "object_1", "basket_1_contain_region"),),
    )
    retrieval = BDDLRetrieval(
        query_concern="language robustness",
        selected=task,
        score=1.0,
        authorized_candidate_count=1,
        not_authorized_candidate_count=0,
        selection_authorization="explicit_run_binding_only",
    )
    compatibility = PolicyTaskCompatibility(
        policy_name="SmolVLA",
        checkpoint="/checkpoint",
        declared_scope="unknown",
        authorized_task_ids={"libero_object": (0,)},
        authorization_source="explicit_run_binding",
        artifact_evidence={"fixture": True},
    )
    contract = ControlledChangeContract(
        source_suite="libero_object",
        source_task_id=0,
        source_problem_name="fixture_problem",
        query_concern="language robustness",
        requested_change_roots=("language",),
        allowed_change_roots=("language", "obj_of_interest", "goal"),
        preserved_roots=("initial_state",),
        status="unsupported",
        reason="language-only is outside the Phase-1 TaskGen axis",
    )
    result = _planner_taskgen_misaligned_result(
        request="test language only",
        root=tmp_path,
        official_success=True,
        retrieval=retrieval,
        compatibility=compatibility,
        change_contract=contract,
    )
    assert result["status"] == "planner_taskgen_misaligned"
    assert result["rollouts_executed"] == 1
    assert result["custom_rollout_authorized"] is False
    assert result["method_chain_valid"] is False
    assert result["paper_performance_evidence"] is False
    assert result["scientific_evidence_eligible"] is False


def test_official_control_failure_short_circuits_custom_rollout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import mea.libero.chain as chain_module

    task = BDDLTaskRecord(
        suite="libero_object",
        task_id=0,
        problem_name="fixture_problem",
        language="pick object",
        bddl_path="/fixture/task0.bddl",
        init_state_path="/fixture/task0.pruned_init",
    )
    retrieval = BDDLRetrieval(
        query_concern="object identity",
        selected=task,
        score=1.0,
        authorized_candidate_count=1,
        not_authorized_candidate_count=0,
        selection_authorization="explicit_run_binding_only",
    )
    compatibility = PolicyTaskCompatibility(
        policy_name="SmolVLA",
        checkpoint="/checkpoint",
        declared_scope="unknown",
        authorized_task_ids={"libero_object": (0,)},
        authorization_source="explicit_run_binding",
        artifact_evidence={"fixture": True},
    )
    pending = ControlledChangeContract(
        source_suite="libero_object",
        source_task_id=0,
        source_problem_name="fixture_problem",
        query_concern="object identity",
        requested_change_roots=(),
        allowed_change_roots=("language", "obj_of_interest", "goal"),
        preserved_roots=("initial_state",),
        status="pending",
        reason="awaiting Planner",
    )
    official = TaskContract(
        schema_version=1,
        benchmark="libero",
        suite="libero_object",
        official_task_id=0,
        bddl_path="/fixture/task0.bddl",
        bddl_sha256="fixture",
        problem_name="fixture_problem",
        domain="robosuite",
        language="pick object",
        objects={"object": ["object_1"]},
        regions=["basket_1_contain_region"],
        initial_state_sha256="fixture",
        goal_predicates=[["in", "object_1", "basket_1_contain_region"]],
        python_problem_impl="fixture.Problem",
        initial_state_source="/fixture/task0.pruned_init",
    )

    class FakeBenchmark:
        def __init__(self, **_kwargs):
            pass

        def make_official_env(self):
            raise AssertionError("fake policy must not construct the simulator")

    class FakePolicy:
        run_count = 0

        def __init__(self, **_kwargs):
            pass

        def load(self, *, seed):
            return {"status": "passed", "seed": seed}

        def run(self, **kwargs):
            type(self).run_count += 1
            return EpisodeRecord(
                schema_version=1,
                benchmark="libero",
                policy_name="smolvla",
                checkpoint="/checkpoint",
                suite="libero_object",
                task_id=kwargs["task_id"],
                seed=kwargs["seed"],
                horizon_steps=BATCH23_PARITY_HORIZON_STEPS,
                executed_steps=BATCH23_PARITY_HORIZON_STEPS,
                success=False,
                reward_sum=0.0,
                goal_predicate_satisfied=False,
                elapsed_seconds=1.0,
                bddl_path=kwargs["bddl_path"],
                video_path=None,
                task_contract_path=str(kwargs["task_contract_path"]),
                actions_path="/fixture/actions.npy",
            )

        def unload(self):
            pass

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Planner/TaskGen must not run after control failure")

    monkeypatch.setenv("UIUI_API_KEY", "fixture")
    monkeypatch.setattr(
        chain_module,
        "build_official_task_contract",
        lambda **_kwargs: official,
    )
    monkeypatch.setattr(
        chain_module,
        "_open_query_retrieval",
        lambda **_kwargs: (compatibility, retrieval, pending),
    )
    monkeypatch.setattr(
        chain_module,
        "_gate0",
        lambda **_kwargs: {"schema_version": 1, "status": "passed"},
    )
    monkeypatch.setattr(chain_module, "OpenAICompatibleProvider", lambda **_kwargs: object())
    monkeypatch.setattr(chain_module, "LiberoBenchmarkAdapter", FakeBenchmark)
    monkeypatch.setattr(chain_module, "LeRobotPolicyAdapter", FakePolicy)
    monkeypatch.setattr(chain_module, "PlanAgent", forbidden)
    monkeypatch.setattr(chain_module, "LiberoTaskGenBackend", forbidden)

    result = chain_module.run_libero_method_chain(
        repo_root=tmp_path,
        request="object identity robustness",
        evaluation_id="control_fail",
        bound_suite="libero_object",
        bound_task_id=0,
    )
    assert FakePolicy.run_count == 1
    assert result["status"] == "control_failed"
    assert result["custom_rollout_authorized"] is False
    assert result["method_chain_valid"] is False
    assert result["paper_performance_evidence"] is False


def test_unknown_checkpoint_scope_requires_explicit_suite_task_binding() -> None:
    assert parse_bound_libero_task("libero_object/task0") == ("libero_object", 0)
    assert parse_bound_libero_task("libero_goal/task4") == ("libero_goal", 4)
    with unittest.TestCase().assertRaisesRegex(ValueError, "scope is unknown"):
        parse_bound_libero_task(None)


def test_preserve_only_goal_text_is_not_a_change_authorization() -> None:
    official = build_official_task_contract()
    index = BDDLTaskIndex.from_contracts([official])
    compatibility = smolvla_policy_compatibility(
        checkpoint="/checkpoint",
        explicit_task_binding=index.tasks[0],
    )
    retrieval = index.retrieve_nearest(
        "object identity robustness",
        compatibility=compatibility,
    )
    contract = authorize_controlled_change(
        retrieval,
        {
            "proposal": {
                "requested_perturbation": {
                    "controlled_changes": ["preserve goal object identity"]
                }
            }
        },
    )
    assert contract.status == "unsupported"


def test_taskgen_rejects_planner_language_only_request(tmp_path: Path) -> None:
    official = build_official_task_contract()
    index = BDDLTaskIndex.from_contracts([official])
    compatibility = smolvla_policy_compatibility(
        checkpoint="/checkpoint",
        explicit_task_binding=index.tasks[0],
    )
    retrieval = index.retrieve_nearest(
        "object identity robustness",
        compatibility=compatibility,
    )
    proposal = {
        "proposal": {
            "requested_perturbation": {
                "description": "paraphrase only",
                "controlled_changes": ["language"],
                "preserve": ["goal semantics"],
            }
        }
    }
    change_contract = authorize_controlled_change(retrieval, proposal)
    with unittest.TestCase().assertRaisesRegex(
        LiberoContractError,
        "controlled-change contract is not authorized",
    ):
        LiberoTaskGenBackend(_FakeProvider(), model="fixture").generate(
            user_query="object identity robustness",
            proposal_bundle=proposal,
            output_dir=tmp_path / "taskgen",
            seed=100800,
            retrieval=retrieval,
            change_contract=change_contract,
        )


@unittest.skipUnless(
    importlib.util.find_spec("libero") is not None
    and importlib.util.find_spec("lerobot") is not None,
    "requires the dedicated LIBERO/LeRobot environment",
)
class LiberoMethodChainTests(unittest.TestCase):
    def test_dedicated_environment_executes_libero_method_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_path = Path(directory)
            test_taskgen_state_compatible_and_tool_exact_reuse(temp_path)
            test_custom_factory_is_not_stock_task_id()
            test_unknown_checkpoint_scope_requires_explicit_suite_task_binding()
            test_taskgen_rejects_planner_language_only_request(temp_path)
