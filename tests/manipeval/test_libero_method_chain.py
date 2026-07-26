from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

from mea.libero.benchmark import LiberoContractError
from mea.libero.benchmark import EpisodeRecord, LiberoBenchmarkAdapter
from mea.libero.taskgen import LiberoTaskGenBackend
from mea.libero.tool import LiberoPredicateToolBackend
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


def test_taskgen_state_compatible_and_tool_exact_reuse(tmp_path: Path) -> None:
    proposal = {
        "schema_version": 1,
        "proposal": {
            "sub_aspect": "existing_object_identity",
            "requested_perturbation": {
                "description": "change only the goal object",
                "controlled_changes": ["goal object"],
                "preserve": ["initial state"],
            },
            "tool_need": {"description": "goal predicate"},
        },
    }
    contract, result = LiberoTaskGenBackend(
        _FakeProvider(), model="fixture"
    ).generate(
        user_query="test object identity",
        proposal_bundle=proposal,
        output_dir=tmp_path / "taskgen",
        seed=100800,
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
        horizon_steps=100,
        executed_steps=100,
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
    adapter = LiberoBenchmarkAdapter(episode_length=100)
    official = adapter.make_official_env()
    official_path = official._task_bddl_file
    official.close()
    assert "libero_object" in official_path


def test_taskgen_rejects_planner_language_only_request(tmp_path: Path) -> None:
    with unittest.TestCase().assertRaisesRegex(
        LiberoContractError,
        match="Planner requested a different controlled change",
    ):
        LiberoTaskGenBackend(_FakeProvider(), model="fixture").generate(
            user_query="object identity robustness",
            proposal_bundle={
                "proposal": {
                    "requested_perturbation": {
                        "description": "paraphrase only",
                        "controlled_changes": ["language"],
                        "preserve": ["goal semantics"],
                    }
                }
            },
            output_dir=tmp_path / "taskgen",
            seed=100800,
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
            test_taskgen_rejects_planner_language_only_request(temp_path)
