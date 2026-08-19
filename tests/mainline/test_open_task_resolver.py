import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "mea"
    / "planner"
    / "open_task_resolver.py"
)
SPEC = importlib.util.spec_from_file_location("open_task_resolver", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


def concern(task_intent: str) -> dict:
    return {
        "schema_version": 1,
        "source_query": "Which object property first exposes a weakness?",
        "sub_aspect": "novel.reflective_surface_confusion",
        "hypothesis": "A reflective target causes a pre-contact miss.",
        "task_intent": task_intent,
        "requested_variation": "change only target reflectivity",
        "preserved_conditions": [],
        "measurement_need": "target contact and pre-contact trajectory",
    }


def single_task_policy(task_name: str = "beat_block_hammer") -> dict:
    return {
        "policy_name": "ACT",
        "checkpoint_id": f"act-{task_name}/demo_clean-50",
        "single_task_checkpoint": True,
        "language_conditioned": False,
        "task_name": task_name,
        "checkpoint_ready": True,
    }


def multi_task_policy(*, supports_unseen: bool = False) -> dict:
    return {
        "policy_name": "LanguagePolicy",
        "checkpoint_id": "language-policy-v1",
        "single_task_checkpoint": False,
        "language_conditioned": True,
        "training_tasks": ["beat_block_hammer", "open_laptop"],
        "supports_unseen_tasks": supports_unseen,
        "checkpoint_ready": True,
    }


def inventory() -> list[dict]:
    return [
        {
            "schema_version": 1,
            "task_name": "beat_block_hammer",
            "description": "grab the hammer and beat the target block",
            "execution_status": "capability_registered",
            "capability_aspects": ["object.appearance"],
        },
        {
            "schema_version": 1,
            "task_name": "open_laptop",
            "description": "lift the laptop lid until the laptop is open",
            "execution_status": "official_base_only",
            "capability_aspects": [],
        },
        {
            "schema_version": 1,
            "task_name": "place_object_basket",
            "description": "pick up the target object and put it in a basket",
            "execution_status": "official_base_only",
            "capability_aspects": [],
        },
    ]


def click_near_tie_inventory() -> list[dict]:
    return [
        {
            "schema_version": 1,
            "task_name": "scan_object",
            "description": (
                "Use one arm to pick the scanner and use the other arm to "
                "pick the object, and use the scanner to scan the object"
            ),
            "execution_status": "official_base_only",
            "capability_aspects": [],
        },
        {
            "schema_version": 1,
            "task_name": "click_alarmclock",
            "description": (
                "click the alarm clock's center of the top side button "
                "on the table"
            ),
            "execution_status": "official_base_only",
            "capability_aspects": [],
        },
        {
            "schema_version": 1,
            "task_name": "click_bell",
            "description": "click the bell's top center on the table",
            "execution_status": "capability_registered",
            "capability_aspects": ["object_instance"],
        },
    ]


class QueryInterpretationTests(unittest.TestCase):
    class Provider:
        last_metadata = {"provider": "fixture"}

        def __init__(self, responses):
            self.responses = list(responses)
            self.prompts = []

        def text(self, prompt, **_kwargs):
            self.prompts.append(prompt)
            return self.responses.pop(0)

    def test_prompt_has_no_inventory_or_catalog_parameter(self):
        prompt = resolver.build_free_concern_prompt(
            concern("grab hammer and hit block")["source_query"],
            single_task_policy(),
        )
        compact_prompt = " ".join(prompt.split())
        self.assertIn("not available until a later retrieval stage", prompt)
        self.assertIn("invariant base action", prompt)
        self.assertIn("Put distractors", prompt)
        self.assertIn("jointly realizable", prompt)
        self.assertIn("RGB is only", prompt)
        self.assertIn("Preservation is an authority claim", compact_prompt)
        self.assertIn(
            "Task identity and policy checkpoint are already frozen",
            compact_prompt,
        )
        self.assertIn("default preservation list is empty", compact_prompt)
        self.assertIn(
            "observable in policy or simulator metadata is a measurement",
            compact_prompt,
        )
        self.assertIn(
            "actor identity, physics timestep, or object-to-target binding",
            compact_prompt,
        )
        self.assertIn("terminal-state distance threshold", compact_prompt)
        self.assertIn("terminal value of that same", compact_prompt)
        self.assertIn("trajectory peak or maximum", compact_prompt)
        self.assertIn(
            "must not use its scale to relax, replace",
            compact_prompt,
        )
        self.assertIn("Do not invent an absolute perturbation magnitude", prompt)
        self.assertIn("never add reuse_first to either", prompt)
        self.assertNotIn("open_laptop", prompt)
        self.assertNotIn("object.appearance", prompt)

    def test_query_identity_is_preserved(self):
        value = concern("grab hammer and hit block")
        self.assertEqual(
            resolver.validate_free_concern(
                value, expected_query=value["source_query"]
            ),
            value,
        )
        with self.assertRaisesRegex(
            resolver.OpenTaskResolutionError, "differs from the original Query"
        ):
            resolver.validate_free_concern(value, expected_query="another query")

    def test_agent_retries_without_ever_exposing_inventory(self):
        value = concern("grab a hammer and hit the target block")
        provider = self.Provider(["{}", json.dumps(value)])
        result = resolver.PlanAgentQueryInterpreter(
            provider, model="fixture"
        ).propose(
            value["source_query"], policy_card=single_task_policy()
        )
        self.assertEqual(
            result["source"],
            "provider_plan_agent_query_interpretation",
        )
        self.assertEqual(result["concern"], value)
        self.assertEqual(result["provider"]["attempt_count"], 2)
        self.assertEqual(len(provider.prompts), 2)
        self.assertTrue(all("open_laptop" not in prompt for prompt in provider.prompts))

    def test_agent_preserves_provider_authored_independent_experiment_needs(self):
        value = concern("locate the bell and click its top center")
        typed = {
            **value,
            "scene_need": {
                "required": True,
                "description": "Reduce only the bell visual size to 50%.",
            },
            "checker_need": {
                "required": False,
                "description": None,
            },
            "rule_tool_need": {
                "required": True,
                "description": "Measure success and click-position error.",
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
        }
        provider = self.Provider([json.dumps(typed)])

        result = resolver.PlanAgentQueryInterpreter(
            provider,
            model="fixture",
        ).propose(value["source_query"], policy_card=single_task_policy())

        self.assertEqual(result["concern"], value)
        self.assertEqual(
            result["experiment_needs"]["scene_need"],
            typed["scene_need"],
        )
        self.assertIsNone(
            result["experiment_needs"]["checker_need"]["description"]
        )
        self.assertIn("A Tool-only Query must not invent", provider.prompts[0])

    def test_official_only_query_uses_provider_typed_official_rule_reuse(self):
        value = {
            **concern("click the bell's top center"),
            "source_query": "Can it complete only the official task?",
            "requested_variation": "reuse the unchanged official scene",
            "measurement_need": "use official check_success()",
        }
        typed_needs = {
            field: {
                "required": field == "rule_tool_need",
                "description": (
                    "Reuse the official check_success() boolean result."
                    if field == "rule_tool_need"
                    else None
                ),
                **(
                    {"reuse_first": True}
                    if field in {"rule_tool_need", "vqa_tool_need"}
                    else {}
                ),
            }
            for field in resolver._EXPERIMENT_NEED_FIELDS
        }
        provider = self.Provider([json.dumps({**value, **typed_needs})])

        result = resolver.PlanAgentQueryInterpreter(
            provider,
            model="fixture",
        ).propose(value["source_query"], policy_card=single_task_policy())

        needs = result["experiment_needs"]
        self.assertFalse(needs["scene_need"]["required"])
        self.assertFalse(needs["checker_need"]["required"])
        self.assertTrue(needs["rule_tool_need"]["required"])
        self.assertIn(
            "official check_success()",
            needs["rule_tool_need"]["description"],
        )
        self.assertEqual(
            result["concern"]["measurement_need"],
            value["measurement_need"],
        )
        self.assertFalse(needs["vqa_tool_need"]["required"])
        self.assertIn("official-task-only Query", provider.prompts[0])
        self.assertEqual(result["provider"]["attempt_count"], 1)

    def test_non_official_query_cannot_drop_all_evidence_needs(self):
        value = concern("click the bell's top center")
        empty_needs = {
            field: {
                "required": False,
                "description": None,
                **(
                    {"reuse_first": True}
                    if field in {"rule_tool_need", "vqa_tool_need"}
                    else {}
                ),
            }
            for field in resolver._EXPERIMENT_NEED_FIELDS
        }
        provider = self.Provider([json.dumps({**value, **empty_needs})])

        with self.assertRaisesRegex(
            resolver.OpenTaskResolutionError,
            "at least one explicit evidence need",
        ):
            resolver.PlanAgentQueryInterpreter(
                provider,
                model="fixture",
                max_attempts=1,
            ).propose(value["source_query"], policy_card=single_task_policy())

    def test_provider_typed_checker_choice_is_not_overridden_by_query_regex(self):
        query = (
            "是否存在一个成功样本，使 ACT 只抬起目标滚筒且不抬起"
            "非目标滚筒？"
        )
        value = {
            **concern("lift the target roller"),
            "source_query": query,
        }
        false_needs = {
            **value,
            "scene_need": {
                "required": True,
                "description": "Add one physical distractor roller.",
            },
            "checker_need": {"required": False, "description": None},
            "rule_tool_need": {
                "required": True,
                "description": "Measure target and distractor lift.",
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
        }
        provider = self.Provider([json.dumps(false_needs)])

        result = resolver.PlanAgentQueryInterpreter(
            provider,
            model="fixture",
        ).propose(query, policy_card=single_task_policy("grab_roller"))

        self.assertFalse(
            result["experiment_needs"]["checker_need"]["required"]
        )
        self.assertEqual(result["provider"]["attempt_count"], 1)
        self.assertIn(
            '"checker_need": {\n    "required": false',
            provider.prompts[0],
        )
        self.assertIn(
            "keep their roles distinct",
            provider.prompts[0],
        )

    def _cold_agent_can_be_frozen_to_one_attempt(self):
        provider = self.Provider(["{}"])
        with self.assertRaisesRegex(
            resolver.OpenTaskResolutionError, "1 FreeConcern attempt"
        ):
            resolver.PlanAgentQueryInterpreter(
                provider, model="fixture", max_attempts=1
            ).propose(
                "Which concern matters?", policy_card=single_task_policy()
            )
        self.assertEqual(len(provider.prompts), 1)

    def test_query_prose_does_not_reclassify_provider_typed_needs(self):
        query = (
            "Define success as the official goal and one additional "
            "experimental condition."
        )
        value = {
            **concern("test one additional success condition"),
            "source_query": query,
        }
        invalid = {
            **value,
            "scene_need": {"required": False, "description": None},
            "checker_need": {"required": False, "description": None},
            "rule_tool_need": {
                "required": True,
                "description": "Measure the official success boolean.",
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
        }
        provider = self.Provider([json.dumps(invalid), json.dumps(invalid)])

        result = resolver.PlanAgentQueryInterpreter(
            provider,
            model="fixture",
        ).propose(query, policy_card=single_task_policy())

        self.assertEqual(result["experiment_needs"], {
            field: invalid[field] for field in resolver._EXPERIMENT_NEED_FIELDS
        })
        self.assertEqual(result["provider"]["attempt_count"], 1)

    def test_typed_checker_is_not_rewritten_by_cross_field_word_matching(self):
        query = (
            "Define success as the official goal plus one new condition and "
            "report one scalar trajectory observation."
        )
        value = {
            **concern("test clearance during the trajectory"),
            "source_query": query,
        }
        scalar_description = "Measure trajectory deviation as one scalar."
        copied = {
            **value,
            "scene_need": {
                "required": True,
                "description": "Add one inert distractor.",
            },
            "checker_need": {
                "required": True,
                "description": (
                    "Pass only if the official goal succeeds and trajectory "
                    "smoothness remains within an acceptable threshold."
                ),
            },
            "rule_tool_need": {
                "required": True,
                "description": scalar_description,
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
        }
        provider = self.Provider([json.dumps(copied)])

        result = resolver.PlanAgentQueryInterpreter(
            provider,
            model="fixture",
        ).propose(query, policy_card=single_task_policy())

        self.assertEqual(result["provider"]["attempt_count"], 1)
        self.assertEqual(
            result["experiment_needs"]["checker_need"],
            copied["checker_need"],
        )

    def _cold_historical_free_concern_class_name_remains_readable(self):
        self.assertIs(
            resolver.FreeConcernAgent,
            resolver.PlanAgentQueryInterpreter,
        )


class InventoryTests(unittest.TestCase):
    def test_discovers_official_library_and_only_attaches_catalog_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "envs").mkdir()
            instructions = root / "description" / "task_instruction"
            instructions.mkdir(parents=True)
            for task_name, description in (
                ("beat_block_hammer", "grab a hammer and beat a block"),
                ("open_laptop", "open a laptop lid"),
            ):
                (root / "envs" / f"{task_name}.py").write_text(
                    "class Task: pass\n", encoding="utf-8"
                )
                (instructions / f"{task_name}.json").write_text(
                    json.dumps({"full_description": description}),
                    encoding="utf-8",
                )
            (root / "envs" / "missing_instruction.py").write_text(
                "class Task: pass\n", encoding="utf-8"
            )
            discovered = resolver.discover_robotwin_task_inventory(
                root,
                capability_catalog={
                    "tasks": [
                        {
                            "task_name": "beat_block_hammer",
                            "aspects": [{"aspect_id": "novel.free.concern"}],
                        }
                    ]
                },
            )
        self.assertEqual(
            [item["task_name"] for item in discovered],
            ["beat_block_hammer", "open_laptop"],
        )
        self.assertEqual(discovered[0]["execution_status"], "capability_registered")
        self.assertEqual(discovered[1]["execution_status"], "official_base_only")

    def test_runtime_inventory_requires_source_and_schema_not_registration(self):
        root = Path(__file__).resolve().parents[2]
        discovered = resolver.discover_robotwin_runtime_task_inventory(root)
        names = {item["task_name"] for item in discovered}

        self.assertTrue(
            {
                "adjust_bottle",
                "beat_block_hammer",
                "click_bell",
                "grab_roller",
                "place_phone_stand",
            }.issubset(names)
        )
        for task_name in names:
            self.assertTrue((root / "envs" / f"{task_name}.py").is_file())
            self.assertTrue(
                (
                    root
                    / "mea"
                    / "toolkit"
                    / "schemas"
                    / f"{task_name}.json"
                ).is_file()
            )

    def test_catalog_capabilities_do_not_change_semantic_ranking(self):
        base = inventory()
        changed = [dict(item) for item in base]
        changed[0] = {
            **changed[0],
            "capability_aspects": ["completely.different.catalog.entry"],
        }
        query_concern = concern("grab a hammer and strike the target block")
        self.assertEqual(
            [
                (item["task_name"], item["score"])
                for item in resolver.rank_official_tasks(query_concern, base)
            ],
            [
                (item["task_name"], item["score"])
                for item in resolver.rank_official_tasks(query_concern, changed)
            ],
        )


class PolicyGateTests(unittest.TestCase):
    def test_single_task_anchor_wins_within_semantic_near_tie(self):
        value = concern(
            "Click the correct target object based on its predefined "
            "identity and location."
        )
        result = resolver.resolve_open_task(
            value,
            policy_card=single_task_policy("click_bell"),
            inventory=click_near_tie_inventory(),
        )
        self.assertEqual(result["ranked_candidates"][0]["task_name"], "scan_object")
        self.assertEqual(result["decision"], "retrieve_and_adapt")
        self.assertEqual(
            result["reason_code"], "policy_compatible_semantic_near_tie"
        )
        self.assertEqual(result["selected_base_task"]["task_name"], "click_bell")
        self.assertIn(
            "click_bell",
            result["resolution_contract"]["plausible_candidate_names"],
        )

    def test_single_task_retrieves_same_task_for_catalog_external_concern(self):
        result = resolver.resolve_open_task(
            concern("grab a hammer and strike the target block"),
            policy_card=single_task_policy(),
            inventory=inventory(),
        )
        self.assertEqual(result["decision"], "retrieve_and_adapt")
        self.assertEqual(
            result["selected_base_task"]["task_name"], "beat_block_hammer"
        )
        self.assertEqual(
            result["query_interpretation"]["sub_aspect"],
            "novel.reflective_surface_confusion",
        )
        self.assertEqual(
            result["resolution_contract"]["catalog_role"],
            "execution_capability_inventory_only",
        )

    def test_single_task_rejects_query_for_another_official_task(self):
        result = resolver.resolve_open_task(
            concern("lift the laptop lid and open the laptop"),
            policy_card=single_task_policy(),
            inventory=inventory(),
        )
        self.assertEqual(result["decision"], "unsupported")
        self.assertEqual(result["reason_code"], "policy_task_mismatch")
        self.assertIsNone(result["selected_base_task"])
        self.assertEqual(result["ranked_candidates"][0]["task_name"], "open_laptop")

    def _cold_single_task_rejects_clearly_wrong_cross_task_even_with_margin(self):
        result = resolver.resolve_open_task(
            concern("lift the laptop lid until the laptop is fully open"),
            policy_card=single_task_policy(),
            inventory=inventory(),
            near_tie_margin=0.1,
        )
        self.assertEqual(result["decision"], "unsupported")
        self.assertEqual(result["reason_code"], "policy_task_mismatch")
        self.assertNotIn(
            "beat_block_hammer",
            result["resolution_contract"]["plausible_candidate_names"],
        )

    def test_multitask_checkpoint_can_retrieve_its_other_training_task(self):
        result = resolver.resolve_open_task(
            concern("lift the laptop lid until it is open"),
            policy_card=multi_task_policy(),
            inventory=inventory(),
        )
        self.assertEqual(result["decision"], "retrieve_and_adapt")
        self.assertEqual(result["reason_code"], "nearest_training_task")
        self.assertEqual(result["selected_base_task"]["task_name"], "open_laptop")

    def test_open_language_policy_can_generate_when_no_base_is_near(self):
        result = resolver.resolve_open_task(
            concern("water a potted plant with a watering can"),
            policy_card=multi_task_policy(supports_unseen=True),
            inventory=inventory(),
            semantic_threshold=0.3,
            can_generate_new_task=True,
        )
        self.assertEqual(result["decision"], "generate_new")
        self.assertEqual(result["reason_code"], "no_near_official_base")
        self.assertIsNone(result["selected_base_task"])

    def test_generation_is_not_inferred_from_language_conditioning_alone(self):
        result = resolver.resolve_open_task(
            concern("water a potted plant with a watering can"),
            policy_card=multi_task_policy(supports_unseen=False),
            inventory=inventory(),
            semantic_threshold=0.3,
            can_generate_new_task=True,
        )
        self.assertEqual(result["decision"], "unsupported")
        self.assertEqual(result["reason_code"], "policy_not_open_task_capable")

    def test_single_task_cannot_claim_unseen_task_support(self):
        card = single_task_policy()
        card["supports_unseen_tasks"] = True
        card["language_conditioned"] = True
        with self.assertRaisesRegex(
            resolver.OpenTaskResolutionError, "single-task checkpoint"
        ):
            resolver.policy_task_scope_from_card(card)


if __name__ == "__main__":
    unittest.main()
