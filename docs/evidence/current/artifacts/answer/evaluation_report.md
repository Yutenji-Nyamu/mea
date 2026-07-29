# MEA Multi-Round Evaluation Report

## Identity

- evaluation id: `eval_20260729_b30_refinement_live_v2`
- user query: Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?
- executed rounds: `3`
- total episodes: `3`

## Plan Agent decisions

```json
[
  {
    "schema_version": 2,
    "action": "continue",
    "transition": "switch_concern",
    "candidate_id": "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
    "observation_summary": "Testing object position variation directly probes the robustness of the ACT policy under manipulated-object property changes, addressing a key uncertainty in the original Query.",
    "decision_reason": "provider_authored_open_world_step",
    "answered_query": false,
    "plan_step_source": "provider_claim_first_open_query",
    "planning_lineage": {
      "schema_version": 1,
      "decision_kind": "evidence_conditioned_refinement",
      "evidence_conditioned": true,
      "completed_round_ids": [
        "round_1"
      ],
      "completed_round_count": 1,
      "input_digest": "ec67ceece3e66171cc151cd19cab45bb8c2e73af05f6995d2f3570c78db34d1a"
    },
    "plan_step_proposal": {
      "schema_version": 2,
      "action": "propose",
      "aspect_id": "task_execution.object_position_variation",
      "candidate_id": "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
      "execution_mode": "reuse_or_generate",
      "experiment_candidate": {
        "schema_version": 2,
        "candidate_id": "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
        "source_query": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?",
        "base_task": "click_bell",
        "semantic_concern": "task_execution.object_position_variation: The ACT policy fails to achieve success when the bell's position is perturbed within the allowable bounds.",
        "scene_need": {
          "kind": "adapt",
          "description": "Introduce a bounded variation in the bell's position. Preserve unchanged: task identity; policy checkpoint.",
          "reuse_first": true
        },
        "checker_need": null,
        "rule_tool_need": {
          "kind": "measure",
          "description": "Numeric or symbolic Rule Tool observable needed. Hypothesis: The ACT policy fails to achieve success when the bell's position is perturbed within the allowable bounds.",
          "reuse_first": true
        },
        "vqa_tool_need": null,
        "tool_need": {
          "kind": "measure",
          "description": "Numeric or symbolic Rule Tool observable needed. Hypothesis: The ACT policy fails to achieve success when the bell's position is perturbed within the allowable bounds.",
          "reuse_first": true
        },
        "evaluation_intent": {
          "schema_version": 1,
          "intent_id": "intent.fbdf63ddcef3951b",
          "source_query": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?",
          "original_concern": "task_execution.object_position_variation",
          "hypothesis": "The ACT policy fails to achieve success when the bell's position is perturbed within the allowable bounds.",
          "requested_change": "Introduce a bounded variation in the bell's position.",
          "preserved_conditions": [
            "task identity",
            "policy checkpoint"
          ],
          "required_observation": "Numeric or symbolic Rule Tool observable needed."
        },
        "intent_alignment": {
          "schema_version": 1,
          "relationship": "direct",
          "rationale": "Candidate preserves the requested change, hypothesis, and observation semantics.",
          "matched_intent_fields": [
            "requested_change",
            "preserved_conditions",
            "hypothesis",
            "required_observation"
          ],
          "unmatched_intent_fields": []
        }
      },
      "rationale": "Testing object position variation directly probes the robustness of the ACT policy under manipulated-object property changes, addressing a key uncertainty in the original Query.",
      "answered_query": false,
      "planning_lineage": {
        "schema_version": 1,
        "decision_kind": "evidence_conditioned_refinement",
        "evidence_conditioned": true,
        "completed_round_ids": [
          "round_1"
        ],
        "completed_round_count": 1,
        "input_digest": "ec67ceece3e66171cc151cd19cab45bb8c2e73af05f6995d2f3570c78db34d1a"
      }
    },
    "round_budget_before_decision": 2,
    "query_assessment": {
      "schema_version": 1,
      "contract": {
        "schema_version": 3,
        "claim_type": "diagnostic",
        "candidate_universe": [
          "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23"
        ],
        "required_coverage": {
          "candidate_ids": [
            "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23"
          ],
          "minimum_evaluated": 1,
          "minimum_per_group": null
        },
        "round_budget": 2,
        "comparison_groups": null,
        "candidate_universe_closed": false,
        "existential_witness_outcome": null,
        "control_requirement": "required"
      },
      "should_stop": false,
      "stop_reason": "continue",
      "claim_verdict": "inconclusive",
      "evidence_sufficient": false,
      "completed_rounds": 0,
      "round_budget": 2,
      "budget_remaining": 2,
      "candidate_universe_closed": false,
      "candidate_discovery_required": true,
      "observed_candidate_ids": [],
      "decisive_candidate_ids": [],
      "conflict_candidate_ids": [],
      "unknown_candidate_ids": [],
      "untested_required_candidate_ids": [
        "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23"
      ],
      "untested_candidate_ids": [
        "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23"
      ],
      "recommended_candidate_ids": [
        "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23"
      ],
      "rationale": "The query contract still has unresolved required evidence.",
      "statistics": {
        "diagnosed_failure_candidate_ids": []
      },
      "limitations": [
        "This is a finite-domain stopping prototype, not a statistical generalization guarantee.",
        "The candidate universe is open; exhaustive, no-counterexample, and worst-case conclusions are not licensed.",
        "Diagnosis strings are trusted upstream evidence labels; this contract does not independently infer or validate causality."
      ]
    },
    "next_round": {
      "round_id": "round_2",
      "template_id": null,
      "candidate_id": "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
      "experiment_candidate": {
        "schema_version": 2,
        "candidate_id": "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
        "source_query": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?",
        "base_task": "click_bell",
        "semantic_concern": "task_execution.object_position_variation: The ACT policy fails to achieve success when the bell's position is perturbed within the allowable bounds.",
        "scene_need": {
          "kind": "adapt",
          "description": "Introduce a bounded variation in the bell's position. Preserve unchanged: task identity; policy checkpoint.",
          "reuse_first": true
        },
        "checker_need": null,
        "rule_tool_need": {
          "kind": "measure",
          "description": "Numeric or symbolic Rule Tool observable needed. Hypothesis: The ACT policy fails to achieve success when the bell's position is perturbed within the allowable bounds.",
          "reuse_first": true
        },
        "vqa_tool_need": null,
        "tool_need": {
          "kind": "measure",
          "description": "Numeric or symbolic Rule Tool observable needed. Hypothesis: The ACT policy fails to achieve success when the bell's position is perturbed within the allowable bounds.",
          "reuse_first": true
        },
        "evaluation_intent": {
          "schema_version": 1,
          "intent_id": "intent.fbdf63ddcef3951b",
          "source_query": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?",
          "original_concern": "task_execution.object_position_variation",
          "hypothesis": "The ACT policy fails to achieve success when the bell's position is perturbed within the allowable bounds.",
          "requested_change": "Introduce a bounded variation in the bell's position.",
          "preserved_conditions": [
            "task identity",
            "policy checkpoint"
          ],
          "required_observation": "Numeric or symbolic Rule Tool observable needed."
        },
        "intent_alignment": {
          "schema_version": 1,
          "relationship": "direct",
          "rationale": "Candidate preserves the requested change, hypothesis, and observation semantics.",
          "matched_intent_fields": [
            "requested_change",
            "preserved_conditions",
            "hypothesis",
            "required_observation"
          ],
          "unmatched_intent_fields": []
        }
      },
      "sub_aspect": "task_execution.object_position_variation",
      "rationale": "Materialize only the Query-derived Task or Tool needs; no catalog template authorizes this round.",
      "task_instruction": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?\nScene need: Introduce a bounded variation in the bell's position. Preserve unchanged: task identity; policy checkpoint.\nChecker need: reuse the official implementation",
      "task_name": "click_bell",
      "task_module": null,
      "telemetry_profile": "balanced_v1",
      "route": "generic_provider_scene_checker_codegen",
      "variant_hint": {},
      "execution": {
        "backend": "act",
        "seeds": [
          100000
        ],
        "num_episodes": 1,
        "gates": [
          "ast",
          "render",
          "visual_diagnosis",
          "expert",
          "act",
          "toolkit",
          "planned_tool",
          "aggregate"
        ]
      },
      "observations": [
        "scene_alignment",
        "expert_solvable",
        "trusted_tools",
        "planned_tool",
        "aggregate"
      ],
      "tool_request": {
        "schema_version": 1,
        "task_name": "click_bell",
        "metric": "official_check_success",
        "question": "Fallback only: did the task success predicate pass?"
      },
      "open_tool_request_deferred": true,
      "vqa_phenomenon_ids": [],
      "semantic_need_execution": {
        "schema_version": 2,
        "candidate_id": "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
        "task": {
          "requested": true,
          "description": "Introduce a bounded variation in the bell's position. Preserve unchanged: task identity; policy checkpoint.",
          "route": "generic_provider_scene_checker_codegen",
          "status": "selected"
        },
        "checker": {
          "requested": false,
          "description": null,
          "route": "official_checker_reuse",
          "status": "not_requested"
        },
        "rule_tool": {
          "requested": true,
          "description": "Numeric or symbolic Rule Tool observable needed. Hypothesis: The ACT policy fails to achieve success when the bell's position is perturbed within the allowable bounds.",
          "route": "after_executed_telemetry_schema",
          "status": "pending"
        },
        "vqa_tool": {
          "requested": false,
          "description": null,
          "route": "not_requested",
          "status": "not_requested"
        }
      }
    }
  },
  {
    "schema_version": 2,
    "action": "continue",
    "transition": "switch_concern",
    "candidate_id": "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee",
    "observation_summary": "Testing object instance variation is the next most informative step to identify potential weaknesses in the ACT policy under manipulated-object property changes, as it isolates a new causal factor not yet tested.",
    "decision_reason": "provider_authored_open_world_step",
    "answered_query": false,
    "plan_step_source": "provider_claim_first_open_query",
    "planning_lineage": {
      "schema_version": 1,
      "decision_kind": "evidence_conditioned_refinement",
      "evidence_conditioned": true,
      "completed_round_ids": [
        "round_1",
        "round_2"
      ],
      "completed_round_count": 2,
      "input_digest": "c0aea84e3ddc811ceecb455e7e5f8556adec93e518669fa41c398efcc283d2f3"
    },
    "plan_step_proposal": {
      "schema_version": 2,
      "action": "propose",
      "aspect_id": "task_execution.object_instance_variation",
      "candidate_id": "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee",
      "execution_mode": "reuse_or_generate",
      "experiment_candidate": {
        "schema_version": 2,
        "candidate_id": "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee",
        "source_query": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?",
        "base_task": "click_bell",
        "semantic_concern": "task_execution.object_instance_variation: The ACT policy fails to achieve success when the bell's object instance is perturbed within allowable bounds.",
        "scene_need": {
          "kind": "adapt",
          "description": "Introduce a bounded variation in the bell's object instance. Preserve unchanged: task identity; policy checkpoint.",
          "reuse_first": true
        },
        "checker_need": null,
        "rule_tool_need": {
          "kind": "measure",
          "description": "Numeric or symbolic Rule Tool observable needed. Hypothesis: The ACT policy fails to achieve success when the bell's object instance is perturbed within allowable bounds.",
          "reuse_first": true
        },
        "vqa_tool_need": null,
        "tool_need": {
          "kind": "measure",
          "description": "Numeric or symbolic Rule Tool observable needed. Hypothesis: The ACT policy fails to achieve success when the bell's object instance is perturbed within allowable bounds.",
          "reuse_first": true
        },
        "evaluation_intent": {
          "schema_version": 1,
          "intent_id": "intent.6f8ac1db644ae8b0",
          "source_query": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?",
          "original_concern": "task_execution.object_instance_variation",
          "hypothesis": "The ACT policy fails to achieve success when the bell's object instance is perturbed within allowable bounds.",
          "requested_change": "Introduce a bounded variation in the bell's object instance.",
          "preserved_conditions": [
            "task identity",
            "policy checkpoint"
          ],
          "required_observation": "Numeric or symbolic Rule Tool observable needed."
        },
        "intent_alignment": {
          "schema_version": 1,
          "relationship": "direct",
          "rationale": "Candidate preserves the requested change, hypothesis, and observation semantics.",
          "matched_intent_fields": [
            "requested_change",
            "preserved_conditions",
            "hypothesis",
            "required_observation"
          ],
          "unmatched_intent_fields": []
        }
      },
      "rationale": "Testing object instance variation is the next most informative step to identify potential weaknesses in the ACT policy under manipulated-object property changes, as it isolates a new causal factor not yet tested.",
      "answered_query": false,
      "planning_lineage": {
        "schema_version": 1,
        "decision_kind": "evidence_conditioned_refinement",
        "evidence_conditioned": true,
        "completed_round_ids": [
          "round_1",
          "round_2"
        ],
        "completed_round_count": 2,
        "input_digest": "c0aea84e3ddc811ceecb455e7e5f8556adec93e518669fa41c398efcc283d2f3"
      }
    },
    "round_budget_before_decision": 1,
    "query_assessment": {
      "schema_version": 1,
      "contract": {
        "schema_version": 3,
        "claim_type": "diagnostic",
        "candidate_universe": [
          "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
          "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee"
        ],
        "required_coverage": {
          "candidate_ids": [
            "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
            "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee"
          ],
          "minimum_evaluated": 1,
          "minimum_per_group": null
        },
        "round_budget": 2,
        "comparison_groups": null,
        "candidate_universe_closed": false,
        "existential_witness_outcome": null,
        "control_requirement": "required"
      },
      "should_stop": false,
      "stop_reason": "continue",
      "claim_verdict": "inconclusive",
      "evidence_sufficient": false,
      "completed_rounds": 1,
      "round_budget": 2,
      "budget_remaining": 1,
      "candidate_universe_closed": false,
      "candidate_discovery_required": true,
      "observed_candidate_ids": [
        "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23"
      ],
      "decisive_candidate_ids": [
        "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23"
      ],
      "conflict_candidate_ids": [],
      "unknown_candidate_ids": [],
      "untested_required_candidate_ids": [
        "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee"
      ],
      "untested_candidate_ids": [
        "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee"
      ],
      "recommended_candidate_ids": [
        "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee"
      ],
      "rationale": "The query contract still has unresolved required evidence.",
      "statistics": {
        "diagnosed_failure_candidate_ids": []
      },
      "limitations": [
        "This is a finite-domain stopping prototype, not a statistical generalization guarantee.",
        "The candidate universe is open; exhaustive, no-counterexample, and worst-case conclusions are not licensed.",
        "Diagnosis strings are trusted upstream evidence labels; this contract does not independently infer or validate causality."
      ]
    },
    "next_round": {
      "round_id": "round_3",
      "template_id": null,
      "candidate_id": "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee",
      "experiment_candidate": {
        "schema_version": 2,
        "candidate_id": "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee",
        "source_query": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?",
        "base_task": "click_bell",
        "semantic_concern": "task_execution.object_instance_variation: The ACT policy fails to achieve success when the bell's object instance is perturbed within allowable bounds.",
        "scene_need": {
          "kind": "adapt",
          "description": "Introduce a bounded variation in the bell's object instance. Preserve unchanged: task identity; policy checkpoint.",
          "reuse_first": true
        },
        "checker_need": null,
        "rule_tool_need": {
          "kind": "measure",
          "description": "Numeric or symbolic Rule Tool observable needed. Hypothesis: The ACT policy fails to achieve success when the bell's object instance is perturbed within allowable bounds.",
          "reuse_first": true
        },
        "vqa_tool_need": null,
        "tool_need": {
          "kind": "measure",
          "description": "Numeric or symbolic Rule Tool observable needed. Hypothesis: The ACT policy fails to achieve success when the bell's object instance is perturbed within allowable bounds.",
          "reuse_first": true
        },
        "evaluation_intent": {
          "schema_version": 1,
          "intent_id": "intent.6f8ac1db644ae8b0",
          "source_query": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?",
          "original_concern": "task_execution.object_instance_variation",
          "hypothesis": "The ACT policy fails to achieve success when the bell's object instance is perturbed within allowable bounds.",
          "requested_change": "Introduce a bounded variation in the bell's object instance.",
          "preserved_conditions": [
            "task identity",
            "policy checkpoint"
          ],
          "required_observation": "Numeric or symbolic Rule Tool observable needed."
        },
        "intent_alignment": {
          "schema_version": 1,
          "relationship": "direct",
          "rationale": "Candidate preserves the requested change, hypothesis, and observation semantics.",
          "matched_intent_fields": [
            "requested_change",
            "preserved_conditions",
            "hypothesis",
            "required_observation"
          ],
          "unmatched_intent_fields": []
        }
      },
      "sub_aspect": "task_execution.object_instance_variation",
      "rationale": "Materialize only the Query-derived Task or Tool needs; no catalog template authorizes this round.",
      "task_instruction": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?\nScene need: Introduce a bounded variation in the bell's object instance. Preserve unchanged: task identity; policy checkpoint.\nChecker need: reuse the official implementation",
      "task_name": "click_bell",
      "task_module": null,
      "telemetry_profile": "balanced_v1",
      "route": "generic_provider_scene_checker_codegen",
      "variant_hint": {},
      "execution": {
        "backend": "act",
        "seeds": [
          100000
        ],
        "num_episodes": 1,
        "gates": [
          "ast",
          "render",
          "visual_diagnosis",
          "expert",
          "act",
          "toolkit",
          "planned_tool",
          "aggregate"
        ]
      },
      "observations": [
        "scene_alignment",
        "expert_solvable",
        "trusted_tools",
        "planned_tool",
        "aggregate"
      ],
      "tool_request": {
        "schema_version": 1,
        "task_name": "click_bell",
        "metric": "official_check_success",
        "question": "Fallback only: did the task success predicate pass?"
      },
      "open_tool_request_deferred": true,
      "vqa_phenomenon_ids": [],
      "semantic_need_execution": {
        "schema_version": 2,
        "candidate_id": "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee",
        "task": {
          "requested": true,
          "description": "Introduce a bounded variation in the bell's object instance. Preserve unchanged: task identity; policy checkpoint.",
          "route": "generic_provider_scene_checker_codegen",
          "status": "selected"
        },
        "checker": {
          "requested": false,
          "description": null,
          "route": "official_checker_reuse",
          "status": "not_requested"
        },
        "rule_tool": {
          "requested": true,
          "description": "Numeric or symbolic Rule Tool observable needed. Hypothesis: The ACT policy fails to achieve success when the bell's object instance is perturbed within allowable bounds.",
          "route": "after_executed_telemetry_schema",
          "status": "pending"
        },
        "vqa_tool": {
          "requested": false,
          "description": null,
          "route": "not_requested",
          "status": "not_requested"
        }
      }
    }
  },
  {
    "schema_version": 3,
    "action": "stop",
    "transition": "stop",
    "next_aspect_id": null,
    "next_template_id": null,
    "observation_summary": "A measured failure has an evidence-backed diagnosis and the minimum diagnostic coverage is met.",
    "decision_reason": "claim_first_query_sufficiency_contract",
    "answered_query": true,
    "plan_step_source": "deterministic_query_sufficiency_contract",
    "round_budget_before_decision": 0,
    "evidence_assessment": {
      "schema_version": 1,
      "contract": {
        "schema_version": 3,
        "claim_type": "diagnostic",
        "candidate_universe": [
          "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
          "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee"
        ],
        "required_coverage": {
          "candidate_ids": [
            "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
            "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee"
          ],
          "minimum_evaluated": 1,
          "minimum_per_group": null
        },
        "round_budget": 2,
        "comparison_groups": null,
        "candidate_universe_closed": false,
        "existential_witness_outcome": null,
        "control_requirement": "required"
      },
      "should_stop": true,
      "stop_reason": "evidence_sufficient",
      "claim_verdict": "diagnosed",
      "evidence_sufficient": true,
      "completed_rounds": 2,
      "round_budget": 2,
      "budget_remaining": 0,
      "candidate_universe_closed": false,
      "candidate_discovery_required": false,
      "observed_candidate_ids": [
        "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
        "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee"
      ],
      "decisive_candidate_ids": [
        "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
        "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee"
      ],
      "conflict_candidate_ids": [],
      "unknown_candidate_ids": [],
      "untested_required_candidate_ids": [],
      "untested_candidate_ids": [],
      "recommended_candidate_ids": [],
      "rationale": "A measured failure has an evidence-backed diagnosis and the minimum diagnostic coverage is met.",
      "statistics": {
        "diagnosed_failure_candidate_ids": [
          "dynamic.click.bell.task.execution.object.instance.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.object.instance.is.perturbed.within.allowable.bounds.5cc7bb074cee"
        ]
      },
      "limitations": [
        "This is a finite-domain stopping prototype, not a statistical generalization guarantee.",
        "The candidate universe is open; exhaustive, no-counterexample, and worst-case conclusions are not licensed.",
        "Diagnosis strings are trusted upstream evidence labels; this contract does not independently infer or validate causality."
      ]
    },
    "next_round": null
  }
]
```

## Historical planning retrieval

以下历史只作为 planning prior，不属于本次 policy evidence。

```json
{
  "schema_version": 1,
  "status": "disabled",
  "match_count": 0,
  "matches": []
}
```

## Round evidence

### round_1: `task_execution.official_baseline`

- TaskGen route: `official`
- instruction: Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?
- seeds: `[100000]`
- episodes: `1`
- selected retrieval tasks: none (reuse route)
- observed color: `None`
- expert solvable: `None`
- execution backend: `ACT`
- ACT pipeline status: `True`
- policy success: `1.0`
- pipeline passed: `True`
- planned Tool requested route: `auto`
- planned Tool resolved route: `reuse`
- planned Tool source: `trusted_catalog`
- planned Tool: `official_check_success`
- planned Tool validation: `{'provider_called': False, 'catalog_tool_found': True, 'episode_count': 1}`
- planned Tool results:
  - ACT (policy_under_evaluation) seed 100000: value=True, evidence_steps=[5505]
- position samples:
  - none
- trusted Tool results:
  - ACT seed 100000: time_to_success=22.02 s

### round_2: `task_execution.object_position_variation`

- TaskGen route: `generic_provider_scene_checker_codegen`
- instruction: Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?
Scene need: Introduce a bounded variation in the bell's position. Preserve unchanged: task identity; policy checkpoint.
Checker need: reuse the official implementation
- seeds: `[100000]`
- episodes: `1`
- selected retrieval tasks: none (reuse route)
- observed color: `None`
- expert solvable: `True`
- execution backend: `ACT`
- ACT pipeline status: `True`
- policy success: `1.0`
- pipeline passed: `True`
- planned Tool requested route: `auto`
- planned Tool resolved route: `typed_metric_spec_compile`
- planned Tool source: `run_local_generated`
- planned Tool: `query_derived_metric`
- planned Tool validation: `{'provider_called': False, 'episode_count': 1}`
- planned Tool results:
  - ACT (policy_under_evaluation) seed 100000: value=0.021466496280102353, evidence_steps=[5514]
- position samples:
  - none
- trusted Tool results:
  - ACT seed 100000: official_check_success=True, time_to_success=22.056 s

### round_3: `task_execution.object_instance_variation`

- TaskGen route: `generic_provider_scene_checker_codegen`
- instruction: Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?
Scene need: Introduce a bounded variation in the bell's object instance. Preserve unchanged: task identity; policy checkpoint.
Checker need: reuse the official implementation
- seeds: `[100000]`
- episodes: `1`
- selected retrieval tasks: none (reuse route)
- observed color: `None`
- expert solvable: `True`
- execution backend: `ACT`
- ACT pipeline status: `True`
- policy success: `0.0`
- pipeline passed: `True`
- planned Tool requested route: `auto`
- planned Tool resolved route: `run_local_reuse`
- planned Tool source: `run_local_registry`
- planned Tool: `query_derived_metric`
- planned Tool validation: `{'provider_called': False, 'episode_count': 1}`
- planned Tool results:
  - ACT (policy_under_evaluation) seed 100000: value=0.0027531830083962435, evidence_steps=[3880]
- position samples:
  - none
- trusted Tool results:
  - ACT seed 100000: official_check_success=False, time_to_success=None s


## Aggregate observations

- scene alignment: `True`
- observed color by round: `[None, None, None]`
- expert solvable: `True`
- execution backends: `['ACT']`
- ACT pipeline status: `True`
- weighted policy success: `0.6666666666666666`
- policy success by round: `[1.0, 1.0, 0.0]`
- position varied: `None`
- position metrics: `{}`
- pipeline passed: `True`

## Deterministic Aggregate Toolkit

以下数值直接来自 deterministic Aggregate Toolkit；Feedback Agent 没有重新计算 episode 统计量。

```json
{
  "schema_version": 1,
  "status": "passed",
  "source_count": 5,
  "unique_episode_count": 5,
  "input_issues": [],
  "metrics": [
    {
      "metric": "official_check_success",
      "value_kind": "boolean",
      "unit": null,
      "cohorts": [
        {
          "role": "policy_under_evaluation",
          "policy_names": [
            "ACT"
          ],
          "summary": {
            "episode_result_count": 3,
            "quality": {
              "valid": 3,
              "missing": 0,
              "invalid": 0
            },
            "statistics": {
              "true_count": {
                "value": 2
              },
              "true_rate": {
                "value": 0.6666666666666666,
                "numerator": 2,
                "denominator": 3
              },
              "false_count": {
                "value": 1
              },
              "false_rate": {
                "value": 0.3333333333333333,
                "numerator": 1,
                "denominator": 3
              },
              "success_count": {
                "value": 2
              },
              "success_rate": {
                "value": 0.6666666666666666,
                "numerator": 2,
                "denominator": 3
              }
            }
          },
          "passed_summary": {
            "episode_result_count": 3,
            "quality": {
              "valid": 3,
              "missing": 0,
              "invalid": 0
            },
            "statistics": {
              "true_count": {
                "value": 2
              },
              "true_rate": {
                "value": 0.6666666666666666,
                "numerator": 2,
                "denominator": 3
              },
              "false_count": {
                "value": 1
              },
              "false_rate": {
                "value": 0.3333333333333333,
                "numerator": 1,
                "denominator": 3
              }
            }
          },
          "groups": {
            "seed": [
              {
                "value": 100000,
                "summary": {
                  "episode_result_count": 3,
                  "quality": {
                    "valid": 3,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 2
                    },
                    "true_rate": {
                      "value": 0.6666666666666666,
                      "numerator": 2,
                      "denominator": 3
                    },
                    "false_count": {
                      "value": 1
                    },
                    "false_rate": {
                      "value": 0.3333333333333333,
                      "numerator": 1,
                      "denominator": 3
                    },
                    "success_count": {
                      "value": 2
                    },
                    "success_rate": {
                      "value": 0.6666666666666666,
                      "numerator": 2,
                      "denominator": 3
                    }
                  }
                },
                "passed_summary": {
                  "episode_result_count": 3,
                  "quality": {
                    "valid": 3,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 2
                    },
                    "true_rate": {
                      "value": 0.6666666666666666,
                      "numerator": 2,
                      "denominator": 3
                    },
                    "false_count": {
                      "value": 1
                    },
                    "false_rate": {
                      "value": 0.3333333333333333,
                      "numerator": 1,
                      "denominator": 3
                    }
                  }
                }
              }
            ],
            "round_id": [
              {
                "value": "round_1",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 1
                    },
                    "true_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 0
                    },
                    "false_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    },
                    "success_count": {
                      "value": 1
                    },
                    "success_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    }
                  }
                },
                "passed_summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 1
                    },
                    "true_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 0
                    },
                    "false_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    }
                  }
                }
              },
              {
                "value": "round_2",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 1
                    },
                    "true_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 0
                    },
                    "false_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    },
                    "success_count": {
                      "value": 1
                    },
                    "success_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    }
                  }
                },
                "passed_summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 1
                    },
                    "true_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 0
                    },
                    "false_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    }
                  }
                }
              },
              {
                "value": "round_3",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 0
                    },
                    "true_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 1
                    },
                    "false_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    },
                    "success_count": {
                      "value": 0
                    },
                    "success_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    }
                  }
                },
                "passed_summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 0
                    },
                    "true_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 1
                    },
                    "false_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    }
                  }
                }
              }
            ],
            "variant": [
              {
                "value": "task_execution.object_instance_variation",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 0
                    },
                    "true_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 1
                    },
                    "false_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    },
                    "success_count": {
                      "value": 0
                    },
                    "success_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    }
                  }
                },
                "passed_summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 0
                    },
                    "true_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 1
                    },
                    "false_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    }
                  }
                }
              },
              {
                "value": "task_execution.object_position_variation",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 1
                    },
                    "true_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 0
                    },
                    "false_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    },
                    "success_count": {
                      "value": 1
                    },
                    "success_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    }
                  }
                },
                "passed_summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 1
                    },
                    "true_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 0
                    },
                    "false_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    }
                  }
                }
              },
              {
                "value": "task_execution.official_baseline",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 1
                    },
                    "true_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 0
                    },
                    "false_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    },
                    "success_count": {
                      "value": 1
                    },
                    "success_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    }
                  }
                },
                "passed_summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 1
                    },
                    "true_rate": {
                      "value": 1.0,
                      "numerator": 1,
                      "denominator": 1
                    },
                    "false_count": {
                      "value": 0
                    },
                    "false_rate": {
                      "value": 0.0,
                      "numerator": 0,
                      "denominator": 1
                    }
                  }
                }
              }
            ],
            "policy_name": [
              {
                "value": "ACT",
                "summary": {
                  "episode_result_count": 3,
                  "quality": {
                    "valid": 3,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 2
                    },
                    "true_rate": {
                      "value": 0.6666666666666666,
                      "numerator": 2,
                      "denominator": 3
                    },
                    "false_count": {
                      "value": 1
                    },
                    "false_rate": {
                      "value": 0.3333333333333333,
                      "numerator": 1,
                      "denominator": 3
                    },
                    "success_count": {
                      "value": 2
                    },
                    "success_rate": {
                      "value": 0.6666666666666666,
                      "numerator": 2,
                      "denominator": 3
                    }
                  }
                },
                "passed_summary": {
                  "episode_result_count": 3,
                  "quality": {
                    "valid": 3,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "true_count": {
                      "value": 2
                    },
                    "true_rate": {
                      "value": 0.6666666666666666,
                      "numerator": 2,
                      "denominator": 3
                    },
                    "false_count": {
                      "value": 1
                    },
                    "false_rate": {
                      "value": 0.3333333333333333,
                      "numerator": 1,
                      "denominator": 3
                    }
                  }
                }
              }
            ]
          }
        }
      ]
    },
    {
      "metric": "query_derived_metric",
      "value_kind": "numeric",
      "unit": "m",
      "cohorts": [
        {
          "role": "policy_under_evaluation",
          "policy_names": [
            "ACT"
          ],
          "summary": {
            "episode_result_count": 2,
            "quality": {
              "valid": 2,
              "missing": 0,
              "invalid": 0
            },
            "statistics": {
              "mean": {
                "value": 0.012109839644249299
              },
              "median": {
                "value": 0.012109839644249299
              },
              "min": {
                "value": 0.0027531830083962435
              },
              "max": {
                "value": 0.021466496280102353
              },
              "population_stddev": {
                "value": 0.009356656635853054
              }
            }
          },
          "passed_summary": null,
          "groups": {
            "seed": [
              {
                "value": 100000,
                "summary": {
                  "episode_result_count": 2,
                  "quality": {
                    "valid": 2,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 0.012109839644249299
                    },
                    "median": {
                      "value": 0.012109839644249299
                    },
                    "min": {
                      "value": 0.0027531830083962435
                    },
                    "max": {
                      "value": 0.021466496280102353
                    },
                    "population_stddev": {
                      "value": 0.009356656635853054
                    }
                  }
                },
                "passed_summary": null
              }
            ],
            "round_id": [
              {
                "value": "round_2",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 0.021466496280102353
                    },
                    "median": {
                      "value": 0.021466496280102353
                    },
                    "min": {
                      "value": 0.021466496280102353
                    },
                    "max": {
                      "value": 0.021466496280102353
                    },
                    "population_stddev": {
                      "value": 0.0
                    }
                  }
                },
                "passed_summary": null
              },
              {
                "value": "round_3",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 0.0027531830083962435
                    },
                    "median": {
                      "value": 0.0027531830083962435
                    },
                    "min": {
                      "value": 0.0027531830083962435
                    },
                    "max": {
                      "value": 0.0027531830083962435
                    },
                    "population_stddev": {
                      "value": 0.0
                    }
                  }
                },
                "passed_summary": null
              }
            ],
            "variant": [
              {
                "value": "task_execution.object_instance_variation",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 0.0027531830083962435
                    },
                    "median": {
                      "value": 0.0027531830083962435
                    },
                    "min": {
                      "value": 0.0027531830083962435
                    },
                    "max": {
                      "value": 0.0027531830083962435
                    },
                    "population_stddev": {
                      "value": 0.0
                    }
                  }
                },
                "passed_summary": null
              },
              {
                "value": "task_execution.object_position_variation",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 0.021466496280102353
                    },
                    "median": {
                      "value": 0.021466496280102353
                    },
                    "min": {
                      "value": 0.021466496280102353
                    },
                    "max": {
                      "value": 0.021466496280102353
                    },
                    "population_stddev": {
                      "value": 0.0
                    }
                  }
                },
                "passed_summary": null
              }
            ],
            "policy_name": [
              {
                "value": "ACT",
                "summary": {
                  "episode_result_count": 2,
                  "quality": {
                    "valid": 2,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 0.012109839644249299
                    },
                    "median": {
                      "value": 0.012109839644249299
                    },
                    "min": {
                      "value": 0.0027531830083962435
                    },
                    "max": {
                      "value": 0.021466496280102353
                    },
                    "population_stddev": {
                      "value": 0.009356656635853054
                    }
                  }
                },
                "passed_summary": null
              }
            ]
          }
        }
      ]
    },
    {
      "metric": "time_to_success",
      "value_kind": "numeric",
      "unit": "s",
      "cohorts": [
        {
          "role": "policy_under_evaluation",
          "policy_names": [
            "ACT"
          ],
          "summary": {
            "episode_result_count": 3,
            "quality": {
              "valid": 2,
              "missing": 1,
              "invalid": 0
            },
            "statistics": {
              "mean": {
                "value": 22.038
              },
              "median": {
                "value": 22.038
              },
              "min": {
                "value": 22.02
              },
              "max": {
                "value": 22.056
              },
              "population_stddev": {
                "value": 0.018000000000000682
              }
            }
          },
          "passed_summary": null,
          "groups": {
            "seed": [
              {
                "value": 100000,
                "summary": {
                  "episode_result_count": 3,
                  "quality": {
                    "valid": 2,
                    "missing": 1,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 22.038
                    },
                    "median": {
                      "value": 22.038
                    },
                    "min": {
                      "value": 22.02
                    },
                    "max": {
                      "value": 22.056
                    },
                    "population_stddev": {
                      "value": 0.018000000000000682
                    }
                  }
                },
                "passed_summary": null
              }
            ],
            "round_id": [
              {
                "value": "round_1",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 22.02
                    },
                    "median": {
                      "value": 22.02
                    },
                    "min": {
                      "value": 22.02
                    },
                    "max": {
                      "value": 22.02
                    },
                    "population_stddev": {
                      "value": 0.0
                    }
                  }
                },
                "passed_summary": null
              },
              {
                "value": "round_2",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 22.056
                    },
                    "median": {
                      "value": 22.056
                    },
                    "min": {
                      "value": 22.056
                    },
                    "max": {
                      "value": 22.056
                    },
                    "population_stddev": {
                      "value": 0.0
                    }
                  }
                },
                "passed_summary": null
              },
              {
                "value": "round_3",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 0,
                    "missing": 1,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": null
                    },
                    "median": {
                      "value": null
                    },
                    "min": {
                      "value": null
                    },
                    "max": {
                      "value": null
                    },
                    "population_stddev": {
                      "value": null
                    }
                  }
                },
                "passed_summary": null
              }
            ],
            "variant": [
              {
                "value": "task_execution.object_instance_variation",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 0,
                    "missing": 1,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": null
                    },
                    "median": {
                      "value": null
                    },
                    "min": {
                      "value": null
                    },
                    "max": {
                      "value": null
                    },
                    "population_stddev": {
                      "value": null
                    }
                  }
                },
                "passed_summary": null
              },
              {
                "value": "task_execution.object_position_variation",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 22.056
                    },
                    "median": {
                      "value": 22.056
                    },
                    "min": {
                      "value": 22.056
                    },
                    "max": {
                      "value": 22.056
                    },
                    "population_stddev": {
                      "value": 0.0
                    }
                  }
                },
                "passed_summary": null
              },
              {
                "value": "task_execution.official_baseline",
                "summary": {
                  "episode_result_count": 1,
                  "quality": {
                    "valid": 1,
                    "missing": 0,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 22.02
                    },
                    "median": {
                      "value": 22.02
                    },
                    "min": {
                      "value": 22.02
                    },
                    "max": {
                      "value": 22.02
                    },
                    "population_stddev": {
                      "value": 0.0
                    }
                  }
                },
                "passed_summary": null
              }
            ],
            "policy_name": [
              {
                "value": "ACT",
                "summary": {
                  "episode_result_count": 3,
                  "quality": {
                    "valid": 2,
                    "missing": 1,
                    "invalid": 0
                  },
                  "statistics": {
                    "mean": {
                      "value": 22.038
                    },
                    "median": {
                      "value": 22.038
                    },
                    "min": {
                      "value": 22.02
                    },
                    "max": {
                      "value": 22.056
                    },
                    "population_stddev": {
                      "value": 0.018000000000000682
                    }
                  }
                },
                "passed_summary": null
              }
            ]
          }
        }
      ]
    }
  ]
}
```

## Execution VQA

Execution VQA 只提供视觉补充证据。Simulator numeric Tool 保持权威；不一致会原样保留为 `evidence_conflict`。

```json
[
  {
    "round_id": "round_1",
    "status": "passed",
    "model_requested": "gpt-4o-2024-11-20",
    "representative_episode": "mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_1/evaluation/telemetry/act/episode_000_seed_100000",
    "evidence_conflict": false,
    "observation": {
      "phenomena": [
        {
          "id": "bell_visibly_pressed",
          "observed": true,
          "description": "The robot visibly presses the target bell.",
          "confidence": 1.0,
          "frame_ids": [
            "success_before",
            "success_after"
          ]
        }
      ],
      "confidence": 1.0,
      "frame_ids": [
        "success_before",
        "success_after"
      ],
      "numeric_consistency": "consistent",
      "conflicts": [],
      "evidence_conflict": false
    },
    "selected_frames": [
      {
        "frame_id": "initial",
        "frame_index": 0,
        "source": "video_boundary"
      },
      {
        "frame_id": "success_before",
        "frame_index": 75,
        "source": "official_success_event"
      },
      {
        "frame_id": "success_after",
        "frame_index": 76,
        "source": "official_success_event"
      },
      {
        "frame_id": "final",
        "frame_index": 77,
        "source": "video_boundary"
      }
    ],
    "artifacts": {
      "result": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/execution_vqa/execution_vqa.json",
      "prompt": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/execution_vqa/execution_vqa_prompt.md",
      "response": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/execution_vqa/execution_vqa_response.txt",
      "montage": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/execution_vqa/execution_montage.png",
      "selection": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/execution_vqa/keyframe_selection.json",
      "query": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/execution_vqa/execution_vqa_query.json"
    },
    "reason": null,
    "query": {
      "schema_version": 1,
      "profile": "dynamic_v1",
      "task_name": "click_bell",
      "template_id": "task_execution.official_baseline",
      "sub_aspect": "task_execution.official_baseline",
      "tool_metric": "official_check_success",
      "phenomenon_ids": [
        "bell_visibly_pressed"
      ],
      "questions": [
        {
          "id": "bell_visibly_pressed",
          "question_type": "visible_state_change",
          "target_role": "task_target",
          "question": "Does the robot visibly press or actuate the target bell?",
          "visual_scope": "rollout_change",
          "numeric_authority": "official_core_predicate_is_authoritative_when_available_else_official_check_success"
        }
      ],
      "selection_reasons": [
        "capability_adapter:click_bell:task_execution.official_baseline",
        "task_metric:click_bell:official_check_success"
      ],
      "answer_contract": {
        "required_response_keys": [
          "phenomena",
          "confidence",
          "frame_ids",
          "numeric_consistency",
          "conflicts"
        ],
        "phenomenon_item_keys": [
          "id",
          "observed",
          "description",
          "confidence",
          "frame_ids"
        ],
        "observed_type": "boolean_or_null",
        "numeric_consistency_values": [
          "consistent",
          "conflict",
          "uncertain"
        ]
      }
    }
  },
  {
    "round_id": "round_2",
    "status": "passed",
    "model_requested": "gpt-4o-2024-11-20",
    "representative_episode": "mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_2/evaluation/telemetry/act/episode_000_seed_100000",
    "evidence_conflict": false,
    "observation": {
      "phenomena": [
        {
          "id": "bell_visibly_pressed",
          "observed": true,
          "description": "The robot visibly presses the bell in the rollout.",
          "confidence": 1.0,
          "frame_ids": [
            "success_before",
            "success_after"
          ]
        }
      ],
      "confidence": 1.0,
      "frame_ids": [
        "success_before",
        "success_after"
      ],
      "numeric_consistency": "consistent",
      "conflicts": [],
      "evidence_conflict": false
    },
    "selected_frames": [
      {
        "frame_id": "initial",
        "frame_index": 0,
        "source": "video_boundary"
      },
      {
        "frame_id": "success_before",
        "frame_index": 73,
        "source": "official_success_event"
      },
      {
        "frame_id": "success_after",
        "frame_index": 74,
        "source": "official_success_event"
      },
      {
        "frame_id": "final",
        "frame_index": 75,
        "source": "video_boundary"
      }
    ],
    "artifacts": {
      "result": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/execution_vqa/execution_vqa.json",
      "prompt": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/execution_vqa/execution_vqa_prompt.md",
      "response": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/execution_vqa/execution_vqa_response.txt",
      "montage": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/execution_vqa/execution_montage.png",
      "selection": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/execution_vqa/keyframe_selection.json",
      "query": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/execution_vqa/execution_vqa_query.json"
    },
    "reason": null,
    "query": {
      "schema_version": 1,
      "profile": "dynamic_v1",
      "task_name": "click_bell",
      "template_id": null,
      "sub_aspect": "task_execution.object_position_variation",
      "tool_metric": "query_derived_metric",
      "phenomenon_ids": [
        "bell_visibly_pressed"
      ],
      "questions": [
        {
          "id": "bell_visibly_pressed",
          "question_type": "visible_state_change",
          "target_role": "task_target",
          "question": "Does the robot visibly press or actuate the target bell?",
          "visual_scope": "rollout_change",
          "numeric_authority": "official_core_predicate_is_authoritative_when_available_else_official_check_success"
        }
      ],
      "selection_reasons": [
        "task_owned_fallback:click_bell"
      ],
      "answer_contract": {
        "required_response_keys": [
          "phenomena",
          "confidence",
          "frame_ids",
          "numeric_consistency",
          "conflicts"
        ],
        "phenomenon_item_keys": [
          "id",
          "observed",
          "description",
          "confidence",
          "frame_ids"
        ],
        "observed_type": "boolean_or_null",
        "numeric_consistency_values": [
          "consistent",
          "conflict",
          "uncertain"
        ]
      }
    }
  },
  {
    "round_id": "round_3",
    "status": "passed",
    "model_requested": "gpt-4o-2024-11-20",
    "representative_episode": "mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_3/evaluation/telemetry/act/episode_000_seed_100000",
    "evidence_conflict": false,
    "observation": {
      "phenomena": [
        {
          "id": "bell_visibly_pressed",
          "observed": false,
          "description": "The robot does not visibly press or actuate the target bell.",
          "confidence": 0.95,
          "frame_ids": [
            "initial",
            "context_1",
            "context_2",
            "final"
          ]
        }
      ],
      "confidence": 0.95,
      "frame_ids": [
        "initial",
        "context_1",
        "context_2",
        "final"
      ],
      "numeric_consistency": "consistent",
      "conflicts": [],
      "evidence_conflict": false
    },
    "selected_frames": [
      {
        "frame_id": "initial",
        "frame_index": 0,
        "source": "video_boundary"
      },
      {
        "frame_id": "context_1",
        "frame_index": 133,
        "source": "uniform_fallback"
      },
      {
        "frame_id": "context_2",
        "frame_index": 266,
        "source": "uniform_fallback"
      },
      {
        "frame_id": "final",
        "frame_index": 399,
        "source": "video_boundary"
      }
    ],
    "artifacts": {
      "result": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/execution_vqa/execution_vqa.json",
      "prompt": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/execution_vqa/execution_vqa_prompt.md",
      "response": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/execution_vqa/execution_vqa_response.txt",
      "montage": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/execution_vqa/execution_montage.png",
      "selection": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/execution_vqa/keyframe_selection.json",
      "query": "mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/execution_vqa/execution_vqa_query.json"
    },
    "reason": null,
    "query": {
      "schema_version": 1,
      "profile": "dynamic_v1",
      "task_name": "click_bell",
      "template_id": null,
      "sub_aspect": "task_execution.object_instance_variation",
      "tool_metric": "query_derived_metric",
      "phenomenon_ids": [
        "bell_visibly_pressed"
      ],
      "questions": [
        {
          "id": "bell_visibly_pressed",
          "question_type": "visible_state_change",
          "target_role": "task_target",
          "question": "Does the robot visibly press or actuate the target bell?",
          "visual_scope": "rollout_change",
          "numeric_authority": "official_core_predicate_is_authoritative_when_available_else_official_check_success"
        }
      ],
      "selection_reasons": [
        "task_owned_fallback:click_bell"
      ],
      "answer_contract": {
        "required_response_keys": [
          "phenomena",
          "confidence",
          "frame_ids",
          "numeric_consistency",
          "conflicts"
        ],
        "phenomenon_item_keys": [
          "id",
          "observed",
          "description",
          "confidence",
          "frame_ids"
        ],
        "observed_type": "boolean_or_null",
        "numeric_consistency_values": [
          "consistent",
          "conflict",
          "uncertain"
        ]
      }
    }
  }
]
```

## Feedback Agent answer

ACT策略在测试对象实例变化时首次暴露弱点，未能完成任务。

### Findings

- 在对象位置变化测试中，ACT策略成功完成任务，成功率为1.0。
- 在对象实例变化测试中，ACT策略未能完成任务，成功率为0.0。
- 未发现证据冲突，所有测试均满足有限域证据充分性合同。

### Limitations

- 测试仅包含3个episode，无法提供统计泛化保证。
- 测试范围局限于指定的任务、策略检查点和种子。
- 有限域停止原型不支持最坏情况或无反例结论。
- Evidence contains N=3 policy episodes at seeds [100000].
- The run stopped because the finite query-sufficiency contract was satisfied; this is not a statistical generalization guarantee.

### Recommended next step

建议进一步测试其他对象属性变化（如纹理或背景干扰），以全面评估策略的鲁棒性。

## Artifact index

- evaluation plan: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/plan/evaluation_plan.json`
- history retrieval: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/plan/history_retrieval.json`
- Plan decision: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/plan/decision_after_round_1.json`
- Plan decision: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/plan/decision_after_round_2.json`
- Plan decision: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/plan/decision_after_round_3.json`
- Evidence assessment: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/plan/evidence_after_round_1.json`
- Evidence assessment: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/plan/evidence_after_round_2.json`
- Evidence assessment: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/plan/evidence_after_round_3.json`
- machine-readable summary: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/summary/summary.json`
- deterministic aggregate: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/summary/aggregate_result.json`
- round 1 `generated_task`: `envs/click_bell.py`
- round 1 `scene_image`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_1/evidence/initial_head.png`
- round 1 `vision_result`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_1/validation/vision.json`
- round 1 `position_samples`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_1/validation/position_samples.json`
- round 1 `reflection_summary`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_1/reflection/summary.json`
- round 1 `act_videos`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_1/evaluation/episode0.mp4`
- round 1 `rollout_videos`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_1/evaluation/telemetry/act/episode_000_seed_100000/video.mp4`
- round 1 `act_result`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_1/evaluation/_result.txt`
- round 1 `trusted_tools`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_1/evaluation/telemetry/tool_results.json`
- round 1 `planned_tool`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/planned_tool/tool_execution.json`
- round 1 `aggregate`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/aggregate_result.json`
- round 1 `evidence_aggregate`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/evidence_aggregate.json`
- round 1 `execution_vqa`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/execution_vqa/execution_vqa.json`
- round 1 `execution_vqa_query`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/execution_vqa_query.json`
- round 1 `execution_vqa_montage`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/execution_vqa/execution_montage.png`
- round 1 `execution_vqa_selection`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_1/execution_vqa/keyframe_selection.json`
- round 1 `child_manifest`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_1/manifest.json`
- round 2 `generated_task`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_2/task.py`
- round 2 `scene_image`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_2/evidence/initial_head.png`
- round 2 `vision_result`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_2/validation/vision.json`
- round 2 `position_samples`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_2/validation/position_samples.json`
- round 2 `reflection_summary`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_2/reflection/summary.json`
- round 2 `act_videos`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_2/evaluation/episode0.mp4`
- round 2 `rollout_videos`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_2/evaluation/telemetry/act/episode_000_seed_100000/video.mp4`
- round 2 `act_result`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_2/evaluation/_result.txt`
- round 2 `trusted_tools`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_2/evaluation/telemetry/tool_results.json`
- round 2 `planned_tool`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/planned_tool/tool_execution.json`
- round 2 `aggregate`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/aggregate_result.json`
- round 2 `evidence_aggregate`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/evidence_aggregate.json`
- round 2 `method_runtime`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/method_runtime_projection.json`
- round 2 `execution_vqa`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/execution_vqa/execution_vqa.json`
- round 2 `execution_vqa_query`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/execution_vqa_query.json`
- round 2 `execution_vqa_montage`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/execution_vqa/execution_montage.png`
- round 2 `execution_vqa_selection`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_2/execution_vqa/keyframe_selection.json`
- round 2 `child_manifest`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_2/manifest.json`
- round 3 `generated_task`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_3/task.py`
- round 3 `scene_image`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_3/evidence/initial_head.png`
- round 3 `vision_result`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_3/validation/vision.json`
- round 3 `position_samples`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_3/validation/position_samples.json`
- round 3 `reflection_summary`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_3/reflection/summary.json`
- round 3 `act_videos`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_3/evaluation/episode0.mp4`
- round 3 `rollout_videos`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_3/evaluation/telemetry/act/episode_000_seed_100000/video.mp4`
- round 3 `act_result`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_3/evaluation/_result.txt`
- round 3 `trusted_tools`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_3/evaluation/telemetry/tool_results.json`
- round 3 `planned_tool`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/planned_tool/tool_execution.json`
- round 3 `aggregate`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/aggregate_result.json`
- round 3 `evidence_aggregate`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/evidence_aggregate.json`
- round 3 `method_runtime`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/method_runtime_projection.json`
- round 3 `execution_vqa`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/execution_vqa/execution_vqa.json`
- round 3 `execution_vqa_query`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/execution_vqa_query.json`
- round 3 `execution_vqa_montage`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/execution_vqa/execution_montage.png`
- round 3 `execution_vqa_selection`: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/execution/round_3/execution_vqa/keyframe_selection.json`
- round 3 `child_manifest`: `mea/generated_tasks/run_20260729_b30_refinement_live_v2_round_3/manifest.json`
