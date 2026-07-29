# MEA method evidence: eval_20260729_b30_refinement_live_v2

> This is a compact view of real run artifacts. The complete machine audit remains in the evaluation directory.

## 1. Query and fixed policy scope

> Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?

```json
{
  "binding_mode": "single_task_single_checkpoint_open_world",
  "task_name": "click_bell",
  "task_profile": null,
  "policy": {
    "name": "ACT",
    "checkpoint_setting": "demo_clean",
    "expert_data_num": 50,
    "language_conditioned": false
  },
  "checkpoint": {
    "policy_name": "ACT",
    "checkpoint_setting": "demo_clean",
    "expert_data_num": 50,
    "checkpoint_id": "act-click_bell/demo_clean-50",
    "ready": true
  },
  "round_budget": 3,
  "episodes_per_round": [
    1,
    1,
    1
  ]
}
```

One evaluation keeps this task and ACT checkpoint fixed. Adaptation happens only across this task's sub-aspects/variants.

## 2. Paper-level data flow

```mermaid
flowchart LR
  Q["Open Query"] --> P["Plan Agent / sub-aspect"]
  P --> T["TaskGen: reuse or generate"]
  T --> I["Render / visual reflection"]
  I --> E["ACT rollout"]
  E --> V["Rule Tool + dynamic VQA"]
  V --> A["Aggregate"]
  A -->|"evidence"| P
  A --> R["Final answer"]
```

## 3. Initial decomposition

```json
{
  "evaluation_goal": "answer_open_query_with_evidence: Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?",
  "selected_aspect_ids": null,
  "requested_template_ids": [
    "task_execution.official_baseline"
  ],
  "first_round": "round_1",
  "planning_state": "stopped_after_round_3_evidence_sufficient"
}
```

## 4.1. round_1: task_execution.official_baseline

### Compatibility task projection (not used for planning)

```json
{
  "proposal_status": "not_projected_in_compatibility_view",
  "task_name": "click_bell",
  "aspect_id": "task_execution.official_baseline",
  "task_instruction": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?"
}
```

### TaskGen output

- Route: `official`
- Materialization: `official_passthrough`
- Child run: `run_20260729_b30_refinement_live_v2_round_1`
- Full task artifact: [round_1_overlay.yml](code/round_1_overlay.yml)

```yaml
{}
```
- VariantSpec: [round_1_variant_spec.json](data/round_1_variant_spec.json)

### Render / scene check

![round_1 initial scene](assets/round_1_scene.png)

### ACT rollout

```json
{
  "backend": "ACT",
  "seeds": [
    100000
  ],
  "pipeline_passed": true,
  "policy_success": 1.0
}
```

[Open ACT video](assets/round_1_act.mp4)

<video src="assets/round_1_act.mp4" controls width="720"></video>

### Compatibility Tool projection (not used for planning)

```json
{
  "proposal_status": "not_projected_in_compatibility_view",
  "tool_request": {
    "schema_version": 1,
    "task_name": "click_bell",
    "metric": "official_check_success",
    "question": "Did the rollout satisfy the official RoboTwin success check?"
  }
}
```

```json
{
  "route": "reuse",
  "metric": "official_check_success",
  "episodes": [
    {
      "role": "policy_under_evaluation",
      "policy_name": "ACT",
      "seed": 100000,
      "value": true,
      "unit": null,
      "passed": true
    }
  ]
}
```

### Dynamic VQA

```json
{
  "status": "passed",
  "questions": [
    {
      "id": "bell_visibly_pressed",
      "question": "Does the robot visibly press or actuate the target bell?"
    }
  ],
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
  "numeric_consistency": "consistent",
  "evidence_conflict": false
}
```

![round_1 VQA keyframes](assets/round_1_vqa_montage.png)

### Aggregate -> next decision

```json
{
  "aggregate_status": "passed",
  "policy_success": 1.0,
  "decision": {
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
  }
}
```

## 4.2. round_2: task_execution.object_position_variation

### Compatibility task projection (not used for planning)

```json
{
  "proposal_status": "not_projected_in_compatibility_view",
  "task_name": "click_bell",
  "aspect_id": "task_execution.object_position_variation",
  "task_instruction": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?\nScene need: Introduce a bounded variation in the bell's position. Preserve unchanged: task identity; policy checkpoint.\nChecker need: reuse the official implementation"
}
```

### TaskGen output

- Route: `generic_provider_scene_checker_codegen`
- Materialization: `generic_provider_scene_checker_codegen`
- Child run: `run_20260729_b30_refinement_live_v2_round_2`
- Full task artifact: [round_2_task.py](code/round_2_task.py)

```python
"""Provider-generated RoboTwin task candidate."""

import envs.click_bell as _official_task_module
from envs.click_bell import *


class click_bell(_official_task_module.click_bell):
    def load_actors(self):
            rand_pos = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
            )
            while abs(rand_pos.p[0]) < 0.05:
                rand_pos = rand_pose(
                    xlim=[-0.25, 0.25],
                    ylim=[-0.2, 0.0],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                )

            # Introduce bounded variation in the bell's position
            perturbation = np.random.uniform(-0.02, 0.02, size=3)
            perturbed_position = rand_pos.p + perturbation
            rand_pos = sapien.Pose(p=perturbed_position, q=rand_pos.q)

            self.bell_id = np.random.choice([0, 1], 1)[0]
            self.bell = create_actor(
                scene=self,
                pose=rand_pos,
                modelname="050_bell",
                convex=True,
                model_id=self.bell_id,
                is_static=True,
            )

            self.add_prohibit_area(self.bell, padding=0.07)
            self.check_arm_function = self.is_left_gripper_close if self.bell.get_pose().p[0] < 0 else self.is_right_gripper_close

    def mea_official_check_success(self):
        """Evaluate the untouched official core predicate."""
        return _official_task_module.click_bell.check_success(self)
```

### Render / scene check

![round_2 initial scene](assets/round_2_scene.png)

### ACT rollout

```json
{
  "backend": "ACT",
  "seeds": [
    100000
  ],
  "pipeline_passed": true,
  "policy_success": 1.0
}
```

[Open ACT video](assets/round_2_act.mp4)

<video src="assets/round_2_act.mp4" controls width="720"></video>

### Compatibility Tool projection (not used for planning)

```json
{
  "proposal_status": "not_projected_in_compatibility_view",
  "tool_request": {
    "schema_version": 2,
    "task_name": "click_bell",
    "metric": "query_derived_metric",
    "question": "Where does the ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?",
    "metric_spec": {
      "schema_version": 1,
      "operation": "minimum_distance",
      "left_signal": "right_tcp_position",
      "right_signal": "bell_contact_position",
      "dimensions": [
        "x",
        "y"
      ],
      "unit": "m",
      "null_semantics": "null_if_no_finite_sample"
    }
  }
}
```

```json
{
  "route": "typed_metric_spec_compile",
  "metric": "query_derived_metric",
  "episodes": [
    {
      "role": "policy_under_evaluation",
      "policy_name": "ACT",
      "seed": 100000,
      "value": 0.021466496280102353,
      "unit": "m",
      "passed": null
    }
  ]
}
```

[Open generated/reused Tool source](code/round_2_tool.py)

```python
def generated_tool(trajectory):
    left = np.asarray(trajectory.trace['right_tcp_position'], dtype=float)
    right = np.asarray(trajectory.trace['bell_contact_position'], dtype=float)
    left_view = left[:, [0, 1]]
    right_view = right[:, [0, 1]]
    valid = np.all(np.isfinite(left_view) & np.isfinite(right_view), axis=1)
    distances = np.linalg.norm(left_view - right_view, axis=1)
    masked = np.where(valid, distances, np.inf)
    index = int(np.argmin(masked))
    value = float(masked[index])
    if not np.isfinite(value):
        return {
            "value": None,
            "unit": 'm',
            "passed": None,
            "evidence_steps": [],
            "details": {
                "operation": 'minimum_distance',
                "left_signal": 'right_tcp_position',
                "right_signal": 'bell_contact_position',
                "dimensions": ['x', 'y'],
                "min_index": None,
                "reason": "no_finite_sample",
            },
        }
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    step = int(physics[index])
    return {
        "value": value,
        "unit": 'm',
        "passed": None,
        "evidence_steps": [step],
        "details": {
            "operation": 'minimum_distance',
            "left_signal": 'right_tcp_position',
            "right_signal": 'bell_contact_position',
            "dimensions": ['x', 'y'],
            "min_index": index,
            "reason": "measured",
        },
    }
```

### Dynamic VQA

```json
{
  "status": "passed",
  "questions": [
    {
      "id": "bell_visibly_pressed",
      "question": "Does the robot visibly press or actuate the target bell?"
    }
  ],
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
  "numeric_consistency": "consistent",
  "evidence_conflict": false
}
```

![round_2 VQA keyframes](assets/round_2_vqa_montage.png)

### Aggregate -> next decision

```json
{
  "aggregate_status": "passed",
  "policy_success": 1.0,
  "decision": {
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
  }
}
```

## 4.3. round_3: task_execution.object_instance_variation

### Compatibility task projection (not used for planning)

```json
{
  "proposal_status": "not_projected_in_compatibility_view",
  "task_name": "click_bell",
  "aspect_id": "task_execution.object_instance_variation",
  "task_instruction": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?\nScene need: Introduce a bounded variation in the bell's object instance. Preserve unchanged: task identity; policy checkpoint.\nChecker need: reuse the official implementation"
}
```

### TaskGen output

- Route: `generic_provider_scene_checker_codegen`
- Materialization: `generic_provider_scene_checker_codegen`
- Child run: `run_20260729_b30_refinement_live_v2_round_3`
- Full task artifact: [round_3_task.py](code/round_3_task.py)

```python
"""Provider-generated RoboTwin task candidate."""

import envs.click_bell as _official_task_module
from envs.click_bell import *


class click_bell(_official_task_module.click_bell):
    def load_actors(self):
            rand_pos = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
            )
            while abs(rand_pos.p[0]) < 0.05:
                rand_pos = rand_pose(
                    xlim=[-0.25, 0.25],
                    ylim=[-0.2, 0.0],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                )

            # Apply a bounded variation to the bell's scale
            scale_multiplier = 1.2  # Increase size by 20%

            self.bell_id = np.random.choice([0, 1], 1)[0]
            self.bell = create_actor(
                scene=self,
                pose=rand_pos,
                modelname="050_bell",
                convex=True,
                model_id=self.bell_id,
                is_static=True,
                scale_multiplier=scale_multiplier,
            )

            self.add_prohibit_area(self.bell, padding=0.07)
            self.check_arm_function = self.is_left_gripper_close if self.bell.get_pose().p[0] < 0 else self.is_right_gripper_close

    def mea_official_check_success(self):
        """Evaluate the untouched official core predicate."""
        return _official_task_module.click_bell.check_success(self)
```

### Render / scene check

![round_3 initial scene](assets/round_3_scene.png)

### ACT rollout

```json
{
  "backend": "ACT",
  "seeds": [
    100000
  ],
  "pipeline_passed": true,
  "policy_success": 0.0
}
```

[Open ACT video](assets/round_3_act.mp4)

<video src="assets/round_3_act.mp4" controls width="720"></video>

### Compatibility Tool projection (not used for planning)

```json
{
  "proposal_status": "not_projected_in_compatibility_view",
  "tool_request": {
    "schema_version": 2,
    "task_name": "click_bell",
    "metric": "query_derived_metric",
    "question": "Reuse the validated query_derived_metric measurement.",
    "metric_spec": {
      "schema_version": 1,
      "operation": "minimum_distance",
      "left_signal": "right_tcp_position",
      "right_signal": "bell_contact_position",
      "dimensions": [
        "x",
        "y"
      ],
      "unit": "m",
      "null_semantics": "null_if_no_finite_sample"
    }
  }
}
```

```json
{
  "route": "run_local_reuse",
  "metric": "query_derived_metric",
  "episodes": [
    {
      "role": "policy_under_evaluation",
      "policy_name": "ACT",
      "seed": 100000,
      "value": 0.0027531830083962435,
      "unit": "m",
      "passed": null
    }
  ]
}
```

### Dynamic VQA

```json
{
  "status": "passed",
  "questions": [
    {
      "id": "bell_visibly_pressed",
      "question": "Does the robot visibly press or actuate the target bell?"
    }
  ],
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
  "numeric_consistency": "consistent",
  "evidence_conflict": false
}
```

![round_3 VQA keyframes](assets/round_3_vqa_montage.png)

### Aggregate -> next decision

```json
{
  "aggregate_status": "passed",
  "policy_success": 0.0,
  "decision": {
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
}
```

## 5. Final answer to the original Query

> ACT策略在测试对象实例变化时首次暴露弱点，未能完成任务。

```json
{
  "findings": [
    "在对象位置变化测试中，ACT策略成功完成任务，成功率为1.0。",
    "在对象实例变化测试中，ACT策略未能完成任务，成功率为0.0。",
    "未发现证据冲突，所有测试均满足有限域证据充分性合同。"
  ],
  "recommended_next_step": "建议进一步测试其他对象属性变化（如纹理或背景干扰），以全面评估策略的鲁棒性。",
  "limitations": [
    "测试仅包含3个episode，无法提供统计泛化保证。",
    "测试范围局限于指定的任务、策略检查点和种子。",
    "有限域停止原型不支持最坏情况或无反例结论。",
    "Evidence contains N=3 policy episodes at seeds [100000].",
    "The run stopped because the finite query-sufficiency contract was satisfied; this is not a statistical generalization guarantee."
  ]
}
```

## 6. Boundaries

- Policy results and pipeline status are reported separately.
- Expert evidence, when present, is a solvability/instrumentation gate, not ACT performance.
- Few-shot N=1 rounds demonstrate method wiring, not benchmark-level generalization.
- Missing artifacts are shown as N/A; this report never substitutes proxy images or invented values.

## 7. Raw artifact index

- Server source: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/manifest.json`
- Server source: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/plan/evaluation_plan.json`
- Server source: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/plan/bound_task_session.json`
- Server source: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/summary/evidence_bundle.json`
- Server source: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/feedback/feedback.json`
- Server source: `mea/evaluation_runs/eval_20260729_b30_refinement_live_v2/evaluation_report.md`
