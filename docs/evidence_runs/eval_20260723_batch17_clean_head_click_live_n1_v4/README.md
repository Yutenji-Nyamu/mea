# MEA method evidence: eval_20260723_batch17_clean_head_click_live_n1_v4

> This is a compact view of real run artifacts. The complete machine audit remains in the evaluation directory.

## 1. Query and fixed policy scope

> How well does the click_bell ACT policy generalize across properties of the operated bell?

```json
{
  "binding_mode": "single_task_single_checkpoint",
  "task_name": "click_bell",
  "task_profile": "adaptive_properties",
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
  "round_budget": 2,
  "episodes_per_round": [
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
  "evaluation_goal": "evaluate generalization across supported properties of the operated bell",
  "selected_aspect_ids": [
    "object_position",
    "object_instance"
  ],
  "requested_template_ids": [
    "object_position.left_fixed",
    "object_position.right_fixed",
    "object_instance.base0",
    "object_instance.base1"
  ],
  "first_round": "round_1",
  "planning_state": "stopped_after_round_2_by_hard_cap"
}
```

## 4.1. round_1: object_position

### Plan -> TaskProposal

```json
{
  "schema_version": 1,
  "proposal_id": "object_position.query_generated_1",
  "task_name": "click_bell",
  "aspect_id": "object_position",
  "intent": "evaluate a query-relevant bounded variation",
  "capability_id": "object_position.fixed_xy",
  "reuse_first": true,
  "changes": {
    "bell": {
      "position_mode": "fixed",
      "xy": [
        -0.14,
        -0.12
      ]
    }
  },
  "preserve_success_semantics": true
}
```

### TaskGen output

- Route: `reuse`
- Materialization: `bounded_variant_overlay`
- Child run: `run_20260723_batch17_clean_head_click_live_n1_v4_round_1`
- Full task artifact: [round_1_overlay.yml](code/round_1_overlay.yml)

```yaml
mea:
  enabled: true
  bell:
    position_mode: fixed
    xy:
    - -0.14
    - -0.12
```
- VariantSpec: [round_1_variant_spec.json](data/round_1_variant_spec.json)

### Render / scene check

![round_1 initial scene](assets/round_1_scene.png)

### ACT rollout

```json
{
  "backend": "ACT",
  "seeds": [
    100502
  ],
  "pipeline_passed": true,
  "policy_success": 1.0
}
```

[Open ACT video](assets/round_1_act.mp4)

<video src="assets/round_1_act.mp4" controls width="720"></video>

### ToolProposal -> ToolGen / reuse

```json
{
  "schema_version": 2,
  "proposal_id": "object_position.query_generated_1.tool",
  "task_name": "click_bell",
  "aspect_id": "object_position",
  "evaluation_goal": "measure task outcome and visible behavior",
  "metric": "bell_active_tcp_min_xy_error",
  "question": "What simulator measurement best diagnoses this aspect?",
  "vqa_phenomenon_ids": [
    "bell_visibly_pressed",
    "run_local.click_bell.object_position.query_observation"
  ],
  "vqa_question_specs": [
    {
      "question_type": "visible_state_change",
      "target_role": "task_target",
      "question": "Does the rollout visibly show the robot making task-relevant progress under the query-generated variation?",
      "visual_scope": "rollout_change",
      "numeric_authority": "official_check_success_is_authoritative",
      "id": "run_local.click_bell.object_position.query_observation"
    }
  ],
  "reuse_first": true
}
```

```json
{
  "route": "force_codegen",
  "metric": "bell_active_tcp_min_xy_error",
  "episodes": [
    {
      "role": "policy_under_evaluation",
      "policy_name": "ACT",
      "seed": 100502,
      "value": 0.009174784645438194,
      "unit": "m",
      "passed": null
    },
    {
      "role": "expert_validation",
      "policy_name": "expert",
      "seed": 100502,
      "value": 0.0005667416262440383,
      "unit": "m",
      "passed": null
    }
  ]
}
```

[Open generated/reused Tool source](code/round_1_tool.py)

```python
def generated_tool(trajectory):
    bell_position = trajectory.trace["bell_position"]
    bell_contact_position = trajectory.trace["bell_contact_position"]
    active_arm = (
        "left"
        if bell_position[0, 0] < 0
        else "right"
    )
    tcp_position = trajectory.trace[
        "left_tcp_position" if active_arm == "left" else "right_tcp_position"
    ]
    delta_xy = tcp_position[:, :2] - bell_contact_position[:, :2]
    d = np.sqrt(np.sum(delta_xy * delta_xy, axis=1))
    min_index = np.argmin(np.where(np.isfinite(d), d, np.inf))
    min_error = d[min_index]
    physics_step = trajectory.trace["physics_step"][min_index]
    simulation_time_seconds = trajectory.trace["simulation_time_seconds"][min_index]
    return {
        "value": float(min_error) if np.isfinite(min_error) else None,
        "unit": "m",
        "passed": None,
        "evidence_steps": [int(physics_step)] if np.isfinite(min_error) else [],
        "details": {
            "active_arm": active_arm,
            "min_error_physics_step": int(physics_step) if np.isfinite(min_error) else None,
            "simulation_time_seconds": (
                float(simulation_time_seconds) if np.isfinite(min_error) else None
            ),
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
    },
    {
      "id": "run_local.click_bell.object_position.query_observation",
      "question": "Does the rollout visibly show the robot making task-relevant progress under the query-generated variation?"
    }
  ],
  "phenomena": [
    {
      "id": "bell_visibly_pressed",
      "observed": true,
      "description": "The robot end effector is visibly positioned down over the bell in the success frames, consistent with a pressing action.",
      "confidence": 0.89,
      "frame_ids": [
        "success_before",
        "success_after",
        "final"
      ]
    },
    {
      "id": "run_local.click_bell.object_position.query_observation",
      "observed": true,
      "description": "The robot moves from away from the bell initially to directly over it in the success frames, showing task-relevant progress.",
      "confidence": 0.94,
      "frame_ids": [
        "initial",
        "success_before",
        "success_after",
        "final"
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
    "schema_version": 3,
    "action": "continue",
    "transition": "switch_aspect",
    "next_aspect_id": "object_instance",
    "next_template_id": "object_instance.base0",
    "observation_summary": "The object-position evaluation produced complete, valid Rule evidence (aggregate_status passed), a passed VQA result without evidence conflict, and a policy success rate of 1.0, so the observed result is reliable for that evaluated position condition. However, the original query concerns generalization across bell properties, and the initially required object-instance aspect remains uncovered. Evaluating an official alternate bell instance is therefore the next necessary step to assess instance-property generalization.",
    "decision_reason": "provider_authored_plan_step",
    "answered_query": false,
    "plan_step_source": "provider",
    "plan_step_proposal": {
      "schema_version": 1,
      "action": "propose",
      "aspect_id": "object_instance",
      "template_id": "object_instance.base0",
      "rationale": "The object-position evaluation produced complete, valid Rule evidence (aggregate_status passed), a passed VQA result without evidence conflict, and a policy success rate of 1.0, so the observed result is reliable for that evaluated position condition. However, the original query concerns generalization across bell properties, and the initially required object-instance aspect remains uncovered. Evaluating an official alternate bell instance is therefore the next necessary step to assess instance-property generalization.",
      "answered_query": false
    },
    "round_budget_before_decision": 1,
    "evidence_assessment": {
      "schema_version": 1,
      "task_name": "click_bell",
      "checkpoint_id": "act-click_bell/demo_clean-50",
      "current_aspect_id": "object_position",
      "round_budget_remaining": 1,
      "evidence_state": "sufficient",
      "evidence_packet": {
        "schema_version": 1,
        "round_id": "round_1",
        "template_id": "object_position.left_fixed",
        "pipeline": {
          "passed": true,
          "failure_stage": null
        },
        "policy": {
          "success_rate": 1.0,
          "reported": true
        },
        "rule": {
          "metric": "bell_active_tcp_min_xy_error",
          "expected_policy_episodes": 1,
          "aggregate_status": "passed",
          "input_issue_count": 0,
          "valid": 1,
          "missing": 0,
          "invalid": 0,
          "semantic_missing": 0,
          "semantic_missing_reasons": [],
          "observed_policy_episodes": 1,
          "complete": true,
          "reasons": []
        },
        "vqa": {
          "required": true,
          "status": "passed",
          "evidence_conflict": false
        },
        "evidence_strength": "sufficient",
        "reason_codes": []
      },
      "initial_required_aspect_ids": [
        "object_position",
        "object_instance"
      ],
      "covered_aspect_ids": [
        "object_position"
      ],
      "uncovered_initial_required_aspect_ids": [
        "object_instance"
      ],
      "discoverable_aspect_ids": [
        "performance.completion_time_stability",
        "robustness.scene_clutter",
        "scene_background_texture",
        "scene_lighting"
      ],
      "available_steps": {
        "refine": [
          {
            "aspect_id": "object_position",
            "template_ids": [
              "object_position.right_fixed"
            ]
          }
        ],
        "propose": [
          {
            "aspect_id": "object_instance",
            "template_ids": [
              "object_instance.base0",
              "object_instance.base1"
            ],
            "initially_required": true
          },
          {
            "aspect_id": "performance.completion_time_stability",
            "template_ids": [
              "performance.completion_time_stability.official"
            ],
            "initially_required": false
          },
          {
            "aspect_id": "robustness.scene_clutter",
            "template_ids": [
              "robustness.scene_clutter.official_table"
            ],
            "initially_required": false
          },
          {
            "aspect_id": "scene_background_texture",
            "template_ids": [
              "scene_background_texture.unseen"
            ],
            "initially_required": false
          },
          {
            "aspect_id": "scene_lighting",
            "template_ids": [
              "scene_lighting.static_random"
            ],
            "initially_required": false
          }
        ],
        "stop": false
      },
      "stop_requires_answered_query": false,
      "forced_stop": false,
      "fallback_step": {
        "schema_version": 1,
        "action": "propose",
        "aspect_id": "object_instance",
        "template_id": "object_instance.base0",
        "rationale": "Deterministic fallback selected the first legal evidence-conditioned step.",
        "answered_query": false
      }
    },
    "next_round": {
      "round_id": "round_2",
      "template_id": "object_instance.base0",
      "capability_id": "object_instance.official_id",
      "task_variant_id": "object_instance.query_generated_1",
      "capability_contract": {
        "schema_version": 1,
        "task_name": "click_bell",
        "template_id": "object_instance.base0",
        "aspect": {
          "aspect_id": "object_instance",
          "semantic_scope": "object",
          "target_role": "task_target"
        },
        "taskgen": {
          "operation": "bounded_variant_overlay",
          "capability_id": "object_instance.official_id",
          "task_variant_id": "object_instance.base0",
          "controlled_axis": "object_instance",
          "change_scope": "object",
          "generation_mode": "bounded_variant_overlay",
          "allowed_change_roots": [
            "bell"
          ],
          "changes": {
            "bell": {
              "position_mode": "official_random",
              "instance_mode": "fixed",
              "bell_id": 0
            }
          }
        },
        "tool": {
          "request_factory_id": "official_success_tool_request",
          "metric": "official_check_success"
        },
        "vqa": {
          "phenomenon_ids": [
            "bell_visibly_pressed"
          ]
        },
        "required_gates": [
          "variant_spec",
          "render",
          "rule",
          "scene_variant",
          "vision",
          "expert",
          "act",
          "toolkit",
          "planned_tool",
          "aggregate",
          "execution_vqa"
        ]
      },
      "sub_aspect": "object_instance",
      "aspect_id": "object_instance",
      "probe_role": "sentinel",
      "rationale": "Official larger white/black base0 bell instance.",
      "task_instruction": "How well does the click_bell ACT policy generalize across properties of the operated bell? Trusted bounded variant: Official larger white/black base0 bell instance. Query-generated bounded variation: evaluate a query-relevant bounded variation",
      "task_name": "click_bell",
      "task_module": "mea.tasks.click_bell",
      "telemetry_profile": "balanced_v1",
      "route": "reuse",
      "variant_hint": {
        "bell": {
          "position_mode": "official_random",
          "instance_mode": "fixed",
          "bell_id": 0
        }
      },
      "execution": {
        "backend": "act",
        "seeds": [
          100502
        ],
        "num_episodes": 1,
        "gates": [
          "variant_spec",
          "render",
          "rule",
          "scene_variant",
          "vision",
          "expert",
          "act",
          "toolkit",
          "planned_tool",
          "aggregate",
          "execution_vqa"
        ]
      },
      "observations": [
        "scene_alignment",
        "bell_position",
        "bell_instance_id",
        "scene_clutter",
        "scene_background_texture",
        "scene_lighting",
        "expert_solvable",
        "policy_success",
        "trusted_tools",
        "completion_time_statistics",
        "execution_vqa"
      ],
      "tool_request": {
        "schema_version": 1,
        "task_name": "click_bell",
        "metric": "official_check_success",
        "question": "What simulator measurement best diagnoses this aspect?"
      },
      "vqa_phenomenon_ids": [
        "bell_visibly_pressed",
        "run_local.click_bell.object_instance.query_observation"
      ],
      "task_proposal": {
        "schema_version": 1,
        "proposal_id": "object_instance.query_generated_1",
        "task_name": "click_bell",
        "aspect_id": "object_instance",
        "intent": "evaluate a query-relevant bounded variation",
        "capability_id": "object_instance.official_id",
        "reuse_first": true,
        "changes": {
          "bell": {
            "position_mode": "official_random",
            "instance_mode": "fixed",
            "bell_id": 0
          }
        },
        "preserve_success_semantics": true
      },
      "tool_proposal": {
        "schema_version": 2,
        "proposal_id": "object_instance.query_generated_1.tool",
        "task_name": "click_bell",
        "aspect_id": "object_instance",
        "evaluation_goal": "measure task outcome and visible behavior",
        "metric": "official_check_success",
        "question": "What simulator measurement best diagnoses this aspect?",
        "vqa_phenomenon_ids": [
          "bell_visibly_pressed",
          "run_local.click_bell.object_instance.query_observation"
        ],
        "vqa_question_specs": [
          {
            "question_type": "visible_state_change",
            "target_role": "task_target",
            "question": "Does the rollout visibly show the robot making task-relevant progress under the query-generated variation?",
            "visual_scope": "rollout_change",
            "numeric_authority": "official_check_success_is_authoritative",
            "id": "run_local.click_bell.object_instance.query_observation"
          }
        ],
        "reuse_first": true
      },
      "proposal_materialization": {
        "schema_version": 1,
        "mode": "query_generated_bounded_variation",
        "base_template_id": "object_instance.base0",
        "capability_contract_is_authority_envelope": true,
        "task_proposal_is_round_variation_authority": true
      }
    }
  }
}
```

## 4.2. round_2: object_instance

### Plan -> TaskProposal

```json
{
  "schema_version": 1,
  "proposal_id": "object_instance.query_generated_1",
  "task_name": "click_bell",
  "aspect_id": "object_instance",
  "intent": "evaluate a query-relevant bounded variation",
  "capability_id": "object_instance.official_id",
  "reuse_first": true,
  "changes": {
    "bell": {
      "position_mode": "official_random",
      "instance_mode": "fixed",
      "bell_id": 0
    }
  },
  "preserve_success_semantics": true
}
```

### TaskGen output

- Route: `reuse`
- Materialization: `bounded_variant_overlay`
- Child run: `run_20260723_batch17_clean_head_click_live_n1_v4_round_2`
- Full task artifact: [round_2_overlay.yml](code/round_2_overlay.yml)

```yaml
mea:
  enabled: true
  bell:
    position_mode: official_random
    instance_mode: fixed
    bell_id: 0
```
- VariantSpec: [round_2_variant_spec.json](data/round_2_variant_spec.json)

### Render / scene check

![round_2 initial scene](assets/round_2_scene.png)

### ACT rollout

```json
{
  "backend": "ACT",
  "seeds": [
    100502
  ],
  "pipeline_passed": true,
  "policy_success": 1.0
}
```

[Open ACT video](assets/round_2_act.mp4)

<video src="assets/round_2_act.mp4" controls width="720"></video>

### ToolProposal -> ToolGen / reuse

```json
{
  "schema_version": 2,
  "proposal_id": "object_instance.query_generated_1.tool",
  "task_name": "click_bell",
  "aspect_id": "object_instance",
  "evaluation_goal": "measure task outcome and visible behavior",
  "metric": "official_check_success",
  "question": "What simulator measurement best diagnoses this aspect?",
  "vqa_phenomenon_ids": [
    "bell_visibly_pressed",
    "run_local.click_bell.object_instance.query_observation"
  ],
  "vqa_question_specs": [
    {
      "question_type": "visible_state_change",
      "target_role": "task_target",
      "question": "Does the rollout visibly show the robot making task-relevant progress under the query-generated variation?",
      "visual_scope": "rollout_change",
      "numeric_authority": "official_check_success_is_authoritative",
      "id": "run_local.click_bell.object_instance.query_observation"
    }
  ],
  "reuse_first": true
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
      "seed": 100502,
      "value": true,
      "unit": null,
      "passed": true
    },
    {
      "role": "expert_validation",
      "policy_name": "expert",
      "seed": 100502,
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
    },
    {
      "id": "run_local.click_bell.object_instance.query_observation",
      "question": "Does the rollout visibly show the robot making task-relevant progress under the query-generated variation?"
    }
  ],
  "phenomena": [
    {
      "id": "bell_visibly_pressed",
      "observed": true,
      "description": "The robot end effector moves onto the bell and visibly depresses/actuates its central button by success_after.",
      "confidence": 0.9,
      "frame_ids": [
        "initial",
        "success_before",
        "success_after"
      ]
    },
    {
      "id": "run_local.click_bell.object_instance.query_observation",
      "observed": true,
      "description": "The end effector approaches the bell from the initial frame and performs a task-relevant pressing motion in the success frames.",
      "confidence": 0.93,
      "frame_ids": [
        "initial",
        "success_before",
        "success_after",
        "final"
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
    "action": "stop",
    "observation_summary": "Completed 2 round(s); the task-agnostic hard execution cap is now exhausted.",
    "decision_reason": "external_max_agent_rounds_budget",
    "next_template_id": null,
    "remaining_template_ids_before_decision": [
      "object_position.right_fixed",
      "object_instance.base1"
    ],
    "round_budget_before_decision": 0,
    "evidence_assessment": {
      "schema_version": 1,
      "state": "external_hard_round_cap_reached",
      "required_action": "stop",
      "completed_rounds": 2,
      "max_agent_rounds": 2,
      "remaining_template_ids": [
        "object_position.right_fixed",
        "object_instance.base1"
      ],
      "policy_outcome_not_inferred": true
    },
    "next_round": null
  }
}
```

## 5. Final answer to the original Query

> 在本次 click_bell 评估中，ACT 策略在已测的两个钟属性条件下均完成任务：固定左侧位置与官方 base0 钟实例。ACT cohort 的官方成功率为 1.0（2/2，seed 100502）；这说明其在这两个已测条件上表现成功，但不足以据此宣称对钟属性具有广泛泛化能力。

```json
{
  "findings": [
    "ACT policy_under_evaluation cohort 的 official_check_success 聚合结果为 2/2 成功，success_rate=1.0，质量为 valid=2、missing=0、invalid=0；这是策略任务完成的官方数值结论。",
    "固定左侧位置条件下，ACT 的 official_check_success 为 1/1，success_rate=1.0；其 time_to_success 均值为 19.484 s。计划工具 bell_active_tcp_min_xy_error 的 ACT 均值为 0.009174784645438194 m，质量 valid=1、missing=0、invalid=0。",
    "官方 base0 钟实例条件下，ACT 的 official_check_success 为 1/1，success_rate=1.0；其 time_to_success 均值为 20.216 s，质量 valid=1、missing=0、invalid=0。",
    "跨本次两个 ACT episode，time_to_success 聚合均值为 19.85 s，中位数为 19.85 s，最小值 19.484 s，最大值 20.216 s，population_stddev 为 0.36599999999999966 s；该汇总仅包含 ACT cohort。",
    "两轮 expert_validation cohort 均为 1/1 官方成功，用于支持场景可解性和测量链路，不计入 ACT 性能。",
    "Execution VQA 在两轮均观察到机械臂对铃的可见按压及任务相关运动，且与数值结论一致；没有视觉与 simulator Tool 的证据冲突。",
    "评估流水线通过不等同于策略成功；本次策略成功由 ACT cohort 的 official_check_success 聚合结果支持。"
  ],
  "recommended_next_step": "优先在新的评估预算中执行未覆盖的 object_position.right_fixed 与 object_instance.base1，并为每个条件增加多个独立 seed/episode；继续以 official_check_success 为 ACT 的主结果，单独保留 expert_validation 作为可解性与仪器控制。",
  "limitations": [
    "每个已测条件仅有 1 个 ACT episode，且两轮均使用同一 seed 100502；少量 episode 不能构成泛化基准，也不能证明跨未测属性的稳健性。",
    "计划的 object_position.right_fixed 和 object_instance.base1 尚未执行，原因是 2 轮硬上限已耗尽。",
    "本次不支持或未评估钟的颜色、材质光泽、纹理、质量和尺度属性；因此不能对这些属性作出结论。",
    "固定左侧位置轮的计划工具是距离误差，而实例轮的计划工具是官方成功，二者度量不同，不应据此比较两种条件的接近精度。"
  ]
}
```

## 6. Boundaries

- Policy results and pipeline status are reported separately.
- Expert evidence, when present, is a solvability/instrumentation gate, not ACT performance.
- Few-shot N=1 rounds demonstrate method wiring, not benchmark-level generalization.
- Missing artifacts are shown as N/A; this report never substitutes proxy images or invented values.

## 7. Raw artifact index

- Server source: `mea/evaluation_runs/eval_20260723_batch17_clean_head_click_live_n1_v4/manifest.json`
- Server source: `mea/evaluation_runs/eval_20260723_batch17_clean_head_click_live_n1_v4/plan/evaluation_plan.json`
- Server source: `mea/evaluation_runs/eval_20260723_batch17_clean_head_click_live_n1_v4/plan/bound_task_session.json`
- Server source: `mea/evaluation_runs/eval_20260723_batch17_clean_head_click_live_n1_v4/summary/evidence_bundle.json`
- Server source: `mea/evaluation_runs/eval_20260723_batch17_clean_head_click_live_n1_v4/feedback/feedback.json`
- Server source: `mea/evaluation_runs/eval_20260723_batch17_clean_head_click_live_n1_v4/evaluation_report.md`
