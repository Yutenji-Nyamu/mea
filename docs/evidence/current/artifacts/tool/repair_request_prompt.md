You are ToolGen in ManipEvalAgent. Derive the smallest executable measurement needed by the open Query. First inspect both the trusted static registry and validated_generated_tools. For an exact static match, return schema_version=1 with its metric id. For an exact generated match, copy that entry's schema_version=2 request and MetricSpec exactly. Otherwise return schema_version=2 and a MetricSpec using only the advertised typed operator contracts and telemetry names. Replace angle-bracket placeholders with real advertised names or null. A registered composite target is an exact static match and may be selected by its schema_version=1 metric id; it will be generated and validated when no compatible registration exists. A fixed left/right signal does not satisfy an active-arm or active-gripper need when both sides are advertised. Do not invent an unavailable signal, task name, template, or aspect. Return strict JSON only.

ORIGINAL QUERY:
Where does this ACT policy first expose a weakness when generalizing over manipulated-object properties?

SEMANTIC CONCERN:
object_generalization.position_translation: The ACT policy's first weakness in manipulated-object generalization is spatial: translating the bell within a bounded reachable workspace will reduce official press success or increase final bell-contact distance relative to the unchanged control.

MEASUREMENT NEED:
ToolGen should reuse the official success checker and generate a rule metric reporting the final target-contact distance and whether the active gripper is closed, so partial spatial degradation is observable even if the binary outcome remains successful.

TELEMETRY AND TOOL CONTEXT:
{
  "forbidden_metric_ids": [
    "official_check_success",
    "time_to_success"
  ],
  "outcome_semantics": "generated_checker_experimental",
  "schema_version": 1,
  "task_name": "click_bell",
  "telemetry_schema": {
    "common_events": [
      "contact_interval",
      "success_transition"
    ],
    "semantic_fields": [
      {
        "actor_id": "bell",
        "name": "bell_position",
        "point_id": null,
        "side": null,
        "source": "actor_position"
      },
      {
        "actor_id": "bell",
        "name": "bell_contact_position",
        "point_id": 0,
        "side": null,
        "source": "actor_contact_position"
      },
      {
        "actor_id": null,
        "name": "left_tcp_position",
        "point_id": null,
        "side": "left",
        "source": "robot_tcp_position"
      },
      {
        "actor_id": null,
        "name": "right_tcp_position",
        "point_id": null,
        "side": "right",
        "source": "robot_tcp_position"
      }
    ],
    "tracked_actors": [
      {
        "contact_points": [
          0
        ],
        "functional_points": [],
        "id": "bell",
        "scene_name": "050_bell"
      }
    ]
  },
  "telemetry_schema_source": "executed_episode_schema",
  "tool_registry": {
    "composite_targets": [
      {
        "description": "Minimum XY distance between the official active-arm TCP and the bell contact point over the recorded trajectory.",
        "metric": "bell_active_tcp_min_xy_error",
        "oracle_kind": "private_semantic_trace_oracle",
        "supporting_examples": [
          "time_to_success"
        ]
      },
      {
        "description": "Elapsed simulator time from the first hammer pickup threshold crossing to the first strict physical hammer-block contact.",
        "metric": "pickup_to_first_contact_time",
        "oracle_kind": "composite_trusted_tools",
        "supporting_examples": [
          "first_hammer_pickup_step",
          "first_contact_step",
          "time_to_success"
        ]
      }
    ],
    "matching_policy": "strict_exact_metric_id",
    "schema_version": 1,
    "snapshot_sha256": "5cfb1e5d6b975611811c2b1f16306a3a928f375d44fb08a90a9f15cd9f270efd",
    "trusted_tools": [
      {
        "description": "Active-arm TCP path length at physics resolution.",
        "name": "ee_path_length",
        "sha256": "f36f7af78b78c16a3a8d95aca5e158b837c2ec518ab5a81847a101965fc447e7",
        "supported_task_names": [
          "beat_block_hammer"
        ],
        "tags": [
          "path",
          "motion",
          "轨迹",
          "路径",
          "运动"
        ],
        "version": 1
      },
      {
        "description": "First hammer-block contact physics step and time.",
        "name": "first_contact_step",
        "sha256": "efc807f0d52a634f948cf50d7cbf36f8a344df7cc41f394b15e587357c62b923",
        "supported_task_names": [
          "beat_block_hammer"
        ],
        "tags": [
          "first",
          "contact",
          "首次",
          "接触",
          "时间"
        ],
        "version": 1
      },
      {
        "description": "First physics step where hammer height rise reaches the task pickup threshold.",
        "name": "first_hammer_pickup_step",
        "sha256": "141205b152af5c33ca55e9d52ea2a05b75c25b49a59570b6c0d088e00b0ebe3d",
        "supported_task_names": [
          "beat_block_hammer"
        ],
        "tags": [
          "hammer",
          "pickup",
          "first",
          "step",
          "拿起",
          "首次",
          "时间"
        ],
        "version": 1
      },
      {
        "description": "Latched outcome from a validated experimental generated checker.",
        "name": "generated_check_success",
        "sha256": "03c152fabc89c773516e92ba5a880951c01904941866fa76147e3b367372045c",
        "supported_task_names": [
          "*"
        ],
        "tags": [
          "success",
          "generated",
          "experimental",
          "result"
        ],
        "version": 1
      },
      {
        "description": "Whether hammer and block ever had physical contact.",
        "name": "hammer_block_contact_ever",
        "sha256": "6a86ff262629c81c6ff7b2b8c1a00546a2d7399f8923f2d5a2376aba24d4c0cb",
        "supported_task_names": [
          "beat_block_hammer"
        ],
        "tags": [
          "contact",
          "hit",
          "接触",
          "敲"
        ],
        "version": 1
      },
      {
        "description": "Minimum official functional-point XY alignment error.",
        "name": "hammer_block_min_xy_error",
        "sha256": "73c258ea7f3c0a195da6e84b53890c023d4603d6f5a71d4c2aea9ce89383ee5f",
        "supported_task_names": [
          "beat_block_hammer"
        ],
        "tags": [
          "distance",
          "alignment",
          "接近",
          "距离",
          "敲"
        ],
        "version": 1
      },
      {
        "description": "Count physical hammer-left_camera contact intervals as one bounded unintended-contact proxy.",
        "name": "hammer_left_camera_contact_count",
        "sha256": "ceb22b4687b7d74db32ef8f2572a2874445043e2a455314e777cf7a409c9ab8c",
        "supported_task_names": [
          "beat_block_hammer"
        ],
        "tags": [
          "safety",
          "unintended",
          "camera",
          "contact",
          "collision"
        ],
        "version": 1
      },
      {
        "description": "Maximum hammer center height rise from the initial state.",
        "name": "hammer_pickup_height",
        "sha256": "81d30382c0b5216af3dbac643878ee17990dc9fa9500ce308fa924c641701149",
        "supported_task_names": [
          "beat_block_hammer"
        ],
        "tags": [
          "hammer",
          "pickup",
          "grasp",
          "拿起",
          "抬起"
        ],
        "version": 1
      },
      {
        "description": "Maximum contact-point impulse during hammer-block contact.",
        "name": "max_contact_impulse",
        "sha256": "10defb7d6e05b02a5b1607bb1534a5e3d102ce5efc1af2fec5e9677c8e9dce81",
        "supported_task_names": [
          "beat_block_hammer"
        ],
        "tags": [
          "impulse",
          "force",
          "contact",
          "冲量",
          "力度",
          "接触"
        ],
        "version": 1
      }
    ],
    "typed_metric_spec": {
      "execution": "compile_validate_register",
      "operations": [
        "event_count",
        "minimum_distance",
        "time_between_events"
      ],
      "schema_version": 1
    }
  },
  "typed_operator_contracts": {
    "event_count": {
      "event": {
        "actors": "<null_or_two_advertised_actor_ids>",
        "event_type": "contact_interval",
        "physical_only": true
      },
      "null_semantics": "zero_if_absent",
      "operation": "event_count",
      "schema_version": 1,
      "unit": "count"
    },
    "minimum_distance": {
      "dimensions": [
        "x",
        "y",
        "z"
      ],
      "left_signal": "<advertised_semantic_field_name>",
      "null_semantics": "null_if_no_finite_sample",
      "operation": "minimum_distance",
      "right_signal": "<different_advertised_semantic_field_name>",
      "schema_version": 1,
      "unit": "m"
    },
    "time_between_events": {
      "end_event": {
        "actors": null,
        "event_type": "success_transition",
        "physical_only": false
      },
      "null_semantics": "null_if_missing_or_reversed",
      "operation": "time_between_events",
      "schema_version": 1,
      "start_event": {
        "actors": "<null_or_two_advertised_actor_ids>",
        "event_type": "contact_interval",
        "physical_only": true
      },
      "unit": "s"
    }
  },
  "validated_generated_tools": []
}

OUTPUT SHAPE EXAMPLE (choose fields according to schema version):
{
  "schema_version": 2,
  "task_name": "click_bell",
  "metric": "query_derived_metric",
  "question": "The precise question answered by the metric.",
  "metric_spec": {
    "schema_version": 1,
    "operation": "minimum_distance",
    "left_signal": "actor_a_position",
    "right_signal": "actor_b_position",
    "dimensions": [
      "x",
      "y",
      "z"
    ],
    "unit": "m",
    "null_semantics": "null_if_no_finite_sample"
  }
}