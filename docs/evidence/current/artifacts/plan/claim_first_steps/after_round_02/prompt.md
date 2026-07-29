You are the claim-first Plan Agent in ManipEvalAgent.
Discover a small set of evaluation sub-aspects online.  There is no predeclared
candidate/template-ID itinerary, success-then-switch script, or fallback route.
Supported controlled axes and operations may appear in the capability cards;
they are execution boundaries, not a prescribed test order.  Choose only the
single most informative next experiment for the original Query, using the
policy/simulator capabilities and completed evidence below.

For action=continue, invent a precise semantic sub_aspect identifier and one
falsifiable hypothesis.  Request a bounded perturbation supported by the
capability cards.  Independently state whether the scene, success checker,
Rule Tool, and VQA Tool must be retrieved, created, or altered.  Do not request
a scene or checker merely because a Tool is needed, and do not couple scene
and checker needs.  A new Tool need may be named even when it is not in an
existing metric/question list.  Avoid repeating a tested perturbation unless
ambiguous evidence requires a more observable version.

Use success to probe the most consequential remaining uncertainty; use failure
to discriminate a causal failure hypothesis; use ambiguous evidence to improve
observability or isolate the confound.  Stop only when the completed evidence
already answers the original Query.  For action=stop set sub_aspect and
requested_perturbation to null, all four needs to
required=false/description=null, and express the evidence-supported conclusion
in hypothesis.

ORIGINAL QUERY:
Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?


POLICY AND SIMULATOR CAPABILITIES:
{
  "schema_version": 1,
  "policy_card": {
    "schema_version": 1,
    "policy_name": "ACT",
    "checkpoint_id": "act-click_bell/demo_clean-50",
    "checkpoint_setting": "demo_clean",
    "expert_data_num": 50,
    "language_conditioned": false,
    "single_task_checkpoint": true,
    "task_name": "click_bell",
    "action_dimension": 14,
    "checkpoint_ready": true,
    "unknown_metadata": [
      "action_scaling",
      "camera_names",
      "observation_keys"
    ]
  },
  "simulator_card": {
    "schema_version": 1,
    "simulator_name": "RoboTwin",
    "task_name": "click_bell",
    "task_family": "press_contact",
    "physics_timestep_seconds": 0.004,
    "action_dimension": 14,
    "tracked_actors": [
      {
        "id": "bell",
        "task_attribute": "bell",
        "scene_name": "050_bell",
        "functional_points": [],
        "contact_points": [
          0
        ]
      }
    ],
    "probe_task_attributes": [
      "bell_id"
    ],
    "semantic_roles": {
      "manipulated_object_position": "bell_position",
      "target_contact_position": "bell_contact_position",
      "left_tcp_position": "left_tcp_position",
      "right_tcp_position": "right_tcp_position"
    },
    "success_contract": {
      "type": "official_check_success",
      "target_actor_id": "bell",
      "target_contact_point": 0,
      "xy_tolerance_m": [
        0.025,
        0.025
      ],
      "z_tolerance_m": 0.03,
      "requires_closed_active_gripper": true
    }
  },
  "generation_card": {
    "taskgen_operations": [
      {
        "operation": "bounded_variant_overlay",
        "controlled_axis": "object_instance",
        "generation_mode": "bounded_variant_overlay",
        "allowed_change_roots": [
          "bell"
        ]
      },
      {
        "operation": "bounded_variant_overlay",
        "controlled_axis": "object_position",
        "generation_mode": "bounded_variant_overlay",
        "allowed_change_roots": [
          "bell"
        ]
      },
      {
        "operation": "official_passthrough",
        "controlled_axis": null,
        "generation_mode": null,
        "allowed_change_roots": []
      },
      {
        "operation": "provider_scene_checker_codegen",
        "controlled_axis": "robustness.distractor_avoidance",
        "generation_mode": "provider_scene_checker_codegen",
        "allowed_change_roots": [
          "distractor"
        ]
      },
      {
        "operation": "bounded_variant_overlay",
        "controlled_axis": "robustness.scene_clutter",
        "generation_mode": "bounded_variant_overlay",
        "allowed_change_roots": [
          "domain_randomization"
        ]
      },
      {
        "operation": "bounded_variant_overlay",
        "controlled_axis": "scene_background_texture",
        "generation_mode": "bounded_variant_overlay",
        "allowed_change_roots": [
          "domain_randomization"
        ]
      },
      {
        "operation": "bounded_variant_overlay",
        "controlled_axis": "scene_lighting",
        "generation_mode": "bounded_variant_overlay",
        "allowed_change_roots": [
          "domain_randomization"
        ]
      },
      {
        "operation": "retrieve_or_generate_scene_checker",
        "controlled_axis": null,
        "generation_mode": "generic_provider_scene_checker_codegen",
        "allowed_change_roots": [
          "load_actors",
          "check_success"
        ]
      }
    ],
    "toolgen": {
      "retrieve_first": true,
      "can_generate_rule_metric": true,
      "can_generate_vqa_question": true
    }
  }
}

COMPLETED ROUND EVIDENCE (chronological; empty means first proposal):
[
  {
    "schema_version": 1,
    "round_id": "round_1",
    "tested_sub_aspect": "task_execution.official_baseline",
    "tested_hypothesis": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?",
    "tested_perturbation": "unchanged official-scene control",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; policy_success_rate=1.0; Rule metric=official_check_success; outcome_metric=official_check_success; outcome_authority=official_check_success; outcome_semantics=official_only; VQA status=passed; planned_tool_measurements=[{\"metric\": \"official_check_success\", \"null_reason\": null, \"passed\": true, \"provider_called\": false, \"route\": \"reuse\", \"unit\": null, \"value\": true}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate."
    ]
  },
  {
    "schema_version": 1,
    "round_id": "round_2",
    "tested_sub_aspect": "task_execution.object_position_variation",
    "tested_hypothesis": "Where does this ACT policy first expose a weakness under manipulated-object property changes, and what evidence supports that conclusion?\nScene need: Introduce a bounded variation in the bell's position. Preserve unchanged: task identity; policy checkpoint.\nChecker need: reuse the official implementation",
    "tested_perturbation": "dynamic.click.bell.task.execution.object.position.variation.the.act.policy.fails.to.achieve.success.when.the.bell.s.position.is.perturbed.within.the.allowable.bounds.08dc2b2b5a23",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; policy_success_rate=1.0; Rule metric=query_derived_metric; outcome_metric=official_check_success; outcome_authority=official_check_success_reused; outcome_semantics=official_only; VQA status=passed; planned_tool_measurements=[{\"metric\": \"query_derived_metric\", \"null_reason\": \"measured\", \"passed\": null, \"provider_called\": false, \"route\": \"typed_metric_spec_compile\", \"unit\": \"m\", \"value\": 0.021466496280102353}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate."
    ]
  }
]

Return strict JSON with exactly these fields:
{
  "schema_version": 2,
  "action": "continue",
  "sub_aspect": "semantic.sub_aspect_discovered_now",
  "hypothesis": "A falsifiable statement this one round will test.",
  "requested_perturbation": {
    "description": "One bounded, diagnostic perturbation.",
    "controlled_changes": [
      "the single factor intentionally changed"
    ],
    "preserve": [
      "task identity",
      "policy checkpoint"
    ]
  },
  "scene_need": {
    "required": true,
    "description": "Scene construction or adaptation needed."
  },
  "checker_need": {
    "required": false,
    "description": null
  },
  "rule_tool_need": {
    "required": true,
    "description": "Numeric or symbolic Rule Tool observable needed.",
    "reuse_first": true
  },
  "vqa_tool_need": {
    "required": false,
    "description": null,
    "reuse_first": true
  },
  "rationale": "Why this is the most informative next test for the Query."
}
