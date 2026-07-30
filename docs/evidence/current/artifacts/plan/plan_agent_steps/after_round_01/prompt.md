You are the Plan Agent in ManipEvalAgent.
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
The generic RoboTwin TaskGen surface can change only what the advertised
allowed_change_roots directly implement.  In particular, load_actors can alter
actors, assets, appearance, scale, pose, clutter, lighting, and other simulator
scene state; it cannot reduce policy/controller/gripper precision, inject
action noise or latency, or change policy weights unless an explicit runtime
intervention root is advertised.  After successful evidence, refine to another
executable physical scene/checker/tool concern instead of relabelling a scene
change as an unavailable policy intervention.
If a Query calls an episode successful only when the official goal and any additional experimental condition both hold, request checker_need. A numeric difference Tool reports magnitude but cannot supply that pass/fail predicate.

Use success to probe the most consequential remaining uncertainty; use failure
to discriminate a causal failure hypothesis; use ambiguous evidence to improve
observability or isolate the confound.  When completed evidence is non-empty,
the rationale must cite a concrete observed outcome or limitation and explain
why it changed the priority of this sub-aspect.  Do not present a candidate
that was already frozen before seeing that evidence as evidence-conditioned
refinement.  For a broad robustness Query, for example, a successful control
can justify selecting the highest-risk supported perturbation, while a failed
control should redirect to baseline reliability or failure diagnosis.

Stop only when the completed evidence already answers the original Query.  For
action=stop set sub_aspect and
requested_perturbation to null, all four needs to
required=false/description=null, and express the evidence-supported conclusion
in hypothesis.

ORIGINAL QUERY:
这个ACT策略在grab_roller任务中最先会在哪种可执行物体属性或场景变化上暴露弱点？


POLICY AND SIMULATOR CAPABILITIES:
{
  "schema_version": 1,
  "policy_card": {
    "schema_version": 1,
    "policy_name": "ACT",
    "checkpoint_id": "act-grab_roller/demo_clean-50",
    "checkpoint_setting": "demo_clean",
    "expert_data_num": 50,
    "language_conditioned": false,
    "single_task_checkpoint": true,
    "task_name": "grab_roller",
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
    "task_name": "grab_roller",
    "task_family": "dual_arm_lift",
    "physics_timestep_seconds": 0.004,
    "action_dimension": 14,
    "tracked_actors": [
      {
        "id": "roller",
        "task_attribute": "roller",
        "scene_name": "102_roller",
        "functional_points": [],
        "contact_points": [
          0,
          1
        ]
      }
    ],
    "probe_task_attributes": [],
    "semantic_roles": {
      "manipulated_object_position": "roller_position",
      "left_target_contact_position": "roller_left_contact_position",
      "right_target_contact_position": "roller_right_contact_position",
      "left_tcp_position": "left_tcp_position",
      "right_tcp_position": "right_tcp_position"
    },
    "success_contract": {
      "type": "official_check_success",
      "target_actor_id": "roller",
      "minimum_height_m": 0.8,
      "requires_left_gripper_closed": true,
      "requires_right_gripper_closed": true
    }
  },
  "generation_card": {
    "taskgen_operations": [
      {
        "operation": "official_passthrough",
        "controlled_axis": null,
        "generation_mode": null,
        "allowed_change_roots": []
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
    "tested_hypothesis": "这个ACT策略在grab_roller任务中最先会在哪种可执行物体属性或场景变化上暴露弱点？",
    "tested_perturbation": "unchanged official-scene control",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; policy_success_rate=1.0; Rule metric=official_check_success; outcome_metric=official_check_success; outcome_authority=official_check_success; outcome_semantics=official_only; VQA status=passed; planned_tool_measurements=[{\"metric\": \"official_check_success\", \"null_reason\": null, \"passed\": true, \"provider_called\": false, \"route\": \"reuse\", \"unit\": null, \"value\": true}].",
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
