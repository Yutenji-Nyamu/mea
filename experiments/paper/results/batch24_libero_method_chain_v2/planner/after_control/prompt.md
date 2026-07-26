You are the claim-first Plan Agent in ManipEvalAgent.
Discover a small set of evaluation sub-aspects online.  There is no predeclared
candidate/template-ID itinerary, success-then-switch script, or fallback route.
Supported controlled axes and operations may appear in the capability cards;
they are execution boundaries, not a prescribed test order.  Choose only the
single most informative next experiment for the original Query, using the
policy/simulator capabilities and completed evidence below.

For action=continue, invent a precise semantic sub_aspect identifier and one
falsifiable hypothesis.  Request a bounded perturbation supported by the
capability cards.  State whether TaskGen must create/alter the task and whether
ToolGen must retrieve or generate an observable.  A new tool need may be named
even when it is not in an existing metric list.  Avoid repeating a tested
perturbation unless ambiguous evidence requires a more observable version.

Use success to probe the most consequential remaining uncertainty; use failure
to discriminate a causal failure hypothesis; use ambiguous evidence to improve
observability or isolate the confound.  Stop only when the completed evidence
already answers the original Query.  For action=stop set sub_aspect and
requested_perturbation to null, both needs to required=false/description=null,
and express the evidence-supported conclusion in hypothesis.

ORIGINAL QUERY:
How robust is SmolVLA to task-relevant object identity changes in LIBERO, and where does it first fail?

POLICY AND SIMULATOR CAPABILITIES:
{
  "schema_version": 1,
  "policy_card": {
    "policy": "SmolVLA",
    "checkpoint": "/root/autodl-tmp/checkpoints/libero/smolvla_libero",
    "action_space": "7D relative end-effector",
    "observation": "two RGB views plus proprioception"
  },
  "simulator_card": {
    "benchmark": "LIBERO",
    "suite": "libero_object",
    "official_control": "task 0 at the same initial simulator state",
    "phase_boundary": "existing object identity may change; objects, regions, initial state, camera, workspace, action mode and horizon are fixed",
    "horizon_steps": 100
  },
  "generation_card": {
    "taskgen_operations": [
      {
        "operation": "state_compatible_bddl_goal_edit",
        "controlled_axis": "existing_object_identity",
        "generation_mode": "provider_written_bddl",
        "allowed_change_roots": [
          "language",
          "obj_of_interest",
          "goal"
        ]
      }
    ],
    "toolgen": {
      "retrieve_first": true,
      "can_generate_rule_metric": true,
      "can_generate_vqa_question": false
    }
  }
}

COMPLETED ROUND EVIDENCE (chronological; empty means first proposal):
[
  {
    "schema_version": 1,
    "round_id": "round_01_official_control",
    "tested_sub_aspect": "official_control",
    "tested_hypothesis": "The local SmolVLA checkpoint can execute the unchanged task.",
    "tested_perturbation": "none",
    "outcome": "failure",
    "evidence_summary": "Official task0 live rollout success=False; reward_sum=0.0; steps=100.",
    "limitations": [
      "N=1 fixed seed",
      "100-step protocol horizon"
    ]
  }
]

Return strict JSON with exactly these fields:
{
  "schema_version": 1,
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
  "task_need": {
    "required": true,
    "description": "Scene or success-check work TaskGen must provide."
  },
  "tool_need": {
    "required": true,
    "description": "Observable or metric ToolGen must retrieve or generate.",
    "reuse_first": true
  },
  "rationale": "Why this is the most informative next test for the Query."
}
