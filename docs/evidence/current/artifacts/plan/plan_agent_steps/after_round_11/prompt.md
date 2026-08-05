You are the Plan Agent in ManipEvalAgent.
Discover a small set of evaluation sub-aspects online.  There is no predeclared
candidate/template-ID itinerary, success-then-switch script, or fallback route.
The capability card exposes only backend primitives such as scene/checker
generation, telemetry, Rule/VQA Tools, and artifact retrieval.  It is an
execution boundary, not an operation menu or prescribed test order.  Choose
only the single most informative next experiment for the original Query, using
the policy/simulator capabilities and completed evidence below.

CURRENT QUERY CONTRACT: every action=continue Proposal MUST set checker_need.required=true and describe the directly observable experimental predicate; false is invalid.

For action=continue, invent a precise semantic sub_aspect identifier and one
falsifiable hypothesis.  Request a bounded perturbation supported by the
capability cards.  Independently state whether the scene, success checker,
Rule Tool, and VQA Tool must be retrieved, created, or altered.  Do not request
a scene or checker merely because a Tool is needed, and do not couple scene
and checker needs.  A new Tool need may be named even when it is not in an
existing metric/question list.  Avoid repeating a tested perturbation unless
ambiguous evidence requires a more observable version.
Each Rule/VQA need must name one primary scalar or boolean observation for this
round.  Leave independent measurements for a later evidence-conditioned round
instead of bundling them into one Tool request.
For both rule_tool_need and vqa_tool_need, reuse_first MUST always be true,
including when required=false: retrieve-first is the ToolGen method contract,
not a choice to bypass reuse.
State the intentional delta in requested_perturbation.description and
controlled_changes with an explicit operation and concrete value or direction;
put unchanged conditions only in preserve.  When scene_need.required is true,
repeat that same explicit delta in scene_need.description.  Preserve only the
isolation-critical factors supported by a current preservation authority.
Fields merely listed as observable in the simulator card are measurement
capabilities, not preservation authorities.  Use "task identity" and "policy
checkpoint" as the default preserve set; add another condition only when the
current input identifies an authority that can compare it.  Do not add actor
identity, physics timestep, or object-to-target binding merely because those
fields appear in simulator metadata.  When an additional experimental checker
must retain the official goal, add exactly "official core predicate as a
required conjunct" to preserve.  Do not call the extended checker "official
success semantics" or claim full equivalence.
Request a generated checker only when every added relation is directly
observable from the advertised current-state simulator API.  Gripper closure
is not target contact, sequential events are not simultaneous events, and
height is not placement.  A declared actor contact point is a geometric
reference, not a PhysX contact-event identity: do not request that "point i is
physically contacted" unless the runtime explicitly binds collision contacts
to that point ID.  Prefer a directly observable point/TCP distance condition
or an entity-pair contact condition with exactly the semantics the API
supports.  If the exact relation is unavailable, choose a
scene-only experiment with a Rule/VQA observation, or another informative
sub-aspect, instead of asking TaskGen to implement a correlated proxy.
The generated checker is an experimental success criterion, not a way to
encode the predicted policy failure.  It must remain satisfiable by the expert
on the proposed scene.  In particular, do not request an added relation that
the controlled scene change itself makes deterministically false for both the
expert and the policy; any weakness must be established by rollout evidence.
If the original Query explicitly requires an experimental checker for every
generated round, scene-only is not a valid fallback: choose another directly
observable relation or stop with the unsupported limitation stated plainly.
TaskGen may retrieve or generate scene and checker code; ToolGen may retrieve
or generate Rule/VQA Tools.  These artifact primitives do not authorize policy
or controller intervention: do not reduce gripper precision, inject action
noise or latency, or change policy weights.  After successful evidence, refine
to another executable scene/checker/tool concern instead of relabelling a scene
change as an unavailable policy intervention.
If a Query calls an episode successful only when the official goal and any additional experimental condition both hold, request checker_need. A numeric difference Tool reports magnitude but cannot supply that pass/fail predicate. Mentioning the official goal or official predicate as one component of a combined condition does not make the Query official-only; record that invariant as 'official core predicate as a required conjunct', never as full official-success equivalence, and preserve every additional condition from the original Query. When both checker_need and rule_tool_need are required, keep their roles distinct: checker_need must describe a boolean conjunction such as 'official goal AND distractor remains uncontacted', while rule_tool_need describes the scalar or boolean observation used to diagnose it. Never copy a raw numeric measurement into checker_need as though it were a pass/fail predicate. If the checker applies a terminal-state distance threshold, the same-round Rule Tool must report the terminal value of that same distance. A trajectory peak or maximum is a separate trajectory weakness, not a scalar for setting the terminal threshold; later evidence refinement must not use its scale to relax, replace, or calibrate the terminal predicate. check_success is evaluated from simulator state, not from a whole-trajectory derived metric: smoothness, deviation, jerk, path length, or trajectory clearance belongs in rule_tool_need, never behind an invented checker helper.

Use success to probe the most consequential remaining uncertainty; use failure
to discriminate a causal failure hypothesis; use ambiguous evidence to improve
observability or isolate the confound.  When completed evidence is non-empty,
the rationale must cite a concrete observed outcome or limitation and explain
why it changed the priority of this sub-aspect.  Do not present a candidate
that was already frozen before seeing that evidence as evidence-conditioned
refinement.  If completed evidence contains a finite scalar, bracket the next
intervention or falsifiable threshold around that observed scale.  Never put a
numeric boundary into a generated success checker unless that exact boundary
comes from the original Query or from completed finite scalar/state evidence.
A successful control alone is not numeric calibration.  After a checker
fixture fails, use its expert-terminal actor/TCP coordinates to derive or
bracket a new observable boundary; do not repeat the same arbitrary threshold
with only an actor or robot-side relabel.  When no grounded boundary exists,
choose an exact discrete relation supported by the current-state API, request
scene-only diagnostic evidence, or report the need unsupported.  For a broad
robustness Query, for example, a successful control
can justify selecting the highest-risk supported perturbation, while a failed
control should redirect to baseline reliability or failure diagnosis.
For a pre-policy TaskGen failure, inspect `bounded_repair_evidence` as well as
the terminal diagnosis.  If an earlier expert fixture gives concrete terminal
state showing that the requested boolean relation is false, do not repeat that
same relation merely because the local repair later violated the Proposal.
Use the simulator state to correct the Proposal itself or switch concern.
Failure example from prior runs: after a successful generated test whose live
scalar shows a comfortable margin, merely increasing the same scene factor is
not a new sub-aspect unless that value brackets a clear boundary.  When the
scalar instead weakens the current hypothesis, switch to the most informative
orthogonal concern that the capability card can execute.  State explicitly
which observed Tool value or outcome caused the switch.  Conversely, do not
manufacture another concern
after the completed evidence already satisfies the Query contract: propose
action=stop so the contract can validate the answer.

Interpret completed evidence by its declared role.  The top-level `outcome`
is the authoritative verdict for the tested hypothesis.  A
`diagnostic_tool_measurements` value is supporting diagnosis only and never
rewrites that verdict.  Preserve the Tool's temporal semantics exactly:
`peak`/`maximum over the rollout` is not a terminal/current-state value.
Failure example: if `outcome="success"` for a terminal checker while a
trajectory-peak distance is large, do not call that a terminal failure or a
failing existential witness.  The correct next step may diagnose the large
transient or choose a stronger scene challenge, but it must retain the
successful terminal-checker result.

Stop when the completed evidence answers the original Query.  You may also
stop when the candidate universe remains open but the advertised runtime
capabilities contain no further distinct informative executable concern.  In
that case state explicitly that the result is inconclusive and that untested
concerns may remain; information saturation is not evidence sufficiency.  For
action=stop set sub_aspect and
requested_perturbation to null, all four needs to
required=false/description=null, and express the evidence-supported conclusion
in hypothesis.

ORIGINAL QUERY:
Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.


POLICY AND SIMULATOR CAPABILITIES:
{
  "schema_version": 2,
  "policy_card": {
    "schema_version": 1,
    "policy_name": "SmolVLA",
    "checkpoint_id": "lerobot/smolvla_robotwin",
    "checkpoint_setting": "shared_official",
    "expert_data_num": null,
    "language_conditioned": true,
    "single_task_checkpoint": false,
    "training_tasks": [
      "press_stapler"
    ],
    "supports_unseen_tasks": false,
    "task_name": "press_stapler",
    "action_dimension": 14,
    "checkpoint_ready": true,
    "unknown_metadata": [
      "action_scaling",
      "camera_names",
      "observation_keys",
      "expert_data_num",
      "semantic_actor_schema"
    ]
  },
  "simulator_card": {
    "schema_version": 1,
    "simulator_name": "RoboTwin",
    "task_name": "press_stapler",
    "task_family": "robotwin_official_task",
    "physics_timestep_seconds": 0.004,
    "action_dimension": 14,
    "tracked_actors": [],
    "probe_task_attributes": [],
    "semantic_roles": {},
    "success_contract": {
      "authority": "official_task_check_success_only",
      "semantic_telemetry_available": false
    }
  },
  "generation_card": {
    "backend_primitives": {
      "scene": true,
      "checker": true,
      "telemetry": false,
      "rule": true,
      "vqa": true,
      "retrieve": true,
      "generate": true
    }
  }
}

COMPLETED ROUND EVIDENCE (chronological; empty means first proposal):
[
  {
    "schema_version": 1,
    "round_id": "round_1",
    "tested_sub_aspect": "task_execution.official_baseline",
    "tested_hypothesis": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.",
    "tested_perturbation": "unchanged official-scene control",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; authoritative_candidate_outcome=success; success_predicate_metric=official_check_success; success_predicate_value=1.0; success_predicate_authority=official_check_success; success_predicate_semantics=official_only; policy_success_rate=1.0; Rule metric=official_check_success; VQA status=skipped; diagnostic_tool_role=supporting_measurement_not_success_authority; diagnostic_tool_measurements=[{\"metric\": \"official_check_success\", \"null_reason\": null, \"passed\": true, \"provider_called\": false, \"route\": \"reuse\", \"unit\": null, \"value\": true}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate."
    ]
  },
  {
    "schema_version": 1,
    "round_id": "round_2",
    "tested_sub_aspect": "scene_robustness.initial_lateral_displacement",
    "tested_hypothesis": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.\nScene need: Construct or adapt the official press_stapler scene by translating the stapler initial position from the official reset pose by +0.03 m along the world x-axis. Preserve unchanged: task identity; policy checkpoint.\nChecker need: Boolean terminal predicate: official goal AND the stapler is physically contacting either robot gripper in the terminal simulator state.",
    "tested_perturbation": "dynamic.press.stapler.scene.robustness.initial.lateral.displacement.translating.the.stapler.s.initial.position.by.0.03.m.along.the.world.x.axis.will.expose.a.policy.weakness.the.policy.will.not.satisfy.the.combined.predicate.of.official.task.success.and.terminal.physical.contact.between.the.stapler.and.either.robot.gripper.ee000124987f",
    "outcome": "ambiguous",
    "evidence_summary": "EvidencePacket strength=pipeline_invalid; authoritative_candidate_outcome=ambiguous; success_predicate_metric=None; success_predicate_value=None; success_predicate_authority=None; success_predicate_semantics=non_comparable; policy_success_rate=None; Rule metric=generated_check_success; VQA status=skipped; diagnostic_tool_role=supporting_measurement_not_success_authority; diagnostic_tool_measurements=[]. planning_observation=taskgen_materialization_failed; policy_sample_count=0; diagnosis=generated checker failed live negative/positive fixtures: {\"expert_terminal_actor_xyz_m\": {\"stapler\": [-0.097097, 0.011531, 0.741]}, \"expert_terminal_actor_z_m\": {\"stapler\": 0.741}, \"expert_terminal_robot_tcp_xyz_m\": {\"left\": [-0.127417, 0.022307, 0.810949], \"right\": [0.305233, -0.192807, 0.941005]}, \"failed_fixtures\": [{\"expected\": true, \"fixture_id\": \"official_expert_terminal_positive\", \"observed\": false}], \"initial_actor_xyz_m\": {\"stapler\": [-0.097097, 0.011531, 0.741]}, \"initial_actor_z_m\": {\"stapler\": 0.741}, \"initial_robot_tcp_xyz_m\": {\"left\": [-0.300323, -0.193828, 0.941998], \"right\": [0.305233, -0.192806, 0.940998]}}. bounded_repair_evidence=[{\"failure_kind\": \"invalid_candidate\", \"message\": \"generated checker must enforce self.mea_official_check_success() as a required boolean conjunct when the Proposal preserves or composes the official task goal\", \"stage\": \"scene_codegen\"}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate.",
      "The typed Rule/VQA/pipeline evidence is not sufficient: latest_pipeline_failed",
      "Policy success was not reported for this round.",
      "TaskGen rejected this candidate before policy execution; it is planning evidence only and contributes N=0 policy samples."
    ]
  },
  {
    "schema_version": 1,
    "round_id": "round_3",
    "tested_sub_aspect": "scene_robustness.lateral_shift_terminal_left_tcp_proximity",
    "tested_hypothesis": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.\nScene need: Construct or adapt the official press_stapler scene by translating the stapler initial position by +0.03 m along the world x-axis. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.\nChecker need: Boolean terminal predicate: official core predicate as a required conjunct AND Euclidean distance between left_tcp_position and stapler_position is less than or equal to 0.080 m.",
    "tested_perturbation": "dynamic.press.stapler.scene.robustness.lateral.shift.terminal.left.tcp.proximity.after.translating.the.stapler.initial.position.by.0.03.m.along.the.world.x.axis.the.policy.will.fail.the.satisfiable.combined.terminal.criterion.of.official.press.stapler.success.and.left.tcp.to.stapler.distance.no.greater.than.0.080.m.exposing.a.spatial.retargeting.weakness.5536913db934",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; authoritative_candidate_outcome=success; success_predicate_metric=generated_check_success; success_predicate_value=1.0; success_predicate_authority=llm_generated_python_ast_validated; success_predicate_semantics=expected_semantic_extension; policy_success_rate=1.0; Rule metric=terminal_left_tcp_to_stapler_distance; VQA status=skipped; diagnostic_tool_role=supporting_measurement_not_success_authority; diagnostic_tool_measurements=[{\"metric\": \"terminal_left_tcp_to_stapler_distance\", \"null_reason\": \"measured\", \"passed\": null, \"provider_called\": true, \"returns_diagnostic_not_success\": true, \"route\": \"provider_python_codegen\", \"unit\": \"m\", \"value\": 0.07994672000122097}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate.",
      "This round is judged by the bounded generated_check_success predicate and is not an official RoboTwin success result.",
      "The generated checker has not been certified as equivalent to the official core predicate; its verdict must be treated as experimental."
    ]
  },
  {
    "schema_version": 1,
    "round_id": "round_4",
    "tested_sub_aspect": "scene_robustness.lateral_shift_boundary_refinement",
    "tested_hypothesis": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.\nScene need: Alter the retrieved official press_stapler scene by setting the stapler initial world-x offset to +0.031 m relative to the official reset pose; no other scene modification is requested. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.\nChecker need: Retrieve the official checker and create an experimental terminal checker whose boolean predicate is: official core predicate as a required conjunct AND Euclidean distance(left_tcp_position, stapler_position) <= 0.080 m.",
    "tested_perturbation": "dynamic.press.stapler.scene.robustness.lateral.shift.boundary.refinement.increasing.the.stapler.initial.world.x.displacement.from.0.030.m.to.0.031.m.relative.to.the.official.reset.pose.will.cause.the.policy.to.fail.the.terminal.conjunction.of.the.official.core.predicate.and.left.tcp.to.stapler.distance.less.than.or.equal.to.0.080.m.exposing.a.bounded.spatial.retargeting.weakness.2cfa9bfa7a31",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; authoritative_candidate_outcome=success; success_predicate_metric=generated_check_success; success_predicate_value=1.0; success_predicate_authority=llm_generated_python_ast_validated; success_predicate_semantics=expected_semantic_extension; policy_success_rate=1.0; Rule metric=terminal_left_tcp_to_stapler_distance; VQA status=skipped; diagnostic_tool_role=supporting_measurement_not_success_authority; diagnostic_tool_measurements=[{\"metric\": \"terminal_left_tcp_to_stapler_distance\", \"null_reason\": \"measured\", \"passed\": null, \"provider_called\": false, \"returns_diagnostic_not_success\": true, \"route\": \"run_local_reuse\", \"unit\": \"m\", \"value\": 0.07990267815148812}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate.",
      "This round is judged by the bounded generated_check_success predicate and is not an official RoboTwin success result.",
      "The generated checker has not been certified as equivalent to the official core predicate; its verdict must be treated as experimental."
    ]
  },
  {
    "schema_version": 1,
    "round_id": "round_5",
    "tested_sub_aspect": "scene_robustness.lateral_shift_terminal_proximity_boundary",
    "tested_hypothesis": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.\nScene need: Retrieve or adapt the official press_stapler scene by translating the stapler initial position from the official reset pose by +0.030 m along the world x-axis. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.\nChecker need: Retrieve the official checker and create an experimental terminal checker with the boolean predicate: official core predicate as a required conjunct AND Euclidean distance(left_tcp_position, stapler_position) <= 0.07992 m.",
    "tested_perturbation": "dynamic.press.stapler.scene.robustness.lateral.shift.terminal.proximity.boundary.with.the.stapler.translated.by.0.030.m.along.world.x.from.the.official.reset.pose.the.policy.will.fail.the.experimental.terminal.conjunction.of.the.official.core.predicate.and.left.tcp.to.stapler.distance.less.than.or.equal.to.0.07992.m.revealing.that.its.previously.successful.spatial.retargeting.margin.is.below.the.evidence.grounded.boundary.0946283b238c",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; authoritative_candidate_outcome=success; success_predicate_metric=generated_check_success; success_predicate_value=1.0; success_predicate_authority=llm_generated_python_ast_validated; success_predicate_semantics=expected_semantic_extension; policy_success_rate=1.0; Rule metric=terminal_left_tcp_to_stapler_distance; VQA status=skipped; diagnostic_tool_role=supporting_measurement_not_success_authority; diagnostic_tool_measurements=[{\"metric\": \"terminal_left_tcp_to_stapler_distance\", \"null_reason\": \"measured\", \"passed\": null, \"provider_called\": false, \"returns_diagnostic_not_success\": true, \"route\": \"run_local_reuse\", \"unit\": \"m\", \"value\": 0.07982503806186039}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate.",
      "This round is judged by the bounded generated_check_success predicate and is not an official RoboTwin success result.",
      "The generated checker has not been certified as equivalent to the official core predicate; its verdict must be treated as experimental."
    ]
  },
  {
    "schema_version": 1,
    "round_id": "round_6",
    "tested_sub_aspect": "scene_robustness.orthogonal_world_y_displacement_terminal_tcp_proximity",
    "tested_hypothesis": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.\nScene need: Construct or adapt the official press_stapler scene by translating the stapler initial position from the official reset pose by +0.030 m along the world y-axis; make no other scene modification. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.\nChecker need: Create an experimental terminal checker whose boolean predicate is the official core predicate as a required conjunct AND Euclidean distance between left_tcp_position and stapler_position is less than or equal to 0.080 m.",
    "tested_perturbation": "dynamic.press.stapler.scene.robustness.orthogonal.world.y.displacement.terminal.tcp.proximity.translating.the.stapler.s.initial.position.by.0.030.m.along.the.world.y.axis.will.cause.the.policy.to.fail.the.experimental.terminal.predicate.consisting.of.the.official.core.predicate.and.left.tcp.to.stapler.distance.no.greater.than.0.080.m.exposing.an.orthogonal.lateral.retargeting.weakness.while.the.expert.remains.able.to.satisfy.the.predicate.58f7e703ef15",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; authoritative_candidate_outcome=success; success_predicate_metric=generated_check_success; success_predicate_value=1.0; success_predicate_authority=llm_generated_python_ast_validated; success_predicate_semantics=expected_semantic_extension; policy_success_rate=1.0; Rule metric=terminal_left_tcp_to_stapler_distance; VQA status=skipped; diagnostic_tool_role=supporting_measurement_not_success_authority; diagnostic_tool_measurements=[{\"metric\": \"terminal_left_tcp_to_stapler_distance\", \"null_reason\": \"measured\", \"passed\": null, \"provider_called\": false, \"returns_diagnostic_not_success\": true, \"route\": \"run_local_reuse\", \"unit\": \"m\", \"value\": 0.07998626327914184}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate.",
      "This round is judged by the bounded generated_check_success predicate and is not an official RoboTwin success result.",
      "The generated checker has not been certified as equivalent to the official core predicate; its verdict must be treated as experimental."
    ]
  },
  {
    "schema_version": 1,
    "round_id": "round_7",
    "tested_sub_aspect": "scene_robustness.vertical_reach_displacement_terminal_tcp_proximity",
    "tested_hypothesis": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.\nScene need: Retrieve or adapt the official press_stapler scene and translate only the stapler initial position by +0.030 m along world z. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.\nChecker need: Retrieve the official checker and create an experimental terminal checker with the boolean predicate: official core predicate as a required conjunct AND Euclidean distance(left_tcp_position, stapler_position) <= 0.080 m.",
    "tested_perturbation": "dynamic.press.stapler.scene.robustness.vertical.reach.displacement.terminal.tcp.proximity.translating.the.stapler.initial.position.by.0.030.m.along.the.world.z.axis.will.cause.the.policy.to.fail.the.terminal.conjunction.of.the.official.core.predicate.and.a.left.tcp.to.stapler.distance.of.at.most.0.080.m.while.the.expert.can.satisfy.that.criterion.8b06bca69155",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; authoritative_candidate_outcome=success; success_predicate_metric=generated_check_success; success_predicate_value=1.0; success_predicate_authority=llm_generated_python_ast_validated; success_predicate_semantics=expected_semantic_extension; policy_success_rate=1.0; Rule metric=terminal_left_tcp_to_stapler_distance; VQA status=skipped; diagnostic_tool_role=supporting_measurement_not_success_authority; diagnostic_tool_measurements=[{\"metric\": \"terminal_left_tcp_to_stapler_distance\", \"null_reason\": \"measured\", \"passed\": null, \"provider_called\": false, \"returns_diagnostic_not_success\": true, \"route\": \"run_local_reuse\", \"unit\": \"m\", \"value\": 0.07992299451213755}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate.",
      "This round is judged by the bounded generated_check_success predicate and is not an official RoboTwin success result.",
      "The generated checker has not been certified as equivalent to the official core predicate; its verdict must be treated as experimental."
    ]
  },
  {
    "schema_version": 1,
    "round_id": "round_8",
    "tested_sub_aspect": "scene_robustness.lateral_shift.evidence_bracketed_terminal_tcp_margin",
    "tested_hypothesis": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.\nScene need: Retrieve or adapt the official press_stapler scene by translating only the stapler initial position from the official reset pose by +0.030 m along world x. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.\nChecker need: Retrieve the official checker and create an experimental terminal checker with the boolean predicate: official core predicate as a required conjunct AND Euclidean distance(left_tcp_position, stapler_position) <= 0.07980 m.",
    "tested_perturbation": "dynamic.press.stapler.scene.robustness.lateral.shift.evidence.bracketed.terminal.tcp.margin.with.the.stapler.initially.translated.by.0.030.m.along.world.x.the.policy.will.fail.the.satisfiable.terminal.conjunction.of.the.official.core.predicate.and.left.tcp.to.stapler.distance.0.07980.m.exposing.an.evidence.bracketed.spatial.retargeting.margin.weakness.6ad07b3dba9b",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; authoritative_candidate_outcome=success; success_predicate_metric=generated_check_success; success_predicate_value=1.0; success_predicate_authority=llm_generated_python_ast_validated; success_predicate_semantics=expected_semantic_extension; policy_success_rate=1.0; Rule metric=terminal_left_tcp_to_stapler_distance; VQA status=skipped; diagnostic_tool_role=supporting_measurement_not_success_authority; diagnostic_tool_measurements=[{\"metric\": \"terminal_left_tcp_to_stapler_distance\", \"null_reason\": \"measured\", \"passed\": null, \"provider_called\": false, \"returns_diagnostic_not_success\": true, \"route\": \"run_local_reuse\", \"unit\": \"m\", \"value\": 0.0796477768203362}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate.",
      "This round is judged by the bounded generated_check_success predicate and is not an official RoboTwin success result.",
      "The generated checker has not been certified as equivalent to the official core predicate; its verdict must be treated as experimental."
    ]
  },
  {
    "schema_version": 1,
    "round_id": "round_9",
    "tested_sub_aspect": "scene_robustness.initial_stapler_yaw_orientation",
    "tested_hypothesis": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.\nScene need: Retrieve the official press_stapler scene and alter only the stapler initial world-z yaw by +15 degrees relative to its official reset orientation. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.\nChecker need: Retrieve the official checker and create an experimental terminal checker with the directly observable boolean predicate: official core predicate as a required conjunct AND Euclidean distance(left_tcp_position, stapler_position) <= 0.07980 m.",
    "tested_perturbation": "dynamic.press.stapler.scene.robustness.initial.stapler.yaw.orientation.with.the.stapler.rotated.by.15.degrees.about.the.world.z.axis.from.its.official.reset.orientation.the.policy.will.fail.the.terminal.conjunction.of.the.official.core.predicate.and.left.tcp.to.stapler.distance.less.than.or.equal.to.0.07980.m.revealing.orientation.sensitive.retargeting.despite.its.successful.translational.perturbation.results.7b8aa62577e6",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; authoritative_candidate_outcome=success; success_predicate_metric=generated_check_success; success_predicate_value=1.0; success_predicate_authority=llm_generated_python_ast_validated; success_predicate_semantics=expected_semantic_extension; policy_success_rate=1.0; Rule metric=terminal_left_tcp_to_stapler_distance; VQA status=skipped; diagnostic_tool_role=supporting_measurement_not_success_authority; diagnostic_tool_measurements=[{\"metric\": \"terminal_left_tcp_to_stapler_distance\", \"null_reason\": \"measured\", \"passed\": null, \"provider_called\": false, \"returns_diagnostic_not_success\": true, \"route\": \"run_local_reuse\", \"unit\": \"m\", \"value\": 0.07974978433448847}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate.",
      "This round is judged by the bounded generated_check_success predicate and is not an official RoboTwin success result.",
      "The generated checker has not been certified as equivalent to the official core predicate; its verdict must be treated as experimental."
    ]
  },
  {
    "schema_version": 1,
    "round_id": "round_10",
    "tested_sub_aspect": "scene_robustness.initial_stapler_pitch_alignment",
    "tested_hypothesis": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.\nScene need: Retrieve or adapt the official press_stapler scene and alter only the stapler initial orientation by applying a +15 degree rotation about the world-y axis relative to the official reset orientation. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.\nChecker need: Retrieve the official checker and create an experimental terminal checker with the directly observable boolean predicate: official core predicate as a required conjunct AND Euclidean distance(left_tcp_position, stapler_position) <= 0.07980 m.",
    "tested_perturbation": "dynamic.press.stapler.scene.robustness.initial.stapler.pitch.alignment.rotating.the.stapler.by.15.degrees.about.the.world.y.axis.from.its.official.reset.orientation.will.cause.the.policy.to.fail.the.satisfiable.terminal.conjunction.of.the.official.core.predicate.and.left.tcp.to.stapler.distance.less.than.or.equal.to.0.07980.m.exposing.sensitivity.to.press.surface.pitch.alignment.d77104b18bec",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; authoritative_candidate_outcome=success; success_predicate_metric=generated_check_success; success_predicate_value=1.0; success_predicate_authority=llm_generated_python_ast_validated; success_predicate_semantics=expected_semantic_extension; policy_success_rate=1.0; Rule metric=terminal_left_tcp_to_stapler_distance; VQA status=skipped; diagnostic_tool_role=supporting_measurement_not_success_authority; diagnostic_tool_measurements=[{\"metric\": \"terminal_left_tcp_to_stapler_distance\", \"null_reason\": \"measured\", \"passed\": null, \"provider_called\": false, \"returns_diagnostic_not_success\": true, \"route\": \"run_local_reuse\", \"unit\": \"m\", \"value\": 0.07976609109967475}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate.",
      "This round is judged by the bounded generated_check_success predicate and is not an official RoboTwin success result.",
      "The generated checker has not been certified as equivalent to the official core predicate; its verdict must be treated as experimental."
    ]
  },
  {
    "schema_version": 1,
    "round_id": "round_11",
    "tested_sub_aspect": "scene_robustness.initial_stapler_roll_orientation",
    "tested_hypothesis": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.\nScene need: Retrieve or adapt the official press_stapler scene and apply only a +15 degree world-x roll rotation to the stapler initial orientation relative to its official reset orientation. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.\nChecker need: Retrieve the official checker and create an experimental terminal checker whose boolean predicate is: official core predicate as a required conjunct AND Euclidean distance(left_tcp_position, stapler_position) <= 0.07980 m.",
    "tested_perturbation": "dynamic.press.stapler.scene.robustness.initial.stapler.roll.orientation.applying.a.bounded.15.degree.roll.rotation.about.the.world.x.axis.to.the.stapler.from.its.official.reset.orientation.will.cause.the.policy.to.fail.the.satisfiable.terminal.conjunction.of.the.official.core.predicate.and.left.tcp.to.stapler.distance.less.than.or.equal.to.0.07980.m.exposing.sensitivity.to.an.untested.press.surface.roll.alignment.concern.a60098bc03e2",
    "outcome": "success",
    "evidence_summary": "EvidencePacket strength=sufficient; authoritative_candidate_outcome=success; success_predicate_metric=generated_check_success; success_predicate_value=1.0; success_predicate_authority=llm_generated_python_ast_validated; success_predicate_semantics=expected_semantic_extension; policy_success_rate=1.0; Rule metric=terminal_left_tcp_to_stapler_distance; VQA status=skipped; diagnostic_tool_role=supporting_measurement_not_success_authority; diagnostic_tool_measurements=[{\"metric\": \"terminal_left_tcp_to_stapler_distance\", \"null_reason\": \"measured\", \"passed\": null, \"provider_called\": false, \"returns_diagnostic_not_success\": true, \"route\": \"run_local_reuse\", \"unit\": \"m\", \"value\": 0.07961811625797877}].",
    "limitations": [
      "One bounded runtime round is not a statistical generalization estimate.",
      "This round is judged by the bounded generated_check_success predicate and is not an official RoboTwin success result.",
      "The generated checker has not been certified as equivalent to the official core predicate; its verdict must be treated as experimental."
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
    "description": "Set one advertised factor from its baseline to one bounded diagnostic value.",
    "controlled_changes": [
      "factor: baseline -> diagnostic value"
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
    "required": true,
    "description": "The additional experimental success predicate."
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
