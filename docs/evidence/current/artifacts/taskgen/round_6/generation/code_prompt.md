Generate one RoboTwin experiment from the open Query-derived candidate below. Retrieve semantics from the official base program, but implement the requested scene and checker rather than selecting a catalog template.

EXPERIMENT CANDIDATE:
{
  "base_task": "press_stapler",
  "candidate_id": "dynamic.press.stapler.scene.robustness.orthogonal.world.y.displacement.terminal.tcp.proximity.translating.the.stapler.s.initial.position.by.0.030.m.along.the.world.y.axis.will.cause.the.policy.to.fail.the.experimental.terminal.predicate.consisting.of.the.official.core.predicate.and.left.tcp.to.stapler.distance.no.greater.than.0.080.m.exposing.an.orthogonal.lateral.retargeting.weakness.while.the.expert.remains.able.to.satisfy.the.predicate.58f7e703ef15",
  "checker_need": {
    "description": "Create an experimental terminal checker whose boolean predicate is the official core predicate as a required conjunct AND Euclidean distance between left_tcp_position and stapler_position is less than or equal to 0.080 m.",
    "kind": "generate",
    "reuse_first": true
  },
  "evaluation_intent": {
    "hypothesis": "Translating the stapler's initial position by +0.030 m along the world y-axis will cause the policy to fail the experimental terminal predicate consisting of the official core predicate and left-TCP-to-stapler distance no greater than 0.080 m, exposing an orthogonal lateral retargeting weakness while the expert remains able to satisfy the predicate.",
    "intent_id": "intent.b21106e7813967dd",
    "original_concern": "scene_robustness.orthogonal_world_y_displacement_terminal_tcp_proximity",
    "preserved_conditions": [
      "task identity",
      "policy checkpoint",
      "official core predicate as a required conjunct"
    ],
    "requested_change": "Translate the stapler initial position from the official reset pose by +0.030 m along the world y-axis.",
    "required_observation": "Create an experimental terminal checker whose boolean predicate is the official core predicate as a required conjunct AND Euclidean distance between left_tcp_position and stapler_position is less than or equal to 0.080 m. Report the single primary scalar observation terminal_left_tcp_to_stapler_distance: the terminal Euclidean distance in meters between left_tcp_position and stapler_position.",
    "schema_version": 1,
    "source_query": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope."
  },
  "intent_alignment": {
    "matched_intent_fields": [
      "requested_change",
      "preserved_conditions",
      "hypothesis",
      "required_observation"
    ],
    "rationale": "Candidate preserves the requested change, hypothesis, and observation semantics.",
    "relationship": "direct",
    "schema_version": 1,
    "unmatched_intent_fields": []
  },
  "rule_tool_need": {
    "description": "Report the single primary scalar observation terminal_left_tcp_to_stapler_distance: the terminal Euclidean distance in meters between left_tcp_position and stapler_position.",
    "kind": "measure",
    "reuse_first": true
  },
  "scene_need": {
    "description": "Construct or adapt the official press_stapler scene by translating the stapler initial position from the official reset pose by +0.030 m along the world y-axis; make no other scene modification. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.",
    "kind": "adapt",
    "reuse_first": true
  },
  "schema_version": 2,
  "semantic_concern": "scene_robustness.orthogonal_world_y_displacement_terminal_tcp_proximity: Translating the stapler's initial position by +0.030 m along the world y-axis will cause the policy to fail the experimental terminal predicate consisting of the official core predicate and left-TCP-to-stapler distance no greater than 0.080 m, exposing an orthogonal lateral retargeting weakness while the expert remains able to satisfy the predicate.",
  "source_query": "Does there exist a bounded, executable scene concern beyond the unchanged official press_stapler task under which this policy exposes a measured weakness? Observe the control, then let the Plan Agent invent and refine the most informative concerns from evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template, checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool actually required by each Proposal. A generated checker must preserve official success as a required conjunct and add only directly observable current-state semantics. A diagnostic Tool must remain separate from success. After a valid success, the evidence must choose a genuinely different semantic concern or an evidence-grounded boundary refinement rather than repeat the same test. The Plan Agent must propose stop as soon as a definitive failure witness has an evidence-backed diagnosis. If executable supported concerns become informationally saturated without such a witness, it must actively stop and answer only the tested scope.",
  "tool_need": {
    "description": "Report the single primary scalar observation terminal_left_tcp_to_stapler_distance: the terminal Euclidean distance in meters between left_tcp_position and stapler_position.",
    "kind": "measure",
    "reuse_first": true
  },
  "vqa_tool_need": null
}

THIN TASK ADAPTER:
{
  "asset_paths": [
    "description/objects_description/048_stapler/base0.json",
    "description/objects_description/048_stapler/base1.json",
    "description/objects_description/048_stapler/base2.json",
    "description/objects_description/048_stapler/base3.json",
    "description/objects_description/048_stapler/base4.json",
    "description/objects_description/048_stapler/base5.json",
    "description/objects_description/048_stapler/base6.json"
  ],
  "documentation_paths": [
    "description/task_instruction/press_stapler.json"
  ],
  "generation_hook_contract": {
    "checker_semantic_review": "taskgen_checker_semantic_review_v1",
    "expert_preflight": true,
    "local_regeneration_limit": 1,
    "methods": [
      "load_actors",
      "check_success"
    ],
    "render_preflight": true,
    "semantic_validation": "task_schema_contract_v2",
    "static_and_fixture_validation": true
  },
  "official_class": "press_stapler",
  "official_source": "envs/press_stapler.py",
  "schema_version": 1,
  "task_context": {
    "authority": {
      "actor_telemetry": "fresh_simulator_reset_probe",
      "official_source": "repository_source_sha256",
      "success": "official_check_success_runtime_callable"
    },
    "declared_methods": [
      "check_success",
      "load_actors",
      "play_once",
      "setup_demo"
    ],
    "official_class": "press_stapler",
    "official_source": "envs/press_stapler.py",
    "official_source_sha256": "69274cf09aa1d4ba51469621461683b5225b8f98cf7837f821e848196ce51f9c",
    "runtime_probe": {
      "action_dimension": 14,
      "actors": [
        {
          "scene_name": "048_stapler",
          "task_attribute": "stapler"
        }
      ],
      "observables": {
        "contact_events": true,
        "policy_action": true,
        "robot_tcp": {
          "left": true,
          "right": true
        },
        "simulation_clock": true
      },
      "official_check_success_callable": true,
      "official_source": "envs/press_stapler.py",
      "official_source_sha256": "69274cf09aa1d4ba51469621461683b5225b8f98cf7837f821e848196ce51f9c",
      "physics_timestep_seconds": 0.004000000189989805,
      "schema_version": 1,
      "setup_success": true,
      "task_name": "press_stapler"
    },
    "schema_origin": "runtime_probe",
    "schema_version": 1,
    "source_task_attributes": [
      "add_prohibit_area",
      "close_gripper",
      "get_gripper_actor_contact_position",
      "grasp_actor",
      "info",
      "move",
      "stage_success_tag",
      "stapler",
      "stapler_id"
    ],
    "task_name": "press_stapler",
    "task_schema": {
      "action_dimension": 14,
      "contact_focus_actor_ids": [
        "stapler"
      ],
      "physics_timestep_seconds": 0.004000000189989805,
      "probe_task_attributes": [],
      "schema_version": 1,
      "semantic_fields": [
        {
          "actor_id": "stapler",
          "name": "stapler_position",
          "source": "actor_position"
        },
        {
          "name": "left_tcp_position",
          "side": "left",
          "source": "robot_tcp_position"
        },
        {
          "name": "right_tcp_position",
          "side": "right",
          "source": "robot_tcp_position"
        }
      ],
      "semantic_roles": {},
      "success_contract": {
        "authority": "official_check_success_runtime_callable",
        "official_source_sha256": "69274cf09aa1d4ba51469621461683b5225b8f98cf7837f821e848196ce51f9c",
        "semantic_telemetry_available": true,
        "type": "official_check_success"
      },
      "task_family": "robotwin_runtime_discovered",
      "task_name": "press_stapler",
      "telemetry_observables": {
        "actor_pose_signals": [
          "stapler_position"
        ],
        "authority": "validated_task_schema_and_recorder_contract",
        "contact_events": {
          "available": true,
          "scope": "declared_contact_focus_actors",
          "signals": [
            "contact_pair",
            "physical_contact",
            "start_simulation_time_seconds",
            "end_simulation_time_seconds"
          ]
        },
        "policy_action": {
          "available": true,
          "dimension": 14,
          "signals": [
            "action.0",
            "action.1",
            "action.2",
            "action.3",
            "action.4",
            "action.5",
            "action.6",
            "action.7",
            "action.8",
            "action.9",
            "action.10",
            "action.11",
            "action.12",
            "action.13"
          ]
        },
        "robot_tcp": {
          "available_sides": [
            "left",
            "right"
          ],
          "signals": [
            "left_tcp_position",
            "right_tcp_position"
          ]
        },
        "schema_version": 1,
        "simulation_clock": {
          "available": true,
          "signals": [
            "physics_step",
            "policy_step",
            "simulation_time_seconds"
          ]
        }
      },
      "tracked_actors": [
        {
          "contact_points": [],
          "functional_points": [],
          "id": "stapler",
          "scene_name": "048_stapler",
          "task_attribute": "stapler"
        }
      ],
      "trusted_tool_profile": "runtime_actor_positions"
    },
    "taskgen_ready": true,
    "telemetry_observables": {
      "actor_pose_signals": [
        "stapler_position"
      ],
      "authority": "validated_task_schema_and_recorder_contract",
      "contact_events": {
        "available": true,
        "scope": "declared_contact_focus_actors",
        "signals": [
          "contact_pair",
          "physical_contact",
          "start_simulation_time_seconds",
          "end_simulation_time_seconds"
        ]
      },
      "policy_action": {
        "available": true,
        "dimension": 14,
        "signals": [
          "action.0",
          "action.1",
          "action.2",
          "action.3",
          "action.4",
          "action.5",
          "action.6",
          "action.7",
          "action.8",
          "action.9",
          "action.10",
          "action.11",
          "action.12",
          "action.13"
        ]
      },
      "robot_tcp": {
        "available_sides": [
          "left",
          "right"
        ],
        "signals": [
          "left_tcp_position",
          "right_tcp_position"
        ]
      },
      "schema_version": 1,
      "simulation_clock": {
        "available": true,
        "signals": [
          "physics_step",
          "policy_step",
          "simulation_time_seconds"
        ]
      }
    }
  },
  "task_name": "press_stapler",
  "task_schema": {
    "action_dimension": 14,
    "contact_focus_actor_ids": [
      "stapler"
    ],
    "physics_timestep_seconds": 0.004000000189989805,
    "probe_task_attributes": [],
    "schema_version": 1,
    "semantic_fields": [
      {
        "actor_id": "stapler",
        "name": "stapler_position",
        "source": "actor_position"
      },
      {
        "name": "left_tcp_position",
        "side": "left",
        "source": "robot_tcp_position"
      },
      {
        "name": "right_tcp_position",
        "side": "right",
        "source": "robot_tcp_position"
      }
    ],
    "semantic_roles": {},
    "success_contract": {
      "authority": "official_check_success_runtime_callable",
      "official_source_sha256": "69274cf09aa1d4ba51469621461683b5225b8f98cf7837f821e848196ce51f9c",
      "semantic_telemetry_available": true,
      "type": "official_check_success"
    },
    "task_family": "robotwin_runtime_discovered",
    "task_name": "press_stapler",
    "telemetry_observables": {
      "actor_pose_signals": [
        "stapler_position"
      ],
      "authority": "validated_task_schema_and_recorder_contract",
      "contact_events": {
        "available": true,
        "scope": "declared_contact_focus_actors",
        "signals": [
          "contact_pair",
          "physical_contact",
          "start_simulation_time_seconds",
          "end_simulation_time_seconds"
        ]
      },
      "policy_action": {
        "available": true,
        "dimension": 14,
        "signals": [
          "action.0",
          "action.1",
          "action.2",
          "action.3",
          "action.4",
          "action.5",
          "action.6",
          "action.7",
          "action.8",
          "action.9",
          "action.10",
          "action.11",
          "action.12",
          "action.13"
        ]
      },
      "robot_tcp": {
        "available_sides": [
          "left",
          "right"
        ],
        "signals": [
          "left_tcp_position",
          "right_tcp_position"
        ]
      },
      "schema_version": 1,
      "simulation_clock": {
        "available": true,
        "signals": [
          "physics_step",
          "policy_step",
          "simulation_time_seconds"
        ]
      }
    },
    "tracked_actors": [
      {
        "contact_points": [],
        "functional_points": [],
        "id": "stapler",
        "scene_name": "048_stapler",
        "task_attribute": "stapler"
      }
    ],
    "trusted_tool_profile": "runtime_actor_positions"
  }
}

READ-ONLY CURRENT-STATE FIELD ACCESS:
- stapler_position: `self.stapler.get_pose().p`
- left_tcp_position: `self.robot.get_left_tcp_pose()[:3]`
- right_tcp_position: `self.robot.get_right_tcp_pose()[:3]`
When checker_need names one of these semantic fields, use its exact expression. Do not invent a similarly named helper such as get_contact_position, and do not replace a declared actor point identity with an arbitrary PhysX collision point. Semantic field names describe evidence; they are not necessarily Python attributes. For example, do not rewrite `stapler_position` as an assumed `self.stapler` access unless that exact expression is listed above.

SIMULATOR-SPECIFIC API CONSTRAINTS:
Keep the official class identity and policy action interface. Use only assets and simulator APIs present in retrieved context. The generated initial scene must differ observably from the same-seed official scene in simulator state or rendered pixels when scene_need is non-null; when scene_need is null, preserve the official load_actors implementation exactly. When checker_need is null, preserve official check_success exactly. SAPIEN Pose.p and Pose.q values must not be modified by indexed assignment or +=/-= because those writes do not update the Pose; construct a new sapien.Pose from a copied position array and the original quaternion before passing it to create_actor. The upstream create_actor scale argument is normally replaced by asset model_data. scale_multiplier is the final/original size ratio: increase by 50% uses 1.5; reduce by 50%, or reduce to 50%, uses 0.5. Use scale_override only for a known absolute asset scale. Both opt-ins update the built mesh scale and Actor point metadata. If load_actors adds an actor that later measurement may need, also assign self.mea_telemetry_tracked_actors to a list of dicts with exactly id, task_attribute, scene_name, functional_points, contact_points, and contact_focus; task_attribute must name the public self attribute holding that actor, and contact_focus must be a boolean. Actors already listed in the TASK TELEMETRY/EXECUTION SCHEMA remain tracked automatically when their pose or instance is replaced: do not assign mea_telemetry_tracked_actors merely to repeat them. Include only entirely new actors in that list. Every new actor must have a unique simulator/contact identity distinct from every base actor: pass a unique runtime_name to create_actor when the asset modelname is reused, and declare that exact runtime get_name() value as scene_name. The asset modelname is not a unique runtime identity. Do not redeclare an actor already present in the TASK TELEMETRY/EXECUTION SCHEMA; that schema remains valid when the generated scene replaces the same public actor attribute and scene name. The initial state must not satisfy check_success; the official expert terminal state must satisfy it.

OUTPUT CONTRACT:
Return one strict JSON object with exactly two string fields, load_actors and check_success. Each field must contain one complete Python method with only self when its corresponding need is non-null. A non-null scene_need requires a changed load_actors method. A non-null checker_need requires a changed check_success method. Both JSON fields remain required for transport, but when a need is null return an empty string for that field: the runtime ignores that text and injects the exact official method before AST, fixture, render, and expert validation. A changed load_actors method must directly implement the requested scene change. Comments or an unrelated actor/pose change are not implementation evidence. load_actors cannot alter policy weights, controller or gripper precision, action noise, latency, or inference. Those require an explicit runtime intervention and must not be simulated by relabelling a scene change. For a pose change, reuse the official pose construction and alter only the Proposal-named component. For a fixed-angle rotation, the AST contract does not admit np.sin, np.cos, np.deg2rad, or math trigonometry; emit a normalized quaternion as numeric literals instead. When the candidate leaves the perturbation magnitude open, derive the smallest measurable change from the retrieved spawn or workspace range, stay away from its boundary, and keep every task-critical actor fully inside the unchanged camera view. Actors already present in the TASK TELEMETRY/EXECUTION SCHEMA are tracked automatically even when their pose or instance is replaced. Do not assign self.mea_telemetry_tracked_actors merely to repeat one of those base actors. Do not add helper state beyond self assignments already present in the official method and new actor handles/telemetry; in particular, do not cache initial poses, heights, thresholds, or flags on self. Compute checker values from current simulator state and literal or Query-specified thresholds. check_success cannot read the completed trajectory or invoke a derived Rule metric such as trajectory deviation, smoothness, jerk, path length, or minimum clearance. Leave that scalar observation to ToolGen and never invent calculate_* or measure_* helper methods. Implement every checker_need relation literally. Do not replace an exact relation with a correlated proxy: a closed gripper is not target contact, height is not placement, and sequential contacts are not simultaneous contacts. If the requested predicate is not available from current simulator state or is false in the supplied expert terminal fixture, let validation reject the candidate rather than weakening its meaning. When checker_need composes the official task goal with an additional experimental condition, call self.mea_official_check_success() directly and use its result as a required conjunct; do not copy or reimplement the official predicate. This preserves the official core without claiming that the extended checker is official-equivalent. For a simulator-verifiable robot-contact condition, inspect self.scene.get_contacts(). A SAPIEN PhysxContact exposes bodies, not actor0/actor1; each body.entity is the scene entity, while a RoboTwin Actor wrapper exposes its scene entity as .actor. The RoboTwin Robot wrapper has no get_links() method; when robot link entities are needed, combine self.robot.left_entity.get_links() and self.robot.right_entity.get_links(). When the Proposal specifically requires left/right gripper contact, use tuple(item[0].child_link for item in self.robot.left_gripper) and the corresponding right_gripper expression; all arm links are not equivalent to gripper links; do not invent a helper such as self.check_contact unless that exact method appears in the retrieved official source. Build checker-local entity collections with `+` or tuple literals; do not call `.append()` or `.extend()`. These APIs are read-only and must not mutate simulator state. self.mea_telemetry_tracked_actors is the metadata exception. Assign it only when adding an entirely new actor, include only new actors, and give every entry exactly id, task_attribute, scene_name, functional_points, contact_points, and a boolean contact_focus. When adding a distractor or obstacle, inspect the retrieved asset scale/collision geometry and place it initially disjoint from the target and the official expert contact path. A small center offset is not sufficient when reused asset extents overlap. Do not return Markdown, a template id, or an explanation. When the retrieved API supports scale_multiplier, it is the final-size/original-size ratio: increasing size by 50% uses 1.5, while reducing size by 50% (or to 50%) uses 0.5.

README.AGENT CONTEXT:
# TaskGen output rules

- Return complete methods, never patches or prose.
- Change only fields authorized by the validated Proposal.
- Use only actors, poses, telemetry, thresholds, and assets declared by the
  supplied TaskContext/TaskSchema or retrieved official source. If that context
  is insufficient, do not fabricate the missing field.
- Preserve actor identity, collision behavior, random-call order, and required
  telemetry names.
- For a pose change, reuse the official pose construction and alter only the
  Proposal-named component. If the Proposal gives only a direction, derive the
  smallest measurable change from the retrieved spawn/workspace range, stay
  away from its boundary, and keep every key actor fully in the unchanged
  camera view.
- Do not import or access files, network, environment variables, processes,
  dynamic execution, dunder attributes, or `super()`.
- Use wrapper-provided `np`, `sapien`, `create_actor`, `create_box`, and
  `rand_pose`.
- The ordinary scene-only route must not replace `check_success()`.
- Actors already listed in the task telemetry/execution schema remain tracked
  when a generated scene moves or replaces the same public actor. Do not
  redeclare them in `self.mea_telemetry_tracked_actors`.
- Every newly added actor must have a stable runtime name distinct from all
  base-scene actor names. When reusing an existing asset model, pass a unique
  `runtime_name` to `create_actor()` and make telemetry `scene_name` exactly
  equal to it; the asset `modelname` is not the actor identity. Declare only
  new actors, with exactly `id`, `task_attribute`, `scene_name`,
  `functional_points`, `contact_points`, and boolean `contact_focus`.
- When a Proposal requests both scene and checker, generate `load_actors()` and
  `check_success()` together. The checker is experimental and must never be
  relabeled as official success.
- Every added checker conjunct must be false initially and true in the supplied
  official-expert terminal state. Do not require instantaneous PhysX contact
  when the official terminal state may release contact; use the repair
  evidence's current actor state or another Proposal-consistent predicate.
- Implement the exact checker relation in the Proposal. Never replace it with
  a correlated proxy: gripper closure is not target contact, height is not
  placement, and sequential events are not simultaneous events. If the exact
  predicate is unavailable from current simulator state or cannot pass the
  supplied expert fixture, leave the candidate rejected/unsupported rather
  than weakening its semantics during repair.
- A generated challenge must preserve at least one feasible official action
  path to every required contact or functional point. A fixed center offset is
  not proof of approach clearance, and `add_prohibit_area()` is not a robot
  reachability check. Never relax an official success threshold to compensate
  for an expert-solvability failure.
- For robot-contact checks, `PhysxContact` uses `bodies[*].entity`; the
  RoboTwin Robot wrapper has no `get_links()`, so use
  `left_entity.get_links()` plus `right_entity.get_links()`.
- Keep checker-local collections immutable: build link/entity collections with
  `+` or tuple literals rather than `.append()`/`.extend()`. Those mutators are
  outside the generated-checker API contract and can waste the one repair
  attempt before simulator validation.
- Before adding an obstacle or distractor, use the retrieved asset scale and
  collision geometry to keep it initially disjoint from the target and from
  the official expert contact path. Reusing the target asset at a small center
  offset is invalid when their collision extents overlap.
- For lift checkers, derive bounds from TaskSchema or initial actor state:
  near-zero world z does not mean an actor stayed unlifted on raised support.
- A paper-claim run requires compile/semantic fixtures, render, expert
  solvability, and the generated scene/checker to remain bound to one artifact.

RETRIEVED ROBOTWIN API AND TASK CONTEXT:
OFFICIAL BASE TASK METHODS:
```python
def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.2, 0.2],
            ylim=[-0.1, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi, 0],
        )

        self.stapler_id = np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0]
        self.stapler = create_actor(self,
                                    pose=rand_pos,
                                    modelname="048_stapler",
                                    convex=True,
                                    model_id=self.stapler_id,
                                    is_static=True)

        self.add_prohibit_area(self.stapler, padding=0.05)

def check_success(self):
        if self.stage_success_tag:
            return True
        stapler_pose = self.stapler.get_contact_point(2)[:3]
        positions = self.get_gripper_actor_contact_position("048_stapler")
        eps = [0.03, 0.03]
        for position in positions:
            if (np.all(np.abs(position[:2] - stapler_pose[:2]) < eps) and abs(position[2] - stapler_pose[2]) < 0.03):
                self.stage_success_tag = True
                return True
        return False

def play_once(self):
        # Determine which arm to use based on stapler's position (left if negative x, right otherwise)
        arm_tag = ArmTag("left" if self.stapler.get_pose().p[0] < 0 else "right")

        # Move arm to the overhead position of the stapler and close the gripper
        self.move(self.grasp_actor(self.stapler, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0.1, contact_point_id=2))
        self.move(self.close_gripper(arm_tag=arm_tag))

        # Move the stapler down slightly to press it
        self.move(
            self.grasp_actor(self.stapler, arm_tag=arm_tag, pre_grasp_dis=0.02, grasp_dis=0.02, contact_point_id=2))

        self.info["info"] = {"{A}": f"048_stapler/base{self.stapler_id}", "{a}": str(arm_tag)}
        return self.info
```

TASK TELEMETRY/EXECUTION SCHEMA:
{
  "action_dimension": 14,
  "contact_focus_actor_ids": [
    "stapler"
  ],
  "physics_timestep_seconds": 0.004000000189989805,
  "probe_task_attributes": [],
  "schema_version": 1,
  "semantic_fields": [
    {
      "actor_id": "stapler",
      "name": "stapler_position",
      "source": "actor_position"
    },
    {
      "name": "left_tcp_position",
      "side": "left",
      "source": "robot_tcp_position"
    },
    {
      "name": "right_tcp_position",
      "side": "right",
      "source": "robot_tcp_position"
    }
  ],
  "semantic_roles": {},
  "success_contract": {
    "authority": "official_check_success_runtime_callable",
    "official_source_sha256": "69274cf09aa1d4ba51469621461683b5225b8f98cf7837f821e848196ce51f9c",
    "semantic_telemetry_available": true,
    "type": "official_check_success"
  },
  "task_family": "robotwin_runtime_discovered",
  "task_name": "press_stapler",
  "telemetry_observables": {
    "actor_pose_signals": [
      "stapler_position"
    ],
    "authority": "validated_task_schema_and_recorder_contract",
    "contact_events": {
      "available": true,
      "scope": "declared_contact_focus_actors",
      "signals": [
        "contact_pair",
        "physical_contact",
        "start_simulation_time_seconds",
        "end_simulation_time_seconds"
      ]
    },
    "policy_action": {
      "available": true,
      "dimension": 14,
      "signals": [
        "action.0",
        "action.1",
        "action.2",
        "action.3",
        "action.4",
        "action.5",
        "action.6",
        "action.7",
        "action.8",
        "action.9",
        "action.10",
        "action.11",
        "action.12",
        "action.13"
      ]
    },
    "robot_tcp": {
      "available_sides": [
        "left",
        "right"
      ],
      "signals": [
        "left_tcp_position",
        "right_tcp_position"
      ]
    },
    "schema_version": 1,
    "simulation_clock": {
      "available": true,
      "signals": [
        "physics_step",
        "policy_step",
        "simulation_time_seconds"
      ]
    }
  },
  "tracked_actors": [
    {
      "contact_points": [],
      "functional_points": [],
      "id": "stapler",
      "scene_name": "048_stapler",
      "task_attribute": "stapler"
    }
  ],
  "trusted_tool_profile": "runtime_actor_positions"
}

TASK CONTEXT AUTHORITY:
{
  "authority": {
    "actor_telemetry": "fresh_simulator_reset_probe",
    "official_source": "repository_source_sha256",
    "success": "official_check_success_runtime_callable"
  },
  "declared_methods": [
    "check_success",
    "load_actors",
    "play_once",
    "setup_demo"
  ],
  "official_class": "press_stapler",
  "official_source": "envs/press_stapler.py",
  "official_source_sha256": "69274cf09aa1d4ba51469621461683b5225b8f98cf7837f821e848196ce51f9c",
  "runtime_probe": {
    "action_dimension": 14,
    "actors": [
      {
        "scene_name": "048_stapler",
        "task_attribute": "stapler"
      }
    ],
    "observables": {
      "contact_events": true,
      "policy_action": true,
      "robot_tcp": {
        "left": true,
        "right": true
      },
      "simulation_clock": true
    },
    "official_check_success_callable": true,
    "official_source": "envs/press_stapler.py",
    "official_source_sha256": "69274cf09aa1d4ba51469621461683b5225b8f98cf7837f821e848196ce51f9c",
    "physics_timestep_seconds": 0.004000000189989805,
    "schema_version": 1,
    "setup_success": true,
    "task_name": "press_stapler"
  },
  "schema_origin": "runtime_probe",
  "schema_version": 1,
  "source_task_attributes": [
    "add_prohibit_area",
    "close_gripper",
    "get_gripper_actor_contact_position",
    "grasp_actor",
    "info",
    "move",
    "stage_success_tag",
    "stapler",
    "stapler_id"
  ],
  "task_name": "press_stapler",
  "task_schema": {
    "action_dimension": 14,
    "contact_focus_actor_ids": [
      "stapler"
    ],
    "physics_timestep_seconds": 0.004000000189989805,
    "probe_task_attributes": [],
    "schema_version": 1,
    "semantic_fields": [
      {
        "actor_id": "stapler",
        "name": "stapler_position",
        "source": "actor_position"
      },
      {
        "name": "left_tcp_position",
        "side": "left",
        "source": "robot_tcp_position"
      },
      {
        "name": "right_tcp_position",
        "side": "right",
        "source": "robot_tcp_position"
      }
    ],
    "semantic_roles": {},
    "success_contract": {
      "authority": "official_check_success_runtime_callable",
      "official_source_sha256": "69274cf09aa1d4ba51469621461683b5225b8f98cf7837f821e848196ce51f9c",
      "semantic_telemetry_available": true,
      "type": "official_check_success"
    },
    "task_family": "robotwin_runtime_discovered",
    "task_name": "press_stapler",
    "telemetry_observables": {
      "actor_pose_signals": [
        "stapler_position"
      ],
      "authority": "validated_task_schema_and_recorder_contract",
      "contact_events": {
        "available": true,
        "scope": "declared_contact_focus_actors",
        "signals": [
          "contact_pair",
          "physical_contact",
          "start_simulation_time_seconds",
          "end_simulation_time_seconds"
        ]
      },
      "policy_action": {
        "available": true,
        "dimension": 14,
        "signals": [
          "action.0",
          "action.1",
          "action.2",
          "action.3",
          "action.4",
          "action.5",
          "action.6",
          "action.7",
          "action.8",
          "action.9",
          "action.10",
          "action.11",
          "action.12",
          "action.13"
        ]
      },
      "robot_tcp": {
        "available_sides": [
          "left",
          "right"
        ],
        "signals": [
          "left_tcp_position",
          "right_tcp_position"
        ]
      },
      "schema_version": 1,
      "simulation_clock": {
        "available": true,
        "signals": [
          "physics_step",
          "policy_step",
          "simulation_time_seconds"
        ]
      }
    },
    "tracked_actors": [
      {
        "contact_points": [],
        "functional_points": [],
        "id": "stapler",
        "scene_name": "048_stapler",
        "task_attribute": "stapler"
      }
    ],
    "trusted_tool_profile": "runtime_actor_positions"
  },
  "taskgen_ready": true,
  "telemetry_observables": {
    "actor_pose_signals": [
      "stapler_position"
    ],
    "authority": "validated_task_schema_and_recorder_contract",
    "contact_events": {
      "available": true,
      "scope": "declared_contact_focus_actors",
      "signals": [
        "contact_pair",
        "physical_contact",
        "start_simulation_time_seconds",
        "end_simulation_time_seconds"
      ]
    },
    "policy_action": {
      "available": true,
      "dimension": 14,
      "signals": [
        "action.0",
        "action.1",
        "action.2",
        "action.3",
        "action.4",
        "action.5",
        "action.6",
        "action.7",
        "action.8",
        "action.9",
        "action.10",
        "action.11",
        "action.12",
        "action.13"
      ]
    },
    "robot_tcp": {
      "available_sides": [
        "left",
        "right"
      ],
      "signals": [
        "left_tcp_position",
        "right_tcp_position"
      ]
    },
    "schema_version": 1,
    "simulation_clock": {
      "available": true,
      "signals": [
        "physics_step",
        "policy_step",
        "simulation_time_seconds"
      ]
    }
  }
}

DOCUMENTATION `description/task_instruction/press_stapler.json`:
{
  "full_description": "Use one arm to press the stapler.",
  "schema": "{A} notifies the stapler, {a} notifies the arm to press the stapler",
  "preference": "num of words should not exceed 15",
  "seen": [
    "Push {A} using one arm",
    "Push down on {A} to staple",
    "Press {A} until it works",
    "Apply pressure to {A} firmly",
    "Push {A} down completely",
    "Lower {A} using an arm",
    "Use an arm to press {A}",
    "Push the top of {A} down",
    "Apply force to {A} with {a}",
    "Press {A} firmly to staple",
    "Press on {A} using {a}.",
    "Push {A} down firmly.",
    "Press {A} to complete the task.",
    "Apply force to {A} to staple.",
    "Use one arm to press {A}.",
    "Push the stapler {A} with {a}.",
    "Push {A} until it staples.",
    "Push down on {A} firmly.",
    "Apply pressure to {A} with {a}.",
    "Firmly push down on {A}.",
    "Firmly press {A} using an arm.",
    "Push down on the stapler {A}.",
    "Apply pressure on {A} until it works.",
    "Press on {A} to activate it.",
    "Use your {a} to press the stapler {A}.",
    "Firmly push {A} using {a} to operate it.",
    "Push the stapler {A} to activate it.",
    "Press down on {A} until it functions.",
    "Press the stapler {A} using an arm {a}.",
    "Apply enough force to {A} to press it.",
    "Place pressure on {A} to staple.",
    "Use one arm {a} to press {A}.",
    "Push down on {A} to staple papers.",
    "Apply pressure to {A} using one arm {a}.",
    "Press {A} to staple the sheets together.",
    "Use your arm {a} to push {A} down.",
    "Push down firmly on {A} to complete.",
    "Apply one arm {a} to press {A} firmly.",
    "Press {A} downward using your arm {a}.",
    "Firmly press down on {A} to staple.",
    "Push down on the stapler {A}.",
    "Apply pressure to {A} with {a}.",
    "Push the stapler {A} to operate it.",
    "Lower {a} onto the stapler {A}.",
    "Push down hard on {A} with {a}.",
    "Press down on the stapler {A} firmly.",
    "Operate {A} by pressing it with {a}.",
    "Push down forcefully on the stapler {A}.",
    "Engage {A} by using {a} to press it.",
    "Simply press the stapler {A} downward."
  ],
  "unseen": [
    "Press down on {A} with {a}",
    "Use {a} to press on {A}",
    "Press down on {A} with {a}.",
    "Use {a} to press {A}.",
    "Press down on {A} firmly using {a}.",
    "Use {a} to press the stapler {A}.",
    "Push {A} down with one arm {a}.",
    "Press {A} firmly using your arm.",
    "Press the stapler {A} with the arm {a}.",
    "Use {a} to press down on {A}."
  ]
}

AVAILABLE ASSETS:
[
  {
    "path": "description/objects_description/048_stapler/base0.json",
    "sha256": "f2b4afc362667e173eb035ffa33e493982745a967a28defddc6efcca443dcd6a",
    "size_bytes": 705
  },
  {
    "path": "description/objects_description/048_stapler/base1.json",
    "sha256": "2f7cec14343cb506b5a41c92af1a3de4c2b731d93b46878c301b08284452f2d9",
    "size_bytes": 670
  },
  {
    "path": "description/objects_description/048_stapler/base2.json",
    "sha256": "c7bf28ab91c034f655d3959f027c2e5835fe4ac7c8882ef02c11fa2965d75952",
    "size_bytes": 651
  },
  {
    "path": "description/objects_description/048_stapler/base3.json",
    "sha256": "e408234f817dd227d6cbbdbf30c1596d2b9fa030fdf0da67c409d4fb7ed271c8",
    "size_bytes": 622
  },
  {
    "path": "description/objects_description/048_stapler/base4.json",
    "sha256": "87977541c35f2ec52e6101e9cf2f142902f701e6d70c03eb77dc09c69c681fd4",
    "size_bytes": 665
  },
  {
    "path": "description/objects_description/048_stapler/base5.json",
    "sha256": "9112de96398164b8ada06dd57d27f83f674e5891c5da2abe93a6757eceacf6a7",
    "size_bytes": 645
  },
  {
    "path": "description/objects_description/048_stapler/base6.json",
    "sha256": "aecc597aa26e1e52a08ea2c79691d8bfa030335658ea9f0e3e05ea01f32af6ab",
    "size_bytes": 704
  }
]
