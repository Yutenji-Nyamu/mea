Generate one RoboTwin experiment from the open Query-derived candidate below. Retrieve semantics from the official base program, but implement the requested scene and checker rather than selecting a catalog template.

EXPERIMENT CANDIDATE:
{
  "base_task": "grab_roller",
  "candidate_id": "dynamic.grab.roller.scene.robustness.roller.translation.terminal.tcp.alignment.translating.the.roller.by.exactly.0.05.m.along.the.world.x.axis.will.expose.a.terminal.tcp.alignment.weakness.causing.the.combined.experimental.checker.to.fail.despite.the.successful.official.control.result.f7e3639e20ff",
  "checker_need": {
    "description": "Evaluate the boolean conjunction of the official goal, left TCP distance to the left roller contact point being at most 0.025 m, and right TCP distance to the right roller contact point being at most 0.025 m, using terminal current simulator point positions only.",
    "kind": "generate",
    "reuse_first": true
  },
  "evaluation_intent": {
    "hypothesis": "Translating the roller by exactly 0.05 m along the world x-axis will expose a terminal TCP alignment weakness, causing the combined experimental checker to fail despite the successful official-control result.",
    "intent_id": "intent.78fbce792af89034",
    "original_concern": "scene_robustness.roller_translation.terminal_tcp_alignment",
    "preserved_conditions": [
      "task identity",
      "policy checkpoint",
      "official core predicate as a required conjunct"
    ],
    "requested_change": "Translate the manipulated roller exactly 0.05 m along the world x-axis from its official-scene position while retaining an expert-solvable arrangement.",
    "required_observation": "Evaluate the boolean conjunction of the official goal, left TCP distance to the left roller contact point being at most 0.025 m, and right TCP distance to the right roller contact point being at most 0.025 m, using terminal current simulator point positions only. Report the terminal maximum of the left TCP-to-left-contact and right TCP-to-right-contact distances as one scalar. Report the peak over the rollout of the larger of the left TCP-to-left-contact and right TCP-to-right-contact distances as one diagnostic trajectory scalar.",
    "schema_version": 1,
    "source_query": "Relative to the official grab task, does there exist a newly generated executable scene challenge that exposes a terminal alignment weakness in this policy? After observing official-control evidence, let the Plan Agent choose the most informative supported scene change without an aspect or template from me. To avoid a trivial perturbation, the chosen geometric scene change must displace the manipulated roller by at least 0.05 m while remaining expert-solvable; the Plan Agent chooses the axis and exact magnitude. Define experimental success as the official task goal AND both terminal TCPs being within 0.025 m of their corresponding roller contact points, using only current simulator point positions; do not require episode history, accumulated contact, or a trajectory-derived success threshold. Independently report one scalar metric computed from the rollout trajectory that diagnoses the chosen hypothesis, but treat that scalar strictly as diagnostic evidence and never as the terminal success outcome."
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
    "description": "Report the terminal maximum of the left TCP-to-left-contact and right TCP-to-right-contact distances as one scalar.",
    "kind": "measure",
    "reuse_first": true
  },
  "scene_need": {
    "description": "Generate an executable expert-solvable scene by translating the roller exactly 0.05 m along the world x-axis from its official-scene position. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.",
    "kind": "adapt",
    "reuse_first": true
  },
  "schema_version": 2,
  "semantic_concern": "scene_robustness.roller_translation.terminal_tcp_alignment: Translating the roller by exactly 0.05 m along the world x-axis will expose a terminal TCP alignment weakness, causing the combined experimental checker to fail despite the successful official-control result.",
  "source_query": "Relative to the official grab task, does there exist a newly generated executable scene challenge that exposes a terminal alignment weakness in this policy? After observing official-control evidence, let the Plan Agent choose the most informative supported scene change without an aspect or template from me. To avoid a trivial perturbation, the chosen geometric scene change must displace the manipulated roller by at least 0.05 m while remaining expert-solvable; the Plan Agent chooses the axis and exact magnitude. Define experimental success as the official task goal AND both terminal TCPs being within 0.025 m of their corresponding roller contact points, using only current simulator point positions; do not require episode history, accumulated contact, or a trajectory-derived success threshold. Independently report one scalar metric computed from the rollout trajectory that diagnoses the chosen hypothesis, but treat that scalar strictly as diagnostic evidence and never as the terminal success outcome.",
  "tool_need": {
    "description": "Report the terminal maximum of the left TCP-to-left-contact and right TCP-to-right-contact distances as one scalar.",
    "kind": "measure",
    "reuse_first": true
  },
  "vqa_tool_need": {
    "description": "Report the peak over the rollout of the larger of the left TCP-to-left-contact and right TCP-to-right-contact distances as one diagnostic trajectory scalar.",
    "kind": "vqa",
    "reuse_first": true
  }
}

THIN TASK ADAPTER:
{
  "asset_paths": [
    "description/objects_description/102_roller/base0.json",
    "description/objects_description/102_roller/base1.json",
    "description/objects_description/102_roller/base2.json"
  ],
  "documentation_paths": [
    "description/task_instruction/grab_roller.json"
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
  "official_class": "grab_roller",
  "official_source": "envs/grab_roller.py",
  "schema_version": 1,
  "task_context": {
    "authority": {
      "actor_telemetry": "reviewed_task_schema",
      "official_source": "repository_source_sha256",
      "success": "reviewed_task_schema"
    },
    "declared_methods": [
      "check_success",
      "load_actors",
      "play_once",
      "setup_demo"
    ],
    "official_class": "grab_roller",
    "official_source": "envs/grab_roller.py",
    "official_source_sha256": "f0e1864199605acf59cf8982dded05129d873bb3934f200e47abafd918d41c89",
    "runtime_probe": null,
    "schema_origin": "reviewed_task_schema",
    "schema_version": 1,
    "source_task_attributes": [
      "add_prohibit_area",
      "grasp_actor",
      "info",
      "is_left_gripper_close",
      "is_right_gripper_close",
      "model_id",
      "move",
      "move_by_displacement",
      "roller"
    ],
    "task_name": "grab_roller",
    "task_schema": {
      "action_dimension": 14,
      "contact_focus_actor_ids": [
        "roller"
      ],
      "physics_timestep_seconds": 0.004,
      "schema_version": 1,
      "semantic_fields": [
        {
          "actor_id": "roller",
          "name": "roller_position",
          "source": "actor_position"
        },
        {
          "actor_id": "roller",
          "name": "roller_left_contact_position",
          "point_id": 0,
          "source": "actor_contact_position"
        },
        {
          "actor_id": "roller",
          "name": "roller_right_contact_position",
          "point_id": 1,
          "source": "actor_contact_position"
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
      "semantic_roles": {
        "left_target_contact_position": "roller_left_contact_position",
        "left_tcp_position": "left_tcp_position",
        "manipulated_object_position": "roller_position",
        "right_target_contact_position": "roller_right_contact_position",
        "right_tcp_position": "right_tcp_position"
      },
      "success_contract": {
        "minimum_height_m": 0.8,
        "requires_left_gripper_closed": true,
        "requires_right_gripper_closed": true,
        "target_actor_id": "roller",
        "type": "official_check_success"
      },
      "task_family": "dual_arm_lift",
      "task_name": "grab_roller",
      "tracked_actors": [
        {
          "contact_points": [
            0,
            1
          ],
          "functional_points": [],
          "id": "roller",
          "scene_name": "102_roller",
          "task_attribute": "roller"
        }
      ],
      "trusted_tool_profile": "generic_success"
    },
    "taskgen_ready": true,
    "telemetry_observables": {
      "actor_pose_signals": [
        "roller_position",
        "roller_left_contact_position",
        "roller_right_contact_position"
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
  "task_name": "grab_roller",
  "task_schema": {
    "action_dimension": 14,
    "contact_focus_actor_ids": [
      "roller"
    ],
    "physics_timestep_seconds": 0.004,
    "schema_version": 1,
    "semantic_fields": [
      {
        "actor_id": "roller",
        "name": "roller_position",
        "source": "actor_position"
      },
      {
        "actor_id": "roller",
        "name": "roller_left_contact_position",
        "point_id": 0,
        "source": "actor_contact_position"
      },
      {
        "actor_id": "roller",
        "name": "roller_right_contact_position",
        "point_id": 1,
        "source": "actor_contact_position"
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
    "semantic_roles": {
      "left_target_contact_position": "roller_left_contact_position",
      "left_tcp_position": "left_tcp_position",
      "manipulated_object_position": "roller_position",
      "right_target_contact_position": "roller_right_contact_position",
      "right_tcp_position": "right_tcp_position"
    },
    "success_contract": {
      "minimum_height_m": 0.8,
      "requires_left_gripper_closed": true,
      "requires_right_gripper_closed": true,
      "target_actor_id": "roller",
      "type": "official_check_success"
    },
    "task_family": "dual_arm_lift",
    "task_name": "grab_roller",
    "tracked_actors": [
      {
        "contact_points": [
          0,
          1
        ],
        "functional_points": [],
        "id": "roller",
        "scene_name": "102_roller",
        "task_attribute": "roller"
      }
    ],
    "trusted_tool_profile": "generic_success"
  }
}

READ-ONLY CURRENT-STATE FIELD ACCESS:
- roller_position: `self.roller.get_pose().p`
- roller_left_contact_position: `self.roller.get_contact_point(0, "pose").p`
- roller_right_contact_position: `self.roller.get_contact_point(1, "pose").p`
- left_tcp_position: `self.robot.get_left_tcp_pose()[:3]`
- right_tcp_position: `self.robot.get_right_tcp_pose()[:3]`
When checker_need names one of these semantic fields, use its exact expression. Do not invent a similarly named helper such as get_contact_position, and do not replace a declared actor point identity with an arbitrary PhysX collision point.

SIMULATOR-SPECIFIC API CONSTRAINTS:
Keep the official class identity and policy action interface. Use only assets and simulator APIs present in retrieved context. The generated initial scene must differ observably from the same-seed official scene in simulator state or rendered pixels when scene_need is non-null; when scene_need is null, preserve the official load_actors implementation exactly. When checker_need is null, preserve official check_success exactly. SAPIEN Pose.p and Pose.q values must not be modified by indexed assignment or +=/-= because those writes do not update the Pose; construct a new sapien.Pose from a copied position array and the original quaternion before passing it to create_actor. The upstream create_actor scale argument is normally replaced by asset model_data. scale_multiplier is the final/original size ratio: increase by 50% uses 1.5; reduce by 50%, or reduce to 50%, uses 0.5. Use scale_override only for a known absolute asset scale. Both opt-ins update the built mesh scale and Actor point metadata. If load_actors adds an actor that later measurement may need, also assign self.mea_telemetry_tracked_actors to a list of dicts with exactly id, task_attribute, scene_name, functional_points, contact_points, and contact_focus; task_attribute must name the public self attribute holding that actor, and contact_focus must be a boolean. Actors already listed in the TASK TELEMETRY/EXECUTION SCHEMA remain tracked automatically when their pose or instance is replaced: do not assign mea_telemetry_tracked_actors merely to repeat them. Include only entirely new actors in that list. Every new actor must have a unique simulator/contact identity distinct from every base actor: pass a unique runtime_name to create_actor when the asset modelname is reused, and declare that exact runtime get_name() value as scene_name. The asset modelname is not a unique runtime identity. Do not redeclare an actor already present in the TASK TELEMETRY/EXECUTION SCHEMA; that schema remains valid when the generated scene replaces the same public actor attribute and scene name. The initial state must not satisfy check_success; the official expert terminal state must satisfy it.

OUTPUT CONTRACT:
Return one strict JSON object with exactly two string fields, load_actors and check_success. Each field must contain one complete Python method with only self when its corresponding need is non-null. A non-null scene_need requires a changed load_actors method. A non-null checker_need requires a changed check_success method. Both JSON fields remain required for transport, but when a need is null return an empty string for that field: the runtime ignores that text and injects the exact official method before AST, fixture, render, and expert validation. A changed load_actors method must directly implement the requested scene change. Comments or an unrelated actor/pose change are not implementation evidence. load_actors cannot alter policy weights, controller or gripper precision, action noise, latency, or inference. Those require an explicit runtime intervention and must not be simulated by relabelling a scene change. For a pose change, reuse the official pose construction and alter only the Proposal-named component. When the candidate leaves the perturbation magnitude open, derive the smallest measurable change from the retrieved spawn or workspace range, stay away from its boundary, and keep every task-critical actor fully inside the unchanged camera view. Actors already present in the TASK TELEMETRY/EXECUTION SCHEMA are tracked automatically even when their pose or instance is replaced. Do not assign self.mea_telemetry_tracked_actors merely to repeat one of those base actors. Do not add helper state beyond self assignments already present in the official method and new actor handles/telemetry; in particular, do not cache initial poses, heights, thresholds, or flags on self. Compute checker values from current simulator state and literal or Query-specified thresholds. check_success cannot read the completed trajectory or invoke a derived Rule metric such as trajectory deviation, smoothness, jerk, path length, or minimum clearance. Leave that scalar observation to ToolGen and never invent calculate_* or measure_* helper methods. Implement every checker_need relation literally. Do not replace an exact relation with a correlated proxy: a closed gripper is not target contact, height is not placement, and sequential contacts are not simultaneous contacts. If the requested predicate is not available from current simulator state or is false in the supplied expert terminal fixture, let validation reject the candidate rather than weakening its meaning. When checker_need composes the official task goal with an additional experimental condition, call self.mea_official_check_success() directly and use its result as a required conjunct; do not copy or reimplement the official predicate. This preserves the official core without claiming that the extended checker is official-equivalent. For a simulator-verifiable robot-contact condition, inspect self.scene.get_contacts(). A SAPIEN PhysxContact exposes bodies, not actor0/actor1; each body.entity is the scene entity, while a RoboTwin Actor wrapper exposes its scene entity as .actor. The RoboTwin Robot wrapper has no get_links() method; when robot link entities are needed, combine self.robot.left_entity.get_links() and self.robot.right_entity.get_links(). When the Proposal specifically requires left/right gripper contact, use tuple(item[0].child_link for item in self.robot.left_gripper) and the corresponding right_gripper expression; all arm links are not equivalent to gripper links; do not invent a helper such as self.check_contact unless that exact method appears in the retrieved official source. Build checker-local entity collections with `+` or tuple literals; do not call `.append()` or `.extend()`. These APIs are read-only and must not mutate simulator state. self.mea_telemetry_tracked_actors is the metadata exception. Assign it only when adding an entirely new actor, include only new actors, and give every entry exactly id, task_attribute, scene_name, functional_points, contact_points, and a boolean contact_focus. When adding a distractor or obstacle, inspect the retrieved asset scale/collision geometry and place it initially disjoint from the target and the official expert contact path. A small center offset is not sufficient when reused asset extents overlap. Do not return Markdown, a template id, or an explanation. When the retrieved API supports scale_multiplier, it is the final-size/original-size ratio: increasing size by 50% uses 1.5, while reducing size by 50% (or to 50%) uses 0.5.

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
        ori_qpos = [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0, 0, 0.707, 0.707]]
        self.model_id = np.random.choice([0, 2], 1)[0]
        rand_pos = rand_pose(
            xlim=[-0.15, 0.15],
            ylim=[-0.25, -0.05],
            qpos=ori_qpos[self.model_id],
            rotate_rand=True,
            rotate_lim=[0, 0.8, 0],
        )
        self.roller = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="102_roller",
            convex=True,
            model_id=self.model_id,
        )

        self.add_prohibit_area(self.roller, padding=0.1)

def check_success(self):
        roller_pose = self.roller.get_pose().p
        return (self.is_left_gripper_close() and self.is_right_gripper_close() and roller_pose[2] > 0.8)

def play_once(self):
        # Initialize arm tags for left and right arms
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")

        # Grasp the roller with both arms simultaneously at different contact points
        self.move(
            self.grasp_actor(self.roller, left_arm_tag, pre_grasp_dis=0.08, contact_point_id=0),
            self.grasp_actor(self.roller, right_arm_tag, pre_grasp_dis=0.08, contact_point_id=1),
        )

        # Lift the roller to height 0.85 by moving both arms upward simultaneously
        self.move(
            self.move_by_displacement(left_arm_tag, z=0.85 - self.roller.get_pose().p[2]),
            self.move_by_displacement(right_arm_tag, z=0.85 - self.roller.get_pose().p[2]),
        )

        # Record information about the roller in the info dictionary
        self.info["info"] = {"{A}": f"102_roller/base{self.model_id}"}
        return self.info
```

TASK TELEMETRY/EXECUTION SCHEMA:
{
  "action_dimension": 14,
  "contact_focus_actor_ids": [
    "roller"
  ],
  "physics_timestep_seconds": 0.004,
  "schema_version": 1,
  "semantic_fields": [
    {
      "actor_id": "roller",
      "name": "roller_position",
      "source": "actor_position"
    },
    {
      "actor_id": "roller",
      "name": "roller_left_contact_position",
      "point_id": 0,
      "source": "actor_contact_position"
    },
    {
      "actor_id": "roller",
      "name": "roller_right_contact_position",
      "point_id": 1,
      "source": "actor_contact_position"
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
  "semantic_roles": {
    "left_target_contact_position": "roller_left_contact_position",
    "left_tcp_position": "left_tcp_position",
    "manipulated_object_position": "roller_position",
    "right_target_contact_position": "roller_right_contact_position",
    "right_tcp_position": "right_tcp_position"
  },
  "success_contract": {
    "minimum_height_m": 0.8,
    "requires_left_gripper_closed": true,
    "requires_right_gripper_closed": true,
    "target_actor_id": "roller",
    "type": "official_check_success"
  },
  "task_family": "dual_arm_lift",
  "task_name": "grab_roller",
  "tracked_actors": [
    {
      "contact_points": [
        0,
        1
      ],
      "functional_points": [],
      "id": "roller",
      "scene_name": "102_roller",
      "task_attribute": "roller"
    }
  ],
  "trusted_tool_profile": "generic_success"
}

TASK CONTEXT AUTHORITY:
{
  "authority": {
    "actor_telemetry": "reviewed_task_schema",
    "official_source": "repository_source_sha256",
    "success": "reviewed_task_schema"
  },
  "declared_methods": [
    "check_success",
    "load_actors",
    "play_once",
    "setup_demo"
  ],
  "official_class": "grab_roller",
  "official_source": "envs/grab_roller.py",
  "official_source_sha256": "f0e1864199605acf59cf8982dded05129d873bb3934f200e47abafd918d41c89",
  "runtime_probe": null,
  "schema_origin": "reviewed_task_schema",
  "schema_version": 1,
  "source_task_attributes": [
    "add_prohibit_area",
    "grasp_actor",
    "info",
    "is_left_gripper_close",
    "is_right_gripper_close",
    "model_id",
    "move",
    "move_by_displacement",
    "roller"
  ],
  "task_name": "grab_roller",
  "task_schema": {
    "action_dimension": 14,
    "contact_focus_actor_ids": [
      "roller"
    ],
    "physics_timestep_seconds": 0.004,
    "schema_version": 1,
    "semantic_fields": [
      {
        "actor_id": "roller",
        "name": "roller_position",
        "source": "actor_position"
      },
      {
        "actor_id": "roller",
        "name": "roller_left_contact_position",
        "point_id": 0,
        "source": "actor_contact_position"
      },
      {
        "actor_id": "roller",
        "name": "roller_right_contact_position",
        "point_id": 1,
        "source": "actor_contact_position"
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
    "semantic_roles": {
      "left_target_contact_position": "roller_left_contact_position",
      "left_tcp_position": "left_tcp_position",
      "manipulated_object_position": "roller_position",
      "right_target_contact_position": "roller_right_contact_position",
      "right_tcp_position": "right_tcp_position"
    },
    "success_contract": {
      "minimum_height_m": 0.8,
      "requires_left_gripper_closed": true,
      "requires_right_gripper_closed": true,
      "target_actor_id": "roller",
      "type": "official_check_success"
    },
    "task_family": "dual_arm_lift",
    "task_name": "grab_roller",
    "tracked_actors": [
      {
        "contact_points": [
          0,
          1
        ],
        "functional_points": [],
        "id": "roller",
        "scene_name": "102_roller",
        "task_attribute": "roller"
      }
    ],
    "trusted_tool_profile": "generic_success"
  },
  "taskgen_ready": true,
  "telemetry_observables": {
    "actor_pose_signals": [
      "roller_position",
      "roller_left_contact_position",
      "roller_right_contact_position"
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

DOCUMENTATION `description/task_instruction/grab_roller.json`:
{
  "full_description": "use both arms to grab the roller on the table",
  "schema": "{A} notifies the roller. 'arm' use literal here",
  "preference": "num of words should not exceed 10.",
  "seen": [
    "Take hold of {A} using your arms.",
    "Firmly grip {A} on the table now.",
    "Grab {A} directly from the table.",
    "Take both arms to grasp {A}.",
    "Hold {A} on the table with hands.",
    "Reach for {A} and grab it firmly.",
    "Securely grab {A} using both arms.",
    "Use both arms to grip {A} tightly.",
    "Grasp {A} firmly from the table.",
    "Lift {A} off the table with arms.",
    "Secure {A} on the table using arms.",
    "Take hold of {A} with both arms.",
    "Grab the roller on the table.",
    "Hold {A} firmly from the table.",
    "Lift {A} from the table carefully.",
    "Take {A} directly from the table.",
    "Grasp {A} on the table with arms.",
    "Use arms to firmly grab {A}.",
    "Both arms should grab {A} now.",
    "Pick up {A} from the table directly.",
    "Get hold of {A} using your arms",
    "Secure {A} from the table using arms",
    "Grab the roller placed on the table",
    "Lift {A} off the table with both arms",
    "Reach for {A} and hold it firmly",
    "Lift {A} from its place on the table",
    "Pick up {A} using both arms equally",
    "Pick up the roller using any method",
    "Hold {A} with both arms to pick it up",
    "Reach out to grab {A} from the table",
    "Grab the roller on the table.",
    "Secure {A} with both arms.",
    "Grab {A} placed on the table.",
    "Lift {A} with your arms.",
    "Pick up {A} using both arms.",
    "Grab roller using both hands.",
    "Grasp {A} firmly with arms.",
    "Hold {A} from the table.",
    "Take hold of {A} with arms.",
    "Lift {A} from the table.",
    "Grip {A} firmly with arms.",
    "Use both arms to grab {A}.",
    "Take hold of {A}.",
    "Secure {A} using your arms.",
    "Pick up {A} from the table.",
    "Bring both arms to grab {A}.",
    "Place hands on {A} and lift.",
    "Use arms to hold {A} tightly.",
    "Grasp {A} on the table.",
    "Firmly grab {A} using arms."
  ],
  "unseen": [
    "Grab {A} on the table with arms.",
    "Use both arms to grab {A}.",
    "Grab {A} on the table with arms.",
    "Use both arms to grab {A}.",
    "Grab {A} on the table using arms",
    "Reach and grab {A} with both arms",
    "Grab {A} with both arms.",
    "Use arms to grab {A}.",
    "Hold {A} with both arms.",
    "Grab {A} on the table."
  ]
}

AVAILABLE ASSETS:
[
  {
    "path": "description/objects_description/102_roller/base0.json",
    "sha256": "70d032760ec6d0c161d036a0a7dceb6e74d5283b154bf6bc1acff8a03eceaab8",
    "size_bytes": 620
  },
  {
    "path": "description/objects_description/102_roller/base1.json",
    "sha256": "c73730c36d420a692126fb89117080829d50134548a2cfa6a4aa586233f3d0ea",
    "size_bytes": 685
  },
  {
    "path": "description/objects_description/102_roller/base2.json",
    "sha256": "cab5e8cec78767df7563713595ee1f55bc5ac6d8a73e80b81f63049dc5e2f72b",
    "size_bytes": 559
  }
]
