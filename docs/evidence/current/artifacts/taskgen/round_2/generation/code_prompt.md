Generate one RoboTwin experiment from the open Query-derived candidate below. Retrieve semantics from the official base program, but implement the requested scene and checker rather than selecting a catalog template.

EXPERIMENT CANDIDATE:
{
  "base_task": "grab_roller",
  "candidate_id": "dynamic.grab.roller.object.geometry.graspable.scale.reduction.roller.15.act.5e696fb1d45c",
  "checker_need": null,
  "evaluation_intent": {
    "hypothesis": "将roller的整体几何尺度在保持其初始位姿和材质不变的情况下缩小15%后，ACT的双夹爪接触或抬升高度将失败，导致官方抓取成功率低于未扰动基线。",
    "intent_id": "intent.7d4cb852954896a8",
    "original_concern": "object_geometry.graspable_scale_reduction",
    "preserved_conditions": [
      "task identity",
      "policy checkpoint",
      "roller初始位置和姿态",
      "roller外观与材质",
      "lighting and clutter"
    ],
    "requested_change": "仅将roller的统一物体尺度设为原始尺度的0.85，保持位置、姿态、外观、光照、杂物和任务场景其他状态不变。",
    "required_observation": "复用或生成数值Rule Tool，记录roller峰值高度、双夹爪闭合状态以及左右TCP到对应接触位置的最小距离，用于区分接触对齐失败与抬升失败。",
    "schema_version": 1,
    "source_query": "这个ACT策略在grab_roller任务中最先会在哪种可执行物体属性或场景变化上暴露弱点？"
  },
  "intent_alignment": {
    "matched_intent_fields": [
      "preserved_conditions",
      "hypothesis",
      "required_observation"
    ],
    "rationale": "Candidate is a nearby diagnostic, not a direct implementation of candidate-contract fields: ['requested_change'].",
    "relationship": "diagnostic_proxy",
    "schema_version": 1,
    "unmatched_intent_fields": [
      "requested_change"
    ]
  },
  "rule_tool_need": {
    "description": "复用或生成数值Rule Tool，记录roller峰值高度、双夹爪闭合状态以及左右TCP到对应接触位置的最小距离，用于区分接触对齐失败与抬升失败。 Hypothesis: 将roller的整体几何尺度在保持其初始位姿和材质不变的情况下缩小15%后，ACT的双夹爪接触或抬升高度将失败，导致官方抓取成功率低于未扰动基线。",
    "kind": "measure",
    "reuse_first": true
  },
  "scene_need": {
    "description": "仅将roller的统一物体尺度设为原始尺度的0.85，保持位置、姿态、外观、光照、杂物和任务场景其他状态不变。 Preserve unchanged: task identity; policy checkpoint; roller初始位置和姿态; roller外观与材质; lighting and clutter.",
    "kind": "adapt",
    "reuse_first": true
  },
  "schema_version": 2,
  "semantic_concern": "object_geometry.graspable_scale_reduction: 将roller的整体几何尺度在保持其初始位姿和材质不变的情况下缩小15%后，ACT的双夹爪接触或抬升高度将失败，导致官方抓取成功率低于未扰动基线。",
  "source_query": "这个ACT策略在grab_roller任务中最先会在哪种可执行物体属性或场景变化上暴露弱点？",
  "tool_need": {
    "description": "复用或生成数值Rule Tool，记录roller峰值高度、双夹爪闭合状态以及左右TCP到对应接触位置的最小距离，用于区分接触对齐失败与抬升失败。 Hypothesis: 将roller的整体几何尺度在保持其初始位姿和材质不变的情况下缩小15%后，ACT的双夹爪接触或抬升高度将失败，导致官方抓取成功率低于未扰动基线。",
    "kind": "measure",
    "reuse_first": true
  },
  "vqa_tool_need": null
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

SIMULATOR-SPECIFIC API CONSTRAINTS:
Keep the official class identity and policy action interface. Use only assets and simulator APIs present in retrieved context. The generated initial scene must differ observably from the same-seed official scene in simulator state or rendered pixels when scene_need is non-null; when scene_need is null, preserve the official load_actors implementation exactly. When checker_need is null, preserve official check_success exactly. SAPIEN Pose.p and Pose.q values must not be modified by indexed assignment or +=/-= because those writes do not update the Pose; construct a new sapien.Pose from a copied position array and the original quaternion before passing it to create_actor. The upstream create_actor scale argument is normally replaced by asset model_data. scale_multiplier is the final/original size ratio: increase by 50% uses 1.5; reduce by 50%, or reduce to 50%, uses 0.5. Use scale_override only for a known absolute asset scale. Both opt-ins update the built mesh scale and Actor point metadata. If load_actors adds an actor that later measurement may need, also assign self.mea_telemetry_tracked_actors to a list of dicts with exactly id, task_attribute, scene_name, functional_points, contact_points, and contact_focus; task_attribute must name the public self attribute holding that actor, and contact_focus must be a boolean. Actors already listed in the TASK TELEMETRY/EXECUTION SCHEMA remain tracked automatically when their pose or instance is replaced: do not assign mea_telemetry_tracked_actors merely to repeat them. Include only entirely new actors in that list. Every new actor must have a unique simulator/contact identity distinct from every base actor: pass a unique runtime_name to create_actor when the asset modelname is reused, and declare that exact runtime get_name() value as scene_name. The asset modelname is not a unique runtime identity. Do not redeclare an actor already present in the TASK TELEMETRY/EXECUTION SCHEMA; that schema remains valid when the generated scene replaces the same public actor attribute and scene name. The initial state must not satisfy check_success; the official expert terminal state must satisfy it.

OUTPUT CONTRACT:
Return one strict JSON object with exactly two string fields, load_actors and check_success. Each field must contain one complete Python method with only self when its corresponding need is non-null. A non-null scene_need requires a changed load_actors method. A non-null checker_need requires a changed check_success method. Both JSON fields remain required for transport, but when a need is null return an empty string for that field: the runtime ignores that text and injects the exact official method before AST, fixture, render, and expert validation. A changed load_actors method must directly implement the requested scene change. Comments or an unrelated actor/pose change are not implementation evidence. load_actors cannot alter policy weights, controller or gripper precision, action noise, latency, or inference. Those require an explicit runtime intervention and must not be simulated by relabelling a scene change. Actors already present in the TASK TELEMETRY/EXECUTION SCHEMA are tracked automatically even when their pose or instance is replaced. Do not assign self.mea_telemetry_tracked_actors merely to repeat one of those base actors. Assign it only when adding an entirely new actor, include only new actors, and give every entry exactly id, task_attribute, scene_name, functional_points, contact_points, and a boolean contact_focus. Do not return Markdown, a template id, or an explanation. When the retrieved API supports scale_multiplier, it is the final-size/original-size ratio: increasing size by 50% uses 1.5, while reducing size by 50% (or to 50%) uses 0.5.

README.AGENT CONTEXT:
# TaskGen output rules

- Return complete methods, never patches or prose.
- Change only fields authorized by the validated Proposal.
- Preserve actor identity, collision behavior, random-call order, and required
  telemetry names.
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
