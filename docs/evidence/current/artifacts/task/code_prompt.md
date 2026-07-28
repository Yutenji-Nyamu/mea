Generate one RoboTwin experiment from the open Query-derived candidate below. Retrieve semantics from the official base program, but implement the requested scene and checker rather than selecting a catalog template.

EXPERIMENT CANDIDATE:
{
  "base_task": "adjust_bottle",
  "candidate_id": "dynamic.adjust.bottle.task.execution.success.margin.components.act.50450b5e2502",
  "checker_need": {
    "description": "Generate an experimental check_success predicate that decides: 在保持官方场景与 ACT 检查点不变的情况下，基线成功主要由瓶子的高度达标或横向越界裕量之一决定；分解后的轨迹观测将显示最先接近失败的成功条件，从而定位潜在弱点。",
    "kind": "generate",
    "reuse_first": true
  },
  "scene_need": {
    "description": "TaskGen must retrieve or generate a scene checker that preserves official success while exposing bottle functional-point height and absolute-x margin components.",
    "kind": "adapt",
    "reuse_first": true
  },
  "schema_version": 2,
  "semantic_concern": "task_execution.success_margin_components: 在保持官方场景与 ACT 检查点不变的情况下，基线成功主要由瓶子的高度达标或横向越界裕量之一决定；分解后的轨迹观测将显示最先接近失败的成功条件，从而定位潜在弱点。",
  "source_query": "这个 ACT 策略执行调整瓶子任务时，对未见对象属性的泛化能力如何，最先在哪里暴露弱点？",
  "tool_need": {
    "description": "ToolGen must retrieve or generate observable rule metrics for final bottle functional-point height, absolute x, height margin, x margin, and official success, reusing the official checker where possible.",
    "kind": "measure",
    "reuse_first": true
  }
}

THIN TASK ADAPTER:
{
  "asset_paths": [
    "description/objects_description/001_bottle/base0.json",
    "description/objects_description/001_bottle/base1.json",
    "description/objects_description/001_bottle/base10.json",
    "description/objects_description/001_bottle/base11.json",
    "description/objects_description/001_bottle/base12.json",
    "description/objects_description/001_bottle/base13.json",
    "description/objects_description/001_bottle/base14.json",
    "description/objects_description/001_bottle/base15.json",
    "description/objects_description/001_bottle/base16.json",
    "description/objects_description/001_bottle/base17.json",
    "description/objects_description/001_bottle/base18.json",
    "description/objects_description/001_bottle/base19.json",
    "description/objects_description/001_bottle/base2.json",
    "description/objects_description/001_bottle/base20.json",
    "description/objects_description/001_bottle/base21.json",
    "description/objects_description/001_bottle/base22.json",
    "description/objects_description/001_bottle/base3.json",
    "description/objects_description/001_bottle/base4.json",
    "description/objects_description/001_bottle/base5.json",
    "description/objects_description/001_bottle/base6.json",
    "description/objects_description/001_bottle/base7.json",
    "description/objects_description/001_bottle/base8.json",
    "description/objects_description/001_bottle/base9.json"
  ],
  "documentation_paths": [
    "description/task_instruction/adjust_bottle.json"
  ],
  "generation_hook_contract": {
    "expert_preflight": true,
    "local_regeneration_limit": 1,
    "methods": [
      "load_actors",
      "check_success"
    ],
    "render_preflight": true,
    "static_and_fixture_validation": true
  },
  "official_class": "adjust_bottle",
  "official_source": "envs/adjust_bottle.py",
  "schema_version": 1,
  "task_name": "adjust_bottle",
  "task_schema": {
    "action_dimension": 14,
    "contact_focus_actor_ids": [
      "bottle"
    ],
    "physics_timestep_seconds": 0.004,
    "schema_version": 1,
    "semantic_fields": [
      {
        "actor_id": "bottle",
        "name": "bottle_position",
        "source": "actor_position"
      },
      {
        "actor_id": "bottle",
        "name": "bottle_functional_position",
        "point_id": 0,
        "source": "actor_functional_position"
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
      "left_tcp_position": "left_tcp_position",
      "manipulated_functional_position": "bottle_functional_position",
      "manipulated_object_position": "bottle_position",
      "right_tcp_position": "right_tcp_position"
    },
    "success_contract": {
      "minimum_absolute_x_m": 0.15,
      "minimum_height_m": 0.9,
      "target_actor_id": "bottle",
      "target_functional_point": 0,
      "target_side_semantic_field": "bottle_functional_position",
      "target_side_source": "initial_bottle_x_sign",
      "type": "official_check_success"
    },
    "task_family": "object_reposition",
    "task_name": "adjust_bottle",
    "tracked_actors": [
      {
        "contact_points": [],
        "functional_points": [
          0
        ],
        "id": "bottle",
        "scene_name": "001_bottle",
        "task_attribute": "bottle"
      }
    ],
    "trusted_tool_profile": "generic_success"
  }
}

SIMULATOR-SPECIFIC API CONSTRAINTS:
Keep the official class identity and policy action interface. Use only assets and simulator APIs present in retrieved context. The generated initial scene must differ observably from the same-seed official scene in simulator state or rendered pixels when scene_need is non-null; when scene_need is null, preserve the official load_actors implementation exactly. When checker_need is null, preserve official check_success exactly. SAPIEN Pose.p and Pose.q values must not be modified by indexed assignment or +=/-= because those writes do not update the Pose; construct a new sapien.Pose from a copied position array and the original quaternion before passing it to create_actor. If load_actors adds an actor that later measurement may need, also assign self.mea_telemetry_tracked_actors to a list of dicts with exactly id, task_attribute, scene_name, functional_points, contact_points, and contact_focus; task_attribute must name the public self attribute holding that actor. Do not redeclare an actor already present in the TASK TELEMETRY/EXECUTION SCHEMA; that schema remains valid when the generated scene replaces the same public actor attribute and scene name. The initial state must not satisfy check_success; the official expert terminal state must satisfy it.

OUTPUT CONTRACT:
Return one strict JSON object with exactly two string fields, load_actors and check_success. Each field must contain one complete Python method with only self. A non-null scene_need requires a changed load_actors method; a null scene_need requires the exact official load_actors method from the retrieved source. A non-null checker_need requires a changed check_success method; a null checker_need requires the exact official check_success method. Do not return Markdown, a template id, or an explanation.

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
- The explicit `provider_scene_checker_codegen` route must generate both
  `load_actors()` and `check_success()` from the same Proposal. Its checker is
  experimental and must never be relabeled as official success.
- A paper-claim run requires compile/semantic fixtures, render, expert
  solvability, and the generated scene/checker to remain bound to one artifact.

RETRIEVED ROBOTWIN API AND TASK CONTEXT:
OFFICIAL BASE TASK METHODS:
```python
def load_actors(self):
        self.qpose_tag = np.random.randint(0, 2)
        qposes = [[0.707, 0.0, 0.0, -0.707], [0.707, 0.0, 0.0, 0.707]]
        xlims = [[-0.12, -0.08], [0.08, 0.12]]

        self.model_id = np.random.choice([13, 16])

        self.bottle = rand_create_actor(
            self,
            xlim=xlims[self.qpose_tag],
            ylim=[-0.13, -0.08],
            zlim=[0.752],
            rotate_rand=True,
            qpos=qposes[self.qpose_tag],
            modelname="001_bottle",
            convex=True,
            rotate_lim=(0, 0, 0.4),
            model_id=self.model_id,
        )
        self.delay(4)
        self.add_prohibit_area(self.bottle, padding=0.15)
        self.left_target_pose = [-0.25, -0.12, 0.95, 0, 1, 0, 0]
        self.right_target_pose = [0.25, -0.12, 0.95, 0, 1, 0, 0]

def check_success(self):
        target_hight = 0.9
        bottle_pose = self.bottle.get_functional_point(0)
        return ((self.qpose_tag == 0 and bottle_pose[0] < -0.15) or
                (self.qpose_tag == 1 and bottle_pose[0] > 0.15)) and bottle_pose[2] > target_hight
```

TASK TELEMETRY/EXECUTION SCHEMA:
{
  "action_dimension": 14,
  "contact_focus_actor_ids": [
    "bottle"
  ],
  "physics_timestep_seconds": 0.004,
  "schema_version": 1,
  "semantic_fields": [
    {
      "actor_id": "bottle",
      "name": "bottle_position",
      "source": "actor_position"
    },
    {
      "actor_id": "bottle",
      "name": "bottle_functional_position",
      "point_id": 0,
      "source": "actor_functional_position"
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
    "left_tcp_position": "left_tcp_position",
    "manipulated_functional_position": "bottle_functional_position",
    "manipulated_object_position": "bottle_position",
    "right_tcp_position": "right_tcp_position"
  },
  "success_contract": {
    "minimum_absolute_x_m": 0.15,
    "minimum_height_m": 0.9,
    "target_actor_id": "bottle",
    "target_functional_point": 0,
    "target_side_semantic_field": "bottle_functional_position",
    "target_side_source": "initial_bottle_x_sign",
    "type": "official_check_success"
  },
  "task_family": "object_reposition",
  "task_name": "adjust_bottle",
  "tracked_actors": [
    {
      "contact_points": [],
      "functional_points": [
        0
      ],
      "id": "bottle",
      "scene_name": "001_bottle",
      "task_attribute": "bottle"
    }
  ],
  "trusted_tool_profile": "generic_success"
}

DOCUMENTATION `description/task_instruction/adjust_bottle.json`:
{
  "full_description": "Pick up the bottle on the table headup with the correct arm",
  "schema": "{A} notifies the bottle, {a} notifies the arm to manipulate the bottle",
  "preference": "num of words should not exceed 15",
  "seen": [
    "Lift {A} head-up from the table.",
    "Pick {A} up with {a} ensuring it stays head-up.",
    "Grab {A} from the table and hold it head-up.",
    "Utilize {a} to lift {A} while keeping it head-up.",
    "Lift {A} ensuring it remains upright.",
    "Grab {A} head-up using {a} from the table.",
    "Hold {A} head-up after lifting it.",
    "Use {a} to pick {A} up and keep it head-up.",
    "Pick {A} head-up and hold it steady.",
    "Use {a} to grab and lift {A} head-up.",
    "Grab {A} from the table with {a}",
    "Lift the bottle {A} headup from the table",
    "Raise {A} in a head-up position",
    "Use {a} to lift {A} head-up",
    "Position {A} head-up and lift it",
    "Grab {A} with {a} in a head-up way",
    "Lift the bottle {A} up from the table",
    "Pick {A} head-up using the right arm",
    "Grab {A} and lift it into a head-up position",
    "Use {a} to pick up {A} in the correct orientation",
    "Lift {A} from the table upright",
    "Use {a} to hold {A} from the table",
    "Identify {A} and grab it with {a}",
    "Pick {A} upright from the table",
    "Lift {A} using {a} and hold upright",
    "Take {A} from the table and keep upright",
    "Grab {A} and lift it with {a}",
    "Pick up {A} upright from the table",
    "Hold {A} upright after lifting with {a}",
    "Lift {A} from the table and secure upright",
    "Lift {A} from the table with {a}",
    "Pick {A} upright from the table",
    "Grab {A} and lift it upright",
    "Lift {A} head-up from the table",
    "Using {a}, pick {A} upright",
    "Secure {A} upright with {a}",
    "Hold {A} upright from the table",
    "Pick {A} and keep it upright",
    "Lift {A} upright carefully using {a}",
    "Carefully grab {A} head-up",
    "Pick up {A} from the table carefully.",
    "Use {a} to pick up {A} from the table.",
    "Locate {A} and lift it upright with {a}.",
    "Raise {A} from the table using the correct arm, {a}.",
    "Grab {A} and lift it upward from the table.",
    "Use the correct arm to pick up {A}.",
    "Lift {A} off the table and hold it upright.",
    "Pick up {A} from the table using {a}.",
    "Lift {A} from the table without mentioning the arm.",
    "Find {A} on the table and raise it using {a}."
  ],
  "unseen": [
    "Pick up {A} from the table head-up.",
    "Use {a} to grab {A} head-up.",
    "Use {a} to grab the bottle {A}",
    "Pick up {A} using the correct arm",
    "Grab {A} from the table with {a}",
    "Pick up {A} carefully using {a}",
    "Pick up {A} using {a} in an upright position",
    "Use {a} to grab {A} upright",
    "Lift {A} from the table using {a}.",
    "Grab {A} on the table and raise it."
  ]
}

AVAILABLE ASSETS:
[
  {
    "path": "description/objects_description/001_bottle/base0.json",
    "sha256": "4927366c88ece99e61bb06c4ae78303825e97011b01e0f164a0b346f8ebb6049",
    "size_bytes": 634
  },
  {
    "path": "description/objects_description/001_bottle/base1.json",
    "sha256": "a983d35344e69c2a723f14c48940340855d678809b9b3d2f6a5c5bb208a8343a",
    "size_bytes": 618
  },
  {
    "path": "description/objects_description/001_bottle/base10.json",
    "sha256": "b1d2a1d2f9bbf3420d4c570bb58cee5bd09b7bc090ba242696c0bfd8d7815200",
    "size_bytes": 745
  },
  {
    "path": "description/objects_description/001_bottle/base11.json",
    "sha256": "931e15d76717f420cceb94ec693500ab4d4e7b63bd2a0cb4afefb74768860bca",
    "size_bytes": 692
  },
  {
    "path": "description/objects_description/001_bottle/base12.json",
    "sha256": "bcd691a0d6cc3e9fe643e858f66d1bda402cde9275db1ce84f1a64d14a52bc51",
    "size_bytes": 696
  },
  {
    "path": "description/objects_description/001_bottle/base13.json",
    "sha256": "49da9966fd4e19adaaea661f76c9e70bd6d876b81ae027b5fceb8365b9be7544",
    "size_bytes": 590
  },
  {
    "path": "description/objects_description/001_bottle/base14.json",
    "sha256": "13bbd355a3aad2aaad23c652589a49f5b33b51edc1967249858f679d0c67af93",
    "size_bytes": 572
  },
  {
    "path": "description/objects_description/001_bottle/base15.json",
    "sha256": "7d9b9506ef0e9f4907f120948f258e4bc9027538a93dd0ed034471f79f8c3d1d",
    "size_bytes": 696
  },
  {
    "path": "description/objects_description/001_bottle/base16.json",
    "sha256": "d9d44cf675f0d6f6253447de31f29ecb0eed829a365adc1af2f9bf54e42015a5",
    "size_bytes": 634
  },
  {
    "path": "description/objects_description/001_bottle/base17.json",
    "sha256": "8fd5f1fba205322d28d60a282e79999def98c842a54ea18b6ee605b9deebbaba",
    "size_bytes": 616
  },
  {
    "path": "description/objects_description/001_bottle/base18.json",
    "sha256": "4b9263e19a00accb3b4cc9f56502d8938cdc2a3010c594bb8786b22aba0cb3c2",
    "size_bytes": 698
  },
  {
    "path": "description/objects_description/001_bottle/base19.json",
    "sha256": "43fbb4b53560eca8d170fe541e119b639be27ee40a3ddec35da0fd39b547f5c9",
    "size_bytes": 685
  },
  {
    "path": "description/objects_description/001_bottle/base2.json",
    "sha256": "adbd5b0e14e2ea67e841f2d1716b20367802ee59dea1bc94e2182e0129898d3d",
    "size_bytes": 725
  },
  {
    "path": "description/objects_description/001_bottle/base20.json",
    "sha256": "4d5779917bb88f245d9216528e23496bddcb9ce1232fcb2029666749c3289fe5",
    "size_bytes": 718
  },
  {
    "path": "description/objects_description/001_bottle/base21.json",
    "sha256": "45f26d6372df9190621e897154d4fbaa9ae33889fa1bfd849b6345aa52a6cd33",
    "size_bytes": 591
  },
  {
    "path": "description/objects_description/001_bottle/base22.json",
    "sha256": "4cbd73bb645c30a942947025a49b9ef7c23fb08ec20487b88a94823c88d5c201",
    "size_bytes": 706
  },
  {
    "path": "description/objects_description/001_bottle/base3.json",
    "sha256": "dbaee62ec1b04e8046dcd70e97b79273ada76cbad3c5168dd3e8dee11dcb2c87",
    "size_bytes": 594
  },
  {
    "path": "description/objects_description/001_bottle/base4.json",
    "sha256": "b260cd25881b1dcdf8402602f434acfb639e63337ae1584a3fa7ea38642bb9ec",
    "size_bytes": 658
  },
  {
    "path": "description/objects_description/001_bottle/base5.json",
    "sha256": "d5b8d6563bec2f4d8fb0cc834aeec613721a6398faa157fdb83ca0940cfda9cb",
    "size_bytes": 651
  },
  {
    "path": "description/objects_description/001_bottle/base6.json",
    "sha256": "7408422fdb5626ae9048bb5018d7407123779a08fa8c98f6c8ee0f70bbf26c73",
    "size_bytes": 632
  },
  {
    "path": "description/objects_description/001_bottle/base7.json",
    "sha256": "6068368b228990ff6733e015b7a2f1dc97392c622e20ea2966db2f1692ab8c1d",
    "size_bytes": 731
  },
  {
    "path": "description/objects_description/001_bottle/base8.json",
    "sha256": "3737b823ad0d39e9fd52f009c0416e992e2e5916b9bdfde67c604253ca0d8f0e",
    "size_bytes": 649
  },
  {
    "path": "description/objects_description/001_bottle/base9.json",
    "sha256": "53f23e1da33b59807e01b77e352562f1d39ee8a872096e2ff19c365a42d80987",
    "size_bytes": 697
  }
]
