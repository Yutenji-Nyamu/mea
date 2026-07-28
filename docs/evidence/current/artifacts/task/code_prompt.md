Generate one RoboTwin experiment from the open Query-derived candidate below. Retrieve semantics from the official base program, but implement the requested scene and checker rather than selecting a catalog template.

EXPERIMENT CANDIDATE:
{
  "base_task": "click_bell",
  "candidate_id": "dynamic.click.bell.object.generalization.position.translation.the.act.policy.s.first.weakness.in.manipulated.object.generalization.is.spatial.translating.the.bell.within.a.bounded.reachable.workspace.will.reduce.official.press.success.or.increase.final.bell.contact.distance.relative.to.the.unchanged.control.5142e88b1d4d",
  "checker_need": {
    "description": "Generate an experimental check_success predicate that decides: The ACT policy's first weakness in manipulated-object generalization is spatial: translating the bell within a bounded reachable workspace will reduce official press success or increase final bell-contact distance relative to the unchanged control.",
    "kind": "generate",
    "reuse_first": true
  },
  "scene_need": {
    "description": "TaskGen must create a bounded object_position variant overlay for the bell and preserve the official click_bell success contract.",
    "kind": "adapt",
    "reuse_first": true
  },
  "schema_version": 2,
  "semantic_concern": "object_generalization.position_translation: The ACT policy's first weakness in manipulated-object generalization is spatial: translating the bell within a bounded reachable workspace will reduce official press success or increase final bell-contact distance relative to the unchanged control.",
  "source_query": "Where does this ACT policy first expose a weakness when generalizing over manipulated-object properties?",
  "tool_need": {
    "description": "ToolGen should reuse the official success checker and generate a rule metric reporting the final target-contact distance and whether the active gripper is closed, so partial spatial degradation is observable even if the binary outcome remains successful.",
    "kind": "measure",
    "reuse_first": true
  }
}

THIN TASK ADAPTER:
{
  "asset_paths": [
    "description/objects_description/050_bell/base0.json",
    "description/objects_description/050_bell/base1.json"
  ],
  "documentation_paths": [
    "description/task_instruction/click_bell.json",
    "mea/knowledge/tasks/click_bell.md"
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
  "official_class": "click_bell",
  "official_source": "envs/click_bell.py",
  "schema_version": 1,
  "task_name": "click_bell",
  "task_schema": {
    "action_dimension": 14,
    "contact_focus_actor_ids": [
      "bell"
    ],
    "physics_timestep_seconds": 0.004,
    "probe_task_attributes": [
      "bell_id"
    ],
    "schema_version": 1,
    "semantic_fields": [
      {
        "actor_id": "bell",
        "name": "bell_position",
        "source": "actor_position"
      },
      {
        "actor_id": "bell",
        "name": "bell_contact_position",
        "point_id": 0,
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
      "left_tcp_position": "left_tcp_position",
      "manipulated_object_position": "bell_position",
      "right_tcp_position": "right_tcp_position",
      "target_contact_position": "bell_contact_position"
    },
    "success_contract": {
      "requires_closed_active_gripper": true,
      "target_actor_id": "bell",
      "target_contact_point": 0,
      "type": "official_check_success",
      "xy_tolerance_m": [
        0.025,
        0.025
      ],
      "z_tolerance_m": 0.03
    },
    "task_family": "press_contact",
    "task_name": "click_bell",
    "tracked_actors": [
      {
        "contact_points": [
          0
        ],
        "functional_points": [],
        "id": "bell",
        "scene_name": "050_bell",
        "task_attribute": "bell"
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
        rand_pos = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        while abs(rand_pos.p[0]) < 0.05:
            rand_pos = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
            )

        self.bell_id = np.random.choice([0, 1], 1)[0]
        self.bell = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )

        self.add_prohibit_area(self.bell, padding=0.07)
        self.check_arm_function = self.is_left_gripper_close if self.bell.get_pose().p[0] < 0 else self.is_right_gripper_close

def check_success(self):
        if self.stage_success_tag:
            return True
        if not self.check_arm_function():
            return False
        bell_pose = self.bell.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("050_bell")
        eps = [0.025, 0.025]
        for position in positions:
            if (np.all(np.abs(position[:2] - bell_pose[:2]) < eps) and abs(position[2] - bell_pose[2]) < 0.03):
                self.stage_success_tag = True
                return True
        return False
```

TASK TELEMETRY/EXECUTION SCHEMA:
{
  "action_dimension": 14,
  "contact_focus_actor_ids": [
    "bell"
  ],
  "physics_timestep_seconds": 0.004,
  "probe_task_attributes": [
    "bell_id"
  ],
  "schema_version": 1,
  "semantic_fields": [
    {
      "actor_id": "bell",
      "name": "bell_position",
      "source": "actor_position"
    },
    {
      "actor_id": "bell",
      "name": "bell_contact_position",
      "point_id": 0,
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
    "left_tcp_position": "left_tcp_position",
    "manipulated_object_position": "bell_position",
    "right_tcp_position": "right_tcp_position",
    "target_contact_position": "bell_contact_position"
  },
  "success_contract": {
    "requires_closed_active_gripper": true,
    "target_actor_id": "bell",
    "target_contact_point": 0,
    "type": "official_check_success",
    "xy_tolerance_m": [
      0.025,
      0.025
    ],
    "z_tolerance_m": 0.03
  },
  "task_family": "press_contact",
  "task_name": "click_bell",
  "tracked_actors": [
    {
      "contact_points": [
        0
      ],
      "functional_points": [],
      "id": "bell",
      "scene_name": "050_bell",
      "task_attribute": "bell"
    }
  ],
  "trusted_tool_profile": "generic_success"
}

DOCUMENTATION `description/task_instruction/click_bell.json`:
{
  "full_description": "click the <bell's top center> on the table",
  "schema": "{A} notifies the bell, {a} notifies the arm to click the bell",
  "preference": "num of words should not exceed 10",
  "seen": [
    "Press <bell's top center> using {a} on the table",
    "Instruct {a} to press <bell's top center>",
    "Push <bell's top center> on the table",
    "Click {A}'s <top center> using {a}",
    "Make {a} press <bell's top center>",
    "Press the <bell's top center> directly",
    "Direct {a} to click <bell's top center>",
    "Push {A}'s <top center> on the table",
    "Click <bell's top center> using {a}",
    "Press <bell's top center> placed on the table",
    "Press the center top of {A}.",
    "Command {a} to press {A}'s top.",
    "Click at the bell's top center.",
    "Direct {a} to touch {A}'s top.",
    "Press down on the bell's top.",
    "Guide {a} to click the bell's top.",
    "Click the designated center of {A}.",
    "Request {a} to press the bell's top.",
    "Press the specified top area of {A}.",
    "Ask {a} to interact with {A}'s top.",
    "Press the center of {A} using {a}.",
    "Click the bell's center on the table.",
    "Tap {A}'s top center with {a}.",
    "Tap the top center of {A}.",
    "Press {A}'s top center on the table.",
    "Click using {a} on {A}'s center.",
    "Push the center of {A} using {a}.",
    "Push the bell's center on the table.",
    "Press down {A}'s top center gently.",
    "Press down the top of {A} using {a}.",
    "Click the top center of {A} on table.",
    "Direct {a} to click the top of {A}.",
    "Pinpoint {A} and click its top center.",
    "Have {a} click at {A}'s top center.",
    "Press the top center of {A} on table.",
    "Make {a} interact with {A}'s top center.",
    "Click {A} at its top center on table.",
    "Guide {a} to click {A}'s top center.",
    "Locate {A} and click its top center.",
    "Use {a} to press {A}'s top section.",
    "Engage the top center of the bell.",
    "Click {A}'s top center using {a}.",
    "Press the bell's top center on the table.",
    "Tap {A}'s top center with {a}.",
    "Touch the bell at its top center.",
    "Use {a} to touch {A}'s top center.",
    "Engage the bell's top center gently.",
    "Activate {A} by pressing its top center.",
    "Press {A}'s top center with {a} firmly.",
    "Tap the bell's top center on the table."
  ],
  "unseen": [
    "Click the <bell's top center> on the table",
    "Tap the <bell's top center> placed on the table",
    "Click the top center of {A}.",
    "Direct {a} to click {A}'s top.",
    "Click {A}'s top center on the table.",
    "Use {a} to press {A}'s top center.",
    "Find {A} and click its top center.",
    "Use {a} to press {A}'s top center.",
    "Click the bell at its top center.",
    "Use {a} to press {A}'s top center."
  ]
}

DOCUMENTATION `mea/knowledge/tasks/click_bell.md`:
# ClickBell scene contract

`click_bell.load_actors()` creates exactly one static `050_bell`, records its
`bell_id`, and selects the arm from the sign of the bell X coordinate. Position
variants must remain inside the official workspace and consume the official
pose and instance RNG before applying a bounded override.

`check_success()` remains the upstream RoboTwin authority. It requires the
selected gripper to close and contact the bell's functional point. TaskGen may
change only the declared position, instance, or simulator-native scene axis;
it must preserve `play_once()`, `check_success()`, actor identity, and policy
checkpoint semantics.

AVAILABLE ASSETS:
[
  {
    "path": "description/objects_description/050_bell/base0.json",
    "sha256": "543240a1b5b88f5a2ec5975dcbfd865c2eb9b8eda7546775ef3069359225b09f",
    "size_bytes": 639
  },
  {
    "path": "description/objects_description/050_bell/base1.json",
    "sha256": "6c4efb216f006a833724dfe4a760c914f305921c850aae457ac48575754f8ea2",
    "size_bytes": 565
  }
]
