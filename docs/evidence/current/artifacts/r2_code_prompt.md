Generate one RoboTwin ClickBell candidate from the immutable proposal below. The same candidate must define the scene and replacement checker.

PROPOSAL:
{
  "intent": "Can this ACT policy click the intended bell without touching a nearby visually similar distractor bell? Trusted bounded variant: One alternate official bell instance is placed 0.12 m from the target; success requires the correct-arm target press and forbids every latched distractor contact.",
  "proposal_id": "robustness.distractor_avoidance.lookalike_bell",
  "query": "Can this ACT policy click the intended bell without touching a nearby visually similar distractor bell? Trusted bounded variant: One alternate official bell instance is placed 0.12 m from the target; success requires the correct-arm target press and forbids every latched distractor contact. Query-generated bounded variation: Can this ACT policy click the intended bell without touching a nearby visually similar distractor bell? Trusted bounded variant: One alternate official bell instance is placed 0.12 m from the target; success requires the correct-arm target press and forbids every latched distractor contact.",
  "scene": {
    "distractor_name": "distractor_bell",
    "distractor_offset_xy_m": [
      0.0,
      0.12
    ],
    "instance_relation": "alternate_official_instance",
    "target_name": "050_bell"
  },
  "schema_version": 1,
  "success": {
    "forbid_distractor_contact": true,
    "latch_distractor_contact": true,
    "require_correct_arm": true,
    "target_xy_threshold_m": [
      0.025,
      0.025
    ],
    "target_z_threshold_m": 0.03
  },
  "task_name": "click_bell"
}

OUTPUT CONTRACT:
Return one strict JSON object with exactly two string fields, load_actors and check_success. Each field contains one complete Python method. Do not return Markdown.

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
RETRIEVED OFFICIAL CLICK_BELL METHODS (envs/click_bell.py; preserve these public APIs):
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

SUPPORTED DELTA API:
- Preserve the official rand_pose arguments, bell instance sampling, self.bell, self.bell_id, self.check_arm_function, and inherited play_once.
- Create the second bell with create_actor(scene=self, pose=..., modelname="050_bell", convex=True, model_id=1 - self.bell_id, is_static=True). Store it as self.distractor and rename the actor with self.distractor.set_name("distractor_bell").
- Construct its pose with sapien.Pose and the proposal offset. Initialize self._mea_distractor_contact_seen = False.
- Contact APIs are only self.get_gripper_actor_contact_position(actor.get_name()) and self.bell.get_contact_point(0). Do not invent helper methods.
- Preserve the official correct-arm check, target-contact thresholds, and boolean self.stage_success_tag latch. Update the distractor latch before returning the prior target-success latch.
