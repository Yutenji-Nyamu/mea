Generate one RoboTwin BeatBlockHammer candidate from the immutable proposal below. The same candidate must define the scene and its replacement success checker.

PROPOSAL:
{
  "intent": "evaluate a query-relevant bounded variation",
  "proposal_id": "robustness.distractor_avoidance.query_generated_1",
  "query": "Generate a BeatBlockHammer scene containing the official target block and one same-size physical look-alike distractor. Replace check_success() so the rollout succeeds only after target alignment/contact and fails after any distractor contact. Validate the generated scene and checker before running one ACT episode. Query-generated bounded variation: evaluate a query-relevant bounded variation",
  "scene": {
    "distractor_color": [
      0.85,
      0.05,
      0.05
    ],
    "distractor_name": "distractor_box",
    "distractor_offset_xy_m": [
      0.1,
      0.0
    ],
    "half_size_m": [
      0.025,
      0.025,
      0.025
    ],
    "target_color": [
      1.0,
      0.0,
      0.0
    ],
    "target_name": "box"
  },
  "schema_version": 1,
  "success": {
    "forbid_distractor_contact": true,
    "latch_distractor_contact": true,
    "require_target_contact": true,
    "target_alignment_thresholds_m": [
      0.025,
      0.025
    ]
  },
  "task_name": "beat_block_hammer"
}

OUTPUT CONTRACT:
Return one strict JSON object with exactly two string fields, load_actors and check_success. Each field must contain one complete Python method. The target and a same-size physical look-alike distractor must both exist. Success requires target alignment/contact and no distractor contact. Do not return Markdown.

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
Do not use imports, files, network, processes, dunder attributes, dynamic execution, super(), or extra helpers. Preserve the official hammer and random target pose, add a static same-size distractor at the declared offset, and latch any distractor contact. The immutable official hammer contract is create_actor(scene=self, pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]), modelname="020_hammer", convex=True, model_id=0), followed by self.hammer.set_mass(0.001). Sample the target with rand_pose(xlim=[-0.25, 0.25], ylim=[-0.05, 0.15], zlim=[0.76], qpos=[1, 0, 0, 0], rotate_rand=True, rotate_lim=[0, 0, 0.5]). Pass is_static=True when creating both boxes. Assign the target actor to self.block because inherited play_once() reads self.block; additional aliases are allowed. Assign the distractor to a stable public attribute such as self.distractor for use by check_success. self.add_prohibit_area(self.hammer, padding=0.10), then call self.prohibited_area.append([pose.p[0] - 0.05, pose.p[1] - 0.05, pose.p[0] + 0.05, pose.p[1] + 0.05]) once for the target pose and once for the distractor pose; do not invent a prohibit_regions attribute. Choose two public contact-latch attribute names, initialize both to false, and reuse those names in check_success. Use only np.array, np.asarray, np.sum, np.all, np.any, np.abs, sapien.Pose, create_actor, create_box, the global rand_pose function, and the listed task/actor methods. The base task has no self.create_actor, self.create_box, self.rand_pose, or self._get_random_pose methods; call the global functions directly. Actors have no get_contacts method. Detect contact only with self.check_actors_contact(self.hammer.get_name(), self.block.get_name()) and the analogous call using self.distractor.get_name(); pass actor-name strings, never actor objects. Read alignment with self.hammer.get_functional_point(0, "pose").p and the target actor's get_functional_point(1, "pose").p, compare their first two coordinates against np.array([0.025, 0.025]). Equivalent structure is allowed; scene and checker semantics are validated by fixtures.
