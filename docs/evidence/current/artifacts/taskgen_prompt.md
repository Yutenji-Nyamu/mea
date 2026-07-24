You are the bounded RoboTwin BeatBlockHammer TaskGen code agent.
The same immutable proposal must produce both scene construction and a replacement success checker.

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

Return one strict JSON object with exactly two string fields: load_actors and check_success. Each string contains one complete Python method. Do not use imports, files, network, processes, dunder attributes, dynamic execution, super(), or extra helpers. Preserve the official hammer and random target pose, add a static same-size distractor at the declared offset, and latch any distractor contact. The immutable official hammer contract is create_actor(scene=self, pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]), modelname="020_hammer", convex=True, model_id=0), followed by self.hammer.set_mass(0.001). Sample the target with rand_pose(xlim=[-0.25, 0.25], ylim=[-0.05, 0.15], zlim=[0.76], qpos=[1, 0, 0, 0], rotate_rand=True, rotate_lim=[0, 0, 0.5]). Pass is_static=True when creating both boxes. Call self.add_prohibit_area(self.hammer, padding=0.10), then call self.prohibited_area.append([pose.p[0] - 0.05, pose.p[1] - 0.05, pose.p[0] + 0.05, pose.p[1] + 0.05]) once for the target pose and once for the distractor pose; do not invent a prohibit_regions attribute. Choose two public contact-latch attribute names, initialize both to false, and reuse those names in check_success. Success requires target alignment/contact and no distractor contact. Use only np.array, np.asarray, np.sum, np.all, np.any, np.abs, sapien.Pose, create_actor, create_box, the global rand_pose function, and the listed task/actor methods needed for those requirements. The base task has no self.create_actor, self.create_box, self.rand_pose, or self._get_random_pose methods; call create_actor(...), create_box(...), and rand_pose(...) directly as global functions. Actors have no get_contacts method. Detect contact only with self.check_actors_contact(self.hammer.get_name(), self.block.get_name()) and the equivalent call for self.distractor. Read alignment only from self.hammer.get_functional_point(0, "pose").p and self.block.get_functional_point(1, "pose").p; compare only their first two coordinates against np.array([0.025, 0.025]). Equivalent implementation structure is allowed; the result is checked with scene and success fixtures. Do not return Markdown.