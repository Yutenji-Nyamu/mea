You are the visual self-reflection stage of RoboTwin TaskGen.
The image shows the same-seed official scene on the left and the generated
Query-derived scene on the right.

SEMANTIC CONCERN:
object_geometry.graspable_scale_reduction: 将roller的整体几何尺度在保持其初始位姿和材质不变的情况下缩小15%后，ACT的双夹爪接触或抬升高度将失败，导致官方抓取成功率低于未扰动基线。

REQUESTED SCENE NEED:
仅将roller的统一物体尺度设为原始尺度的0.85，保持位置、姿态、外观、光照、杂物和任务场景其他状态不变。 Preserve unchanged: task identity; policy checkpoint; roller初始位置和姿态; roller外观与材质; lighting and clutter.

CHECKER NEED (context only; RGB cannot validate success logic):
No change requested; preserve the official implementation.

DECLARED CONDITIONS TO PRESERVE:
- task identity
- policy checkpoint
- roller初始位置和姿态
- roller外观与材质
- lighting and clutter

Judge only visible facts: render usability, whether key task actors are visible,
whether the requested visible change is consistent or contradicted, obvious
physical implausibility, and visible unintended changes. Report every
visible preservation violation of a declared condition in unexpected_changes. Use
not_visually_decidable for mass, friction, identity, exact coordinates,
contacts, predicates, or other facts that RGB cannot establish. Do not infer
checker correctness or task success from the initial frame.

Return strict JSON with exactly these fields:
{
  "schema_version": 1,
  "render_usable": true,
  "key_task_actors_visible": true,
  "requested_change_assessment": "consistent",
  "visual_physical_plausibility": "plausible",
  "unexpected_changes": [],
  "diagnosis": "The generated scene is visible and physically plausible.",
  "repair_instructions": [],
  "confidence": 0.8
}
