You are the visual self-reflection stage of RoboTwin TaskGen.
The image shows the same-seed official scene on the left and the generated
Query-derived scene on the right.

SEMANTIC CONCERN:
object_geometry.graspable_scale_strong_reduction: 将roller统一尺度进一步降至原始尺度的0.70后，ACT的双侧抓取几何余量会明显下降，并导致官方抬升成功率低于未扰动基线；若仍成功，则说明0.85尺度测试中的22毫米左侧接触距离并未揭示当前最早弱点。

REQUESTED SCENE NEED:
仅将roller的统一物体尺度设为原始尺度的0.70，保持其初始位置、姿态、外观、材质、光照、杂物及其他场景状态不变。 Preserve unchanged: task identity; policy checkpoint; roller初始位置和姿态; roller外观与材质; lighting; clutter; 其他任务场景状态.

CHECKER NEED (context only; RGB cannot validate success logic):
No change requested; preserve the official implementation.

DECLARED CONDITIONS TO PRESERVE:
- task identity
- policy checkpoint
- roller初始位置和姿态
- roller外观与材质
- lighting
- clutter
- 其他任务场景状态

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
