You are the visual self-reflection stage of RoboTwin TaskGen.
The image shows the same-seed official scene on the left and the generated
Query-derived scene on the right.

SEMANTIC CONCERN:
robustness of the policy to positional variation of the bell: The ACT policy can successfully complete the click_bell task when the bell is shifted to a new position within a bounded range.

REQUESTED SCENE NEED:
Shift the bell's position horizontally by a small, bounded distance while keeping its size, shape, material, and the overall scene layout unchanged. Preserve unchanged: 任务目标与接触几何语义; size; shape; material; the overall scene layout.

CHECKER NEED (context only; RGB cannot validate success logic):
No change requested; preserve the official implementation.

DECLARED CONDITIONS TO PRESERVE:
- 任务目标与接触几何语义
- size
- shape
- material
- the overall scene layout

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
