# README.Agent: open evidence-conditioned evaluation planning

This file is the Plan Agent design and README.Agent-ablation reference.
Production prompts keep only the relevant rules below; this is not a menu of
allowed tasks, aspects, templates, metrics, or prewritten experiment sequences.

## Query and task boundary

Start from the original open Query. Express a semantic concern and a falsifiable
hypothesis before seeing any task or artifact inventory. Task retrieval happens
afterward and only binds the concern to an executable policy checkpoint and the
closest official base task.

A RoboTwin policy binding is executable when runtime code validates:

- the official `envs/<task>.py` source and its `load_actors()` /
  `check_success()` methods;
- non-empty policy weights and dataset statistics for that same task.

Scene generation and semantic measurement additionally require a validated
runtime TaskContext or TaskSchema. Runtime may retrieve or derive that context
from official source, reset actors, and telemetry; if required fields remain
unknown, report that need as `unsupported` rather than inventing it. Missing
semantic context does not by itself prohibit an unchanged official rollout.

CapabilityAdapter/catalog membership is not execution authority. Known
Task/Tool/VQA entries are retrieval hints only:

- exact semantic match: reuse and revalidate on the current seed;
- no exact match: generate only the Task or Tool needs requested by the Query;
- incompatible checkpoint or unverifiable physical request: report
  `unsupported` or `inconclusive`.

Never invent paths, modules, checkpoint payloads, seeds, gates, or executable
commands. Runtime owns those fields.

## Evidence-conditioned planning

The first tested concern may come directly from a no-control Query. When the
Query requires a baseline, run only the unchanged official control first; do
not freeze a later concern before control evidence exists.

After each completed round:

1. consume the complete Rule/VQA/Aggregate evidence from completed rounds;
2. assess the original Query's truth condition and remaining uncertainty;
3. stop if the evidence contract is satisfied;
4. otherwise propose the next most informative semantic concern.

Only put a numeric boundary into a generated checker when the original Query
or completed finite scalar/state evidence grounds that boundary. A successful
control alone is not numeric calibration. After a checker fixture failure,
use its expert-terminal actor/TCP coordinates to derive or bracket the next
observable boundary; do not repeat an arbitrary threshold with only a side or
actor relabel. If no grounded boundary exists, choose an exact observable
relation, run a scene-only diagnostic, or report the need unsupported.
For a TaskGen failure, read `bounded_repair_evidence` as well as the terminal
diagnosis. If an earlier expert fixture disproves the requested relation, fix
the Proposal or switch concern; do not repeat it because a later local repair
was correctly rejected for changing semantics.

The next concern may be outside the retrieval inventory. It becomes an
`ExperimentCandidate` with independent optional needs:

- `scene_need`;
- `checker_need`;
- `rule_tool_need`;
- `vqa_tool_need`.

A new concern is an independent official-base experiment unless the Proposal
explicitly says it refines a prior validated scene. A refinement must restate
every retained prior scene delta plus the new delta in the current
`controlled_changes` and scene need. TaskGen does not silently inherit generated
code or simulator state from an earlier round. "Reuse the previous/exact scene"
is not executable by itself: an exact, prior, or refinement request must copy
the simulator-observed actor value or signed delta with coordinate axis and
units. If completed evidence contains no such value, do not call the next scene
the same scene; choose an independent official-base experiment or stop. A
retained prior delta is a current controlled change, not a
`preserve_world_position` fact: preservation is checked against the same-seed
official scene. Preserve only orthogonal official properties such as an
unchanged z coordinate or orientation.

For a numeric position change, `requested_perturbation.controlled_changes`
must carry the executable typed fact
`{actor, property="position", axis, signed_delta, unit="m",
reference="same_seed_official_reset"}`. Description prose explains why the
experiment matters; it is not the numerical execution contract. TaskGen must
compare this signed delta with the same-seed official/generated simulator
states before accepting the scene.

A Tool-only Query must not be forced to generate a scene or checker. A
scene-only request may reuse the official checker. A generated checker defines
experimental success and must remain distinct from official benchmark success.
If success combines the official goal with any additional experimental
condition, request a checker; a numeric Tool alone has no pass/fail authority.
Request a generated checker only when every added relation is directly
observable through the advertised current-state API. Do not ask TaskGen to
infer target contact from gripper closure, simultaneity from sequential events,
or placement from height. If the exact relation is unavailable, choose a
scene-only plus Rule/VQA observation experiment or another informative
sub-aspect instead of knowingly creating an unverifiable checker.
If the original Query explicitly requires a checker for every generated round,
scene-only is not a valid fallback; choose another observable relation or stop
and state the unsupported limitation.

For each continued round, state one explicit bounded scene delta. An independent
official-base concern may let TaskGen choose a source-supported bounded magnitude;
later refinements must use the numeric value reported by completed simulator
evidence. The runtime binding already fixes task identity and policy checkpoint;
do not duplicate those bindings as scene-preservation claims. Other preservation
claims require an advertised authority. Each Rule/VQA need asks for one primary
observation and keeps `reuse_first=true`. Task/Tool generation never authorizes
changing policy weights, controller precision, action noise, or latency.
One Rule Tool request must reduce to one executable metric contract. If height,
left distance, and right distance could all be diagnostic, select the single
highest-information measurement now; later evidence may justify another Tool.
When one hypothesis compares symmetric or grouped signals and the advertised
capabilities include independent derived-observable validation, prefer one
scalar aggregate observable (for example a maximum or absolute difference)
over splitting left and right into duplicate scene rollouts. Otherwise request
the single highest-information typed metric; do not invent an unvalidated
aggregate merely to save a rollout.

Write each `preserved_conditions` entry as one atomic object with exactly
`{actor, property, axis, relation}`. Name a coordinate axis when only that
coordinate is fixed, and distinguish `preserve_local_offsets` from
`preserve_world_position` for contact points. Use `actor=null` for task-wide
facts such as `official_goal`; use `axis=null` unless `property=position`.
TaskGen lets simulator, checker, or visual authority decide each fact. A
non-typed preservation entry is rejected at the Proposal boundary; historical
prose conversion belongs only in cold reproduction code, not production.

Do not repeat `task_identity` or `policy_checkpoint` as per-candidate scene
preservation facts. The outer runtime binding already freezes both, so listing
them here creates a second owner that TaskGen cannot verify from scene probes.
Preserve only actor properties that simulator state can compare and
`official_goal` when exact official-checker reuse or a required conjunct can
verify it.

Interpret evidence by its declared role. The round outcome is the tested
hypothesis verdict; diagnostic Tool values explain it but do not rewrite it.
Terminal state is not trajectory peak, expert evidence is not evaluated-policy
evidence, and visual evidence does not override numeric simulator authority.
After a successful test with comfortable margin, do not repeatedly vary the
same axis unless completed finite evidence brackets a boundary. Simulator
sampling support is not evidence that the official expert can solve every
sampled pose. Do not claim expert executability as a preserved fact unless a
positive expert probe supports it. If both the generated candidate and the
unchanged same-seed expert fail through the same grasp/IK path, that is an
expert-oracle limitation, not a policy weakness and not evidence that the
generated scene broke a contact reference. Base another attempt on the nearest
expert-positive artifact and one genuinely new observable; otherwise actively
stop inconclusive instead of guessing another pose, orientation, or instance.
Switch to an orthogonal concern, or actively stop inconclusive when no distinct
supported concern remains. For `action=stop`, clear the perturbation and set
every artifact need to not required.

Do not treat a valid policy failure as pipeline failure. Do not turn a few
successful episodes into a generalization claim. Open candidate universes
cannot license exhaustive, worst-case, or no-counterexample conclusions.

## Generation and repair

TaskGen retrieves first, then generates a thin official-task subclass or patch
only on a miss. Generated artifacts must pass AST, semantic fixture, simulator
state, render/visual diagnosis, expert-solvability, rollout, and evidence
gates. Visual judgment cannot replace numeric or simulator authority.

TaskGen and ToolGen each permit at most one bounded local repair. There is no
whole-round automatic restart and policy failure is never silently rerun.

## Compatibility boundary

Fixed suites, catalog navigation, task-specific planners, and
registered paper protocols are explicit `experiments/paper/` compatibility
paths. They must not determine the production Plan Agent candidate domain.

Return only the strict JSON object requested by the active prompt.
