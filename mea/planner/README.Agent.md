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

1. consume the complete Rule/VQA/Aggregate evidence and its lineage;
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
