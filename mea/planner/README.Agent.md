# README.Agent: open evidence-conditioned evaluation planning

This file is runtime context for the production Plan Agent. It describes the
method boundary, not a menu of allowed tasks, aspects, templates, metrics, or
prewritten experiment sequences.

## Query and task boundary

Start from the original open Query. Express a semantic concern and a falsifiable
hypothesis before seeing any task or artifact inventory. Task retrieval happens
afterward and only binds the concern to an executable policy checkpoint and the
closest official base task.

A RoboTwin task is executable when runtime code validates all of:

- the official `envs/<task>.py` source and its `load_actors()` /
  `check_success()` methods;
- a repository-owned TaskSchema;
- non-empty policy weights and dataset statistics for that same task.

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

The next concern may be outside the retrieval inventory. It becomes an
`ExperimentCandidate` with independent optional needs:

- `scene_need`;
- `checker_need`;
- `rule_tool_need`;
- `vqa_tool_need`.

A Tool-only Query must not be forced to generate a scene or checker. A
scene-only request may reuse the official checker. A generated checker defines
experimental success and must remain distinct from official benchmark success.

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

Fixed suites, catalog navigation, task-specific BBH/ClickBell planners, and
registered paper protocols are explicit `experiments/paper/` compatibility
paths. They must not determine the production ClaimFirst candidate domain.

Return only the strict JSON object requested by the active prompt.
