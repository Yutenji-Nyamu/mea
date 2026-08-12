# Failure-driven Agent prompt iteration

This cold protocol keeps prompt development tied to observed method failures.
It is not a task menu, a second registry, or a replacement for simulator and
semantic validation.

## Knowledge placement

- `mea/*/README.Agent.md`: short cross-task invariants used on every matching
  Agent call.
- `mea/knowledge/tasks/<task>.md`: source-backed task facts and a few observed
  task-local lessons, retrieved only after runtime task binding.
- evaluation artifacts / this ledger: raw prompt, response, validator error,
  correction and server regression result. Raw runtime bundles stay on AutoDL.

There is deliberately no automatic failure index or token-overlap retrieval.
A task-local lesson goes into its short task card; only a failure repeated
across tasks becomes one concise rule in the owning `README.Agent.md`. A repair
receives the previous complete output and the current concrete error directly.

`description/task_instruction/*.json` describes user-facing task language. It
must not be treated as simulator implementation guidance.

## Iteration contract

1. Classify the failure as Agent semantics/generation, simulator/runtime,
   provider/transport, or policy outcome. Only the first class changes prompts.
2. Give a retry the previous complete output and the exact validator stage and
   error. Keep one bounded repair; do not add recovery state machines.
3. Prefer one short task guide, or one concise shared rule after the same
   failure repeats across tasks, over another code branch. Preserve AST,
   fixture, render/expert, Tool oracle and evidence-scope checks because prompts
   do not establish simulator truth.
4. First replay the frozen failure without rollout. Then run the smallest live
   case that can show the failure disappeared or the next Proposal changed.
5. Recheck at least one unaffected case before promoting a task-local lesson to
   a shared `README.Agent.md` invariant.

## Active vertical cases

| order | case | observed failure | acceptance |
| --- | --- | --- | --- |
| 1 | `grab_roller` | official succeeded; generated TaskGen expert hook ended with `target_pose cannot be None` before policy rollout | Plan, TaskGen and Tool prompts retrieve one source-backed guide; retry prompts carry prior output; the frozen failure either passes or becomes an explicit evidence-conditioned new Proposal |
| 2 | `press_stapler` | repeated same-observable axis refinements ended inconclusive | Plan Agent switches to a distinct concern or actively stops when no new information remains |
| 3 | temporal VQA | identical frozen montage/question produced conflicting booleans | ordered-frame evidence is cited; insufficient evidence returns `observed=null`; repeated conflict remains explicit |

Do not pre-write guides for all RoboTwin tasks. Add one only when source facts or
an observed failure supply information that generic source/TaskContext retrieval
does not already expose. The breadth harness may characterize coverage, but it
does not determine this prompt curriculum and is not evidence for the paper's
dynamic-evaluation claim.
