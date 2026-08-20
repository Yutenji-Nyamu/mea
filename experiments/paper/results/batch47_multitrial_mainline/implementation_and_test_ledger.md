# Batch47 multi-trial mainline implementation and test ledger

This cold ledger records the implementation and server validation of the
paper-aligned within-task policy-trial path. It contains no credential.

## 2026-08-20 — source and method audit (Windows PC, read-only)

- Workspace source: `C:\Users\86136\Documents\mea\worktree`.
- Starting revision: `9d3b517495eaf3585ff88d213c7761141b6986af`.
- Starting tracked state: clean.
- No test, import, compile, provider, simulator, or policy command was run on
  Windows.
- The ICLR 2026 proceedings paper was checked directly. Section 3.1 separates
  adaptive sub-aspects/rounds from the per-task trajectory index, and Appendix
  A.1.1 states that each constructed task executes five trials by default.
  Primary source:
  <https://proceedings.iclr.cc/paper_files/paper/2026/file/8833c8aa10542d24d693bbaf6a4598f5-Paper-Conference.pdf>.
- The paper does not prescribe an official-control protocol or paired
  control/candidate seeds. This implementation uses one shared exact seed group
  for fair comparison and records that choice separately from the paper claim.

## Implementation batch (Windows PC, source edits only)

Planned/implemented contraction and extension:

1. CLI/default execution
   - change ordinary `--num-episodes` default from 1 to 5;
   - retain explicit `M=1` only for transport/mechanism debugging;
   - keep adaptive Plan round allowance independent from `M`.
2. Native RoboTwin execution
   - accept one non-empty exact seed group;
   - materialize the official/generated Task once, anchored at the first seed;
   - execute the same candidate once per exact seed with isolated artifacts;
   - reject duplicate, missing, negative, or substituted seeds;
   - persist one aggregate success rate and one method-round evidence object.
3. Policy RNG pairing
   - include the trial seed in SmolVLA/Hy-VLA reset requests;
   - reset Torch/NumPy RNG before each policy reset;
   - record server-side trial seeds in the terminal server summary.
4. Evidence shown to Plan
   - preserve exact seed group, trial count, and aggregate success rate;
   - distinguish full success, full failure, and a mixed rate;
   - treat a finite mixed aggregate as completed decisive evidence whose Query
     meaning remains owned by Plan;
   - project numeric Rule results as one compact aggregate for the prompt while
     retaining raw episode artifacts outside the prompt.
5. Control semantics
   - require an authoritative completed official aggregate;
   - do not require a perfect `1.0` control rate as a transport gate.
6. Documentation and durable rules
   - update the root example to `M=5`;
   - retain Batch46 unchanged as a frozen historical `M=1` method proof and
     mark its statistical limitation;
   - record the round/trial distinction in workspace rules and long-term
     memory.

## AutoDL validation plan

Exact commands and outputs will be appended after the source batch is synced.
The intended sequence is:

1. run only the precise multi-trial native/evidence/CLI nodes needed to locate
   integration failures;
2. run default `tests/mainline` once;
3. commit and push only after both pass and the remote/worktree state is clean;
4. run a fresh anchor-free SmolVLA evaluation with seeds `1000..1004`, `M=5`,
   one aggregate per round, no history, and an adaptive round allowance;
5. report final successful-run episodes separately from episodes consumed by
   failed attempts.

## 2026-08-20 — AutoDL regression

Server repository:
`/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime`, starting from
`9d3b517495eaf3585ff88d213c7761141b6986af`.

Precise multi-trial regression command:

```bash
PYTHONPATH=. /root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q \
  tests/mainline/test_policy_backends.py::test_native_control_runs_exact_paired_seeds_and_aggregates_once \
  tests/mainline/test_policy_backends.py::test_native_method_evidence_uses_trusted_generated_checker_result \
  tests/mainline/test_agent_evidence_integration.py::AgentEvidenceIntegrationTests::test_plan_record_uses_one_five_trial_rule_aggregate \
  tests/mainline/test_agent_evidence_integration.py::AgentEvidenceIntegrationTests::test_partial_trial_rate_is_one_decisive_mixed_round \
  tests/mainline/test_agent_evidence_integration.py::AgentEvidenceIntegrationTests::test_completed_round_rejects_substituted_trial_seed \
  tests/mainline/test_plan_agent_session.py::PlanAgentRuntimeTests::test_control_identity_is_tracked_without_forcing_budgeted_stop \
  tests/mainline/test_plan_agent_session.py::PlanAgentRuntimeTests::test_zero_rate_control_is_valid_and_can_schedule_an_unchanged_retry \
  tests/mainline/test_plan_runtime_limits.py::PlanAgentRuntimeLimitsTests::test_partial_trial_aggregate_is_decisive_without_becoming_pass_or_fail \
  tests/mainline/test_production_cli_boundary.py::ProductionCliBoundaryTests::test_policy_trials_default_to_five_and_allow_explicit_debug_one
```

Result: `9 passed, 7 subtests passed in 0.47s`.

Default mainline command:

```bash
PYTHONPATH=. /root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q tests/mainline
```

Result: `229 passed, 28 subtests passed in 15.07s`.

Both commands ran on AutoDL. No test, import, compilation, simulator, provider,
or policy command ran on Windows PC.

## 2026-08-20 — first live attempts and observed integration failure

The frozen Query, checkpoint, seeds `1000..1004`, `M=5`, five-round allowance,
models, and stopping conditions remained unchanged across attempts.

- v1 (`eval_20260820_batch47_move_playingcard_anchorfree_m5_s1000_v1`)
  failed before any policy episode. The run package had not changed into the
  repository working directory, so RoboTwin could not resolve the relative
  `assets/objects/objaverse/list.json` path. Result: zero completed rounds and
  zero policy episodes. v1 remains an immutable failed artifact.
- v2 (`eval_20260820_batch47_move_playingcard_anchorfree_m5_s1000_v2`)
  corrected only the working directory. It executed exactly five official
  control episodes on seeds `1000..1004`; every raw result reported
  `official_check_success=true`. The run then failed before round evidence was
  assembled because Rule Tool discovery still assumed the historical
  two-level telemetry layout. Result: zero completed rounds and five
  failed-attempt policy episodes. These episodes are diagnostic artifacts, not
  part of a final scientific sample.

Root repair:

- `mea/toolgen/tool_execution.py` now discovers episode metadata recursively
  below the current child run's `evaluation/telemetry` root.
- `mea/toolgen/tool_routing.py` uses the same bounded recursive discovery for
  typed MetricSpec execution.
- The new regression builds two isolated per-trial directories and verifies one
  Rule request sees both exact seeds and outcomes.

AutoDL regression after the repair:

```bash
PYTHONPATH=. /root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q \
  tests/mainline/test_tool_orchestration.py::ToolOrchestrationTests::test_auto_router_reads_isolated_multi_trial_episode_directories
PYTHONPATH=. /root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q tests/mainline
```

Results: `1 passed in 0.26s`; then
`230 passed, 28 subtests passed in 15.00s`.

The independent typed-MetricSpec route was then moved to the same nested
per-trial fixture and checked directly:

```bash
PYTHONPATH=. /root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q \
  tests/mainline/test_open_python_toolgen.py::OpenPythonToolGenTests::test_orchestration_labels_runtime_validation_without_numeric_oracle
PYTHONPATH=. /root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q tests/mainline
```

Results: `1 passed in 0.88s`; then
`230 passed, 28 subtests passed in 15.01s`. Both ran on AutoDL.
