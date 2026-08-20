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

## 2026-08-20 — live attempts v3 and v4

The scientific inputs remained frozen: the same anchor-free Query, checkpoint,
paired seed group `1000..1004`, `M=5`, and five-round allowance were used. Each
attempt is retained as an immutable failed artifact rather than resumed across
a provider failure.

- v3 (`eval_20260820_batch47_move_playingcard_anchorfree_m5_s1000_v3`)
  completed the official-control aggregate from exactly five policy episodes.
  Before any round-2 candidate policy episode, TaskGen visual validation
  received an upstream HTTP 429 response. Result: five failed-attempt policy
  episodes.
- v4 (`eval_20260820_batch47_move_playingcard_anchorfree_m5_s1000_v4`)
  completed three evidence rounds, each over the exact five-seed group:
  - round 1: official control, `5/5` policy successes;
  - round 2: Plan proposed an `x +0.01 m` perturbation, which produced `5/5`
    policy successes plus per-seed Rule evidence;
  - round 3: Plan proposed an orthogonal `y +0.05 m` perturbation, which
    produced `4/5` policy successes plus per-seed Rule evidence;
  - after consuming the round-3 mixed aggregate, Plan proposed a
    `y +0.025 m` boundary refinement for round 4. The provider returned HTTP
    429 before TaskGen produced the candidate, so round 4 consumed zero policy
    episodes.

v4 therefore consumed 15 failed-attempt policy episodes and is not a completed
scientific evaluation. Across all failed attempts, policy-episode consumption
is reported separately as `v1=0 + v2=5 + v3=5 + v4=15 = 25`.

## 2026-08-20 — provider retry root repair and AutoDL regression

The 429 failures exposed a production wiring bug rather than a missing retry
implementation. `OpenAICompatibleProvider` already defaults to
`max_retries=2` and `retry_delay=1`, giving at most three attempts with bounded
one- and two-second backoff. Standard `--auto-route` construction in
`mea/agent_query_routing.py` immediately overwrote the shared provider with
`max_retries=0`; because Plan, TaskGen, ToolGen, and visual validation reuse
that provider, the ordinary runtime silently disabled its existing retry
budget.

The minimal repair deletes only that override and retains the provider's
existing defaults. No outer evaluation retry, state-resume path, larger retry
budget, or alternate provider was added. A focused regression makes the first
two requests return HTTP 429 and verifies that the third succeeds after the
expected `1.0`- and `2.0`-second backoff calls.

AutoDL commands:

```bash
PYTHONPATH=. /root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q \
  tests/mainline/test_openai_compatible_provider.py
PYTHONPATH=. /root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q tests/mainline
```

Results: `3 passed`; then `231 passed, 28 subtests passed`. Both commands ran
on AutoDL. No test, import, compilation, simulator, provider, or policy command
ran on Windows PC.

## 2026-08-20 — completed v5 live evaluation and final audit

The provider credential was injected only into the auth-probe and launch
processes. A Terra text probe returned AUTH_OK with retry_count=0 before
launch. The frozen package retained the same Query, checkpoint, paired seeds
1000..1004, M=5, no-history setting, no-VQA instruction, five-round allowance,
and 7200-second wall bound. Only the new evaluation ID and repaired revision
changed:

~~~text
eval_20260820_batch47_move_playingcard_anchorfree_m5_s1000_v5
f03a93a19d0d561c21eac4284f1a7a851b5fc485
~~~

Launch command:

~~~bash
bash /root/autodl-tmp/run-packages/
  batch47_move_playingcard_anchorfree_m5_s1000_v5/launch.sh
~~~

The launcher verified HEAD=origin/main at the expected revision, a clean
tracked worktree, a new output/log directory, a bindable port 18771, the
resolved JSON configuration, and shell syntax before starting. Result:
agent exit 0, manifest status=completed, lifecycle_status=completed, no failure
stage, and three completed child runs.

The completed evidence chain was:

1. official control: exact seeds 1000..1004, 5/5 success, one aggregate;
2. fresh same-seed playingcards x +0.05 m Task: observed anchor delta
   0.049999999999999996 m, 4/5 success, and five finite
   minimum_terminal_abs_playingcards_x results;
3. after reading that 5/5 to 4/5 change and the Rule range, Plan proposed a
   total playingcards x +0.10 m follow-up; the fresh Task observed
   0.09999999999999999 m at the anchor seed, again produced 4/5 success, and
   returned five finite terminal_absolute_playingcards_x results.

Plan then stopped with supported/evidence_sufficient=true while two candidate
round allowances remained. Its Answer is byte-for-byte identical across the
query-answer file, evidence bundle, final answer, and top-level manifest.
Structured scope is sample_count=15, seeds 1000..1004, two tested candidates,
no conflict, termination=agent_stop.

The final assertion audit checked every summary and child manifest:

- requested=actual seeds 1000..1004 in all three rounds;
- five episode results and five MethodRuntime rollouts per child;
- materialization_anchor_seed=1000 and aggregate unique_episode_count=5;
- each requested Rule contains exactly five seed-indexed results;
- total_policy_episodes=15;
- full ordered evidence history is present in each next Plan prompt and the
  bound Proposal equals the materialized next-round Proposal;
- final Answer, verdict, stop, sample count, and seed group agree in all
  published artifacts.

Result: mechanical_audit=passed. The same scan found 44 retry_count=0 artifact
copies and two distinct logical calls with retry_count=1 (duplicated in their
transport copies): Plan after round 1 and TaskGen round 3. Thus the bounded
provider repair was exercised in the successful run. After completion the
Agent, supervisor, and policy server had exited, port 18771 was free, and
AutoDL HEAD/origin remained aligned with a clean tracked worktree.

Scientific accounting remains separate: v5 contributes 15 completed episodes.
The immutable failed attempts consumed 25 additional diagnostic episodes
(v1=0, v2=5, v3=5, v4=15) that are not included in v5's sample.
