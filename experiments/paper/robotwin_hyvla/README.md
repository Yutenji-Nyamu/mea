# RoboTwin / Hy-VLA pilot

This directory is a cold, explicit paper-experiment adapter. It verifies that the
official multi-task Hy-VLA RoboTwin policy can run behind a task-neutral transport;
it is not a second MEA production runtime.

## What is here

| File | Role |
| --- | --- |
| `download_checkpoint.py` / `download_checkpoint_server.sh` | Exact bounded server-side checkpoint downloader and network wrapper |
| `validate_install.py` | Offline official `encode_obs` and one-forward validation |
| `policy_server.py` | Hy-VLA process using the official wrapper and policy environment |
| `sim_client.py` | Generic RoboTwin task client using official reset/action/success APIs |
| `production_control.py` | One bound official control through the production MEA RoundExecutor |
| `transport.py` | Loopback-only length-prefixed message transport |
| `deployment_ledger.md` | Full cold installation, network, validation, and rollout ledger |

The task is a CLI value. None of these files contains a `press_stapler` branch or
changes the official policy/task source.

## Accepted result

Server paths and revisions are frozen in `deployment_ledger.md`. The first official
N=1 acceptance used `press_stapler`, `demo_clean`, seed `10000`:

- official `eval_success=true` and `check_success()=true`;
- 24 of 400 allowed actions, four actual network forwards;
- 41.15 s rollout wall time after model load;
- about 9.81 GB peak CUDA allocation on an RTX 4090.

This proves one runnable generalist-policy adapter and one official task success.
Production v9 separately completed a bounded official-control MEA round and cached
Answer finalization. Neither result establishes 50-task success, a policy ranking,
or generated TaskGen/ToolGen behavior.

## Production MEA binding

The production hook reuses the same explicitly started loopback server. MEA does
not start the Hy-VLA environment. Start `policy_server.py` in the Hy-VLA Python
environment with explicit `--source`, `--checkpoint`, `--ready-file`, and
`--summary-file`; then run the normal Agent command with:

```text
--policy-backend hyvla
--hyvla-source /root/autodl-tmp/third_party/Hy-Embodied-0.5-VLA
--hyvla-checkpoint /root/autodl-tmp/checkpoints/robotwin/hyvla_robotwin
--hyvla-python-env /root/autodl-tmp/envs/mea-hyvla
--hyvla-port 18781
```

For the bounded admission-free acceptance, invoke `production_control.py` with
the same explicit paths, port, and the server's `--ready-file`. It bypasses only
open-Query admission; binding, MethodRuntime, Rule Tool, Aggregate, and the
production round summary still execute.

The task remains runtime data (`--bound-task-name` or query routing). The hook
enters the existing Plan Agent → TaskGen/ToolGen → RoundExecutor → Aggregate loop;
this experiment directory does not acquire method orchestration or task-specific
logic. Production v9 reached policy success, official Rule reuse, Aggregate, Agent
stop, and QueryContract evidence sufficiency. Its first terminal summary call
failed with UIUI HTTP 503 model_not_found; provider-only cached finalization then
completed the same evaluation with 0 new rollout and unchanged cached-artifact
hashes. See the ledger for the exact boundary.

For the short repeat procedure, see
[`docs/robotwin_hyvla_reproduction_zh.md`](../../../docs/robotwin_hyvla_reproduction_zh.md).
