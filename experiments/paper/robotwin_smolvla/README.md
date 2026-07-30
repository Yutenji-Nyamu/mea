# RoboTwin SmolVLA paper experiment

This directory contains the post-run cleaned version of the isolated
two-process runtime used for the five-task deployment pilot documented in
`docs/robotwin_smolvla_reproduction_zh.md`.

- `policy_server.py` runs under Python 3.12 with LeRobot 0.6 and SmolVLA.
- `sim_client.py` runs under the existing Python 3.10 RoboTwin environment.
- The server binds only to `127.0.0.1`.
- The protocol transports raw image bytes plus public shape lists, a 14D state
  list, and a 50-by-14 action list. It deliberately avoids pickled NumPy arrays
  across NumPy 1.26 and 2.x.

The scripts do not contain a task allowlist. `--task` is resolved from
`envs.<task>` at runtime. The policy contract is intentionally fixed to the
checkpoint's three cameras, 14D state/action and 50-step chunk.

The published five-task evidence used two policy processes (one client, then
four ordered clients), all with simulator scene seed 1000. A policy process is
seeded once; `policy.reset()` does not reseed its diffusion RNG. Start one
`--max-clients 1` server per task for independent repeatability, or freeze the
client order when batching. The repository copy adds loopback, argument and
stale-ready-file checks after the run; server-side CLI/import validation covers
those edits, while the evidence episodes retain hashes of the exact temporary
runners that produced them.

The exact tested runners and raw command ledger are kept out of the active
context under [`history/20260729/`](history/20260729/README.md). They are
provenance only and must not be imported by the maintained runner.

Run the server:

```bash
python policy_server.py \
  --checkpoint /path/to/smolvla_robotwin \
  --backbone-metadata /path/to/SmolVLM2-500M-Video-Instruct-metadata \
  --ready-file /tmp/smolvla.ready.json \
  --max-clients 1
```

Run one simulator client from the RoboTwin repository:

```bash
PYTHONPATH=/path/to/RoboTwin python sim_client.py \
  --task beat_block_hammer \
  --seed 1000 \
  --output-dir /path/to/output
```

These are experiment-layer adapters, not a second MEA production orchestration
path. A production policy backend should reuse the same observation/action
contract through the shared MethodRuntime interface.
