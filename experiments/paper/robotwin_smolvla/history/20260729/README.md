# 2026-07-29 tested-run provenance

Cold archive for the exact temporary runners and raw command ledger used by the
five-task RoboTwin/SmolVLA deployment pilot.

- `command_ledger.md` records server downloads, environment setup, probes,
  failures and rollout commands.
- `policy_server.py` SHA-256:
  `e6291b936f08b68f371a0421fd577da6bb0f5b829db6f76df152cdb9bbb9f9b0`
- `robotwin_client.py` SHA-256:
  `1aa619c81e3685f9505303aeca43eb86624b41d59ca9998e2412dae6d56afd56`

These files are immutable evidence, not active entry points. Use the cleaned
`../../policy_server.py` and `../../sim_client.py` for future runs. Read
`docs/robotwin_smolvla_reproduction_zh.md` first; it is the maintained
reproduction guide and explains what this pilot does and does not establish.
