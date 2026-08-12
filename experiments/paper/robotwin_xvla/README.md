# RoboTwin / X-VLA (paper experiment adapter)

This cold experiment directory records the isolated official deployment of
`2toINF/X-VLA-RoboTwin2`. It is a policy backend feasibility artifact, not a
second MEA method runtime and not evidence that all 50 tasks succeed.

- [Complete deployment ledger](deployment_ledger.md)
- [Offline model-load/action validator](validate_install.py)
- concise Chinese runbook:
  [`docs/robotwin_xvla_reproduction_zh.md`](../../../docs/robotwin_xvla_reproduction_zh.md)

The model process and RoboTwin simulator should remain isolated when a live
adapter is added. Reuse the shared MEA `MethodRuntime` outside this directory;
do not copy Plan Agent, TaskGen, ToolGen, Aggregate, or Answer logic here.
