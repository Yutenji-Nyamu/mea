# 2026-07-31 方法诊断补充证据

本目录不替代 [`docs/evidence/current`](../../current/README.md) 的唯一 accepted
bundle。它只保存 v10/v11 暴露方法缺口所需的最小证据；完整 raw evaluation 与
telemetry 留在 AutoDL。

- `v10_full/summary.json`：真实 official-control episode 后产生新 Proposal，但完整运行
  在 round-2 TaskGen 失败。
- `v10_taskgen_repair/`：同一 Proposal 的 `0`-policy bounded repair 组件正例。
- `v11_full/`：上一轮 evidence 触发的 Plan Agent 决策、round-2 Proposal 与失败边界。
- `v11_posthoc_rejected/`：执行 gate 通过但语义漂移的 checker，以及撤销 registry 准入的
  development-agent proxy 审计。

这些文件证明“为什么修改方法”，不构成新的 policy-performance 结论。
