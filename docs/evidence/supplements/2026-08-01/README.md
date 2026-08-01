# 2026-08-01 补充证据

本目录不替代 [`docs/evidence/current`](../../current/README.md) 的完整生成式
旗舰。它保存新增 policy backend 与跨任务运行所需的最小、可移动证据。

- [`hyvla_v9_control/`](hyvla_v9_control/README.md)：Hy-VLA 在无手写
  `TaskSchema` 的 `press_stapler` 上，经共享生产入口完成一轮官方任务、官方
  Rule Tool 精确复用、Aggregate、Plan Agent 主动停止和受限 Answer。

该结果只有一个 seed、一个 official episode；它证明 backend-neutral 方法运输层
可运行，不证明 50 任务成功率、策略排名或生成式 TaskGen 跨任务泛化。
