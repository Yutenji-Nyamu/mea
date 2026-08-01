# ManipEvalAgent 文档

根 `README.md` 保持上游项目说明，不承载本复现实验日志。当前文档只保留以下入口：

- [架构与干净数据流](architecture_and_dataflow_zh.md)：`--auto-route` 开放 Query 的
  Plan Agent、Query interpretation、Proposal、Plan Agent session 主链及每轮证据结构；
  旧 catalog/fixed 入口仅作兼容与论文消融。
- [运行指南](running_guide_zh.md)：plan-only、live rollout 与证据查看。
- [LIBERO / SmolVLA 复现与 MEA 接入](libero_smolvla_reproduction_zh.md)：服务器安装、镜像/限流问题、seed/RNG parity、official/custom 两回合方法链及复现协议；这是长期保留、按需读取的 cold reference。
- [RoboTwin / SmolVLA 复现](robotwin_smolvla_reproduction_zh.md)：checkpoint 固定与校验、服务器网络、Python/NumPy 隔离、双进程 IPC、policy adapter 协议和完整回滚边界；这是长期保留、按需读取的 cold reference。
- [RoboTwin / Hy-VLA 复现](robotwin_hyvla_reproduction_zh.md)：第二个多任务 policy 的服务器部署、版本固定、隔离环境、official N=1 验收与 MEA 接入边界；完整逐命令流水保存在 paper 实验层。
- [论文 claim 与 gap](paper_claim_gap_zh.md)：论文声称、当前证据和下一步。
- [开发者参考](developer_reference_zh.md)：扩展任务、生成器和工具时的最小接口。
- [当前证据](evidence/current/README.md)：当前接受的可发布运行的紧凑索引。
- [历史索引](evidence/history.jsonl)：旧运行只保留结论、边界和 revision。

## 上下文路由

- **Hot**：本索引、架构、运行指南、paper claim/gap；方法开发再读取
  `mea/{planner,taskgen,toolgen,robotwin}` 与两个生产 CLI。
- **Warm**：开发者参考、当前证据摘要、与当前模块对应的 `README.Agent.md`。
  `README.Agent.md` 是运行时 prompt 与论文消融组件，不能作为普通文档删除。
- **Cold**：环境安装流水账、`evidence/current/artifacts|assets`、
  [`2026-07-31 negative supplements`](evidence/supplements/2026-07-31/)、
  [`Batch33 方法与多任务 policy 结果`](../experiments/paper/results/batch33_open_cross_task/README.md)、
  `experiments/paper/inputs|results`、vendor policy 文档和 tests。仅在复现、审计、
  消融或定位 caller 时按索引打开；paper 实验层入口见
  [`experiments/paper/README.md`](../experiments/paper/README.md)。
- **默认忽略**：`.git`、`tmp`、`__pycache__` 和服务器原始 runtime 目录；它们不是
  论文方法上下文。

Git 的 current bundle 只保留最近一次运行的短视频、render、生成代码、关键
provider 输出和结论。完整 telemetry/VQA bundle、其他 provider 中间结果与 checkpoint
只保存在 canonical AutoDL；旧开发日志与重复 evidence bundle 可通过 Git 历史恢复。
