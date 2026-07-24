# ManipEvalAgent 文档

根 `README.md` 保持上游项目说明，不承载本复现实验日志。当前文档只保留以下入口：

- [架构与干净数据流](architecture_and_dataflow_zh.md)：当前唯一生产主链及每轮证据结构。
- [运行指南](running_guide_zh.md)：plan-only、live rollout 与证据查看。
- [论文 claim 与 gap](paper_claim_gap_zh.md)：论文声称、当前证据和下一步。
- [开发者参考](developer_reference_zh.md)：扩展任务、生成器和工具时的最小接口。
- [当前证据](evidence/current/README.md)：最近一次可审计运行的紧凑索引。
- [历史索引](evidence/history.jsonl)：旧批次只保留结论、边界和 Git revision。

原始视频、telemetry、provider 请求/响应及 checkpoint 体积较大，只保存在 canonical
AutoDL。Git 中的 current manifest 给出服务器相对路径和摘要；旧开发日志与重复 evidence
bundle 可通过 Git 历史恢复。
