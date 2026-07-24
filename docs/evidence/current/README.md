# 当前紧凑证据：ClaimFirst 主链

当前发布对应 `eval_20260724_batch21_semantic_needs_live_n1_v5`。它真实完成：

```text
开放 Query
→ official control Proposal
→ ACT rollout 成功
→ evidence-conditioned appearance Proposal
→ TaskGen 新蓝色场景 + 实验 checker
→ render / gate
→ 新 XY 距离 Tool
→ 第二次 ACT
→ Rule/VQA/Aggregate
→ 受限 Answer
```

结果为 `inconclusive`：两轮中 control 成功、生成候选失败，但 timing 候选未测且预算耗尽。
它证明主链的机制能够运行，不证明属性泛化，也不进入论文表格。

Git 只保留 [manifest.json](manifest.json)。raw 视频、telemetry、generated code、render、
provider response 和完整回答位于 canonical AutoDL 的：

```text
/root/autodl-tmp/mea/mea/evaluation_runs/
  eval_20260724_batch21_semantic_needs_live_n1_v5/
```

manifest 按 `Query → Proposal → generated artifacts → render → rollout →
Rule/VQA → Aggregate → Answer` 给出路径。需要查看完整证据时在服务器依次打开这些文件，
不要从历史开发日志拼接结论。
