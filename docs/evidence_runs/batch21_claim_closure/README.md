# Batch21：低成本 claim closure

本目录发布 2026-07-24 batch21 的 compact、可审计结论。原始视频、telemetry、provider
响应、checkpoint 和运行日志只保存在 canonical AutoDL，不进入 Git；其路径与 SHA-256
记录在 `evidence_manifest.json`。

本批最重要的正向结果有两条：

1. `eval_20260724_batch21_semantic_needs_live_n1_v5` 用两个 ACT 完成
   official control → evidence-conditioned semantic Proposal → 新场景/实验 checker →
   新 typed metric → Aggregate → Answer。最终因候选未覆盖而诚实输出
   `inconclusive`。
2. `batch21_position_universal_n1` 的 independent fixed/adaptive arms 分别使用 2/1 个
   ACT，对有限 universal Query 得到相同 `refuted` 结论，并实测节省 58.151 秒。

这两条都只是 one-task/one-seed 的 mechanism evidence，不是论文规模结果。VQA 的
12 个 case 来自 3 个缓存 montage 和图像级扰动；Plan 使用 development-agent proxy；
Fig. 6 只有 5 个冻结 operation；ACT/DP 因缺少 DP checkpoint 和环境而没有启动任何
policy rollout。

详细实现、限制和下一步见
`docs/development_log_20260724_batch21_low_cost_claim_closure_zh.md`，当前总览见
`docs/evidence_snapshot_current.json`。根 `README.md` 未修改。
