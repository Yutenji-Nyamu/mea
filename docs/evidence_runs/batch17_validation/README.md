# Batch 17 紧凑验证证据

本目录发布两份 0-ACT 冻结产物，目的是让文档中的机制结论可以直接审阅，而不是把服务器侧大规模运行目录提交进 Git。

- `method_coverage_report.json`：在 canonical server worktree 上运行的 16 项接口与最小 evidence-gate 审计；结果为 `16/16 implemented`，但其中真实 evidence validator 读取了服务器侧忽略的历史运行产物。
- `clean_head_click_counterfactual.json`：固定非 policy 证据，仅把第一轮 `policy_success` 改为 `0` 或 `1` 的确定性 PlanSession 重放；新 ACT 数为 0。失败分支留在 `object_position` 深挖，成功分支切到 `object_instance`。

可信边界：

- 两份文件都不是新 rollout，也不增加统计样本；
- coverage 的 `implemented` 只表示源码接口与声明的最小 evidence gate 通过，不等于论文方法语义或论文结论已经复现；
- counterfactual 只证明当前确定性 session policy 会随输入证据改变下一动作，不证明 LLM 独立产生同一反事实，也不证明原 Query 已有充分证据；
- fresh clone 未包含 `mea/evaluation_runs/`、`mea/validation_runs/` 等服务器原始大产物，因此可审阅这份冻结报告，但不能仅凭 clone 完整重算它。
