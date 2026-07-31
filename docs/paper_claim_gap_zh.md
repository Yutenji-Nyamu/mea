# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–5、Tables 1–10 与 Appendix。本文只维护当前真值；
历史运行摘要见 [`docs/evidence/history.jsonl`](evidence/history.jsonl)。

## 当前方法真值

当前证据入口是
[`eval_20260731_batch32_clean_flagship_live_v18`](evidence/current/README.md)。
这是首个把论文方法主干合入同一真实 RoboTwin bundle 的小范围正例：

1. broad Query 未给 aspect/template；official control 成功后，Plan Agent 才选择
   `grab_roller` 的世界 x 轴 `+0.05 m` 变化。
2. 通用 TaskGen 由 provider 编写 scene 与实验 checker；AST、`2/2` fixture、
   render/VLM、expert 与 preservation gate 均通过。
3. SmolVLA 在新场景仍满足 official core，但未满足“official core 且双侧 terminal
   TCP-to-contact 距离均不超过 `0.025 m`”的实验谓词。
4. ToolGen 生成 `terminal_max_tcp_contact_distance`，从真实 telemetry 得到
   `0.24384725093841553 m`，进入 Aggregate；VQA 无证据冲突。
5. Plan Agent 主动提出 stop，QueryContract 验证
   `evidence_sufficient=true / counterexample_found`，随后回答原 Query。
6. 一个独立 follow-up Query 在零 rollout、零 provider 下命中同一 registration，
   `run_local_reuse` 得到相同数值；registry 与原始 summary 字节未改变。

本次只有两个 policy episode、一个 seed `100401`、一个任务与一个生成候选。
generated checker 是有界实验语义，不是官方 benchmark checker；因此这是方法链
验收，不是统计泛化、benchmark 排名或论文全部实验结论。原始 v18 summary 在运行时
仍使用旧 acceptance 投影而显示 false；当前代码的 append-only 重投影为 true，两者均
保留在证据包。原始 final-summary 还多写了一条“缺少轨迹峰值”的保守限制，而 Query
只要求从 rollout telemetry 得到一个诊断标量；提示词已由该失败例修正，原始产物未被
回写。

## 方法 claim

| 论文 claim | 当前项目 | 判断 |
| --- | --- | --- |
| Fig. 2/5：开放 Query 驱动 Plan Agent 自主提出 sub-aspect | v18 在 completed control evidence 后才选择轴与精确位移；catalog 对模型不可见 | **小范围单例完成**；Query 仍限定了 terminal-alignment concern，尚非开放弱点搜索 |
| evidence 决定下一轮，并在充分时停止 | v18 为 `continue → 新 Proposal → Agent stop → QueryContract 验证` | **小范围单例完成**；有限 existential 合同不代表广泛充分性 |
| Fig. 3：Proposal → retrieve/generate scene + `check_success()` → rollout | v18 同链生成 scene/checker 并裁决真实 policy episode | **小范围单例完成**；跨任务冷启动仍受 TaskContext/schema 与 simulator hook 限制 |
| 首帧视觉诊断与局部重新生成 | render/VLM 与一次局部 repair 路径已在历史真实案例触发；v18 无需 repair | **组件完成**；视觉只审外观，数值关系仍由 simulator/fixture 审计 |
| Fig. 4：ToolGen retrieve/generate/validate/register/reuse | v18 新 Python Tool 有 live finite 值；独立 follow-up Query exact reuse | **同一 evaluation 内完成**；尚未证明 reviewed registry 的跨 evaluation 长期复用 |
| rollout → Rule/VQA → Aggregate → Plan Agent → Answer | RoboTwin 的 SmolVLA/ACT 共用 `RoundExecutor` 与方法外层 | **RoboTwin 小范围完成**；LIBERO 仍有独立外层 |
| 回答原 Query 并约束确定性 | `AnswerScope` 报告 N、seed、候选域、冲突、停止原因与语义边界 | **完成度较高**；还需避免模型凭空收紧 Query 子要求 |

## 实验 claim

| 论文 claim | 当前证据 | 判断 |
| --- | --- | --- |
| Tables 1–2：更少 samples/time 保持 dense 结论 | 三 seed toy 从 12 降到 6 ACT，但完整 failure set 仅保持 2/3 | **真实节省、结论不完全一致** |
| Table 3：RAG、visual self-check、README.Agent 提升 codegen | 五个 frozen Proposal 的小型消融，只有 RAG 有方向信号 | **未复现结论** |
| Table 6：Plan 与机器人研究者一致 | 30 条 development-agent proxy；无人类 gold | **未复现** |
| Tables 7–8：VQA 四条件 accuracy/AUROC | 8 个缓存样本、单 VLM、proxy gold | **仅协议 smoke** |
| Table 9：少样本保持多 policy 排名 | ACT/DP3 三 seed 为 2/3 对 2/3，Spearman 不可算 | **未复现** |
| Fig. 6：系统错误率与模块分布 | 固定 operation 分母很小 | **不可比较** |
| RoboTwin/LIBERO 跨任务适配 | SmolVLA 可低成本扩 official rollout；LIBERO 有 basic adaptation | **完整方法外层尚未跨环境统一** |

## 下一步主干

1. **解除跨任务 TaskContext 准入。** 将手写 TaskSchema 降级为缓存；缺失时从 official
   source、reset actors 与 telemetry retrieve/generate/validate，禁止新增任务名分支。
2. **把 LIBERO 接入同一方法外层。** simulator backend 可以不同，但
   QueryContract、Plan Agent session、RoundExecutor、Aggregate、stop 与 Answer
   不应重复实现。
3. **继续收束生产结构。** `PlanAgentApplication` 拥有 route/round/stop/answer；
   `MethodRuntime` 是唯一 TaskGen materialization owner。迁移 caller 后再删除 legacy
   planner、任务方言和重复 registry，不再增加 façade。
4. **方法稳定后补实验。** 先做小型三 seed dense/adaptive 保真；独立人工 Plan/VQA、
   多 policy ranking 与大规模任务后置。

## 软件工程边界

- 生产只保留 Plan Agent 主链；fixed/catalog/task-specific planner 属于
  `experiments/paper/` 或 compat。
- Task binding 只保存 task/checkpoint/schema/official-success/runtime hooks，不承载
  aspect、metric 或 Planner 菜单。
- TaskGen、ToolGen 各只保留一次局部 repair；失败不触发中央 whole-round restart。
- 动态运行真值只维护在本文与 `docs/evidence/current/`；安装/网络故障放 cold
  runbook，历史结果放 `docs/evidence/history.jsonl`。
