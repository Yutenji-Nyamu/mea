# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–5、Tables 1–10 与 Appendix。这里严格区分：

- **方法入口存在**：代码能表达该组件；
- **真实小闭环**：组件在 simulator/policy rollout 中形成可审计数据流；
- **论文实验复现**：任务、policy、样本、人工标注与统计协议达到论文口径。

## 当前真值

当前可发布正证据仍是
[`eval_20260730_batch31_grab_roller_broad_live_v3`](evidence/current/README.md)：开放
Query 经 official control 后由 Plan Agent 自选 scale concern，再依据 0.85 evidence
细化到 0.70；通用 TaskGen、Rule Tool、Aggregate 与受限 Answer 真实运行。它只有一个
task/seed，最终因预算停止，不能证明广泛泛化或 evidence-sufficient stop。

此后生产 `RoundExecutor` 已同时接入 ACT 与 SmolVLA，SmolVLA 也进入通用 TaskGen；
但最新 scene+checker live 在 TaskGen 验证阶段终止、没有启动 policy rollout。因此这
只是代码统一和负向诊断，不是新的方法正证据。旧运行结论只在
[`docs/evidence/history.jsonl`](evidence/history.jsonl)保留一行摘要，不在本文展开。

## 方法 claim

| 论文 claim | 当前项目 | 判断 |
| --- | --- | --- |
| Fig. 2/5：开放 Query 驱动 Plan Agent 自主提出 sub-aspect | 生产入口不调用 catalog/task-specific planner；现有 live 已从 broad Query 自选并细化 scale concern | **小范围完成**；仍只有单 task/seed 与单 concern 方向 |
| 上一轮 evidence 决定下一轮，并在充分时停止 | evidence-conditioned refinement 与有限合同的 sufficient stop 各有真实案例 | **尚未在同一 broad flagship 合一** |
| Fig. 3：Proposal → retrieve/generate scene + `check_success()` → rollout | 通用 scene generation 有多轮正例，实验 checker 有独立正例；SmolVLA 路径已接线 | **组合正例缺失**；当前 SmolVLA scene+checker 在 rollout 前失败 |
| 首帧视觉诊断与局部重新生成 | VLM、simulator/fixture、render、expert 与一次 repair 已接入 | **机制存在**；缺一次 repair 后通过的 clean live 正例 |
| Fig. 4：ToolGen retrieve/generate/validate/register/reuse | Python Rule Tool 已在 live 取得非空值；exact reuse 另有 0-rollout 案例 | **部分完成**；“live 值 → 影响下一轮 → 第二 Query exact reuse”尚未同链证明 |
| rollout → Rule/VQA → Aggregate → Plan Agent → Answer | RoboTwin 小范围已闭环；ACT/SmolVLA 共用 `RoundExecutor` | **RoboTwin 基本完成**；LIBERO 仍是独立外层 chain |
| 回答原 Query 并约束确定性 | `AnswerScope` 报告 N、seed、未覆盖项、冲突与停止原因 | **完成度较高**；不是统计泛化保证 |

## 实验 claim

| 论文 claim | 当前证据 | 判断 |
| --- | --- | --- |
| Tables 1–2：更少 samples/time 保持 dense 结论 | 三 seed toy 从 12 降到 6 ACT，但完整 failure set 仅保持 2/3 | **真实节省、结论不完全一致** |
| Table 3：RAG、visual self-check、README.Agent 提升 codegen | 五个 frozen Proposal 的小型消融，只有 RAG 有方向信号 | **未复现结论** |
| Table 6：Plan 与机器人研究者一致 | 30 条 development-agent proxy；无人类 gold | **未复现** |
| Tables 7–8：VQA 四条件 accuracy/AUROC | 8 个缓存样本、单 VLM、proxy gold | **仅协议 smoke** |
| Table 9：少样本保持多 policy 排名 | ACT/DP3 三 seed 为 2/3 对 2/3，Spearman 不可算 | **未复现** |
| Fig. 6：系统错误率与模块分布 | 固定 operation 分母很小 | **不可比较** |
| RoboTwin/LIBERO 跨任务适配 | SmolVLA 可低成本扩 official rollout；LIBERO 有 basic adaptation | **方法外层与生成式证据尚未跨环境统一** |

## 当前方法优先级

1. **完成一个 backend-neutral clean flagship。** broad Query 不给 aspect/template；
   Plan Agent 自选可实现 concern；按需生成 scene、实验 checker 与独立 scalar Tool；
   通过 TaskGen gate 后执行一次轻量 policy rollout；live Tool 非空并进入 Aggregate。
2. **让 evidence 同时决定 refinement 与停止。** 下一 Proposal 必须在上一轮 completed
   evidence 后产生；Plan Agent 主动提出 stop，QueryContract 只验证，不替它决策。
3. **在同一方法链证明 Tool reuse。** 新 Tool 影响下一轮后，由第二 Query exact reuse；
   无独立 oracle 时诚实返回 unsupported，不回退到相近 MetricSpec。
4. **统一 LIBERO 外层 loop。** simulator backend 可以不同，QueryContract、Plan Agent
   session、RoundExecutor、Aggregate、stop 与 Answer 不应重复实现。
5. **方法稳定后再扩大实验。** 首先做小型三 seed dense/adaptive 保真；独立人工
   Plan/VQA、多 policy ranking 和大规模任务后置。

## 软件工程边界

- 生产只保留 Plan Agent 主链；fixed/catalog/task-specific planner 属于
  `experiments/paper/` 或 compat。
- task identity、schema、policy binding 与 runtime hooks 不得重新膨胀成
  aspect/metric/Planner 菜单。
- TaskGen、ToolGen 各只保留一次局部 repair；失败不触发中央 whole-round restart。
- registry caller 迁移完成前不直接删除旧持久层，也不继续增加 façade。
- 动态运行真值只维护在本文与 `docs/evidence/current/`；安装和失败排查放 cold
  runbook，历史结果放 `docs/evidence/history.jsonl`。
