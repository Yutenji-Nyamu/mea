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

此后生产 `RoundExecutor` 已同时接入 ACT 与 SmolVLA，SmolVLA 也进入通用 TaskGen。
`eval_20260730_native_smolvla_broad_live_v10` 与
`eval_20260731_native_smolvla_broad_live_v11` 都已产生真实 official-control policy
episode，并在该 evidence 之后由 Plan Agent 提出新的 round-2 Proposal；因此 v5
“没有 observation/episode”的结论已过时。

这两次仍不是 clean flagship 正证据。v10 在 round 2 的 TaskGen 中先遇到非法
`.extend`，修复后又因全尺寸 distractor 阻断 expert official success 而终止；同一
Proposal 的零-policy replay 随后通过一次局部 repair 将 distractor 缩至 `0.1`，使模型
编写的 scene+checker 通过 AST、`2/2` fixtures、render/VLM 和 expert gate。v11 的
round-2 checker 要求 terminal state 中仍存在瞬时 PhysX contact，首次生成和一次 repair
均未通过 expert positive fixture；后续零-policy replay 虽改用“夹爪闭合”而通过执行
gate，但这只是 contact need 的相关代理，语义上不等价，必须拒绝。故当前最重要的缺口
是 **Proposal-checker 语义忠实性与完整同链闭合**，而不是再证明 transport 可以启动
rollout。当前代码已新增独立 development-agent checker semantic review，逐项核对
量词/同时性/对象关系/直接观测，拒绝相关代理，并把批准结果与 Proposal、checker 哈希
绑定到 registry 和 pre-policy gate；AutoDL 主干回归通过，但尚未形成新的 live 正例。
生产结构也已收束为 `PlanAgentApplication` 直接拥有
round → evidence → next/stop → Answer 生命周期，并让
`MethodRuntime.materialize_candidate()` 成为唯一生产 TaskGen materialization
owner；对应 AutoDL 回归为 `124/124` 聚焦测试和 `299/299 + 16 subtests` 默认主干。
这些是代码级验收，不替代下一次 live flagship。
历史运行只在 [`docs/evidence/history.jsonl`](evidence/history.jsonl)保留摘要。

## 方法 claim

| 论文 claim | 当前项目 | 判断 |
| --- | --- | --- |
| Fig. 2/5：开放 Query 驱动 Plan Agent 自主提出 sub-aspect | 生产入口不调用 catalog/task-specific planner；v10/v11 均由 broad Query 的 official-control evidence 触发新 Proposal | **小范围完成**；仍缺同一运行中的后续成功 round 与充分停止 |
| 上一轮 evidence 决定下一轮，并在充分时停止 | v10/v11 已证明 evidence 后才提出新 sub-aspect；有限合同的 sufficient stop 另有真实案例 | **尚未在同一 broad flagship 合一** |
| Fig. 3：Proposal → retrieve/generate scene + `check_success()` → rollout | v10 零-policy replay 中，模型编写的 scene+checker 经一次 repair 后通过执行 gate；v11 代理实现促成了独立语义审查与哈希绑定 | **组合正例缺失**；新语义门尚无同一 evaluation 的 live policy episode |
| 首帧视觉诊断与局部重新生成 | v10 已有一次 repair 后通过 AST、fixture、render/VLM 与 expert gate 的正例 | **组件正例完成**；视觉不能替代 checker 的 simulator-state 语义审计 |
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
   Plan Agent 在 completed evidence 后自选可实现 concern；按需生成 scene、实验 checker
   与单独声明的 scalar Tool；同一 evaluation 中通过 TaskGen gate、执行轻量 rollout，
   使 live Tool 非空并进入 Aggregate。
2. **让 evidence 同时决定 refinement 与停止。** 下一 Proposal 必须在上一轮 completed
   evidence 后产生；Plan Agent 主动提出 stop，QueryContract 只验证，不替它决策。
3. **用 live TaskGen 验收通用 checker 语义门。** 代码已做到 fixture 与语义审查分责；
   下一次运行必须证明忠实 checker 被批准、相关代理被拒绝，且批准哈希在 registry 和
   policy 前保持一致。若 expert terminal state 不支持精确 predicate，应报告
   unsupported，而不是改写语义。
4. **在同一方法链证明 Tool reuse。** 新 Tool 影响下一轮后，由第二 Query exact reuse。
   对没有 caller-supplied independent numeric oracle 的 derived observable，ToolGen 使用
   separate development-agent semantic review，再执行 declared-signal AST、determinism、
   finite/unit、evidence-step 与 artifact-immutability runtime gates；产物必须显式记录
   `independent_numeric_oracle=false`、`oracle_agreement=null`，且只能作为诊断量，不能
   拥有 success/reward authority。存在独立 numeric oracle 时才报告其 agreement。
5. **统一 LIBERO 外层 loop。** simulator backend 可以不同，QueryContract、Plan Agent
   session、RoundExecutor、Aggregate、stop 与 Answer 不应重复实现。
6. **方法稳定后再扩大实验。** 首先做小型三 seed dense/adaptive 保真；独立人工
   Plan/VQA、多 policy ranking 和大规模任务后置。

## 软件工程边界

- 生产只保留 Plan Agent 主链；fixed/catalog/task-specific planner 属于
  `experiments/paper/` 或 compat。
- task identity、schema、policy binding 与 runtime hooks 不得重新膨胀成
  aspect/metric/Planner 菜单。
- TaskGen、ToolGen 各只保留一次局部 repair；失败不触发中央 whole-round restart。
- registry caller 迁移完成前不直接删除旧持久层，也不继续增加 façade。
- `PlanAgentSession` 是唯一公开生产 session；旧 execution session 只保留内部冻结
  运输和历史 reader 兼容。
- 动态运行真值只维护在本文与 `docs/evidence/current/`；安装和失败排查放 cold
  runbook，历史结果放 `docs/evidence/history.jsonl`。
