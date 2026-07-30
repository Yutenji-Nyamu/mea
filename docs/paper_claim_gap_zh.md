# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–6、Tables 1–10 与 Appendix。这里严格区分：

- **方法入口存在**：代码可以表达论文组件；
- **真实小闭环**：组件在 simulator/policy rollout 上形成过可审计数据流；
- **论文实验复现**：样本、任务、策略、人工标注与统计协议达到论文口径。

本文统一使用论文术语 Plan Agent、Query interpretation、Proposal 与 Plan Agent
session。代码和不可变历史 artifact 中的 `ClaimFirst`、`FreeConcern`、
`ExperimentCandidate` 仅是兼容旧名称，不代表三套额外的方法组件。

当前方法主证据是
[current evidence](evidence/current/README.md) 对应的
`eval_20260730_b44_grab_roller_plan_agent_live13`。Query 未提供
aspect/template，但已经限定一个可证伪场景和实验成功语义；round 1 仅执行 official
control。Plan Agent 在 control 通过后才 materialize 这个 Query-derived Proposal，并由同一
Proposal 生成新 scene 与新 `check_success()`、生成高度差 Rule Tool、完成
render/VLM/expert、第二次 ACT、Aggregate、充分性停止与最终 Answer。实验 checker
和 official core 均为真；目标/非目标终态高度为 `0.80005/0.74166 m`，新 Tool
测得 `0.05838 m`。随后以已完成 telemetry 做 0-ACT 审计，第二次请求走
`run_local_reuse` 且未调用 provider。

这次运行也留下了有价值的模型失败：Query interpretation 与下一步 Plan Agent
首答都漏掉 checker；精简的失败提示和现有一次语义反馈使第二答修正，而不是新增任务
方言。它证明的是单 task、单 seed、有限 existential Query 的完整方法闭环，不是统计
泛化或跨 Query registry reuse。

## 方法 claim

| 论文 claim | 当前项目证据 | 判断 |
| --- | --- | --- |
| Fig. 2/5：开放 Query 驱动，Plan Agent 自主提出 sub-aspect | 生产入口由 `PlanAgentInitialPlanBuilder` 直接建立首轮计划，不调用 catalog/task-specific planner。live13 未给 aspect/template，但 Query 已完整描述唯一候选；系统将其解释并 materialize 为 `non_target_proximity_effect` | **Query-derived Proposal 正例完成，自主发现未完成**。尚未证明 broad Query 下连续发现多个未知 sub-aspect |
| 上一轮 evidence 决定下一轮，并在充分时停止 | live13 的 round 1 official evidence 只负责 control gate 和 lineage；它没有决定测试方向。round 2 checker pass 后 existential contract 以 `evidence_sufficient` 停止 | **evidence-gated continuation/stop 完成**。`eval_20260726_batch23_open_query_live_n1_v5` 另有 scale→distractor refinement，但两种能力尚未在同一旗舰合一 |
| Fig. 3：Proposal → retrieve/generate scene + `check_success()` → rollout | live13 在原本 official-only 的 `grab_roller` 上通过通用 backend 同时生成 scene/checker；2/2 simulator fixtures、render、VLM、expert 通过后直接裁决 ACT，且未新增任务名分支 | **一个完整冷启动正例完成**。生成 checker 明确标为 experimental，不冒充 official benchmark success |
| 首帧视觉诊断，失败时局部重生成 | 通用 TaskGen 已运行真实 VLM visual diagnosis，并只允许一次局部 repair；preservation 仍由 simulator state、collision geometry、AST/checker fixture 与 frozen binding 独立裁决 | **职责边界已完成**。最新旗舰没有触发 repair，因此尚缺“视觉发现问题→一次修复→通过”的在线正例；视觉不能替代数值 authority |
| Fig. 4：ToolGen retrieve/generate/validate/register/reuse | live13 的 Query 诱发 typed MetricSpec，编译和差分验证后得到 `0.05838 m`，进入 Aggregate/Plan Agent；追加 0-ACT replay 的第二次请求精确走 `run_local_reuse` | **同一 completed-round 审计内闭合**。仍缺第二个独立 Query/evaluation 从持久 registry 复用 |
| rollout → Rule/VQA → Aggregate → Plan Agent session → Answer | live13 的单一命令生成 Query、逐轮 prompt/decision、TaskGen 代码、render、ACT video/telemetry、Tool/VQA、Aggregate 和受限 Answer | **RoboTwin 小范围基本完成** |
| 回答原 Query 并显式约束确定性 | `AnswerScope` 强制记录 N、唯一 seeds、候选域、冲突、stop reason 与限制；live13 正确表述 2 个 episode 只有一个 seed `[100301]` | **当前完成度高**。输出明确不是统计泛化保证 |
| RoboTwin/LIBERO 使用同一外层方法语义 | `MethodRuntime`、`QueryContract` 与 policy-task binding 已抽成共享接口；RoboTwin 真实 child bundle 会经过 typed runtime projection，LIBERO 仍进入独立 backend chain | **合同开始统一，原生执行环未统一**。projection 验证既有执行结果，不等于生产编排已经委托给共享 runtime |

## 实验 claim

| 论文 claim | 当前项目证据 | 判断 |
| --- | --- | --- |
| Tables 1–2：更少 samples/time 得到 dense benchmark 可比结论 | ClickBell 三 seed toy 中 dense 12 ACT → adaptive 6 ACT；预注册 universal verdict 与 paired weak-axis 为 3/3，但完整 failure set 仅 2/3 | **受限 toy，未复现论文主实验**。它同时给出了真实节省和真实不一致，不能只报告正面字段 |
| Table 3：RAG、visual self-check、README.Agent 提升 codegen | 五个 frozen unseen BBH Proposal × 五条件、25 次 generation；只有 RAG 呈论文方向，visual/README 出现 ceiling 或反向结果 | **协议已运行，结论未复现** |
| Table 6：Plan 与机器人研究者标注一致 | 30 条五类 Query，development-agent proxy precision=0.513、micro-F1=0.519、exact=0.367 | **未复现人工有效性**。human_count=0，冻结包可由后续独立标注直接替换 |
| Tables 7–8：VQA 在四类视觉条件下保持 accuracy/AUROC | 8 个缓存样本、单 VLM、development proxy，accuracy=0.875、AUROC=1.0 | **只属协议 smoke**。缺 simulator-native 条件、独立人工 gold 与多 VLM |
| Table 9：少样本保持 ACT/DP/DP3/RDT/π0 排名 | ACT/DP3 三 seed pilot 为 2/3 对 2/3，平局；Spearman 不可计算 | **未复现**。暂不为制造排名扩大策略 |
| Fig. 6：约 5% 系统错误率与模块分布 | 冻结 10 个 terminal operations，10/10 pass | **分母不足，未复现** |
| 数百条、五类、人工 sub-aspect Query 数据集 | 30 条中文 proxy Query、五类各 6 条 | **最小协议样本，不是论文数据贡献** |
| 多任务、跨 policy、RoboTwin/LIBERO 一致性 | RoboTwin 当前有五份数据完备 TaskSchema 和相应 checkpoint；生产 binding 已不要求五任务 TaskAdapter。SmolVLA 统一 checkpoint 在五任务 N=1 顺序 pilot 为 4/5，但未进入 MEA 方法链。LIBERO task0 为 official-positive/custom-negative basic adaptation | **跨任务/跨环境结论未复现**。代码通用边界与多任务 policy 可运行性都不等于跨任务生成式闭环证据或 50-task 成功率 |

## 当前最重要的 gap

1. **把 broad Query refinement 与完整生成闭环合到同一运行。**
   `eval_20260726_batch23_open_query_live_n1_v5` 证明多轮 concern
   转向，live13 证明通用 scene+checker+Tool 冷启动；下一步应由 broad Query 和第一轮
   evidence 自主决定第二轮候选，而不是由 Query 先限定唯一场景。
2. **统一已知与未知 concern 的 materialization。** retrieval index 外 concern 已走
   `Proposal → GenericRoboTwinTaskGenBackend`；但命中旧 template 时仍可能进入
   `BoundedProposalAgent` 与 BBH/ClickBell dialect。下一步应先做语义 artifact exact
   retrieval，miss 后统一走 generic backend，使 task-specific dialect 只留在
   `experiments/paper/`。
3. **让跨 evaluation Tool reuse 成为正常 Query 路径。** 当前只证明同一已完成
   telemetry 的 exact reuse；还需第二个独立 Query 直接检索持久 artifact。
4. **把共享 `MethodRuntime` 从兼容投影变成原生执行边界。** 当前 projection 不重复
   TaskGen/ACT，但真正生产 mechanics 仍在旧 child pipeline；应逐阶段迁移，而不是再加一套编排。
5. **统一 RoboTwin/LIBERO 的外层 loop。** simulator backend 可以不同，但 route/session、
   RoundExecutor、Aggregate、stop contract 与 Answer 不应重复实现。
6. **方法稳定后再补实验规模。** 首先做三 seed、非平凡 compare/worst-case 的
   dense/adaptive 保真；独立人工 Plan/VQA、多策略排名与更多 LIBERO task 后置。

## 软件工程边界

- 生产路径只保留 Plan Agent；fixed/catalog/task-specific legacy planner 仅作为
  `experiments/paper/` 消融兼容层。
- `mea/artifact_retrieval_index.py` 是生产 known-artifact retrieval API；
  `CapabilityAdapter` 只保留旧数据和兼容导出。task identity、schema、checkpoint 与
  official runtime hooks 由 runtime binding 单独验证，不能再把旧五任务成员资格当执行许可。
- generic TaskGen 与 ToolGen 各只保留一次共享局部 repair；checker fixture 失败时
  TaskGen 可保持已验证 scene、只重生成 checker，但不能再借另一份预算重试；不恢复中央
  whole-round restart。
- `ArtifactIndex` 是迁移 façade。旧 Task/Tool/VQA registry 仍有真实 caller，在 caller
  迁移前不能直接删除或声称已统一。
- `execution_receipt` 仍被 ACT evaluator、recorder 和 probe 使用；论文协议 plumbing
  应先迁移 caller，再裁剪。
- README 保持简洁不改；动态运行状态只维护在本文件和 `docs/evidence/current/`。
