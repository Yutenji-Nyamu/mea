# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–6、Tables 1–10 与 Appendix。这里严格区分：

- **方法入口存在**：代码可以表达论文组件；
- **真实小闭环**：组件在 simulator/policy rollout 上形成过可审计数据流；
- **论文实验复现**：样本、任务、策略、人工标注与统计协议达到论文口径。

当前方法主证据是
[current evidence](evidence/current/README.md) 对应的
`eval_20260729_batch31_open_flagship_live_v13`。该运行从未指定 aspect/template 的开放
Query 出发，完成 official control 和一个在线选择的有界位置变化，共 2 次真实 ACT；
第二轮成功后由 existential Query 合同以 `evidence_sufficient` 停止，最终
`flagship_acceptance.accepted=true`。这只证明一个有限候选、单 seed 的存在性命题，
不证明广泛泛化。

## 方法 claim

| 论文 claim | 当前项目证据 | 判断 |
| --- | --- | --- |
| Fig. 2/5：开放 Query 驱动，Plan Agent 自主提出 sub-aspect | 生产入口由 `ClaimFirstInitialPlanBuilder` 直接建立首轮计划，不再调用 catalog/task-specific Planner。v13 Query 未给 aspect/template，在线 Planner 选择“bell 有界水平位置变化” | **小范围完成**。`OpenWorldEvaluationTarget` 已缩为 `PolicyTaskBinding + max_rounds`；catalog/capability 只提供检索提示，不再进入候选许可 schema |
| 上一轮 evidence 决定下一轮，并在充分时停止 | v13 official control 成功后才 materialize 动态 candidate；动态 rollout 成功，existential witness 触发 `evidence_sufficient`，不是 hard cap | **单例正验收完成**。只适用于冻结候选与存在性真值条件；开放世界的 `all`、`worst-case` 仍必须保持 inconclusive |
| Fig. 3：Proposal → retrieve/generate scene + `check_success()` → rollout | v13 的 scene 由 provider 编写代码并通过 AST、fixture、same-seed simulator state、collision geometry、render、VLM、expert 与 ACT；本例按需复用 official checker。此前 BBH 有同一 Proposal 生成 scene+checker 并裁决 rollout 的真实案例 | **组件与组合均有真实案例**，但“新 scene + 新 checker + 干净旗舰”仍分散在两个案例，尚未形成多任务成功率 |
| 首帧视觉诊断，失败时局部重生成 | 通用 TaskGen 已运行真实 VLM visual diagnosis，并只允许一次局部 repair；preservation 同时由 simulator state、collision geometry、AST/checker fixture 与 frozen binding 独立裁决 | **职责边界已修正**。v13 视觉检查一次通过，因此尚缺真实触发 repair 后成功的在线正例；视觉不能替代数值 authority |
| Fig. 4：ToolGen retrieve/generate/validate/register/reuse | v13 根据 Query 生成 typed MetricSpec Tool，在真实 telemetry 上得到非空 `0.4100531051 m` 并进入 Aggregate/Planner。完成回合上的 0-ACT 追加审计再次执行同一请求，第二次走 `run_local_reuse`，无 provider | **小范围闭合**。精确复用目前是 append-only completed-round audit，不是另一个独立在线 Query 的跨 evaluation 复用证据 |
| rollout → Rule/VQA → Aggregate → Planner → Answer | v13 由单一命令生成 Query、plan、TaskGen 代码、render、ACT video/telemetry、Tool/VQA、Aggregate、decision 和受限 Answer | **RoboTwin 小范围基本完成** |
| 回答原 Query 并显式约束确定性 | `AnswerScope` 强制记录 N、seed、候选域、未测项、冲突、stop reason 与限制；v13 回答为有限域 existential `supported` | **当前完成度高**。输出明确不是统计泛化保证 |
| RoboTwin/LIBERO 使用同一外层方法语义 | `MethodRuntime`、`QueryContract` 与 policy-task binding 已抽成共享接口；LIBERO 仍在主 CLI 中进入独立 backend chain | **接口开始统一，执行环未统一**。不能声称 LIBERO 已共享完整 RoundExecutor/Answer loop |

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
| 多任务、跨 policy、RoboTwin/LIBERO 一致性 | RoboTwin 有五个数据化 TaskAdapter，深入证据集中于 BBH/ClickBell；AdjustBottle 有一次生成式运行。LIBERO task0 为 official-positive/custom-negative basic adaptation | **跨任务/跨环境结论未复现**。adapter 广度不等于生成式闭环证据 |

## 当前最重要的 gap

1. **把干净旗舰从一个存在性单例扩到不同 need 组合。** 下一步优先做
   checker-only、Tool-only，以及一个 official-only RoboTwin 任务的冷启动；不得新增任务名
   专属 Planner/TaskGen 分支。
2. **继续压缩检索层的任务菜单耦合。** 生产 target 已不携带 aspect/template，但
   `OpenWorldPlanSession.retrieval_aspects` 仍由 `CapabilityAdapter` 的预注册 contracts
   投影。下一步应允许语义检索直接返回相似 artifact/unsupported，而不是先形成任务内
   aspect 菜单。
3. **让跨 evaluation Tool reuse 成为正常 Query 路径。** v13 已证明同一完成回合中的
   exact reuse；还需第二个独立 Query 直接检索 reviewed artifact，并保持非空 live measurement。
4. **统一 RoboTwin/LIBERO 的外层 loop。** simulator backend 可以不同，但 route/session、
   RoundExecutor、Aggregate、stop contract 与 Answer 不应重复实现。
5. **方法稳定后再补实验规模。** 首先做三 seed、非平凡 compare/worst-case 的
   dense/adaptive 保真；独立人工 Plan/VQA、多策略排名与更多 LIBERO task 后置。

## 软件工程边界

- 生产路径只保留 ClaimFirst；fixed/catalog/task-specific Planner 仅作为
  `experiments/paper/` 消融兼容层。
- `CapabilityAdapter` 只应保存 task identity、checkpoint、schema、official success、
  render/rollout hooks；aspect/metric 菜单属于 retrieval index。
- TaskGen/ToolGen 各保留一次局部 repair；不恢复中央 whole-round restart。
- `ArtifactIndex` 是迁移 façade。旧 Task/Tool/VQA registry 仍有真实 caller，在 caller
  迁移前不能直接删除或声称已统一。
- `execution_receipt` 仍被 ACT evaluator、recorder 和 probe 使用；论文协议 plumbing
  应先迁移 caller，再裁剪。
- README 保持简洁不改；动态运行状态只维护在本文件和 `docs/evidence/current/`。

本轮所有 pytest、provider、simulator 与 ACT 均在 SeetaCloud 服务器运行；Windows 仅用于
代码、文档、diff、Git 与产物同步。
