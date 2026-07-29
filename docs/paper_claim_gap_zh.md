# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–6、Tables 1–10 与 Appendix。这里严格区分：

- **方法入口存在**：代码可以表达论文组件；
- **真实小闭环**：组件在 simulator/policy rollout 上形成过可审计数据流；
- **论文实验复现**：样本、任务、策略、人工标注与统计协议达到论文口径。

当前方法主证据是
[current evidence](evidence/current/README.md) 对应的
`eval_20260729_b30_refinement_live_v2`。该运行从未指定 aspect/template 的开放 Query
出发：round 1 只执行 official control；Planner 读取该轮 evidence 后才提出
`object_position`；读取前两轮累计 evidence 后又转向 `object_instance`。第三轮首次失败，
QueryContract 随后以 `evidence_sufficient` 停止。全程共 3 次真实 ACT，
`flagship_acceptance.accepted=true`，无 history/cache replay 或人工串接。这证明了一个
有限域、单 task、单 seed 的 evidence-conditioned refinement 正例，不证明统计泛化。

## 方法 claim

| 论文 claim | 当前项目证据 | 判断 |
| --- | --- | --- |
| Fig. 2/5：开放 Query 驱动，Plan Agent 自主提出 sub-aspect | 生产入口由 `ClaimFirstInitialPlanBuilder` 直接建立首轮计划，不再调用 catalog/task-specific Planner。最新 Query 未给 aspect/template；control 前没有冻结候选，Planner 在 evidence 后依次提出 position、instance | **受限正例完成**。catalog/capability 只提供全域检索提示，不再充当执行许可；但真实证据仍只有一个 task/query |
| 上一轮 evidence 决定下一轮，并在充分时停止 | R1→R2 lineage 为 `[round_1]`，R2→R3 为 `[round_1, round_2]`，输入摘要不同；position 成功后才转向 instance，instance 失败后按 diagnostic Query 合同停止 | **Fig. 5 的关键机制已有一个干净 live 正验收**。它只支持冻结有限域中的首个弱点诊断；开放世界的 `all`、`worst-case` 仍必须保持 inconclusive |
| Fig. 3：Proposal → retrieve/generate scene + `check_success()` → rollout | 最新 round 2/3 由通用 provider 路径生成 scene，但 `checker_need=null`，运行时精确保留 official `check_success()`；scene 经过 AST、simulator state、render、VLM、expert 后直接进入 ACT。此前 BBH 有同一 Proposal 生成 scene+checker 并裁决 rollout 的真实案例 | **组件均有真实案例，但尚未在本次干净旗舰中合一**。下一步应在原本 official-only task 上无专属分支地生成 scene+checker |
| 首帧视觉诊断，失败时局部重生成 | 通用 TaskGen 已运行真实 VLM visual diagnosis，并只允许一次局部 repair；preservation 仍由 simulator state、collision geometry、AST/checker fixture 与 frozen binding 独立裁决 | **职责边界已完成**。最新旗舰没有触发 repair，因此尚缺“视觉发现问题→一次修复→通过”的在线正例；视觉不能替代数值 authority |
| Fig. 4：ToolGen retrieve/generate/validate/register/reuse | round 2 由 Query 诱发 typed MetricSpec，编译、差分验证并注册 `query_derived_metric`，真实值 `0.021466 m` 进入 Aggregate/Planner；round 3 按相同 code hash 走 `run_local_reuse`，得到非空 `0.002753 m` | **同一干净案例内闭合**。本例 `provider_called=false`，生成空间仍受 MetricSpec DSL 限制；另缺第二个独立 Query/evaluation 从 reviewed registry exact reuse |
| rollout → Rule/VQA → Aggregate → Planner → Answer | 单一命令生成 Query、逐轮 prompt/decision、TaskGen 代码、render、ACT video/telemetry、Tool/VQA、Aggregate 和受限 Answer | **RoboTwin 小范围基本完成** |
| 回答原 Query 并显式约束确定性 | `AnswerScope` 强制记录 N、seed、候选域、未测项、冲突、stop reason 与限制；最新回答只定位有限域中首个暴露的弱点 | **当前完成度高**。输出明确不是统计泛化保证 |
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
| 多任务、跨 policy、RoboTwin/LIBERO 一致性 | RoboTwin 有五个数据化 TaskAdapter；SmolVLA 统一 checkpoint 在五任务 N=1 顺序 pilot 为 4/5，但未进入 MEA 方法链。LIBERO task0 为 official-positive/custom-negative basic adaptation | **跨任务/跨环境结论未复现**。多任务 policy 可运行性不等于生成式闭环证据或 50-task 成功率 |

## 当前最重要的 gap

1. **把 evidence-conditioned refinement 从单一 ClickBell 正例推广到一次真正冷启动。**
   下一步优先在一个原本 official-only task 上，由第一轮 evidence 决定第二轮 concern；
   不得新增任务名专属 Planner/TaskGen 分支。
2. **继续压缩检索层的任务菜单耦合。** 生产 target 已不携带 aspect/template，但
   `OpenWorldPlanSession.retrieval_aspects` 仍由 `CapabilityAdapter` 的预注册 contracts
   投影。下一步应允许语义检索直接返回相似 artifact/unsupported，而不是先形成任务内
   aspect 菜单。
3. **把共享 `MethodRuntime` 从兼容投影变成原生执行边界。** 当前 projection 不重复
   TaskGen/ACT，但真正生产 mechanics 仍在旧 child pipeline；应逐阶段迁移，而不是再加一套编排。
4. **让跨 evaluation Tool reuse 成为正常 Query 路径。** 当前已证明同一在线 evaluation
   的 exact reuse；还需第二个独立 Query 直接检索 reviewed artifact，并保持非空 live measurement。
5. **统一 RoboTwin/LIBERO 的外层 loop。** simulator backend 可以不同，但 route/session、
   RoundExecutor、Aggregate、stop contract 与 Answer 不应重复实现。
6. **方法稳定后再补实验规模。** 首先做三 seed、非平凡 compare/worst-case 的
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
