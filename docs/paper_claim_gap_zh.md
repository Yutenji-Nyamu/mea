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
`eval_20260730_batch31_grab_roller_broad_live_v3`。Query 只问 ACT 在
`grab_roller` 的哪种可执行物体属性或场景变化上最先暴露弱点，没有给
aspect/template。round 1 official 成功后，Plan Agent 自选物体尺度，将 roller
缩至 `0.85`；该轮仍成功且左侧 TCP 最小距离为 `0.02206 m`。这份不充分 evidence
使下一轮把同一 sub-aspect 细化为 `0.70`，并补测右侧 TCP 最小距离
`0.04654 m`。两轮均由通用 TaskGen 生成 scene/task subclass，并复用 official
checker binding，经过 render、VLM、expert 后各执行一次 ACT；新 Python Rule Tool
在原 live round 得到非空值并进入 Aggregate。Execution VQA 是旧编排自动运行的
辅助观察；两个 Proposal 均未请求 VQA Tool，不能作为 Query-conditioned VQA
materialization 证据。

三轮均为同一 seed `[100401]` 且官方成功，最终因预算耗尽而不是证据充分停止；
Answer 因此保持 `inconclusive`，没有虚构“最早弱点”。0-ACT 语义重审修正了中文
“尺度设为/调整为”被误判为 unchanged scene 的词法错误：两个 Proposal 均为
direct alignment；但 execution coverage 仍为 partial，`preserved_conditions` 缺
完整 simulator authority，`required_observation` 也未完整覆盖（左右距离来自不同
scale，且没有测完全部 requested signals）。原 Answer 与原始运行保持不可变，重审
只作为附加审计。

## 方法 claim

| 论文 claim | 当前项目证据 | 判断 |
| --- | --- | --- |
| Fig. 2/5：开放 Query 驱动，Plan Agent 自主提出 sub-aspect | 生产入口由 `PlanAgentInitialPlanBuilder` 直接建立首轮计划，不调用 catalog/task-specific planner。batch31 broad Query 未给候选；official evidence 后 Plan Agent 自选 scale，并依据 0.85 evidence 细化到 0.70 | **小范围 evidence-conditioned sub-aspect refinement 已真实完成**。仍只有一个 task/seed 和一条尺度方向，尚未证明跨多类 concern 的自主搜索 |
| 上一轮 evidence 决定下一轮，并在充分时停止 | batch31 证明 evidence 改变下一 Proposal，但三轮均成功后因 `budget_exhausted` 停止；旧 live13 证明有限 existential contract 可因 `evidence_sufficient` 停止，但候选已由 Query 限定 | **“动态细化”和“充分停止”分别有正例，尚未在同一 broad flagship 合一** |
| Fig. 3：Proposal → retrieve/generate scene + `check_success()` → rollout | batch31 在 `grab_roller` 上两次使用同一通用 backend 生成 scene/task subclass，并用 official checker wrapper 裁决 ACT；两个 Proposal 的 `checker_need` 均为空，且没有任务名专属分支 | **通用 scene generation 已有多轮正例，本次没有新增 checker codegen 证据**。早期实验 checker 案例仍单独成立；本次 preservation 与 observation coverage 都不完整 |
| 首帧视觉诊断，失败时局部重生成 | 通用 TaskGen 已运行真实 VLM visual diagnosis，并只允许一次局部 repair；preservation 仍由 simulator state、collision geometry、AST/checker fixture 与 frozen binding 独立裁决 | **职责边界已完成**。最新旗舰没有触发 repair，因此尚缺“视觉发现问题→一次修复→通过”的在线正例；视觉不能替代数值 authority |
| Fig. 4：ToolGen retrieve/generate/validate/register/reuse | batch31 两个 live round 均由 provider 生成 Python Rule Tool、取得非空 TCP 距离并进入 Aggregate；Batch30 另有 AST/确定性/oracle/artifact gate 与第二 Query `run_local_reuse`。验证器也支持 caller 提供独立 oracle/fixtures 的 `derived_observable`，无 oracle 时生产默认不广告 | **live 生成与测量已完成，精确复用仍是分离的 0-ACT 案例**。尚缺同一旗舰和跨 evaluation reviewed reuse |
| rollout → Rule/VQA → Aggregate → Plan Agent session → Answer | batch31 的单一命令保存 Query、逐轮 prompt/decision、两份 TaskGen 代码、render、三次 ACT video/telemetry、Rule Tool、辅助 Execution VQA、Aggregate 和受限 Answer；两个 Proposal 均未请求 VQA | **RoboTwin 小范围基本完成**。Execution VQA 不能误记为 Query 诱发的新 Tool |
| 回答原 Query 并显式约束确定性 | `AnswerScope` 强制记录 N、唯一 seeds、候选域、冲突、stop reason 与限制；live13 正确表述 2 个 episode 只有一个 seed `[100301]` | **当前完成度高**。输出明确不是统计泛化保证 |
| RoboTwin/LIBERO 使用同一外层方法语义 | `MethodRuntime`、`QueryContract` 与 policy-task binding 已抽成共享接口。生产 Plan Agent CLI 已用 `--policy-backend smolvla` 在 schema-backed `click_bell` 完成一次 N=1 official rollout，并写出 Rule/Aggregate；该 episode 的 official check 为 true。原在线 pipeline 因未请求 VQA 却被执行而失败，修复后只做了 append-only 0-rollout 投影并得到 pipeline pass；Planner 仍以 `budget_exhausted`、`inconclusive` 结束。LIBERO 仍进入独立 backend chain | **只证明 live backend/post-rollout mechanism 与事后投影，不是 clean online acceptance**。generic scene/checker 与请求型 VQA 尚未接入 SmolVLA；跨 simulator 外层 loop 未统一 |

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
| 多任务、跨 policy、RoboTwin/LIBERO 一致性 | RoboTwin 动态发现 50 个 official task，均可由共享 SmolVLA binding 尝试 official rollout；仅 5 个有 semantic TaskSchema。五任务 N=1 为 4/5；新增 `click_alarmclock` 与 `turn_switch` 均合法执行但 policy failure，后者走原生 MethodRuntime。LIBERO task0 为 official-positive/custom-negative basic adaptation | **跨任务/跨环境结论未复现**。50 个 rollout-ready 只表示执行资格，不表示 50-task 成功率、训练覆盖证明或生成式闭环 |

## 当前最重要的 gap

1. **让 broad refinement 因 evidence sufficient 而停。** batch31 已把 broad
   Query、自主 scale concern、两轮通用 scene generation、official checker reuse
   与 live Python Tool 合在同一数据流，但仍因预算停止。下一次应冻结可满足的有限
   Query contract，让 evidence 决定继续、换 concern 或充分停止，而不是再增加接口。
2. **让 Proposal 的 preservation 与 required observation 可被真实 authority 裁决。**
   batch31 的尺度变化经重审已是 direct alignment，但模型写入了当前 simulator/VLM
   无法完整验证的保持条件，而且两轮没有覆盖各自声明的全部 observation。Prompt 应只
   声明可观测的条件；否则先生成对应 state/geometry Tool，不能把“未发现变化”当作
   已保持，也不能把跨不同 scale 的左右单侧测量合成同一候选的 bilateral evidence。
3. **把开放 Tool 变成可验证的生产能力。** `derived_observable` 已能在
   caller-owned oracle 下生成、验证和注册；下一步提供独立 oracle/fixture broker，
   只在该能力存在时广告 v2，并在同一 live run 证明非空 metric、影响下一轮和第二
   Query exact reuse。
4. **补齐 SmolVLA 原生生成候选边界。** 先让 schema-backed official round 在修复后
   得到一次 clean online acceptance；再把 generic TaskGen scene/checker 与请求型 VQA
   capability 接到同一 `MethodRuntime`，保持共享 post-rollout orchestration，不增加
   第二套 mini-chain。
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
  known concern 先做语义 artifact exact retrieval，miss 后统一进入 generic backend；
  BBH/ClickBell dialect 只作为 paper/compat 迁移源。
- generic TaskGen 与 ToolGen 各只保留一次共享局部 repair；checker fixture 失败时
  TaskGen 可保持已验证 scene、只重生成 checker，但不能再借另一份预算重试；不恢复中央
  whole-round restart。
- `ArtifactIndex` 是迁移 façade。旧 Task/Tool/VQA registry 仍有真实 caller，在 caller
  迁移前不能直接删除或声称已统一。
- `execution_receipt` 仍被 ACT evaluator、recorder 和 probe 使用；论文协议 plumbing
  应先迁移 caller，再裁剪。
- README 保持简洁不改；动态运行状态只维护在本文件和 `docs/evidence/current/`。
