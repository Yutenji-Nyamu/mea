# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–6、Tables 1–10 与 Appendix。状态严格区分
“接口存在”“小规模真实闭环”和“达到论文实验规模”。
本批四项结果统一索引见
[batch25 open retrieval + TaskGen](../experiments/paper/results/batch25_open_retrieval_taskgen/summary.json)；
上一批冻结协议仍见
[batch24 claim closure](../experiments/paper/results/batch24_claim_closure/summary.json)。

| 论文 claim | 当前项目证据 | 判断 |
| --- | --- | --- |
| 开放 Query 驱动多轮评估；evidence 决定下一步 | 新 resolver 先做 inventory-free `FreeConcern`，再从 50 个 official RoboTwin task 中检索。ClickBell plan-only v1 因 concern 写错任务意图而拒绝，v2 concern 正确但 lexical top-1 仍误选 `scan_object`；0-provider 确定性 near-tie replay 才选回 policy-compatible `click_bell`。既有宽泛 Query v5 则已证明真实 evidence 能改变下一 aspect | **部分实现**。50-task discovery 扩大了形式上的开放入口，但两个 live provider plan-only 都是负结果；确定性 replay 不是 clean provider→route 成功证据，单任务 checkpoint 也不能执行任意被发现 task |
| 证据充分后停止并回答原 Query | v7 对单候选合同以 `evidence_sufficient` 停止；宽泛 v5 的 scale official-equivalent 判定通过，distractor 实验 checker=true 但 official success=false，最终以 `budget_exhausted / inconclusive` 停止并列出未测 color/position/timing | **停止与限制合同已实现，语义冲突处理仍有缺口**。最终回答披露 checker authority 限制，但内部 `evidence_conflict` 未把这组语义不一致标成冲突；N=3 不能回答“最先在哪里失败” |
| TaskGen 为同一 Proposal 生成 scene 与 `check_success()`，并裁决 rollout | BBH v5 已有一个完整案例。ClickBell 首版在初次生成和一次有界重生成中均虚构不存在的 API，静态验证 fail closed、0 simulator/ACT；补齐 exact-code retrieval 后，独立 v2 一次生成通过 compile、6/6 fixtures、render/rule、expert，并以同 seed 各跑 1 个 official control 与 custom ACT，二者均成功 | **第二个任务的最小正例完成，但结论仍 inconclusive**。online resolver v2 先以 unsupported 结束，后续 0-provider replay 与 standalone TaskGen/ACT driver 并非统一 CLI 自动两轮；且只覆盖 seed 100405、一个第二 bell 几何和 0.12m offset |
| 首帧视觉检查与重新生成 | 通用 wrong-color 注入案例完成 fail→一次 provider repair→rerender→accept；v5 视觉检查 confidence=0.8、无需 repair | **机制完成一个真实 repair 案例**，但未证明它提高总体生成成功率 |
| ToolGen retrieve/generate/validate/register/reuse | jerk Query 现场生成 `tcp_jerk_peak_l2_pre_contact`，oracle 验证后在真实 ACT telemetry 得到 58.145 m/s³；第二 Query exact reuse、`provider_called=false`；含 Tool 的 evidence 改变下一 Proposal，v5 另有 live XY-distance Tool | **生命周期最小闭环完成一例**。Planner 影响来自缓存同轮反事实，不是重复随机试验的因果估计 |
| rollout → Rule/VQA → Aggregate → Planner → Answer | BBH/click_bell、ACT、N=1–3 范围内完整运行；AnswerScope 强制列 N、候选域、冲突、unsupported 与停止原因 | **小范围基本实现** |
| 更少 samples/time 得到与 dense benchmark 可比结论 | 三个未见 seed、四个冻结 click_bell 候选、fixed/adaptive 独立新 rollout：universal conclusion 与 paired weak axis 均 3/3 一致；fixed 12 ACT 对 adaptive 6 ACT，少 50% episode、46.8% wall、43.2% policy steps | **受限 toy 正证据**，不是 Tables 1–2。完整 failure-candidate set 仅 2/3 一致；seed 100406 的 adaptive 漏掉两个 dense instance failures |
| 少样本保持 ACT、DP、DP3、RDT、π0 相对排名 | BBH 共享 expert/scene gate 后，ACT 与 DP3 同 3 seeds 均 2/3 | **未复现**。两策略打平，pair order 与 Spearman 均为空；未扩 DP/RDT/π0 |
| RAG、visual self-check、README.Agent 提升 codegen 成功率 | 五个冻结 unseen BBH proposals × 五条件、共 25 个 provider scene+checker cells：complete=4/5、−RAG=0/5、base=0/5、−visual=3/5、−README=5/5；0 ACT | **RAG 有受限方向性信号，其他 claim 未复现**。同模型族 blind development-proxy、单 seed；visual 仅弱单对信号，README 未显示增益，且 AST allowlist 构成主要失败源 |
| Plan 与机器人研究者 sub-aspect 标注一致 | 30 条五类 Query（各 6）及可替换盲化标注包；单 Codex development proxy 下 sub-aspect precision=0.513、micro-F1=0.519、exact=0.367 | **未复现**。human_count=0；30 query invocations 因 schema repair 产生 50 provider attempts/42 recorded responses，属于已披露的协议偏差 |
| VQA 在四种视觉条件下保持 accuracy/AUROC | 现有 8 个 cached montage predictions 被冻结到可替换盲包；numeric telemetry proxy 下 coverage 7/8、accuracy 0.875、AUROC 1.0，本批 0 新 VLM/ACT | **协议 smoke**。human_count=0、每条件仅一正一负、扰动非 simulator-native、单 VLM；人工回来只替换 annotation 文件，当前数值仍非 Tables 7–8 证据 |
| 约 5% 系统错误率及模块分布 | 冻结 10 个语义 artifact operations，10/10 pass；另一次 v4 真实暴露 fenced-JSON adapter error 并由回归测试修复 | **未复现**。10 个样本的 0% 无法估计 5% 或模块分布，v4 也不在冻结分母内 |
| 数百条、五类、人工 sub-aspect 的开放 Query 数据集 | 30 条中文 proxy Query，五类各 6 条；14 supported、16 unsupported；已冻结四人多数票空槽 | **只完成可替换协议样本**，不是论文数据集贡献；正式人工只替换 annotation JSON，不重跑 Query 或 prediction |
| 多任务、跨 policy、RoboTwin/LIBERO 一致性 | RoboTwin 可发现 50 个 official task，但 ACT official 深入入口仍是 BBH/click_bell，adjust_bottle/grab_roller 只有 adapter；DP3 仅 BBH。LIBERO checkpoint 的声明 scope 为 unknown，因此无绑定请求在创建 run 前拒绝；显式 `libero_object/task0` 的 plan-only 只形成 1 个授权候选、0 rollout，change contract 仍 pending。旧 batch24 official/custom 两 episode 因 control 失败且变更不对齐而协议无效 | **没有跨环境或任意任务能力证据**。显式绑定是执行授权，不是训练支持证明；新 live chain 只在代码和测试中保证 official control 失败后短路 custom rollout，本批未新增 LIBERO ACT |
| 相对 benchmark 的可解释、动态生成、开放工具能力 | 当前主链能输出生成代码、render、rollout、Tool/VQA、Aggregate 与受限自然语言结论 | **受限实现**。项目不声称传统 benchmark 的 absolute correctness；生成 checker 只能支持其声明的实验语义 |

## 第一性原理上的首要 gap

1. **停止策略的完整结论保真性**：三 seed toy 已保住预注册的 universal conclusion 与
   paired weak axis，却在 1/3 seed 漏掉 dense instance failures。下一步必须先明确 Query
   只要求主弱轴还是要求完整 failure set；后者不能沿用当前 position-pair 即停的合同。
2. **生成成功率的独立归因**：Table 3 已扩到 5 Proposal/25 cells，但 RAG 信号同时受
   AST compatibility floor、单 seed 和同模型族 proxy 影响。下一步是人工盲评与更多
   独立 generation repeats，而不是继续增加 ablation 开关。
3. **开放检索的语义可靠性与端到端统一**：本批已把 concern 抽取放到 inventory 之前，
   并可发现 50 个 official task；但两个 provider plan-only 都未直接得到正确可执行
   route，near-tie 修复也只是 0-call replay。下一步应在冻结 Query 上跑通一次 clean
   provider concern → semantic retrieval → policy gate → TaskGen 的同一 bundle；同时把
   lexical 近邻替换成对 base action/goal 更稳健的语义检索。catalog 外 concern 可以
   generate 或明确 unsupported，但不能被单任务 checkpoint 静默接管。
4. **有效性证据**：Plan/VQA 已有冻结 prediction 与可替换 annotation 入口；关键缺口
   现在明确是四名独立机器人标注者、senior tie-break、真实 simulator-native 扰动和
   多 VLM，而不是继续增加 development proxy。Plan 的 50-attempt 偏差也需在正式协议
   中禁用 repair 或按 HTTP attempt 预注册。
5. **policy ranking 与跨环境**：ACT/DP3 平局没有排名信息；至少需要第三策略和多个 task
   才可能形成 Spearman。当前按成本约束后置。LIBERO 已证明 BDDL/custom env/确定性
   predicate MetricSpec adapter/reuse 组件能运行，但没有证明模型生成新 Tool；当前
   runtime 已在 provider/TaskGen 前强制 Planner–TaskGen controlled-change alignment，
   并在 control fail 后短路 custom rollout。下一步先为 checkpoint 建立可审计的 task
   scope，再取得成功的 official control；不能用显式 CLI binding 代替 scope 证据。

项目后续只增加直接支撑这些 claim 的主链或实验；不恢复已删除的平行 planner、旧
Table 3/efficiency 栈、中央 recovery、开发日志或多层可信性封装。
