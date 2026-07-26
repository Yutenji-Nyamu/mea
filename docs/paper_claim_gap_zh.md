# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–6、Tables 1–10 与 Appendix。状态严格区分
“接口存在”“小规模真实闭环”和“达到论文实验规模”。

| 论文 claim | 当前项目证据 | 判断 |
| --- | --- | --- |
| 开放 Query 驱动多轮评估；evidence 决定下一步 | 宽泛 Query v5 未给 aspect 顺序；official control 后，ClaimFirst 自主选择 1.2× scale，再根据结果选择 lookalike distractor；三轮均为真实 ACT | **小范围完成**。候选仍来自有界 capability catalog，尚不是任意属性发现 |
| 证据充分后停止并回答原 Query | v7 对单候选合同以 `evidence_sufficient` 停止；宽泛 v5 的 scale official-equivalent 判定通过，distractor 实验 checker=true 但 official success=false，最终以 `budget_exhausted / inconclusive` 停止并列出未测 color/position/timing | **停止与限制合同已实现，语义冲突处理仍有缺口**。最终回答披露 checker authority 限制，但内部 `evidence_conflict` 未把这组语义不一致标成冲突；N=3 不能回答“最先在哪里失败” |
| TaskGen 为同一 Proposal 生成 scene 与 `check_success()`，并裁决 rollout | v5 round 3 由模型一次编写 target+distractor scene/checker，6/6 fixtures、render、visual、expert、ACT 均通过；生成 checker=true 而 official success=false | **论文式最小案例完成**。实验 checker 与 official semantics 分开报告；仅 BBH、少量 proposal |
| 首帧视觉检查与重新生成 | 通用 wrong-color 注入案例完成 fail→一次 provider repair→rerender→accept；v5 视觉检查 confidence=0.8、无需 repair | **机制完成一个真实 repair 案例**，但未证明它提高总体生成成功率 |
| ToolGen retrieve/generate/validate/register/reuse | jerk Query 现场生成 `tcp_jerk_peak_l2_pre_contact`，oracle 验证后在真实 ACT telemetry 得到 58.145 m/s³；第二 Query exact reuse、`provider_called=false`；含 Tool 的 evidence 改变下一 Proposal，v5 另有 live XY-distance Tool | **生命周期最小闭环完成一例**。Planner 影响来自缓存同轮反事实，不是重复随机试验的因果估计 |
| rollout → Rule/VQA → Aggregate → Planner → Answer | BBH/click_bell、ACT、N=1–3 范围内完整运行；AnswerScope 强制列 N、候选域、冲突、unsupported 与停止原因 | **小范围基本实现** |
| 更少 samples/time 得到与 dense benchmark 可比结论 | 预注册四候选单 seed：fixed 4 ACT 为 left fail、其余 pass；adaptive 首项 left fail 后停止，结论一致，少 3 ACT、170.00 s、205 steps | **正向 toy evidence**，不是 Tables 1–2。另一个多轴 pilot 曾因漏测弱轴得到负结果，说明停止合同仍脆弱 |
| 少样本保持 ACT、DP、DP3、RDT、π0 相对排名 | BBH 共享 expert/scene gate 后，ACT 与 DP3 同 3 seeds 均 2/3 | **未复现**。两策略打平，pair order 与 Spearman 均为空；未扩 DP/RDT/π0 |
| RAG、visual self-check、README.Agent 提升 codegen 成功率 | 两个冻结 unseen proposals × 五条件真实 provider scene+checker：complete=2/2、−RAG=0/2、base=0/2、−visual=2/2、−README=2/2 | **只得到 RAG 的微型方向性信号**。非盲 development-agent proxy、N=2；没有证明 visual/README 增益，也未做 ToolGen 消融 |
| Plan 与机器人研究者 sub-aspect 标注一致 | 20 条五类 Query development-agent proxy；aspect micro-F1≈0.462 | **未复现**。无独立人工 gold、多人多数票，且指标/规模与 Table 6 不可比 |
| VQA 在四种视觉条件下保持 accuracy/AUROC | 两个真实 RoboTwin rollout montage，经 clean/clutter/texture/lighting 确定性变换得到 8 样本；单 VLM coverage 7/8、accuracy 0.875、AUROC 1.0 | **协议 smoke**。proxy gold、每条件仅一正一负、扰动非 simulator-native、无多 VLM；AUROC 统计意义极弱 |
| 约 5% 系统错误率及模块分布 | 冻结 10 个语义 artifact operations，10/10 pass；另一次 v4 真实暴露 fenced-JSON adapter error 并由回归测试修复 | **未复现**。10 个样本的 0% 无法估计 5% 或模块分布，v4 也不在冻结分母内 |
| 数百条、五类、人工 sub-aspect 的开放 Query 数据集 | 20 条中文 proxy Query，五类各 4 条；10 supported、10 unsupported | **只完成协议样本**，不是论文数据集贡献 |
| 多任务、跨 policy、RoboTwin/LIBERO 一致性 | RoboTwin ACT official 入口覆盖 BBH、click_bell、adjust_bottle、grab_roller；DP3 仅 BBH；LIBERO/SmolVLA 只有独立 adapter smoke | **执行面扩大，方法证据仍窄**。论文自身也说明 LIBERO 仅基本适配、未达到 RoboTwin 实验规模 |
| 相对 benchmark 的可解释、动态生成、开放工具能力 | 当前主链能输出生成代码、render、rollout、Tool/VQA、Aggregate 与受限自然语言结论 | **受限实现**。项目不声称传统 benchmark 的 absolute correctness；生成 checker 只能支持其声明的实验语义 |

## 第一性原理上的首要 gap

1. **停止策略的结论保真性**：省样本只有在结论不变时才有价值。下一步应把四候选
   toy 扩成 3 个冻结 seed；若任一 seed 漏掉 dense 弱轴，直接判定停止合同失败。约
   15–24 ACT，先做 N=3 总计 15 ACT 的最小版。
2. **生成成功率而非单例成功**：TaskGen 的核心不是“能生成一次”，而是对未见 Proposal
   稳定生成 scene+checker。先把 Table 3 扩到 5 个 Proposal；仍为 0 ACT，独立人工
   review 后再讨论 visual/README 增益。
3. **开放规划的候选发现**：v5 证明 evidence 能改变顺序，但 candidate universe 仍预建。
   下一批应允许 Planner 从 Query 提出一个 catalog 外 concern，再由 capability/TaskGen
   判定可执行或 unsupported，而不是默默映射回现有 aspect。
4. **有效性证据**：Table 6–8 的关键缺口是独立 gold 与真实 simulator-native 扰动，
   不是再增加 proxy 数量。开发代理标注只用于调协议，正式结论等待人工和多 VLM。
5. **policy ranking 与跨环境**：ACT/DP3 平局没有排名信息；至少需要第三策略和多个 task
   才可能形成 Spearman。当前按成本约束后置。LIBERO 下一步若继续，应迁移
   BDDL + Problem/checker + predicate Tool，而不是只增加 policy rollout。

项目后续只增加直接支撑这些 claim 的主链或实验；不恢复已删除的平行 planner、旧
Table 3/efficiency 栈、中央 recovery、开发日志或多层可信性封装。
