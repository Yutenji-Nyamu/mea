# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–6、Tables 1–10 与 Appendix。状态严格区分
“接口存在”“小规模真实闭环”和“达到论文实验规模”。本批统一结果见
[batch26 claim closure](../experiments/paper/results/batch26_claim_closure/summary.json)，
干净旗舰的逐步数据流见
[current evidence](evidence/current/README.md)。

| 论文 claim | 当前项目证据 | 判断 |
| --- | --- | --- |
| 开放 Query 驱动多轮评估；evidence 决定下一步 | batch26 ClickBell v6 从一条未带 candidate/aspect 顺序的 Query 出发，在线生成 `FreeConcern`，解析到 `robustness.distractor_avoidance.lookalike_bell`，依次执行 official control 与 provider scene+checker；无 history replay、无 CLI candidate hint | **有限域正例完成**。同一命令和 bundle 已闭合，但本例最终候选域只有 1；尚未证明 Planner 能在多个 catalog 外 concern 中自主寻找最有信息量的方向 |
| 证据充分后停止并回答原 Query | v6 两轮 ACT 后以 `evidence_sufficient` 停止；官方轮标为 `official_only`，生成 checker 轮标为 `expected_semantic_extension`；Answer 只回答有限实验语义，并明确不回答 official benchmark | **有限合同完成**。不再因 hard cap 停止，也没有把 official/generated checker 的语义差异误报成 benchmark 冲突；N=2、单候选仍不是广泛泛化结论 |
| TaskGen 为同一 Proposal 生成 scene 与 `check_success()`，并裁决 rollout | BBH 已有完整案例；batch26 又把 ClickBell provider scene+checker 放进同一旗舰命令：代码生成、静态检查、6/6 fixtures、render/expert gate、ACT 与生成 checker 裁决均进入同一 bundle | **两个任务上的最小正例**。生成 checker 成功而 official terminal success=false，只支持声明的实验扩展语义；还没有跨多个 unseen Proposal 的稳定 live 成功率 |
| 首帧视觉检查与重新生成 | 通用 wrong-color 案例完成 fail→一次 provider repair→rerender→accept；batch26 干净旗舰无需 repair | **机制有真实个例**，但没有证明 visual self-check 提高总体生成成功率 |
| ToolGen retrieve/generate/validate/register/reuse | jerk Query 已完成新 metric→oracle→真实 telemetry 非空值→第二 Query exact reuse；batch26 同 bundle 中 checker Tool 被绑定、执行并复用到 Aggregate/Planner | **生命周期小范围闭合**。跨 Query exact reuse 与“同 bundle 影响 Planner”仍来自两个案例，而不是同一个旗舰 |
| rollout → Rule/VQA → Aggregate → Planner → Answer | batch26 v6 的两个 ACT round 均产生 episode、Rule/Tool、VQA、Aggregate、next decision 与受限 Answer | **RoboTwin 小范围基本实现** |
| 更少 samples/time 得到与 dense benchmark 可比结论 | batch26 单 seed、四候选 universal-refutation：fixed 4 ACT，adaptive 首轮发现反例后停在 1 ACT；两者原 Query verdict 均为 refuted，少 3 ACT、170.7 秒和 200 policy steps。既有三 seed实验为 12→6 ACT，但完整 failure set 只保持 2/3 | **受限 toy 正证据，不是 Tables 1–2**。它只保持预注册的“存在反例”结论；adaptive 未覆盖其余三候选，因此完整 failure set 不可比较 |
| 少样本保持 ACT、DP、DP3、RDT、π0 相对排名 | 既有 ACT/DP3 三 seed pilot 为 2/3 对 2/3，平局。batch26 N=5 预注册先跑共享 expert gate，仅 2/5 seed eligible，按协议在策略 rollout 前中止 | **未复现**。pair order 与 Spearman 为空；本批 5 次 expert probe、0 policy rollout，未通过 gate 不能靠筛 seed 制造排名 |
| RAG、visual self-check、README.Agent 提升 codegen 成功率 | 五个冻结 unseen BBH Proposal × 五条件、25 次 provider generation：complete=4/5、base=0/5、−RAG=0/5、−visual=3/5、−README=5/5；0 ACT | **仅 RAG 有方向性信号**。单模型族、单 repeat、development-agent proxy；visual/README 未复现论文方向，不能称 Table 3 |
| Plan 与机器人研究者 sub-aspect 标注一致 | 30 条五类 Query；单 Codex development proxy 下 precision=0.513、micro-F1=0.519、exact=0.367 | **未复现论文人工有效性**。human_count=0；已保留可直接替换 annotation 的冻结包 |
| VQA 在四种视觉条件下保持 accuracy/AUROC | 8 个缓存样本、每条件一正一负；proxy accuracy=0.875、AUROC=1.0 | **协议 smoke**。human_count=0、单 VLM、扰动非 simulator-native；不能作为 Tables 7–8 |
| 约 5% 系统错误率及模块分布 | 冻结 10 个语义 terminal operations，10/10 pass，0 个 paper-defined error | **未复现 Fig. 6**。分母虽达到最小可报告阈值，但 10 个样本无法估计约 5% 或稳定模块分布 |
| 数百条、五类、人工 sub-aspect 的开放 Query 数据集 | 30 条中文 proxy Query，五类各 6 条，并留有四人多数票替换槽 | **只完成最小协议样本**，不是论文数据集贡献 |
| 多任务、跨 policy、RoboTwin/LIBERO 一致性 | RoboTwin 可检索 official task inventory，实际深入主链为 BBH/ClickBell，adjust_bottle/grab_roller 仅 official adapter。LIBERO batch26 按 `libero_object/task0`、relative action、280-step 上限执行 SmolVLA official control；1 rollout 失败后严格禁止 custom BDDL | **跨环境方法证据仍为空**。LIBERO adapter/inference 可运行，但 control 未通过，`method_chain_valid=false`，不能进入第二轮或宣称 basic adaptation 成功 |
| 相对 benchmark 的可解释、动态生成、开放工具能力 | 当前旗舰能保留 Query、FreeConcern、生成代码、render、rollout、Tool/VQA、Aggregate、Planner decision 与受限 Answer | **受限实现**。项目不声称 traditional benchmark absolute correctness；生成 checker 只拥有其显式实验语义的 authority |

## 第一性原理上的首要 gap

1. **从单候选闭环走向真正的信息增益规划**：v6 已消除人工串接和缓存 replay，但
   singleton concern 几乎不需要“规划”。下一次应冻结一个包含 2–3 个可执行 concern
   的开放 Query，让第一轮 control evidence 决定第二轮生成哪个此前未指定的测试；仍
   限制每轮 N=1、最多 2–3 ACT。
2. **结论保真，而不只是快速找到一个反例**：universal-refutation 可以合法早停，
   但“完整弱项/最差属性/比较”需要不同 coverage contract。下一批应在 3 个 seed 上
   预注册一种更难的 claim，只有 adaptive 与 dense 在该 claim 的全部所需字段一致且
   rollout/time 下降才算正结果。
3. **跨环境链先过 control gate**：LIBERO 的主要缺口不是再写接口，而是同一
   checkpoint/task 的 official control 可复现成功。control 成功后才允许一个对齐的
   custom BDDL、deterministic checker Tool、Aggregate 与第二 Query reuse；继续保持
   最多 2 episodes。
4. **独立有效性**：Plan/VQA 已有冻结 prediction 与可替换 annotation 包。真正缺的是
   四名独立机器人标注者、senior tie-break、真实 simulator-native 四条件 clips 与
   多 VLM，而不是更多 development proxy。
5. **生成归因与 policy ranking**：Table 3 需要独立人工盲评和少量重复 generation；
   ACT/DP3 需要先得到共同 eligible seeds，第三策略与五策略 Spearman继续后置。严格
   gate 失败本身应报告，不能事后筛 seed。

后续只保留直接支撑上述 claim 的主链和实验入口；不恢复已被替代的平行 planner、
中央 whole-round recovery、多层 provenance/receipt 封装、旧 evidence bundle 或
逐批 development log。根 README 继续不改。
