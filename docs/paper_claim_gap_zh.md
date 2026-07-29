# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–6、Tables 1–10 与 Appendix。状态严格区分
“接口存在”“小规模真实闭环”和“达到论文实验规模”。方法实现的最新证据是
v19 ClickBell 三轮在线运行及其最终 preservation audit：在线机械链完整，但两个动态
candidate 均未取得完整语义 authority，因此该运行是 confounded negative，不是成功
旗舰。跨环境结果见
[batch27 unified adapter / LIBERO evidence](../experiments/paper/results/batch27_unified_adapter_libero/)，
上一批协议结果见
[batch26 claim closure](../experiments/paper/results/batch26_claim_closure/summary.json)，
v19 受审计运行的逐步数据流见
[current evidence](evidence/current/README.md)。

| 论文 claim | 当前项目证据 | 判断 |
| --- | --- | --- |
| 开放 Query 驱动多轮评估；evidence 决定下一步 | 生产入口由 `ClaimFirstInitialPlanBuilder` 直接建计划，不调用 catalog/task-specific Planner。v19 Query 未提供 aspect/template：runtime 先选择 color；该轮 rollout 成功后才提出 80% size | **机械转向真实发生，语义证据无效**。round2 preservation 只有部分 authority，因此不能把该转向当作已验证的开放规划正例 |
| 证据充分后停止并回答原 Query | v19 official/color/size 为 `1/1、1/1、0/1`，但 round2 geometry preservation 未证实，round3 contact-point z 改变 `-0.0051580801 m` | **未完成**。最终 audit 必须 `accepted=false`、Query inconclusive；失败关闭本身是纠正错误方法结论的有效进展 |
| TaskGen 为同一 Proposal 生成 scene 与 `check_success()`，并裁决 rollout | v19 两个动态 candidate 都由 provider materialize scene、复用 official checker，并运行 fixture、VLM visual diagnosis、expert preflight 和 ACT；round2 trace=`direct+partial`，round3=`direct+partial/repair_required`。此前 BBH/ClickBell 另有 provider-written scene+checker 个例 | **执行链存在，semantic preservation gate 尚未完成正验收**。VLM/expert 通过不能替代 same-seed simulator/AST authority |
| 首帧视觉检查与重新生成 | v19 两个动态 candidate 都真实运行 VLM visual diagnosis 与 expert preflight，但它们没有发现 geometry authority 缺失和 contact-point 漂移 | **当前关键方法 gap**。代码已把 preservation `false` 接到一次局部 repair；scale+center+contact feasibility gate 已实现并通过服务器定向反例测试，仍需可实现 candidate 的在线正验收 |
| ToolGen retrieve/generate/validate/register/reuse | v19 round2 生成 XY Tool，live=`0.0077674431 m`；round3 exact run-local reuse，live=`0.0452577472 m`，两次均执行于真实 episode | **Tool 机械链闭合，Query 归因未闭合**。数值描述实际混杂场景，不能被解释为纯 color/size 改变的证据 |
| rollout → Rule/VQA → Aggregate → Planner → Answer | v19 三次 ACT、动态 TaskGen、Rule/VQA、Aggregate、transition 与 Answer 由一个命令完成，无缓存 rollout replay；最终 semantic audit 否决结果 | **在线机械链完整，方法结论未完成**。正确状态是 preservation-confounded negative，而不是 evidence-sufficient flagship |
| 更少 samples/time 得到与 dense benchmark 可比结论 | batch24 ClickBell 三 seed 实验中，dense 12 ACT→adaptive 6 ACT；universal conclusion 与 paired weak-axis 均 3/3 一致，但完整 failure set 仅 2/3。batch26 单 seed universal-refutation 另有 4→1 ACT 的早停案例 | **受限 toy 正证据，不能称 Tables 1–2 复现**。只证明预注册结论字段在该有限协议下可保真；“完整弱项/最差候选”仍未保真 |
| 少样本保持 ACT、DP、DP3、RDT、π0 相对排名 | 既有 ACT/DP3 三 seed pilot 为 2/3 对 2/3，平局。batch26 N=5 预注册先跑共享 expert gate，仅 2/5 seed eligible，按协议在策略 rollout 前中止 | **未复现**。pair order 与 Spearman 为空；本批 5 次 expert probe、0 policy rollout，未通过 gate 不能靠筛 seed 制造排名 |
| RAG、visual self-check、README.Agent 提升 codegen 成功率 | 五个冻结 unseen BBH Proposal × 五条件、25 次 provider generation：complete=4/5、base=0/5、−RAG=0/5、−visual=3/5、−README=5/5；0 ACT | **仅 RAG 有方向性信号**。单模型族、单 repeat、development-agent proxy；visual/README 未复现论文方向，不能称 Table 3 |
| Plan 与机器人研究者 sub-aspect 标注一致 | 30 条五类 Query；单 Codex development proxy 下 precision=0.513、micro-F1=0.519、exact=0.367 | **未复现论文人工有效性**。human_count=0；已保留可直接替换 annotation 的冻结包 |
| VQA 在四种视觉条件下保持 accuracy/AUROC | 8 个缓存样本、每条件一正一负；proxy accuracy=0.875、AUROC=1.0 | **协议 smoke**。human_count=0、单 VLM、扰动非 simulator-native；不能作为 Tables 7–8 |
| 约 5% 系统错误率及模块分布 | 冻结 10 个语义 terminal operations，10/10 pass，0 个 paper-defined error | **未复现 Fig. 6**。分母虽达到最小可报告阈值，但 10 个样本无法估计约 5% 或稳定模块分布 |
| 数百条、五类、人工 sub-aspect 的开放 Query 数据集 | 30 条中文 proxy Query，五类各 6 条，并留有四人多数票替换槽 | **只完成最小协议样本**，不是论文数据集贡献 |
| 多任务、跨 policy、RoboTwin/LIBERO 一致性 | RoboTwin 有五个 official `TaskAdapter`，深入证据仍集中在 BBH/ClickBell，AdjustBottle 有一次生成式运行；GrabRoller/PlacePhoneStand official-only。LIBERO batch27 task0 是 official-positive/custom-negative 两轮 basic adaptation | **未复现跨任务/跨环境结论**。适配器广度不能代替生成式闭环或一致性实验 |
| 相对 benchmark 的可解释、动态生成、开放工具能力 | 当前受审计 bundle 能保留 Query、FreeConcern、生成代码、render、rollout、Tool/VQA、Aggregate、Planner decision 与受限 Answer | **受限实现**。项目不声称 traditional benchmark absolute correctness；生成 checker 只拥有其显式实验语义的 authority |

## 第一性原理上的首要 gap

方法优先于扩大实验规模，当前应按以下顺序处理：

1. **先闭合 preservation authority**：exact spatial/contact 约束必须比较 same-seed
   simulator state；shape/size 等 geometry 若没有 simulator/AST authority 必须为
   `unknown`；明确 `false` 只允许一次局部 repair。代码已加入
   scale+center+contact 的生成前 feasibility gate，并完成服务器定向反例验收；下一步是
   用可实现 candidate 重跑 v19 型 Query，取得在线正验收。
2. **彻底分离“语义开放”与“执行绑定”**：catalog 已不再是 candidate 许可表，但当前
   `OpenWorldEvaluationTarget` 为兼容检索仍携带 `aspects/control_template_id`。
   下一步可把生产 target 缩到 task/checkpoint/schema/render/rollout hooks；检索到的
   artifact hints 单独传入 TaskGen/ToolGen。这样新增 concern 不再以 catalog-shaped
   transport 表达。
3. **取得一个语义有效的干净在线旗舰后再扩 need 组合**：随后才验证 scene-only、
   checker-only、Tool-only、不同真值条件和跨 evaluation reviewed Tool reuse；不为
   GrabRoller/PlacePhoneStand 复制深链。
4. **方法主干稳定后再验证结论充分性**：预注册有限 candidate universe、coverage 与
   stop contract，在 3 个 seed 上比较 adaptive 与 dense 的原 Query verdict/最弱轴。
   只有必需字段一致且 rollout/time 同时下降才是 Tables 1–2 的小型正证据。
5. **实验性证据继续后置**：LIBERO 多 seed、独立人工 Plan/VQA、Table 3 归因与
   ACT/DP3 ranking 都重要，但不能替代 RoboTwin 主方法同包闭环。严格 gate 失败本身应
   报告，不能事后筛 seed。

后续只保留直接支撑上述 claim 的主链和实验入口；不恢复已被替代的平行 planner、
中央 whole-round recovery、多层 provenance/receipt 封装、旧 evidence bundle 或
逐批 development log。服务器完整回归为 `903 passed + 181 subtests`，这只证明代码级
闭合，不是论文实验结果。根 README 继续不改。
