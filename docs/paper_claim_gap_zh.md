# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–6、Tables 1–10 与 Appendix。状态严格区分
“接口存在”“小规模真实闭环”和“达到论文实验规模”。方法实现的最新证据为
`eval_20260728_adjust_bottle_open_live_v2` 及其 append-only、0-ACT Tool/VQA replay；
源在线 bundle 始终保持 inconclusive，replay 不倒改其结果。跨环境结果见
[batch27 unified adapter / LIBERO evidence](../experiments/paper/results/batch27_unified_adapter_libero/)，
上一批协议结果见
[batch26 claim closure](../experiments/paper/results/batch26_claim_closure/summary.json)，
干净旗舰的逐步数据流见
[current evidence](evidence/current/README.md)。

| 论文 claim | 当前项目证据 | 判断 |
| --- | --- | --- |
| 开放 Query 驱动多轮评估；evidence 决定下一步 | 生产入口现在由 `ClaimFirstInitialPlanBuilder` 直接建立初始计划，不调用 `CatalogPlanAgent` 或任务专属 Planner；`OpenWorldPlanSession.from_target()` 只接收已冻结的 task/checkpoint binding。AdjustBottle v2 的 Query 未提供 aspect/template：FreeConcern 先提出未见瓶体几何，official control evidence 再促使 Planner 在线提出 `task_execution.success_margin_components` | **开放语义主链已有第二任务真实验收**。catalog 仅作 retrieval index，typed scene/checker/tool need 由 runtime candidate 决定；但仍是单 task/seed 的两轮案例，而且 target transport 为兼容旧工件仍携带 catalog-shaped aspects，不能声称开放域完成 |
| 证据充分后停止并回答原 Query | AdjustBottle v2 源在线 bundle 因原 Tool=null、VQA 问题跨任务而以 `budget_exhausted`/inconclusive 停止。中间 Tool-only repair v3 曾得到 `evidence_sufficient/diagnosed`，但仍消费源错误 BBH VQA。canonical composed v7 同时消费修正 Tool 与 task-owned VQA 后得到 `should_stop=true`、`stop_reason=evidence_conflict`、`verdict=inconclusive`、`evidence_sufficient=false`；batch26 另有在线有限合同的 sufficient stop | **Planner 会因充分证据或证据冲突按不同语义停止，且组合 replay 没有掩盖冲突**。v7 是 0-ACT 方法审计，源在线结果未改写；还缺一次修正后生产代码的干净在线同包 stop |
| TaskGen 为同一 Proposal 生成 scene 与 `check_success()`，并裁决 rollout | BBH 与 ClickBell 已有正例；AdjustBottle v2 把 runtime candidate、provider-written scene/checker、静态验证、2/2 checker fixture、VLM visual diagnosis（0.88）、expert gate 与第二次 ACT 放进同一在线链 | **第三任务已有生成式方法证据**。该 checker 的 policy 结果为 false，只裁决实验语义，不能视为 official-equivalent；单例不证明跨 unseen Proposal 的生成稳定性 |
| 首帧视觉检查与重新生成 | AdjustBottle v2 确实调用 VLM visual diagnosis 并以 0.88 通过，未触发 repair；既有 wrong-color 案例完成 fail→一次 provider repair→rerender→accept | **诊断与一次局部 repair 的两个分支均有个例证据**，但还没有在同一通用冷启动任务上触发并恢复，也没有证明 Table 3 的成功率增益 |
| ToolGen retrieve/generate/validate/register/reuse | jerk Query 已有新 metric→oracle→真实非空值→exact reuse。AdjustBottle v3/v6 使用通用 `terminal_signal_component`：首次 `typed_metric_spec_compile`、`provider_called=false`，测得 `bottle_functional_position.z=0.771909236907959 m`；相同第二 Query 为 `run_local_reuse`、`provider_called=false`，Aggregate passed | **metric need→typed Tool→非空 live telemetry→Aggregate→exact reuse 已在同一已完成 rollout 上统一**。v6 又把 task-owned VQA 合入 Planner；这是 0-ACT 缓存 replay，不是新的 policy sample，仍需生产在线路径自然生成同类 Tool |
| rollout → Rule/VQA → Aggregate → Planner → Answer | AdjustBottle v2 在线完成两轮 rollout、TaskGen、Rule/VQA、Aggregate 与 Answer，但原 Tool=null，且无匹配问题时错误继承 BBH VQA。`vqa_task_owned_replay_v1` 只询问 `bottle_visibly_repositioned`，VLM=true/0.98；因 generated/official 核心 predicate=false，保留 `numeric_consistency=conflict`。canonical composed `terminal_tool_plus_task_vqa_repair_v7` 将该 VQA 与 terminal Tool 在同一次 replay 中重算，EvidencePacket=`conflicting`，Planner 以 `evidence_conflict` 停止并返回 inconclusive | **反馈闭环的相关性、组合和冲突保真已有追加式方法证据**。v7 是对已完成 rollout 的审计，不是无瑕在线旗舰；它正确地拒绝让高置信 VQA 覆盖 Rule/checker 冲突 |
| 更少 samples/time 得到与 dense benchmark 可比结论 | batch24 ClickBell 三 seed 实验中，dense 12 ACT→adaptive 6 ACT；universal conclusion 与 paired weak-axis 均 3/3 一致，但完整 failure set 仅 2/3。batch26 单 seed universal-refutation 另有 4→1 ACT 的早停案例 | **受限 toy 正证据，不能称 Tables 1–2 复现**。只证明预注册结论字段在该有限协议下可保真；“完整弱项/最差候选”仍未保真 |
| 少样本保持 ACT、DP、DP3、RDT、π0 相对排名 | 既有 ACT/DP3 三 seed pilot 为 2/3 对 2/3，平局。batch26 N=5 预注册先跑共享 expert gate，仅 2/5 seed eligible，按协议在策略 rollout 前中止 | **未复现**。pair order 与 Spearman 为空；本批 5 次 expert probe、0 policy rollout，未通过 gate 不能靠筛 seed 制造排名 |
| RAG、visual self-check、README.Agent 提升 codegen 成功率 | 五个冻结 unseen BBH Proposal × 五条件、25 次 provider generation：complete=4/5、base=0/5、−RAG=0/5、−visual=3/5、−README=5/5；0 ACT | **仅 RAG 有方向性信号**。单模型族、单 repeat、development-agent proxy；visual/README 未复现论文方向，不能称 Table 3 |
| Plan 与机器人研究者 sub-aspect 标注一致 | 30 条五类 Query；单 Codex development proxy 下 precision=0.513、micro-F1=0.519、exact=0.367 | **未复现论文人工有效性**。human_count=0；已保留可直接替换 annotation 的冻结包 |
| VQA 在四种视觉条件下保持 accuracy/AUROC | 8 个缓存样本、每条件一正一负；proxy accuracy=0.875、AUROC=1.0 | **协议 smoke**。human_count=0、单 VLM、扰动非 simulator-native；不能作为 Tables 7–8 |
| 约 5% 系统错误率及模块分布 | 冻结 10 个语义 terminal operations，10/10 pass，0 个 paper-defined error | **未复现 Fig. 6**。分母虽达到最小可报告阈值，但 10 个样本无法估计约 5% 或稳定模块分布 |
| 数百条、五类、人工 sub-aspect 的开放 Query 数据集 | 30 条中文 proxy Query，五类各 6 条，并留有四人多数票替换槽 | **只完成最小协议样本**，不是论文数据集贡献 |
| 多任务、跨 policy、RoboTwin/LIBERO 一致性 | RoboTwin 有五个 official `TaskAdapter`；BBH/ClickBell 较深入，AdjustBottle 有一次真实生成式链及独立修复 replay，GrabRoller/PlacePhoneStand 仍为 official-only。LIBERO batch27 task0 official 成功后执行 evidence-triggered custom BDDL，custom 失败；两回合共 132.698 s，Tool exact reuse，`method_chain_valid=true`、`query_sufficient=false` | **方法广度推进到第三任务，但还不是三个干净旗舰**。LIBERO 只证明 basic-adaptation chain 可执行；没有证明 robustness、RoboTwin/LIBERO 一致性或论文表格结论 |
| 相对 benchmark 的可解释、动态生成、开放工具能力 | 当前旗舰能保留 Query、FreeConcern、生成代码、render、rollout、Tool/VQA、Aggregate、Planner decision 与受限 Answer | **受限实现**。项目不声称 traditional benchmark absolute correctness；生成 checker 只拥有其显式实验语义的 authority |

## 第一性原理上的首要 gap

方法优先于扩大实验规模，当前应按以下顺序处理：

1. **一条无需事后 repair 的干净在线主链**：直接 ClaimFirst 已摆脱 CatalogPlan 壳，
   generic TaskGen、terminal Tool 与 task-owned VQA 已在 composed v7 中共同通过方法
   replay，但正确证据仍来自 0-ACT append-only 审计。下一次最小 live 应让修正后代码在原始
   bundle 内自然完成非空 Tool、相关 VQA、冲突保留、Aggregate、Planner stop 与 Answer；
   不要求把冲突强行变成成功结论。
2. **彻底分离“语义开放”与“执行绑定”**：catalog 已不再是 candidate 许可表，但当前
   `OpenWorldEvaluationTarget` 为兼容检索仍携带 `aspects/control_template_id`。
   下一步可把生产 target 缩到 task/checkpoint/schema/render/rollout hooks；检索到的
   artifact hints 单独传入 TaskGen/ToolGen。这样新增 concern 不再以 catalog-shaped
   transport 表达。
3. **通用 TaskGen/ToolGen 的多样性而非任务方言数量**：AdjustBottle 已证明无需新增
   任务名 Planner/TaskGen 分支也能冷启动。下一步只需再用少量 catalog-external
   scene-only、checker-only、Tool-only Query 验证 typed need 的独立组合，以及一次真实
   visual fail→单次 repair；不为 GrabRoller/PlacePhoneStand 复制深链。
4. **方法主干稳定后再验证结论充分性**：预注册有限 candidate universe、coverage 与
   stop contract，在 3 个 seed 上比较 adaptive 与 dense 的原 Query verdict/最弱轴。
   只有必需字段一致且 rollout/time 同时下降才是 Tables 1–2 的小型正证据。
5. **实验性证据继续后置**：LIBERO 多 seed、独立人工 Plan/VQA、Table 3 归因与
   ACT/DP3 ranking 都重要，但不能替代 RoboTwin 主方法同包闭环。严格 gate 失败本身应
   报告，不能事后筛 seed。

后续只保留直接支撑上述 claim 的主链和实验入口；不恢复已被替代的平行 planner、
中央 whole-round recovery、多层 provenance/receipt 封装、旧 evidence bundle 或
逐批 development log。根 README 继续不改。
