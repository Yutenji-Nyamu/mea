# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–6、Tables 1–10 与 Appendix。状态严格区分
“接口存在”“小规模真实闭环”和“达到论文实验规模”。最新 batch28 v4 已完成一次
broad Query 驱动的两 ACT 真实闭环；跨环境结果见
[batch27 unified adapter / LIBERO evidence](../experiments/paper/results/batch27_unified_adapter_libero/)，
上一批协议结果见
[batch26 claim closure](../experiments/paper/results/batch26_claim_closure/summary.json)，
干净旗舰的逐步数据流见
[current evidence](evidence/current/README.md)。

| 论文 claim | 当前项目证据 | 判断 |
| --- | --- | --- |
| 开放 Query 驱动多轮评估；evidence 决定下一步 | batch28 v4 从 broad Query 出发，先执行 neutral ClickBell control，再由 control evidence 在线产生 runtime candidate；provider 生成 scene+checker，通过真实 visual/expert gate 后启动第二次 ACT | **开放主链已有单次真实验收**。不再依赖预先给定 template/aspect/candidate，但仍是单任务、单 seed、两 ACT，不能据此声称广泛开放评估 |
| 证据充分后停止并回答原 Query | batch26 v6 已展示 `evidence_sufficient` 停止；batch28 v4 则在两 ACT 后得到 `budget_exhausted`、`evidence_sufficient=false`，最终诚实返回 inconclusive；后续零 ACT Tool 语义修复不改变该结论 | **两种终止语义均已真实执行**。batch28 证明系统没有把完整链路或事后修复误报成充分结论；剩余问题是候选 coverage、预算与多 seed 结论充分性 |
| TaskGen 为同一 Proposal 生成 scene 与 `check_success()`，并裁决 rollout | BBH 与 batch26 ClickBell 已有正例；batch28 v4 又把在线 runtime candidate、provider scene+checker、visual/expert gate 与第二次 ACT 放进同一链。另一个独立改写 Query 对同一 Task 达成 exact reuse，新增 0 ACT | **Task 生成与复用路径均有真实证据**。0-ACT exact reuse 只证明语义匹配和产物复用，不是新的 policy 性能样本；仍缺跨多个 unseen Proposal 的稳定成功率 |
| 首帧视觉检查与重新生成 | 通用 wrong-color 案例完成 fail→一次 provider repair→rerender→accept；batch26 干净旗舰无需 repair | **机制有真实个例**，但没有证明 visual self-check 提高总体生成成功率 |
| ToolGen retrieve/generate/validate/register/reuse | jerk Query 已完成新 metric→oracle→真实 telemetry 非空值→第二 Query exact reuse；batch26 checker Tool 已进入 Aggregate/Planner。batch28 v4 原先固定使用 left TCP 的数值已判为无效并排除，当前代码对此 fail-closed；零新 ACT 修复在缓存 control/translated telemetry 上生成 `bell_active_tcp_min_xy_error`，值分别为 0.0092059225 m/0.0057088756 m，两个 episode 的 `active_arm` 均为 `right`。第二个改写 Query 路由为 `run_local_reuse`，且 codegen `provider_called=false` | **Tool 的语义绑定、非空执行与 run-local reuse 已有修复证据**。这是复用既有 telemetry 的零新 ACT Tool 证据，不是新 policy 证据，也不建立 metric 的统计有效性；run-local 与 reviewed registry 的物理存储仍未合并 |
| rollout → Rule/VQA → Aggregate → Planner → Answer | batch28 v4 完成 neutral control、在线 candidate、provider TaskGen、visual/expert、第二次 ACT 与 Aggregate，但原 typed MetricSpec 的 fixed-left 语义错误在运行后才发现，现已从有效证据排除；正确 Tool 使用同批缓存 telemetry 重算，尚未重新进入该次在线 Planner。整体 Query 始终返回 inconclusive | **RoboTwin 小范围基本实现，但本批不是完全无瑕的同包 Tool 正例**。链路完成、事后语义修复与 Query 得到充分答案是三件事 |
| 更少 samples/time 得到与 dense benchmark 可比结论 | batch24 ClickBell 三 seed 实验中，dense 12 ACT→adaptive 6 ACT；universal conclusion 与 paired weak-axis 均 3/3 一致，但完整 failure set 仅 2/3。batch26 单 seed universal-refutation 另有 4→1 ACT 的早停案例 | **受限 toy 正证据，不能称 Tables 1–2 复现**。只证明预注册结论字段在该有限协议下可保真；“完整弱项/最差候选”仍未保真 |
| 少样本保持 ACT、DP、DP3、RDT、π0 相对排名 | 既有 ACT/DP3 三 seed pilot 为 2/3 对 2/3，平局。batch26 N=5 预注册先跑共享 expert gate，仅 2/5 seed eligible，按协议在策略 rollout 前中止 | **未复现**。pair order 与 Spearman 为空；本批 5 次 expert probe、0 policy rollout，未通过 gate 不能靠筛 seed 制造排名 |
| RAG、visual self-check、README.Agent 提升 codegen 成功率 | 五个冻结 unseen BBH Proposal × 五条件、25 次 provider generation：complete=4/5、base=0/5、−RAG=0/5、−visual=3/5、−README=5/5；0 ACT | **仅 RAG 有方向性信号**。单模型族、单 repeat、development-agent proxy；visual/README 未复现论文方向，不能称 Table 3 |
| Plan 与机器人研究者 sub-aspect 标注一致 | 30 条五类 Query；单 Codex development proxy 下 precision=0.513、micro-F1=0.519、exact=0.367 | **未复现论文人工有效性**。human_count=0；已保留可直接替换 annotation 的冻结包 |
| VQA 在四种视觉条件下保持 accuracy/AUROC | 8 个缓存样本、每条件一正一负；proxy accuracy=0.875、AUROC=1.0 | **协议 smoke**。human_count=0、单 VLM、扰动非 simulator-native；不能作为 Tables 7–8 |
| 约 5% 系统错误率及模块分布 | 冻结 10 个语义 terminal operations，10/10 pass，0 个 paper-defined error | **未复现 Fig. 6**。分母虽达到最小可报告阈值，但 10 个样本无法估计约 5% 或稳定模块分布 |
| 数百条、五类、人工 sub-aspect 的开放 Query 数据集 | 30 条中文 proxy Query，五类各 6 条，并留有四人多数票替换槽 | **只完成最小协议样本**，不是论文数据集贡献 |
| 多任务、跨 policy、RoboTwin/LIBERO 一致性 | RoboTwin 现有五个 official `TaskAdapter`：BBH/ClickBell 深入，adjust_bottle/grab_roller/place_phone_stand 仅 official-only；place_phone_stand expert N=1 成功、ACT N=1 失败。LIBERO batch27 task0 official 成功后执行 evidence-triggered custom BDDL，custom 失败；两回合共 132.698 s，Tool exact reuse，`method_chain_valid=true`、`query_sufficient=false` | **广度接口扩大，跨环境方法链有最小结构证据**。不能把“五任务接入、两任务深入”称为五任务复现；LIBERO 结果只证明 basic-adaptation chain 可执行，未证明 robustness、RoboTwin/LIBERO 一致性或论文表格结论 |
| 相对 benchmark 的可解释、动态生成、开放工具能力 | 当前旗舰能保留 Query、FreeConcern、生成代码、render、rollout、Tool/VQA、Aggregate、Planner decision 与受限 Answer | **受限实现**。项目不声称 traditional benchmark absolute correctness；生成 checker 只拥有其显式实验语义的 authority |

## 第一性原理上的首要 gap

1. **结论充分性与保真，而不只是跑完整条链**：batch28 v4 已关闭 broad Query 到
   runtime candidate、Task/Tool 与 Aggregate 的 live 接线 gap，但两 ACT 后仍是
   `budget_exhausted`/inconclusive。下一步应预注册有限 candidate universe、coverage
   与停止合同；在 3 个 seed 上验证比 universal-refutation 更难的“完整弱项/最差属性/
   比较”claim。只有 adaptive 与 dense 的全部必需字段一致且 rollout/time 下降才算正结果。
2. **从接口广度推进第三个 live 深入任务**：generic adapter 已从 official
   source/schema 自动发现任务，不再要求为 adjust_bottle/grab_roller/place_phone_stand
   写专属方言；但后三者仍无完整的 model-written
   scene+checker→rollout→Tool/VQA→Answer 证据。只需选一个 official control 成功的
   任务做两回合正例，不应为所有可发现任务复制特定 planner。
3. **LIBERO 结论充分性**：两回合方法链已合法执行，但 official positive/custom
   negative 在单 seed 下只能回答“该 variation 出现失败”，且 Planner 继续请求证据，
   最终因预算停止。下一步应先预注册有限 candidate universe 与停止合同，再以最多
   3 个 seed 判断同一 Query；不扩 OpenVLA、更多 suite 或大规模 ranking。
4. **独立有效性**：Plan/VQA 已有冻结 prediction 与可替换 annotation 包。真正缺的是
   四名独立机器人标注者、senior tie-break、真实 simulator-native 四条件 clips 与
   多 VLM，而不是更多 development proxy。
5. **生成归因与 policy ranking**：Table 3 需要独立人工盲评和少量重复 generation；
   ACT/DP3 需要先得到共同 eligible seeds，第三策略与五策略 Spearman继续后置。严格
   gate 失败本身应报告，不能事后筛 seed。

后续只保留直接支撑上述 claim 的主链和实验入口；不恢复已被替代的平行 planner、
中央 whole-round recovery、多层 provenance/receipt 封装、旧 evidence bundle 或
逐批 development log。根 README 继续不改。
