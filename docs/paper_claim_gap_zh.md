# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–6、Tables 1–9 与 Appendix。状态严格区分“接口存在”、
“小规模真实闭环”和“达到论文实验规模”。

| 论文 claim | 当前项目 | 判断与首要 gap |
| --- | --- | --- |
| 开放 Query 驱动多轮评估；evidence 决定下一步；充分后停止 | ClaimFirst 已是默认自动路由；v7 中 official control 的证据触发 distractor Proposal，第二轮后由 QueryContract 以 `evidence_sufficient` 停止 | **受限闭环完成一例**。本 Query 已明确 distractor，候选域只有一个；尚未证明从宽泛 Query 自主发现未知弱点 |
| TaskGen 对同一 Proposal 生成 scene 与 `check_success()`，render 后诊断/修复并裁决 rollout | v7 由 provider 编写 target+distractor scene 与 checker；非 restricted compiler；6/6 fixtures、render/visual/expert gate 和一次 ACT 均通过 | **论文式最小案例完成**。仍只有 BBH 一个生成式 variation，缺跨 task/多 proposal 成功率 |
| ToolGen retrieve/generate/validate/register/reuse | v7 直接复用与生成 checker 绑定的 Tool，裁决同一真实 episode，进入 Aggregate 并影响最终停止/回答 | **同轮闭环完成**。缺“新 Query 诱发新 metric → 持久注册 → 第二个 Query exact reuse”的同一案例 |
| Rule/VQA → Aggregate → Planner → 可解释回答 | BBH/click_bell、ACT、N=1/2 范围内跑通；Answer 强制列 N、候选域、冲突与限制 | **基本实现，范围小** |
| 显著减少 samples/time，同时得到与完整 benchmark 可比结论 | 只有 fixed2/adaptive1 的单 seed toy/proxy | **论文最重要的科学证据仍缺失**；需预先冻结非平凡 dense reference，并真实比较 rollout 与 wall time |
| 少样本保持 ACT、DP、DP3、RDT、π0 相对排名 | 服务器有 ACT 多任务 checkpoint；官方 DP3 公共 checkpoint 仅 BBH；已有 adapter/protocol smoke | **未复现**。当前只做 ACT/DP3 同 task、同 seed 的双 policy pair-order pilot，不声称 Table 9 或 Spearman |
| RAG、visual self-check、README.Agent 提升生成成功率 | 三个开关及真实 codegen/render/oracle 路径存在 | **没有真实增益数据**；需 matched unseen proposals，而不是“开关能运行” |
| Plan 与机器人研究者的 sub-aspect 标注一致 | 只有 development-agent proxy | **缺独立人工 gold、多人一致性与论文规模** |
| VQA 在 clean/clutter/texture/lighting 下保持 accuracy/AUROC | 有缓存图像扰动和少量真实 VQA 路径 | **缺真实 simulator 四条件、正负 clips、独立 gold 和多 VLM** |
| 约 5% 系统错误率及 Plan/TaskGen/ToolGen/simulator 分布 | 已能前瞻记录 operation 状态 | **固定分母太小**，不能与 Fig. 6 比较 |
| 多任务、不同 policy 与 RoboTwin/LIBERO 一致性 | ACT official 入口覆盖 BBH、click_bell、adjust_bottle、grab_roller；生成式主链集中在前两者 | **只扩了执行面，没有方法证据面**；RDT、π0、LIBERO 按当前范围后置 |

## 当前最值得做的批次

1. **真实效率 toy（6 ACT 左右）**：预注册 4 个 BBH/click_bell 候选；fixed 全跑 4，
   adaptive 最多 2。只有结论一致且 rollout、wall time 都更少才记为正结果，否则报告未复现。
2. **跨 Query Tool reuse（0–1 ACT）**：第一个 Query 生成并持久注册 metric；第二个 Query
   exact lookup，要求 `provider_called=false`，并让该值改变停止或回答。
3. **ACT/DP3 pair-order（6 rollouts）**：BBH 上两个 policy 各跑相同 3 seeds。只能回答
   二者顺序是否稳定；两 policy 不能计算或宣称论文的多 policy ranking。
4. **Table 3 小消融（0 ACT）**：5 个冻结 unseen proposals，比较 complete、−RAG、
   −visual、−README.Agent 的 codegen→compile→render→oracle 成功率。
5. **独立有效性包**：先冻结 20–30 Query 和四类正负 clips；当前 development-agent 标注
   只用于发现协议问题，正式表格等待独立机器人研究者与多 VLM。

项目后续只增加直接支撑这些 claim 的证据，不再恢复平行 planner、中央 recovery、
receipt/ledger 或多层 registry 封装。
