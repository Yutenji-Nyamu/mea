# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–6、Tables 1–9 与 Appendix。这里区分“接口存在”、
“机制真实跑通”和“达到论文实验规模”。

| 论文 claim | 当前项目 | 主要缺口 |
| --- | --- | --- |
| 开放 Query 驱动多轮评估；evidence 动态细化；充分后停止 | ClaimFirst 两轮 live 主链已跑通，证据会改变下一轮 | 候选域仍有界；最近运行因预算而非充分性停止；需让 ClaimFirst 成为唯一默认 Planner |
| TaskGen 检索/生成 scene 与 `check_success()`，render 后自检修复 | scene、实验 checker、render、fixture/expert gate 与 ACT 均有真实案例 | model-written scene+checker 与统一主链仍未完全合一；需允许多种语义等价代码，而非复制 reference AST |
| ToolGen retrieve/generate/validate/register/reuse | 新 XY metric 在真实 episode 得到非空值并进入 Aggregate；另有 exact reuse 案例 | “非空测量→影响 Planner→下一 Query 复用”仍分散在不同运行 |
| Rule/VQA→Aggregate→Planner→解释性回答 | 在少量 RoboTwin task、ACT、N=1/2 范围内基本跑通 | 需要更多任务和不同 evidence pattern，而不是继续增加框架层 |
| 显著减少 samples/time，同时与完整 benchmark 结论可比 | fixed 2 / adaptive 1 的 universal toy 得到相同 `refuted` | 单 task/seed 的简单反例；需非平凡 fixed 3–4 vs adaptive 1–3，随后 N=3 |
| 少样本保持 ACT/DP/DP3/RDT/π0 相对排名 | ACT/DP3 有初步接入/资产准备 | 尚无同 task、同 seed 的有效双 policy ranking；本阶段只做 ACT/DP3 pilot |
| RAG、visual self-check、README.Agent 提升生成成功率 | 对应开关和路径存在 | 缺少 matched unseen proposals、真实 codegen/render/oracle 的 Table 3 消融 |
| Plan 与机器人研究者 sub-aspect 标注一致 | 有 development-agent proxy | 缺独立人工 gold、多人一致性及论文规模 |
| VQA 在 clean/clutter/texture/lighting 下保持 accuracy/AUROC | 有缓存图像扰动 proxy | 需真实 simulator 条件、正负 clips、人工 gold 与至少多个 VLM |
| 约 5% 系统错误率及 Plan/TaskGen/ToolGen/simulator 分布 | 有小分母 prospective smoke | 固定分母过小，不能与 Fig. 6 比较 |

## 当前优先级

1. **统一 Fig. 2–5 主链（1–3 ACT）**：ClaimFirst 默认化；model-written scene+checker；
   新 Tool 在同一运行中测量、影响下一 plan，并在后续 Query exact reuse。
2. **扩大到少量任务（低成本）**：优先下载已有 ACT/DP3 checkpoint，为 3–5 个 official
   RoboTwin task 增加 TaskSchema/通用 telemetry 支持；不复制 task-specific planner。
3. **非平凡效率 toy（5–7 ACT）**：冻结候选 universe、fixed 顺序、adaptive stop 和 seeds，
   比较 verdict、rollout 数及 wall time。
4. **ACT/DP3 双 policy pilot**：同 task、同 seeds 各少量 rollout；若 tie 或失败则报告
   inconclusive，不冒充 Table 9。
5. **低 ACT validity**：真实 Table 3 小消融；先建立可交给人的 Plan/VQA 标注包。

多任务大模型、RDT、π0、LIBERO 和论文规模重复暂缓。项目的目标是逐项提供论文 claim 的
直接证据，不再增加与 claim 无关的安全封装、ledger、fallback 或平行 planner。
