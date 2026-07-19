# 2026-07-19：声明式 capability、真实模块开关、整轮恢复与 scene pair

## 1. 本批目标与论文对应

本批不扩展 policy，也不追求大样本结果；继续只用 ACT，以最小成本补齐论文主体方法的控制流和
可审计证据：

| 实现 | 论文对应 | 本批证据边界 |
| --- | --- | --- |
| 通用 declarative capability adapter | Sec. 3.3、Figs. 3–4 | BBH 与 `click_bell` 共用 Plan → VariantSpec → Tool/VQA/gates 合同 |
| aspect canonicalization 与语义范围 | Sec. 3.2、Table 6 | 显式 alias、object/scene/performance/execution scope；不是论文人工标注精度 |
| 真正执行的 TaskGen/ToolGen switches | Sec. 3.3、Table 3 | 7 条 0-provider/0-simulator/0-ACT 控制流 smoke；不是生成成功率 |
| completed-run scene collector | Tables 7–8 | 真实 artifact 的 hash、绑定和缺失诊断；没有 gold label |
| registered fixed/dynamic ACT pair | Tables 1–2 的效率机制 | 单任务、单 seed、4 ACT 的 micro-pilot；不进入论文表格 |
| stage-aware whole-round recovery | App. A.3.4 | typed action table 与 0-ACT 单元测试；注册实验明确禁用恢复 |

## 2. 实现摘要

### 2.1 一份 capability 合同贯穿整条链

`mea/capability_adapter.py` 把受信 template 编译为同一合同：canonical aspect、TaskGen operation、
capability/task-variant identity、允许修改的根字段、Tool metric、VQA phenomena 和 required gates。
`mea/aspects.py` 只接受显式 alias；object 变化只能改受信 object 根，scene 变化只能改
`domain_randomization`。

Agent 在 TaskGen 前核对 planner 输出与合同；TaskGen 物化后再次 exact bind。BBH 的 VariantSpec
由 planner capability contract 直接授权，provider proposal 不能覆盖；official passthrough 还会
验证 task module、固定 official spec 与空 overlay。Tool request、VQA allowlist 和 gate list 也必须
与同一轮的合同逐项一致，防止跨轮或跨任务 artifact 被误接入。

### 2.2 Table 3 开关与 typed outcomes

`mea/module_ablation_execution.py` 实际执行论文列出的 TaskGen
`complete/no_rag/no_visual_self_check/no_readme_agent/base` 和 ToolGen `complete/no_rag` 分支。
每个 item 输出 candidate、execution trace、模块调用计数、deterministic judge 和 typed outcome；
development artifact 与 formal artifact root 分离。内置 smoke 的 7 个 condition 全部按预期执行：
complete 分支通过，移除必要模块的分支由 judge 拒绝，Tool validation 在两条 ToolGen 分支均保持
启用。运行计数固定为 provider/simulator/ACT = 0，`paper_table_eligible=false`。

### 2.3 stage-aware whole-round recovery

只有 typed `tool_execution/unexpected_exception` 可以触发新 attempt、新 child run、新 execution
目录和新 ACT；policy/simulator 失败不重试，未知异常不升级为整轮恢复。失败 attempt 保留，汇总
记录 restart 次数、总 ACT 和 recovery 额外 ACT。registered execution 强制两个 recovery budget
都为 0，保证预注册样本预算不会被自动扩大。

### 2.4 scene-shift collector

collector 只扫描 completed parent，验证 parent-child membership、TaskGen scene contract、ACT
episode/video、Execution VQA/query/montage 的相互绑定，并对七类来源逐文件写 SHA-256。它不会
启动 provider、simulator 或 ACT，也不会把被测 VQA prediction 当作 gold label。

真实产物第一次收集暴露两个 post-hoc validator bug：

1. Agent 在 provider 响应通过后追加了 `evidence_conflict`，collector 却把该派生字段送回严格
   provider schema，导致四条 VQA artifact 被误判；
2. static lighting 正确使用 `random_light=true`、`crazy_random_light_rate=0` 和 simulator light
   colors，但 collector 错误地同时要求只属于 unseen-texture route 的 `eval_mode=true`。

修复只修改 collector/validator 和回归 fixture，没有修改或重跑 ACT。`scene_collection_v2` 对原有
4 个候选得到 ready `4/4`、diagnostic `0`；加入未知扩展字段、派生冲突一致性与布尔 rate
fail-closed 后，最终严格版本的 `scene_collection_v4` 结果相同。每个 condition 仍只有 1 个
unique seed，label status 为
`not_requested`，因此 `suite_validated=false`、`paper_table_eligible=false`。

## 3. 注册 fixed/dynamic micro-pilot

实验冻结在 commit `37fbaef3c552ac7e270c493611da5080f7a75439`：

- registration id：`batch9_scene_pair_37fbaef`；
- manifest payload SHA-256：
  `951f0f9f37e4d4586e4cc7d2bc6cdd4eb1b3ced751d981cd9977e5ef7d2d992e`；
- command-plan SHA-256：
  `3bf4d1602f58f4fc688476598c41cc2cec4a5ae260430c9cdfde5954ade59191`；
- task/policy：`click_bell` / ACT `demo_clean-50`；
- candidates：`scene_background_texture.unseen`、`scene_lighting.static_random`；
- seed：`100402`；fixed 2 ACT + dynamic 2 ACT，总计恰好 4；
- recovery budgets：Tool `0`、whole-round `0`，实际 restart 和额外 ACT 均为 `0`。

真实结果：

| Strategy | Unseen texture | Static lighting | Successes / ACT | Rollout wall time |
| --- | ---: | ---: | ---: | ---: |
| fixed predeclared | fail | pass | 1 / 2 | 87.18 s |
| dynamic evidence | fail | pass | 1 / 2 | 86.59 s |

两条策略在两个 `(variant_id, seed)` 上 exact success agreement 为 `1.0`。texture rollout 都执行到
400 policy steps 后失败；lighting rollout 都在 69 policy steps 成功。所有轮次的 capability binding、
scene/render/rule/expert/ACT/Tool/Aggregate/Execution-VQA gates 均通过；这里的 pipeline pass 表示
证据链完整，不等于 policy success。

动态 Planner 在第一轮读到 `policy_success=0`、Aggregate 完整且 VQA 无冲突；因为 background
texture 方面没有剩余 drill-down template，而 lighting 尚未覆盖，所以选择
`switch_aspect → scene_lighting.static_random`。第二轮成功且预算耗尽后停止。这证明了真实证据驱动
transition，但冻结 suite 只有两个候选，fixed 也必须跑两轮，故 rollout savings 为 `0`。

这不是论文 Table 1 标准 benchmark，也无法在 N=1 下计算 Table 2 consistency；结论仅限于
“注册身份、完整调用链、证据驱动决策和比较器能够真实运行”。

## 4. 验证

- Windows 小型 fixture：scene collector / VQA validator `21/21` 通过；
- 服务器 RoboTwin Python：全套 `378/378` 通过，耗时约 40 秒；
- 0-ACT module-switch smoke：7 个论文 condition 全部产生 typed outcome；
- registered pair：两个 parent 均为 `completed`，总 ACT 恰好 4；
- post-hoc collector v4：ready `4/4`、diagnostics `0`，未调用 provider/simulator/ACT。

## 5. 仍未补齐的论文 gap

1. provider 与 ACT 目前统计完成调用；若进程在调用中途崩溃，尚缺独立的 call-start ledger。
2. scene suite 尚未把 `round_id + evidence_bundle hash` 放进 suite schema；当前依靠七类逐文件 hash
   和 parent/child/result 自绑定。
3. Tables 7–8 仍缺每个 condition 至少 2 个独立 seed 与独立人工 gold；当前 4 条 candidate 来自
   fixed/dynamic 对同一 seed 的重复观察。
4. Table 3 只有真实开关与 deterministic judge，尚无 live provider generation + 独立审核的微型
   matched outcomes。
5. 主体方法仍缺一次受控的 TaskGen scene error → visual diagnosis → repair 真实演示，以及把
   `click_bell`/BBH 的通用开放 query 结果合成一个最终强项、弱项、建议和局限报告。
6. 仍只使用 ACT；这是当前有意范围，不是遗漏。

## 6. 下一批建议

按“论文核心 gap / 实现成本”排序：

1. **调用开始账本 + round provenance hash（0 ACT）**：补齐 provider/ACT 中途崩溃审计，并把
   round/evidence identity 纳入 suite。对应 Sec. 3.3、App. A.3.4，成本低，先做。
2. **真实 scene error → diagnosis → repair（0–1 ACT）**：注入一种确定的错误 overlay，先让 visual
   gate 失败，再依据 typed diagnosis 修复；修复后最多跑 1 次 ACT。对应 Fig. 3，是 TaskGen 反馈闭环
   最关键的剩余功能证据。
3. **scene VQA 小 suite（2–4 ACT）**：每个 scene condition 补到 2 个 unique seeds；先由开发代理
   临时代标并明确 `unvalidated`，之后再换人工 gold。对应 Tables 7–8。
4. **旗舰开放 Query 最终报告（1–3 ACT）**：同一入口路由到 `click_bell` 与 BBH 已有 capability，
   汇总强项、弱项、建议和局限。对应 Fig. 2、Secs. 3.2–3.4。

完成前两项后，论文主体方法的功能闭环基本齐全；再完成第 3–4 项，可称“整篇论文主体的小规模
功能复现”，但实验规模和多 policy 对照仍明确缩减。
