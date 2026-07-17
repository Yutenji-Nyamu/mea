# 2026-07-17 Planner/VQA 代理验证、真实 Clutter 与协议补齐

本批面向 ManipEvalAgent 论文 Tables 1–3、6–8 和 Sec. 3.2–3.3.2，目标是用 ACT-only、
`N=1` 或缓存证据补齐五条最小可运行通路。这里的人工环节由 Codex 临时充当
`development_agent_proxy`；所有产物固定声明 `human_reviewer_count=0`、
`paper_table_eligible=false`，不能写成 human gold、majority vote 或论文表格复现。

## 1. 本批实现

### 1.1 Table 6-facing：20-query live Planner scorer

- 新增 20 条经过开发代理复核的开放 query/aspect 数据集，覆盖论文的 object、scene、
  performance、safety、language/multi-task 五类。
- scorer 对每条 query 真实调用 `GlobalQueryRouter`，检查 schema、任务、受支持能力、
  task-qualified unsupported gap、aspect 集和 first aspect；预算只允许 `1/3/5/20`。
- 提交后的 runner 固定 dataset 与 catalog SHA-256，防止同名数据集或能力目录静默漂移；本批
  budget-5 artifact 生成在该加固前，保留为开发结果，不冒充预注册 run。
- unsupported gap 从全局 aspect 字符串改为 `(task_name, aspect_id)`，消除“某能力在一个
  任务受支持，便被误判为所有任务都受支持”的 catalog union 漏洞。

真实 budget-5 开发验证：

```text
run: mea/validation_runs/query_validation_20260717_batch7_budget5
schema valid rate:                 1.0000
capability decision accuracy:      0.8000
task accuracy:                     0.8000
task-qualified gap coverage:       0.6667
aspect micro precision / recall:   0.6667 / 0.5714
aspect micro F1:                   0.6154
exact aspect-set accuracy:         0.6000
first-aspect accuracy:             1.0000
provider failures:                 0
```

两个主要错误是：把 completion-time stability 错路由到 BBH pickup-to-contact；对一个
language/multi-task query 只找出一个而不是两个 unsupported gap。样本太少且标签不是独立
人工 gold，以上数值只能定位路由问题。

### 1.2 Tables 7–8-facing：RoboTwin 原生 Scene Clutter

- `click_bell/robustness.scene_clutter` 编译为 RoboTwin 原生
  `cluttered_table=true, clean_background_rate=0`，不做 RGB 后处理。
- probe 从 `task.info.cluttered_table_info` 读取实际物体列表/数量，并把 simulator authority
  写入 manifest；scene、render、rule 与 expert gate 均必须通过。
- reviewed VQAQuerySpec 只能选择代码 allowlist 中的视觉问题；spec、review、index 和内容均由
  SHA-256 锁定，路径越界、symlink、篡改和多重匹配 fail closed。
- 离线 validator 绑定 TaskGen manifest、同一 ACT episode、Execution VQA、query 和 montage，
  并要求 clean/clutter 的 seed、bell pose、quaternion、instance 相同。

真实 Clutter `N=1`：

```text
evaluation: eval_20260717_batch7_click_bell_clutter_seed100402_n1
child:      run_20260717_batch7_click_bell_clutter_seed100402_n1_round_1
seed:       100402
clutter:    10 objects
authority:  simulator_task_info:cluttered_table_info
gates:      passed
expert:     success
ACT:        failure (official_check_success=false)
VQA:        bell_visibly_pressed=false
            bell_target_selected_among_clutter=false
proxy:      both false; matched VQA
```

第一次尝试的 seed `100401` 被 scene-stability gate 因不稳定 bowl 拒绝，未启动 ACT；之后先用
0-ACT seed probe 找到稳定的 `100402`。这是 gate 正常工作，不应删除或改写为策略失败。

同 seed 的 Clean 对照和二条件汇总见本文第 2 节。

### 1.3 Tables 1–2-facing：Fixed suite 与 Dynamic MEA 对照合同

- `ClickBellFixedSuitePlanAgent` 在任何 rollout 前冻结与 dynamic Planner 相同的 candidate suite；
  后续 policy/VQA 证据仍记录，但不参与路由。
- artifact-only comparator 只接受同 task、ACT 逻辑配置、telemetry profile、base commit、开放
  query、global route、candidate-suite hash 和 `(variant_id, seed)` 样本身份；fixed 必须完整覆盖
  冻结 suite，缺失 success 与路径越界 fail closed。
- `N=1` 时 `table2_consistency=null`；没有 repetition 方差时拒绝伪造结论一致性。

本批完成的是 runner、严格合同和单元 smoke，没有花 6–8 个 ACT rollout 跑完整 fixed-vs-dynamic
矩阵。这个自定义对照只面向 Table 1 的效率机制，并非论文标准 benchmark vs MEA 的原始对照；
checkpoint 内容 hash 也尚未记录。因此只能称 plumbing，不是 Tables 1–2 的真实实验结果。

### 1.4 Sec. 3.2/3.3.2-facing：分类与 reviewed VQAQuerySpec

- capability/unsupported gap 均带 task 身份；proxy 数据集显式覆盖 performance、safety、
  robustness 等论文 taxonomy。
- Scene Clutter 的 VQA spec 由显式开发代理审核并持久注册；它只能组合
  `bell_visibly_pressed` 和 `bell_target_selected_among_clutter`，不能注入自由 prompt。
- reviewed VQA registry 与 reviewed Tool registry 都是 selection/verification 层，不能把生成内容
  自动晋升为全局 Trusted Tool 或人类 gold。

### 1.5 Table 3 / App. A.3.4-facing：微消融与有界恢复

- cached micro-ablation 读取真实 TaskGen/ToolGen artifact，用确定性 counterfactual 验证：关闭
  visual gate 会接收错误颜色，关闭 Tool AST/oracle validation 会接收非法工具；4/4 functional
  gate checks 通过。另有 1 条 RAG provenance check，但没有 matched no-RAG artifact，不计入
  functional summary，也不估计 RAG effect。
- 加固后的汇总位于
  `mea/validation_runs/validation_20260717_batch7_micro_ablation_v2/summary.json`。
- 它不启动 provider、simulator 或 ACT，故 Table 3 成功率固定不可用。
- Agent 的 Tool orchestration 仅对未预期 runtime exception 最多重试一次；完整 telemetry tree
  内容 hash 必须相同，attempt started/result append-only，禁止重跑 ACT、simulator、policy
  failure 或语义/验证失败。generated route 的 provider/registry 工作可能重复，因此这是保守工程
  retry，不是论文 App. A.3.4 的整轮 restart。
- 真实 Clutter run 中注入一次开发故障：attempt 1 失败、attempt 2 成功，
  `same_telemetry_reused=true`、`restarts_used=1`。该 v1 smoke 的 `act_rollouts_started=0` 实际含义是
  recovery 额外启动 0 次；提交后的 schema v2 已重命名为
  `additional_act_rollouts_started_by_recovery=0`，避免被误读成整轮没有 ACT。

## 2. 真实 Clean/Clutter 二条件验证

受控配对使用同一 ACT checkpoint、telemetry profile 和 seed `100402`。当前开发代理标签来自
逐帧 montage 复核，不是独立人类多数投票。

```text
validation: mea/validation_runs/vqa_real_simulator_clean_clutter_seed100402_n1_v2/
            validation_summary.json
target identity:   same seed / bell position / quaternion / instance = passed
protocol identity: same base commit / checkpoint setting / telemetry hash = passed

condition       clutter  ACT official success  VQA proxy items correct
clean           0        false                 1/1
scene_clutter   10       false                 2/2

pooled proxy accuracy: 3/3 = 1.0
coverage:              clean 1/1, scene_clutter 2/2
AUROC:                 null (N=1 per condition)
human reviewer count:  0
paper eligible:        false
```

该结果证明 parent evaluation、TaskGen child、真实 simulator 条件、ACT episode/video、query、
montage 和代理标签能被严格串联，也证明 VQA 在
这两个负样本上与开发代理一致；它不能证明 VQA 区分正负样本的能力，也不能支持“clutter 使
ACT 下降”的结论，因为 clean 与 clutter 中 ACT 均失败。Clean run 的正常 Tool analysis 只执行
一次，`restarts_used=0`；Clutter 故障注入才执行一次有界恢复。

## 3. 调用链与证据边界

```text
open query
  -> GlobalQueryRouter (task + supported capability or task-qualified gap)
  -> dynamic or fixed-suite Planner
  -> bounded click_bell TaskGen overlay
  -> simulator-native clutter + scene/render/rule/expert gates
  -> ACT N=1 + telemetry/video
  -> Trusted/Generated Tool -> deterministic Aggregate
  -> reviewed VQAQuerySpec -> Dynamic Execution VQA
  -> evidence transition / final answer
  -> offline proxy validation and protocol reports
```

权威层级保持不变：RoboTwin `check_success()` 判任务成功；simulator state 和确定性 Tool 判数值；
VQA 只判可见现象，冲突必须保留。pipeline passed 只表示链路与 gate 完成，不代表 ACT success。

## 4. 验证与问题修复

- targeted：68 tests passed；cross-task entrypoint：19 tests passed。
- cached micro-ablation：4/4 functional gate checks + 1/1 provenance check；runtime
  provider/simulator/ACT 均为 false/false/0，且没有 matched no-RAG effect estimate。
- 首个 Clean `seed=100401` 的 ACT 与 VQA 已完成，但 parent summary 暴露
  `position_metrics` 未初始化。修复为从 position evidence 显式取 metrics，并增加跨任务回归测试；
  没有把该 parent run 当作受控配对结果。
- 首次完整测试的 285 项中，旧多轮夹具缺少新恢复合同要求的 telemetry 目录，旧 capability
  断言仍写死 click_bell 只有两个能力。补齐 fixture、把断言升级为三个精确 capability id 后，
  相关回归测试通过；再加入 comparator 路径/query/coverage 与 VQA parent/video 绑定测试后，
  最终服务器全量为 `289 passed in 36.275 s`。

## 5. 尚未完成的论文距离

- Table 6 仍缺独立多人 gold/majority review 和足量 query；当前只有开发代理 budget-5。
- Tables 7–8 仍缺人类多数视频标签、每种扰动的正负样本和 `N>=3`；单样本 AUROC 必须为 null。
- Tables 1–2 的 fixed-vs-dynamic comparator 尚缺真实 matched candidate suite 实验和 repetition。
- Table 3 只有 cached/fault functional evidence，尚无 matched live ablation success rate。
- 当前 taxonomy 中 performance/safety 大多只能识别为 unsupported gap，仍缺真实 telemetry/tool/task
  capability；这比继续添加零散 metric 更重要。
