# 2026-07-23：reviewed Task、partial route 与 clean-head v4

本批继续按 ManipEvalAgent Secs. 3.2–3.4、Figs. 2–5 的因果链补最小公共能力，不增加 policy，
也不把 N=1 功能验收写成论文统计复现。当前结论是：两个任务族、少量可信 capability 内已经有一条
完成态、evidence-conditioned 的两轮 clean-head 链路；Query evidence sufficiency、proposal-derived
新成功语义和论文效率结论仍未完成。

## 1. 本批实现

### 1.1 reviewed generated Task 与 production acceptance

新增 reviewed Task registry 的 template/install/find 流程，正常 Agent/TaskGen 可通过
`--reviewed-task-registry` 做 exact executable semantic lookup。复用边界分成：

- immutable copied inputs：task.py、VariantSpec、overlay/load_actors、可选 SuccessSpec、
  validation/static；
- run-local derived outputs：`TaskArtifactBundle` 与 `SceneCheckSpec`，新 run 中重新生成；
- 当前登记的 5 个 Python runtime dependency hashes。

registry-authoritative variant id 可与新 run-local id 不同，但 capability、changes、SuccessSpec、
contract 和 provenance 必须一致。production acceptance 在 ACT 前重新核验最终 candidate、场景 gate、
artifact contract 和 `act_rollouts_started=0`；它不会因为 registry 命中而跳过新 run 的物化和验证。

### 1.2 partial route

全局 Router 现在区分：

```text
同一 Query 有 supported subset
→ route 该 subset
→ 保留 task-qualified unsupported gaps

没有任何 material answer
→ status=unsupported
→ 0 ACT
```

validator 会拒绝虚构 gap、跨 task 混入 capability，或在调用方显式绑定父 aspect 时擅自扩成 partial
route。它只改变“部分可回答”的表达方式，不扩张 ACT catalog。

### 1.3 bounded Proposal 与失败耐久性

Proposal Agent 可对 v2/v3 VQA question binding 做 capability-derived structural repair，并保存 repair
trace；它不修改 scene、SuccessSpec、metric 或其他可执行语义。Proposal 阶段使用 taskgen model role，
避免把 Task Proposal 错接到 planner model role。

轮后 PlanStep 应先持久化 prompt/response/decision，再物化下一 Proposal；terminal failure 要写明
failure stage 和已完成轮数。最终源码还补了失败/restart attempt ledger 与 feedback `finally` 收口。
本批成功 v4 没有触发 whole-round recovery/restart，因此这些新增失败分支只有源码/回归合同，不能
借 v4 冒充 live restart 证据。

恢复继续遵循论文 App. A.3.4 的 stage-specific action；论文不要求所有分支共用一个中央 recovery
controller。

## 2. reviewed Task 真实复用

当前 reviewed registration：

- registration：`reviewed_task_80dbe26908344a3465c2`；
- artifact：`task_artifact_b0906f5572867b12c0c7`；
- source run：`run_20260722_batch14_bbh_scale_codegen_v1`；
- 变化：BBH block scale `1.2x`，成功语义保持 official-equivalent；
- reviewer kind：`development_agent`，`paper_table_eligible=false`。

真实复用证据：

- `run_20260723_batch17_reviewed_scale_reuse_expert_v4`：exact registry match，TaskGen provider=0，
  setup/render/rule/expert 通过；
- `run_20260723_batch17_reviewed_scale_reuse_act_n1_v1`：production acceptance 通过后启动 ACT；
  expert seed `1000`，ACT 实际 seed `1001`，official success=true。

后一 run 中 `hammer_block_contact_ever=false`，与 official success 构成值得保留的 rule/policy
观察差异。它只有一个 ACT 样本，而且 expert/ACT seed 不同，不能写成严格 paired 结论。这里的
provider=0 只指 TaskGen reviewed reuse；全 Agent 的 Router、Planner、VQA 或 Feedback 仍可调用模型。

## 3. clean-head 路由与失败历史

原始 Query：

> How well does the click_bell ACT policy generalize across properties of the operated bell?

plan-only `eval_20260723_batch17_clean_head_click_plan_v2` 使用 task-only binding
`--bound-task-name click_bell`；Query 本身也点名 `click_bell`，但没有 history、bound aspect 或内部顺序提示：

- route supported：`object_position`、`object_instance`；
- 首个方面：`object_position`；
- partial gaps：color、gloss、texture、mass、scale；
- ACT starts=0。

失败尝试按原状态保留：

- `eval_20260723_batch17_clean_head_click_live_n1_v2`：首轮 ACT/Tool/VQA/Aggregate 完成，下一
  Proposal 先返回空 content，重试收到 HTTP 403，evaluation 未完成；
- v3：Luna vision HTTP 403，在 ACT 前以 pipeline failure 终止，ACT starts=0。

这些失败说明 key 本身不是唯一变量：同一 key 可以覆盖所有角色，但具体模型服务、网关和配额仍可能
瞬时不可用。不能把 v4 成功反写成 v2/v3 也已完成。

## 4. clean-head live v4

完成 evaluation：

`eval_20260723_batch17_clean_head_click_live_n1_v4`

所有模型角色显式覆盖为 `gpt-5.6-terra`。运行数据流：

| 阶段 | 结果 |
| --- | --- |
| Global route | supported=`object_position, object_instance`；gaps=`color, gloss, texture, mass, scale` |
| Round 1 proposal | 自主选择 `object_position.left_fixed`；`query_generated xy=[-0.14,-0.12]` |
| Round 1 ACT | seed=`100502`，success=`1` |
| Round 1 evidence | pipeline/Aggregate/Dynamic VQA passed，`evidence_conflict=false` |
| Adaptive PlanStep | provider action=`propose`，transition=`switch_aspect`，target=`object_instance.base0` |
| Round 2 ACT | 同 seed `100502`，success=`1` |
| Round 2 evidence | pipeline/Aggregate/Dynamic VQA passed，`evidence_conflict=false` |
| Terminal | hard cap=`2`；status/lifecycle=`completed` |

runtime ledger 精确为：

- logical provider calls：`11`；
- transport attempts：`16`；
- ACT starts：`2`；
- whole-round recovery/restart：`0`。

logical calls、transport attempts 和 ACT starts 是不同计量，不能合并。v4 只测了
`left_fixed` 与 `base0`；`right_fixed`、`base1` 没有执行。最终 completed 表示执行合同走完，不表示
宽 Query 已有充分证据。

已发布 compact evidence bundle：
[eval_20260723_batch17_clean_head_click_live_n1_v4](evidence_runs/eval_20260723_batch17_clean_head_click_live_n1_v4/)。
bundle 共 15 个文件，文件内容合计 600,465 bytes、目录占用约 628K；包含两轮短视频、scene、VQA
montage、代码和紧凑结构化数据。完整
machine audit 仍在服务器 evaluation 目录。compact bundle 改善人工核验，不替代原始 telemetry 或论文
规模统计。

本批开发资源核账不能只看成功 v4：新增 ACT starts 合计 `4`，其中 reviewed BBH reuse=`1`、clean-head
v2 首轮=`1`、v3=`0`、v4=`2`。clean-head 路径累计支付 `3` 个 ACT，左侧 position 在 v2/v4 重复；
因此“完成态 artifact 含 2 ACT”不等于开发总成本 2，也不产生 sampling savings 结论。

## 5. cached evidence branch

0-ACT cached replay v2 固定同一份真实 round-1 非 policy evidence，只修改
`policy_success`：

- `policy_success=0` → `drill_down position`；
- `policy_success=1` → `switch instance`。

该 replay 与 v4 共同证明 evidence 可以改变下一动作；它不产生新 rollout，也不能证明 stop 的充分性、
成功率或采样节省。

## 6. 当前能声称与不能声称

可以声称：

- 宽 Query 可在同一 task 内执行 supported subset，并显式保留 unsupported gaps；
- reviewed generated Task 可在 exact semantic/provenance 匹配下进入新 run，且 ACT 前重新 acceptance；
- 当前公共 `AdaptivePlanStepAgent` 已有一次完成态真实证据：position 成功后切到 instance；
- Rule/VQA/Aggregate/Planner 的主链在两个 N=1 round 中完成。

不能声称：

- 两个成功样本证明 click_bell 对位置、实例或“物体属性”广泛泛化；
- hard cap stop 等于 evidence sufficient；
- dynamic 比 fixed 更省样本；当前没有 matched efficiency 结果，旧 pilot savings=`0`；
- public Proposal Agent 已自然产生并让 ACT 消费新的 SuccessSpec v2；
- development-agent review 等于 human gold 或 paper eligibility；
- 全角色 Terra 成功证明 Luna、默认混合 profile 或外部网关稳定；
- v4 验证了最终源码新增的 failure/restart live 路径。

## 7. 下一批优先级

1. **Query-conditioned evidence sufficiency**：分开 `candidate_universe`、`required_coverage` 和
   `budget_cap`；让 `all/across/some/worst-case/compare` 等量词决定 stop contract，明确区分
   `evidence_sufficient`、`budget_exhausted`、`unsupported_gap`。先使用 v4/cached evidence，0 ACT。
2. **公共 Proposal → SuccessSpec v2**：让开放 Query 自然提出受控 scene + 新成功语义，通过正负
   fixture、oracle、render/expert 与 production acceptance；live 后置。
3. **adaptive vs fixed 科学证据**：同 Query、seed、checkpoint、最大预算和 sufficiency contract，
   先 N=1，再 N=3；同时报告结论一致性、样本数、墙钟和失败状态。
4. **独立人工 gold 与 VQA 鲁棒性**：真实扰动、正负平衡、多人复核。
5. **最后扩大 task、policy 与 repetition**。

## 8. 源码回归与覆盖审计

- 服务器最终全量：`552/552` tests passed；
- 0-ACT `batch17_method_coverage_v1`：`16/16` interface claims=`implemented`。

coverage audit 只检查源码接口和既有最小 evidence gate 是否存在；它不判断 Query sufficiency、方法语义、
统计一致性或论文结论。`16/16 implemented` 不能改写为“论文方法 100% 复现”。冻结报告与缓存反事实已发布到
[`evidence_runs/batch17_validation/`](evidence_runs/batch17_validation/)；fresh clone 缺少服务器侧被忽略的
`mea/evaluation_runs/` 与 `mea/validation_runs/` 原始产物，因此不能仅凭 clone 重新得到同一 `16/16`。
