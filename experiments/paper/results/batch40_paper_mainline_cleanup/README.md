# Batch40：论文主链语义清理与充分停止正例

本目录是冷开发证据，不替换 `docs/evidence/current/`，也不据此声称跨任务、跨 seed
或统计泛化。Batch40 先断开一项 compat-only 生产运输，修正 evidence、停止所有权与
preservation 语义，再用同一条生产 Plan Agent 主链做纵向回归。

## 本批方法变化

- evidence writer 明确区分原始 pipeline 是否执行完成与 evidence 是否可供 Plan 使用；
  typed `N=0` planning failure 不再伪装成 pipeline failure。
- control 失败和充分停止都交给 Plan Agent 作出 `continue/stop` 决定；QueryContract 只验证
  决定是否被当前 evidence 支持。
- preservation 从词项重叠改为结构化事实：actor、property、axis 与 relation；无法解析的
  prose 保持 `unverified`，不猜测通过。
- Proposal 与 intended scene 的一致性按结构化 owner 判断，不再用 token overlap 将直接
  实验误标成 diagnostic proxy。
- ordinary production CLI 不再传递 reviewed Task registry；该 registry 只留在冷的
  standalone TaskGen 兼容路径。reviewed Tool/VQA 复用仍属于有效论文能力，未被误删。
- flagship acceptance 从生产模块迁入 paper layer；普通 finalize 不再执行论文批次专属
  验收总表。

没有新增任务专属 Planner、TaskGen 方言、registry、恢复状态机或 rollout gate。

## AutoDL live v1–v4

| run | policy episode | 结果与由此产生的最小修正 |
| --- | ---: | --- |
| v1 | 1 | unchanged official control 真实执行但 policy 失败；旧 precheck 在 Plan Agent 看到该 evidence 前直接终止。修正为失败 control 也到达 Agent，由 Agent 决定停止。 |
| v2 | 0 | Query 只引用 `official_check_success`，旧正则却误判为用户定义了 custom success semantics，因而在 rollout 前拒绝。收窄为只有明确说明 `check_success means/requires/iff` 才触发 custom-checker gate。 |
| v3 | 1 | official control 成功，Plan 选择 `x=+0.15 m` scene concern；词法 intent gate 将 Proposal 与同一 intended scene 误标为 diagnostic proxy，generated rollout 未启动。改成结构化 scene/checker/Tool owner 对齐。 |
| v4 | 2 | 完整退出：official control 与一个 generated scene episode 均成功；TaskGen 的 scene materialization、official checker 复用、live finite Rule Tool、Aggregate、Plan stop、QueryContract 验证和最终 Answer 在同一个 broad Query 中合一。原 live prompt 暴露了预计算的 sufficiency verdict；当前 prompt 的独立停止所有权由下述冻结回放补验。 |

v4 的 evaluation id 是
`eval_20260813_batch40_grab_roller_evidence_sufficient_s1000_v4`，使用 SmolVLA、scene seed
`1000`。Query 没有指定 aspect、object、axis、template、scene edit、checker、metric 或停止
脚本；它只要求寻找一个同时满足 official success 与 finite live Rule Tool measurement 的
generated witness，并允许 evidence 决定候选。

v4 中，official control 只作基线而不充当 witness。Plan 根据 control 与检索到的任务卡选择
其中已有正向证据锚点的 `grab_roller` x scene concern，再做一次 fresh generation 与验证；
用户 Query 没有给菜单，但精确的 `x=+0.15 m` 来自任务卡，而不是本次 control 冷发现。
v4 simulator probe 实测 roller `x=0.15001997 m`；intent transport 标为 `direct`，
preservation 是结构化且完整的。通用 TaskGen materialize 所需 scene，复用 official checker，
真实 SmolVLA episode 成功；provider Python Tool
`query_terminal_roller_z_position` 经独立 oracle 验证后，从同一 episode 返回
`0.8000384569168091 m` 并进入 Aggregate。该 aggregate 使用 evidence schema v2，
`pipeline.passed=true`、`valid_for_planning=true`、intent preservation complete 且
`sufficient=true`。Plan 的原始响应给出 `action=stop`、
`stop_reason=evidence_sufficient`、`claim_verdict=supported` 与
`answered_query=true`；QueryContract 验证通过，最终 Answer 对原 Query 给出受已测试范围
约束的肯定回答。TaskGen 只生成一次，2/2 fixtures、VLM 与 expert 均通过。

冻结 v4 artifact 随后用当前代码做了 0-rollout 真值重算：final Aggregate 的 3 条 metric
result 来自 2 个物理 policy episodes；短路径与仓库相对 episode 路径现会归一为同一 identity，
`unique_episode_count=2`。compact decision 也会从当前 `query_assessment` 正确投影
`evidence_sufficient/supported/evidence_sufficient`，不再留下空字段。

原 live v4 的 Plan prompt 曾包含 QueryContract 预计算的 sufficiency verdict，因此不能单独用来
证明 Agent 独立判断停止。当前代码已把调用合同和 prompt 都缩成仅含预算与运行限制；随后对
同一冻结两轮 evidence 做了一次 **provider-only、0 simulator、0 policy rollout** 回放。prompt
未包含 `should_stop/evidence_sufficient/stop_reason/claim_verdict` 字段，Plan Agent 一次调用即
原始输出 `action=stop`，后置 QueryContract 再验证为
`query_contract_validated_stop/supported/answered=true`。该回放没有改写 v4 live artifact。

最终代码在 AutoDL server 通过 226 项定向主干测试（另 13 个 subtests）和 287 项默认
mainline（另 12 个 subtests）。测试日志与冻结回放位于
`/root/autodl-tmp/mea-run-logs/batch40_final_validation/`；所有 Windows 操作仅限源代码、文档与
diff，没有运行测试、导入、provider、simulator 或 policy。

## 结论边界

Batch40 关闭的是一个具体方法缺口：在一个不提供实验菜单的 Query 内，成功执行的 evidence
能进入 Agent，并由 Agent 主动提出一个有充分证据支持的 stop，验证器只负责核验而不替代
推理。它同时证明 TaskGen、真实 rollout、live Rule Tool、Aggregate 和 Answer 可以在这条
正例中合一。

候选选择使用了任务卡中既有的同 seed 正向锚点，因此它证明的是检索知识指导的 fresh
generation/revalidation 与当前 evidence 的充分停止，不是从 control 独立冷发现一个从未见过的
concern。它也不证明 50-task 泛化、不同 seed 稳定性、Tool 跨 evaluation exact reuse、VQA 鲁棒性、
少样本效率或 policy ranking；也不替换 Batch37 的较宽多轮当前证据包。v4 是单任务、单 seed、
两个 policy episode 的 existential 正例。

此外，v4 的具体 x 改变有 simulator 数值证据，但当前通用 matcher 只验证 scene owner 与
可观测变化，还不能证明任意自然语言 Proposal 的 actor/axis/value 与生成 artifact 精确蕴含。

机器可读摘要见 [`summary.json`](summary.json)。raw evaluation 位于 AutoDL server：

```text
/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime/mea/evaluation_runs/
eval_20260813_batch40_grab_roller_evidence_sufficient_s1000_v4
```

v1–v4 的运行日志位于 `/root/autodl-tmp/mea-run-logs/` 对应
`batch40_grab_roller_evidence_sufficient_v*` 目录；provider credential 未写入仓库。
