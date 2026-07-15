# 开发记录：Auto Tool Router 与有界自适应规划

日期：2026-07-15

## 1. 本批目标

本批把 Tool 的选择从 Plan Agent 的显式 `reuse` / `force_codegen` 决定，
收敛为由可信代码执行的自动路由，并把单轮验证扩展为最多三轮的 bounded
adaptive planning：

```text
User Query
→ bounded adaptive Plan Agent
→ route-free ToolRequest（只描述 metric / question）
→ Auto Tool Router
   ├─ 已有完全匹配 Tool：reuse
   └─ 没有完全匹配 Tool：force_codegen
→ Tool observation
→ 下一轮 Plan Agent 或停止
→ Feedback / evaluation report
```

目标不是让 GPT 自己决定是否执行任意代码，而是让 Plan Agent 描述需要测量的
语义，由 Router 根据可审计的 catalog、contract 与 telemetry signal 决定复用
或生成。

## 2. 实现内容

### 2.1 Auto Tool Router

Router 接收结构化 ToolSpec，并输出实际执行路由及其依据：

- `auto → reuse`：Toolkit 中已有满足当前 Tool contract 的可信 Tool；
- `auto → force_codegen`：没有完全匹配的已有 Tool，需要运行 ToolGen；
- Router 的选择、候选 Tool、provider 是否被调用以及最终结果进入 evaluation
  artifact，便于复查；
- reuse 路径不得调用生成模型；force_codegen 路径必须经过现有 AST、契约、
  determinism、oracle 与 artifact integrity 验证。

这使 Plan Agent 负责“要测什么”，可信 Router 负责“已有工具能否直接测”。

### 2.2 最多三轮的 bounded adaptive planner

规划器允许根据上一轮 observation 继续提出下一个 ToolRequest，但设置最多三轮的
硬边界。每轮只能在允许的任务、sub-aspect、telemetry signal 与 Tool route
协议内工作；达到证据目标、发生 pipeline failure、没有合法下一轮或达到轮数
上限时停止。

该设计对应论文中的：

```text
Proposal
→ Generation / Retrieval
→ Execution
→ Observation
→ 下一轮 Proposal
```

同时避免原型进入不受限的 agent 循环。

### 2.3 Tool observation 回流

Router 的实际路由、policy/expert 结果、有效性与证据 step 会作为结构化
observation 返回 planner，并进入最终 evidence bundle 与 Feedback。policy 与
expert 的角色保持分离：policy 是被评估对象，expert 仅用于任务与 Tool 的
可解性/正确性验证，不能混合计算结论。

## 3. 真实验证

本批 live evaluation id：

```text
eval_20260715_auto_router_adaptive_live_v2
```

本次验证复用了已经存在的真实 ACT 与 expert telemetry，没有重新运行 ACT 或
expert rollout。因此，下面验证的是 Proposal、Router、Tool execution、
observation 回流和 Feedback 链路，不把复用数据描述为一次新策略评估。

### 3.1 已有 contact Tool：`auto → reuse`

验证结果：

- Plan Agent 请求判断 hammer 与 block 是否发生物理接触；
- Auto Tool Router 发现已有匹配的可信 contact Tool；
- 实际路由：`reuse`；
- provider 调用：`false`；
- ACT policy：`false`，没有检测到目标物理接触；
- expert validation：`true`，检测到目标物理接触。

该结果证明已有 Tool 命中时 Router 能跳过 GPT 代码生成，并保持 policy 与
expert 证据分离。

首次 live run 暴露了一个 provenance 投影问题：数值正确，但 reuse 结果的
`tool` / `tool_sha256` 被 differential oracle 的精简投影剥离。修复后，exact
catalog reuse 直接执行 Trusted Tool；v2 已验证 policy/expert 的 Tool 名与
source SHA 均完整保留。原 run 继续保留为可审计的失败历史。

### 3.2 新 pickup-to-contact Tool：`auto → force_codegen`

验证结果：

- Plan Agent 请求测量 hammer 首次抬升到首次接触 block 的时间；
- Toolkit 中没有完全匹配的已有 Tool；
- 实际路由：`force_codegen`；
- provider 调用：`true`；
- ACT policy：结果为 `null`，原因是该 episode 没有目标 contact，时间差未定义；
- expert validation：`1.66 s`。

ACT 的 `null` 是符合 Tool contract 的语义结果，不是把缺失 contact 错误记成
零秒。expert 的有效数值为生成 Tool 提供了正例，并与已有 pickup/contact
证据共同用于验证。

### 3.3 两条路由的联合意义

这两个真实分支共同验证：

1. `auto` 并不固定等于 ToolGen；
2. 已有 Tool 可以无 provider 调用直接复用；
3. 缺少完全匹配的指标时可以进入受限 ToolGen；
4. 同一 Tool contract 能处理 policy 的合法空值与 expert 的有效数值；
5. Tool observation 可以成为下一轮规划与最终反馈的证据。

## 4. 数据与 Recorder 边界

本批没有修改 runtime Recorder，也没有改变现有 telemetry artifact 格式。
contact 与 pickup-to-contact 所需信号已经存在于：

- `semantic_trace.npz`：250 Hz hammer position、physics step、simulation time；
- `events.jsonl`：首次 strict physical contact 与相关 evidence step；
- `schema.json`：pickup height threshold 与任务语义契约；
- `states.csv`：policy boundary 的 action、机器人与目标 actor 状态。

本批只整理了 `balanced_v1` 多频率 Recorder 的文档设计：250 Hz 保留关键
semantic/contact，50 Hz 增加完整机器人与目标 actor dynamics，10 Hz 保持
action 与 H264 RGB，事件关键帧留作后续。设计目标 `2–5 MB/episode` 是估算，
不是本批 runtime 实测结果。

将 Auto Router 与 Recorder schema 迁移拆开，可以避免路由回归与数据格式
回归相互干扰。

## 5. 测试与检查

最终测试完成后补充以下数字：

- 自动化测试：`61` 个；
- Auto Tool Router contract / 路由测试：`6` 个；
- bounded adaptive planner 测试：`9` 个；
- ToolGen / Toolkit 回归测试：`17` 个；
- 远端全量测试结果：`Ran 61 tests ... OK`；
- `py_compile`、`git diff --check`、README 与官方 `eval.sh` 未改检查：通过。

已完成的真实验证事实以第 3 节为准：一个 `auto → reuse` 分支、一个
`auto → force_codegen` 分支，均使用既有真实 ACT/expert telemetry，未新增
rollout。

## 6. 后续建议

在该闭环稳定后，优先补充：

1. 跨 episode Aggregate Toolkit，让 planner 获得统计 observation；
2. execution VQA，为数值 Tool 补充关键帧视觉证据；
3. run-local generated Tool 的后续轮次复用与受控 promote；
4. 第二个 RoboTwin task 的 Documentation RAG、TaskSchema 与 Tool 迁移；
5. token、Vision call、rollout 与 wall-time budget 及停止规则。
