# Plan Agent → Tool → Feedback 最短闭环

## 实现结果

本阶段把原先独立的 Offline ToolGen 接回外层评估编排，并进一步分离“指标意图”和“执行 route”。Plan Agent 只选择受限 sub-aspect template；系统注入场景指令、seed、gate 与无 route 的 `ToolRequest`。每轮 rollout 完成后，Runtime 根据 metric 精确匹配 Trusted catalog 或 composite target，再决定复用或生成 Tool。Tool observation 会在下一轮 planning 前写回，最终 Feedback Agent 只基于 evidence bundle 回答用户。

## 调用链

官方 RoboTwin ACT：

```text
policy/ACT/eval.sh
→ script/eval_policy.py
→ ACT checkpoint / deploy_policy
→ get_obs / get_action
→ Base_Task.take_action() / 250 Hz physics
→ task.check_success()
→ video / _result.txt
```

当前 MEA：

```text
自然语言 request
→ Plan Agent
   └─ 选择受限 sub-aspect template
→ Runtime 注入 scene round plan + route-free ToolRequest
→ Task / Documentation Retrieval
→ TaskGen / AST / render / Visual Self-Reflection
→ expert gate
→ eval_mea.sh → 官方 ACT 主干
→ Trajectory Recorder
→ Tool Router
   ├─ exact Trusted metric → reuse；零 GPT 调用
   ├─ exact composite target → force_codegen → differential/property gates
   └─ unknown/近似名称 → unsupported；零 GPT 调用
→ normalized tool_execution.json
→ round observation → 下一轮 Plan Agent
→ evidence_bundle.json
→ Feedback Agent
→ evaluation_report.md
```

MEA 没有改写 ACT 推理与 RoboTwin 物理执行主干；新增部分主要位于官方链之前的 proposal/generation，以及之后的 observation/tool/feedback。

## ToolRequest、Router 与 runtime 的职责边界

Plan Agent 实际只选择 template id。可信 template 再声明：

- task 与 metric；
- 问题语义；
- route-free metric 与问题语义。

Plan Agent 不能声明 Tool route、telemetry 路径、ACT/expert 的实际结果、生成源码路径或 hash。Router 使用 `strict_exact_metric_id`：metric 精确命中 `TOOL_CATALOG` 才复用，精确命中已登记 `COMPOSITE_TARGETS` 才生成；tags 和模糊相似度不能决定是否执行生成代码。Runtime 再从真实 artifact 解析完整内部 `ToolSpec` 与 episode，形成 `resolved_tool_spec.json`。

当前主要验证两个 metric：

- `hammer_block_contact_ever` 精确命中 Trusted catalog，自动 `reuse`，不调用 provider；
- `pickup_to_first_contact_time` 不存在于 Trusted catalog，只允许 `force_codegen`；它由 first-pickup 与 first-contact 两个 Trusted primitive 组成私有 oracle，要求 ACT `null` 与 expert numeric 正例；
- 新时间指标的 `pickup` 是 hammer Z 首次跨过 schema 的 `0.03 m` 阈值，不是最大高度，也不声称是首次稳定 grasp；
- ACT 标记为 `policy_under_evaluation`；
- expert 标记为 `expert_validation`，不能混入 ACT 表现。

## 当前 telemetry 是否足够

当前不是完整 simulator dump，而是 BeatBlockHammer 任务切片：

- 250 Hz：hammer/block 与 functional point、双臂 TCP、success、physics/policy step/time；
- policy boundary：action、双臂 qpos/qvel、EE/TCP、gripper、hammer/block pose/velocity；
- sparse events：strict contact、first contact、impulse、separation、success/error；
- episode/schema metadata 与 MP4。

它足够判断 pickup、接近/对齐、contact、首次接触、impulse、路径、官方成功和成功耗时。若要判断 250 Hz 关节振荡、全场 actor 碰撞、视觉遮挡、controller/planner/ACT 内部行为或其他任务语义，必须先扩展 Recorder/TaskSchema。第一版不追求“把整个 simulator 全存下来”，而是按指标补充稳定 signal。

## 统一产物

每轮 Tool 执行写入：

```text
execution/<round_id>/planned_tool/
├── tool_request.json
├── catalog_snapshot.json
├── route_decision.json
├── tool_spec.json
├── resolved_tool_spec.json
├── tool_execution.json
└── generated/                 # 仅 force_codegen
    ├── generated_tool.py
    ├── registration.json
    ├── execution_results.json
    ├── manifest.json
    └── attempts/attempt_*/
```

`route_decision.json` 记录 requested=`auto`、resolved route、精确命中的 registry、catalog hash、是否需要/实际调用 provider，以及失败原因。未知 metric 或 telemetry preflight 失败时也保留 Router 审计产物。`tool_execution.json` 对两条 route 使用同一 envelope，包含 source scope、ACT/expert role、value、evidence steps、validation gates 与相对 artifact path。

## 最多三轮的受限自适应规划

当前只开放三个可信 template：

1. `object_appearance.color_blue`：蓝色方块，1 episode；
2. `object_position.official_random`：官方位置/朝向采样，2 episodes；
3. `performance.pickup_to_contact_timing`：pickup-to-contact 时间，1 episode。

初始 GPT 只输出用户请求的 template ids 与首个 id；完整 round 由系统 materialize。每轮后 GPT 只输出 `continue/stop` 与下一个 template id，不能改 seed、gate、TaskGen route 或 Tool route。Runtime 使用完整 `observation_history`，禁止重复、越权和超过三轮；pipeline 失败、预算耗尽或没有剩余请求时强制停止，pipeline 通过且仍有用户明确请求的 template 时必须继续。每次决策写入 `decision_after_round_N.*`。

TaskGen 子进程的 exit code 也属于 pipeline evidence：非零退出时不会继续运行 Tool，round 会记录 skipped reason 且 `pipeline_passed=false`。外层 manifest 用 `lifecycle_status=completed` 表示编排已经结束，用 `completed_with_pipeline_failure` 区分评估失败，避免把“流程结束”误写成“评估通过”。

## 正式验证

为避免重复耗时 rollout，本次复用已经完成的蓝色方块 seed 100000 telemetry：

```text
mea/generated_tasks/run_20260715_telemetry_blue_seed100000/
```

闭环验证目录：

```text
mea/evaluation_runs/eval_20260715_plan_tool_closed_loop_v2/
```

结果：

- Live Plan Agent 输出 Round 1 `force_codegen` ToolSpec。
- Live ToolGen 第 0 次 attempt 即通过。
- ACT：contact=`false`，无 evidence step。
- expert：contact=`true`，first physical-contact physics step=`1454`。
- 两条轨迹的 deterministic、oracle agreement、artifacts unchanged 全部通过。
- Tool observation 返回后，Live Plan Agent 为下一轮输出 `reuse` ToolSpec。
- `reuse` route 的 `provider_called=false`，值与 Trusted Tool 一致。
- Live Feedback 明确区分 ACT 失败与 expert 对照成功，并指出只执行了一个正式 episode，尚未执行 Round 2 位置评估。

## 新指标正式验证

`eval_20260715_new_tool_duration_v5` 复用同一组 ACT/expert telemetry，完成了真正新指标的 live Plan → ToolGen → Feedback：

- Plan schema version 为 4，Round 1 ToolSpec metric 是 `pickup_to_first_contact_time`，`reference_tool=null`；
- Retriever 只给 GPT `first_hammer_pickup_step`、`first_contact_step` 和 `time_to_success` 三个基础示例，没有提供目标函数答案；
- ACT 在 physics step `6284` 首次达到 pickup 阈值，但没有 strict contact，所以 `value=null`，reason 为 `contact_not_observed_after_pickup`；
- expert 在 step `1039` pickup、step `1454` first contact，相隔 415 physics steps / `1.66 s`；
- 生成代码第 0 次即通过，两个真实 episode 的 deterministic、composition-oracle agreement 与 artifacts unchanged 全部为 true；
- 另外构造两个不改磁盘轨迹的 counterfactual property scenarios：`pickup_not_observed` 和 `contact_precedes_pickup`；生成代码对两者也 deterministic 且与 oracle 完全一致；
- v1/v2/v4 失败 evaluation id 均保留，分别暴露 reason enum、schema key 和 `.append()` AST 约束文档不明确；修正 prompt contract 后 v5 通过。

## Auto Router 与三轮状态机 live 验证

本批控制面验证目录为：

```text
mea/evaluation_runs/eval_20260715_auto_router_adaptive_live_v2/
```

用户请求同时包含蓝色外观、官方位置变化和 pickup-to-contact 时间。Live Plan
Agent 依次选择：

```text
object_appearance.color_blue
→ object_position.official_random
→ performance.pickup_to_contact_timing
→ stopped_after_round_3
```

Router 的两个真实分支均通过：

- `hammer_block_contact_ever`：requested=`auto`，resolved=`reuse`，provider
  未调用；ACT 没有 strict contact，expert 在 physics step 1454 有 contact；
- `pickup_to_first_contact_time`：requested=`auto`，resolved=`force_codegen`，
  provider 被调用；ACT 在 step 6284 达到 pickup 阈值但之后没有 strict
  contact，所以 duration 合法地为 `null`；expert 从 step 1039 到 1454，
  duration=`1.66 s`。

该验证明确复用既有真实 ACT/expert telemetry，没有重新执行三轮 rollout；它验证
的是 Plan Agent 状态转换、Router、Tool execution 与 observation 回流，不应描述
为一次新的“三轮 ACT E2E 评估”。完整 Runner 中 round 的 `route` 表示 TaskGen
场景路由，与 Tool Router 的 resolved route 是两个独立概念。

## 尚未解决的 gap

自动 Router 和三轮状态机已经完成，但 composite target 仍需人工登记可靠 oracle，不代表任意未知 metric 都能安全生成。下一批优先补多 episode Aggregate、执行期 VQA、同一 evaluation 内 run-local Tool 复用与 promote；随后扩展第二个 RoboTwin task。Recorder 的 `balanced_v1` 作为独立批次实现，避免把 Tool/Planner 回归与采样格式迁移混在一起。
