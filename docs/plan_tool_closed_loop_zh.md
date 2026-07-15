# Plan Agent → Tool → Feedback 最短闭环

## 实现结果

本阶段把原先独立的 Offline ToolGen 接回外层评估编排。Plan Agent 现在不仅规划场景和 episode，还输出严格 `ToolSpec`；每轮 rollout 完成后，runtime 自动选择 Trusted Tool 复用或受约束 ToolGen，并在下一轮 planning 之前把 Tool observation 写回。最终 Feedback Agent 只基于 evidence bundle 回答用户。

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
   ├─ scene round plan
   └─ ToolSpec
→ Task / Documentation Retrieval
→ TaskGen / AST / render / Visual Self-Reflection
→ expert gate
→ eval_mea.sh → 官方 ACT 主干
→ Trajectory Recorder
→ Tool Router
   ├─ reuse → Trusted Tool；零 GPT 调用
   └─ force_codegen → examples → GPT ToolGen → differential gates
→ normalized tool_execution.json
→ round observation → 下一轮 Plan Agent
→ evidence_bundle.json
→ Feedback Agent
→ evaluation_report.md
```

MEA 没有改写 ACT 推理与 RoboTwin 物理执行主干；新增部分主要位于官方链之前的 proposal/generation，以及之后的 observation/tool/feedback。

## ToolSpec 与 runtime 的职责边界

Plan Agent 只声明：

- task 与 metric；
- 问题语义；
- `reuse` 或 `force_codegen` route；
- 需要的稳定 signal；
- 输出和验证契约。

它不能声明 telemetry 路径、ACT/expert 的实际结果、生成源码路径或 hash。runtime 从真实 artifact 中解析这些内容，形成 `resolved_tool_spec.json`。

第一版只开放 `hammer_block_contact_ever`：

- `force_codegen` 必须同时找到 reference=false 与 true 的不同 trajectory，用于 differential gate；
- `reuse` 直接运行已测试 Trusted Tool，不要求 false/true 对照，也不调用 provider；
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

`tool_execution.json` 对两条 route 使用同一 envelope，包含 source scope、ACT/expert role、value、evidence steps、validation gates 与相对 artifact path。

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

## 尚未解决的 gap

当前 `force_codegen` 生成的是已有 Trusted Tool 的同义实现；它证明了 Proposal → Generation/Retrieval → Execution → Feedback 的编排链路，但还不等于能可靠验证任意未知 metric。下一步最小扩展应生成“从 hammer pickup 到 first physical contact 的耗时”，用已有两个 Trusted Tool 的组合结果构造 oracle，再考虑跨第二个 RoboTwin task 泛化。
