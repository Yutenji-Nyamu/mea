# Offline ToolGen 原型

## 1. 这次实现解决什么问题

论文把 rule-based Tool 表示为 `trajectory → result`，同时也允许 Tool 直接监控 simulator。本原型选择一种解耦实现：先把所需 simulator signals 记录成 `TrajectoryView`，再做 post-rollout offline evaluation。因此，同一条 ACT rollout 可以被不同 Tool 反复分析，不需要每生成一个指标就重新运行机器人策略；这不是论文限定的唯一执行方式。

原型先用已有 contact Tool 验证 plumbing，当前又补上真正新指标的最短链：

```text
“判断锤子是否真正接触方块”
→ 检索 3 个已验证 contact Tool 示例
→ UIUI GPT 生成完整 generated_tool(trajectory)
→ AST 与返回契约检查
→ 同一输入重复执行，检查确定性
→ 检查轨迹文件未被修改
→ 在 ACT 负例和 expert 正例上执行
→ 与 Trusted Tool oracle 做 differential equality
→ 保存源码、prompt、response、hash 和结果

“从首次抬升达到阈值到首次严格接触经过多久”
→ 检索 first-pickup / first-contact / nullable-time 三个基础示例
→ UIUI GPT 生成目标函数；目标函数本身不在 Trusted catalog
→ ACT null 与 expert numeric trajectory 做 differential gate
→ 与两个 Trusted primitive 的私有 composition oracle 逐字段比较
```

已有 `hammer_block_contact_ever` 的 force-codegen 回归仍保留；新 `pickup_to_first_contact_time` 不进入 Trusted catalog，因此不是重复生成已有函数。它的 composition oracle 只参与 validation，不会作为检索答案交给 GPT。

### 官方 RoboTwin ACT 调用链

```text
policy/ACT/eval.sh
→ script/eval_policy.py
→ deploy_policy.get_model() / ACT checkpoint
→ expert seed gate
→ Base_Task.get_obs()
→ deploy_policy.eval() / ACT.get_action()
→ Base_Task.take_action() 展开为 250 Hz physics steps
→ task.check_success()
→ video 与 _result.txt
```

### 当前 MEA 调用链

```text
自然语言 request
→ Plan Agent 输出 round plan 与 ToolSpec
→ Task / Documentation Retrieval
→ TaskGen 生成完整 load_actors()
→ AST / render / Visual Self-Reflection
→ expert gate
→ eval_mea.sh 进入原始 ACT 主干
→ Trajectory Recorder
→ Trusted Tool Retriever / execution
→ Tool Router（reuse 或 force_codegen）
→ Feedback Agent / evaluation_report.md
```

ToolGen 已接入 `manipeval_agent.py` 的每轮执行边界：

```text
本轮 TaskGen / ACT 完成 telemetry
→ 解析 Plan Agent 的 ToolSpec
→ 自动发现 ACT 与 expert trajectory
→ reuse：直接运行 Trusted Tool，不调用 GPT
  或 force_codegen：生成、差分验证并 run-local registration
→ normalized tool_execution.json
→ round observation
→ 下一轮 Plan Agent 与最终 Feedback Agent
```

MEA 没有替换 ACT 推理和 RoboTwin 物理执行主干；主要是在官方链前增加 request 理解、检索、场景生成与验证，在链后增加 trajectory、Tool 与证据报告。

## 2. 当前是否记录了完整 RoboTwin 数据

没有。准确表述是：

> Recorder 挂载后的正式 expert/ACT 动作主循环中，每个已接入的 `scene.step()` 都有 BeatBlockHammer 关键语义记录；但状态维度只是任务相关切片。

也就是“时间轴基本完整，状态维度不完整”。Recorder 在正式 expert/ACT 动作执行阶段挂载，不包含 `setup_demo()` 的场景创建与稳定阶段。

### `semantic_trace.npz`：250 Hz physics step

Recorder 挂载后，每个已接入的 physics step 保存：

- `physics_step`、`policy_step`、`simulation_time_seconds`
- `success`
- `hammer_position`、`block_position`
- `hammer_functional_position`、`block_functional_position`
- `left_tcp_position`、`right_tcp_position`

这些数组适合计算距离、位移、路径、对齐误差和 time-to-event。

### `states.csv`：policy boundary

initial、每个 policy action 结束和 final 保存；expert 的整个 `expert_plan` 当前作为一个宏观 policy boundary，而不是每个 planner primitive 各存一行：

- 14 维 action 与 action type
- policy/physics step、simulation/wall time、video frame
- 左右臂 qpos/qvel、EE/TCP pose、gripper
- hammer/block pose、linear/angular velocity
- schema 声明的 functional-point pose
- success

它适合检查 action、关节、gripper 和 policy-level 状态，但不是 250 Hz 全量机器人状态。

### `events.jsonl`：稀疏事件

保存：

- 涉及 hammer 或 block 的 contact interval
- reported contact 与 strict physical contact
- interval start/end
- first physical contact
- maximum impulse、minimum separation
- peak contact position/normal
- success transition 与 error

`reported_contact=true` 只代表 SAPIEN 返回了 contact pair。当前严格物理接触还要求 contact point 的 impulse norm `> 1e-8` 或 separation `<= 0`。

### `episode.json` 与 `schema.json`

`episode.json` 保存 task、policy、seed、success、步数、耗时和 artifact 映射；`schema.json` 保存 actor 身份、functional point、阈值、physics timestep 等任务语义。

### 当前不存在的数据

- 每个 physics step 的双臂完整 qpos/qvel、gripper 与 orientation
- table、robot links、wall、clutter 等全场 scene actor/link 状态
- controller、planner、ACT hidden state 或 logits
- 全部逐点 contact 原始序列和摩擦信息
- 每步 RGB/depth/segmentation/pointcloud；RGB 仍单独保存为 MP4
- 其他 RoboTwin task 的通用语义字段

因此 GPT 只能生成“现有 trajectory 能表达的指标”。需要未记录数据的 request 应明确返回不可实现，或先扩展 TaskSchema/Recorder。

对于当前 BeatBlockHammer 垂直切片，现有数据足以可靠判断：hammer pickup、hammer/block 距离与对齐、strict physical contact、first contact、impulse、TCP path、official success 和 time-to-success。只有要分析 250 Hz 关节振荡、全场物体碰撞、视觉遮挡、控制器/规划器内部行为，或切换到尚未定义 TaskSchema 的其他任务时，才需要扩展记录维度。与其保存整个 simulator dump，更合理的策略是按新指标补充稳定 signal。

## 3. GPT 可使用的接口

生成函数固定为：

```python
def generated_tool(trajectory):
    return {
        "value": ...,
        "unit": ...,
        "passed": ...,
        "evidence_steps": [...],
        "details": {...},
    }
```

允许读取：

- `trajectory.trace`
- `trajectory.events`
- `trajectory.policy_states`
- `trajectory.schema`
- `trajectory.metadata`
- `trajectory.contact_intervals`
- `trajectory.success_events`
- `trajectory.hammer_block_contacts()`

worker 已注入 `np`，生成代码不需要也不允许 `import numpy`；可调用的 NumPy attribute chain 受纯数值 allowlist 限制。

框架而不是 GPT 负责添加 `tool`、`version`、`generated` 和 `tool_sha256`。这样模型不能自行声明来源或伪造 source hash。

## 4. 检索与验证

第一版不用 vector database。一个确定性 Retriever 从经过测试的 standalone examples 中选择最多 3 个源码片段。接触 request 会选择：

- `hammer_block_contact_ever`
- `first_contact_step`
- `max_contact_impulse`

这些 example 在进入 prompt 前，会先在所有输入 episode 上与对应 Trusted Tool 比较；只有一致的示例才交给 GPT。

新时间指标会选择：

- `first_hammer_pickup_step`
- `first_contact_step`
- `time_to_success`

第三项只演示 duration/null 输出习惯。GPT 看不到最终 composite 实现。

生成代码必须通过：

1. module 只能包含一个 `generated_tool(trajectory)`。
2. 无 import、文件、网络、process、environment 或 introspection。
3. 只能读取公开的 `TrajectoryView` 字段。
4. 返回值可以严格 JSON 序列化。
5. `evidence_steps` 必须存在于 physics trace。
6. 五个 core artifacts 必须齐全，`episode.error` 为空，metadata/schema task 一致。
7. 验证集路径不得重复；contact 原型必须同时包含 oracle=True 和 False。
8. 生成函数在独立 Python subprocess 中运行，并有 5 秒 timeout。
9. 同一 episode 执行两次结果完全一致。
10. 核心五个 trajectory artifact 执行前后 SHA256 不变。
11. 已有 metric 与单个 Trusted Tool 一致；新 metric 与私有 composition oracle 一致。时间指标必须同时有 `null` 负例和非负有限 numeric 正例。
12. 失败时把诊断加入下一次 prompt；默认共尝试 2 次，即最多重新生成 1 次，CLI 上限为 3 次。

第一版采用严格 attribute-chain allowlist，并在独立 Python subprocess 中执行经过 AST gate 的代码；显式 `for/while`、递归和动态执行均被禁止，超时会终止 worker。它仍不是 OS/container 级安全沙箱，且 `TrajectoryView` 的内存对象不是真正 immutable；后续允许更自由的新 Tool 前，应继续加入 container 资源限制和只读 facade。

## 5. 使用方式

```bash
export UIUI_API_KEY='...'

python scripts/manipeval_toolgen.py \
  --request '生成一个工具，计算锤子首次抬升到首次严格接触经过多久' \
  --target-metric pickup_to_first_contact_time \
  --trajectory mea/generated_tasks/run_20260715_telemetry_blue_seed100000/evaluation/telemetry/act/episode_000_seed_100000 \
  --trajectory mea/generated_tasks/run_20260715_telemetry_blue_seed100000/evaluation/telemetry/expert/episode_000_seed_100000 \
  --output-dir mea/generated_tasks/run_20260715_telemetry_blue_seed100000/toolgen/hammer_block_contact_v1
```

输出目录包含：

```text
request.json
retrieval.json
example_validation.json
manifest.json
generated_tool.py
execution_results.json
registration.json
attempts/attempt_*/
  prompt.md
  response.txt
  generated_tool.py
  validation.json
```

## 6. 与论文的对应及后续 gap

这条链对应论文 Figure 4 的 retrieval-first、相近 Tool few-shot、code generation 和 unit-test gate，也符合论文中 `rule-based tool: trajectory → result` 的定义。

已经补齐：

1. Plan Agent 输出严格 `ToolSpec`，只声明 metric、所需 signal、输出契约和 route，不声明 telemetry 路径或结果。
2. Runtime 解析 ACT/expert 角色与 trajectory；`force_codegen` 执行 differential gate，`reuse` 直接调用 Trusted Tool 且不调用 GPT。
3. 两条 route 统一生成 `tool_execution.json`，并在下一轮 planning 前进入 round observation。
4. Tool 证据进入 `evidence_bundle.json`、Feedback prompt 和 `evaluation_report.md`；ACT 是被评策略，expert 只作为验证对照。

后续优先级建议：

1. Tool Retriever 自动判断 exact reuse 或 unknown force-codegen，并记录路由理由。
2. 为更多未知指标加入 property tests、人工 reference 或组合式 Trusted Tool oracle。
3. run-local `registration.json` 经审核后可 promote 到永久 Toolkit。
4. 扩展 TaskSchema/Recorder 到第二个 RoboTwin task，验证不是 hammer-specific demo。
5. 再做视觉型 Tool/VQA Tool 与数值 Tool 的联合证据。

## 7. 正式实跑记录

实现提交 `f9cdd830a082251014e5b7948c3d3562a8b398d5` 后，在已有蓝色方块 run 上完成正式 live generation：

- evaluation run：`run_20260715_telemetry_blue_seed100000`
- ToolGen output：`toolgen/hammer_block_contact_v2`
- model：`gpt-4o-2024-11-20`
- prompt / completion / total tokens：1097 / 189 / 1286
- successful attempt：0；未发生 regeneration
- generator source SHA256：`4f296f3968096fde9a6cb01a28981c0d25d4b4f4487fe6ce3868ba54e1c4df37`
- prompt contract SHA256：`87068fb62d487a839a29d51672db57560ac130cbf834ef5be634f6d24b99e18e`
- generated Tool SHA256：`4683ba1229c8696f574aa3b5207bed323dc9cf1c2a75300c1174fc83b4feaada`
- ACT：`value=false`，无 physical-contact evidence step
- expert：`value=true`，first physical-contact evidence step 为 `1454`
- 两个 episode 均满足 deterministic、oracle agreement、artifacts unchanged
- 三个 retrieved standalone examples 在 ACT/expert 两条轨迹上共 6 次 oracle comparison 全部通过
- `registration.json` 为 `scope=run_local`、`status=validated`

独立 audit 重新计算了 implementation、contract、generated source 与 registration hash，并检查 manifest、正负例结果和全部 gate，所有检查均为 true。生成代码是完整 `generated_tool(trajectory)`，没有调用 Trusted Tool，也没有在执行时访问 simulator；它只读取 `trajectory.hammer_block_contacts()`。

服务器产物：

```text
/root/autodl-tmp/mea/mea/generated_tasks/run_20260715_telemetry_blue_seed100000/toolgen/hammer_block_contact_v2/
```

操作日志：

```text
/root/autodl-tmp/mea/_ops_logs/toolgen_static_20260715_115838.log
/root/autodl-tmp/mea/_ops_logs/toolgen_stage_commit_20260715_120059.log
/root/autodl-tmp/mea/_ops_logs/toolgen_live_20260715_120235.log
```

## 8. Plan → Tool → Feedback 正式闭环验证

闭环实现后，使用已有蓝色方块 ACT/expert telemetry 验证编排，不重新运行 ACT：

```text
用户 request
→ UIUI Plan Agent 输出 Round 1 force_codegen ToolSpec
→ Runtime 自动定位 seed 100000 的 ACT/expert telemetry
→ UIUI ToolGen 生成完整 generated_tool(trajectory)
→ deterministic / oracle / artifact gates
→ Tool observation 返回 Plan Agent
→ Plan Agent 为下一轮输出 reuse ToolSpec
→ reuse route 零 provider 调用
→ UIUI Feedback Agent 生成证据化结论
```

最终正式验证目录：

```text
/root/autodl-tmp/mea/mea/evaluation_runs/eval_20260715_plan_tool_closed_loop_v2/
```

关键结果：

- Plan schema version：3。
- Round 1 Tool route：`force_codegen`；successful attempt：0。
- 生成 Tool 在 ACT 上返回 `false`，在 expert 对照上返回 `true`。
- expert first physical-contact physics step：`1454`。
- 两条 trajectory 的 deterministic、oracle agreement、artifacts unchanged 全部为 true。
- 下一轮 Plan Agent 输出 `reuse`；该 route 的 `provider_called=false`，结果与 Trusted Tool 一致。
- `force_codegen` 才要求 false/true differential contrast；普通 `reuse` 不要求策略必须失败，因此 ACT 成功时不会被错误拒绝。
- 本次只复用了 Round 1 telemetry 来验证 Tool 编排；没有把它冒充 Round 2 位置评估。
- 完整的 prompt、response、ToolSpec、resolved episode、生成源码、验证结果、evidence bundle 与最终报告都保存在同一 evaluation id 下。

## 9. 真正新指标的 live generation

`eval_20260715_new_tool_duration_v5` 使用 Plan schema 4 和 `pickup_to_first_contact_time` ToolSpec。该 metric 没有同名 Trusted Tool，Retriever 只提供 first-pickup、first-contact 和 nullable-time 基础实现；runtime 用两个 primitive 的私有 composition oracle 验证生成结果。

真实轨迹结果：

- ACT：pickup step `6284` / `25.136 s`，没有 strict contact，`value=null`；
- expert：pickup step `1039` / `4.156 s`，contact step `1454` / `5.816 s`，`value=1.66 s`；
- 两条真实 trajectory 的 deterministic、oracle agreement、artifacts unchanged 全部通过；
- 两个内存 counterfactual `pickup_not_observed` 与 `contact_precedes_pickup` 也通过 deterministic/oracle gate，后者明确要求 `duration_physics_steps=null` 且 evidence steps 升序；
- successful attempt 为 `0`。

开发过程中保留了 v1/v2/v4 失败 evaluation：v1 暴露 reason enum 没有在 prompt 中精确列出；v2 暴露 `physics_timestep_seconds` 字段名没有明确说明；加入 property gate 后又发现 v3 对未见的 contact-before-pickup 边界不完整，v4 则因 `.append()` 被 AST gate 拒绝。补充 exact edge/AST contract 后 v5 完成 Plan → ToolGen → counterfactual properties → Feedback，并生成 `evaluation_report.md`。
