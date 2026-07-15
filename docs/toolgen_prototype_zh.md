# Offline ToolGen 原型

## 1. 这次实现解决什么问题

论文把 rule-based Tool 表示为 `trajectory → result`，同时也允许 Tool 直接监控 simulator。本原型选择一种解耦实现：先把所需 simulator signals 记录成 `TrajectoryView`，再做 post-rollout offline evaluation。因此，同一条 ACT rollout 可以被不同 Tool 反复分析，不需要每生成一个指标就重新运行机器人策略；这不是论文限定的唯一执行方式。

本原型只验证最短的一条链：

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
```

因为系统已经存在 `hammer_block_contact_ever`，正常的 retrieval-first 路径会直接复用它。本原型显式采用 force-codegen 思路生成一个重复 Tool，目的不是创造新指标，而是先证明 ToolGen plumbing 可以工作。

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
→ Plan Agent 与 round orchestration
→ Task / Documentation Retrieval
→ TaskGen 生成完整 load_actors()
→ AST / render / Visual Self-Reflection
→ expert gate
→ eval_mea.sh 进入原始 ACT 主干
→ Trajectory Recorder
→ Trusted Tool Retriever / execution
→ Feedback Agent / evaluation_report.md
```

本次 ToolGen 仍是独立验证支路，尚未自动接入 `manipeval_agent.py`：

```text
已有 telemetry → scripts/manipeval_toolgen.py → run-local registration
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
11. 每条输入 trajectory 都必须与对应 Trusted Tool oracle 一致；本次实跑恰好使用 ACT 负例与 expert 正例。
12. 失败时把诊断加入下一次 prompt；默认共尝试 2 次，即最多重新生成 1 次，CLI 上限为 3 次。

第一版采用严格 attribute-chain allowlist，并在独立 Python subprocess 中执行经过 AST gate 的代码；显式 `for/while`、递归和动态执行均被禁止，超时会终止 worker。它仍不是 OS/container 级安全沙箱，且 `TrajectoryView` 的内存对象不是真正 immutable；后续允许更自由的新 Tool 前，应继续加入 container 资源限制和只读 facade。

## 5. 使用方式

```bash
export UIUI_API_KEY='...'

python scripts/manipeval_toolgen.py \
  --request '生成一个工具，判断锤子是否真正接触过蓝色方块' \
  --reference-tool hammer_block_contact_ever \
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

下一步优先级建议：

1. 正常模式先复用已有 Tool，只有 library 缺失时才进入 ToolGen。
2. 让 Plan Agent 输出结构化 `ToolSpec`，并把 generated result 自动送回 Feedback Agent。
3. 当前由人显式指定 `--reference-tool`；下一步应让 Plan Agent 提出未知指标的结构化 `ToolSpec`。
4. run-local `registration.json` 尚未自动 promote 到永久 Toolkit，也未回接 Plan/Feedback。
5. 增加真正 immutable Trajectory facade 以及 container/CPU/memory 资源限制。
6. 扩展 TaskSchema/Recorder 到第二个 RoboTwin task，验证不是 hammer-specific demo。
7. 再做视觉型 Tool/VQA Tool 与数值 Tool 的联合证据。

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
