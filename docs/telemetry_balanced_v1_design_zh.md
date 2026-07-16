# 跨任务 Telemetry 与 `balanced_v1` 设计

## 1. 当前 Recorder 的真实边界

当前数据不是完整 simulator dump，而是面向 `beat_block_hammer` 的任务语义切片：

| 频率 | 产物 | 已记录内容 |
| --- | --- | --- |
| 250 Hz physics step | `semantic_trace.npz` | hammer/block position、两个 functional point、双臂 TCP、success、physics/policy step 与 simulation time |
| 250 Hz event monitor | `events.jsonl` | contact interval、first physical contact、peak impulse、minimum separation、success/error |
| policy boundary，约 10 Hz | `states.csv` | action、双臂 qpos/qvel、EE/TCP、gripper、TaskSchema tracked actor 的 pose/velocity/functional pose |
| 10 FPS | `video.mp4` | 压缩 RGB 视觉证据 |
| episode | `episode.json`、`schema.json` | seed、policy/expert、结果、行数、耗时与 TaskSchema 快照 |

这些信号足够分析当前任务的 pickup、接近与对齐、hammer-block contact、first
contact、impulse、TCP 路径、官方成功和成功耗时。它们不能可靠回答：

- 250 Hz joint oscillation、jerk、瞬时 qvel 峰值、torque 或 controller tracking
  error；
- actor 的高频 quaternion、linear/angular velocity、撞击速度和撞击角度；
- hammer/block 之外的全场 actor 运动、碰撞与障碍物距离；
- `setup_demo()` / stabilization 阶段的初始碰撞、落体或穿模；
- depth、segmentation、point cloud、精确遮挡和 policy 视觉输入异常；
- planner/IK 内部状态，以及 ACT hidden state、action chunk 和 temporal
  aggregation uncertainty。

`states.csv` 的 actor 列已经由 `TaskSchema.tracked_actors` 驱动，但当前
`_semantic_state()`、`semantic_trace.npz` 的列和 `TrajectoryView` 仍直接使用
hammer/block 名称。因此它还不是可直接迁移到任意 RoboTwin task 的通用格式。

## 2. 跨任务通用设计

下一版应把“怎样采样”与“任务语义是什么”分开：

- `TelemetryProfile`：可信、allowlisted 的采样频率、field group、dtype 与
  artifact 规则；
- `TaskSchema`：tracked actor、task attribute、scene name、functional/contact
  point、success contract、threshold 与语义 alias；
- `SignalCatalog`：稳定的跨任务 signal id、shape、dtype、unit 和可用 stream；
- Tool contract：只声明 `required_signals`，不依赖某个 NPZ 的私有列名。

推荐稳定 signal namespace：

```text
core.physics_step
core.policy_step
core.simulation_time_seconds
task.success
robot.left.qpos / qvel / ee_pose / tcp_pose / gripper
robot.right.qpos / qvel / ee_pose / tcp_pose / gripper
actor.<actor_id>.pose
actor.<actor_id>.linear_velocity
actor.<actor_id>.angular_velocity
actor.<actor_id>.functional.<point_id>.pose
contact.<focus_id>.interval
```

NPZ 内应使用固定列顺序的 typed arrays，并在同目录写字段表，避免为每个 sample
构造大型 Python dict。TaskSchema 可以把任务专用名称映射到稳定 signal，例如：

```json
{
  "semantic_aliases": {
    "manipulated_object": "actor.block",
    "tool_object": "actor.hammer",
    "success_target_point": "actor.block.functional.1"
  }
}
```

这允许通用 Recorder 只理解 actor/robot/contact，Task-specific Tool 再通过 schema
理解“hammer”“block”或其他任务角色。旧 Tool 可继续从 legacy facade 读取原字段；
新 Tool 应优先按 signal id 读取。

## 3. `balanced_v1` 多频率 profile

`balanced_v1` 是设计目标，当前 runtime 尚未实现。推荐按信号用途分频：

- 250 Hz semantic stream：TaskSchema tracked actor 的关键 position、functional
  point、TCP、success 与时间索引；
- 250 Hz event monitor：contact begin/end、first physical contact、peak impulse、
  minimum separation 与 success/error；
- 50 Hz dynamics stream：完整双臂 qpos/qvel、EE/TCP、gripper，以及 selected
  actor 的 position、quaternion、linear/angular velocity 和 functional pose；
- policy boundary：action 与完整 state snapshot；
- 10 FPS：继续使用现有 H264 RGB；
- event keyframes：initial、pickup、contact、success/failure 的
  RGB/depth/segmentation，作为后续独立功能，不进入第一版。

建议 profile：

```json
{
  "schema_version": 1,
  "profile_id": "balanced_v1",
  "preserve_legacy_artifacts": true,
  "force_initial_sample": true,
  "force_final_sample": true,
  "streams": {
    "policy_state": {
      "sampling": "policy_boundary",
      "field_groups": ["legacy_full_state"],
      "artifact": "states.csv"
    },
    "semantic_trace": {
      "sampling": "physics_period",
      "every_physics_steps": 1,
      "field_groups": ["legacy_semantic"],
      "artifact": "semantic_trace.npz"
    },
    "dynamics_trace": {
      "sampling": "physics_period",
      "every_physics_steps": 5,
      "field_groups": [
        "robot_joint_state",
        "robot_end_effector_state",
        "tracked_actor_rigid_state",
        "tracked_actor_functional_pose"
      ],
      "artifact": "dynamics_trace.npz",
      "float_dtype": "float32"
    },
    "contact_events": {
      "sampling": "physics_period",
      "every_physics_steps": 1,
      "mode": "interval_summary",
      "scope": "task_schema_contact_focus",
      "artifact": "events.jsonl"
    }
  }
}
```

50 Hz 是对可分析性和体积的折中：足以观察大多数机器人/刚体动态趋势，同时
短暂 contact 仍由独立的 250 Hz event monitor 捕获。若未来某个 metric 确实要求
250 Hz joint/dynamics，应新增显式高频 profile，而不是把默认 profile 无限制扩大。

## 4. 空间估算

既有蓝色方块 ACT episode 的实测 telemetry 约为 2.75 MB：其中
`states.csv` 约 1.9 MB、`semantic_trace.npz` 约 0.46 MB、H264 video 约
0.27 MB，其余 metadata/events 很小。CSV 改为 typed compressed array 可以进一步
降低体积，但第一版为兼容已有 Tool 暂不删除它。

当前估算边界：

- `balanced_v1` selected-actor：目标约 2–5 MB/episode；
- 50 Hz 覆盖全场 actor：约 5–20 MB/episode，需实现后实测；
- 250 Hz、约 50 actor 的完整数值状态：约 53 MB raw、10–30 MB compressed；
- 三相机 RGB/depth/segmentation 全部以 250 Hz 原始保存：约
  37.6 GB/episode；再保存 XYZ point cloud 约增加 41 GB。

最后两项不适合作为默认模式。应继续把 RGB 保存为视频，把 depth/segmentation
限制为事件关键帧，并按 Tool 的真实 signal contract 扩展数值流。

## 5. 向后兼容迁移

第一版采用 additive migration：

- 原样保留 `states.csv`、`semantic_trace.npz` 和 `events.jsonl`；
- 新增 `dynamics_trace.npz` 与字段表，不改变 legacy semantic stream 的采样率；
- 每个 episode 保存 `telemetry_profile.json` 快照；
- `episode.json` 新增 profile id、配置 hash、每个 stream 的周期、行数、dtype
  和 artifact 路径；
- initial 与 final 必须强制采样，即使 final step 不是 5 的倍数；
- contact monitor 始终逐 physics step 运行，不能跟随 dynamics 降采样；
- 旧 `TrajectoryView` 忽略未知 artifact 后仍可工作；新 `TrajectoryView v2`
  按 `required_signals` 惰性读取相应 stream；
- schema/profile hash 进入 Tool registration compatibility，避免错误复用。

当前 Recorder 在 `setup_demo()` 完成后才 attach，现有 RGB 也由
`Base_Task.take_action()` 写入。因此 setup/stabilization telemetry 和事件关键帧需要
单独修改生命周期，不能假装通过 profile 配置已经覆盖。

## 6. 后续实现与验收

建议把 Recorder v2 作为独立批次：先完成 profile validator、通用 signal extractor、
`dynamics_trace.npz` writer 与 `TrajectoryView v2`，再接第二个 RoboTwin task。

最低验收：

1. legacy 与 balanced profile 的全部现有 Trusted Tool 结果一致；
2. dynamics sample steps 精确满足 `0, 5, 10, ..., final`；
3. 短暂 contact 不因 50 Hz dynamics 而丢失；
4. early success、异常退出和不足 5 个 physics steps 的 episode 仍写 final sample；
5. ACT 与 expert 各跑 1 episode，比较 policy behavior、video hash 或关键结果，确认
   Recorder 没有改变 rollout；
6. 实测 selected-actor 与全场 profile 的 storage、内存和 wall-time overhead；
7. 用第二个 task 验证 Recorder core 无需复制，只有 TaskSchema/adapter 发生变化；
8. Tool preflight 能明确报告缺失 signal，而不是在运行中出现模糊 KeyError。

在这些验收通过前，`balanced_v1` 只能称为设计，不应写成已经部署的 runtime
行为。
