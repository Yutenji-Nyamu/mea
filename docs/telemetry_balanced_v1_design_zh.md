# 跨任务 Telemetry 与 `balanced_v1`

> 状态更新：`balanced_v1` 已实现为可选择的 runtime profile。本文件前半保留设计
> 动机，实际实现与验证结果以本文第 5、6 节及本批开发记录为准。

## 1. 当前 Recorder 的真实边界

当前数据不是完整 simulator dump，而是由 TaskSchema 声明的任务语义切片：

| 频率 | 产物 | 已记录内容 |
| --- | --- | --- |
| 250 Hz physics step | `semantic_trace.npz` | hammer/block position、两个 functional point、双臂 TCP、success、physics/policy step 与 simulation time |
| 250 Hz event monitor | `events.jsonl` | contact interval、first physical contact、peak impulse、minimum separation、success/error |
| 50 Hz physics step | `dynamics_trace.npz` | 双臂 qpos/qvel、EE/TCP、gripper，以及 selected actor pose、速度与 functional/contact pose |
| policy boundary，约 10 Hz | `states.csv` | action、双臂 qpos/qvel、EE/TCP、gripper、TaskSchema tracked actor 的 pose/velocity/functional pose |
| policy/ACT 约 10 FPS；official expert 事件帧 2 FPS | `video.mp4` | 连续或稀疏 H264 RGB 视觉证据，具体语义见 episode metadata |
| episode | `episode.json`、`schema.json` | seed、policy/expert、结果、行数、耗时与 TaskSchema 快照 |

这些信号足够分析当前 schema 所声明 actor 的 pickup、接近、对齐、first contact、
impulse、TCP 路径、官方成功和成功耗时。它们不能可靠回答：

- 250 Hz joint oscillation、jerk、瞬时 qvel 峰值、torque 或 controller tracking
  error；
- actor 的高频 quaternion、linear/angular velocity、撞击速度和撞击角度；
- hammer/block 之外的全场 actor 运动、碰撞与障碍物距离；
- `setup_demo()` / stabilization 阶段的初始碰撞、落体或穿模；
- depth、segmentation、point cloud、精确遮挡和 policy 视觉输入异常；
- planner/IK 内部状态，以及 ACT hidden state、action chunk 和 temporal
  aggregation uncertainty。

`states.csv`、`semantic_trace.npz`、`dynamics_trace.npz` 与 `TrajectoryView` 现在都按
episode 内的 TaskSchema 工作。BBH 的旧字段名仍作为兼容 contract 保留；当前 schema
覆盖 `beat_block_hammer`、`click_bell`、`adjust_bottle` 与 `grab_roller`。后三项分别
声明 bell、bottle、roller 的任务语义字段与左右 TCP，不复制 Recorder core。

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

`balanced_v1` 已按下列用途分频实现：

- 250 Hz semantic stream：TaskSchema tracked actor 的关键 position、functional
  point、TCP、success 与时间索引；
- 250 Hz event monitor：contact begin/end、first physical contact、peak impulse、
  minimum separation 与 success/error；
- 50 Hz dynamics stream：完整双臂 qpos/qvel、EE/TCP、gripper，以及 selected
  actor 的 position、quaternion、linear/angular velocity 和 functional pose；
- policy boundary：action 与完整 state snapshot；
- policy/ACT 约 10 FPS：继续使用现有连续 H264 RGB；
- official expert `event_keyframes_v1`：捕获 initial、动作引起的首次物理接触、
  success transition 与 final head-camera RGB，同一 physics step 去重并编码为
  2 FPS H264；初始支撑接触不占动作接触关键帧。

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

## 5. 已实现的向后兼容迁移

第一版采用 additive migration：

- 原样保留 `states.csv`、`semantic_trace.npz` 和 `events.jsonl`；
- 新增 `dynamics_trace.npz` 与 array manifest，不改变 legacy semantic stream 的采样率；
- 每个 episode 保存 `telemetry_profile.json` 快照；
- `episode.json` 新增 profile id、配置 hash、每个 stream 的周期、行数、dtype
  和 artifact 路径；
- initial 与 final 必须强制采样，即使 final step 不是 5 的倍数；
- contact monitor 始终逐 physics step 运行，不能跟随 dynamics 降采样；
- `TrajectoryView` 可选加载 dynamics；旧 episode 没有该文件时仍可工作；
- Runner 将 profile 与 dynamics artifact 纳入 SHA-256 审计。

当前 Recorder 在 `setup_demo()` 完成后才 attach，因此 `event_keyframes_v1` 的
initial 表示 expert 执行起点，不覆盖 setup/stabilization。Recorder 会先快照已有物理
接触，把桌面等支撑接触排除在“首次动作接触”之外；这一机制不能被解释为已记录 setup
阶段的完整视觉或数值状态。

## 6. 当前验收与剩余边界

当前已通过的代码级验收：

1. `legacy_v1` 不生成 dynamics，旧 `TrajectoryView` 仍可读取；
2. dynamics sample steps 精确满足 `0, 5, 10, ..., final`；
3. contact/semantic stream 在 profile 与 Recorder 中仍逐 physics step 运行，不随
   dynamics 降采样；
4. 正常结束时强制写 final sample，且 final 恰逢周期边界时不会重复；
5. 多任务 synthetic trajectory 与 generic Tools 已通过；
6. `--telemetry-profile balanced_v1|legacy_v1` 已贯穿 Agent、TaskGen/probe 与 ACT
   wrapper；
7. `event_keyframes_v1` 已验证 initial、动作接触、success、final 的逐步去重，事件与
   semantic stream 可通过精确 `video_frame_index` 对齐；视觉捕获失败时数值 telemetry
   仍保留且明确标记 failure。

仍需逐步补充：early success/异常退出的真实环境集成测试、短接触的真实物理回归、
ACT 与 expert 的长期性能开销、全场 actor profile、setup/stabilization telemetry、
事件触发 depth/segmentation，以及要求 250 Hz joint dynamics 的专用高频 profile。
`click_bell`、`adjust_bottle`、`grab_roller` direct ACT smoke 均已完成 2 episodes；它们
只验证 checkpoint、连续视频、`balanced_v1` telemetry 与 Trusted Tool 通路，不是统计
充分的性能验收。

## 7. 2026-07-16 真实运行结果

`click_bell`（seeds 100100/100101）、`adjust_bottle`（100201/100202）与
`grab_roller`（100300/100301）各完成两条 `balanced_v1` official expert 轨迹，六条
均通过官方 success。`adjust_bottle` seed 100200 出现 simulator unstable，已保留
审计并改用 100201/100202，不将不稳定 attempt 混入成功样本。

六条接受轨迹均完成 `event_keyframes_v1`：生成 keyframe manifest、H264
`video.mp4` 与可追溯帧索引；初始支撑接触没有误占动作接触关键帧。这验证的是 official
expert、Recorder 与视觉证据链。

另以官方 `click_bell demo_clean-50` checkpoint 运行 ACT seeds 100400/100401：前者
400 policy steps 失败，后者 70 policy steps 成功；连续视频分别为 400/71 帧、10 FPS、
H264 320x240。两条 `balanced_v1` telemetry 均可由 generic Trusted Tools 重算，结果与
官方 1/2 一致。该小样本只作为端到端格式 smoke。

同日以 paired expert seeds 运行 `adjust_bottle` ACT 100201/100202 与 `grab_roller` ACT
100300/100301，均为 2/2 official success。四条连续视频是 H264 320x240、10 FPS，
generic `official_check_success` 和 `time_to_success` 与 episode outcome 逐条一致。每任务
仅两条，因此仍不能把 2/2 解读为论文协议下的稳定 100% 成功率。
