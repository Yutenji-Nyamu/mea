# 轨迹数据、可测量范围与多频率 Recorder 设计

## 1. 当前实际记录了什么

当前 Recorder 不是完整的 simulator dump，而是面向
`beat_block_hammer` 的任务语义切片：

| 采样时机 | 产物 | 主要内容 |
| --- | --- | --- |
| 每个 250 Hz physics step | `semantic_trace.npz` | hammer/block position、functional points、双臂 TCP、success、physics/policy step、simulation time |
| policy boundary | `states.csv` | action、双臂 qpos/qvel、EE/TCP、gripper、目标 actor pose/velocity、success |
| 每个 physics step 检查、稀疏写入 | `events.jsonl` | contact interval、strict physical contact、peak impulse、minimum separation、success/error |
| 每个 policy step | `video.mp4` | 10 FPS H264 RGB 视觉证据 |
| 每个 episode | `episode.json`、`schema.json` | policy、seed、结果、步数、TaskSchema signal 与 threshold 契约 |

这些数据已经足够支持当前 Auto Tool Router、Trusted Tools，以及新的
pickup-to-contact Tool。后者使用 250 Hz `hammer_position` 判断首次抬升，
再使用 contact event 确定首次物理接触，不需要重新设计 Recorder。

这里的 `pickup` 精确定义为：hammer center 相对初始 Z 首次上升至少
TaskSchema 中的阈值，当前为 `0.03 m`。它表示 first lift，不等同于
gripper 首次稳定抓住 hammer。若要分析真正的 grasp onset，需要新增
gripper-hammer contact 或 constraint signal。

## 2. 当前空间实测

对蓝色方块 seed 100000 的 canonical ACT episode 实测：

| 产物 | 实际大小 |
| --- | ---: |
| `states.csv`，402 行 × 251 列 | 1,980,406 B |
| `semantic_trace.npz`，14,853 physics rows | 469,427 B |
| H264 RGB，320×240、10 FPS、400 帧 | 277,479 B |
| events、schema、episode、tool results | 约 14 KB |
| ACT telemetry 合计 | 约 2.75 MB |
| expert telemetry | 约 0.09 MB |
| ACT + expert | 约 2.86 MB |

`states.csv` 占主要空间。只改变编码而不删除字段时，gzip 后约为
0.435 MB；把数值列转为 float32 compressed NPZ 后约为 0.224 MB。
因此，当前 ACT episode 理论上可在基本不损失字段的情况下由约
2.75 MB 降至约 0.98 MB。正式迁移格式前仍需兼容已有
`TrajectoryView` 与 ToolGen artifact preflight。

可复现实测命令：

```bash
python scripts/analyze_trajectory_storage.py \
  mea/generated_tasks/run_20260715_telemetry_blue_seed100000/evaluation/telemetry/act/episode_000_seed_100000
```

## 3. 当前不能可靠分析的内容

当前数据不适合直接判断：

- 250 Hz joint oscillation、jerk、瞬时 qvel 峰值、torque 与 controller
  tracking error；
- hammer/block 的高频 orientation、冲击速度、撞击角度与完整 force
  curve；
- hammer/block 之外的碰撞、障碍物距离与全场 actor 运动；
- `setup_demo()` / stabilization 阶段的落体、初始碰撞与穿模；
- depth、segmentation、point cloud、精确遮挡与视觉输入异常；
- planner/IK 内部状态，以及 ACT hidden state、action chunk 与 temporal
  aggregation uncertainty。

“完整数值状态”和“完整传感器数据”必须区分。以本次 episode 的长度
估算，250 Hz 全数值状态加约 50 个 actor 约为 53 MB raw、约
10–30 MB compressed；三相机 RGB、depth、segmentation 若都以 250 Hz
原始保存约为 37.6 GB/episode，再保存 XYZ point cloud 还会增加约
41 GB。后者不适合作为默认模式。

## 4. `balanced_v1` 多频率设计

`balanced_v1` 是下一版 Recorder 的设计目标，不是本轮已经实现或实测的
runtime 行为。建议按信号用途分频：

- 250 Hz semantic stream：TaskSchema tracked actor 的关键 position、
  functional points、TCP、success 与时间索引；
- 250 Hz event monitor：contact begin/end、first physical contact、peak
  impulse、minimum separation 与 success/error；
- 50 Hz dynamics stream：完整双臂 qpos/qvel、EE/TCP、gripper，以及目标
  actor 的 position、quaternion、linear/angular velocity 与 functional
  pose；
- policy boundary / 约 10 Hz：action 与当前完整 state snapshot；
- 10 FPS：继续使用现有 H264 RGB，不在 Recorder v2 中改变视频路径；
- 事件关键帧：initial、pickup、contact、success、failure 时保存
  RGB/depth/segmentation，列为后续功能，本版不实现。

建议使用可信、allowlisted profile，而不是让 GPT 自由生成采样配置：

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

## 5. 向后兼容约束

第一版 `balanced_v1` 应采用 additive migration：

- 原样保留 `states.csv`、`semantic_trace.npz` 与 `events.jsonl`；
- 新增 `dynamics_trace.npz`，而不是改变现有 semantic stream 的采样率；
- `episode.json` 新增 profile id、配置 hash、各 stream 的采样周期、行数
  与 artifact 路径；
- episode 内保存 `telemetry_profile.json` 快照；
- 旧 `TrajectoryView` 忽略未知 artifact 后仍能运行，新 Tool 才按需读取
  dynamics stream；
- initial 与 final 必须强制采样，即使 final physics step 不是 5 的倍数；
- contact monitor 始终逐 physics step 运行，不能随 dynamics 一同降采样。

现有 Recorder 在 `setup_demo()` 完成后才 attach，所以仅增加 profile 不能
补录 setup/stabilization。现有 RGB 写入也位于 `Base_Task.take_action()`，
不由 Recorder 控制。这两个问题需要单独修改生命周期，不属于最小
`balanced_v1`。

50 Hz selected-actor stream 可继续在 episode 结束时写 compressed NPZ，
但应使用固定列顺序的 typed array，避免为每个样本建立大型 Python dict。
未来若实现 250 Hz 全场 actor 模式，应改用 chunked writer，不能把全部数据
长期积存在内存中。

## 6. 空间目标与本轮边界

`balanced_v1` selected-actor profile 的初步设计目标是 2–5 MB/episode。
这是依据当前行数、字段数和压缩率得到的数量级估计，不是已经运行得到的
实测结果。若 50 Hz 覆盖全场 actor，预计约为 5–20 MB/episode，同样需要
实现后通过真实 rollout 验证。

本轮 Auto Tool Router 与 pickup-to-contact Tool 开发不修改 runtime
Recorder，理由是：

1. 当前数据已经满足 Router 和新 Tool 的 signal contract；
2. 现有 ToolGen preflight 与 `TrajectoryView` 依赖 legacy artifacts；
3. 将 Tool 路由变化和 Recorder schema 迁移放在同一提交中会扩大回归面；
4. `balanced_v1` 应在独立开发批次中通过 expert 与 ACT 1-episode 回归验证。

后续验收至少包括：legacy 与 balanced 的全部 Trusted Tool 结果一致；采样
steps 满足 `0, 5, 10, ..., final`；短暂 contact 不因 50 Hz dynamics 而
丢失；early success 与异常退出仍写 final sample；并真实测量 storage 与
wall-time overhead。
