# 轨迹数据、可测量范围与存储折中

## 1. 当前记录的是什么

当前 Recorder 不是完整 simulator dump，而是 `beat_block_hammer` 的多频率任务语义切片：

| 频率 | 产物 | 主要内容 |
| --- | --- | --- |
| 250 Hz physics step | `semantic_trace.npz` | hammer/block 与 functional point position、双臂 TCP、success、physics/policy step、simulation time |
| policy boundary，ACT 约 10 Hz | `states.csv` | action、双臂 qpos/qvel、EE/TCP、gripper、目标 actor pose/velocity、success |
| 事件触发 | `events.jsonl` | contact interval、strict physical contact、peak impulse、minimum separation、success/error |
| 10 FPS | `video.mp4` | 压缩 RGB 视觉证据 |
| episode 级 | `episode.json`、`schema.json` | policy/seed/结果/步数与 task signal/threshold 契约 |

它已经足够计算 pickup、functional-point 距离与对齐、strict contact、首次接触、peak impulse、TCP 路径、官方成功、完成时间，以及本次新增的 pickup-to-contact duration。

本项目中 `pickup` 的精确定义是 hammer center 相对初始 Z 首次上升至少 `0.03 m`；它是 first lift，不等同于夹爪首次稳定抓住 hammer。若要分析真实 grasp onset，需要新增 gripper-hammer contact 或 constraint signal。

## 2. 真实空间实测

对蓝色方块 seed 100000 的 canonical ACT episode 实测：

| 产物 | 大小 |
| --- | ---: |
| `states.csv`，402 行 × 251 列 | 1,980,406 B |
| `semantic_trace.npz`，14,853 physics rows | 469,427 B |
| H264 RGB，320×240、10 FPS、400 帧 | 277,479 B |
| events/schema/episode/tool results | 约 14 KB |
| ACT telemetry 合计 | 约 2.75 MB |
| expert telemetry | 约 0.09 MB |
| ACT + expert | 约 2.86 MB |

`states.csv` 占主要空间。只改变编码而不删除字段时，gzip 约为 0.435 MB；把数值列转为 float32 compressed NPZ 约为 0.224 MB。当前 ACT episode 因而可以从约 2.75 MB 降至约 0.98 MB。这个优化优先级高于删 signal。

可复现实测命令：

```bash
python scripts/analyze_trajectory_storage.py \
  mea/generated_tasks/run_20260715_telemetry_blue_seed100000/evaluation/telemetry/act/episode_000_seed_100000
```

## 3. 当前损失了哪些分析能力

当前不能可靠测量：

- 250 Hz joint oscillation、jerk、瞬时 qvel 峰值、torque 与 controller tracking error；
- hammer/block 的高频 orientation、冲击速度、撞击角度和完整 force curve；
- hammer/block 之外的碰撞、障碍物距离和全场 actor 运动；
- `setup_demo()` / stabilization 阶段的落体、初始碰撞与穿模；
- depth、segmentation、point cloud、精确遮挡与视觉输入异常；
- planner/IK 内部状态和 ACT hidden state、action chunk、temporal aggregation uncertainty。

“完整数值状态”和“完整传感器数据”必须分开讨论。250 Hz 全数值状态加约 50 个 actor 估计为 53 MB raw、约 10–30 MB compressed/episode；三相机 RGB+depth+segmentation 若全部以 250 Hz 原始保存约为 37.6 GB/episode，再保存 XYZ point cloud 还会增加约 41 GB。后者比当前高四个数量级，不适合作为默认模式。

## 4. 推荐的下一版多频率 Profile

建议保留按用途分频，而不是无差别全存：

- 250 Hz：task functional points、关键 actor position/quaternion/velocity、TCP、success、contact begin/end 与 peak；
- 50 Hz：完整双臂 qpos/qvel、EE/TCP、gripper、action target、任务 actor pose/velocity；
- 10 Hz：policy action、observation metadata 与 H264 RGB；
- 事件关键帧：initial、pickup、contact、success、failure 时保存 RGB/depth/segmentation；
- setup/stabilization：单独以 25 Hz 保存数值和首尾关键帧；
- 调试模式：在 contact/error 前后约 0.5–1 s 临时提升所需 signal 到 250 Hz。

预计选定任务 actor 的 profile 为 2–5 MB/episode；50 Hz 覆盖全场 actor 时约 5–20 MB/episode。它能覆盖大部分后续 Tool，同时远小于全传感器 dump。Recorder 的下一步应先把 `states.csv` 改为保留同字段的压缩列式/NPZ 格式，再按具体新指标增加 signal；本轮没有改变正式 rollout 的记录格式，避免未经跨版本验证就破坏现有轨迹兼容性。
