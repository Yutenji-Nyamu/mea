# 2026-07-15 新 ToolGen 与轨迹存储审计记录

## 开发范围

本次没有改动 RoboTwin 官方 `policy/ACT/eval.sh`、ACT checkpoint、场景资产或既有 telemetry。开发集中在 MEA 的 post-rollout Tool 路径：

1. 新增 Trusted primitive `first_hammer_pickup_step`；
2. 新增真正不存在于 Trusted catalog 的 metric `pickup_to_first_contact_time`；
3. 用 first-pickup + first-contact 构造私有 composition oracle；
4. 泛化 ToolGen、ToolSpec 与 Plan Agent schema；
5. 新增直接 CLI `--target-metric`；
6. 实测存储规模并增加可复现 audit script 与中文说明。

## 关键定义

- pickup：hammer center Z 相对初始值首次上升至少 schema `pickup_height_threshold_m=0.03 m` 的 250 Hz sample；
- contact：hammer 与 block 的首次 strict physical contact；
- metric：`contact_time - pickup_time`，单位秒；
- 缺 pickup、缺 contact 或 contact 早于 pickup 时 `value=null`；
- 该 metric 是描述性测量，`passed=null`。

## 验证过程

- 定向单元测试：25 项全部通过；
- 全量 `tests/manipeval`：48 项全部通过；
- 真实 oracle 重算：ACT `null`；expert `1.66 s`；
- live UIUI Plan/ToolGen/Feedback：最终 v3 通过；
- ACT/expert 两条轨迹均满足 generated Tool deterministic、composition-oracle agreement、artifacts unchanged。

两个失败尝试被保留而没有覆盖：

- v1：GPT 将缺 contact 的 reason 写成 `contact_absent`，strict gate 拒绝；随后补充 exact reason enum；
- v2：GPT 使用不存在的 `schema["physics_timestep"]`，worker 拒绝；随后明确 simulation timestamp 与 `physics_timestep_seconds` 字段；
- v3：第 0 次生成仍未通过，第 1 次 regeneration 修复成功。

正式产物：

```text
mea/evaluation_runs/eval_20260715_new_tool_duration_v3/
  plan/evaluation_plan.json
  execution/round_1/planned_tool/
  summary/evidence_bundle.json
  summary/feedback/
  evaluation_report.md
```

失败证据：

```text
mea/evaluation_runs/eval_20260715_new_tool_duration_v1/
mea/evaluation_runs/eval_20260715_new_tool_duration_v2/
```

server live log：

```text
_ops_logs/new_tool_duration_live_20260715_160640.log
```

## 存储审计摘要

canonical ACT telemetry 实测约 2.75 MB，其中 `states.csv` 约 1.98 MB；expert 约 0.09 MB。仅把 states 换成 compressed NPZ 而不删字段，ACT episode 预计约 0.98 MB。完整三相机 RGB/depth/segmentation 250 Hz raw 约 37.6 GB/episode，因此建议采用 250 Hz 任务关键量、50 Hz 全机器人/目标 actor、10 Hz 视频、事件关键帧的多频率 profile。本轮只增加审计工具和设计文档，没有改变 rollout 产物格式。
