# 2026-07-16：official 事件视频、跨任务 expert 与 ACT smoke

## 1. 本轮目标

本轮补齐此前 `click_bell` official expert 没有 rollout video、Dynamic Execution VQA
只能跳过的问题，并把同一套 TaskSchema/Recorder/Tool 通路扩展到
`adjust_bottle` 与 `grab_roller`。同时增加选择性 ACT checkpoint 下载入口，更新运行与
架构文档。RoboTwin 官方任务源码、官方 `policy/ACT/eval.sh` 和根执行语义未改写。

## 2. official expert 事件视频

`EpisodeRecorder` 新增默认关闭的 `event_keyframes_v1`。official expert 启用后捕获：

1. expert 执行起点；
2. 动作引起的首次 physical contact；
3. 官方 success transition；
4. final。

同一 physics step 的触发原因合并到一帧。Recorder 在 expert 动作前快照已有 contact，
因此 bottle/roller 与桌面的初始支撑接触不会消耗首次动作接触帧。每帧保存 physics step、
时间、原因和 `video_frame_index`，再以 2 FPS 编码 H264 320x240 `video.mp4`。这是稀疏
事件证据，不是连续运动录像。

相机、ffmpeg 或视觉 manifest 失败都不会中断数值 telemetry；episode 会保存失败阶段和
错误。视觉完成时才声明 video artifact；manifest 失败会删除残留容器，Agent 也会核对
`episode.json` 的 completed 状态和 video 声明，不能只凭路径存在使用视频。ACT 的连续
10 FPS video contract 保持不变。

Execution VQA 现在按 round route 选择证据：official route 只选 expert，generated route
只选 ACT，不允许任意 fallback 混淆角色。多 episode 中优先选择实际存在 video 的 episode；
若所有 official expert 视频都失败，则保留可审计的 skipped 结果。success transition 的
精确稀疏帧索引会进入关键帧选择。

## 3. 新任务与 seed 边界

新增 TaskSchema：

- `adjust_bottle`：bottle pose、functional point、左右 TCP；
- `grab_roller`：roller pose、两侧 contact point、左右 TCP。

两者复用 generic `official_check_success` 与 `time_to_success`；视觉问题分别限制为 bottle
是否被移到目标侧、roller 是否被双臂抬升。schema 阈值用于声明可审计语义，最终 outcome
仍以官方 `check_success()` 为准。

official expert batch 现在有界扫描候选 seed。只对明确的 `UnStableError` 跳过并记录
rejected seed；稳定 seed 上的 expert 失败仍是失败，不能被筛掉。这个规则与 RoboTwin ACT
评估先用 expert 检查可解性的边界一致。

## 4. 真实 official expert 结果

所有运行使用 `balanced_v1 + event_keyframes_v1`：

| task | accepted seeds | official success | time to success | event video |
| --- | --- | ---: | --- | --- |
| `click_bell` | 100100, 100101 | 2/2 | 3.424 s, 3.320 s | 两段各 4 帧，H264 320x240 |
| `adjust_bottle` | 100201, 100202 | 2/2 | 6.700 s, 6.864 s | 两段各 4 帧，H264 320x240 |
| `grab_roller` | 100300, 100301 | 2/2 | 4.844 s, 4.948 s | 两段各 4 帧，H264 320x240 |

`adjust_bottle` seed 100200 在 setup 阶段被官方环境判为 bottle unstable；该 attempt 保留
在 rejected-seed 审计中，不计入 2 条 accepted expert 轨迹。六段视觉采集均 completed、
errors 为空；bottle/roller 的桌面支撑 contact 均未占用动作接触帧。

这 6/6 是 official expert/证据通路验收，不是 learned policy success rate。

## 5. ACT checkpoint 与 direct smoke

新增 `scripts/download_act_checkpoint.py`：

- 接受一个或多个 canonical task name；
- 固定 RoboTwin 2.0 checkpoint release revision；
- 每任务只下载 `policy_last.ckpt` 与 `dataset_stats.pkl`；
- 拒绝路径穿越，支持 `--dry-run`，下载后检查文件存在与大小；
- 网络失败时提示在服务器启用 AutoDL 学术加速或配置 `HF_ENDPOINT` mirror，不建议把常规
  大 checkpoint 经个人电脑中转。

开发服务器的 `click_bell demo_clean-50` 权重与 stats 已按 SHA-256 校验。直接调用
`policy/ACT/eval_mea.sh` 运行 demo_clean seeds 100400/100401：

- checkpoint loader：`All keys matched successfully`；
- seed 100400：400 policy steps / 10,200 physics steps，official failure；
- seed 100401：70 policy steps / 4,909 physics steps，official success，
  `time_to_success=19.636 s`；
- official smoke result：1/2；Trusted Tool 重算一致；
- 连续视频：H264 320x240、10 FPS，分别为 400 帧/40.0 s 与 71 帧/7.1 s。

随后完全在服务器侧启用 AutoDL 学术加速，以较长 Hugging Face 大文件超时和单 worker
续传 `adjust_bottle`、`grab_roller` checkpoint；没有经本机中转。两者使用与 expert
相同的 paired seeds 直接评估：

| task | seeds | official result | policy steps | time to success | continuous video |
| --- | --- | ---: | --- | --- | --- |
| `adjust_bottle` | 100201, 100202 | 2/2 | 117, 126 | 41.888 s, 43.296 s | 118/127 帧，11.8/12.7 s |
| `grab_roller` | 100300, 100301 | 2/2 | 93, 90 | 33.560 s, 31.332 s | 94/91 帧，9.4/9.1 s |

四段视频均为 H264 320x240、10 FPS；两个 checkpoint loader 均报告
`All keys matched successfully`。generic `official_check_success` 与 `time_to_success`
Trusted Tools 对四条 telemetry 的重算和官方结果逐条一致。

这六条 ACT 样本只验证 checkpoint、官方 evaluator、连续视频、telemetry 与 Tool 通路。
论文结果仍需要更多 paired seeds，并区分 Easy `demo_clean` 与 Hard
`demo_randomized`；2/2 不能解读为稳定的 100% 成功率。

## 6. 可读性维护

- 新增 `docs/running_guide_zh.md`，覆盖环境、资产、checkpoint、三个运行入口、产物和
  凭据边界；根 README 只增加一行文档入口；
- 更新 `docs/architecture_and_dataflow_zh.md`，明确 generated+ACT 与 official expert
  两条路线、TaskSchema 扩展边界、证据权威和当前 BBH ACT 硬编码；
- 校准 `docs/multitask_schema_zh.md` 与 `docs/telemetry_balanced_v1_design_zh.md`；
- 仅在 route 选择、初始支撑接触过滤和 BBH-specific ACT bridge 等关键位置补少量注释。

## 7. 当前剩余 gap

1. official expert 已有 VQA video，但真实模型调用仍需 UIUI key 才能做 live Dynamic
   Execution VQA 验收；
2. 外层 Agent 的 ACT bridge 仍固定 BBH；非 BBH checkpoint 可直接评估，但还不能通过
   通用 `expert|act|both` backend 自动编排；
3. `event_keyframes_v1` 适合 outcome/事件证据，不适合判断连续轨迹质量、遮挡全过程、
   depth 或 segmentation；
4. 每任务 2-episode ACT smoke 不能作为论文成功率，需要扩大 paired accepted seeds；
5. 新任务专属动力学 metric 仍需按问题增加稳定 signal/Trusted Tool，不能从未记录字段中
   事后推断。
