# 2026-07-16：通用 official ACT backend

## 本轮最小实现

- 将任务 route 与 execution backend 解耦：`official` 只表示复用官方任务源码；执行可选
  `expert`、`act` 或 `both`。
- `scripts/manipeval_agent.py --execution-backend expert|act|both` 把选择传入 TaskGen。
  `act` 使用 setup/rule probe 并沿用 RoboTwin evaluator 的 expert eligibility；`both`
  同时保留 expert 验证。两者均以 ACT 作为被评 policy 和报告主证据。
- TaskGen 的 ACT runner 按当前任务解析 checkpoint，并在启动仿真前检查
  `policy_last.ckpt` 与 `dataset_stats.pkl`。大文件继续只在服务器侧通过选择性下载脚本、
  AutoDL 学术加速或 Hugging Face mirror 获取，不经个人电脑/Codex 工作区中转。
- Execution VQA 按 backend 选择视频：expert 使用 `event_keyframes_v1` 稀疏事件视频，
  ACT/`both` 使用连续 rollout 视频。expert 成功与 ACT policy success 在汇总中保持分离。

## 真实服务器验收

服务器 `/root/autodl-tmp/mea` 的完整单测为 168/168 通过。以下均使用服务器已有的
官方 `demo_clean-50` checkpoint；它们是通路 smoke，不是论文成功率统计。

| 入口 | 任务 / seed | 结果 | Dynamic Execution VQA |
| --- | --- | --- | --- |
| inner TaskGen `both` | `click_bell` / `100401` | expert 与 ACT 均成功；实际 seed 对齐；expert 3.276s，ACT 19.636s | inner smoke 不调用模型 |
| full Agent `act` | `click_bell` / `100401` | ACT 1/1，19.636s，pipeline completed | `bell_visibly_pressed=true`，confidence 0.96，与数值证据一致 |
| full Agent `act` | `adjust_bottle` / `100201` | ACT 1/1，41.888s，pipeline completed | `bottle_visibly_repositioned=true`，confidence 0.98，与数值证据一致 |

两次 full Agent 均从 10 FPS 连续 ACT 视频选择 initial、success 前后和 final 帧；官方
`check_success()` 与 Trusted Tools 仍是结果权威，VQA 只补充可见现象且没有覆盖数值结论。
模型 key 仅通过当前进程环境和 SSH stdin 注入，未写入仓库或运行 manifest。

## 当前边界

- Agent ACT contract 暂固定为官方 `demo_clean-50` checkpoint；尚未开放任意 task config、
  checkpoint setting 或 expert-data count。
- `both` 已能收集两类证据并拒绝最终实际 seed 不一致，但两边尚未由同一显式 seed
  manifest 预先驱动，也没有 Easy/Hard paired 统计，不能视为论文级 paired 实验。
- 完整 Dynamic Execution VQA 和最终反馈需要有效的 `UIUI_API_KEY`；没有有效 key 时仍可
  完成 inner TaskGen、仿真、telemetry、Trusted Tools 与确定性聚合测试。
- generated task family 仍主要限于 `beat_block_hammer`，本轮没有扩大生成/修复边界。

## 下一轮建议

1. 增加显式 seed manifest，让 expert 与 ACT 在相同 Easy/Hard scene seed 上成对运行。
2. 增加 paired aggregate（成功率、耗时及逐 seed 差异），避免把 smoke test 当论文统计。
3. 在保持 preflight 的前提下参数化 task config/checkpoint setting，并用更多 schema-backed
   official 任务做小样本回归。
