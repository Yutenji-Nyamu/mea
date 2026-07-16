# 2026-07-16：Exact-seed Easy/Hard paired protocol

## 动机

原有 Agent `both` 可以在一次运行结束后核对 official expert 与 ACT 实际使用的 seed，防止
两类证据静默错位；但 RoboTwin 的 expert eligibility 与 ACT evaluator 都可能在失败后向后
扫描候选 seed。两边即使最终列表相同，也不能证明它们执行了最初计划的 seed。该行为适合
提高 smoke test 完成率，却不足以支撑论文中的 Easy/Hard 配对比较。

本次增加专用 exact-seed paired protocol，优先解决一个边界清晰、实现成本较低且直接影响
实验可信度的 gap：预先锁定一组 seed，并让 Easy（`demo_clean`）与 Hard
（`demo_randomized`）在完全相同的 seed 上评估同一个 `demo_clean-50` ACT checkpoint。

## 设计

- `scripts/manipeval_paired.py` 提供无 UIUI 依赖的实验入口；`--seeds` 与 `--manifest`
  二选一，前者会生成并冻结 `seed_manifest.json`。
- manifest 固定 task、seed 的顺序、Easy/Hard condition、checkpoint setting、expert data
  数量与 policy seed；拒绝空列表、重复值、负数和任务名不匹配。
- runner 先以与 ACT rollout 相同的 `eval_mode=True` 分布，对每个请求 seed、每个 condition
  单独运行一次 exact expert eligibility probe，并记录 `passed`、`unstable`、
  `expert_failed` 或 `error`。任何状态都不触发替代 seed。
- ACT 只运行两边共同 eligible 的有序交集。evaluator 仍保留 expert precheck，因为官方
  instruction 生成依赖其中的 episode 信息；若复查结果与冻结 eligibility 不一致，则记录
  protocol violation，而不是扫描下一个 seed。
- `mea/paired.py` 在确定性代码中按 seed join 两边结果，计算 paired coverage、Easy/Hard
  成功率、`Hard - Easy` 差值、四种成败组合和双成功样本的 time-to-success 差值。语言模型
  不参与这些数值计算。
- 运行产物写入 `mea/paired_runs/<run_id>/`，包含冻结 manifest、eligibility/condition 明细、
  ACT exact-seed 结果、telemetry 和 paired summary；运行目录不提交到 Git。

最小运行示例：

```bash
python scripts/manipeval_paired.py \
  --repo-root "$PWD" \
  --task-name click_bell \
  --task-module envs.click_bell \
  --seeds 100400 100401 \
  --run-id click_bell_paired_smoke \
  --gpu 0 \
  --telemetry-profile balanced_v1
```

## 可信边界与限制

- paired runner 只负责 official task 的 eligibility、ACT rollout 和确定性统计，不执行
  planning、Execution VQA 或最终自然语言反馈；因此无需 UIUI key，也不产生视觉结论。
- 共同 eligibility 交集可能小于请求 seed 集。报告必须同时给出 requested、eligible、
  evaluated 和 coverage，不能把筛选后的成功率表述成全体请求 seed 的成功率。
- `time-to-success` 只统计 Easy/Hard 均成功的 survivor 子集，存在条件化偏差；主指标仍应
  是同一 paired denominator 上的成功率与四格成败结果。
- 单 seed 或少量 smoke test 只能验证协议和调用链，不能支持性能结论。正式实验应事先冻结
  足量 seed，保留失败状态和完整 artifact，并避免在观察结果后修改 seed 列表。
- 当前 checkpoint contract 仍是官方 `demo_clean-50`。扩展到其他训练设置时，需要同时
  扩展 checkpoint preflight、manifest contract 和结果分层，不能只替换目录名。
- “exact seed”表示清单、顺序与 numeric seed 不被替换，不表示 Easy/Hard 的底层几何完全
  相同。配置相关的随机化可能在 actor 放置前改变 RNG 消费顺序；当前只能作 same-seed
  paired comparison。严格 identical-scene 因果对照仍需拆分 RNG stream，或保存并重放
  scene specification。
- seed 替换、eligibility 复查漂移、缺 telemetry 或三方成功判定不一致都会令
  `valid_for_comparison=false`，runner 默认非零退出；允许 protocol violation 的显式选项
  仅供诊断。

## 真实服务器 smoke

在 AutoDL RoboTwin 环境中对三个已有服务器 checkpoint 的任务各跑通一个完整配对：

| 任务 | exact seed | paired coverage | Easy | Hard | outcome | 运行目录 |
| --- | ---: | ---: | --- | --- | --- | --- |
| `click_bell` | 100401 | 1/1 | 成功，19.636 s | 失败 | `easy_only` | `run_20260716_click_bell_evalmode_seed100401`（约 4.0 MiB） |
| `adjust_bottle` | 100201 | 1/1 | 成功，41.888 s | 失败 | `easy_only` | `run_20260716_adjust_bottle_evalmode_seed100201`（约 5.5 MiB） |
| `grab_roller` | 100300 | 1/1 | 成功，33.560 s | 失败 | `easy_only` | `run_20260716_grab_roller_evalmode_seed100300`（约 5.4 MiB） |

三项最终实验中，Easy 与 Hard probe 均明确记录 `eval_mode=true`，official expert eligibility
均通过；冻结交集仍是原始单 seed。六次 ACT exact result 均声明
`no_seed_replacement: true`，telemetry 中的实际 seed 与 manifest 一致。每个结果也都通过
evaluator、Recorder metadata 与 `official_check_success` Trusted Tool 三方交叉核验，三个
summary 均为 `valid_for_comparison=true`，没有 protocol violation。Hard 的三次失败都是完整
rollout 的有效策略结果，不是 eligibility、checkpoint、telemetry 或 pipeline 失败。

每项仍然只是 `N=1` 的调用链 smoke。三个不同任务不能直接合并成 `N=3` 的同分布样本；
这些结果只证明 exact-seed 协议可跨 TaskSchema/ACT checkpoint 复用，并能捕捉同一场景标识下
Easy/Hard 的结果差异，不能据此估计泛化成功率或报告论文结论。

## 后续建议

1. 为长实验增加 chunk/resume 与批次状态恢复，避免一个 20–50 seed 任务因单个基础设施异常
   整批重跑。
2. 为每个任务预先固化 20–50 个以上的独立 seed 集，增加置信区间或配对显著性统计，再形成
   可用于论文表格的批量实验报告。
3. 增加跨任务 batch matrix 汇总，但始终保留每任务分母，不能把不同任务的单 seed 直接池化。
4. 如需可视化解释，可让 Agent 读取 paired run 的指定 seed artifact；在此之前不要把 Agent
   `both` 与严格 paired protocol 混称为同一结果。
