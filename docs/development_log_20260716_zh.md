# 开发记录：Aggregate、Execution VQA 与 run-local Tool 复用

日期：2026-07-16

## 1. 本批目标

本批补齐 ManipEvalAgent 从单 episode observation 向多 episode evaluation
observation 的最短通路：

```text
episode ToolResult[]
→ deterministic Aggregate
→ Execution VQA
→ round observation
→ Plan Agent
→ evaluation-level aggregate
→ Feedback Agent
```

同时让同一 evaluation 后续 round 可以复用已经验证的 generated Tool，并加入按
stage 选择 UIUI model 的可信 profile。

## 2. P0：跨 episode Aggregate Toolkit

新增 `mea/toolkit/aggregate.py`。它接受 Trusted Tool summary 或统一
`tool_execution` envelope，并写出 `aggregate_result.json`：

- boolean：true count/rate；
- numeric：mean、median、min、max、population standard deviation；
- `passed`：通过数量与比例，可用于 pickup rate 等判定；
- valid、missing、invalid episode count；
- 按 seed、round、variant、policy 分组；
- 每个统计量保留 episode、seed、role、evidence step 与 source artifact provenance；
- `policy_under_evaluation` 与 `expert_validation` 严格分 cohort；
- missing、invalid 和 `contact_precedes_pickup` 不进入 policy numeric statistics。

`scripts/manipeval_agent.py` 现在在每轮写
`execution/<round_id>/aggregate_result.json`，并在所有 round 结束后写
`summary/aggregate_result.json`。Plan Agent 与 Feedback 接收的是可信代码计算后的
结构化统计，而不是自行从多份 JSON 计算。

## 3. P1：Execution VQA

新增 `mea/execution_vqa/prototype.py`，与场景生成阶段的 scene Visual
Self-Reflection 分离：

1. 从 ACT video、contact events、Tool evidence 和 semantic timeline 选择
   initial、pickup 前后、first contact 前后与 final，共 4–8 帧；
2. 将 reference scene 和 rollout keyframes 组成一张 montage；
3. 把 montage 与原样保留的 numeric ToolResult 交给 Vision provider；
4. 只接受固定 JSON schema，包括 phenomenon、confidence、frame ids、numeric
   consistency 与 conflicts；
5. simulator numeric evidence 保持 authoritative，Vision 不得改写 ToolResult；
6. 冲突写入 `evidence_conflict`，供 Feedback/Plan Agent 决定是否追加验证。

第一版关注适合视觉判断的现象：蓝色外观、锤子是否明显抬起、方块是否发生可见
位移。精确位置、距离、接触、冲量和尺寸仍由 simulator Tool 判断。

## 4. P2：evaluation-local Tool registry

新增 `mea/toolgen/registry.py`，并接入 Tool orchestration：

- Round 1 的 `force_codegen` Tool 只有通过 AST、determinism、oracle agreement、
  property scenario 与真实 trajectory validation 后才能登记；
- registration 保存 ToolSpec contract hash、code hash、required signals、prompt/model
  metadata、validation episodes 和 telemetry schema compatibility hash；
- 同一 evaluation 后续 round 使用相同 ToolSpec 且 schema/hash 完全一致时，Router
  直接执行 registry 中的源码，`provider_called=false`；
- 缺文件、源码被修改、contract/schema 不一致时不会静默复用；
- 新增显式 `request_candidate_promotion()`：只有至少两个正例、两个反例、
  determinism/oracle agreement 全通过、至少一个真实 rollout 且 code hash 未变化时，
  才把 candidate 标记为 `eligible`；可执行 scope 仍保持 `run_local`；
- `trusted` 不会自动提升，状态只会进入
  `requires_code_review_and_tests`，仍需独立代码审查与仓库测试。

主要审计目录：

```text
mea/evaluation_runs/<evaluation_id>/tool_registry/
├── index.json
└── entries/<registration_id>/
    ├── generated_tool.py
    └── registration.json
```

## 5. Model profile

新增 `mea/providers/model_profiles.py` 和 `--model-profile`：

| profile | planner | taskgen | toolgen | vision | feedback |
| --- | --- | --- | --- | --- | --- |
| `legacy` | GPT-4o | GPT-4o | GPT-4o | GPT-4o | GPT-4o |
| `economy` | GPT-5.6 Luna | GPT-5.6 Luna | GPT-5.6 Luna | GPT-5.6 Luna | GPT-5.6 Luna |
| `balanced` | GPT-5.6 Luna | GPT-5.6 Terra | GPT-5.6 Terra | GPT-5.6 Luna | GPT-5.6 Luna |
| `quality` | GPT-5.6 Sol | GPT-5.6 Sol | GPT-5.6 Sol | GPT-5.6 Sol | GPT-5.6 Sol |

任一 stage 都可用 `--planner-model`、`--taskgen-model`、`--toolgen-model`、
`--vision-model` 或 `--feedback-model` 覆盖 profile。profile 只决定 model id，不会
绕过 schema validator、AST gate 或 evidence guard。

## 6. 论文方法覆盖与剩余 gap

本批对应论文中的三个关键连接：

- 多个 episode result 先经 deterministic Aggregate，形成 round/evaluation
  observation；
- Execution Stage 同时返回 scalar Tool 与 VQA observation；
- ToolGen 生成并验证的新 Tool 可以进入 Toolkit retrieval，而不是每轮重新生成。

仍未补齐的主要部分：

1. 跨 evaluation 的历史结果 database、query retrieval 与一致性比较；
2. candidate → Trusted 的正式 code review、仓库合并与版本发布 workflow；
3. 第二个 RoboTwin task，用于验证 RAG、TaskSchema、Recorder 和 Tool portability；
4. 通用 `TelemetryProfile` / `SignalCatalog` 与 `balanced_v1` runtime；
5. 基于 aggregate uncertainty、evidence conflict、预算与边际信息增益的主动停止；
6. 更完整的 Plan/TaskGen/ToolGen failure recovery；
7. 更多 policy、task 和 sub-aspect 上的系统性实验，而非单一 case study。

Recorder 的跨任务设计见 `docs/telemetry_balanced_v1_design_zh.md`；当前实现没有在
本批同时迁移 telemetry schema，以免把 Tool/Agent 回归与采样格式变化混在一起。
当前 Plan Agent 已能看到并报告 `evidence_conflict`，但仍受预先请求的 template
集合和三轮预算约束；尚未实现因视觉冲突而自动插入一个新的复测 round。

## 7. 验证记录

### 7.1 单元与集成测试

- server 全量命令：
  `python -m unittest discover -s tests/manipeval -p 'test_*.py' -v`；
- 结果：93 项全部通过；
- `py_compile`、`git diff --check`、官方 `policy/ACT/eval.sh` 与根
  `README.md` 无差异检查均通过；
- 日志：`_ops_logs/next_batch_tests_20260716_101814.log`。

### 7.2 三个真实 ACT episodes

- run：`run_20260716_aggregate_blue_seed100010`；
- seeds：`100010`、`100011`、`100012`；每个均有独立 telemetry 与 MP4；
- GPT-5.6 Terra 提案得到蓝色 `[0.0, 0.2, 1.0]`、官方位置/yaw 随机化；
- scene VQA 使用 GPT-5.6 Luna，识别为蓝色，confidence `0.92`，无需 repair；
- ACT 三次均跑满 400 policy steps，pipeline return code 为 0；
- pickup rate：`3/3 = 1.0`；
- strict hammer-block contact rate：`0/3 = 0.0`；
- official success rate：`0/3 = 0.0`；
- minimum XY error mean/median/min/max：
  `0.021095 / 0.010782 / 0.001492 / 0.051010 m`；
- 三次 policy duration 均因没有 contact 而标为 missing，mean/median 等保持
  `null`，没有混入 expert；
- expert 单独 cohort：pickup/contact/success 均为 true，
  pickup-to-contact duration 为 `1.928 s`；
- 日志：`_ops_logs/run_20260716_aggregate_blue_seed100010.log`。

### 7.3 Aggregate、Tool reuse 与 candidate gate

- evaluation：`eval_20260716_aggregate_vqa_reuse_live_v2`；
- 首次 duration Tool 为 `force_codegen`，GPT-5.6 Terra 调用一次，
  4281 tokens；
- 相同 ToolSpec 第二次命中 `run_local_reuse`，provider 调用次数为 0，四个
  ACT/expert episode 的结果与第一次完全一致；
- registration：`runlocal_13d26d92bdd67c55e757`；
- 两个正例、两个反例、两个真实 rollout、determinism/oracle agreement 与 code
  integrity 均通过，candidate status 为 `eligible`；scope 仍是 `run_local`，
  Trusted status 为 `requires_code_review_and_tests`；
- Aggregate 共输出 7 个 metric，并严格拆分 policy/expert cohort；
- 完整审计位于该 evaluation 目录下的 `live_validation_final.json`。

### 7.4 Execution VQA 与现场修复

- representative episode：ACT seed `100010`；
- 最终关键帧：initial `0`、pickup before/after `65/66`、final `399`；
- GPT-5.6 Luna 判断蓝色、明显抬锤、方块可见位移，
  `evidence_conflict=false`；本次调用 3019 tokens；
- 首次现场验证暴露了关键帧 fallback 缺陷：hammer-table 等非目标 contact 被误标为
  hammer-block `contact_after`；
- 修复后 contact event 必须同时包含 hammer 与 block，且 duration 的
  `contact_detected=false` 不再生成 contact frame label；新增回归测试并重新运行
  Vision，最终只保留上述四帧；
- 最终操作摘要日志：
  `_ops_logs/batch_p0_p1_p2_live_20260716_102042.log`。

### 7.5 Model profile

- `--model-profile balanced --plan-only` 已真实调用成功；
- manifest 记录 planner/vision/feedback=`gpt-5.6-luna`，
  taskgen/toolgen=`gpt-5.6-terra`；
- 日志：`_ops_logs/eval_20260716_model_profile_balanced_plan.log`；
- feature DCO commit：`c43afc7e4dbddb84c9fa251ba6c807b80afca64b`；
- 已在第一次尝试推送到公开仓库 `Yutenji-Nyamu/mea` 的 `main`，并验证
  `origin/main` 与本地 commit 完全一致；
- commit/push 日志：`_ops_logs/next_batch_commit_20260716_102418.log`、
  `_ops_logs/next_batch_push_20260716_102457.log`。

## 8. 操作边界

- 未修改官方 `policy/ACT/eval.sh`；
- 未把 API key、SSH 密码、checkpoint、video 或 generated evaluation artifacts
  加入 Git；
- 未实现自动 promotion；本批只有显式、可信条件检查后的 candidate eligibility；
- 未把 `balanced_v1` 设计写成已部署行为；
- 根目录 `README.md` 保持简洁，本批详细内容只写入 `docs/`。
