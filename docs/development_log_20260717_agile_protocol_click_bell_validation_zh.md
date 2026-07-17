# 2026-07-17：敏捷 ACT 协议、click_bell generated family 与缓存验证

本轮遵循敏捷预算：默认 1 次，需要时才放大到 3 / 5；仅评估 ACT，不接入第二种 policy。

## 实现

- 新增 ACT-only 完整 Agent protocol runner：1/3/5 repetition 与 episode、chunk/resume、
  append-only attempt、wall-clock/step/success/failure 统计、JSON 与 Markdown 报告。
- 新增 click_bell `position_lr` generated family：左右固定位置、同 seed、受限 overlay、逐 seed
  simulator XY / rule / expert gate、ACT、Tool、Aggregate、Dynamic Execution VQA 与实测位置证据。
- 新增 cached Planner/VQA scorer：严格 1/3/5 case 预算、模型 Planner 排除规则、
  precision/recall/F1/accuracy/AUROC、human/proxy 分层及 artifact hash。
- 运行指引和架构文档已同步；主 README 的既有文档入口不变。

## 边界

- protocol v1 只支持有 TaskSchema 的 official ACT 任务，BBH 仍使用原 generated route。
- click_bell v1 只生成位置 overlay，不做任意资产、纹理或 3D 生成，也不做模型代码修复。
- cached validation 只验证 scorer 与现有 artifact；没有人工小数据集时不能声称复现论文指标。
- 任何 ACT checkpoint 与数据集仍只在服务器侧按任务下载，不经过本机或 Codex 工作区。

## 验证记录

- 全量 `tests/manipeval`：206 tests，全部通过，用时约 24 秒。
- generated plan-only：`eval_20260717_click_bell_position_plan_smoke` 成功生成 left → right 两轮，
  两轮复用 seed `100401`。
- generated live：`eval_20260717_click_bell_position_lr_live_1` 完成两轮真实仿真。左右 bell 实测
  XY 分别约为 `[-0.20, -0.08]`、`[0.20, -0.08]`，均与声明值一致，rule/expert gate 均通过；
  ACT 左侧 `0/1`、右侧 `1/1`，合计成功率 `0.5`。左侧 Execution VQA 正确识别未按铃；右侧
  VQA 请求遇到 HTTP 502，因此该次运行诚实标为 `completed_with_pipeline_failure`，不能记为完整
  Agent E2E 通过。代码随后补充 transient HTTP 408/409/425/429/5xx 重试及单测；没有为了美化
  结果重跑 ACT。
- cached validation：`validation_20260717_cached_real_smoke` 使用已有真实 artifact、预算 1，未调用
  provider。Planner precision/recall/F1/exact/first-template 均为 `1.0`；VQA accuracy/coverage/
  precision 均为 `1.0`。由于只有一个 simulator-proxy 正样本，AUROC 按协议返回 `single_class`，
  不伪造数值。
- official protocol：`protocol_20260717_click_bell_official_smoke` 冻结 Git HEAD `c04ec2f`。
  第一次 attempt 误用 `/root/miniconda3/bin/python`，因缺少 `sapien` 在 setup probe fail-fast；
  该失败记录完整保留。随后用 `/root/autodl-tmp/conda/envs/RoboTwin/bin/python` 对同一
  repetition append-only 重试。最终 `status=completed`、`valid_for_comparison=true`、coverage
  `1.0`，实际 seed 为 `100402`；ACT 成功 `0/1`，400 policy steps、10206 physics steps，
  rollout wall time 约 71.0 秒，完整 Agent wall time约 181.1 秒。Execution VQA 以 0.96 置信度
  判断未观察到按铃，与 official success 一致。两次 attempt 总 wall time 约 195.4 秒；报告保留
  第一次 `taskgen_or_execution` 环境失败计数。协议入口随后增加 `sapien` runtime fail-fast，
  运行指引补充服务器正确 Python 路径，避免再次浪费仿真时间。
