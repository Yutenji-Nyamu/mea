# 2026-07-22 动态 sub-aspect、MetricSpec runtime 与受限恢复

本批按论文主体由上向下修正优先级：最重要的缺口不是继续扩跨任务 graph，而是让同一次固定
task/checkpoint evaluation 的 Planner 在真实 evidence 到来后，动态发现或选择下一 sub-aspect。

## 1. 本批实现

- 新增 `AdaptivePlanStepAgent`：输入原 Query、固定 task/checkpoint scope、coverage、Rule/VQA 与
  `EvidencePacket`，严格输出 `propose`、`refine` 或 `stop`。
- `BoundTaskPlanSession` 区分 initial required、covered、discoverable；初始要求未覆盖时不能提前停止，
  provider 不可用时走确定性有界 fallback，不凭空扩大 rollout。
- 正常 adaptive Agent runtime 在每一真实 round 后调用动态 step；prompt、候选 navigation options、
  原始响应和裁决均落盘。registered/fixed/legacy 路径保持原语义。
- `ToolProposal v3` 携带的 typed `MetricSpec v1` 接入正常 Proposal → request → compile/validate → registry →
  reuse 数据流；支持一条或更多不同真实 episode，以 deterministic rerun + trusted interpreter
  differential 校验，不再错误要求 live 值必须彼此不同。
- 增加 BBH `object_scale.bounded_1_2`，使开放 Query 可物化 1.2× 受限变化，并沿用 render、vision、
  expert gate；本批不为它启动 ACT。
- 错误 `SuccessSpec` 现在生成结构化 diagnosis，并最多一次替换为可信 official-equivalent 默认合同；
  不把非法字段静默合并进结果。
- 增加 `hammer_left_camera_contact_count`：只统计 `020_hammer ↔ left_camera` 的精确 physical contact，
  并把该边界写进 Tool result/VQA/feedback。其 canonical aspect 是
  `safety.hammer_left_camera_contact`；通用 `safety.unintended_contact` 仍明确 unsupported。
- `EvaluationGraph` 增加 required/covered aspect 与跨 child 汇总，但明确标注
  `cross_checkpoint_portfolio`；它是可选父层，不冒充一个 ACT policy 的跨任务评估。
- 方法覆盖审计改为检查动态 step 是否真的在 runtime loop 内消费 evidence，以及 bounded Proposal 是否
  每轮应用，而不是只检查类名或静态文件存在。

## 2. 最小验收与预算

- 新 ACT rollout：**0**。
- scale 验收只允许 setup/render、vision 与 official expert solvability gate；expert 不计 ACT，也不能
  用作 policy 性能。服务器 `run_20260722_batch14_bbh_scale_codegen_v1` 已为
  `completed_without_act`：render/VLM 通过（confidence 0.93），0 次 scene repair，expert gate 通过。
- safety/MetricSpec 优先消费既有真实 BBH telemetry；它是 cached-real smoke，不是新样本。最终 artifact
  为 `mea/validation_runs/batch15_safety_success_recovery_v1/summary.json`：camera contact=1，首次
  `typed_metric_spec_compile`，问句改写后 `run_local_reuse`，provider=0、new ACT=0。
- invalid SuccessSpec 使用 synthetic fixture 检查 diagnosis → bounded trusted fallback → compile；
  同一 artifact 记录 invalid threshold → 2 次 attempt → `trusted_default` → compile，new ACT=0。
- 定向测试：本地 103/103 通过；另 3 个 TaskGen 测试只因轻量本地克隆没有 RoboTwin hammer asset，
  随后由完整服务器 suite 覆盖。
- 服务器完整测试：497/497，51.364 秒。
- 论文方法 coverage：`mea/validation_runs/batch15_method_coverage_v1/report.json` 为
  16 implemented / 0 partial / 0 evidence_pending；这是接口/机制审计，不是论文规模结果。
- 最终版本：随本日志所在 Git 提交发布；用 `git rev-parse HEAD` 获取，避免文件自指 commit SHA。

所有 checkpoint、RoboTwin 资产、render、expert 与 telemetry 均留在服务器；本批没有让模型权重或
rollout 大文件经过 Windows/Codex 工作区。

## 3. 论文对应与诚实限制

| 本批能力 | 论文对应 | 当前能声称 | 仍不能声称 |
| --- | --- | --- | --- |
| evidence-conditioned 动态 sub-aspect | Sec. 3.2；Figs. 2/5 | 公共 runtime/source 与离线分支具备 | 最新实现尚需 clean-head live N=1/轮 |
| scale runnable variation + gates | Sec. 3.3.1；Fig. 3 | 受限 scene variation 可走生成/gate | 未做 scale ACT，不能声称 policy scale 泛化 |
| SuccessSpec diagnosis/repair | Fig. 3；App. A.3.4 | 非法候选 fail closed 后可有界恢复 | repair 是 trusted fallback，不是 Proposal-derived semantic repair |
| typed Rule Tool runtime | Sec. 3.3.2；Fig. 4 | Proposal 中新 metric 可严格编译、验证、注册与复用 | operator/signals 受限；camera proxy 不等于 safety |
| graph coverage/synthesis | Fig. 2 的可选多 child 用法 | 多 checkpoint child 可显式汇总 | 不是论文核心单任务动态规划，也不是单 checkpoint 跨任务 |

更完整的 claim/gap 排序见 [论文主张与当前差距](paper_claim_gap_zh.md)。

## 4. 下一批建议

1. 统一 TaskGen reuse-first resolver，0 ACT。
2. 做受控、Proposal-derived `SuccessSpec v2`，0 ACT。
3. 用当前动态 runtime 做旗舰 adaptive 与 matched fixed N=1；先 2 条 ACT，必要时再扩到 4。
4. 再统一一次 TaskGenerationAttempt 的 generation/vision/success/expert recovery。
5. 核心链稳定后才补独立人工 gold、真实扰动与 N=3。
