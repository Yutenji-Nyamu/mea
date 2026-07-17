# 2026-07-17：generated protocol、Capability、ToolGen 与小型验证

本批目标是以最小成本同时补齐论文主链的五个接口：generated 多变体协议、跨任务 TaskGen
合同、`click_bell` aspect-driven ToolGen、ACT 三任务计量 pilot，以及 Planner/VQA 小型验证
数据通路。本文区分真实运行证据、可运行入口和论文资格；当前没有任何结果可直接替代论文表。

## 1. 论文对应与完成边界

| 本批能力 | 论文对应 | 本批完成 | 仍缺什么 |
| --- | --- | --- | --- |
| generated protocol v2 | Sec. 3.2、Fig. 5、Tables 1–2 的实验基础 | `(variant_id, seed)` 身份、逐变体 coverage/成功/时间、chunk/resume | 3/5 repetition、更多 generated 轴；正式论文预算 |
| shared Capability + `VariantSpec` v2 | Sec. 3.3.1、Fig. 3 | BBH 与 `click_bell` 共用受控轴/generation/preserve envelope | 更多任务族、开放式检索与生成质量实验 |
| `click_bell` aspect-driven ToolGen | Sec. 3.3.2、Fig. 4 | 新连续 metric；generate→validate→register→reuse | 更多 metric/task；跨 evaluation 审核后晋升机制 |
| ACT 三任务 N=1 pilot | Tables 1–2 | 同 task/seed 的 direct 与完整 Agent 计量聚合 | N=1 无方差；尚无论文式结论一致性 |
| 20-query 草稿 | Table 6 | 固定格式、能力覆盖和人工 review 前置校验 | 四人标注/多数票与 live Planner 预测 |
| cached montage image-proxy | Tables 7–8 | 四种图像条件、hash、按扰动聚合入口 | simulator-level 扰动、人工标签与足量正负样本 |

## 2. generated protocol v2

`scripts/manipeval_protocol.py` 新增 `position_lr` profile，`mea/protocol.py` 不再用裸 seed 合并
generated 两轮。每个 repetition 冻结精确预期身份：

```text
(object_position.left_fixed, 100401)
(object_position.right_fixed, 100401)
```

同一 numeric seed 跨 variant 复用是为了控制 policy/实例随机性，只作为 raw-seed duplicate
诊断；缺失、额外或重复的复合身份才会使 `valid_for_comparison=false`。报告新增逐变体
requested/observed、coverage、success、policy/physics steps、simulation time 和 rollout wall。
official profile 继续兼容 v1 seed-only artifact；generated profile 写 v2 manifest/summary。

真实 N=1 运行 `protocol_20260717_click_bell_position_lr_smoke_v2` 最终满足协议：两种 variant
均观察到 1/1 个样本，left 失败、right 成功，pooled success 为 1/2；left/right 分别记录
400/73 policy steps、10589/5005 physics steps、约 71.27/20.89 秒 rollout wall，完整 Agent
wall 约 530.79 秒。它证明复合身份和逐变体数据流在真实 RoboTwin + ACT + VQA + Feedback
中生效，只是 N=1 smoke，不能推断位置泛化率。

## 3. shared Capability 与 `VariantSpec` v2

新增 `mea/taskgen/capabilities.py`，把生成权限与 telemetry `TaskSchema` 分离：

- BBH：`object_appearance.color`，允许受限 codegen/reuse；
- `click_bell`：`object_position.fixed_xy` 与 `object_instance.official_id`，只允许 bounded overlay；
- catalog 注入 `controlled_axis`、generation mode、默认 metric 和 `preserve`；
- `VariantSpec` v2 强制记录 `task_name`、`variant_id`、`capability_id`、intent、changes 和上述
  受信字段；旧两类 spec 只在读取时升级。

这减少了 BBH/`click_bell` 各自定义隐式合同的重复，但还没有实现论文中的任意资产/任务生成。

## 4. `click_bell` aspect-driven ToolGen

新增 target `bell_active_tcp_min_xy_error`。oracle 根据初始 bell x 选择 active arm，计算该 TCP
到 bell contact point 的全轨迹最小 XY 距离，返回单位 m、证据 physics step、仿真时间和 arm；
它没有被人为转成 pass/fail，因此 `passed=null`。Tool router 依据 task metadata 限定 target，
不再把 composite target 全部硬编码为 BBH。

缓存真实 telemetry smoke `eval_20260717_click_bell_toolgen_cached_smoke_v2` 已完成：第一轮
`force_codegen` 调用 provider，候选通过静态、schema、oracle、determinism 与 integrity 检查并
注册；第二轮 `run_local_reuse` 未调用 provider，复用同一注册工具。summary 明确写
`generate_validate_register_reuse=true`。两轮都实际读取 ACT/expert telemetry，并产生不同的
连续距离值；这证明不是固定常量或只改路由标签。

开发中保留了失败记录：早期候选分别触发不受支持的 `np.sqrt`、错误的
`trajectory.semantic_trace`、安全常量 `np.inf` 和 `np.nanargmin`。修复方式是补最小纯数值
allowlist、收紧 prompt 只允许真实 `trajectory.trace` contract，并建议受支持的向量化写法；
没有跳过验证或把失败候选注册为可信工具。

## 5. ACT 三任务 N=1 pilot

`configs/manipeval/act_three_task_n1.json` 固定三个身份：

| task | seed | 已有 direct official ACT 输入 |
| --- | ---: | --- |
| `adjust_bottle` | 100201 | success，117 policy steps，约 35.01 秒 rollout wall |
| `grab_roller` | 100300 | success，93 policy steps，约 29.90 秒 rollout wall |
| `click_bell` | 100401 | success，70 policy steps，约 20.42 秒 rollout wall |

`scripts/manipeval_benchmark_pilot.py` 还要求对应完整 Agent official protocol 都存在且有效；若
任一路线 identity、protocol validity 或 artifact 不完整就拒绝输出。最终 summary 对 direct 与
Agent 分别累计 success、policy/physics steps 和 wall-clock，并报告同 task/seed binary agreement。
它固定标记 `claim_scope=instrumentation_smoke_not_paper_tables`、
`paper_table_eligible=false`、`table2_consistency=null`。配置中的 direct 数据来自既有真实
eval-mode artifact，但 N=1 没有方差，不能据此声称复现 Tables 1–2。

真实聚合 `act_three_task_n1_20260717` 已跑通：direct 与完整 Agent 两条路线都是 3/3 成功、
280 policy steps、23771 physics steps；三组同 `(task, seed)` 的 binary outcome 与 policy steps
均精确一致。两条路线的累计 rollout wall 分别约 85.32 秒和 85.89 秒；完整 Agent 外层 process
wall 约 403.74 秒。direct 旧 artifact 没有可比的外层 process wall，因此这里只验证计量与结果
一致性，不用不完整 wall-clock 作效率结论。

## 6. 20-query 与 VQA image-proxy

`configs/manipeval_validation/query_aspects_draft_v1.json` 有且仅有 20 条开放 query，当前 capability
支持 5 条、未支持 15 条。校验器强制每条 annotation 都是
`model_draft/unreviewed/[]`，所以 `human_agent_agreement=null`、`paper_table_eligible=false`。
这个文件的用途是导出给人工 review，而不是提前制造 Table 6 gold label。

`scripts/manipeval_vqa_perturb.py` 读取 suite 明确列出的既有真实 Execution VQA artifact；每个
source montage 确定性产生：

1. `clean`；
2. `scene_clutter_image_proxy`；
3. `background_texture_image_proxy`；
4. `lighting_image_proxy`。

runner 保存 source/artifact/query/numeric evidence 与 transform hash，并用现有 VQA contract
重跑四个条件。预算 1 / 3 / 5 对应 4 / 12 / 20 次视觉调用。该入口不启动 simulator，且 gold
来自 simulator proxy，因此即使得到 accuracy/AUROC，也只能称为缓存图像代理 smoke，不能
声称复现 Tables 7–8 的真实 scene perturbation 或 human-label 指标。

真实执行先跑 budget=1，再升到 budget=3。budget=1 的 4/4 条件均有有效 schema、完整覆盖且
proxy 标签正确，但只有负类，所以 AUROC 正确返回 unavailable/single_class。budget=3 共 3 个
source clips × 4 条件 = 12 次视觉调用，artifact failure 为 0，schema/coverage/strict accuracy
均为 1.0，proxy precision 与 AUROC 也为 1.0。这个数值只描述 3 个缓存 clip 和图像后处理代理，
`paper_table_eligible=false` 仍保持不变。

## 7. 运行与资产约束

- 新入口和最短命令见 [MEA 简明运行指引](running_guide_zh.md)。开发一律先用预算 1，再按
  风险与趋势放大到 3、5；本批仍只评 ACT，不接第二种 policy。
- `click_bell` 的所有 position/instance variant 复用同一个任务 checkpoint，不为每个 variant
  下载权重；三个 pilot 任务各需要自己的 ACT checkpoint。
- checkpoint、数据集和其他大文件只在服务器通过 AutoDL 学术加速或服务器侧 HF mirror
  获取，不经过 Windows、`C:` 或 Codex 工作区；本批文档和代码不包含凭据或模型权重。
- generated protocol、ToolGen、pilot、query 和 perturbation 都写机器可读 artifact。总结时以
  artifact 的 validity/claim-scope 字段为准，不能只摘取自然语言报告中的成功描述。

## 8. 当前结论

本批把“多个 generated round 能跑”推进到“每个 variant/seed 可审计”，把 TaskGen 的任务专属
隐式约定推进到共享 capability envelope，并让 `click_bell` 首次以 aspect 驱动一个真实生成、
验证和复用的数值 Tool。三任务、query 和 VQA 的低成本框架也已具备，但它们当前分别是 N=1
instrumentation、unreviewed draft 和 image proxy。下一步若要形成论文级证据，应优先增加
simulator-level 扰动与人工标签，再把稳定协议从 1 放大到 3、5，而不是把这些 smoke 直接包装
成论文表。
