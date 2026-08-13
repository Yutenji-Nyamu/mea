# Paper experiment layer

本目录只保存论文复现实验的显式协议、冻结输入和历史结果。它不是 MEA 生产主链：

- 生产入口是 [`scripts/manipeval_agent.py`](../../scripts/manipeval_agent.py)。
- 当前方法证据入口是
  [`docs/evidence/current/README.md`](../../docs/evidence/current/README.md)。
- 论文 claim 的当前判断只以
  [`docs/paper_claim_gap_zh.md`](../../docs/paper_claim_gap_zh.md) 为准。

除非正在复跑相应论文协议，日常理解 Planner、TaskGen、ToolGen 或 runtime 时不应递归
读取本目录的 `inputs/` 与 `results/`。

## 1. Active paper protocols

这些入口仍可用于有明确目的的消融、审计或离线重放，但不会被生产 CLI 隐式调用。

| 文件 | 用途 | 是否启动真实执行 |
| --- | --- | --- |
| [`live_protocols.py`](live_protocols.py) | fail-closed preregistration 与 receipt evaluator | 否 |
| [`manipeval_run_live_paper_protocols.py`](manipeval_run_live_paper_protocols.py) | 显式执行已冻结的 paper-only live command | 可能；仅显式调用 |
| [`manipeval_paper_evidence_protocols.py`](manipeval_paper_evidence_protocols.py) | 建立和审计论文证据协议 | 否 |
| [`manipeval_plan_strategy_pair.py`](manipeval_plan_strategy_pair.py) | 生成 fixed/adaptive 成对命令计划 | 否 |
| [`manipeval_annotation_replacement.py`](manipeval_annotation_replacement.py) | 用可替换人工标注重算 Plan/VQA 分数 | 否 |
| [`manipeval_independent_validity.py`](manipeval_independent_validity.py) | 聚合多标注者与 VQA control | 否 |
| [`manipeval_execution_vqa_replay.py`](manipeval_execution_vqa_replay.py) | 对完成 rollout 做有界 VQA replay | 否 |
| [`manipeval_replay_completed_tool.py`](manipeval_replay_completed_tool.py) | 对完成 telemetry 做 Tool/Planner repair replay | 否 |
| [`manipeval_evidence_manifest.py`](manipeval_evidence_manifest.py) | 建立或检查 hash-pinned evidence manifest | 否 |
| [`manipeval_portfolio.py`](manipeval_portfolio.py) | 规划或审计历史 cross-task portfolio | 否 |
| [`libero_adapter_smoke.py`](libero_adapter_smoke.py) | LIBERO/SmolVLA adapter feasibility smoke | 显式调用时可能 |
| [`generic_method_matrix.py`](generic_method_matrix.py) | 只读聚合冻结的跨任务方法 bundle，并分离方法、policy 与 Answer 状态 | 否 |
| [`robotwin_breadth.py`](robotwin_breadth.py) | 动态发现 RoboTwin task；可续跑地分开 TaskContext/Plan 预检、TaskGen materialization 与 SmolVLA official N=1 | 仅 `official` phase |

X-VLA installation commands and unresolved blockers are kept in the cold
[`robotwin_xvla/deployment_ledger.md`](robotwin_xvla/deployment_ledger.md).
The bounded failure-to-prompt workflow and its active vertical cases live in
[`prompt_learning/README.md`](prompt_learning/README.md); task queues do not
belong in the production README or hot architecture document.

部署与 prompt 迭代均为按需读取的 cold reference：

| 主题 | 索引 |
| --- | --- |
| SmolVLA | [`robotwin_smolvla/`](robotwin_smolvla/README.md)；[短复现说明](../../docs/robotwin_smolvla_reproduction_zh.md) |
| Hy-VLA | [`robotwin_hyvla/`](robotwin_hyvla/README.md)；[短复现说明](../../docs/robotwin_hyvla_reproduction_zh.md) |
| X-VLA | [`robotwin_xvla/`](robotwin_xvla/README.md)；[短复现说明](../../docs/robotwin_xvla_reproduction_zh.md) |
| failure-to-prompt | [`prompt_learning/`](prompt_learning/README.md) |

## 2. Compatibility-only protocols

以下文件保留旧实验的可恢复性，不应成为新功能依赖：

- [`compat_agent_profile.py`](compat_agent_profile.py)：集中解析生产 CLI
  延迟加载的 catalog、fixed、registered 和 task-specific 兼容参数。
- [`legacy_planner_factory.py`](legacy_planner_factory.py)：显式加载 catalog、fixed-suite
  和 task-specific legacy Planner。
- [`registered_execution_adapter.py`](registered_execution_adapter.py)：在 paper protocol
  内延迟加载旧 strategy/receipt stack。
- [`compat_taskgen/`](compat_taskgen/)：冻结 standalone TaskGen 的 BBH、ClickBell、
  registered/reviewed 与 Table-3 兼容执行；生产 Agent 的 generic TaskGen 由
  `mea.taskgen.runtime` / `MethodRuntime` 直接拥有。
- [`manipeval_click_bell_open_taskgen.py`](manipeval_click_bell_open_taskgen.py)：历史
  ClickBell Gate-0 TaskGen 协议。
- [`manipeval_click_bell_open_evidence.py`](manipeval_click_bell_open_evidence.py)：历史
  Batch25 evidence assembly。

删除这些兼容文件之前，必须先确认生产代码、paper protocol 和测试均已没有 caller。

## 3. Frozen inputs

`inputs/` 中的内容是可审计实验输入，不是运行时 registry，也不应随最新结论自动改写。

| 路径 | 内容 |
| --- | --- |
| [`inputs/batch24_human_replaceable/`](inputs/batch24_human_replaceable/) | Plan/VQA blind packets、proxy annotations、predictions 与八张 review 图 |
| [`inputs/table3_proposal_sets/`](inputs/table3_proposal_sets/) | Table 3 unseen Proposal 集合 |
| [`inputs/query_dataset_proxy20_zh.json`](inputs/query_dataset_proxy20_zh.json) | 早期 20 条 Query proxy |
| [`inputs/vqa_proxy_suite_batch23.json`](inputs/vqa_proxy_suite_batch23.json) | 早期 VQA proxy suite |
| [`inputs/paper_error_operations_batch23.json`](inputs/paper_error_operations_batch23.json) | 前瞻错误率 pilot 的冻结 operation |
| [`inputs/table3_proxy_review_batch23_v5.json`](inputs/table3_proxy_review_batch23_v5.json) | Batch23 Table 3 proxy review |
| [`inputs/table3_proxy_review_batch24_v1.json`](inputs/table3_proxy_review_batch24_v1.json) | Batch24 Table 3 proxy review |
| [`inputs/batch29_adjust_bottle_terminal_height_tool_request.json`](inputs/batch29_adjust_bottle_terminal_height_tool_request.json) | AdjustBottle terminal Tool repair 请求 |

这些 proxy 文件不能替代论文所要求的独立机器人研究者标注或人工 VQA gold。

## 4. Historical results

`results/` 是冻结历史证据和 regression fixture；最新方法状态不从这里推断。

| 路径 | 主要内容 |
| --- | --- |
| [`results/batch23_claim_closure/`](results/batch23_claim_closure/) | 早期 claim closure、Table 3、Plan/VQA 和系统错误率 proxy |
| [`results/batch24_claim_closure/`](results/batch24_claim_closure/) | Batch24 汇总索引 |
| [`results/batch24_human_replaceable/`](results/batch24_human_replaceable/) | 可替换标注协议结果 |
| [`results/batch24_libero_method_chain_v2/`](results/batch24_libero_method_chain_v2/) | LIBERO 两回合 method-chain 历史 bundle |
| [`results/batch25_bound_click_open_plan_v2/`](results/batch25_bound_click_open_plan_v2/) | ClickBell open-plan 与 deterministic repair |
| [`results/batch25_click_bell_open_taskgen_v2/`](results/batch25_click_bell_open_taskgen_v2/) | ClickBell provider TaskGen 历史 bundle |
| [`results/batch26_claim_closure/`](results/batch26_claim_closure/) | Batch26 claim/Table3/VQA/LIBERO 汇总 |
| [`results/batch27_unified_adapter_libero/`](results/batch27_unified_adapter_libero/) | unified adapter、LIBERO 和 PlacePhoneStand 历史结果 |
| [`results/batch24_click_bell_conclusion_fidelity_n3.json`](results/batch24_click_bell_conclusion_fidelity_n3.json) | 三 seed conclusion-fidelity toy |
| [`results/batch24_table3_scene_checker_unseen5_v1.json`](results/batch24_table3_scene_checker_unseen5_v1.json) | 五 Proposal Table 3 toy |
| [`results/batch30_smolvla_native_runtime.json`](results/batch30_smolvla_native_runtime.json) | 50-task manifest、两个新任务 policy failure 与原生 MethodRuntime smoke |
| [`results/batch31_smolvla_plan_agent_n1.json`](results/batch31_smolvla_plan_agent_n1.json) | click_bell N=1 生产 Plan Agent rollout 的原 pipeline failure 与 0-rollout append-only 重投影 |
| [`results/batch30_open_python_toolgen_live/`](results/batch30_open_python_toolgen_live/) | 缓存真实 telemetry 上的 provider Python ToolGen、一次 repair 与 exact reuse |
| [`results/batch32_method_mainline_refactor/`](results/batch32_method_mainline_refactor/implementation_and_run_ledger.md) | Plan Agent application、唯一 TaskGen materialization owner、v18 方法运行与复用审计 |
| [`results/batch33_open_cross_task/`](results/batch33_open_cross_task/README.md) | PressStapler evidence refinement、candidate rejection、SmolVLA 五任务 breadth 与 Hy-VLA N=1 |
| [`results/batch34_task_independent_context/`](results/batch34_task_independent_context/probe_summary.json) | 无 policy 的跨任务 reset TaskContext 与嵌套 actor 发现 |
| [`results/batch35_generic_method_matrix/`](results/batch35_generic_method_matrix/README.md) | 五任务通用方法矩阵与 schema-free scene/checker live 补充验收 |
| [`results/batch38_prompt_context/`](results/batch38_prompt_context/README.md) | `grab_roller` 失败驱动 prompt/context 纵向回归；未晋升当前旗舰 |
| [`results/batch39_grab_roller_prompt_mainline/`](results/batch39_grab_roller_prompt_mainline/README.md) | prompt-first 修复后的 `grab_roller` 主链、live Tool 与主动 inconclusive stop；未晋升旗舰 |

旧运行的简短结论与边界另见
[`docs/evidence/history.jsonl`](../../docs/evidence/history.jsonl)。若两处描述冲突，
优先使用原始 artifact，并在 `paper_claim_gap_zh.md` 中修正当前判断。

## 5. Cold-context rule

默认代码审查和方法规划只读取本 README，不递归展开：

```text
experiments/paper/inputs/**
experiments/paper/results/**
```

只有以下任务才按索引打开对应子目录：

- 重跑固定消融或 matched comparison；
- 更换人工标注并重算；
- 审计某次历史结果；
- 复现 LIBERO 或 SmolVLA 的指定批次；
- 验证 caller 后删除兼容协议。

该规则只控制上下文范围，不改变 Git 跟踪、证据保留或实验可恢复性。
