# 2026-07-19：公共 PlanSession、主链新 Proposal 与 run-local VQA

本批只补论文主体数据流，不扩大 policy、任务数量或重复实验。对应论文 Fig. 2、Sec. 3.2 的
证据驱动多轮规划，Fig. 3 / Sec. 3.3.1 的 Proposal→TaskGen，以及 Fig. 4 / Sec. 3.3.2、
App. A.3.5 的 Tool/VQA 问题流。开发预算固定为 0 ACT。

## 已实现

1. `BoundTaskPlanSession` 新增 `directive()` 与 `adjudicate()`。公共 evidence policy 唯一决定
   `action / transition / next_aspect / next_template`；task adapter 只提交物化候选和解释。
   session 会拒绝 task、checkpoint、预算、requested aspects/templates、历史轮次或下一 template
   被改写。主 Agent 的 adaptive click_bell 路径已经接入该最终裁决。
2. 主 Agent 新增显式 `--proposal-mode novel_first_round`。开放 Query 路由并绑定 task/checkpoint
   后，`BoundedProposalAgent` 为首轮生成不精确重复注册 template 的 TaskProposal/ToolProposal。
   注册 capability contract 只作为 executable materializer envelope；本轮 changes、variant id 和
   intent 由 TaskProposal 授权。
3. `scripts/manipeval_taskgen.py` 现在同时校验 capability envelope 与 TaskProposal，并把
   `--task-proposal-json` 真正传进 TaskGen。manifest 区分
   `planner_capability_contract` 与 `planner_task_proposal`，仍不允许 Proposal 改 path、gate、
   success semantics 或 change roots。
4. `ToolProposal` 增加向后兼容 v2，可携带 `run_local.*` VQA question spec。问题 ID、字段、枚举、
   单行问句与长度均受限；保存后的 `execution_vqa_query.json` 可独立重验。run-local VQA 只作视觉
   补充，不进入 numeric guard，也不能覆盖 simulator Tool。

## 真实验证

- 定向单测：49/49；完整 `tests/manipeval`：423/423，42.2 秒。
- live plan-only：
  `eval_20260719_batch12_novel_plan_smoke_v3`。Router 选择 click_bell 的 position + instance；
  Proposal 产生新位置 `[-0.14, -0.12]`，而注册左右位置仍是 `[-0.20, -0.08]` 与
  `[0.20, -0.08]`。首轮计划同时保存 TaskProposal、ToolProposal v2 与 run-local question。
- 真实 simulator setup/render：
  `run_20260719_batch12_novel_taskgen_smoke`，状态 `completed_without_act`；新位置实际出现在
  scene actor state，`setup_success=true`、`render_success=true`、capability binding passed，
  `variant_spec_authority=planner_task_proposal`，provider 未调用，ACT 未启动。
- VQA query smoke：
  `eval_20260719_batch12_novel_plan_smoke_v2/plan/bounded_proposal/execution_vqa_query_smoke.json`；
  catalog 问题与模型生成的 run-local 问题共同通过自包含验证。
- counterfactual replay：
  `mea/validation_runs/batch12_common_plan_session_counterfactual_v2.json`；固定同一真实 round-1
  非 policy 证据时，`policy_success=0` 得到 position `drill_down`，`policy_success=1` 得到
  `switch_aspect→object_instance`。该 replay 启动 0 ACT，只证明控制流会响应不同证据。

## 当前边界

- 新 Proposal 的主链 materialization 第一版只开放 `click_bell/object_position`；其他 capability
  仍走 catalog template。
- 公共 adjudication 已用于 adaptive click_bell；BBH 的 legacy `verify` 语义和 fixed baseline
  暂保留原 adapter 路径，避免本批改变历史协议。
- run-local VQA 问题由模型生成且未人工审核；本批只验证 query/schema/response plumbing，未对该
  问题重新支付 ACT 或 VQA rollout。
- ToolProposal 仍必须落在已登记 metric family；尚未实现“从 TaskGen code 推导全新 MetricSpec”。
- 这些结果是功能 plumbing 与 0-ACT smoke，不是成功率、VQA accuracy/AUROC 或论文表格证据。
