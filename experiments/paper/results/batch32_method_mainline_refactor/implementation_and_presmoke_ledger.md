# Batch 32 method-mainline refactor ledger

本文件是冷归档；Hot 架构与 claim/gap 文档只保留稳定结论。这里记录实现、服务器
预测试和后续 flagship smoke 的可复现流水，不保存服务器密码、provider key 或 token。

## 1. 范围与执行边界

- 目标：真实迁移生产 caller，而不是继续增加 façade。
- 方法范围：Plan Agent application、唯一 TaskGen materialization owner、Plan session
  ownership，以及一次真实失败驱动的 Planner prompt 修正。
- Windows PC：仅代码、文档、diff 与 Git；未运行 test/import/compile/provider/simulator。
- AutoDL server：所有验证均在 canonical clean worktree 执行。
- 本批不在未审批前启动 provider、SmolVLA policy server、simulator 或 rollout。

## 2. 起始真值与远程恢复

| 检查 | 结果 |
| --- | --- |
| Windows worktree | clean `01aceed97f326eb9be195c5fc63a2d7b56cc80d2` |
| 首个已授权实例 | 密码认证与 `hostname; pwd; id -u` 成功，但有限深度内无 MEA Git repo |
| MEA 实例 | 密码认证与身份探针成功 |
| canonical server worktree | `/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime` |
| canonical server HEAD | clean detached `01aceed97f326eb9be195c5fc63a2d7b56cc80d2` |
| 禁止使用目录 | `/root/autodl-tmp/mea`：旧 HEAD 且严重 dirty |

恢复问题与处理：

1. MEA helper 最初固定的是另一网关指纹，因此新端点在认证前被 host-key gate 拒绝；
   这不是密码失败。
2. 从既有、已验证的 AutoDL helper 找到对应固定指纹后，低层
   `Paramiko.Transport.start_client()` + `auth_password()` 身份探针成功。
3. 不跳过 host-key 校验，也不在没有 MEA 环境的实例上临时重建仓库。

## 3. 实现变更

### 3.1 Plan Agent application

- 新建 `mea/plan_agent_application.py`。
- 生产 caller 现在直接执行
  `PlanAgentApplication.run()`：
  `RoundExecutor → evidence → Plan Agent continue/stop → Aggregate → Answer
  → history index`。
- 从 `scripts/manipeval_agent.py` 删除对应的重复生产生命周期；兼容 planner 路径暂时
  保留。
- `update_manifest` 与 hard-round-cap 持久化 owner 一并迁出。
- `scripts/manipeval_agent.py` 由约 4,986 行降到约 4,449 物理行；`main()` 约减少
  414 行。

### 3.2 唯一 TaskGen materialization owner

- 生产 native round 不再“先完整 TaskGen，再从外部绑定 manifest”。
- 唯一生产链改为：

  ```text
  MethodRuntime.materialize_candidate()
  → RoboTwinMethodBackend.materialize_candidate()
  → provider TaskGen
  → accepted-artifact semantic gates
  → rollout candidate
  ```

- `bind_validated_taskgen_candidate()` 只保留冻结 artifact/兼容重放入口。
- TaskGen failure manifest 记录下沉到 backend。
- AST、fixture、render/VLM、expert、preservation、checker semantic review 与
  TaskContext gate 均未删除。

### 3.3 Plan session ownership

- `PlanAgentSession` 成为唯一公开生产 owner。
- 原公开 `PlanAgentExecutionSession` 收缩为内部 `_FrozenExecutionTransport`。
- 历史别名只在原模块保留 reader/fixture 兼容，不再从 `mea.planner` 公共 API 导出。
- 序列化字段未改变。

### 3.4 失败驱动 prompt 修正

v11 曾把“目标同时接触两个对应夹爪点”弱化为“两个夹爪闭合”。本批不增加新的
任务分支，而是在 Plan Agent 公共 prompt 与 `README.Agent.md` 中加入可迁移规则：

- 只有 current-state simulator API 能直接表达新增关系时才请求 checker；
- gripper closure 不是 target contact；
- sequential event 不是 simultaneous relation；
- height 不是 placement；
- exact relation 不可用时，选择 scene-only + Rule/VQA observation 或另一个
  informative sub-aspect，不要求 TaskGen 生成相关代理。

## 4. 服务器预测试

待完成后填写；所有命令从 canonical server worktree 执行。

```bash
python -m pytest -q \
  tests/mainline/test_production_cli_boundary.py \
  tests/mainline/test_round_executor_boundary.py \
  tests/mainline/test_claim_first_runtime.py \
  tests/mainline/test_robotwin_method_runtime.py \
  tests/mainline/test_policy_backends.py \
  tests/manipeval/test_claim_first_open_query.py \
  tests/manipeval/test_open_world_session.py \
  tests/manipeval/test_runtime_task_binding.py

python -m pytest -q tests/mainline
```

结果：`PENDING`

## 5. Formal flagship smoke gate

本节在预测试通过后补齐 resolved command。启动前必须向用户提交并等待确认：

- 原始 broad Query；
- policy/backend、bound task、checkpoint revision；
- seed、最大 1–3 rollout、最大轮数；
- provider model profile；
- policy server 与 Agent 完整命令；
- 输出目录、GPU/时间/provider 预算；
- 逐阶段监控点和立即停止条件。

未获得确认前，不启动 provider、policy server、simulator 或 rollout。
