# Batch 32 method-mainline implementation and run ledger

本文件是冷归档，记录可复现实现与服务器执行；不保存服务器密码、provider key 或
token。MEA 开发批次在既有授权范围内连续执行，不使用 RL 项目的 smoke 审批门。

## 1. 执行边界与远程恢复

- Windows PC 仅编辑代码/文档、检查 diff 和操作 Git；本批没有在 Windows 运行
  test、import、compile、provider、simulator 或 policy inference。
- 所有验证与 rollout 均在 AutoDL server 的
  `/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime` 执行。
- `/root/autodl-tmp/mea` 是旧且 dirty 的目录，未使用。
- 连接采用固定 host-key 的低层 `Paramiko.Transport.start_client()` +
  `auth_password()`。曾遇到的是认证前 SSH banner/host-key 路径问题，不是密码失败；
  恢复后先执行 `hostname; pwd; id -u` 再继续。

## 2. 主干实现

1. `PlanAgentApplication` 直接拥有
   `route → round → evidence → continue/stop → Aggregate → Answer`。
2. `RoundExecutor` 原生执行 ACT/SmolVLA backend，不再把旧 TaskGen 子进程投影成生产
   round。
3. `MethodRuntime.materialize_candidate()` 是唯一生产 TaskGen materialization owner；
   AST、fixture、render/VLM、expert、preservation 与 checker semantic review 仍在。
4. `PlanAgentSession` 是唯一公开生产 session；旧 execution session 仅保留冻结历史
   reader/fixture 兼容。
5. broad/candidate discovery、typed optional needs、control requirement、generated
   checker 语义保持与 diagnostic-only Tool evidence 均进入公共 Plan Agent。
6. v17 失败例推动公共提示词区分 terminal checker outcome 与 trajectory diagnostic；
   `Define experimental success` 也会在后续轮保留 checker need。
7. publisher 可在不修改原始 evaluation 的前提下发布 append-only Tool reuse 与
   current-code acceptance projection，并同时保留旧 projection。

## 3. v18 真实方法运行

冻结配置与复跑模板：

- [`flagship_run_config.json`](flagship_run_config.json)
- [`flagship_run_commands.sh`](flagship_run_commands.sh)

方法代码 HEAD：

```text
2db7f0abfa803f94dbad0dacf76525ed0c19b454
```

运行位置与日志：

```text
evaluation:
  mea/evaluation_runs/eval_20260731_batch32_clean_flagship_live_v18
logs:
  /root/autodl-tmp/mea-run-logs/batch32_clean_flagship_v18
```

真实结果：

- Round 1：SmolVLA official control，seed `100401`，成功。
- Plan Agent 在 control evidence 后选择 roller 世界 x 轴 `+0.05 m`。
- Round 2：provider scene+checker 通过 AST、`2/2` fixture、render/VLM、expert 与
  preservation；policy official core 成功，实验 checker 失败。
- 新 Python Tool 得到 `0.24384725093841553 m`，VQA 无冲突，Aggregate 通过。
- Plan Agent 主动 stop；QueryContract 为
  `evidence_sufficient / counterexample_found`。
- 总计两个 policy episode、一个唯一 seed；不是统计实验。

## 4. 零 rollout append-only 审计

`repairs/independent_query_tool_reuse_v1` 使用生产 `execute_tool_request()`：

- 独立 follow-up Query 只改写展示问题，保持 task/metric/MetricSpec/schema/unit；
- route 为 `run_local_reuse`，provider 未调用，policy rollout 为 0；
- registration 均为 `runlocal_69223f1181c20689d466`；
- 复用值仍为 `0.24384725093841553 m`；
- registry index、原 Tool execution 与原 summary 字节均未改变；
- 当前 acceptance 重投影为 accepted，并与原运行时旧 projection 一同发布。

## 5. 失败与处理

- v17 把 trajectory peak 误当 terminal outcome，并在下一轮丢失 generated checker。
  处理为公共 prompt/evidence-role 修正与 success-semantics 匹配，不增加任务名分支。
- v18 原 final-summary 多写“缺少轨迹峰值”，而原 Query 只要求一个由 rollout telemetry
  计算的诊断标量。原产物保持不变；`mea/feedback/README.Agent.md` 已加入通用失败例，
  禁止把 Query 的 statistic 擅自升级成 peak/extremum。
- publisher 新测试首次失败仅因提示词断言跨源码换行；改为检查稳定语义片段，
  生产逻辑未变。

## 6. 服务器验证与发布

定向回归：

```bash
PYTHONPATH=. /root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q \
  tests/mainline/test_evidence_report.py \
  tests/mainline/test_answer_scope.py
```

结果：

```text
17 passed in 0.18s
```

最终默认主干命令：

```bash
PYTHONPATH=. /root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q \
  tests/mainline
```

结果：

```text
306 passed, 16 subtests passed in 17.55s
```

日志位于：

```text
/root/autodl-tmp/mea-run-logs/batch32_method_mainline_refactor/
  final_mainline.log
  final_mainline.status
```

所有测试均在 AutoDL server；Windows PC 没有运行项目测试。权威紧凑证据为
[`docs/evidence/current/`](../../../../docs/evidence/current/README.md)。
