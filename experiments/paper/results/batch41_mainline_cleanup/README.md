# Batch41：生产主链裁剪

本目录记录一次结构清理，不替换 `docs/evidence/current/`，也不增加论文方法或实验 claim。
目标是让普通运行重新清楚呈现：

```text
Query → Plan/Proposal → Task/Tool → rollout
→ Rule/VQA → evidence → Plan continue/stop → Answer
```

## 实际裁剪

- production CLI/Application/RoundExecutor 不再运输 QueryContract、evidence manifest、
  command plan、registered strategy、reviewed Task/Tool/VQA registry、receipt 或 resume 状态。
- TaskGen attempts 收成 `generation → validate → optional targeted repair → result`；只写一份
  `task_generation_result.json`，不再写 append-only started/result/summary 或 Proposal/module hash gate。
- Task、Rule Tool 与 VQA 的新生成物使用可读 semantic key 检索，并在当前 simulator、telemetry
  或 frames 上重新验证；审批、promotion 与 hash-pinned reuse 已退出普通生产。
- formal QueryContract 从量词、候选宇宙和真值状态机缩成小型 runtime limits 与后置 stop
  validation；Plan Agent 读取 evidence 后自己提出 continue/stop。
- preservation 由 typed `{actor, property, axis, relation}` 事实传输；未知旧 prose 返回
  `unverified`，不再靠字符串正则硬裁决。
- Query interpretation 不再用 token、alias、ontology 或 catalog ranking 授权 concern；typed
  needs 直接进入 Proposal reuse-or-generate。Plan session 中无人消费的 retrieval-aspect/template
  映射同时删除。
- production telemetry 不再写 profile checksum；native run id 改为可读
  `run_native_<backend>_<evaluation>_<round>`。
- caller-zero acceptance、cached finalization、resume、registered strategy 与 reviewed registry
  代码删除；论文 preregistration、旧 receipt/hash 与 validation protocols 只保留在 cold
  `experiments/paper/`。

普通 `manifest.json` 没有被误删：它只是可读 lifecycle/config/path index。TaskGen 的
simulator/checker fixture、render/VLM、expert，Rule Tool oracle 和 VQA abstain/conflict 也保留，
因为它们保护实际执行与 simulator authority，而不是形成第二套审批系统。

## AutoDL 验证

- base commit：`1d567a92292067e9ab08cd5ad536e02c7ed9b8f5`；
- Python compileall：通过；
- focused 收尾回归：65 passed，另 3 subtests；
- 默认 mainline：225 passed，另 10 subtests；
- cold/compat：623 passed，另 162 subtests；
- 3 条 warning 均来自 robosuite 上游弃用接口；
- Windows PC 未运行测试、import、provider、simulator 或 policy；
- 本批 0 provider call、0 simulator call、0 policy rollout。

## 边界与下一步

这批证明的是清理后代码仍可运行，不是论文方法正例。仍有四个明确债务：旧
`artifact_retrieval_index` 仍提供菜单式 task/VQA hint；`ExperimentCandidate` 仍是内部 Proposal
transport 且自动 identity 含技术摘要；默认 HistoryDB 仍用摘要做幂等；`RoundExecutor`、TaskGen
runtime/generic backend 仍应按 owner 继续拆小。candidate/history 摘要不参与 evidence、审批或执行
许可，但应在独立迁移中改成显式 round identity 与直接 JSON 比较，避免在本批尾部冒险改数据库。

下一轮方法工作按顺序进行：选择第二、第三个任务走同一主链；其中至少一个不带任务卡正锚点，
验证 concern 冷发现；随后验证轻量 Tool library 的跨 evaluation exact reuse；最后单独处理冻结 VQA
输入的重复稳定性。真实失败优先进入 owning Agent prompt 或短任务卡，只有 simulator authority
无法由提示保证时才增加最小代码边界。
