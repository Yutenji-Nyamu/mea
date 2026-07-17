# 2026-07-17：开放 Query 核心闭环 Stage 1 开发记录

## 目标与论文对应

本批不继续扩任务数量，而是把论文 Fig. 2 的顶层方法在 RoboTwin + ACT 的受限范围内串成一条
真实链路：

```text
开放 Query
→ 可信 ACT catalog 与全局 route
→ task-specific Planner
→ Task reuse / bounded overlay / true codegen
→ scene、rule、expert gates
→ ACT
→ Trusted 或 reviewed generated Tool + 动态 Execution VQA
→ 通用 evidence-conditioned transition
→ 最终反馈与 completed-only history
```

该范围对应 Sec. 3.2–3.4 与 Figs. 2–5。完成后可称为“两任务族、ACT-only、少量 rollout 的
核心方法受限功能复现”，不能称为论文 Sec. 4 的完整实验复现。

## 主要实现

### 1. 全局开放问题入口

- `mea/planner/catalog.py` 只暴露同时具有 TaskSchema、`dataset_stats.pkl` 和
  `policy_last.ckpt` 的可信 ACT task/profile/aspect/checkpoint id；路径、module、seed 和可执行
  参数不交给模型决定。
- `mea/planner/global_query.py` 用严格结构选择 `route` 或显式 `unsupported`，并把合法选择转换为
  现有 BBH/click_bell proposal。
- `scripts/manipeval_agent.py --auto-route` 先做 completed-only 全局 history retrieval，再做
  task-specific retrieval；路由后的第一轮不重复调用 task Planner。
- `--auto-route` 拒绝 task-specific `--task-module` override；unsupported 路径也使用严格
  `eval_[A-Za-z0-9_]+` 身份，避免 provenance 歧义和路径逃逸。

### 2. 通用证据转移合同

`assess_conditional_transition` 把 task adapter 限制为有序 `aspect → template_ids` catalog，运行时
按可信证据强制选择：

- pipeline failure：停止；
- VQA/数值冲突或 aggregate 不确定：同 aspect 反事实，预算不足时保留 `unresolved=true`；
- policy failure：优先 `drill_down`；
- 清晰成功：优先 `switch_aspect`；
- 无可信 transition 或预算耗尽：停止并明确未覆盖项。

click_bell adaptive Planner 已委托给该通用合同，不再自行复制一套状态机。

### 3. 审核后跨 evaluation Tool 复用

- 新增 reviewed persistent registry 和 `scripts/manipeval_tool_registry.py`。
- generated Tool 只有在显式审核源码、ToolSpec、验证证据和测试后，才以完整
  registration/code/ToolSpec/contract/schema hashes 安装。
- 新进程仍须在当前轨迹上重新跑 determinism 与私有 oracle gate；精确不匹配时回退 codegen，
  `provider_called=false` 只在真实命中时记录。
- reviewed Tool 永不自动进入全局 Trusted Tool catalog；写入侧拒绝 symlinked `entries/`。

### 4. TaskGen 功能验收

`scripts/manipeval_taskgen_acceptance.py` 只读核验四类既有真实 artifact：official reuse、
click_bell bounded overlay、BBH true codegen + retrieval provenance，以及
`static pass → visual reject → diagnosis → repair → static revalidate → visual pass` 错误注入。
验收同时检查 click_bell manifest/VariantSpec 身份；固定标注 cached、no-provider、no-simulator、
no-ACT、not-paper-eligible。

### 5. 项目可读性与运行稳定性

- 项目手册明确 Fig. 2 的“功能上完整”门槛、可信边界、论文映射与 1→3→5 优先级。
- 架构文档加入全局 route、通用 transition、reviewed registry 和 TaskGen acceptance 数据流。
- 运行指引加入 open-query、history smoke、registry 审核、acceptance 和服务器端 detached 长实验
  方法；根 `README.md` 保持不动。
- checkpoint、数据集和权重仍只允许服务器直接下载。本批没有下载 checkpoint，也没有大文件
  经过 Windows/Codex；本地 batch staging 约 1.3 MB，无 10 MB 以上文件。

## 真实验证

### 全局 route 与历史检索

- `eval_20260717_global_query_plan_smoke`：同一开放问题一次模型 route 到 `click_bell /`
  `adaptive_properties`，选择 position + instance，解析 checkpoint
  `act-click_bell/demo_clean-50`；`--plan-only`，execution 为空。
- history rebuild 得到 17 条有效 completed records；重复 query 的
  `eval_20260717_global_query_history_repeat_smoke` 选中 3 条 task-specific 历史，最高相似项为
  `eval_20260717_click_bell_adaptive_live_2`，未执行 rollout。

### reviewed Tool 跨进程 smoke

- 安装 id：`reviewed_2ade3a1c3a0de5a48ee6`。
- `eval_20260717_reviewed_tool_cross_process_smoke` 在新进程命中
  `reviewed_persistent_reuse`，ToolGen provider 未调用，审核、hash、determinism 和 oracle gates
  全部通过。
- 对缓存 ACT/expert 轨迹测得 TCP 最小 XY 距离分别为
  `0.01842462830245495 m` 与 `0.0007090155268087983 m`。这是缓存复用 smoke，不是新 ACT 结果。

### TaskGen acceptance

四个缓存切片全部通过，包括 BBH codegen/retrieval provenance 与一次 visual
reject→diagnosis→repair；验收进程没有模型、simulator 或 ACT 调用。历史
`oversized_block` fixture 曾被 VQA 漏检，未被选作通过证据。

### 旗舰 open-query：click_bell v3

- evaluation：`eval_20260717_stage1_global_click_flagship_v3`
- 时间：2026-07-17 16:22:17–16:32:32（约 10 分 14 秒）
- 预算：3 rounds × 1 ACT episode，seed 100401；expert cohort 独立统计。
- 全局 route：`click_bell / adaptive_properties`，position + instance，checkpoint ready。

| Round | 真实变体 | ACT | Expert | Tool / VQA | 证据驱动决定 |
| --- | --- | --- | --- | --- | --- |
| 1 | `object_position.left_fixed` | 失败 | 成功 | reviewed TCP Tool=`0.0483979434 m`；动态 VQA 未见按铃，与数值一致 | `drill_down` 到镜像右侧 |
| 2 | `object_position.right_fixed` | 成功 | 成功 | reviewed TCP Tool=`0.0184246283 m`；VQA 观察到触发，与数值一致 | `switch_aspect` 到 instance |
| 3 | `object_instance.base0` | 成功 | 成功 | Trusted `official_check_success`；ACT success time=`19.636 s`；VQA 一致 | 预算耗尽并保留 base1 未覆盖，停止 |

ACT 在三个不同变体样本上为 2/3，expert 为 3/3；这说明所测场景可解且仪器有效，但 N=1/变体
不能形成成功率均值或稳定泛化结论。最终反馈如实给出：左/右位置敏感，base0 成功，base1 未测，
不能声称覆盖所有官方实例。

### 第二个 adapter：BBH N=1

- evaluation：`eval_20260717_stage1_global_bbh_n1`
- 时间：2026-07-17 16:33:57–16:41:16（约 7 分 19 秒）
- 全局 route：`beat_block_hammer / generated / object_appearance.color`，checkpoint
  `act-beat_block_hammer/demo_clean-50`。
- TaskGen 真实走 `force_codegen`，调用 proposal/retrieval/codegen，visual self-reflection 通过并确认
  蓝色方块；expert 成功，说明任务可解。
- ACT N=1 失败：`hammer_block_min_xy_error=0.0149173662 m`，但
  `hammer_block_contact_ever=false`、`official_check_success=false`；动态 VQA 与数值证据无冲突。
- 只有一个请求 aspect 且证据充分，Planner 停止。该结果只提供第二 adapter 的端到端功能 smoke，
  不证明任意任务通用性，也不形成外观泛化统计。

## 测试与提交前审查

- 最终服务器全量单元测试：250/250，36.686 秒。
- 三个新 CLI 通过 `py_compile`；`git diff --check` 通过。
- 只读提交前审查发现并修复：unsupported evaluation id 路径逃逸、末轮冲突错误清除
  `unresolved`、reviewed registry symlink 写出、auto-route module provenance 与 overlay 身份验收。
- 剩余已知设计限制：unsupported aspect 仍以跨任务 aspect union 表达，尚不能精确描述
  “某个 task 不支持某个 aspect”的 task-qualified gap；下一批扩 taxonomy 时应一并修正。

## 运行事故与证据取舍

最初两次旗舰启动让主进程 stdout 依赖 SSH：第一次使用 10 秒客户端 timeout，第二次虽加长
timeout 仍被远端断开；子 simulator/ACT artifact 已落盘，但外层均以 `BrokenPipeError` 失败，故
不计为完整闭环证据。v3 改为服务器端 `nohup + stdout/stderr 重定向`，再用短 SSH 只读轮询，
完整完成。该策略已写入运行指引。

两条真实 evaluation 也都发生在本批开发 worktree 尚未提交时，其 manifest 的
`base_commit=f0219d0a766f93da185d5f75fca4dfbcf28d78a5`，不能单靠该字段重建当时的 dirty diff。
核心实现随后提交为 `28a257fb2ce1bc4a8b7b8ad7ce54ab4ec76bf587`，并在实验后补入了路径约束、末轮
`unresolved`、symlink 写入和 provenance 等安全回归；这些修复不改变上述成功/失败样本的
解释，但意味着本批 live 结果只作为开发期功能 smoke，不是 clean-commit、paper-eligible
实验。后续进入论文表实验前，必须从 clean final HEAD（或其后继）重新运行预注册的小协议。

## 当前诚实结论

Stage 1 已满足受限功能门槛：开放 query 能真实选择 task/aspect/checkpoint，Task/Tool 能复用或
生成，ACT、Rule Tool、动态 VQA 和真实证据能改变后续方向，最终反馈能回答原始问题；BBH 又
提供了第二 adapter 的 true-codegen N=1 smoke。

仍未完成论文实验主张：Table 6 缺人工 gold；Tables 7–8 缺真实 simulator 扰动与人工标签；
Tables 1–2 缺公平小型 baseline 和 repetition；Table 3 只有功能 gate，尚无微型消融。多 policy、
LIBERO 与 Table 9 明确不在当前 ACT-only 范围。
