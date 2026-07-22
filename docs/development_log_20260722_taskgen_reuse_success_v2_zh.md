# 2026-07-22：TaskGen resolution 接口、SuccessSpec v2 与 registered adaptive

本批只补论文 Sec. 3.2、Sec. 3.3.1、Figs. 2–3 的主体缺口；不增加新 policy、任务数量或重复
规模。Windows 只保存轻量源码并做编辑；最终验收都在 AutoDL 的 RoboTwin 环境执行，早期误跑的本地
定向测试在第 2 节单独说明。

## 1. 代码变化

### 1.1 capability-bound Task resolution

`mea/taskgen/resolver.py` 用 exact executable semantic key 在 provider 创建前执行。当前 route 已由上游
可信 capability contract 选定，因此本批实现的是可审计的 capability-bound resolution，而不是跨所有
task artifact 的完整全局 selector：

```text
validated TaskProposal + capability contract
→ official task 可满足：official reuse
→ 内置 bounded overlay/reuse 可满足：built-in reuse
→ force_codegen 分支如配置 registry callback，则查询显式审核、exact semantic match 的 generated artifact
→ 正常 runtime 尚未配置该 callback，诚实记录未尝试 lookup 并进入 codegen
```

key 忽略 Query 的措辞与 `proposal_id`，但绑定 task/aspect/capability/changes、success preservation、
SuccessSpec 占位与完整 capability contract hash。正常 TaskGen 在 run 成功物化后保存
`generation/task_resolution.json`，同时记录 requested/resolved route、materialization 与是否需要
provider。当前生产主链尚未实现审核 generated artifact 的持久注册和重新物化；因此不能把该接口写成
论文 Fig. 3 完整的 retrieve-first resolver。

### 1.2 SuccessSpec v2

v1 完全兼容。v2 由可信 envelope 限制 actor、functional point、axis、comparison、谓词类型与阈值：

- `bbh.official_capability` 只接受与 RoboTwin 官方 BBH 相同的 `all`、`0.02 m` 阈值和 contact，
  `act_eligible=true`；
- `bbh.development_fixture` 可离线验证受限阈值及 `all/any`，但固定
  `act_eligible=false`，正常 TaskGen 不提供绕过开关；
- official envelope 继续与可信 AST expansion 和 official oracle 做差分；development `any` 只用
  truth-table fixture 验证。两者都不执行模型给出的 Python。

这证明了 bounded DSL/编译器，不证明模型能任意生成正确 `check_success()`。

### 1.3 TaskGenerationAttempt recovery

`mea/taskgen/attempts.py` 把 SuccessSpec、scene code/static、render/vision 与 expert gate failure 映射到
typed `repair_success_spec / regenerate_candidate / repair_scene / terminal`，最多再尝试一次。每次
attempt 写 append-only artifact；accepted 前必须为 0 ACT，policy failure 永不触发 TaskGen 重试。
fixture 已真正让第一次 visual failure 的 action 进入第二次 repair callback，并只在 accepted 后启动
一次 policy callback。生产 TaskGen 各分支仍需在下一批统一接入该 controller。

### 1.4 registered dynamic planner

旧 registered dynamic pair 会意外跳过公共 `AdaptivePlanStepAgent`。现在
`dynamic_evidence_v1` 使用与正常 adaptive 相同的 Query + evidence step；`fixed_predeclared_v1` 与
legacy planner 不变。registered run 的 navigation 只暴露预注册 candidate suite，不能发现 hash 范围
外的 aspect/template。这里“现在”指源码与服务器回归已经修复；下面在 commit `412fe6e` 上完成的
strict pair 早于最终接线修复，仍走旧 task-specific decision path，不能作为公共 step 的 live 证据。

## 2. 服务器验收

- 第一阶段定向 resolver、SuccessSpec、TaskGen、attempt recovery、PlanSession、registered runtime 与
  strategy 回归：`67/67` 通过；最终接线修复后的 Planner/strategy 定向回归：`34/34` 通过；
- 最终全量：`516/516` 通过，49.857 秒；
- 本批没有在本机新建 Conda/venv、安装依赖或下载 checkpoint/asset。开发前段曾误用已有
  `E:\anaconda` 跑过两组本地定向测试（`114/117`，3 个只因缺 RoboTwin hammer asset；随后
  `103/103`）；这不符合当前约定，且不作为验收证据。此后 MEA 测试全部改在服务器执行。

严格 live pair 使用同一 click_bell Query、seed `100401`、ACT checkpoint、候选集和两轮上限：

- registration：`batch16_instance_pair_n1_412fe6e`；
- fixed：`eval_batch16_instance_pair_n1_412fe6e_fixed`，2 条 ACT，base0 成功、base1 失败；
- dynamic：`eval_batch16_instance_pair_n1_412fe6e_dynamic`，2 条 ACT，得到相同的成功/失败结果；
- dynamic 的旧 task-specific decision path 在 round 1 读取真实 Rule/VQA/Evidence 后选择
  `continue/drill_down`，在 round 2 候选耗尽后选择 `stop`；
- overlap exact success agreement=`1.0`，rollout savings=`0`，`paper_table_eligible=false`；
- comparison：
  `mea/validation_runs/batch16_instance_pair_n1_412fe6e/comparison/summary.json`。

这只证明 strict registered pair 的 ACT/Tool/VQA 数据链可运行，并在同 seed 下保持结论一致；它同时暴露
了 registered command 不带 `--auto-route` 时会跳过公共 `AdaptivePlanStepAgent` 的接线缺口。该缺口已在
源码中通过“trusted catalog + provider 成对初始化”修复并回归，但本批不再追加 ACT，因此公共 step 的
clean-head live 证据仍待下一次最小验收。N=1 也不能计算 Table 2 consistency，不能把该结果写成论文
Tables 1–2 的复现。

随后使用 round 1 的真实缓存 Rule/VQA/telemetry 做了 0-ACT live-provider plan-only smoke。这个 smoke
暴露 `rule.aggregate_status=passed` 容易被模型误读为策略成功，因此 prompt 现在明确：它只表示聚合证据
有效，`policy.success_rate` 才是 ACT 结果；同时当失败且存在同方面 counterfactual 时，`stop` 不再是合法
动作。使用明确的 Query 判据后，同一份非策略 evidence 得到：

- `policy_success=0.0` → provider `refine object_instance.base1`；
- `policy_success=1.0` → provider `stop`；
- 两次均无 provider validation error，且 simulator/ACT 调用数为 0。

这证明修复后的公共 step 能在 plan-only 路径上依据真实缓存 evidence 分支，但仍不能替代一次整合到
TaskGen→ACT→Tool/VQA→下一轮的 clean-head live run。

## 3. 不能声称什么

- 516 个测试是源码/fixture 回归，不是论文表格结果；
- development SuccessSpec v2 不可进入 ACT；
- reviewed generated-task lookup 尚不等于 persistent registry/materializer；
- commit `412fe6e` 的 strict dynamic run 使用旧 task-specific decision path，不是当前公共 step 的 live 证据；
- 0-ACT live-provider replay 是控制流证据，不是新的策略实验；
- strict pair 的 exact agreement 不是统计一致性，0 rollout savings 也不是效率改进；
- assistant-proxy 与跨 commit 历史扰动不等于独立 human gold 或 Tables 7–8；
- matched N=1 只能证明机制与单 seed 运行，不给出均值、方差或泛化结论。
