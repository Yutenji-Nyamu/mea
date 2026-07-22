# 2026-07-22：TaskGen reuse-first、SuccessSpec v2 与 registered adaptive

本批只补论文 Sec. 3.2、Sec. 3.3.1、Figs. 2–3 的主体缺口；不增加新 policy、任务数量或重复
规模。Windows 只保存轻量源码并做编辑，全部测试与运行验收都在 AutoDL 的 RoboTwin 环境执行。

## 1. 代码变化

### 1.1 reuse-first Task resolution

`mea/taskgen/resolver.py` 用 exact executable semantic key 在 provider 创建前执行：

```text
validated TaskProposal + capability contract
→ official task 可满足：official reuse
→ 内置 bounded overlay/reuse 可满足：built-in reuse
→ 查询显式审核、exact semantic match 的 generated artifact
→ 未命中才允许 force_codegen
```

key 忽略 Query 的措辞与 `proposal_id`，但绑定 task/aspect/capability/changes、success preservation、
SuccessSpec 占位与完整 capability contract hash。正常 TaskGen 保存
`generation/task_resolution.json`，同时记录 requested/resolved route、materialization 与是否需要
provider。当前生产主链尚未实现审核 generated artifact 的持久注册和重新物化；该分支仍诚实标为接口。

### 1.2 SuccessSpec v2

v1 完全兼容。v2 由可信 envelope 限制 actor、functional point、axis、comparison、谓词类型与阈值：

- `bbh.official_capability` 只接受与 RoboTwin 官方 BBH 相同的 `all`、`0.02 m` 阈值和 contact，
  `act_eligible=true`；
- `bbh.development_fixture` 可离线验证受限阈值及 `all/any`，但固定
  `act_eligible=false`，正常 TaskGen 不提供绕过开关；
- 编译结果继续与可信 AST expansion 和 official oracle 做差分，而不是执行模型给出的 Python。

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
外的 aspect/template。

## 2. 服务器验收

- 定向 resolver、SuccessSpec、TaskGen、attempt recovery、PlanSession、registered runtime 与 strategy
  回归：`67/67` 通过；
- 全量：`515/515` 通过，49.959 秒；
- 本批本地未运行 unittest/pytest、未新建 Conda/venv、未下载 checkpoint/asset。

严格 live pair 使用同一 click_bell Query、seed `100401`、checkpoint 与两轮上限：fixed 固定执行
`object_instance.base0/base1`，dynamic 在 base0 后由真实 evidence 决定 stop 或 refine。结果与 artifact
路径在该批 live 完成后补入紧凑 snapshot；预算预计 3 条、最多 4 条 ACT。

## 3. 不能声称什么

- 515 个测试是源码/fixture 回归，不是论文表格结果；
- development SuccessSpec v2 不可进入 ACT；
- reviewed generated-task lookup 尚不等于 persistent registry/materializer；
- assistant-proxy 与跨 commit 历史扰动不等于独立 human gold 或 Tables 7–8；
- matched N=1 只能证明机制与单 seed 运行，不给出均值、方差或泛化结论。
