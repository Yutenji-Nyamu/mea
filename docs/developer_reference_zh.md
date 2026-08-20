# 开发者参考

## 公开术语与兼容边界

论文方法、公开文档和新运行产物统一使用以下术语：

- **Plan Agent**：解释 Query、提出 sub-aspect，并依据累计 evidence 决定继续或停止；
- **Query interpretation**：在 task inventory 暴露前抽取任务意图、待测 concern 与
  preservation 条件；
- **Proposal**：Plan Agent 交给 TaskGen/ToolGen 的本轮实验语义及 typed needs；
- **Plan Agent session**：保存逐轮 evidence、Plan 的继续/停止决定与最终回答状态。

代码文件、旧 schema 和不可变历史 artifact 中的 `ClaimFirst`、`FreeConcern` 是兼容名称；
`ExperimentCandidate` 暂时仍是生产内部的 Proposal transport。公开术语统一为 Proposal，
内部重命名尚未完成；不得通过重写历史 artifact 来“修复”旧术语。

## 核心约束

普通生产运行只表达论文主链：

```text
Query → Plan → Task/Tool 检索或生成 → 执行 → 证据回流 → 继续或回答
```

新增或保留机制时先回答：它对应论文的哪一步；若论文没有，它解决了哪个已经真实发生、
且不能靠更精确的 prompt 或更小真实性边界解决的问题。两问都答不上来的机制不进入生产链；
只为旧实验有价值的实现迁入 `experiments/paper/`，caller 清零后依靠 Git 历史恢复，不长期保留
生产 feature flag。不要为 failure exemplar 再建立审批、哈希、晋升或恢复系统。

生产实现只保留三层：论文方法；生成代码可执行、simulator authority、official goal 忠实、
unknown/abstain 等最小真实性边界；以及默认不加载的冷复现资料。生产不再用 SHA/checksum
作为 evidence、lineage、审批或执行许可；candidate 命名和 HistoryDB 去重中仍有技术性摘要，
需另行迁移而不能冒充已全部清零。receipt、多级 reviewed/promotion、flagship acceptance、复杂
resume/attempt ledger、legacy protocol 等概念不得继续穿透普通 CLI/Application/RoundExecutor。
迁移必须先断 caller、再迁冷、最后删除，不能仅凭文件名批量裁剪。

- 只扩展
  `Query interpretation → runtime limits → PlanAgentInitialPlanBuilder → Plan Agent session`
  主链。catalog/global-query、fixed/task-specific planner 及其兼容 factory 已从当前源码删除；
  历史行为只从 Git 与冻结 paper result 审计。
- 生产 generated round 使用 Proposal 与
  `GenericRoboTwinTaskAdapter`；后者只包含 official source/class、runtime TaskContext、检索文档/
  资产和 simulator validation hooks，不得枚举 aspect、variant、metric 或 planner route。
- TaskSchema 是可复用的语义缓存，不是生产准入表。缺失时 fresh reset probe 只从
  official source 声明的 public root 中发现 actor；嵌套访问只允许原生 list/tuple/dict 的
  typed `access_path`，不执行字符串路径，也不猜 target role、contact point 或 success threshold。
- generic Task、Rule Tool 与 VQA artifact 分别由 `GenericTaskArtifactIndex`、
  `toolgen.registry` 与 Execution VQA question library 检索。旧
  `mea/artifact_retrieval_index.py` 仍提供 task/VQA retrieval hint，属于待迁移菜单债务；
  `mea/capability_adapter.py` 只是旧模板/消融 shim。任何成员关系都不得作为 open-world round
  的执行许可。
- Query 不得由预排菜单写入 aspect 顺序。开放 concern 保留
  相互独立的 scene/checker/tool typed need，进入 exact reuse 或
  generate→validate；缺少 template id 不能成为终止理由，Tool-only Proposal 不得被
  强制生成 scene/checker。
- official success 与 generated experimental checker 分开命名和汇报。
- `ImplementationTrace` 必须逐项记录 scene/checker/tool need 的实现来源、preservation
  authority 与验证结果；只有 `direct+complete` 的 Proposal 才能成为充分性判断中的
  已执行证据。
- 开放 Proposal 不能越过 backend 执行面：当前 generic RoboTwin TaskGen 的 mutation
  roots 是 `load_actors` 与 `check_success`，不能用任意 scene diff 冒充
  policy/controller/gripper precision、action noise/latency 或权重变化。只有 Runtime
  Task Binding 显式发布相应 intervention hook 时才允许这些变化。
- Query interpretation 同时保留原始 Query 与 provider 解释中的显式 preservation
  条件；provider 漏写不能删除用户约束。复合空间条件必须按 contact、
  position、orientation 分量全部验证并合取。
- generic TaskGen 与 ToolGen 各只有一份共享的局部 repair 预算；TaskGen 遇到 checker
  fixture 失败时可保持已验证 scene、只修 checker；policy failure 不自动重跑。
- 每个 evaluation 只写一份顶层 lifecycle `manifest.json`（resolved config、Git revision 与
  最终 artifact 路径）；子 artifact 可有局部技术 manifest。hash-pinned evidence/command plan
  只属于 `experiments/paper/`。
- control-required Query 在 control evidence 完成前不得生成、缓存或冻结下一
  Proposal。后续 Proposal 必须由 Plan Agent session 使用 completed-round evidence 生成；
  直接传递 round id 与相关 evidence，不在生产语义中增加 lineage hash 或 input digest gate。
- `mea/round_executor.py` 是生产单轮执行边界；RoboTwin policy backend 不得再先运行
  旧 child bundle 后做事后 projection。
- planner、feedback、retrieval、taskgen、toolgen 的 `README.Agent.md` 都是各 Agent 的生产
  prompt 规则 owner，也是消融输入。Python 只组装动态 evidence、任务卡、schema 与输出格式；
  同一规则不得在两处复写。Table 3 当前只消融 TaskGen 的一份。

## 增加 RoboTwin task

1. 确认 official source 可在固定 seed reset，并验证所选 policy backend 的 task binding。
2. 无 TaskSchema 时由 fresh reset 自动建立 run-local TaskContext；只有需要稳定
   role、functional/contact point 或更丰富 telemetry 时才增加一份可检索 TaskSchema。
3. 不为生产 Plan Agent/TaskGen 增加任务名条目。`runtime_task_binding.py` 从
   source/TaskContext/policy scope 自动建立执行边界，
   `load_generic_robotwin_task_adapter()` 再从 source/TaskContext 自动发现生成 hooks。
   只有确有已审查 artifact 需要复用时，才向 retrieval index 增加数据化条目；它不能
   携带 planner kind、执行许可或预排 aspect 顺序。
4. 用 generic recorder、Rule Tool 和 VQA 跑一个 N=1 official smoke；这只能把任务标成
   `official-only`。
5. 只有 Query 确实需要新场景，且 model-written scene/checker、fixture/render、
   rollout、Tool/VQA/Answer 均在同一链中通过，才能把任务标成“深入”。

当前支持范围、每个 adapter 的证据深度和最新旗舰验收会随运行更新，统一见
[论文 claim 与 gap](paper_claim_gap_zh.md)和[当前证据](evidence/current/README.md)。
新增 task 不应复制 task-specific planner，也不应仅因 checkpoint 存在或事后 replay
通过就宣称干净在线方法闭环。

统一多任务 SmolVLA checkpoint 的 server-only 安装与 runner 见
[RoboTwin / SmolVLA 复现](robotwin_smolvla_reproduction_zh.md)。该 runner 动态导入
`envs.<task>`，不维护任务 allowlist；它只增加 policy backend 广度，不替代上述
Query→TaskGen/ToolGen→evidence planning 验收。

## 增加 policy backend

- 模型、checkpoint、Python/CUDA 依赖和仿真资产只在服务器下载、安装与验证。
- 在 `experiments/paper/<backend>/deployment_ledger.md` 保留一份 cold 流水账：硬件与
  磁盘探针、source/checkpoint revision、路径、大小与 hash、环境创建、实际执行的每条
  命令及结果、遇到的问题、诊断依据和最终解决方法。
- 网络加速、镜像、wheel 重试和临时目录必须记录启用与退出边界；不得记录密码、API key、
  token 或代理凭证。
- 分开报告 offline import/forward、standalone official rollout 与共享 MEA 方法入口；前两项
  通过不能自动升级为完整方法证据。
- 短 runbook 只保存已验证的最短复现路径并链接流水账，不在多个文档复制完整终端历史。

Hy-VLA 的具体范例见
[`experiments/paper/robotwin_hyvla/deployment_ledger.md`](../experiments/paper/robotwin_hyvla/deployment_ledger.md)。

## 扩展 TaskGen

- retrieve-first；未命中时由 provider 只生成 Proposal 声明所需的 scene 和/或
  `check_success()`；`checker_need=null` 时可显式复用 official checker。
- exact spatial/contact preservation 必须使用 same-seed simulator state；无 simulator/
  AST authority 的 geometry 必须返回 `unknown`。当前 generic backend 会在 lookup 和
  provider 之前拒绝同时要求 uniform scale、center/origin 不变与 contact-point world
  position 不变、且没有 custom pivot capability 的不可实现 Proposal。
- exact reuse 只跳过 provider/codegen；当前 seed 的 setup、render、expert 与 checker
  fixtures 必须重新运行，revalidation 通过后才能增加 reuse count。
- 静态边界只限制 import、写路径和危险 API，不要求 AST 与 reference 完全相同。
- 验收至少包含正例、未完成负例和关键反例；再做 render/visual check。
- 新 actor 用 `mea_telemetry_tracked_actors` 声明；recorder 由实际生成任务扩展 telemetry
  schema，ToolGen 必须在 rollout 后消费这个 schema。
- generated checker 只裁决实验定义，不能覆盖 official RoboTwin 成功率。

## 扩展 ToolGen

- Query 先产生 metric need，不在 Plan Agent 中硬编码 operator。
- 显式 final/terminal semantic trace 单信号分量使用
  `terminal_signal_component(signal, component, absolute)`；两信号终态差使用
  `terminal_signal_difference(left_signal, right_signal, component, absolute)`。
  semantic alignment gate 必须拒绝用 event/distance metric 回答 terminal
  `x/y/z/height` need。
- 新 Tool 必须通过 smooth/positive/negative/missing-data 等最小 oracle。
- `semantic_key` 相同才 exact reuse；复用时不得再次调用 provider。
- Tool library 使用可读 semantic key 检索；跨 Query/evaluation 复用时仍在当前 telemetry
  上重跑静态检查、确定性执行与 oracle 校验，不再经过审批/promotion 协议。
- null 是有效结果，必须带原因并进入 Aggregate，不能用旧缓存数值代替。

## 扩展 Execution VQA

- dynamic/open-world Query 先查 task-scoped retrieval hint 或 semantic-key question artifact。
- task 没有可用问题时，只能生成受限的
  `run_local.tracked_object_visible_state_change`；不得继承其他任务的 legacy 问题。
- 无 context 的旧 API 调用才保留 BBH 三问题默认值，生产 dynamic context 不走该 fallback。
- VLM observation 与 Rule/checker 冲突时保留 `evidence_conflict`；VLM confidence 不拥有
  覆盖数值或 checker predicate 的 authority。
- `experiments/paper/manipeval_execution_vqa_replay.py` 只用于对已完成 rollout 做
  append-only 方法审计，不是生产路径，也不增加 policy sample。
- 历史 Tool/Planner repair replay 已从当前项目删除；需要研究旧 artifact 时从 Git 历史或
  冷 evidence 只读恢复，不把其 hash/append-only 协议接回普通生产。

## Review 清单

- 新代码是否直接支撑论文 claim？
- 是否已有主链能力覆盖它？
- 是否引入第二套 Plan Agent/recovery/registry/manifest？
- 该机制对应论文哪一步，或解决了哪个已观察且不能由 prompt/更小边界处理的故障？
- hash、receipt、approval/promotion、compat 或恢复状态是否仍穿透生产构造器？
- 是否把 retrieval catalog 的成员关系误当成 Proposal 执行许可？
- Tool-only/scene-only/checker-only Proposal 是否只运行实际需要的生成阶段？
- 动态 VQA 是否出现与当前 task 无关的对象或现象？
- 测试是否验证当前接口，而不是保活已删除的旧链路？
- 证据是否明确区分 smoke、proxy 和 paper-scale result？
- imports、pytest、provider、simulator 与 policy validation 是否只在 canonical
  AutoDL 执行，而不是 Windows 工作站？

任何两批内不会调用的兼容层应删除；需要恢复时使用 Git 历史，而不是长期 feature flag。
