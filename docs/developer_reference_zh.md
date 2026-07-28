# 开发者参考

## 核心约束

- 只扩展
  `Global route → QueryContract → ClaimFirstInitialPlanBuilder → OpenWorldPlanSession`
  主链。生产 ClaimFirst 不得实例化 `CatalogPlanAgent` 或任务专属 Planner；旧 Planner
  只能由 `experiments/paper/legacy_planner_factory.py` 显式、延迟加载。
- 生产 generated round 使用 `ExperimentCandidate` 与
  `GenericRoboTwinTaskAdapter`；后者只包含 official source/class、TaskSchema、检索文档/
  资产和 simulator validation hooks，不得枚举 aspect、variant、metric 或 planner route。
- `mea/capability_adapter.py` 暂留为 official control 与旧模板的 retrieval/消融兼容层；
  它的 catalog 成员关系不得作为 open-world round 的执行许可。
- catalog 不得在 Query 中预埋 aspect 顺序。catalog 外 concern 保留
  相互独立的 scene/checker/tool typed need，进入 exact reuse 或
  generate→validate；缺少 template id 不能成为终止理由，Tool-only candidate 不得被
  强制生成 scene/checker。
- official success 与 generated experimental checker 分开命名和汇报。
- TaskGen/ToolGen 各允许一次局部 regenerate/repair；policy failure 不自动重跑。
- 生产运行只写一份 `manifest.json`；实验 hash 放在 `experiments/paper/`。
- 五个被运行时读取的 `mea/*/README.Agent.md`（feedback、retrieval、planner、taskgen、
  toolgen）是生成上下文，不能按普通文档删除；Table 3 当前只消融 TaskGen 的一份。

## 增加 RoboTwin task

1. 确认 official task 可以由 expert 在若干固定 seed 初始化并完成。
2. 增加 TaskSchema：actor、接触点、单位、official success 和可用 generic metrics。
3. 下载服务器端 ACT/DP3 checkpoint 与 stats；记录来源和 revision。
4. 只为 official control/checkpoint binding 增加数据化条目；generated round 由
   `load_generic_robotwin_task_adapter()` 从 source/schema 自动发现，不增加任务名分支、
   planner kind、aspect、metric 或 VQA 菜单。
5. 用 generic recorder、Rule Tool 和 VQA 跑一个 N=1 official smoke；这只能把任务标成
   `official-only`。
6. 只有 Query 确实需要新场景，且 model-written scene/checker、fixture/render、
   rollout、Tool/VQA/Answer 均在同一链中通过，才能把任务标成“深入”。

当前基线是五个 official adapters：BBH/ClickBell 较深入；AdjustBottle 有一次
provider scene/checker→visual/expert→ACT 的在线生成式链，以及独立的 0-ACT Tool/VQA
修复 replay，但源在线 bundle 仍为 inconclusive；GrabRoller/PlacePhoneStand
official-only。新增 task 不应复制 task-specific planner，也不应仅因 checkpoint 存在
或事后 replay 通过就宣称干净在线方法闭环。

## 扩展 TaskGen

- retrieve-first；未命中时由 provider 生成 scene 与 `check_success()`。
- exact reuse 只跳过 provider/codegen；当前 seed 的 setup、render、expert 与 checker
  fixtures 必须重新运行，revalidation 通过后才能增加 reuse count。
- 静态边界只限制 import、写路径和危险 API，不要求 AST 与 reference 完全相同。
- 验收至少包含正例、未完成负例和关键反例；再做 render/visual check。
- 新 actor 用 `mea_telemetry_tracked_actors` 声明；recorder 由实际生成任务扩展 telemetry
  schema，ToolGen 必须在 rollout 后消费这个 schema。
- generated checker 只裁决实验定义，不能覆盖 official RoboTwin 成功率。

## 扩展 ToolGen

- Query 先产生 metric need，不在 Planner 中硬编码 operator。
- 显式 final/terminal semantic trace 分量使用通用
  `terminal_signal_component(signal, component, absolute)`；semantic alignment gate
  必须拒绝用 event/distance metric 回答 terminal `x/y/z/height` need。
- 新 Tool 必须通过 smooth/positive/negative/missing-data 等最小 oracle。
- `semantic_key` 相同才 exact reuse；复用时不得再次调用 provider。
- 同 evaluation 使用 run-local registry；跨 Query/evaluation 只允许显式 approved 的
  reviewed typed Tool，并在当前 telemetry 上重跑确定性与 oracle 校验。
- null 是有效结果，必须带原因并进入 Aggregate，不能用旧缓存数值代替。

## 扩展 Execution VQA

- dynamic/open-world Query 先按 task retrieval index 选择 task-owned 已审查问题。
- task 没有可用问题时，只能生成受限的
  `run_local.tracked_object_visible_state_change`；不得继承其他任务的 legacy 问题。
- 无 context 的旧 API 调用才保留 BBH 三问题默认值，生产 dynamic context 不走该 fallback。
- VLM observation 与 Rule/checker 冲突时保留 `evidence_conflict`；VLM confidence 不拥有
  覆盖数值或 checker predicate 的 authority。
- `experiments/paper/manipeval_execution_vqa_replay.py` 只用于对已完成 rollout 做
  append-only 方法审计，不是生产路径，也不增加 policy sample。
- 组合 Tool/VQA replay 必须由
  `manipeval_replay_completed_tool.py --execution-vqa <manifest>` 显式消费前一步
  VQA artifact 并冻结 path/hash；不得手工合并两个独立 summary。冲突证据应让 Planner
  以 `evidence_conflict`/inconclusive 停止，而不是被标成 sufficient。

## Review 清单

- 新代码是否直接支撑论文 claim？
- 是否已有主链能力覆盖它？
- 是否引入第二套 planner/recovery/registry/manifest？
- 是否把 retrieval catalog 的成员关系误当成 candidate 执行许可？
- Tool-only/scene-only/checker-only candidate 是否只运行实际需要的生成阶段？
- 动态 VQA 是否出现与当前 task 无关的对象或现象？
- 测试是否验证当前接口，而不是保活已删除的旧链路？
- 证据是否明确区分 smoke、proxy 和 paper-scale result？

任何两批内不会调用的兼容层应删除；需要恢复时使用 Git 历史，而不是长期 feature flag。
