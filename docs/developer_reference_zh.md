# 开发者参考

## 核心约束

- 只扩展 `Global route → ClaimFirst → QueryContract` 主链。
- catalog 只描述可运行能力；不得在 Query 中预埋 aspect 顺序。
- official success 与 generated experimental checker 分开命名和汇报。
- TaskGen/ToolGen 各允许一次局部 regenerate/repair；policy failure 不自动重跑。
- 生产运行只写一份 `manifest.json`；实验 hash 放在 `experiments/paper/`。
- 六个 `mea/*/README.Agent.md` 是生成时上下文和 Table 3 消融组件，不是普通文档。

## 增加 RoboTwin task

1. 确认 official task 可以由 expert 在若干固定 seed 初始化并完成。
2. 增加 TaskSchema：actor、接触点、单位、official success 和可用 generic metrics。
3. 下载服务器端 ACT/DP3 checkpoint 与 stats；记录来源和 revision。
4. 用 generic recorder、Rule Tool 和 VQA 跑一个 N=1 official smoke。
5. 把 task 加入 capability catalog；不要新增 task-specific planner。
6. 只有 Query 确实需要新场景时才增加 TaskGen capability。

## 扩展 TaskGen

- retrieve-first；未命中时由 provider 生成 scene 与 `check_success()`。
- 静态边界只限制 import、写路径和危险 API，不要求 AST 与 reference 完全相同。
- 验收至少包含正例、未完成负例和关键反例；再做 render/visual check。
- generated checker 只裁决实验定义，不能覆盖 official RoboTwin 成功率。

## 扩展 ToolGen

- Query 先产生 metric need，不在 Planner 中硬编码 operator。
- 新 Tool 必须通过 smooth/positive/negative/missing-data 等最小 oracle。
- `semantic_key` 相同才 exact reuse；复用时不得再次调用 provider。
- null 是有效结果，必须带原因并进入 Aggregate，不能用旧缓存数值代替。

## Review 清单

- 新代码是否直接支撑论文 claim？
- 是否已有主链能力覆盖它？
- 是否引入第二套 planner/recovery/registry/manifest？
- 测试是否验证当前接口，而不是保活已删除的旧链路？
- 证据是否明确区分 smoke、proxy 和 paper-scale result？

任何两批内不会调用的兼容层应删除；需要恢复时使用 Git 历史，而不是长期 feature flag。
