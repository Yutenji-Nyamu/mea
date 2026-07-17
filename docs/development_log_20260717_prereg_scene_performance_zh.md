# 2026-07-17 预注册、原生场景轴与性能能力

本文记录本批新增的功能合同及其论文位置。实验仍遵循敏捷预算，先证明接口和证据链可运行，
不把 `0-ACT` 计划、开发代理标签或单样本 smoke 写成论文结果。

## 1. 论文对应与本批范围

| 本批功能 | 论文对应 | 本批解决的问题 | 当前证据等级 |
| --- | --- | --- | --- |
| hash-pinned evidence preregistration + registered fixed/dynamic command plan | Sec. 3.2、Fig. 2、Fig. 5、Tables 1–2 的实验身份基础 | Query、候选 suite、Git、checkpoint 内容、telemetry、样本表此前没有在执行前共同冻结 | plumbing；生成计划本身为 `0-ACT` |
| TaskGen / ToolGen module-off prepare 与 artifact audit | Sec. 3.3.1–3.3.2、Figs. 3–4、Table 3 | 原缓存微消融只能证明 gate 存在，不能表达严格 matched `complete / module-off` 合同 | plumbing；未完成配对 artifact 时 effect 为 `null` |
| simulator-native background texture 与 lighting capability | Sec. 3.3.1、Fig. 3、Tables 7–8 | 缓存 RGB proxy 不能证明 RoboTwin 场景真的改变 | 代码与 scene-gate 通路；真实服务器结果待验证 |
| official completion-time capability + Trusted Tool | Sec. 3.3.2、Fig. 4、Eq. 3–4 | 系统缺少可真实执行的 performance sub-aspect | 可执行功能；N=1 仅验接线，3/5 才能看小样本稳定性 |
| 20-query proxy 标签随 catalog 更新 | Sec. 3.2、Table 6 | 新增能力后旧标签会 stale，runner 应 fail-fast 而不是静默计错 | development-agent proxy；非人工 gold |

本批没有增加第二种 policy，也没有扩大 ACT repetition。ACT 仍是唯一被评 policy，预算仍优先
`1 → 3 → 5`。

## 2. 预注册身份真正进入执行链

新增 `mea/evidence_manifest.py`、`mea/strategy_plan.py` 及三个命令入口：

- `scripts/manipeval_evidence_manifest.py`：在 clean Git HEAD 上冻结开放 Query、完整候选 suite、
  `base_commit`、checkpoint 文件字节 hash、telemetry profile、N=1 样本表和源 artifact hash；
- `scripts/manipeval_plan_strategy_pair.py`：只生成 inert fixed/dynamic 命令计划、受验证 route 和
  后处理配置，不调用 provider、simulator 或 ACT；
- `scripts/manipeval_compare_registered_strategies.py`：只在两条已完成 run 都满足注册身份时，复用
  严格 comparator 计算 Table 1-facing 的机制指标。

注册身份的数据流为：

```text
repo-local prereg config
→ evidence manifest：Query / suite / Git / checkpoint / telemetry / samples / source hashes
→ registered route + exact fixed/dynamic argv
→ Agent preflight 重新校验 manifest、plan、route 与实参
→ registration_identity 写入 parent evaluation
→ 同一 identity 传给 TaskGen child manifest
→ parent 核对 child identity
→ completed fixed/dynamic artifacts
→ registered post-hoc comparator 再核对身份后才比较
```

因此 manifest 不再只是一个旁路自述文件。缺少任一注册参数、执行时使用 `--auto-route`、
evaluation id/argv 不匹配、checkpoint 或 source 内容变化、candidate suite 漂移、child identity
不一致都会 fail closed。

但应保留两个边界：manifest 的 canonical self-hash 只能发现内容篡改或漂移，不等于第三方证明
命令真的执行过；只有 Agent/child/post-hoc 的逐层绑定加上真实 artifact 才是执行证据。当前 pair
是 `click_bell`、同一 seed、每 candidate 一次的 N=1 微协议，只验证 Table 1 的动态节省样本机制；
它没有 trial distribution，所以 Table 2 consistency 必须为 `null`。

## 3. 严格 module-off 计划与审核

`mea/module_ablation_protocol.py` 和 `scripts/manipeval_module_ablation.py` 把准备与审核分开：

```text
prepare config
→ freeze matched cases + input_identity + execution_identity
→ complete / no_rag / no_visual_gate / no_tool_validation schedule
→ 外部 runner 按冻结合同产生 typed completed artifact
→ audit 校验 schedule hash、case identity、runtime 声明、artifact refs 与 typed outcome
→ 仅完整 matched pair 才产生 functional effect；否则 effect=null
```

TaskGen 的条件是 `complete / no_rag / no_visual_gate`；ToolGen 的条件是
`complete / no_tool_validation`。`input_identity` 与 `execution_identity` 在同一 matched set 内必须
一致；后者可固定 Git、runner 路径与 hash、provider model、配置 hash 和 seed。RAG provenance
只能证明检索来源存在，不能被解释为生成成功或消融 effect。

prepare/audit 命令自身固定报告 provider=`false`、simulator=`false`、ACT=`0`。completed manifest
中的历史 runtime 是 self-attested，审核器并未旁观那次执行；因此在接入一个真正按 condition
切换模块的 runner 以前，这条通路不能称为 live Table 3 消融，更不能从 provenance 推导效果。

## 4. RoboTwin 原生背景与光照

`click_bell/adaptive_properties` 新增两个 bounded capability：

- `scene_background_texture.unseen`：保持官方 bell、pose/instance sampling、任务语义与 ACT
  checkpoint，只启用 RoboTwin `random_background`，并在 eval mode 使用 unseen wall/table
  texture split；
- `scene_lighting.static_random`：保持任务和物体身份，只启用每 episode 随机的方向光/点光颜色，
  `crazy_random_light_rate=0`，因此不是时间闪烁测试。

两者继续经过薄 overlay、simulator-state probe、render/rule/expert gate 和 ACT 执行。背景的数值
权威来自 `task.info.texture_info`；光照的数值权威来自 simulator light configuration。新增的
reviewed VQAQuerySpec 只允许询问“bell 在 unseen 背景下是否清晰可见”和“bell 在随机静态光照下
是否仍可见”，不得覆盖 simulator state 或 `check_success()`。

这是真实 simulator capability，不是 `background_texture_image_proxy` 或 `lighting_image_proxy`。
本文件不填真实 seed/scene-gate/ACT 结果；主线服务器完成验证后应在第 7 节回填 artifact。

## 5. 可执行的 completion-time 性能能力

新增 sub-aspect `performance.completion_time_stability` 与 template
`performance.completion_time_stability.official`。它走 official route：不生成场景变式、不改变
官方任务或 ACT checkpoint，只复用 `task_execution.official_passthrough`，再执行已有受信
`time_to_success` Tool。

```text
open Query
→ GlobalQueryRouter 选择 click_bell / adaptive_properties / performance
→ official unchanged click_bell ACT rollout
→ time_to_success 读取逐步 success trace 的首次成功时间
→ deterministic Aggregate 汇总 success-conditioned completion time
→ bell_visibly_pressed VQA 提供互补视觉证据
→ evidence policy 继续、切换或停止
```

这补的是“系统能真正测量 performance”而不是新增一个名字。N=1 只能证明 Tool、Aggregate、
VQA 和 feedback 接线；至少 3/5 个独立 seed 才能给出小样本均值与离散程度，仍不等于论文完整
repetition。

## 6. 20-query 标签边界

development-agent proxy 集现在把 `click_bell` 的 scene-lighting 与 completion-time query 标成
当前 `(task, profile, aspect)` catalog 支持；现有 background-texture case 测的是
`beat_block_hammer`，仍保持 unsupported，即使 `click_bell` 已实现同名 scene capability。这样
可以继续用 stale-label fail-fast 检查“某方面在一个任务可用，不代表所有任务都可用”。

无论完整 20-query live scorer 的结果如何，这批标签仍固定：

```text
annotation.source = development_agent_proxy
human_reviewer_count = 0
paper_table_eligible = false
```

它可以验证 Planner schema、route、task/capability/aspect 和 first-aspect，但不能报告为 Table 6
的人机一致性或人工 precision。Tables 7–8 的 reviewed VQA spec 同样由开发代理审核，不是人工
majority。

## 7. 服务器验证结果（待主线完成后回填）

以下项目在写本文时尚未取得最终服务器结果，不预填数字：

- unit/full test：`待回填`；
- evidence manifest prepare + validate：`待回填 artifact`；
- registered fixed/dynamic command plan（预期 0-ACT）：`待回填 artifact`；
- module-off schedule prepare/audit（预期 0-ACT）：`待回填 artifact`；
- background texture / static lighting scene-gate smoke：`待回填 seed 与 artifact`；
- completion-time official route smoke：`待回填 budget 与 artifact`；
- 20-query live proxy：`待回填 metrics 与 artifact`。

若本批只完成代码测试和 0-ACT 计划，最终总结必须明确写“真实 ACT pair、live module-off effect、
论文级 human/VQA 指标均未运行”，不能用计划产物替代实验结果。

## 8. 剩余论文距离

本批让实验身份、原生场景轴和性能测量具备最简功能通路，但仍未补齐：

1. 按注册命令完成一次真实 fixed/dynamic matched pair，并验证 parent/child identity 与 post-hoc
   comparator，而不是只生成计划；
2. 为 TaskGen/ToolGen 写真正按 module switch 执行的 matched runner，再产生最小 live module-off
   artifact；
3. 对 background/lighting 各获得 simulator-state 证明的正负或困难 clip，并由独立人工标注；
4. 用 3/5 个 seed 验证 completion-time 小样本分布；
5. 用独立多人 majority 替换 20-query 与 VQA 的 development-agent proxy。

这些工作完成前，本批最准确的成熟度名称是“preregistered plumbing + simulator-native capability
implementation + executable performance route”。
