# ManipEvalAgent 论文主张与当前差距（自顶向下）

本文只按论文 Sec. 3.2–3.4 与 Figs. 2–5 审查主体方法，不把工程审计、hash、跨进程恢复或
更多任务数量误当作论文核心贡献。实验表格的规模复现另算；本项目当前是 ACT-only、两个任务族和
少量可信 capability 内的受限功能原型，不是论文规模或开放世界能力的完整复现。

## 1. 论文主体数据流

```text
开放 Query
→ Plan Agent：结合 Query 与 Y1:t 动态发现/选择 sub-aspect
→ Task Proposal → TaskGen：检索复用或生成 runnable task + check_success
→ 场景渲染、视觉诊断与有界修复
→ 固定的单任务 ACT checkpoint 执行
→ Tool Proposal → ToolGen：检索复用或生成、验证、注册 Rule Tool
→ Dynamic Execution VQA + Rule Tool
→ Aggregate / evidence
→ evidence 改变下一轮方向，或停止并回答原 Query
```

这里的动态性首先是**同一次单任务 evaluation 内，证据到来后再决定下一 sub-aspect**。一个 ACT
checkpoint 只能执行其训练任务并不冲突：系统通用性来自公共合同与 task adapter，而不是让一个
checkpoint 中途切任务。跨任务 `EvaluationGraph` 是可选的多 checkpoint portfolio，不是 Fig. 2
主体闭环成立的前置条件。

## 2. claim → 当前对应 → 诚实 gap

| 优先级 | 论文 claim | 当前最小对应 | 仍有的 gap |
| ---: | --- | --- | --- |
| 1 | Plan Agent 根据 `Query + Y1:t` 动态发现 sub-aspect（Sec. 3.2；Figs. 2/5） | `AdaptivePlanStepAgent` 每轮只接收受信 Rule/VQA/Evidence 与当前 coverage，输出 `propose/refine/stop`；candidate allowlist 限定在当前 task/checkpoint 的可信 capability 内。clean-head live v4 使用与 Query 一致的 task-only `click_bell` binding，但无 bound aspect/history/内部顺序提示；它自主先选 `object_position.left_fixed`，ACT 成功后读取真实 Evidence，并由 provider `propose/switch_aspect` 到 `object_instance.base0`；两轮 ACT/Tool/VQA/Aggregate 完成，最终 lifecycle=`completed` | 当前停止来自 hard cap=2，不是 Planner 证明 Query 已充分回答；`right_fixed`、`base1` 未测。candidate universe、Query 真正 required coverage 与预算上限仍耦合，尚不能根据“all/across/some/worst-case”等量词给出可审计的 evidence sufficiency |
| 2 | TaskGen 对 Proposal 做 retrieve-or-generate，并交付 runnable scene + `check_success()`（Sec. 3.3；Fig. 3） | 上游可信 capability 先选 official/overlay/codegen route；TaskGen 在 provider 前计算 exact executable semantics。reviewed-task registry、正常 runtime callback 和 production acceptance 已接线；一个受限 BBH reviewed variant 已在新 run 中完成 TaskGen provider=0 的 expert 与 ACT N=1 复用。`SuccessSpec v2` 也支持可信 envelope、受限阈值与 `all/any` 编译 | 当前只是 exact semantic match 的单个受限 reviewed variant，不是跨任意资产/文档的全局 RAG selector。公共 Proposal Agent 仍主要交付 v1、official-preserving 语义，尚不能自然提出并让 ACT 消费 proposal-derived SuccessSpec v2；development-agent review 也不是人工 paper-eligible 审核 |
| 3 | TaskGen 用 render/视觉反馈诊断并修复生成错误（Fig. 3；App. A.3.4） | scene reflection、SuccessSpec repair 与 expert gate 已有；相关 stage 使用 typed diagnosis/action 做最多一次局部 repair/regenerate，并保证 accepted 前 0 ACT、policy failure 不重试 | 论文使用 stage-specific recovery，并不要求所有阶段共享一个中央 controller。当前真实 visual repair 和 fixture recovery 的覆盖仍窄；click overlay 与大多数正常路径仍以 validate/accept 为主 |
| 4 | ToolGen 根据 Tool Proposal 检索、生成、验证、注册与复用（Sec. 3.3；Fig. 4） | trusted Tool、Python ToolGen、AST/differential gate、reviewed registry、run-local reuse 已有；`ToolProposal v3` 携带的 `MetricSpec v1` 已接入正常 runtime，并允许单条真实 episode 做 deterministic/oracle 校验 | DSL/operator 与可观测 signals 仍受限；安全只新增精确 camera-contact proxy，不能概括完整 unintended-contact/safety |
| 5 | Rule Tool 与 Dynamic VQA 互补观测执行结果（Sec. 3.3；Fig. 4） | telemetry/events/video → trusted/generated Tool + 事件关键帧 VQA → typed evidence；数值冲突时 Rule 为权威 | 已有真实与缓存机制证据，但真实扰动下的独立人工 gold、正负平衡和 VQA accuracy/AUROC 仍缺 |
| 6 | Aggregate 后的 evidence 改变后续规划并形成 Query-centric feedback（Secs. 3.2–3.4） | clean-head v4 的首轮真实 Evidence 触发 `object_position→object_instance`；缓存 replay v2 在相同非 policy evidence 上只改 `policy_success`，得到 `0→drill_down position`、`1→switch instance`，且 ACT=0 | 真实 run 与反事实共同证明“evidence 能改变下一动作”，但没有证明何时证据足以回答原 Query。最终报告必须把 hard-cap stop、未覆盖候选和 unsupported capability 分开，不能把预算耗尽写成证据充分 |
| 7 | 方法可在多个 concern/task 上使用（Fig. 2） | BBH 与 click_bell 共用 planning/evidence/proposal/runtime 合同；每个 child 固定自己的 ACT checkpoint | materializer 与真正可执行 aspect 仍少。跨任务 graph 只是一种 portfolio 编排，不应被提升为高于单任务动态闭环的核心 gap |
| 8 | 小规模实验能比较 adaptive 与 fixed，并评估 Planner/VQA（Tables 1–3、6–8） | 已完成同 Query/seed/checkpoint/candidate suite 的 strict matched N=1：fixed/dynamic 各 2 ACT，exact success agreement=`1.0` | dynamic 是旧决策路径，rollout savings=`0`，N=1 无 trial distribution，`paper_table_eligible=false`；只保留为执行链 pilot。N=3、独立人工 gold 与更多扰动后置 |

## 3. 当前最值得补的顺序

1. **Query-conditioned evidence sufficiency（先 0 ACT，再按需 1 ACT）**：把 `candidate_universe`、
   `required_coverage` 与 `budget_cap` 分开；解析 Query 中 `all/across/some/worst-case/compare` 等量词，
   让 Planner 的 stop reason 明确区分 `evidence_sufficient`、`budget_exhausted` 和 `unsupported_gap`。
   先用 v4 evidence 与缓存反事实做确定性测试，不能先假定两轮就是充分。
2. **让公共 Proposal Agent 自然提出受控 SuccessSpec v2（0 ACT fixtures，live 后置）**：从开放 Query
   产生 scene 变化与非官方但 oracle-bounded 的新成功语义，通过正负 fixture、编译/差分 gate 和
   production acceptance；不能再由调用方预填全部语义。
3. **预注册 adaptive 与 fixed 的小型一致性/效率实验（先 N=1，再 N=3）**：同 Query、seed、checkpoint、
   最大预算和停止合同；同时报告样本数、墙钟、失败状态与结论一致性。
4. **独立人工 gold 与 VQA 鲁棒性**：补真实扰动、正负平衡、多人复核和预注册 scorer；不能用
   development-agent proxy 填论文 Tables 6–8。
5. **最后扩大任务、policy 与 repetition**：只有前述主链和评估合同稳定后，才支付论文规模实验。

## 4. 声明边界

- clean-head v4 是当前公共 Planner 的两轮完成态功能验收：首轮位置成功后的真实 evidence 触发切换
  instance。它不是泛化结论；两个实际样本均成功，也不能代表未测 `right_fixed`、`base1` 或未支持轴。
- v4 因 hard cap=2 结束；`completed` 表示执行合同完成，不等于 `evidence_sufficient`。当前首要方法 gap
  是让 Query 量词决定 required coverage，并与所有可选候选和预算上限解耦。
- v4 所有角色显式覆盖为 `gpt-5.6-terra`；它不证明默认多模型 profile 或 Luna 路径稳定。账本精确为
  11 个 logical provider calls、16 个 transport attempts、2 个 ACT starts，且没有 whole-round recovery/
  restart；transport 次数不能与逻辑模型调用数混写。
- v4 的完成态证据本身含 2 ACT，但本批新增 ACT starts 合计 4；clean-head v2/v4 累计 3 ACT，并重复
  左侧 position。不能用最终 bundle 的 2 ACT 代替开发成本，也没有 sampling savings 结论。
- 最新 strict pair 的 ACT/Tool/VQA 是 live，但 planner 仍是旧 task-specific path；exact agreement=`1.0`
  只表示两个重叠样本结论相同。N=1 不等于
  论文统计一致性，rollout savings=`0` 也不支持效率提升 claim。
- expert gate 只证明场景可解，不是 ACT 成功；缓存 telemetry replay 不是新 rollout。
- `safety.hammer_left_camera_contact` 只覆盖 `020_hammer ↔ left_camera` 的精确物理接触；通用
  `safety.unintended_contact` 仍 unsupported，不能把该 proxy 命名为完整 safety。
- invalid `SuccessSpec` → trusted fallback 证明诊断/恢复路径，不证明模型能生成任意正确成功函数。
- reviewed task 由 `development_agent` 审核，固定 `paper_table_eligible=false`；它不是独立人工 gold。
- reviewed reuse 固定的是 task.py、VariantSpec、overlay/load_actors、可选 SuccessSpec 和 validation/static
  等 immutable inputs，以及当前列出的 5 个 Python runtime dependencies。`TaskArtifactBundle` 与
  `SceneCheckSpec` 是 run-local derived rebuild，不能声称整个环境或全部产物按字节复现。
- registry-authoritative variant id 可以与当前请求的 run-local id 不同；复用依据是受信可执行语义和
  provenance，不是宽松字符串相似。Proposal Agent 对 VQA 绑定的有界修复也只是 capability-derived
  结构修复，不是模型完成了新的语义推理。
- partial route 只允许执行同一 task 中已支持的子集，同时把 appearance、mass、scale 等 task-qualified
  gaps 留在最终限制中；它没有把这些 axis 变成新 capability。
- 最终源码已补失败/restart attempt 的账本和 feedback `finally` 收口；v4 没有触发 recovery，不能把
  其 `11 logical / 16 transport / 2 ACT` artifact 当作新 restart 路径的 live 验收。
- `EvaluationGraph` 汇总多个任务专属 checkpoint 的 child；它不是单 policy 跨任务执行，也不是论文核心
  Plan Agent gap 的替代品。
