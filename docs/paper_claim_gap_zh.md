# ManipEvalAgent 论文主张与当前差距（自顶向下）

本文只按论文 Sec. 3.2–3.4 与 Figs. 2–5 审查主体方法，不把工程审计、hash、跨进程恢复或
更多任务数量误当作论文核心贡献。实验表格的规模复现另算；本项目当前是 ACT 为主、一个 DP3
单 seed 部署 pilot、两个任务族和少量可信 capability 内的受限功能原型，不是论文规模或开放世界
能力的完整复现。

## 0. 2026-07-23 路线纠偏

上一版路线仍然把 Query identity、AnswerScope、official/experimental authority 和 fail-closed gate
放在最前面。这些工作能改善工程可靠性，但不是论文的科学主张；把它们继续当作主线，会造成
“接口和护栏越来越完整，论文图表仍然没有新证据”的安全性陷阱。

此后路线从论文 Abstract、Introduction contributions、Sec. 4 的三个实验问题以及 Tables 1–9
自顶向下推导。工程护栏只作为 claim 实验的验收条件，不再单独计为复现进展。判断一次开发是否
值得优先做，首先问它会新增哪条论文 claim 的真实可反驳证据，而不是它是否又增加了一层约束。

### 0.1 batch19 实证增量

batch19 不再只是“协议可执行”：

- capability-conditioned 手工链完成三轮真实 ACT：template ID 与顺序隐藏，但 scale/appearance
  受控轴公开；各阶段由开发者串联，没有统一 VQA/Aggregate/Feedback runtime。post-budget clean
  seed1000 也失败，Planner 继续请求 seed1010 clean-vs-blue paired control，原 Query 未回答。
- standalone TaskGen 的生成 scene 与受限编译 `check_success()` 已在同一次 ACT rollout 合一，并以
  `generated_check_success` 与 official success 分开命名；主 Agent 仍阻止 experimental v2 ACT。
- Query-induced Tool v3b 已用受限 physical-time finite-difference DSL 生成 active/right-arm
  `precontact_peak_tcp_jerk`，在缓存真实 telemetry 上 validate/register，并完成 paraphrase/exact reuse；
  但 v3b 的 43.4348 数值来自 nonphysical event，已判无效。v3c corrected runtime 对原 Query/转述
  均返回 null/no-target-contact；旧 v2 inactive-arm 结果只作 superseded failure audit。
- 两候选 `N=1` v4 是 outcomes 已知后的 post-hoc cached-prefix counterfactual：actual rollout/wall
  saving 均为 null；1 次/50% 只是 counterfactual avoidable count，82.306 s 只是 estimate。
- ACT 与 DP3 在 official BBH 同 seed 各执行 1 次且都失败，排序为 tie、Spearman 不可定义。
- Plan/VQA、Table 3 与 error distribution 现在有真实或 proxy 分母，但仍分别受
  development-agent gold、proposal-only 且每 condition N=1、开发 operation universe 限制。

完整边界见 [batch19 开发记录](development_log_20260723_batch19_claim_evidence_zh.md) 与
[紧凑证据包](evidence_runs/batch19_claim_evidence/)。

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
| 1 | Plan Agent 根据 `Query + Y1:t` 动态发现 sub-aspect（Sec. 3.2；Figs. 2/5） | batch19 隐藏 template ID/顺序，用手工链把三轮 failure 依次传给单步 Planner；预算外 clean seed1000 anchor 也失败 | 受控 axes 仍公开；不是统一 Agent run，也无 VQA/Aggregate/Feedback。clean failure使 property attribution 不成立，Planner 又请求 seed1010 paired control，Query 仍未回答 |
| 2 | TaskGen 对 Proposal 做 retrieve-or-generate，并交付 runnable scene + `check_success()`（Sec. 3.3；Fig. 3） | standalone TaskGen 首次把生成 scene、受限编译 SuccessSpec、expert 正例与 1 次 ACT 放进同一 artifact，outcome 标为 `generated_check_success` | 主 Agent 仍阻止 experimental v2 ACT；只有一个 bounded appearance Proposal，也没有 target+distractor、多 proposal 成功率或 library reuse |
| 3 | TaskGen 用 render/视觉反馈诊断并修复生成错误（Fig. 3；App. A.3.4） | scene reflection、SuccessSpec repair 与 expert gate 已有；相关 stage 使用 typed diagnosis/action 做最多一次局部 repair/regenerate，并保证 accepted 前 0 ACT、policy failure 不重试 | 论文使用 stage-specific recovery，并不要求所有阶段共享一个中央 controller。当前真实 visual repair 和 fixture recovery 的覆盖仍窄；click overlay 与大多数正常路径仍以 validate/accept 为主 |
| 4 | ToolGen 根据 Tool Proposal 检索、生成、验证、注册与复用（Sec. 3.3；Fig. 4） | v3b 从自然 Query 选择受限 DSL `precontact_peak_tcp_jerk`，并完成 synthetic oracle/run-local register/semantic reuse；v3c corrected cached execution 返回 null/no-target-contact | bounded route 单例成立；validated_episode_count=0、threshold 未校准、43.4348 数值无效。没有 policy jerk evidence、Agent/VQA 或 live confirmation |
| 5 | Rule Tool 与 Dynamic VQA 互补观测执行结果（Sec. 3.3；Fig. 4） | telemetry/events/video → Tool/VQA plumbing 已有；8 条 smoke 覆盖四个视觉 condition 名称 | 每个 condition 使用不同 proposition，样本手选且未预注册；development-agent gold、单一 cached VLM。AUROC=1.0 不是 robustness estimate |
| 6 | Aggregate 后的 evidence 改变后续规划并形成 Query-centric feedback（Secs. 3.2–3.4） | batch19 手工 evidence chain 先 drill down 到反向 scale，再 switch 到 appearance；post-budget clean failure 后又请求 exact paired control | evidence JSON 由开发者串联，没有统一 Aggregate/Feedback；Planner 未在三轮内安排 control，且 post-budget 后仍不停止 |
| 7 | 方法可在多个 concern/task 上使用（Fig. 2） | BBH 与 click_bell 共用 planning/evidence/proposal/runtime 合同；每个 child 固定自己的 ACT checkpoint | materializer 与真正可执行 aspect 仍少。跨任务 graph 只是一种 portfolio 编排，不应被提升为高于单任务动态闭环的核心 gap |
| 8 | 少样本结论与 benchmark 一致，并评估 Planner/VQA（Tables 1–3、6–9） | 两候选 v4 只做 post-hoc cached-prefix protocol；ACT/DP3 各 0/1；另有 5-query/8-clip protocol smoke、7-call proposal ablation、23-operation error audit | 没有 observed saving、独立 arms 或 dense benchmark；双 policy tie 无 Spearman；无 5 policy、N=3/5、独立人工 gold 或真实 codegen ablation |

## 3. 论文真实声称、但当前项目还没有的证据

下表只列论文自己的方法或实验 claim，不把本项目新增的安全合同列为贡献。

| 层级 | 论文 claim | 论文证据 | 当前项目状态 |
| --- | --- | --- | --- |
| 核心结论 | 用显著更少的时间和 samples，仍得到与完整 simulation benchmark 可比的结论 | Abstract；Tables 1–2、4–5；每个 constructed task 默认 5 trials，ManipEvalAgent 完整流程重复 10 次 | **只有 post-hoc 协议 demo**：outcomes 已观察后构造 cached prefix；actual saving=null。没有预注册独立 arms、测得 wall speedup、dense benchmark、N=3/5 或 10 repetitions |
| 核心结论 | 少样本评估仍能保持五个 policy 的相对排名 | Table 9：ACT、DP、DP3、RDT、π0；RoboTwin/LIBERO clusters；10/20/50 rollouts；Spearman 与 bootstrap CI | **仍未复现**：ACT/DP3 同 seed 各 0/1，observed rank tie、Spearman=null；DP、RDT、π0、更多 seeds 和 clusters 缺失 |
| 核心方法案例 | 开放 Query 在每轮依据 observations 动态选择或细化 sub-aspect，并在需要时现场生成 task/tool | Secs. 3.2–3.4；Figs. 2、5 | **部分真实存在**：手工 chain 会改变下一 probe，但不是统一 runtime；post-budget clean failure 否定 property attribution；v3b Tool 是独立 bounded DSL cached-real 案例，未进入同一 Agent/VQA chain |
| 组件有效性 | TaskGen 生成 scene + `check_success()`，RAG、visual self-check、README.Agent 提升生成成功率 | Fig. 3；Table 3：complete 98%，各消融 93–96% | **standalone 单例能力存在，统计缺失**：scene+checker 在 standalone 路径评价 1 ACT，主 Agent 仍阻止 v2；7-call 只是 proposal prompt ablation，不是 codegen/downstream success |
| 组件有效性 | ToolGen 能现场生成、验证、注册并复用新 rule tool，RAG 提升成功率 | Fig. 4；Table 3：complete 96%，w/o RAG 92% | **bounded route 单例存在，policy evidence 缺失**：v3b 完成 generate/oracle/register/reuse，但 numeric result 无效；v3c corrected 为 null/no-contact。无 Agent/VQA/live confirmation，proposal-only ablation 也不能估计 RAG 效果 |
| 有效性 | Plan Agent 对开放 Query 的 sub-aspect 分解与机器人研究者高度一致 | Table 6；几百条 query dataset；4 名机器人方向 annotator majority、senior tie-break；三种 planner model | **论文证据缺失**：n=5 是 legacy 20260717 taxonomy-routing proxy plumbing，不是 batch19 ClaimFirst 输出，也不是独立 human gold |
| 有效性 | VQA 在 clean、clutter、background texture、lighting 下与人工一致且 AUROC 稳定 | Tables 7–8；真实 clips；三种 VLM；人工二值 gold | **只有 protocol smoke**：8 条 heterogeneous hand-selected cached predictions，preregistered=false；AUROC=1.0 不能解释为视觉 robustness |
| 稳定性 | 整体约 5% 流程受错误影响，并可分解到 Plan/TaskGen/ToolGen/simulator/others | Fig. 6；App. A.1.3 的人工/fixture 错误计数协议 | **只有开发 audit**：同一冻结 23-operation universe retrospective 重算为 8 errors=34.78%；另有 2 个 post-universe Tool failures 不进分母。分母和重复协议不等于论文 |
| 范围 | multi-task VLA 上仍显著节省时间并保持 benchmark conclusion consistency | Tables 4–5；RDT、π0；RoboTwin 与 LIBERO | **完全缺失**：没有 RDT/π0 checkpoint、多任务 suite 或 LIBERO 实验 |

## 4. batch19 前的 Claim-first 候选（实施历史）

下列 A–G 是 batch19 的输入计划，不是当前待办清单。A 得到手工 capability-conditioned chain，
B 得到 standalone scene+checker ACT，D 得到 cached-prefix v4，E 得到 development proxy，
F 只得到 7-call proposal prompt ablation，G 得到 ACT/DP3 adapter pilot；C 的 v2 Tool 因
inactive-arm bug 已撤回，v3b 已用 bounded DSL active-arm 版本替代。当前缺口见 §0.1、§2–3 和 §6。

### A. Fig. 5 完整开放 Query flagship

- **目标 claim**：开放 Query + 中间 observations 真正决定下一轮，并现场生成任务和工具。
- **最小实现**：只给一句无内部路线提示的 object-generalization Query；最多三轮。Planner 自主选择
  初始 sub-aspect；至少一轮生成新的 scene + `check_success()`，至少一轮产生当前 registry 中没有的
  rule metric；上一轮 ambiguous evidence 必须触发更细 sub-aspect。
- **真实预算**：每轮 ACT `N=1`，合计 1–3 ACT；TaskGen/ToolGen provider 和 simulator 正常运行。
- **成功判据**：一个不可拼接的 artifact 同时包含 Query、三轮 Proposal、生成代码/render、ACT、
  Rule/VQA、Aggregate、evidence-conditioned next proposal 和最终回答。
- **地位**：最高优先级；它直接补 Figs. 2/5，而不是再补外围合同。

### B. Proposal-derived TaskGen + `check_success()` 真实 rollout

- **目标 claim**：TaskGen 对一个此前不存在的 Proposal 同时生成 scene 与成功判据，并在视觉修复后
  真正用于 policy evaluation。
- **最小实现**：为 BBH 生成一个含目标 block 与物理 look-alike distractor 的新场景；生成的 checker
  要求击中目标且不误击 distractor；首帧检查能发现重叠、错误 actor 或错误位置。
- **真实预算**：约 4–7 次 provider、3–5 次 simulator/expert probe、1 次 ACT。
- **成功判据**：新 `task.py` 真实实例化两个物理对象；checker 对 expert 正例、误击负例和未击中负例
  判定正确；ACT rollout 直接调用这个生成 checker，而不是继续继承 official success。

### C. Query 触发一个此前不存在的 Tool

- **目标 claim**：ToolGen 从新评估需求生成、验证、注册工具，并在下一 Query 中检索复用。
- **最小实现**：Query 只描述“接触前是否存在明显抖动或急动”，不指定 metric 名；ToolGen 生成
  `precontact_jerk_peak`，用 smooth、oscillatory、missing-contact 三个 oracle trajectory 验证，再在
  真实 telemetry 上执行，并配套一个可见抖动 VQA。
- **真实预算**：先在缓存 telemetry 上 0 ACT 完成 generate/validate/register；再用 1 次 ACT 做 live
  confirmation；第二次 Query 应 0 codegen 直接复用。
- **成功判据**：metric 在 Query 前不在 catalog；生成代码通过独立 oracle；Rule 与 VQA 一同返回
  Planner；下一 Query 从 registry 命中同一实现。

### D. Tables 1–2 的单 policy 小型真比较

- **目标 claim**：少量 adaptive sampling 在成本更低时仍得到与完整 benchmark 相同的能力结论。
- **最小实现**：click_bell 的 left、right、base0、base1 形成 dense reference；fixed 全跑四项，
  adaptive 最多三轮；比较 object-generalization verdict、最弱 axis、rollout 数和 wall-clock。
- **真实预算**：路径验证 `N=1` 为 fixed 4 + adaptive 1–3，共 5–7 ACT；随后放大到 `N=3`
  的 15–21 ACT 和论文 constructed-task 默认 `N=5` 的 25–35 ACT。
- **成功判据**：`N=3/5` 时 adaptive 与 dense reference 对强弱 axis 和总体 verdict 一致，同时真实
  rollout 与 wall-clock 至少降低 25%；若 adaptive 跑满或结论不一致，就判定当前设置未复现该 claim。

### E. Table 6 与 Tables 7–8 的真实 human validity

- **目标 claim**：Plan sub-aspect 和 VQA evidence 都与独立机器人研究者判断一致。
- **最小实现**：Plan 先用五类 taxonomy 各 4–6 条 Query；VQA 为 clean/clutter/background
  texture/lighting 各收集正负 clips；均由 4 名机器人方向 annotator 独立盲标、majority vote、
  senior tie-break。冻结 prompt 后运行三个 planner/VLM model。
- **真实预算**：0 ACT 起步可复用视频；Plan 约 20–30 × 3 provider calls；VQA 先做 8–16 clips，
  不足条件再补 simulator rollout。
- **成功判据**：Plan precision/recall/F1、inter-annotator agreement，以及每个视觉条件的
  accuracy/AUROC；不再使用 development-agent proxy。

### F. Table 3 真实 code-generation ablation

- **目标 claim**：RAG、visual self-check、README.Agent 确实提高 TaskGen/ToolGen 成功率。
- **最小实现**：同一 unseen Proposal 在 complete、base、w/o RAG、w/o visual self-check、
  w/o README.Agent 下真实生成；由不知道 condition 的 reviewer 检查 task code/render/tool oracle。
- **真实预算**：路径验证为 1 proposal × 7 个论文 condition；最小经验批次 `N=5` 约 35 次 generation；
  论文差值仅 3–5 个百分点，真正估计差异可能需要约 20 cases/condition。
- **成功判据**：预先冻结 correct/reasonable/aligned 标准，输出真实分母、成功率、CI 与失败案例；
  “开关能运行”不能代替消融效果。

### G. Table 9 五 policy ranking

- **目标 claim**：agent 少样本评估保持 policy 相对排序。
- **依赖**：必须先具备 DP、DP3、RDT、π0 的可运行 checkpoint 与对应 task/benchmark adapter。
- **真实预算**：最高；先做 ACT/DP/DP3 三 policy pilot，再扩五 policy 和 10/20/50 rollout budget。
- **成功判据**：从真实 scores 计算 Spearman 与 bootstrap CI，不能用合成 ranking。

推荐依赖顺序是 **A → B → C → 把 A/B/C 合成一次完整 Fig. 5 flagship → D (`N=1→3→5`)**；
E、F 可在不大量消耗 ACT 的情况下并行准备，G 在多 policy checkpoint 就绪后执行。科学重要性上
Tables 1–2/9 最高，但在 A/B/C 尚未成为真实方法能力前直接跑效率比较，比较到的仍会是受限脚本而
不是论文的 ManipEvalAgent。

## 5. 声明边界

- batch19 open Query 的三轮 ACT 是开发者手工串联的 capability-conditioned chain：template ID/顺序
  隐藏但受控轴公开，无统一 VQA/Aggregate/Feedback。post-budget clean seed1000 也失败，因此
  1.2x failure 不是 property-specific contrast；Planner 又请求 seed1010 exact control，Query 未回答。
- batch19 generated scene/checker 只在 standalone TaskGen 路径运行；主 Agent 仍阻止 experimental v2。
  runtime label 是 `generated_check_success`，authority 是
  `compiled_success_spec_experimental_bounded`，从不等于 official success。
- batch19 efficiency 是 outcomes 已观察后的 post-hoc cached-prefix counterfactual，不是独立运行。
  actual rollout/wall saving 均为 null；1 次/50% 只是 counterfactual avoidable count，82.306 s 是 estimate。
- batch19 Query-induced Tool v3b 是 bounded DSL、standalone、cached-real 案例；它支持一个受限
  generate/validate/register/reuse 功能 claim，但不支持任意 Python codegen、集成 Agent/VQA 或 live
  confirmation。v2 inactive-arm artifact 是 superseded failure audit。
- ACT/DP3 exact-seed pilot 是 0/1 对 0/1；tie 不支持参考顺序，也不支持“顺序相反”。Spearman=null
  的正确结论是 inconclusive。
- proxy validity、7-call ablation 和 23-operation error audit 的明确 scope 分别是
  development-agent proxy、每 condition N=1 proposal prompt proxy，以及 frozen operations 的
  retrospective status review；后续两项 Tool failure 单列、不进分母。
- clean-head v4 是当前公共 Planner 的两轮完成态功能验收：首轮位置成功后的真实 evidence 触发切换
  instance。它不是泛化结论；两个实际样本均成功，也不能代表未测 `right_fixed`、`base1` 或未支持轴。
- batch18 P1/P4 对 v4 的回放是 cached 0-ACT evidence：完整四候选 suite、预算 3、已完成 2，结果为
  `inconclusive/continue`。它没有改变 live Planner，也不是新 ACT/Feedback run。
- batch18 P2 是真实 provider/simulator/expert 的 TaskGen acceptance，但 ACT=0 且
  `experimental_v2_act_runtime_eligible=false`；生成的 experimental `check_success()` 不能报告为
  official policy success。
- batch18 P3/P5 的发布结果均是 synthetic fixture。`2:1` savings、accuracy/AUROC 只证明协议算术，
  不能进入论文表格；现有 development-agent/proxy labels 不是 human gold。
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

## 6. batch19 后的首要 gap

| 顺序 | 第一性原理缺口 | 最小下一步 | 论文 claim |
| ---: | --- | --- | --- |
| 1 | 现有三轮只是手工 capability-conditioned chain；post-budget clean failure 后仍需另一 paired control，说明控制实验调度过晚 | 先把 plan→task→rollout→VQA/Rule→Aggregate→next plan→Feedback 接进一个 runtime；用缓存反事实强制 anchor-first/paired-control-before-attribution，再最多 1–2 ACT | Sec. 3.2；Figs. 2/5 |
| 2 | scene+checker 已合一，但 checker 语义仍只是近接触+接触，没有证明多对象区分 | 生成 target + physical distractor；expert 正例、误击与未击中负例，最后 1 ACT 直接由新 checker 评价 | Fig. 3 |
| 3 | bounded Tool 尚未进入 Agent/VQA，且 v3c 的 policy execution 因无物理目标接触而为 null | 接入 v3c 的 physical-contact gate 与显式 null 语义；先让 VQA/Aggregate/Planner 正确消费“无可测证据”，再最多 1 ACT 做有真实目标接触的 live confirmation；不得消费已失效的 v3b 数值，也不扩成任意 Python codegen | Fig. 4 |
| 4 | v4 是 post-hoc cached-prefix protocol，未产生 observed saving | 预注册后独立运行 fixed/adaptive arms，再固定 suite/order/seeds 做 N=3；报告 verdict、rollouts 与实测 wall time | Tables 1–2 |
| 5 | ACT/DP3 single-seed tie 对 ranking 没有信息 | 预注册少量 exact seeds 或接入一个轻量 DP checkpoint；禁止看到结果后挑 seed。RDT/π0/LIBERO 继续后置 | Table 9 |
| 6 | proposal prompt ablation 没有生成或执行 task/tool code | 每 condition 至少 5 个 matched unseen cases，纳入 codegen、compile/render/simulator/oracle，冻结 blind success rubric | Table 3 |
| 7 | error audit 的分母是开发 operation，不可和论文约 5% 直接比较 | 预注册独立 run/case universe、stage attribution 和 cutoff，重复后再报告 CI 与模块占比 | Fig. 6 |
