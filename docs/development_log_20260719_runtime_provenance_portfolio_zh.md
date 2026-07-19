# 2026-07-19：运行时 provenance、真实视觉修复、scene seed2 与跨任务报告

## 1. 本批目标与论文对应

本批仍只使用 ACT，并把真实 rollout 控制在最小规模。目标是同时补论文主体方法的可审计性和
自顶向下的最终输出，而不是扩大 benchmark：

| 实现 | 论文对应 | 本批证据边界 |
| --- | --- | --- |
| call-start ledger + round provenance | Fig. 3、App. A.3.4 | 外部调用开始前落盘，轮计划与 child/Tool/VQA/recovery 做 hash 绑定；不代表 policy 成功 |
| scene error → visual diagnosis → repair | Fig. 3、Sec. 3.3.1 | 真实 RoboTwin render/VLM/repair/expert，0 ACT；不是 TaskGen 成功率 |
| live-provider matched micro | Table 3 | 4 次 provider、0 ACT、开发代理审核；不是论文人工消融 |
| scene condition 两独立 seed | Tables 7–8 | texture/light 各 2 seed，开发代理看真实 montage；无正负可见性平衡，不能算论文指标 |
| 一个 Query 的 click_bell + BBH | Fig. 2、Secs. 3.2–3.4 | 两个受信 task adapter、各 1 ACT、确定性最终报告；不是任意任务图 |

## 2. 实现摘要

### 2.1 调用开始账本和轮级 provenance

`mea/runtime_ledger.py` 在 provider HTTP transport 和 ACT subprocess 启动前 append + `fsync`，记录
evaluation、logical round、attempt、child、调用类型和最小计数；不保存 prompt、图片、URL、key、
checkpoint 路径或内容。ledger 写失败会阻止外部调用。provider logical call 与 transport retry、ACT
started 与 completed 分开统计，进程中断后也不会把“没有结果文件”误报成“没有调用”。

`mea/round_provenance.py` 为每轮 exclusive-create sidecar，绑定 round plan、去除自引用后的 round
summary、child manifest、VariantSpec/reflection、ACT、TaskGen command、Tool、Aggregate、Execution
VQA、recovery 和 ledger。严格 verifier 会重算每个文件 hash 与 binding hash；provenance 只证明证据链
身份和完整性，不证明结果正确。

真实实验暴露一个混合事件 bug：同一 ledger 先有 ACT event，随后记录 provider event 时，旧代码直接
访问每一行的 `logical_call_id`，在 ACT 行触发 `KeyError`。失败发生在 VQA HTTP 调用前；scene、expert、
ACT、Tool 和 Aggregate 已完成。修复先按 `event_type` 过滤，并增加“ACT 后接 provider”回归测试；旧
失败 evaluation 保留为 started-accounting 证据，没有原地修改或冒充 completed case。

### 2.2 真实 scene error → diagnosis → repair

`run_20260719_batch10_real_scene_repair_9be751d` 在 BBH blue-block 任务正常 static gate 后注入结构合法但
语义错误的 red fixture。第一次真实 RoboTwin render 被 VLM 判为 red（confidence `0.98`），随后保存
typed diagnosis/suggestion，安装受保护 repair，重新做 diff/AST/static gate；第二次 render 判为 blue
（confidence `0.94`），expert gate 在第 2 attempt 通过。完整 transition 为：

```text
static_pass → visual_reject → diagnosis → repair_installed
→ static_revalidate_pass → visual_pass → expert_pass
```

本路径调用 provider 6 次、simulator 两次视觉 probe、ACT 0 次，证明 Fig. 3 的视觉反馈修复控制流真实
运行；它不产生 policy 结果或 TaskGen 成功率。

### 2.3 live-provider Table 3 micro

`batch10_table3_live_9be751d` 将 generation 与 review 分成两个 append-only 阶段：先由 live provider
生成 matched candidates，success 保持 `null`；再由开发代理审核保存独立标签。两组 matched pair 是
TaskGen complete/no-RAG 和 ToolGen complete/no-RAG，共 4 次 provider、0 simulator、0 ACT。代理审核为
3/4 通过：TaskGen 两项均通过；ToolGen complete 通过，no-RAG 因把要求的顶层合同嵌进
`output_contract` 而拒绝。单个 matched micro 不能支持 RAG 因果结论，固定
`paper_table_eligible=false`。

### 2.4 scene seed2 coverage 与代理标签

旧 seed `100402` 与新 seed `100403` 的真实 simulator-native artifact 被离线 collector 一起审核。
结果为 candidate `4/4` ready、diagnostics `0`、无重复 condition-seed identity：

| Condition | 独立 seeds | ACT 结果（100402 / 100403） | 代理视觉标签 |
| --- | --- | --- | --- |
| unseen background texture | 2 | fail / fail | bell 均清晰可见，均无可见按压 |
| static randomized lighting | 2 | pass / pass | bell 均清晰可见，均有可见按压 |

新 seed 的 lighting ACT time-to-success 为 `21.36 s`；texture 执行 400 policy steps 后失败。四张 montage
由开发代理实际查看后才写标签，没有从被测 VQA prediction 推导 gold。collector 输出
`emitted_unvalidated`；严格 validator 又正确拒绝，因为每个 condition 的 primary visibility 都只有
`true`，没有正/负平衡。因此本批完成真实两-seed coverage，不声称 VQA accuracy、AUROC 或 Tables 7–8。

### 2.5 一个开放 Query 的跨任务父层

`portfolio_batch10_cross_task_ecbf7b1` 从一个 Query 生成 inert command plan，再严格顺序启动两个
普通 Agent child：`click_bell/object_position.left_fixed`（seed `100404`）与
`beat_block_hammer/object_appearance.color_blue`（catalog seed `100000`）。每个 child 的硬上限为
1 round、1 ACT，recovery budget 为 0；两条 pipeline 均 full pass，但 ACT 均为 `0/1`。

父层在 0 新 runtime 的 `reuse` 阶段重新核验 child outcome、逻辑 checkpoint 合同、call-start
ledger 与 round provenance，得到 provider started `14`、ACT started/completed `2/2`，两个 child 都是
精确 ledger accounting。最终报告没有把 pipeline pass 当策略强项：它明确写“0 个任务有非零成功、
2 个任务有失败证据”，将两个任务都列为 weakness，建议各补一个预注册 seed，并保留小样本、
ACT-only、checkpoint bytes 未 hash-bound 等局限。

操作上，这两个 child 确实由同一 shell 中的 command plan 立即启动；但 child manifest 尚未保存
`portfolio_id + command_plan_sha256` 的反向绑定，所以 post-hoc verifier 不能独立证明因果启动，仍
保守标记“reuse evidence is not causal”。这是下一批最容易且重要的父层 provenance gap。

## 3. 验证与真实调用预算

- 服务器全套单元测试：修复后 `401/401` 通过；
- real scene repair：0 ACT，6 provider calls，visual reject/repair/revalidate/expert 全通过；
- live Table 3 micro：0 ACT，4 provider calls，开发代理审核 3/4；
- scene seed2：新启动 3 个 ACT，其中 1 个是修复前已完成但 VQA 未发出的失败链，修复后重新启动
  texture/light 各 1 个；最终 suite 使用 2 个新 completed case；
- scene collector：4/4 ready、0 diagnostics、2 conditions × 2 unique seeds；suite draft 未验证；
- 跨任务 portfolio：2 ACT，两个 child 均 pipeline full pass、policy `0/1`；父层核验 provider
  started `14`、ACT started/completed `2/2`，并生成强项、弱项、建议和局限。

大文件约束保持不变：checkpoint、视频和运行 artifact 全部留在服务器；本机只临时读取 4 张小型
montage（每张小于 0.5 MB）用于代理标注，没有下载 checkpoint、数据集或 rollout 视频。

## 4. 自顶向下状态判断

论文主体从开放 Query 到最终回答的模块和数据流为：Proposal/Plan → reuse-first TaskGen → visual
self-reflection/repair → reuse-first ToolGen → ACT rollout → Rule Tool + dynamic VQA → deterministic
Aggregate → evidence-driven next plan/stop → strengths/weaknesses/recommendations/limitations。当前项目已在
少量受信任务上为这些节点提供真实或严格标记的最小通路，因此可以称“主体方法的小规模功能实现
基本完整”。

仍不能称完整论文复现：task/capability 覆盖少、跨任务父层仍是两个固定 adapter、只有 ACT、实验
N≤2、人工环节由开发代理代替、Table 3 只跑两个 matched pair、Tables 7–8 缺真实困难负例与 AUROC，
也没有论文规模 baseline、repetition 或多 policy 排名。

## 5. 下一批候选（自顶向下）

从论文完整概念向下检查后，推荐下一批按以下顺序取舍：

| 优先级 | 最小实现 | 论文对应与重要性 | 成本 |
| ---: | --- | --- | --- |
| 1 | child 保存并核验 `portfolio_id + command_plan_sha256 + parent_query_sha256`，父/子双向绑定 | Fig. 2 的 Proposal→Execution 因果链；当前最容易消除的证据缺口 | 低，0 ACT，可重放现有 fixture |
| 2 | 把固定两任务 portfolio 抽象为小型 `EvaluationGraph`；每个 aspect/task 节点必须是 executed、unsupported 或 budget-stopped，并用前一 child 的 typed outcome 决定下一节点 | Fig. 2、Secs. 3.2–3.4；当前最大的主体方法 gap 是“全局跨任务证据驱动规划”仍为预排 | 中，先对现有 child 做 0-ACT replay，稳定后最多 1–2 ACT |
| 3 | 抽象通用 `TaskGenRepairAdapter`，让 codegen BBH 与 declarative click_bell 共用 reject→diagnose→repair/regenerate→revalidate 合同 | Fig. 3、Sec. 3.3.1；把一次 BBH 特例提升为方法级合同 | 中，0 ACT，最多一次 simulator probe |
| 4 | live Table 3 从当前两个 matched pair 补齐其余开关，并保持 generation/review 分离 | Table 3；实验完整度高但不是主体闭环阻塞项 | 低，约 3 次 provider、0 ACT |
| 5 | 为 texture/light/clean/clutter 采集真实困难负例，再由独立人工替换代理标签 | Tables 7–8；补 accuracy/AUROC 的必要条件 | 中，约 2 ACT + 人工；放在方法 gap 之后 |

推荐下批实现 `1 + 2` 的 0-ACT 版本，并顺带做 `3` 的共享合同；若 replay 稳定，再决定是否用 1–2
ACT 做一次真正的跨任务自适应 Query。第 4、5 项属于实验逼真度，暂不应挤占主体方法优先级。
