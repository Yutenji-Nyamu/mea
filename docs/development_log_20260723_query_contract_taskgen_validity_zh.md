# 2026-07-23：Query contract、TaskGen v2 与有效性协议

本批目标不是再增加一个“两轮成功”的 demo，而是给此前最重要的 P1–P5 gap 建立最小可执行合同，
并把缓存、真实 TaskGen、真实 ACT 和合成协议严格分层。

## 1. 本批实现

### P1：Query-conditioned evidence sufficiency

新增：

- `mea/planner/query_contract.py`
- `scripts/manipeval_query_sufficiency.py`
- `tests/manipeval/test_query_sufficiency_contract.py`
- `tests/manipeval/test_query_sufficiency_cli.py`

合同显式包含 `claim_type`、`candidate_universe`、`required_coverage`、`round_budget` 和可选 comparison
groups。`universal`、`existential`、`comparative`、`diagnostic` 使用不同停止逻辑；冲突证据不会被当成
充分证据。停止原因只有 `evidence_sufficient`、`budget_exhausted` 或 `continue`。

当前 CLI 是严格的 cached/offline 0-ACT 入口：它绑定真实 `BoundTaskPlanSession`、task catalog 和
预算，但不会修改 live Planner。用 clean-head v4 两轮证据回放完整四候选 suite 后，结果为
`inconclusive/continue`，`right_fixed` 与 `base1` 未测，预算只剩一轮。

### P2：公共 Proposal → scene + experimental SuccessSpec v2

公共 `BoundedProposalAgent` 新增显式 `experimental_success_bounded` capability，只允许：

```text
beat_block_hammer / object_appearance.color / force_codegen
```

TaskProposal v2 同时给出 bounded scene change 与 experimental SuccessSpec。核心 validator、TaskGen、
artifact bundle、standalone Proposal CLI 与 production acceptance 都绑定同一 capability；其他 task/aspect
组合和绕过 ProposalAgent 的 v2 输入 fail closed。

真实服务器验收：

- provider calls：3；
- simulator probes：2；
- expert probes：1；
- ACT starts：0；
- setup/render/rule/expert/VariantSpec/artifact bundle：全部通过；
- production acceptance：`task_generation_only_no_act`；
- scene：紫蓝色 block；
- SuccessSpec：hammer/block 功能点 XY 距离分别 `<0.025m` 且发生物理接触。

主 Agent 与 standalone TaskGen 都明确禁止 experimental v2 ACT，直到 official 与 experimental outcome
具有不同运行时字段。fresh/resume ACT 都以最终重建的
`TaskArtifactBundle.success_semantics.act_runtime_eligible` 为权威；manifest 中缺失或陈旧的
TaskProposal 副本不能绕过。这个限制防止 experimental predicate 被误写成 official policy success。

### P3：matched efficiency preregistration

新增：

- `mea/matched_efficiency_protocol.py`
- `scripts/manipeval_matched_efficiency.py`
- `tests/manipeval/test_matched_efficiency_protocol.py`

协议 fail closed 地匹配 Query、checkpoint hash、candidate suite/order、seed、最大预算与 P1 contract，
并分开记录 ACT starts、completed trials、policy steps、expert/probe、provider logical/transport/retry、
wall time 与 evaluation samples。synthetic `2:1` 与 `2:2` 只验证算术和零节省分支；
`empirical_policy_claim_eligible=false`。

### P4：AnswerScope 与过度声称 guard

新增：

- `mea/feedback/answer_scope.py`
- `tests/manipeval/test_answer_scope.py`

`FeedbackAgent` 现在必须携带结构化 N、seed、tested/untested candidates、unsupported capabilities、
evidence conflict、termination 与 claim verdict。缺字段、漏掉强制 limitation，或文本反称“证据充分/
所有候选已测/没有 unsupported/证据一致”，都会 fail closed。

缓存 v4 的投影结果为 N=2、唯一 seed 100502、两个未测候选、五类 unsupported capability、
`inconclusive/continue`。这是旧 live evidence 的 0-ACT projection，不是新 live Feedback artifact。

### P5：独立有效性协议

新增：

- `mea/independent_validity.py`
- `scripts/manipeval_independent_validity.py`
- `tests/manipeval/test_independent_validity.py`

协议支持显式 rater role、pairwise agreement、majority vote、senior tie-break、正负 VQA controls，以及
clean/clutter/background texture/lighting 四条件下 accuracy/AUROC。当前发布结果完全是 synthetic
fixture；development-agent 明确不是 human gold，也不满足论文 4 名机器人研究者的配置。

## 2. 正式验收位置与结果

canonical 开发与验收环境是 AutoDL：

```text
/root/autodl-tmp/mea
/root/autodl-tmp/conda/envs/RoboTwin/bin/python
```

正式结果：

```text
Ran 597 tests in 53.764s
OK
```

本批开发中曾在 Windows staging 跑过 46 个纯 Python 定向测试以快速定位集成问题，但这不符合项目
“本地不作为验收”的既有约定，因此不再把 `46/46` 列为本批验收结果。Windows 只用于源码阅读、
轻量编辑、changed-file 编译/静态检查与 `git diff --check`；Python 单元/集成、RoboTwin import、
provider、TaskGen、expert、simulator 和 ACT 均以 canonical AutoDL 结果为准。

## 3. 紧凑证据

仓库内发布：

- [`batch18_contracts/`](evidence_runs/batch18_contracts/)

其中 P1/P4 是 cached real replay，P2 是真实 provider/simulator/expert 的 0-ACT TaskGen acceptance，
P3/P5 是 synthetic functional fixture。原始 generated task、validation log 与 probe machine audit 仍在
服务器 ignored 目录，不进入 Git。

## 4. README 与文档边界

根 `README.md` 已恢复为本批开始前的简洁版本。实现细节只进入：

- `docs/index_zh.md`：入口与职责；
- `docs/running_guide_zh.md`：执行位置和命令；
- `docs/architecture_and_dataflow_zh.md`：真实调用链边界；
- `docs/paper_claim_gap_zh.md`：论文 claim 与 gap；
- 本 development log 与 compact evidence bundle：批次事实。

## 5. 重读论文与项目后的第一性原理结论

本批补齐的是协议与 guard，不是统一 empirical chain。当前首要缺口按因果依赖排序为：

1. **同一条 hash-bound Query identity chain**：原 Query、checkpoint、完整 routed suite、预算、
   unsupported axes、CandidateEvidence、P1 stop、P4 answer 与 P3 comparison 必须共享同一身份；当前 P1
   仍是 offline CLI，live Planner 尚未消费它。
2. **official / experimental 双通道执行**：先在缓存 telemetry 上并列计算 official success 与 compiled
   experimental SuccessSpec，再最多跑 1 ACT；任何情况下都不能把 experimental outcome 改名为 official。
3. **真实 matched pilot**：同一预注册合同先做 fixed 2 + adaptive 1；若 adaptive 也需要 2，诚实报告
   2:2、节省为 0。稳定后才扩到论文默认 5 trials/task 与 10 agent runs。
4. **真实独立有效性**：导入带 hash 的四条件 clips，4 名机器人方向 annotator、majority vote、senior
   tie-break 和预注册阈值；当前 proxy/synthetic 只能验证协议形状。
5. **外部有效性最后扩展**：第三任务、更多 checkpoint/policy、新 metric 必须放在上述身份与统计合同
   闭合之后，否则只会复制同一语义缺口。

论文 Sec. 3.2 只说根据 Query 与累积证据动态选择 sub-aspect、证据充分后停止，并没有定义本批的量词
truth table；QuerySufficiencyContract 是可靠性扩展。论文的主要科学 claim 仍是更少 sampling 下与
benchmark 结论一致，本批没有新增支持该 claim 的真实实验。
