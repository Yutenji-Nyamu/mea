# Batch 18：Query contract、TaskGen v2 与有效性协议证据

本目录只发布本批最小、可审计证据。它混合了三种不同等级，不能合并解读：

- `cached real replay`：消费已有 clean-head v4 的两轮真实 ACT 结果，不启动新 ACT；
- `live TaskGen acceptance`：真实 provider、simulator 与 expert probe，但明确为 0 ACT；
- `synthetic functional fixture`：只验证协议计算，不是策略、效率或 VQA 有效性证据。

## 1. P1 + P4：Query sufficiency 与 AnswerScope

输入 Query 对应的 frozen supported suite 为：

```text
object_position.left_fixed
object_position.right_fixed
object_instance.base0
object_instance.base1
```

原始 Query 见 [`query.txt`](query.txt)。当前 contract/CLI 尚未 hash 绑定该文本；这正是下一条统一
identity chain 要补的边界。

合同将它解释为有限域 `universal` claim，但 round budget 只有 3。缓存 v4 已观察
`left_fixed` 与 `base0` 均通过，因此当前确定性结果是：

| 字段 | 值 |
| --- | --- |
| completed rounds | 2 |
| budget remaining | 1 |
| verdict | `inconclusive` |
| stop reason | `continue` |
| evidence sufficient | `false` |
| untested | `right_fixed`、`base1` |
| 新 provider / simulator / ACT | `0 / 0 / 0` |

即使下一轮成功，仍会有一个候选未测并因预算耗尽停止；只有提前观察到确定失败时，有限域
universal claim 才可被反例直接 refute。这是可审计的有限域停止逻辑，不是统计泛化保证。

`answer_scope.json` 进一步强制投影：

- ACT policy episodes `N=2`；
- 两轮使用同一 seed `100502`；
- 两个候选尚未测试；
- color、material gloss、texture、mass、scale 对 `click_bell` 仍 unsupported；
- 当前回答必须标成 interim，不能声称 evidence sufficient。

相关文件：

- [`query_contract.json`](query_contract.json)
- [`query_candidate_evidence.json`](query_candidate_evidence.json)
- [`query_assessment.json`](query_assessment.json)
- [`answer_scope.json`](answer_scope.json)

## 2. P2：公共 Proposal → scene + experimental SuccessSpec v2

显式 experimental BBH 路径在服务器完成了一次真实 TaskGen acceptance：

| 项目 | 结果 |
| --- | --- |
| task / aspect | `beat_block_hammer / object_appearance.color` |
| TaskGen route | `force_codegen` |
| scene variation | block color `(0.25, 0.25, 0.75)` |
| SuccessSpec | planar XY distance `< 0.025 m` on both axes **and** physical hammer-block contact |
| provider calls | 3 |
| simulator probes | 2 |
| expert probes | 1 |
| ACT starts | 0 |
| setup / render / rule / expert | passed |
| production acceptance | `task_generation_only_no_act` |
| experimental ACT runtime eligible | `false` |

![TaskGen initial render](p2/initial_head.png)

这证明 public Proposal/TaskGen 能生成并验收一个受限 scene + `check_success()` 组合。它不证明 ACT 在
该任务上成功；official 与 experimental outcome 尚未拥有独立的运行时标签，因此主 Agent 会 fail
closed，禁止把该 SuccessSpec 当成 official policy success。ACT 前的最终 authority 是重新构建并验证的
`TaskArtifactBundle.success_semantics.act_runtime_eligible`，不是 manifest 中可陈旧的 Proposal 副本。

相关文件：

- [`proposal_bundle.json`](p2/proposal_bundle.json)
- [`task_proposal.json`](p2/task_proposal.json)
- [`variant_spec.json`](p2/variant_spec.json)
- [`overlay.yml`](p2/overlay.yml)
- [`success_spec.json`](p2/success_spec.json)
- [`success_spec_provenance.json`](p2/success_spec_provenance.json)
- [`task.py`](p2/task.py)
- [`task_artifact_bundle.json`](p2/task_artifact_bundle.json)
- [`scene.json`](p2/scene.json)
- [`expert_episode.json`](p2/expert_episode.json)
- [`task_generation_attempt_summary.json`](p2/task_generation_attempt_summary.json)
- [`result.json`](p2/result.json)

`variant_spec.json`、`overlay.yml`、`task.py`、`TaskArtifactBundle`、scene probe、expert episode 与
acceptance summary 一并发布，使 summary 中的 artifact hash 和 expert/render 记录可独立复核；完整日志
和其余 telemetry 仍留服务器。

## 3. P3：matched fixed/adaptive efficiency protocol

[`matched_efficiency_synthetic.json`](matched_efficiency_synthetic.json) 只验证：

- Query、checkpoint hash、candidate suite、seed、最大预算与 Query contract 必须完全 matched；
- ACT starts、completed trials、policy steps、expert/probe、provider call/attempt/retry 与 wall time 分开；
- 同时覆盖 synthetic `fixed 2 vs adaptive 1` 和 `2 vs 2` 零节省 truth table；
- 原 Query 的 structured conclusion 不一致时，节省不能成立；
- paper 的 5 trials/task、10 agent runs 目标未满足；
- paper 的 sample count 不会被静默等同于 ACT starts 或 policy steps。

该文件生成时没有启动 provider、simulator 或 ACT，`empirical_policy_claim_eligible=false`。

## 4. P5：独立标注与 VQA control protocol

[`independent_validity_synthetic.json`](independent_validity_synthetic.json) 验证多人标注、pairwise
agreement、majority/senior tie-break 接口，以及 clean、scene clutter、background texture、lighting
四条件下的 fixed-threshold accuracy/AUROC 计算。

本 fixture 只有 3 个 proxy rater，包含 development-agent label，没有真实 human gold。文件中的
`accuracy=0.875`、`AUROC=0.875` 是故意带一个失败 control 的合成算术 smoke，不能写进论文结果表。

## 5. 不能从本目录推出的结论

- 不能宣称 Query-conditioned stopping 已进入 live `manipeval_agent.py`；
- 不能宣称 experimental SuccessSpec 已执行 ACT；
- 不能宣称 adaptive 比 fixed 节省真实 rollout；
- 不能宣称 VQA 已通过独立人工有效性验证；
- 不能宣称论文的 sampling-efficiency、benchmark-consistency 或统计表已复现。
