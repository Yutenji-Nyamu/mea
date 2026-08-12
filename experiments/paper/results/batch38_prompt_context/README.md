# Batch38：失败驱动 prompt/context 纵向回归

本目录是冷开发流水账，不是论文旗舰证据。Batch38 用 `grab_roller` 的既有真实失败检查
“任务事实和上一版失败是否进入正确 Agent”，并同步收缩生产链上下文。当前旗舰仍是
Batch37；本批不更新 `docs/evidence/current/`。

## 1. 实现范围

- Plan、TaskGen、Tool Request/ToolGen 共用一张按需读取的短任务卡；不再把语言改写用的
  `task_instruction/*.json` 当作 scene/checker 实现知识。
- Plan repair 带上一版完整 Proposal 和字段级错误；跨轮 prompt 单列最近失败。TaskGen
  repair 带上一版 methods 与 diagnosis；Tool repair 带上一版函数、失败阶段与
  observed/required。
- 不维护自动 failure index/corpus。任务特有经验只进入短任务卡；跨任务重复规则才进入
  对应 `README.Agent`。repair 直接携带上一版完整输出与当前具体错误。
- VQA 的 `null` 或一致性不足被投影为 `abstained/unknown`，而不是伪造二值结论。
- 普通反馈 prompt 改用紧凑 evidence projection；原始 evidence 仍供 validator 和冷审计。
  该投影在 v5 结束后实现，**不能**据此改写 v5 的 token 记录。
- 生产模块不再为最终 Answer 调用 flagship acceptance；普通 recorder 也不再自动建立
  execution receipt。历史兼容和论文协议仍留在冷层。
- X-VLA 的隔离下载、环境、离线 action smoke 另见
  [`../../robotwin_xvla/deployment_ledger.md`](../../robotwin_xvla/deployment_ledger.md)；它尚未
  完成 RoboTwin simulator episode，不属于本批 live 方法证据。

## 2. 服务器与命令

- 服务器：SeetaCloud `bjb2`；权威工作树：
  `/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime`。
- 开始基线：`1a0841eb0b380f89c81027cbdc30a3ba9e53540a`。
- provider credential 只注入远程进程；本文和日志索引不保存 key。
- Windows PC 只编辑代码、文档和紧凑 artifact；未运行测试、provider、仿真或 policy。

定向/主干测试均在 AutoDL server 执行。已确认的阶段性结果为：

| 范围 | 结果 |
| --- | --- |
| task-guide / TaskGen prompt | `38 passed, 5 subtests passed` |
| Plan/runtime 修复 | `48 passed` |
| VQA/Tool/cross-task | `63 passed, 6 subtests passed` |
| TaskGen/candidate | `60 passed, 5 subtests passed` |
| Query semantics/runtime/backends | `121 passed, 5 subtests passed` |
| 当时默认 mainline | `313 passed, 22 subtests passed` |
| 最终定向回归 | `159 passed, 21 subtests passed` |
| 最终默认 mainline | `272 passed, 12 subtests passed` |

阶段日志位于服务器
`/root/autodl-tmp/mea-run-logs/batch38_prompt_context/`。最终提交前另在 Windows PC 只执行
`git diff --check` 并通过；没有运行项目测试。

live 命令的解析配置为：`grab_roller`、SmolVLA、scene seed `1000`、policy-server seed
`103500`（v2 为 `103800`）、每轮 `N=1`、最多 4 个 Agent round、history disabled。其核心命令
如下；端点凭据已省略：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /root/autodl-tmp/envs/mea-libero/bin/python \
  experiments/paper/robotwin_smolvla/policy_server.py \
  --checkpoint /root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin \
  --host 127.0.0.1 --port 18871 --seed 103500 --max-clients 4

UIUI_API_KEY='<process-only>' CUDA_VISIBLE_DEVICES=0 \
  /root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python \
  scripts/manipeval_agent.py --repo-root "$PWD" --auto-route \
  --bound-task-name grab_roller --policy-backend smolvla \
  --smolvla-checkpoint /root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin \
  --smolvla-port 18871 --start-seed 1000 --num-episodes 1 \
  --generated-rounds 3 --max-agent-rounds 4 --no-history \
  --request '<broad query without aspect/template/checker/metric>'
```

## 3. live 纵向结果

| run | policy episode | 结果 |
| --- | ---: | --- |
| v1 | 0 | `repo_root` 未传入 Plan Agent，触发 `NameError`；修复初始化合同。 |
| v2 | 1 | official control 为 policy negative，按证据短路并返回 inconclusive。 |
| v3 | 2 | control 与 `+x` scene 均成功；新 Tool 得到 `0.0606393 m`；随后 official baseline expert 的 `target_pose=None` 被误作 terminal system error。 |
| v4 | 2 | 将明确的 zero-rollout candidate-unexecutable 回流 Plan；两项不可执行候选后，`y=-0.15` scene 成功并得到 `0.0248231 m` live Tool；四轮上限停止。 |
| v5 | 2 | control 与 lateral scene 均成功，新 Tool 得到 `0.0606393 m`；evidence 使 Plan 从 lateral 转向 orientation，再转向 longitudinal；后二者被 expert gate 判定为不可执行、未消耗 rollout；四轮上限停止。 |

v5 正常退出，最终仍为 `inconclusive`。它证明失败上下文能改变下一 Proposal，并且 typed
zero-rollout rejection 不再伪装成 policy failure；它没有证明 Agent 主动充分停止，也没有证明
`grab_roller` 泛化。v5 最终 Answer prompt 为 **56,150 tokens**；随后加入的紧凑 evidence
projection 在服务器回归中把 v5 raw evidence 的 `248,226 B` 压为 `18,601 B`，连同独立
`AnswerScope` `3,586 B` 后为 `22,187 B`。这验证了上下文收缩和关键字段保留，但不能回写
或降低 v5 已发生的 token 记录。

## 4. 结论边界与下一步

- 本批的主要正结果是 prompt/context 纵向修复和真实 failure-to-next-Proposal 反馈，不是新论文
  claim closure。
- rollout 采用软预算：五次开发运行共 7 个 policy episode；zero-rollout expert rejection 不计
  policy episode。没有为了追求“一次 rollout”而截断故障定位，也没有开展无研究问题的宽扫。
- 下一步让同一 broad Query 在可执行候选后由 Plan Agent 主动 stop；若仍重复低信息方向，
  优先改短任务卡和 Agent prompt，不增加外围状态机。
- 机器可读摘要见 [`summary.json`](summary.json)。原始 log/evaluation 继续只保存在 AutoDL 冷目录。
