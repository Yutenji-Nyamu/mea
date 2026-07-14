# Multi-Round Plan Agent 实现与验证

## 目标

本阶段实现一个有界的两轮 Plan Agent 原型。用户输入：

```text
评估 ACT 在蓝色方块和位置变化下的表现。
```

系统不会一次性预写两轮计划，而是先执行 Round 1，将真实观察返回给
Plan Agent，再决定是否进入 Round 2。

## 调用链

```text
用户请求
  -> scripts/manipeval_agent.py
  -> Plan Agent: 只规划 Round 1
  -> TaskGen force_codegen: 蓝色方块
  -> AST / render / rule / Vision / expert / ACT (1 episode)
  -> Round 1 observations 返回 Plan Agent
  -> Plan Agent: continue 或 stop
  -> TaskGen reuse: 保持蓝色，官方位置随机化
  -> 每个 seed 的 rule + expert gate
  -> ACT (2 episodes)
  -> simulator position metrics + policy results 聚合
  -> Feedback Agent
  -> evaluation_report.md
```

Round 1 使用 `force_codegen`，让 GPT 生成完整 `load_actors()`；Round 2 使用
已经验证的 `reuse` 路由，避免只为改变评估采样数量重复生成等价代码。

## 关键实现

- `mea/planner/prototype.py`
  - 初始 Plan 只能包含 Round 1。
  - `decide_next_round()` 接收实际 Round 1 observations。
  - Round 1 pipeline 通过后规划 `object_position`；pipeline 失败则停止。
- `scripts/manipeval_taskgen.py`
  - 新增 `--num-episodes`，向 ACT wrapper 透传评估次数。
  - 收集每个 evaluation seed 的精确 `block_pose`。
  - 计算 `unique_xy_count`、`x_span`、`y_span` 和 `position_varied`。
  - 每个 seed 独立运行 expert gate；偶发执行失败最多重建环境重试 3 次。
  - 复制所有 `episode*.mp4`，并检查视频数量与 episode 数一致。
- `mea/providers/openai_compatible.py`
  - 对 timeout 和 connection error 最多重试 2 次。
  - HTTP 业务错误不重试，错误信息仍不包含 API key。
- `mea/feedback/prototype.py`
  - 汇总多轮计划、逐轮证据、位置指标和 policy success。
  - 生成统一的 `evaluation_report.md`。

## 真实验证

成功 run：`eval_20260714_184500_multiround`

### Round 1：蓝色外观

- seed：`100000`
- episodes：`1`
- route：`force_codegen`
- Vision observed color：`blue`
- scene / expert / ACT pipeline：全部通过
- policy success：`0.0`

### Round 2：位置变化

- seeds：`100002`, `100003`
- episodes：`2`
- route：`reuse`
- 两个 seed 均通过独立 rule 和 expert gate
- block position：
  - `100002`: `[-0.140583, 0.080840, 0.760000]`
  - `100003`: `[0.164232, -0.048631, 0.760000]`
- `unique_xy_count=2`
- `x_span=0.304815 m`
- `y_span=0.129470 m`
- `position_varied=true`
- ACT pipeline：通过，保存 2 段视频
- policy success：`0.0`

三段视频均为 `320x240`、`10 FPS`、`40 s`。两轮总计 3 个 episode，
pipeline 全部通过，ACT policy 在三个 episode 中均未完成任务。这个结果只说明
当前三个样本，不构成泛化结论。

## 开发中发现并修复的问题

1. seed `100001` 的位置接近工作区边缘，官方 expert planner 未通过，因此没有把它
   作为 policy 测试样本。`100002/100003` 均通过 expert gate。
2. 同一 pose 的 expert rollout 存在偶发 `check_success` 波动。现在每个 seed 最多
   独立重建环境并尝试 3 次，同时保留每次尝试结果。
3. UIUI 曾发生一次 180 秒 read timeout。provider 现在会对临时网络错误做有界重试。

## 运行方式

```bash
export UIUI_API_KEY='...'
python scripts/manipeval_agent.py \
  --request '评估 ACT 在蓝色方块和位置变化下的表现。' \
  --gpu 0
```

所有 plan prompt/response、逐轮 TaskGen 产物、位置指标、视频、summary、Feedback
和最终报告都绑定到同一个 `evaluation_id`。运行产物被 Git 忽略，不进入公开仓库。

## 关于多图视觉对照

OpenAI-compatible 多模态消息可以在同一个 `content` 数组中放入多条
`image_url`，因此可以同时提供官方参考图和候选图；只支持单图的网关也可以使用
拼接图回退。当前 provider 的 `vision()` 仍是单图接口，本阶段没有扩展它，因为
位置和尺度应优先由 simulator rule 精确检查，参考图对照适合后续用于颜色、材质、
背景和杂物等视觉属性。
