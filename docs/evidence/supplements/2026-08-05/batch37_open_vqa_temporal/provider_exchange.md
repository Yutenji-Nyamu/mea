# VQA Tool provider exchange（精简）

## 原始 Query

> On the unchanged official press_stapler task, before the first successful press, does either gripper visibly overshoot the stapler and reverse direction while re-aligning? Use visual rollout evidence to answer this trajectory question. Retrieve or generate the VQA Tool the Query needs; do not create or alter a scene, success checker, or numeric Rule Tool, and do not infer the visible reversal from telemetry alone. Stop once the current rollout's temporal visual evidence answers the bounded question, and report uncertainty if it does not.

## ToolGen prompt 的约束

Exact retrieval 未命中。VQA ToolGen 被要求只针对下面的 visual need 生成一个可直接从 rollout montage 判断的二值问题；不得发明 actor、scene change、checker result、数值阈值或 success authority；`numeric_authority` 必须为 `no_numeric_oracle`。

> Temporal visual boolean observation: whether either gripper visibly overshoots the stapler and reverses direction while re-aligning before the first successful press in the current rollout.

## ToolGen 原始响应

```json
{"schema_version":1,"question_spec":{"id":"run_local.gripper_overshoot_reversal","question_type":"visible_state_change","target_role":"manipulated_tool","question":"Before the first successful press, does either gripper visibly pass beyond the stapler and then reverse direction to re-align with it?","visual_scope":"rollout_change","numeric_authority":"no_numeric_oracle"}}
```

## 首次 Execution VQA 原始响应

```json
{
  "phenomena": [{
    "id": "run_local.gripper_overshoot_reversal",
    "observed": false,
    "description": "The selected frames do not visibly show a gripper passing beyond the stapler and then reversing to re-align.",
    "confidence": 0.82,
    "frame_ids": ["initial", "context_1", "context_2", "success_before"]
  }],
  "confidence": 0.82,
  "frame_ids": ["initial", "context_1", "context_2", "success_before"],
  "numeric_consistency": "consistent",
  "conflicts": []
}
```

## Reviewed exact reuse 的再次观察

同一问题 artifact、同一冻结 episode、同一 4 帧，0 新 rollout、0 text generation call；新的 vision call 判断：

```json
{
  "observed": true,
  "description": "The gripper moves past the stapler to the left, then reverses and re-aligns over it before pressing.",
  "confidence": 0.86,
  "frame_ids": ["context_1", "context_2", "success_before"]
}
```

两次布尔结论冲突，必须作为稳定性缺口保留，不能用第二次结果覆盖第一次结果。
