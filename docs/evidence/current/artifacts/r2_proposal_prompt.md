你是 ManipEvalAgent 的 Task Proposal Agent。

用户请求：将 beat_block_hammer 的目标方块等比例放大到官方尺寸的 1.2 倍，保持红色、官方位置/朝向采样、成功语义和其余行为不变；先通过 render 与 expert gate。 Query-generated bounded variation: evaluate a query-relevant bounded variation
规范任务名：beat_block_hammer

请把请求转换为严格 JSON，不要输出 Markdown。第一版只处理 beat_block_hammer 的方块变式。
必须返回：
{
  "task_name": "beat_block_hammer",
  "intent": "change_object_appearance",
  "generation_mode": "force_codegen",
  "changes": {
    "block": {
      "position_mode": "official_random",
      "yaw_mode": "official_random",
      "scale": 1.0,
      "color": [0.0, 0.2, 1.0]
    }
  },
  "preserve": [
    "official_position_sampling",
    "official_yaw_sampling",
    "play_once",
    "check_success",
    "checkpoint"
  ]
}

颜色使用 [0,1] RGB。除非用户明确要求，否则保持官方随机位置、随机 yaw、尺度、专家轨迹和成功判定不变。


The planner capability contract is authoritative. The following VariantSpec is immutable; use it for retrieval and code generation:
{"capability_id": "object_scale.bounded", "changes": {"block": {"color": [1.0, 0.0, 0.0], "position_mode": "official_random", "scale": 1.2, "yaw_mode": "official_random"}}, "controlled_axis": "object_scale", "generation_mode": "force_codegen", "intent": "evaluate a query-relevant bounded variation", "preserve": ["official_position_sampling", "official_yaw_sampling", "official_block_color", "play_once", "check_success_semantics", "checkpoint"], "schema_version": 2, "task_name": "beat_block_hammer", "variant_id": "object_scale.query_generated_1"}