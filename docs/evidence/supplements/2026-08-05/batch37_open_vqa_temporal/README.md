# Batch37 open temporal VQA Tool

Query 要求在不改 scene/checker、也不从 telemetry 推断视觉运动的前提下，判断 `press_stapler` rollout 中夹爪是否在首次成功按压前越过订书机并反向重新对齐。

方法链最小证据：

1. SmolVLA 在 official `press_stapler` 上完成 1 次 rollout（seed 1000，official success=true）。
2. VQA ToolGen 在 exact retrieval miss 后生成新的时序二值问题。
3. `temporal_keyframes_v1` 从 10 帧中选 4 帧，VQA 首次判断 `observed=false`、confidence 0.82。
4. 问题 artifact 经 reviewed persistent registry 在新 evaluation 中 exact reuse：0 新 rollout、0 text/provider-generation call；仍对同一冻结 episode 做一次新的视觉判断。

关键负结果：第二次视觉判断对完全相同的冻结 episode、问题和 4 帧给出 `observed=true`、confidence 0.86。两次执行各自都没有标记内部 conflict，但彼此结论相反。因此这里证明了 VQA **问题 artifact 的生成、注册和精确复用**，同时暴露了 VQA 输出不稳定；它不能作为 VQA robustness 或该现象真值的正证据。

- [provider_exchange.md](provider_exchange.md)：Query、生成合同、生成问题及两次观察
- [evidence_summary.json](evidence_summary.json)：精简机器可读数据流
- [round_1_vqa_montage.png](round_1_vqa_montage.png)：两次判断共同使用的 4 帧 montage
- [round_1_act.mp4](round_1_act.mp4)：冻结的 official rollout

范围限制：N=1、单 seed、单 VLM、同一 episode；没有独立人工 gold，也没有四种 simulator 扰动条件。
