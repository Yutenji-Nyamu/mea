# Evaluation Feedback

在 click_bell 任务、ACT checkpoint（demo_clean-50）和 seed 100000 下，最明确的弱点是将铃铛的视觉与碰撞几何统一缩小为原尺寸的 80%，同时保持接触点世界坐标、颜色、材质、场景、视角、指令和成功判定不变。该变更下 ACT 未完成任务：官方成功率为 0/1，最小 TCP-铃铛接触点 XY 误差为 0.045257747173309326 m；VQA 在 initial、context_2、final 帧观察到铃铛未被按下。作为对照，原始场景和仅改变颜色的候选均成功，颜色变更候选官方成功率为 1/1，最小 XY 误差为 0.0077674430795013905 m。因此，本次有限测试诊断出 ACT 对该 20% 尺寸缩小变更存在失败，但证据未单独建立因果机制。

## Evaluation scope

click_bell 任务；ACT policy_under_evaluation；checkpoint 为 act-click_bell/demo_clean-50；共 N=3 个 policy episodes，均使用 seed 100000：原始场景 1 个、铃铛颜色 bounded change 1 个、铃铛视觉与碰撞几何缩小至 80% 1 个。官方成功结果分别为 1/1、1/1、0/1；本次 evaluation pipeline 完成，不能将 pipeline 完成等同于操作成功。

## Findings

- 原始场景官方成功率为 1.0（1/1），time_to_success 为 22.02 s。
- 仅改变铃铛颜色且保持任务相关几何与位置不变时，官方成功率为 1.0（1/1）；ACT 的最小 XY 误差为 0.0077674430795013905 m，time_to_success 为 20.952 s。
- 将铃铛视觉与碰撞几何统一缩小为 80% 且保持接触点位置不变时，官方成功率为 0.0（0/1）；ACT 未完成任务，最小 XY 误差为 0.045257747173309326 m。
- 尺寸缩小候选的 VQA 与 simulator numeric Tool 结论一致：VQA 在 initial、context_2、final 帧观察到铃铛未被按下；未报告 evidence_conflict。
- 本次总体 policy_under_evaluation 官方成功率为 2/3（0.6666666666666666），其中失败由尺寸缩小候选产生。expert_validation 未与 ACT 结果合并。

## Limitations

- 证据包含 N=3 个 policy episodes，且只有 seed 100000；不能据此作统计泛化结论。
- 候选域是开放的，未测试其他尺寸比例、其他 seed 或其他物体属性变更；不能作穷尽、最坏情况或普遍鲁棒性结论。
- 尺寸缩小失败的官方成功结果和 XY 误差已测得，但本次证据未单独证明具体因果机制。
- 尺寸缩小失败 episode 的 time_to_success 为缺失值；不能将其解释为通过或失败的时间指标。
- 本次停止原因是有限 query-sufficiency contract 达到 evidence_sufficient，而非统计显著性或广泛基准完成。
- Evidence contains N=3 policy episodes at seeds [100000].
- The run stopped because the finite query-sufficiency contract was satisfied; this is not a statistical generalization guarantee.

## Recommended next step

在保持同一 checkpoint、场景语义和官方成功判定不变的前提下，使用多个新 seeds 重复 80% 尺寸候选，并加入 90%、70% 等预先限定的尺寸比例；同时继续记录最小 XY 与 Z 误差，以检验该失败是否稳定并进一步区分定位误差与接触高度误差。
