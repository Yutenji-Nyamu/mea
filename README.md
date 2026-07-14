# MEA

MEA 是基于 RoboTwin 的 **agent 编排评估过程**：把自然语言评估请求转换为场景
变式，依次执行代码检查、场景渲染、Vision 检查、expert gate 和 policy
evaluation，并按 `run_id` 保存证据。

主要入口是 [`scripts/manipeval_taskgen.py`](scripts/manipeval_taskgen.py)：

```bash
export UIUI_API_KEY='...'
python scripts/manipeval_taskgen.py \
  --request '把 beat_block_hammer 的红色方块改为蓝色，其他行为保持不变' \
  --mode force_codegen --probe --vision-check --expert --run-act
```

参数化 ACT 入口为 [`policy/ACT/eval_mea.sh`](policy/ACT/eval_mea.sh)。实现说明见
[`docs/manipeval.md`](docs/manipeval.md) 和
[`docs/taskgen_prototype.md`](docs/taskgen_prototype.md)。原始 RoboTwin README 保存在
[`README_RoboTwin.md`](README_RoboTwin.md)。
