# MEA 精简知识检索与轨迹工具包

本文说明 `beat_block_hammer` 第一条垂直通路的设计与使用方式。目标不是建立一个庞大的通用知识库，而是让 TaskGen 得到写好当前场景真正需要的源码、API 约束和相近示例，同时把一次评估记录成可以复查、可以由 Trusted Tools 离线分析的证据。

## 1. 端到端调用链

```text
自然语言请求
  -> scripts/manipeval_taskgen.py
  -> Proposal: 生成 VariantSpec
  -> TaskRetriever: 选择相近 RoboTwin task
  -> Documentation RAG: 选择短知识卡与 method-level 示例
  -> CodeGen: 生成完整 load_actors(self)
  -> AST / protected diff / render / Vision / expert gate
  -> policy/ACT/eval_mea.sh
  -> script/eval_policy.py
  -> Base_Task.take_action()
       -> 10 Hz policy boundary snapshot
       -> 每次 250 Hz scene.step() 后记录 semantic state 与 contact event
  -> episode artifacts
  -> Tool Retriever
  -> 8 个 Trusted Tools
  -> tool_results.json 与 run manifest
```

Recorder 是 opt-in 的：`eval_mea.sh` 不传第 11 个 `TELEMETRY_DIR` 参数时，官方评估行为保持不变。TaskGen 使用 `--expert` 或 `--run-act` 时会自动为 expert 和 ACT episode 建立 telemetry 目录。

## 2. Offline Extractor 与 Documentation RAG

第一版只覆盖 `beat_block_hammer`，避免运行时把整个 RoboTwin 仓库塞给 GPT。Offline Extractor 位于 `mea/knowledge/extractor.py`，它解析指定源码 symbol，并为知识卡、源码 symbol 和 asset 文件记录 SHA-256。这样既能精确引用来源，也能检测文档是否已经落后于源码。

当前知识卡只有三类：

- `task.beat_block_hammer`：任务生命周期、actor 身份、官方随机位姿与 success contract。
- `api.scene_creation`：`create_box`、`create_actor`、`rand_pose` 的参数含义和使用限制。
- `asset.020_hammer`：hammer asset、functional point 与 contact point 约束；只有请求真正涉及 hammer 时才应选择。

对“把红色方块改成蓝色”这一请求，Knowledge Retriever 只选择前两张知识卡；TaskRetriever 选中的相近任务也只抽取 `load_actors()` 方法，而不是注入整个 task 文件。TaskGen 还会看到官方 `beat_block_hammer.load_actors()`，最终 knowledge context 上限为 8000 characters。这个粒度足以解释如何创建蓝色 box，又不会让无关规划、相机或 reward 代码干扰 CodeGen。

知识相关源码与产物：

```text
mea/knowledge/
  index.json                         # Offline Extractor 索引
  tasks/beat_block_hammer.md         # task contract
  api/scene_creation.md              # 场景构造 API
  assets/020_hammer.md               # hammer asset contract
mea/retrieval/knowledge_base.py      # 确定性选择与 context 组装
<run_id>/generation/
  knowledge_catalog.json
  knowledge_retrieval.json
  knowledge_context.md
```

构建与检查索引：

```bash
python scripts/build_mea_knowledge.py
python scripts/build_mea_knowledge.py --check
```

## 3. BeatBlockHammer TaskSchema

`mea/toolkit/schemas/beat_block_hammer.json` 是 Recorder 与 Tools 之间的稳定语义契约。它声明：

- physics timestep 为 `0.004 s`，即 250 Hz；ACT action dimension 为 14。
- `hammer` 对应 task attribute `self.hammer` 和 scene name `020_hammer`。
- `block` 对应 `self.block` 和 scene name `box`。
- 需要记录的 functional points，以及需要关注 contact 的 actor。
- 官方 success contract：hammer functional point 0 与 block functional point 1 的 XY 误差小于 `0.02 m`，并满足 contact。
- hammer pickup 的初版阈值为相对初始高度上升 `0.03 m`。

TaskSchema 让 Recorder 不必猜测每个 task 的属性名，也让 Trusted Tools 不直接依赖生成代码。扩展到新任务时，应先增加一个短小 schema，再增加与该 schema 对应的工具。

## 4. 两级 Trajectory Recorder

一次 policy action 会在 `Base_Task.take_action()` 内展开成许多 `scene.step()`。只在 `eval_policy.py` 外层采样会漏掉短暂 contact，因此当前实现分为两级：

### Policy boundary：约 10 Hz

`states.csv` 保存 initial、每个 post-action 和 final snapshot，主要字段包括：

- 输入 action、policy step、physics step、sim time 与 wall time。
- 左右臂 qpos/qvel、EE/TCP pose、gripper state。
- hammer/block pose、linear/angular velocity。
- schema 指定的 functional point pose 与 success flag。

### Physics step：250 Hz

`semantic_trace.npz` 只保存高频分析真正需要的紧凑数组：

- hammer 与 block position。
- hammer/block functional point position。
- 左右 TCP position。
- policy/physics step、simulation time 与 success。

`events.jsonl` 保存稀疏事件，而不是为每个 physics step 写一行空 contact：

- hammer/block contact interval 的开始与结束。
- 首次 physical contact、最大 impulse、最小 separation、peak point/normal。
- success transition 与运行异常。

Recorder 同时区分 SAPIEN 报告的 contact pair 和由 impulse/separation 支持的 physical contact。RGB 不重复写入 CSV/NPZ，继续保存为压缩 MP4。

## 5. Trusted Tools 与 Tool Retriever

`mea/toolkit/tools.py` 提供 8 个只读、确定性的工具：

| Tool | 解释 |
| --- | --- |
| `hammer_pickup_height` | hammer 相对初始位置的最大抬升高度，并按 schema 阈值判断是否拿起 |
| `hammer_block_min_xy_error` | 两个官方 functional points 的最小 XY L-infinity error |
| `hammer_block_contact_ever` | hammer 与 block 是否发生过 physical contact |
| `first_contact_step` | 首次 physical contact 的 policy step、physics step 与 simulation time |
| `max_contact_impulse` | contact interval 中的最大 impulse 及估算 peak force |
| `ee_path_length` | 根据 block 初始 X 选择 active arm，并计算其 TCP path length |
| `official_check_success` | Recorder 锁存的 RoboTwin 官方 success |
| `time_to_success` | 首次官方 success 的 simulation time；未成功时为 `null` |

每个结果都带 `tool_sha256`、单位、证据 step 和视频前后帧索引，便于从数值结论回查轨迹和 MP4。`TrustedToolRetriever` 第一版不再调用 GPT：对 BeatBlockHammer 固定选择 pickup、距离、contact、首次 contact 和 official success；当请求出现“力度/impulse”“路径”“耗时”等词时，再加入对应工具。选择理由和完整 catalog 会写入结果，便于审计。

## 6. 一次 run 的主要产物

```text
mea/generated_tasks/<run_id>/
  request.json
  variant_spec.json
  manifest.json
  task.py
  generation/
    knowledge_catalog.json
    knowledge_retrieval.json
    knowledge_context.md
    code_prompt.md
    code_response.txt
  validation/
  evidence/
  evaluation/
    act.json
    episode0.mp4
    telemetry/
      expert/episode_000_seed_<seed>/
        episode.json
        schema.json
        states.csv
        semantic_trace.npz
        events.jsonl
        tool_results.json
      act/episode_000_seed_<seed>/
        episode.json
        schema.json
        states.csv
        semantic_trace.npz
        events.jsonl
        video.mp4
        tool_results.json
      tool_results.json
```

`episode.json` 描述 policy、seed、步数、耗时、最终 success 与异常；顶层 `telemetry/tool_results.json` 汇总所有 expert/ACT episodes；`manifest.json` 保存 `knowledge_retrieval` 和 `trusted_tool_evaluation` 摘要。因此一次评估的 prompt、生成代码、渲染、视频、原始轨迹和工具结论都绑定在同一个 `run_id`。

## 7. 基本命令

蓝色方块、expert gate、ACT 1 episode 和 Trusted Tools 的完整最小链路：

```bash
export UIUI_API_KEY='...'
python scripts/build_mea_knowledge.py --check
python scripts/manipeval_taskgen.py \
  --request '把 beat_block_hammer 的红色方块改为蓝色；评估 ACT 1 episode，判断是否拿起锤子、是否接触方块、第一次接触在第几步，并报告最小距离和最终成功结果' \
  --mode force_codegen \
  --probe \
  --vision-check \
  --expert \
  --run-act \
  --num-episodes 1 \
  --seed 100000 \
  --gpu 0
```

仅在已经生成的 run 上恢复后续验证或评估：

```bash
python scripts/manipeval_taskgen.py \
  --resume-run <run_id> \
  --expert \
  --run-act \
  --num-episodes 1 \
  --seed 100000 \
  --gpu 0
```

直接使用参数化 ACT 入口并开启 telemetry：

```bash
policy/ACT/eval_mea.sh \
  beat_block_hammer demo_clean demo_clean 50 0 0 1 \
  <task_module> <overlay.yml> 100000 <telemetry_dir>
```

## 8. 验证原则与当前局限

结论的证据优先级为：simulator 原始状态与 contact > 独立 Trusted Tool 重算 > 视频视觉观察。视频适合确认颜色、材质和明显场景错误；尺寸、距离、接触、抬升高度与 success 应以 simulator 数值工具为准。

建议至少同时保留一个 ACT episode 和一个 expert episode。ACT 可以暴露真实 policy failure，但不一定覆盖成功/contact 正例；expert 用来确认 Recorder 和 Tools 在正例轨迹上确实能报告 pickup、contact 与 success。工具结果应与 `events.jsonl`、`semantic_trace.npz` 和相邻视频帧交叉核对，不能仅因为 official success 为 false 就推断“从未接触”。

当前局限：

- Knowledge Base 与 TaskSchema 只覆盖 `beat_block_hammer`，尚不是 50 个 task 的通用抽取器。
- Tool Retriever 是确定性关键词版本，还没有 GPT 生成新 Tool。
- 高频 trace 有 functional points、TCP 与 contact，但不是 SAPIEN 全场景所有 actor 的无差别 dump。
- contact 发生在一个 policy action 内时，只能映射到该 action 前后的相邻视频帧，不能保证逐 physics step 有 RGB。
- `ee_path_length` 依据 block 初始 X 选择 active arm，这是该任务的规则，不是跨任务通用规则。

## 9. 蓝色方块实跑验证

2026-07-15 在服务器上完成了端到端实验：

- run_id：`run_20260715_telemetry_blue_seed100000`
- request：把 `beat_block_hammer` 的红色方块改为蓝色，运行 ACT 1 episode，并判断 pickup、contact、first contact、minimum distance 与 official success。
- seed：expert 与 ACT 均为 `100000`。
- Task Retriever：`beat_block_hammer`、`blocks_ranking_rgb`。
- Knowledge Retriever：`task.beat_block_hammer`、`api.scene_creation`、`example.blocks_ranking_rgb.load_actors`。
- CodeGen：生成完整 `load_actors()`；AST、受保护文件检查和 Visual Self-Reflection 均一次通过，识别到蓝色方块。

关键结果：

| 指标 | ACT | expert |
| --- | ---: | ---: |
| official success | false | true |
| policy / physics steps | 400 / 14852 | 1 / 1480 |
| hammer pickup height | 0.127615 m | 0.113232 m |
| hammer-block 最小 XY L∞ 误差 | 0.014917 m | 0.000765 m |
| hammer-block physical contact | false | true |
| first physical contact | null | physics step 1454（5.816 s） |
| max contact impulse | 0 N·s | 0.033686 N·s |
| time to official success | null | physics step 1417（5.668 s） |

这说明 ACT 确实拿起了锤子，并把 functional point 对准到官方每轴 `0.02 m` 阈值内，但没有产生严格的 hammer-block physical contact，因此最终失败。expert 正例同时覆盖了 pickup、contact 与 success，证明 Recorder 和 Trusted Tools 能在成功轨迹上工作。official success 早于 strict physical contact，是因为官方判断使用 SAPIEN reported contact，而 Recorder 的 `physical contact` 还要求非零 impulse 或非正 separation；两种语义均被保留，没有互相覆盖。

完整性检查：ACT 的 `semantic_trace.npz` 有 14853 行，physics step 连续覆盖 `0..14852`；expert 有 1481 行，连续覆盖 `0..1480`。独立审计脚本直接读取 NPZ 和 JSONL，重新计算 pickup、minimum distance、contact、first contact 与 success，结果与 Trusted Tools 全部一致。ACT 视频为 320×240、10 FPS、400 帧、40 秒，SHA256 为 `c5c44dd86e3f36400b608c170805f42584b6dfe51901a6ac16b811fadf3acf6e`；该哈希与同 seed 的既有蓝色场景视频一致，未发现 Recorder 改变 rollout。

单个 ACT episode 的主要体积约为：`states.csv` 1.9 MB、`semantic_trace.npz` 460 KB、`events.jsonl` 4 KB、视频 272 KB。这个结果支持当前的混合存储设计：policy boundary 保存可读 CSV，250 Hz 只保存压缩语义 trace 与稀疏事件，RGB 继续使用 MP4。

开发中发现并修复了四个实际问题：contact interval 局部变量遮蔽、非交互 shell 找不到 Python、失败重试误取历史 eval 目录，以及未设置 `TORCH_HOME` 时重复下载已有 ResNet cache。失败尝试均保留在 run 的 `validation/` 或 `evaluation/previous_act_attempts/` 中；完成后的 manifest 会清空当前 `failure` 字段。服务器操作日志位于 `/root/autodl-tmp/mea/_ops_logs/`，运行产物位于 `/root/autodl-tmp/mea/mea/generated_tasks/run_20260715_telemetry_blue_seed100000/`，它们按仓库规则不提交 Git。
