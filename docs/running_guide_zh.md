# MEA 简明运行指引

本文面向第一次运行 MEA 的用户，给出从 RoboTwin 环境到 expert、ACT 和 Agent
入口的最短路径。MEA 是基于 RoboTwin 2.0 的完整 fork；实现边界和数据流另见
[当前架构与数据流](architecture_and_dataflow_zh.md)。项目目的、论文对应关系、开发取舍和
跨对话约定见 [MEA 项目手册](project_playbook_zh.md)。

## 1. 准备 RoboTwin 环境

优先使用 Linux、NVIDIA GPU、Python 3.10，并按
[RoboTwin 官方 Install & Download](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
处理驱动、Vulkan、CUDA 和手动安装故障。克隆 MEA 后，可直接把它当作官方文档中的
RoboTwin 仓库，不需要再克隆一份上游源码：

```bash
git clone https://github.com/Yutenji-Nyamu/mea.git
cd mea

conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
ffmpeg -version
```

`ffmpeg` 用于写 rollout 视频；未安装时按官方文档补装。下载完成后至少检查：

```bash
test -d assets/objects
test -d assets/embodiments
python -c "import sapien, torch; print('runtime imports: ok')"
```

当前开发服务器把 MEA 放在 `/root/autodl-tmp/mea`，并复用
`/root/autodl-tmp/RoboTwin` 的大体积资源。这个布局不是运行协议：独立机器只要在 MEA
根目录安装资源，并始终传 `--repo-root "$PWD"` 即可。共享机器也可以让
`policy/ACT/act_ckpt` 指向统一的 checkpoint 目录，但不要把本地软链接提交到 Git。
当前开发服务器的 RoboTwin 解释器是
`/root/autodl-tmp/conda/envs/RoboTwin/bin/python`；非交互 SSH 没有执行 `conda activate` 时，
应显式使用该路径，并先运行 `.../bin/python -c "import sapien"`。不要误用不含仿真依赖的
`/root/miniconda3/bin/python`。协议 runner 也会在创建或恢复运行前做这项 fail-fast 检查。

## 2. 按任务下载 ACT checkpoint

ACT checkpoint 是按任务独立训练的。运行某个任务的 expert 不需要 checkpoint；运行该任务
的 ACT 才需要下面两个文件：

```text
policy/ACT/act_ckpt/act-<task>/demo_clean-50/
├── dataset_stats.pkl
└── policy_last.ckpt
```

官方文件位于
[RoboTwin2.0 的 `act_ckpt`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/act_ckpt)。
不要下载整个数据集，也不要先下载到个人电脑/Codex 工作区再上传。应在运行 RoboTwin
的服务器上启用 AutoDL 学术加速；若仍不能直连，再按服务器策略配置 Hugging Face
mirror（例如设置 `HF_ENDPOINT`）。仓库提供了带固定 release revision 和下载后完整性
检查的选择性下载脚本：

```bash
python -m pip install -U huggingface_hub
source /etc/network_turbo                 # AutoDL；其他服务器按平台方式配置
export HF_HUB_DOWNLOAD_TIMEOUT=300        # 大文件慢连接可提高读超时
python scripts/download_act_checkpoint.py --dry-run click_bell
python scripts/download_act_checkpoint.py --max-workers 1 \
  click_bell adjust_bottle grab_roller
```

每个任务约 336 MB；只传实际要评估的任务名。若 Hugging Face 限流，再执行
`hf auth login`；token 只保存在用户环境中。ACT 依赖、训练和
Easy/Hard 配置含义以 [RoboTwin 官方 ACT 指南](https://robotwin-platform.github.io/doc/usage/ACT.html)
为准。

MEA 会在启动 ACT 前检查当前任务的 `policy_last.ckpt` 和 `dataset_stats.pkl`；缺失时会
在进入仿真前报错，并给出选择性下载提示。当前 Agent backend 仅支持官方发布的
`demo_clean-50` checkpoint 布局；其他配置仍可直接使用 RoboTwin/ACT 原生入口实验。

## 3. 先跑无模型密钥的 official expert

仓库中已有 TaskSchema 的任务可复用官方任务实现、expert、Recorder 和 Trusted Tools。
可用 `ls mea/toolkit/schemas/*.json` 查看当前任务列表。以下命令不调用 UIUI：

```bash
conda activate RoboTwin
cd /path/to/mea

TASK=click_bell
python scripts/manipeval_taskgen.py \
  --request "运行官方 ${TASK} expert baseline" \
  --repo-root "$PWD" \
  --task-name "$TASK" \
  --task-module "envs.${TASK}" \
  --mode official \
  --seed 100000 \
  --num-episodes 2 \
  --telemetry-profile balanced_v1 \
  --expert
```

先用 1–2 episodes 做 smoke test，再扩大样本。`official` route 不生成或覆盖官方任务代码；
它会扫描可解 seed，并把 expert 作为验证对照，而不是被评 policy。

## 4. 直接运行 ACT

确认该任务 checkpoint 已就位后，可用 MEA 的参数化 wrapper 跑官方任务。下面把 policy
seed 设为 `0`、GPU 设为 `0`，从场景 seed `100000` 开始运行 2 episodes，并开启
`balanced_v1` telemetry：

```bash
TASK=click_bell
policy/ACT/eval_mea.sh \
  "$TASK" demo_clean demo_clean 50 0 0 \
  2 "" "" 100000 \
  "eval_result/manual_telemetry/${TASK}" balanced_v1
```

前六个位置参数与官方 ACT 入口一致：
`TASK TASK_CONFIG CKPT_SETTING EXPERT_DATA_NUM POLICY_SEED GPU`。之后依次是可选的
`NUM_EPISODES TASK_MODULE TASK_OVERLAY START_SEED TELEMETRY_DIR TELEMETRY_PROFILE`。
用 `demo_randomized` 替换第二个 `demo_clean`，即可测试同一 `demo_clean` checkpoint 的
randomized 环境。

### 4.1 用完全相同的 seed 做 Easy/Hard paired 评估

需要比较同一 ACT checkpoint 在 Easy（`demo_clean`）与 Hard（`demo_randomized`）环境中的
表现时，使用专用入口 `scripts/manipeval_paired.py`。它不调用 planning、VQA 或最终反馈，
因此不需要 `UIUI_API_KEY`；仍需要当前任务的 ACT checkpoint、GPU 和完整 RoboTwin 环境。

```bash
python scripts/manipeval_paired.py \
  --repo-root "$PWD" \
  --task-name click_bell \
  --task-module envs.click_bell \
  --seeds 100400 100401 100402 \
  --run-id click_bell_paired_smoke \
  --gpu 0 \
  --telemetry-profile balanced_v1
```

runner 先把请求 seed 和两种 condition 固化到 `seed_manifest.json`，再以与 ACT 相同的
`eval_mode=True` 分布逐个 seed 探测 Easy/Hard expert eligibility；只有两边都通过的交集才
进入 ACT，并且两边严格使用同一有序 seed 列表。初始化不稳定、expert 失败或执行错误都会
保留为该 seed 的状态，绝不会用后续 seed 静默顶替。也可用
`--manifest <seed_manifest.json>` 复用已有 manifest；它与 `--seeds` 互斥。首次先用 1–3 个
seed 做 smoke test，再用预先声明的更大列表做正式实验。

这里的“同 seed”只保证 numeric seed 身份与顺序相同、不发生替换。RoboTwin 在
`demo_clean` 与 `demo_randomized` 中会执行不同的随机化调用，可能在 actor 放置前改变 RNG
消费顺序；因此它不等价于“完全相同的潜在场景只增加视觉扰动”，不能据此作严格因果鲁棒性
解释。要得到 identical-scene 对照，后续还需拆分随机流或保存并重放 scene specification。

主要产物位于 `mea/paired_runs/<run_id>/`：冻结的 `seed_manifest.json`、eligibility/condition
明细、每个 condition 的 exact-seed ACT 结果与 telemetry，以及确定性计算的 paired summary。
summary 明确给出请求数、两边共同 eligible/evaluated 数、Easy/Hard 成功率、`Hard - Easy`
差值和逐 seed 的 `both_success / easy_only / hard_only / neither` 结果。coverage 小于 1 时应
同时报告原始请求数和实际 paired denominator，不能只摘录成功率。若出现 seed 替换、
复查漂移、缺 telemetry 或成功判定不一致，summary 会标记 `valid_for_comparison=false`，
命令默认非零退出；`--allow-protocol-violations` 只用于诊断，不得用于正式统计。

端到端 Agent 已把任务 route 与 execution backend 解耦：`official` 表示复用官方任务、
不生成或改写任务源码；它不再等同于“只运行 expert”。只要任务有 TaskSchema 和对应
checkpoint，official passthrough 也可选择 ACT。`beat_block_hammer` 变体仍走受限
generated route。

## 5. 运行端到端 Agent

Agent 会调用兼容 OpenAI Chat Completions 的文本/视觉模型，用于 planning、Execution VQA
和最终反馈。密钥只通过环境变量传入：

```bash
export UIUI_API_KEY='在当前 shell 中设置，不要写入文件'
# 仅在使用非默认网关时设置：
# export UIUI_BASE_URL='https://example.com/v1'
```

推荐先使用全局开放 Query 入口；不再手写 task/profile/aspect。下面的旗舰命令最多运行 3 个
ACT rollout（每轮 1 个），证据会决定继续深挖当前方面、切换方面或停止：

```bash
python scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'How well does ACT generalize across positions and official instances of the operated bell?' \
  --auto-route \
  --generated-rounds 3 \
  --start-seed 100402 \
  --num-episodes 1 \
  --max-reflections 0 \
  --model-profile economy \
  --reviewed-tool-registry "$PWD/mea/tool_registry/reviewed"
```

`--auto-route` 只会选择 TaskSchema 与 ACT checkpoint 都就绪的可信 catalog 项；当前覆盖
BBH 和 `click_bell`。若 query 超出 catalog，会写一个 `status=unsupported`、不启动仿真的
evaluation。先验证路由和历史复用而不跑 TaskGen/ACT，可增加 `--plan-only`：

```bash
python scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request '评估 ACT 对蓝色方块外观变化的泛化。' \
  --auto-route \
  --plan-only \
  --evaluation-id eval_global_history_smoke
```

重复相近 query 时检查 `plan/global_query_route.json` 与
`plan/history_retrieval.json`；它们只能引用 completed evaluation，plan-only 本身不会写入
历史结果。

仍可显式选择 official task 进行调试：

```bash

python scripts/manipeval_agent.py \
  --request '评估官方 click_bell 任务，并用视觉和轨迹证据解释结果' \
  --repo-root "$PWD" \
  --task-name click_bell \
  --task-module envs.click_bell \
  --start-seed 100000 \
  --num-episodes 2 \
  --execution-backend act \
  --telemetry-profile balanced_v1 \
  --model-profile economy
```

`--execution-backend` 有三种取值：

- `expert`：只运行官方 expert；无需 ACT checkpoint，expert 是被展示的执行证据；
- `act`：TaskGen 先做非 expert 的 setup/render/rule probe，随后 ACT evaluator 沿用
  RoboTwin 的 expert eligibility 筛选；报告中的 policy success 来自 ACT；
- `both`：同时保留 expert 验证与 ACT 评估，ACT 是 VQA 和报告的主 policy 证据。

`act`/`both` 的 Execution VQA 读取 ACT 连续 rollout 视频；`expert` 读取
`event_keyframes_v1` 稀疏事件视频。完整 Agent 仍需要有效的 `UIUI_API_KEY` 才能完成
视觉问答与最终反馈；无 key 时可先用第 3、4 节入口检查仿真、telemetry 和 checkpoint。

`both` 会在本次 Agent 运行结束时比较 expert 与 ACT episode 的实际有序 seed；若不一致
则把流水线标为失败，避免静默混用。但 Agent 的 `both` 仍会扫描替代 seed，也不计算
Easy/Hard paired 统计，因此只适合通路核验。需要预先锁定 seed 的严格对照时，应使用
第 4.1 节的专用 paired runner。

`beat_block_hammer` 还支持受限 TaskGen/ACT 变体流程；日常使用应优先从
`scripts/manipeval_agent.py` 进入，只有调试内部阶段时才直接调用
`scripts/manipeval_taskgen.py` 或 `policy/ACT/eval_mea.sh`。完整参数以
`python scripts/manipeval_agent.py --help` 为准。

SSH 密码、私钥、UIUI key、Hugging Face token 和 checkpoint 都不得提交。运行前后可用
`git status --short` 确认工作树没有凭据或大文件。

## 6. 产物与最小检查

- Agent：`mea/evaluation_runs/<evaluation_id>/`；先看 `evaluation_report.md` 和
  `summary/evidence_bundle.json`。
- TaskGen/official expert：`mea/generated_tasks/<run_id>/`；先看 `manifest.json`、
  `validation/` 和 `evaluation/telemetry/`。
- ACT wrapper：`eval_result/<task>/ACT/`；若传了 `TELEMETRY_DIR`，轨迹也写到指定目录。
- Easy/Hard paired：`mea/paired_runs/<run_id>/`；先看 `seed_manifest.json` 和 paired summary，
  再按 condition/seed 下钻 eligibility、ACT 结果与 telemetry。
- 单个 telemetry episode：`episode.json`、`events.jsonl`、`semantic_trace.npz`；启用视觉
  捕获的 expert/ACT episode 还会有 `video.mp4`。

这些运行产物默认不进入 Git。提交代码前执行：

```bash
python -m unittest discover -s tests/manipeval -p 'test_*.py'
bash -n policy/ACT/eval_mea.sh
git diff --check
git status --short
```

若 rollout 失败，依次检查 GPU/Vulkan、assets、checkpoint 路径、`ffmpeg -version`、
`manifest.json` 的 `failure` 字段和对应 episode 的 `episode.json`。不要在同一 `run_id`
上盲目重跑；先保留失败产物，再使用新的 run/evaluation id。

多轮 ACT 可能持续数分钟，不应让主进程的 stdout 直接依赖临时 SSH 连接。服务器上优先使用
`tmux`，或把 stdout/stderr 重定向后用 `nohup` 脱离运行，再用短 SSH 连接读取 manifest：

```bash
export UIUI_API_KEY='只放在当前 shell 环境变量中'
nohup python scripts/manipeval_agent.py ... \
  > mea/evaluation_runs/<evaluation_id>.launcher.log 2>&1 </dev/null &
grep -n '"status"' mea/evaluation_runs/<evaluation_id>/manifest.json | head
```

`.launcher.log` 和 evaluation 目录均为运行产物，不进入 Git。SSH 断线后先检查原 PID、manifest
与子 run，再决定是否使用新 id 重跑；不要仅因客户端超时就重复占用 GPU。

## 7. ACT-only 敏捷协议 runner（1 / 3 / 5）

完整 Agent 的小规模重复评估使用 `scripts/manipeval_protocol.py`。它接受已有 TaskSchema 的
official 任务，也接受 `click_bell --task-profile position_lr` 的受限 generated 两轮；policy
固定为 ACT，默认 `1 repetition × 1 episode`，3 和 5 只在 smoke 通过后显式放大。它不会
接入第二种 policy，也不支持 BBH generated route。

这里 `--num-episodes`/`--episodes` 是单轮内的 rollout 数；`1 / 3 / 5` 主要指完整 evaluation
repetition 的敏捷预算。Stage 1 默认两者都取 1，只有在通路稳定且问题明确后才分别放大。

```bash
export UIUI_API_KEY='只放在当前 shell 环境变量中'

python scripts/manipeval_protocol.py \
  --repo-root "$PWD" \
  --request '评估 click_bell 的完整 Agent + ACT 通路' \
  --task-name click_bell \
  --task-module envs.click_bell \
  --run-id protocol_click_bell_smoke \
  --repetitions 1 \
  --episodes 1 \
  --chunk-size 1 \
  --start-seed 100402 \
  --model-profile economy
```

放大为 3 次时可每次只执行一个 repetition，避免长任务中断后全部重跑：

```bash
python scripts/manipeval_protocol.py ... \
  --run-id protocol_click_bell_r3 --repetitions 3 --episodes 1 --chunk-size 1
python scripts/manipeval_protocol.py --repo-root "$PWD" \
  --resume-run protocol_click_bell_r3 --chunk-size 1
```

协议目录为 `mea/protocol_runs/<run_id>/`，包含冻结配置、append-only attempt、Agent log、
逐 episode 统计、JSON summary 和 Markdown report。恢复时会校验 Git HEAD、seed schedule 和
配置 hash；缺 episode、重复实际 seed、pipeline failure 或损坏 artifact 不会被算作有效完成。
`1/3/5` 都是开发预算，不等同于论文的 10 次正式重复。

generated `position_lr` 的最小协议命令为：

```bash
python scripts/manipeval_protocol.py \
  --repo-root "$PWD" \
  --request '评估 click_bell 对左右位置变化的 ACT 泛化' \
  --task-name click_bell \
  --task-profile position_lr \
  --generated-rounds 2 \
  --run-id protocol_click_bell_position_lr_smoke \
  --repetitions 1 \
  --episodes 1 \
  --chunk-size 1 \
  --start-seed 100401 \
  --model-profile economy
```

这个 profile 的样本身份是 `(variant_id, seed)`，不是只看 seed。summary 会分别给出 left/right
variant 的 coverage、成功率、policy/physics steps、simulation time 和 rollout wall-clock；同一
numeric seed 跨 variant 复用是预期行为，复合身份缺失、额外或重复才是协议错误。

## 8. click_bell 的受限 generated 属性族

兼容 profile `position_lr` 固定运行两个 round：bell 位于安全左侧与安全右侧，两轮使用相同
ACT seed。它适合确定性回归：

```bash
python scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request '评估 click_bell 对左右位置变化的 ACT 泛化' \
  --task-name click_bell \
  --task-profile position_lr \
  --generated-rounds 2 \
  --execution-backend act \
  --start-seed 100401 \
  --num-episodes 1 \
  --max-reflections 0 \
  --model-profile economy \
  --no-history
```

每个 seed 都经过 simulator tracked-actor XY、rule 和 expert-solvability gate，再运行 ACT、
Trusted Tool、Aggregate 与 Execution VQA。Scene VQA 只判断 bell 可见性和物理合理性，
精确坐标始终以 simulator 数值为准。显式改为 `--num-episodes 3` 或 `5` 会增加每轮 ACT 与
expert gate 成本；日常开发默认保持 1。

开放属性 smoke 使用 `adaptive_properties`。模型只选择查询涉及的方面和解释真实证据；受信
运行时固定 variant、seed、gate 与 Tool，并强制证据允许的下一步方向：

```bash
export UIUI_API_KEY='只放在当前 shell 环境变量中'

python scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'How well does click_bell ACT generalize across properties of the operated bell?' \
  --task-name click_bell \
  --task-profile adaptive_properties \
  --generated-rounds 3 \
  --start-seed 100401 \
  --num-episodes 1 \
  --max-reflections 0 \
  --model-profile economy \
  --no-history
```

`object_position` 的受信 variant 是 left/right fixed XY；`object_instance` 是 RoboTwin 官方
base0/base1 实例，位置保持官方随机。实例 ID 以 simulator `task_attributes.bell_id` 为权威，
不是由 VQA 猜测。两种属性都复用同一个 `click_bell` ACT checkpoint，不需要为每个 variant
另下权重。位置方面会请求 `bell_active_tcp_min_xy_error`，触发 ToolGen 的
`generate → validate → register → reuse`；生成工具只在当前 evaluation 内注册，测量最小 XY
误差而不自行发明成功阈值。若它经过第 8.1 节的显式源码/证据审核并固定完整 hashes，则可进入
reviewed persistent registry，供后续 evaluation 精确复用；这仍不等于 Trusted Tool。可在
`execution/<round_id>/planned_tool/` 和 `tool_registry/` 核对生成、验证及后轮复用证据。

`--generated-rounds` 最多接受 5；每轮默认 1 个 rollout，真实开发应先跑 1，再按需要
放大到每轮 3 或 5。上面的直接 Agent 命令本身不做跨 repetition 统计；需要冻结复合身份和
逐变体汇总时，使用第 7 节的 `position_lr` protocol 命令。任何 N=1 结果都只能称为通路 smoke。

### 8.1 审核后跨 evaluation 复用生成 Tool

ToolGen 首次生成并通过静态、schema、determinism 和私有 oracle 校验后，只会自动进入当前
evaluation 的 run-local registry。跨 evaluation 复用必须显式生成 review 模板、审阅源码和
验证证据、把模板改成 `decision=approved` 并填写 reviewer/time/checks，再安装：

```bash
python scripts/manipeval_tool_registry.py template \
  --source-registry mea/evaluation_runs/<source_eval>/tool_registry \
  --registration-id <runlocal_registration_id> \
  > /tmp/tool_review.json

# 人工或开发代理实际审阅后再编辑 /tmp/tool_review.json；pending 文件不能安装。
python scripts/manipeval_tool_registry.py install \
  --source-registry mea/evaluation_runs/<source_eval>/tool_registry \
  --registration-id <runlocal_registration_id> \
  --review-manifest /tmp/tool_review.json \
  --reviewed-registry mea/tool_registry/reviewed
```

后续 Agent 显式增加：

```bash
--reviewed-tool-registry "$PWD/mea/tool_registry/reviewed"
```

只有 code、ToolSpec、完整 contract 和当前 telemetry schema 均精确匹配时才走
`reviewed_persistent_reuse`；每次仍重新跑当前轨迹的 determinism 与 oracle gate，且
`provider_called=false`。它不会把生成工具加入 Trusted Tool catalog。registry 运行 artifact
已由 `.gitignore` 排除。

### 8.2 无 ACT 的 TaskGen 功能验收

下面的命令只读复核既有真实 artifact，不调用模型、仿真或 ACT：

```bash
python scripts/manipeval_taskgen_acceptance.py \
  --repo-root "$PWD" \
  --output mea/validation_runs/taskgen_acceptance_stage1/acceptance.json
```

它同时核验 official reuse、`click_bell` overlay、BBH codegen/retrieval provenance 和一次
`wrong_color` 场景错误的 visual reject→diagnosis→repair。输出固定标注
`cached_artifact=true` 与 `paper_table_eligible=false`，不可当作新 rollout 或论文表结果。
默认四个 run id 指向 canonical 服务器上被 Git 忽略的历史缓存；fresh clone 不会自带这些
artifact。新环境须先分别生成四类真实 run，再用 `--official-run-id`、`--overlay-run-id`、
`--codegen-run-id` 和 `--reflection-run-id` 显式传入，不能把缺少缓存误判成代码失败。

## 9. 缓存 Planner / VQA 小验证

`scripts/manipeval_validate.py` 只评分已有 artifact，不调用模型、不重跑仿真：

```bash
cp configs/manipeval_validation_suite.example.json /tmp/mea_validation.json
# 把示例中的 artifact 路径替换为当前机器上的真实 evaluation artifact
python scripts/manipeval_validate.py \
  --repo-root "$PWD" \
  --suite /tmp/mea_validation.json \
  --budget 1 \
  --target both
```

预算仍严格为 1、3 或 5，并要求每个所选 target 至少提供相同数量的 case。Planner 报告
template precision/recall/F1、exact-set 与 first-template accuracy；VQA 报告 strict accuracy、
coverage、precision 和 AUROC。`human`、`simulator_proxy` 标签分层统计，proxy 结果不得描述为
论文的人类标注指标。产物位于 `mea/validation_runs/<run_id>/`。

## 10. ACT 三任务 N=1 instrumentation pilot

先分别完成配置指定的三个 official ACT Agent protocol，再聚合与既有 direct official ACT 的
同 task/seed 结果：

```bash
python scripts/manipeval_benchmark_pilot.py \
  --repo-root "$PWD" \
  --config configs/manipeval/act_three_task_n1.json \
  --output-dir "$PWD/mea/benchmark_runs/act_three_task_n1_smoke"
```

输出目录必须位于仓库内且尚不存在。聚合器会验证 direct paired 和 Agent protocol 的
`valid_for_comparison`，然后报告三个任务的 binary success、steps、rollout wall-clock 与同 seed
二元结论是否一致。这只是 Tables 1–2 的计量 smoke：N=1 无方差，不能计算论文式结论一致性，
报告会固定写 `paper_table_eligible=false`。

## 11. 校验 20-query aspect 草稿

这个入口无需模型 key，也不会启动仿真：

```bash
python scripts/manipeval_query_dataset.py \
  --dataset configs/manipeval_validation/query_aspects_draft_v1.json
```

它只检查 20 条 query 的字段、aspect 集和 annotation 边界，并汇总当前 capability 支持/未支持
数量。所有条目都是 `model_draft_unreviewed`，不是 human gold；在完成多人 review 与 majority
import 前，`human_agent_agreement` 必须为 `null`，因此不能作为论文 Table 6 的结果。

## 12. 缓存 montage 的 VQA image-proxy 扰动

该入口从 suite 指向的已有 Execution VQA artifact 读取真实 rollout montage；它需要视觉模型
key，但不重跑 RoboTwin：

```bash
export UIUI_API_KEY='只放在当前 shell 环境变量中'

python scripts/manipeval_vqa_perturb.py \
  --repo-root "$PWD" \
  --suite configs/manipeval_validation/vqa_perturbation_suite_v1.json \
  --budget 1 \
  --run-id validation_vqa_proxy_budget1 \
  --model-profile economy
```

预算只能为 1 / 3 / 5；每个 source clip 固定产生 clean、scene-clutter image proxy、
background-texture image proxy、lighting image proxy 四个视觉调用，所以预算 1/3/5 分别请求
4/12/20 次调用。产物位于 `mea/validation_runs/<run_id>/`，保存派生图与 source/query/numeric
evidence hash，并分别汇总各扰动。这里的变化是缓存图像变换，不是 simulator-level clutter、
纹理或光照；标签也是 simulator proxy 而非人工标注。它只验证 Tables 7–8 所需的数据通路，
accuracy/AUROC 不得作为论文复现指标。

## 13. 在线 Planner proxy 验证（Table 6-facing）

先用 1 条确认模型与 schema，再按 3 / 5 扩大；20 只用于通路稳定后的完整 proxy 集。key 只放
当前进程环境变量：

```bash
export UIUI_API_KEY='当前会话临时 key'

python scripts/manipeval_query_planner_validate.py \
  --repo-root "$PWD" \
  --dataset configs/manipeval_validation/query_aspects_development_agent_proxy_v1.json \
  --budget 5 \
  --run-id query_validation_budget5 \
  --model-profile economy
```

产物位于 `mea/validation_runs/<run_id>/`，逐 case 保存 prompt、原始 response、严格 route trace
和 score。这里的“人工”由 development agent 暂代，固定写
`human_reviewer_count=0`、`paper_table_eligible=false`；不能称为 human-agent precision。

## 14. 真实 simulator clutter 的 click_bell N=1

下面的开放 query 会经全局 Planner 选择 `robustness.scene_clutter`；overlay 调用 RoboTwin 原生
clutter generator，不生成图片代理。先保持一条 ACT rollout：

```bash
export UIUI_API_KEY='当前会话临时 key'

python scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'Evaluate click_bell ACT robustness to simulator-native table clutter.' \
  --auto-route \
  --planning-policy dynamic_evidence_v1 \
  --generated-rounds 1 \
  --start-seed 100401 \
  --num-episodes 1 \
  --telemetry-profile balanced_v1 \
  --model-profile economy \
  --reviewed-vqa-registry "$PWD/mea/vqa_query_registry/reviewed" \
  --no-history
```

clean control 使用同 seed、同 checkpoint：

```bash
python scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'Evaluate ACT click_bell on the clean official scene.' \
  --task-name click_bell \
  --task-profile official \
  --execution-backend both \
  --start-seed 100401 \
  --num-episodes 1 \
  --model-profile economy \
  --reviewed-vqa-registry "$PWD/mea/vqa_query_registry/reviewed" \
  --no-history
```

完成两条 run 后，开发代理先查看两个真实 montage，再填写一个 suite JSON，分别引用
`execution_vqa.json`、TaskGen `manifest.json`、montage、seed 和逐 phenomenon 二元标签；然后只读
汇总：

```bash
python scripts/manipeval_vqa_simulator_validate.py \
  --repo-root "$PWD" \
  --suite /tmp/click_bell_clean_clutter_proxy_labels.json \
  --output mea/validation_runs/<run_id>/validation_summary.json
```

验证器要求同 seed 的 bell pose/quaternion/id 一致，并核对 VQA 来自同一 ACT episode。
N=1 的 AUROC 必须为 `null`；开发代理标签仍不是 paper human gold。
RoboTwin 原生 clutter 可能令某个 seed 的非目标物体不稳定；scene-stability gate 拒绝时不要强行
进入 ACT，应先用 TaskGen/probe 的 0-ACT 路径寻找稳定 seed，再对 clean/clutter 使用完全相同的
seed。`100402` 是 2026-07-17 当前环境验证过的最小示例，不是跨版本永久保留的 benchmark seed。

## 15. fixed-suite 对照与 0-ACT 微消融

固定策略用 `--task-profile fixed_suite --planning-policy fixed_predeclared_v1`；动态策略用
`adaptive_properties / dynamic_evidence_v1`。公平比较前必须让两次 run 的 candidate suite 完全
一致。完成后建立只含两个 evaluation 目录的 config，再聚合：

```bash
python scripts/manipeval_compare_strategies.py \
  --repo-root "$PWD" \
  --config /tmp/fixed_dynamic_n1.json \
  --output-dir "$PWD/mea/validation_runs/fixed_dynamic_n1"
```

聚合器不启动新 rollout，并拒绝 task、ACT 逻辑配置、telemetry、base commit、开放 query、
global route、suite hash 或样本身份不一致；fixed 必须完整覆盖冻结 suite。N=1 只能称 Table 1
效率机制 facing micro-pilot，并不是论文“标准 benchmark vs MEA”的原始对照；checkpoint 文件
内容 hash 尚未记录，Table 2 consistency 固定不可用。

缓存微消融同样启动 0 次 ACT：

```bash
python scripts/manipeval_micro_ablation.py \
  --repo-root "$PWD" \
  --output-dir "$PWD/mea/validation_runs/cached_micro_ablation"
```

它只证明 gate 的功能作用，不输出论文 Table 3 成功率。

## 16. 旧 same-telemetry Tool 恢复（显式兼容）

2026-07-17 的兼容路径允许一次未预期 Tool orchestration runtime exception 的保守恢复，并校验
前后 telemetry 内容 hash 不变；它不重跑 ACT，也不是论文整轮 restart。2026-07-19 起默认关闭；
只有明确测试旧行为时才同时禁用整轮恢复并开启该路径：

```bash
python scripts/manipeval_agent.py ... \
  --tool-recovery-max-restarts 1 \
  --round-recovery-max-restarts 0 \
  --inject-tool-exception-once
```

该组合只用于兼容性开发 smoke。论文对齐的默认整轮恢复见第 20 节。旧证据位于
`execution/<round_id>/tool_recovery/attempt_*/attempt_started.json`、
`attempt_result.json` 与
`recovery_summary.json`；正常实验不要启用故障注入。

## 17. Hash 预注册与 fixed/dynamic `0-ACT` 计划

这组入口目前只支持 `click_bell demo_clean-50` 的最小 matched N=1 协议。先确认 tracked files
clean、读取当前完整 commit，并在服务器侧查看 checkpoint hash；不要把 checkpoint 复制到本机：

```bash
git status --short
git rev-parse HEAD
sha256sum \
  policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt \
  policy/ACT/act_ckpt/act-click_bell/demo_clean-50/dataset_stats.pkl
```

在被 Git 忽略的 `mea/validation_runs/prereg_inputs/<id>/` 中准备
`prereg_config.json`。它必须使用当前 `base_commit`，并显式列出 `registration_id / claim_scope /
task_name / query / candidate_suite / checkpoint_setting / expert_data_num / checkpoint_files /
telemetry_profile / sample_schedule / source_artifacts`；fixed 与 dynamic 对每个 candidate 登记相同
seed 的一条样本。checkpoint 路径必须就是上面两项，`source_artifacts` 应包含本次依赖的 runner、
catalog、配置和 reviewed registry 文件。然后生成并立即复核 manifest：

```bash
python scripts/manipeval_evidence_manifest.py \
  --repo-root "$PWD" prepare \
  --config mea/validation_runs/prereg_inputs/<id>/prereg_config.json \
  --output mea/validation_runs/prereg_inputs/<id>/evidence_manifest.json

python scripts/manipeval_evidence_manifest.py \
  --repo-root "$PWD" validate \
  --manifest mea/validation_runs/prereg_inputs/<id>/evidence_manifest.json
```

再在同一输入目录准备 `strategy_plan_config.json`，字段为 `schema_version / plan_id /
evidence_manifest / task_name / model_profile / python_executable / gpu / reviewed_tool_registry /
reviewed_vqa_registry`。registry 可为 `null`；若使用，目录内依赖文件必须已包含在 manifest 的
`source_artifacts`。`plan_id` 对应的输出目录必须尚不存在：

```bash
python scripts/manipeval_plan_strategy_pair.py \
  --repo-root "$PWD" \
  --config mea/validation_runs/prereg_inputs/<id>/strategy_plan_config.json \
  --output-dir mea/validation_runs/<plan_id>
```

输出的 `commands.md`、`command_plan.json`、`registered_route.json` 与
`strategy_comparison_config.json` 均为计划，不会自动运行命令，固定报告 provider=`false`、
ACT=`0`。先人工核对 `commands.md` 中的预算和 identity；只有决定支付最多
`2 × candidate_count` 条 ACT rollout 后，才依次执行其中的 validate、fixed、dynamic、validate
与 registered compare 命令。不要给 registered Agent 命令另加 `--auto-route` 或改 argv；任何
漂移都应被 preflight/post-hoc 拒绝。registered argv 会把 Tool 子阶段与整轮 recovery budget
都冻结为 0，使 `pair_max_act_rollouts` 仍是硬上限；异常直接记录失败，不在预注册之外增加 ACT。
N=1 的 Table 2 consistency 必须保持不可用。

## 18. TaskGen / ToolGen module-off prepare、execute 与 audit

在 ignored 输入目录创建 `module_ablation_config.json`，冻结 `study_id`、repo-relative
`artifact_root` 以及 TaskGen/ToolGen matched cases。论文 Table 3 对应的 TaskGen condition 是
`complete / no_rag / no_visual_self_check / no_readme_agent / base`，ToolGen condition 是
`complete / no_rag`；`no_visual_gate / no_tool_validation` 仅保留为旧工程兼容条件，不得写成
论文消融。每个 case
同时提供相同的 `input_identity` 和 `execution_identity`（Git、runner+hash、provider model、
config hash、seed）。先只生成 schedule：

```bash
python scripts/manipeval_module_ablation.py \
  --repo-root "$PWD" prepare \
  --config mea/validation_runs/prereg_inputs/<study>/module_ablation_config.json \
  --output-dir mea/validation_runs/<study>_schedule
```

零成本开发 smoke 可真正执行冻结的开关，并为每个 item 写入 append-only candidate、typed
outcome、execution trace 和 manifest：

```bash
python scripts/manipeval_module_ablation.py \
  --repo-root "$PWD" execute \
  --schedule mea/validation_runs/<study>_schedule/schedule.json \
  --output-dir mea/validation_runs/<study>_execution
```

development item artifact 写到 `<output-dir>/artifacts`，execution summary 与 report 也保存在
`--output-dir`；schedule 内冻结的 formal `artifact_root` 保持未占用。该 deterministic driver 只证明
开关分支和证据合同生效，固定为 provider/simulator/ACT=`0`、`paper_table_eligible=false`，不能当作
论文生成成功率。同一冻结 schedule 后续可交给真正的 provider/human-review runner 写入 formal
`artifact_root`；development manifest 绝不能交给 formal artifact audit。真实 runner 完成 schedule
指向的 typed artifact 后，再用一个全新输出目录审核：

```bash
python scripts/manipeval_module_ablation.py \
  --repo-root "$PWD" audit \
  --schedule mea/validation_runs/<study>_schedule/schedule.json \
  --output-dir mea/validation_runs/<study>_audit
```

`prepare`、内置 `execute` 和 `audit` 本身都不调用 provider、simulator 或 ACT。缺 artifact、
缺 matched pair、只有 provenance
或 runtime/identity 不可核验时，effect 必须为 `null`。completed manifest 中的历史 runtime 是
self-attested；只有接入真实生成、独立审核并形成完整 matched outcome 后，才可称论文 Table 3-facing
functional ablation。

## 19. 原生背景/光照 scene gate 与 completion-time

下面两个命令使用 RoboTwin 原生变化，运行 probe、official expert 与视觉 scene gate，但不加
`--run-act`，所以 ACT rollout 为 0。视觉 gate 需要 key 仅存在当前 shell：

```bash
export UIUI_API_KEY='当前会话临时 key'

python scripts/manipeval_taskgen.py \
  --repo-root "$PWD" \
  --request 'Validate click_bell under unseen simulator background textures.' \
  --run-id click_bell_background_scene_gate_n1 \
  --task-name click_bell --mode reuse \
  --variant-id scene_background_texture.unseen \
  --variant-hint-json '{"domain_randomization":{"random_background":true,"clean_background_rate":0.0}}' \
  --seed 100401 --num-episodes 1 --probe --expert --vision-check --max-reflections 0

python scripts/manipeval_taskgen.py \
  --repo-root "$PWD" \
  --request 'Validate click_bell under static randomized simulator lighting.' \
  --run-id click_bell_lighting_scene_gate_n1 \
  --task-name click_bell --mode reuse \
  --variant-id scene_lighting.static_random \
  --variant-hint-json '{"domain_randomization":{"random_light":true,"crazy_random_light_rate":0.0}}' \
  --seed 100401 --num-episodes 1 --probe --expert --vision-check --max-reflections 0
```

背景变化以 `task.info.texture_info` 为权威；光照变化以 simulator light configuration 为权威。
若 gate 失败，先换一个 seed 或诊断场景，不要直接启动 ACT，也不要把缓存 image proxy 当作
真实 simulator 结果。

completion-time 复用未修改的 official scene 和同一 `click_bell` ACT checkpoint。先跑一轮只验
接线；明确需要小样本稳定性时再把 episode 扩到 3 或 5：

```bash
python scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'How stable is click_bell ACT completion time across seeds?' \
  --task-name click_bell \
  --task-profile adaptive_properties \
  --generated-rounds 1 \
  --start-seed 100401 \
  --num-episodes 1 \
  --telemetry-profile balanced_v1 \
  --model-profile economy \
  --no-history
```

核对 proposal 选择 `performance.completion_time_stability.official`，TaskGen child 为 official
passthrough，Tool 为 Trusted `time_to_success`。N=1 只能证明 route/Tool/Aggregate/VQA/feedback
接线，不能称 completion-time stability 结论。

## 20. Scene-shift 证据收集与整轮恢复 smoke

只读扫描所有 completed evaluation，生成来源 hash 清单和缺失诊断：

```bash
python scripts/manipeval_scene_shift_collect.py \
  --repo-root "$PWD" \
  --output-dir mea/validation_runs/scene_shift_collection_<id>
```

也可重复传 `--evaluation-id eval_...` 限定父 run。若由 development agent 临时代替人工标注，
把 `candidate_id -> phenomenon_id -> bool` JSON 通过 `--labels` 传入，并显式传
`--reviewer-id`；输出仍是 `suite_draft`、`suite_validated=false`，不能命名为 human gold。

收集器只读取已有 artifact，不调用 provider、simulator 或 ACT。输出目录必须是新目录；修复或
升级 validator 后，应写到 `scene_collection_v2` 一类的新目录，保留旧诊断以便审计。背景条件
要求 unseen texture 的 `eval_mode=true`；静态随机光照不要求 eval mode，而读取 simulator
记录的 light configuration。Execution VQA artifact 可含 Agent 派生的 `evidence_conflict`，收集器
只用原始五字段重验 provider 响应，同时仍对完整结果文件做 hash。

Agent 的论文对齐默认值是：

```text
--tool-recovery-max-restarts 0
--round-recovery-max-restarts 1
```

正常运行无需显式传参。开发时可加 `--inject-tool-exception-once`，证明第一次 Tool 执行异常后
创建 `round_attempt_02` 与新 child run；这会真的重跑该轮 ACT，先只用一条 episode。若同时把
两个 restart budget 设为 0，fault injection 会 fail-fast。不要对 policy/simulator failure 开启
自动重试。

## 21. 检查 call-start ledger 与 round provenance

新 Agent 会在每个 provider transport 和 ACT subprocess 开始前写 append-only ledger，并在每轮
完成后写 provenance sidecar。运行结束先看 manifest 的总量，再下钻单轮：

```bash
# 当前开发服务器的非交互 shell；其他机器在 conda activate RoboTwin 后可设 PYTHON=python。
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/RoboTwin/bin/python}
EVAL=eval_example
"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); print(json.dumps({k:p.get(k) for k in ("status","runtime_totals","runtime_ledgers")},ensure_ascii=False,indent=2))' \
  "mea/evaluation_runs/${EVAL}/manifest.json"

find "mea/evaluation_runs/${EVAL}/runtime" -name call_starts.jsonl -print
find "mea/evaluation_runs/${EVAL}/summary" -name '*.provenance.json' -print
```

`provider_calls_started` 是唯一 logical call 数；`provider_transport_attempts_started` 包含重试；
`act_rollouts_started` 来自 ACT batch 在 subprocess 前声明的 rollout 数。它们都不是 completed
计数。若进程中断，应同时报告 started 与 completed episode，不要用旧的完成后计数覆盖账本。
账本不含 prompt、图像或密钥；如果账本不可写，runner 会在外部调用前失败。

每个 `summary/round_*.json` 的 `provenance` 指针应指向对应 sidecar。portfolio verifier 会对新格式
child 自动重算 sidecar 与可达 artifact hash；手工检查时至少确认 child manifest、ACT、Tool、
Execution VQA 和 runtime ledger 都在 sidecar 的 `binding.artifacts` 中。provenance 只证明引用
完整性，不证明 policy 成功。

若只复核一次真实 scene error→diagnosis→repair 源 run，不要求其他历史缓存：

```bash
"$PYTHON" scripts/manipeval_taskgen_acceptance.py \
  --repo-root "$PWD" \
  --only-reflection \
  --reflection-run-id <completed_scene_repair_run_id> \
  --output mea/validation_runs/<id>/acceptance.json
```

这是只读 post-hoc 验收；命令本身不调用 provider、simulator 或 ACT。源 run 是否真实执行必须
由其 reflection、render、expert 和 call-start artifact 证明。

需要给单任务 Agent 加一个不受 task-specific Planner 预算影响的外层硬上限时，使用
`--max-agent-rounds 1`（可选 1–5）。达到上限后 Agent 写出 auditable stop decision，不会把预算
停止推断成 policy 成败。

## 22. live-provider Table 3 micro（0 ACT）

先按第 18 节生成 hash-bound schedule。下面只选择 TaskGen/ToolGen 的 `complete` 与 `no_rag`
matched items；item id 以实际 `schedule.json` 为准：

```bash
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/RoboTwin/bin/python}
"$PYTHON" -c 'import json; p=json.load(open("mea/validation_runs/<study>_schedule/schedule.json")); print("\n".join(x["schedule_item_id"] for x in p["items"]))'

export UIUI_API_KEY='当前 shell 临时 key'
"$PYTHON" scripts/manipeval_module_ablation_live.py \
  --repo-root "$PWD" generate \
  --schedule mea/validation_runs/<study>_schedule/schedule.json \
  --output-dir mea/validation_runs/<study>_live \
  --item-id <taskgen.complete.item> \
  --item-id <taskgen.no_rag.item> \
  --item-id <toolgen.complete.item> \
  --item-id <toolgen.no_rag.item> \
  --model <schedule_frozen_model>
```

generation 结束时 `success_rates=null`，每个 item 仍是
`awaiting_development_agent_proxy`。先逐个查看 `items/*/prompt.txt`、`candidate.txt` 和
`manifest.json`，再在仓库内被忽略的目录写标签：

```json
{
  "schema_version": 1,
  "reviewer": "development_agent_proxy",
  "labels": {
    "<schedule_item_id>": {
      "success": true,
      "rationale": "说明 candidate 是否满足冻结输出合同"
    }
  }
}
```

`labels` 必须精确覆盖 generation 中的全部 item，随后单独 review：

```bash
"$PYTHON" scripts/manipeval_module_ablation_live.py \
  --repo-root "$PWD" review \
  --run-dir mea/validation_runs/<study>_live \
  --labels mea/validation_runs/<study>_proxy_labels.json
```

结果见 `review_summary.json`。当前这类运行固定为 0 simulator、0 ACT；即使调用了真实 provider，
development-agent proxy 也不是独立人工审核，不能填论文 Table 3。输出目录是 append-only；重跑
必须换新目录，不能覆盖 candidate 或 review。

## 23. 一个 Query 的 click_bell + BBH portfolio

先生成 inert command plan；这一步只检查两个 checkpoint-ready 任务并固定总 ACT 上限 2，不会
执行 child：

```bash
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/RoboTwin/bin/python}
QUERY='Across click_bell position and BBH appearance generalization, what are ACT strengths, weaknesses, and next tests?'
PORT=portfolio_cross_task_smoke

"$PYTHON" scripts/manipeval_portfolio.py \
  --repo-root "$PWD" plan \
  --portfolio-id "$PORT" \
  --query "$QUERY" \
  --output-dir "mea/portfolio_runs/${PORT}_plan" \
  --start-seed 100404 \
  --model-profile economy
```

先查看 `command_plan.json`：应恰有 `click_bell`、`beat_block_hammer` 两个 child，每个
`act_rollouts_started=1`、`--max-agent-rounds 1`、recovery budgets 均为 0。决定支付 2 ACT 后，
严格按 JSON 中的 `children[*].argv` 顺序各执行一次，不要手改 argv。两个 ordinary Agent child
完成后，用其显式 evaluation id 做 0-runtime 汇总：

```bash
"$PYTHON" scripts/manipeval_portfolio.py \
  --repo-root "$PWD" reuse \
  --portfolio-id "$PORT" \
  --query "$QUERY" \
  --output-dir "mea/portfolio_runs/${PORT}_result" \
  --click-bell-evaluation-id <completed_click_bell_eval> \
  --bbh-evaluation-id <completed_bbh_eval>
```

检查 `summary.json` 与 `report.md`：`historical_child_runtime` 应分别报告 started/completed，
每个 child 应有权威 ACT `policy_success`，最终 synthesis 应同时包含 strengths、weaknesses、
recommendations、limitations。`reuse` 本身启动 0 provider/0 simulator/0 ACT，也不能单独证明
这些 child 由当前 plan 因果启动；当前必须用本次保存的 command plan 和严格执行记录说明这一点。
两任务 N=1 仍固定 `paper_table_eligible=false`。

## 24. Bound PlanSession 的 plan-only 检查

一次 evaluation 绑定一个 task 与对应 ACT checkpoint；跨 sub-aspect 可以动态规划，不能在中途
切 task。先用 plan-only 看 `EvaluationTarget`、`TaskProposal` 和 `ToolProposal`，不启动 ACT：

```bash
cd /root/autodl-tmp/mea
PYTHON=/root/autodl-tmp/conda/envs/RoboTwin/bin/python
export UIUI_API_KEY='当前 shell 临时 key'

"$PYTHON" scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'Evaluate BBH ACT across object appearance and completion timing.' \
  --auto-route \
  --bound-task-name beat_block_hammer \
  --plan-only \
  --max-agent-rounds 2 \
  --evaluation-id eval_bound_bbh_plan_smoke \
  --model-profile economy \
  --no-history
```

检查 `mea/evaluation_runs/eval_bound_bbh_plan_smoke/plan/bound_task_session.json` 和
`evaluation_plan.json`。target 应固定 `beat_block_hammer` 与其 checkpoint；每轮必须同时有
`task_proposal`、`tool_proposal`。同理可把 `--bound-task-name` 改为 `click_bell`，Query 写 position
与 instance，并加 `--generated-rounds 2`。

检查 unsupported 能力不会诱发 task switch：

```bash
"$PYTHON" scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'Evaluate click_bell friction-coefficient generalization.' \
  --auto-route \
  --bound-task-name click_bell \
  --plan-only \
  --evaluation-id eval_bound_click_friction_unsupported \
  --model-profile economy \
  --no-history
```

这条命令应生成显式 unsupported、0-ACT evaluation，而不是改选 BBH。output 目录是 append-only；
重跑请换 evaluation id。

需要支付两条真实 ACT 验证完整链时，移除 `--plan-only`，保留 `--max-agent-rounds 2` 和
`--num-episodes 1`。例如 BBH：

```bash
"$PYTHON" scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'Evaluate BBH ACT across appearance changes, then diagnose completion timing from evidence.' \
  --auto-route \
  --bound-task-name beat_block_hammer \
  --max-agent-rounds 2 \
  --num-episodes 1 \
  --evaluation-id eval_bound_bbh_live_n1 \
  --model-profile economy \
  --no-history
```

expert 只作 solvability/control；ACT 的 `policy_success` 才是被评结果。失败的集成 run 也会消耗
ACT started，不能从批次预算中删除。

## 25. 生成 bounded Task/Tool proposal，并只物化 Task（0 ACT）

下面让模型在已经绑定的 `click_bell/object_position` capability 内提出一个不同于注册 template 的
XY 位置，同时生成 ToolProposal；`--materialize` 只运行 TaskGen setup/render probe：

```bash
PYTHON=/root/autodl-tmp/conda/envs/RoboTwin/bin/python
export UIUI_API_KEY='当前 shell 临时 key'

"$PYTHON" scripts/manipeval_proposal.py \
  --repo-root "$PWD" \
  --request 'Create one safe unseen left-side bell position and propose evidence for reachability.' \
  --task-name click_bell \
  --aspect-id object_position \
  --output-dir mea/validation_runs/proposal_click_xy_smoke \
  --run-id run_proposal_click_xy_smoke \
  --seed 100405 \
  --model-profile economy \
  --materialize
```

先看 `proposal_bundle.json`：TaskProposal 不能改 task/checkpoint/success semantics，只能改 capability
允许的根；ToolProposal 的 metric 必须可由 ToolGen resolve，VQA ids 必须来自可信问题目录。
`tool_resolution.json` 在此 CLI 中只记录 `validated_and_routed_not_executed`；实际 ToolGen/reuse 在完整
Agent rollout 中消费，不能把 route preview 写成已执行 Tool。再看 `taskgen_result.json`、
`taskgen.log` 和 `mea/generated_tasks/run_proposal_click_xy_smoke/` 的
VariantSpec、overlay/code 与真实 render。该命令固定 `act_rollouts_started=0`，不产生 policy 结论。

## 26. 生成或发布 illustrated evidence report

completed Agent 会自动在 evaluation 内写 `evidence_report.md`。也可重新生成：

```bash
PYTHON=/root/autodl-tmp/conda/envs/RoboTwin/bin/python
EVAL=eval_20260719_batch11_bbh_adaptive_n1_v3

"$PYTHON" scripts/manipeval_evidence_report.py \
  --repo-root "$PWD" \
  --evaluation-id "$EVAL"
```

要发布一个便于 GitHub/手机阅读的小 bundle：

```bash
"$PYTHON" scripts/manipeval_evidence_report.py \
  --repo-root "$PWD" \
  --evaluation-id "$EVAL" \
  --publish-dir "docs/evidence_runs/${EVAL}" \
  --max-video-mb 2
```

输出 `README.md`、`evidence_bundle_manifest.json` 以及只被报告引用的 `assets/`、`code/`、`data/`。
报告依次展示固定 task/checkpoint、TaskProposal、生成/复用代码、真实 render、ACT 视频与结果、
ToolProposal/Tool 结果、VQA montage、Aggregate/下一轮 decision 和最终回答。视频超过上限时不会
复制，缺失 artifact 显示 `N/A`。发布前用下面命令核对体积和文件清单；不要把原始大视频、
checkpoint 或完整 evaluation 目录提交 Git：

```bash
du -sh "docs/evidence_runs/${EVAL}"
find "docs/evidence_runs/${EVAL}" -maxdepth 3 -type f -print
```

## 27. 在完整 Agent 主链生成首轮新 Proposal（先 0 ACT）

`scripts/manipeval_proposal.py` 适合单独调试 Proposal；要验证它已经真正进入 Fig. 2 主调用链，
使用 `--proposal-mode novel_first_round`。第一版只开放已经有严格数值范围校验的
`click_bell/object_position`：

```bash
PYTHON=/root/autodl-tmp/conda/envs/RoboTwin/bin/python
export UIUI_API_KEY='当前 shell 临时 key'

"$PYTHON" scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'Evaluate click_bell ACT across unseen target positions and official bell instances.' \
  --auto-route \
  --bound-task-name click_bell \
  --proposal-mode novel_first_round \
  --plan-only \
  --generated-rounds 2 \
  --max-agent-rounds 2 \
  --num-episodes 1 \
  --evaluation-id eval_novel_click_plan_smoke \
  --model-profile economy \
  --no-history
```

这是 0-ACT smoke。依次检查：

- `plan/bounded_proposal/proposal_bundle.json`：新 XY 不得精确重复注册的左右 template；
- `plan/evaluation_plan.json`：registry `capability_contract` 保持不变，`TaskProposal.changes` 和
  `task_variant_id` 表示本轮新变化；
- `ToolProposal.schema_version=2`：`vqa_phenomenon_ids` 同时列出 catalog 与 `run_local.*` ID，
  `vqa_question_specs` 保存完整受限问句；
- `plan/bound_task_session.json`：task/checkpoint/round budget 未被 Proposal 改写。

移除 `--plan-only` 前，先用 `scripts/manipeval_taskgen.py` 对同一 proposal 做 setup/render probe，
确认 manifest 的 `variant_spec_authority=planner_task_proposal`、`setup_success=true`、
`render_success=true`，且 `act_rollouts_started=0`。真实 Agent 执行后，公共
`BoundTaskPlanSession` 会把证据变成唯一 directive，并在接受 task adapter 物化的下一轮前进行
`adjudicate()`；对应 artifact 为 `plan/runtime_directive_after_*.json`。

run-local VQA 只进入执行视频关键帧问题。它必须通过严格 schema，不能改写 rule Tool、官方
`check_success()` 或 simulator 数值；发生冲突时仍以数值证据为权威并保留冲突记录。

## 28. 审计论文主体方法 16 项（0 provider / 0 simulator / 0 ACT）

```bash
cd /root/autodl-tmp/mea
PYTHON=/root/autodl-tmp/conda/envs/RoboTwin/bin/python
OUT=mea/validation_runs/batch13_method_coverage

"$PYTHON" scripts/manipeval_method_coverage.py \
  --repo-root "$PWD" \
  --output "$OUT/report.json" \
  --markdown "$OUT/report.md"
```

脚本只读源码与既有 JSON artifact。`partial` 表示源码检查缺失；`evidence_pending` 表示代码已就绪，
但需要的真实 VQA/matched protocol artifact 未通过 validator；`implemented` 表示该项声明的最小检查
已通过。只有希望 CI 在任何 pending/partial 时失败，才加 `--require-all-implemented`。输出目录是
append-only；重跑请换目录。该命令即使得到 16 项 implemented，也不等于论文统计结论复现。

## 29. `bounded_each_round`：先验首轮，再用 N=1 触发第二轮

先做 0-ACT 首轮 Proposal/PlanningContext 检查：

```bash
export UIUI_API_KEY='只放在当前 shell 环境变量中'

"$PYTHON" scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'Evaluate click_bell ACT across target positions and official bell instances.' \
  --auto-route \
  --bound-task-name click_bell \
  --proposal-mode bounded_each_round \
  --plan-only \
  --generated-rounds 2 \
  --max-agent-rounds 2 \
  --num-episodes 1 \
  --evaluation-id eval_bounded_each_round_plan_smoke \
  --model-profile economy \
  --no-history
```

检查 `plan/planning_context.json`、`plan/bounded_proposal/round_01/`、
`plan/evaluation_plan.json` 和 `plan/bound_task_session.json`。`plan-only` 没有 observation，所以不会
凭空生成“证据驱动第二轮”。决定支付最小真实预算后，移除 `--plan-only` 并更换 evaluation id；
首轮 N=1 的 Aggregate/EvidencePacket 若裁决为 `continue`，才应出现
`plan/bounded_proposal/round_02/` 与 `plan/runtime_directive_after_round_1.json`。失败的 ACT started
也计入预算；稳定后才把每轮 episode 扩到 3 或 5。

## 30. 检查 typed `MetricSpec v1`（0 provider / 0 ACT）

先运行严格 DSL、编译、差分 oracle、注册与问句改写复用测试：

```bash
"$PYTHON" -m unittest -v tests.manipeval.test_metric_spec
```

真实缓存 telemetry 验收使用公开 API `mea.toolgen.execute_metric_spec(...)`：传入同一 task/schema 的
一条或更多不同 episode、一个新 metric id、`MetricSpec`、append-only output 和 registry 目录。第一遍应为
`route=typed_metric_spec_compile`，第二遍仅改自然语言 question 应为 `route=run_local_reuse`；两遍都应
`provider_called=false`。当前 operator 为 `minimum_distance`、`event_count` 和
`time_between_events`。距离算子的 signals 必须真实存在；事件算子只读取 `events.jsonl` 中的
`contact_interval` / `success_transition`，并使用受限 actor/physical selector。差分 gate 会对每条
episode 复跑 compiled Tool 并与 trusted interpreter 对照，但不要求不同 episode 的观测值彼此不同；
合法的 `0/0` 结果不能因此被拒绝。不要把 unit-test 合成 telemetry 写成真实 RoboTwin 证据，也不要
用 typed metric 覆盖 `official_check_success` 等已注册指标。

## 31. 在既有真实 rollout 上 replay run-local Dynamic VQA（0 新 ACT）

下面的示例复用一个已完成 `click_bell` evaluation 的 ACT 视频、事件、telemetry 和 Rule Tool；它会
支付一次视觉 provider 调用，但不会重新运行 TaskGen、simulator 或 ACT：

```bash
export UIUI_API_KEY='只放在当前 shell 环境变量中'

"$PYTHON" scripts/manipeval_execution_vqa_replay.py \
  --repo-root "$PWD" \
  --evaluation-id eval_20260717_stage1_global_click_flagship_v3 \
  --round-id round_1 \
  --output-dir mea/validation_runs/batch13_run_local_vqa_replay \
  --vision-model gpt-5.6-luna
```

输出目录必须事先不存在。核对 `replay_manifest.json` 的
`evidence_kind=cached_rollout_dynamic_vqa_replay`、`development_evidence_only=true`、
`act_rollouts_started=0`，以及 Execution VQA 的 live model identity、`run_local.*` question、montage 和
boolean-or-null observation。source evaluation/round/child 必须可追溯。这个结果证明 Dynamic VQA
能够消费真实 rollout keyframes，不是新的 ACT 样本，也不提供 VQA accuracy/AUROC；若源 evaluation
不存在，应换成另一条真实 completed `click_bell` run，不能用空壳或图片 proxy 冒充。

## 32. 公共 Proposal、SuccessSpec 与 EvaluationGraph 快速检查

三个核心 source contract 都可先用 0-ACT 单测检查：

```bash
cd /root/autodl-tmp/mea
PYTHON=/root/autodl-tmp/conda/envs/RoboTwin/bin/python

"$PYTHON" -m unittest -v \
  tests.manipeval.test_common_plan_proposal_adapter \
  tests.manipeval.test_success_spec \
  tests.manipeval.test_evaluation_graph
```

EvaluationGraph 的 `plan` 只调用一次 Planner，不启动 simulator/ACT；输出中的每个 node 都固定一个
task、checkpoint、aspect 和 N=1 child command：

```bash
OUT=mea/validation_runs/graph_smoke
"$PYTHON" scripts/manipeval_evaluation_graph.py --repo-root "$PWD" plan \
  --graph-id graph_smoke \
  --query 'Evaluate ACT object generalization; inspect another task family only if the first is failed or uncertain.' \
  --output "$OUT/plan.json" \
  --model-profile economy
```

先读 `plan.json` 取得真实 `node_id`、派生 `evaluation_id` 和 inert child command。只有人工决定支付
预算并按该命令得到 completed child 后，才把它转成 typed outcome，再 replay 父图；`outcome` 会精确
核验原 Query、task、aspect、单轮预算、ACT started 数和 graph 派生 ID，不能拿无关旧 run 代替：

```bash
NODE_ID='<从 plan.json 读取>'
EVAL='<该 node 对应的 completed evaluation id>'

"$PYTHON" scripts/manipeval_evaluation_graph.py --repo-root "$PWD" outcome \
  --plan "$OUT/plan.json" \
  --node-id "$NODE_ID" --evaluation-id "$EVAL" \
  --output "$OUT/outcome_1.json"

jq -s '.' "$OUT/outcome_1.json" > "$OUT/outcomes.json"
"$PYTHON" scripts/manipeval_evaluation_graph.py --repo-root "$PWD" replay \
  --plan "$OUT/plan.json" \
  --outcomes "$OUT/outcomes.json" --output "$OUT/replay_1.json"
```

必须先读取 replay 的 `next_node` 再决定是否运行第二 child。当前 CLI 不自动执行 child；plan/replay
本身不是 ACT 证据。想做完全离线的 0-provider 检查，可向 `plan` 传一个符合严格 schema 的
`--proposal-json`。

完成真实 Agent run 后，可把报告实际展示的小文件发布到 Git，而让原始 telemetry/checkpoint 留在服务器：

```bash
EVAL=eval_20260722_batch14_click_flagship_n1_v2
PUB="docs/evidence_runs/$EVAL"
test -f "mea/evaluation_runs/$EVAL/manifest.json"
test ! -e "$PUB"

"$PYTHON" scripts/manipeval_evidence_report.py \
  --repo-root "$PWD" --evaluation-id "$EVAL" \
  --publish-dir "$PUB" --max-video-mb 2
du -sh "$PUB"
```

发布前需人工核对 source evaluation 已 completed，并使用全新目录；publisher 不清旧目录。它只复制
overlay/task code、render、短视频、Tool、VQA montage 和 compact round JSON，不能替代原始 evaluation
或论文规模统计。

## 33. 动态 PlanStep、SuccessSpec repair 与 safety proxy 快速检查

先在服务器跑 0 provider / 0 simulator / 0 ACT 的定向测试：

```bash
cd /root/autodl-tmp/mea
PYTHON=/root/autodl-tmp/conda/envs/RoboTwin/bin/python

"$PYTHON" -m unittest -v \
  tests.manipeval.test_plan_session \
  tests.manipeval.test_proposal_agent \
  tests.manipeval.test_metric_spec \
  tests.manipeval.test_success_spec \
  tests.manipeval.test_taskgen \
  tests.manipeval.test_toolkit \
  tests.manipeval.test_method_coverage
```

要检查 `object_scale` 的开放 Query 路由和首轮物化，先做 `--plan-only`；它不启动 ACT，也不证明
render/expert gate：

```bash
export UIUI_API_KEY='只放在当前 shell 环境变量中'

"$PYTHON" scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'Evaluate whether the beat_block_hammer ACT policy is robust to a bounded object scale change.' \
  --auto-route \
  --bound-task-name beat_block_hammer \
  --bound-requested-aspect-id object_scale \
  --proposal-mode bounded_each_round \
  --plan-only \
  --generated-rounds 1 \
  --max-agent-rounds 1 \
  --num-episodes 1 \
  --evaluation-id eval_scale_plan_smoke \
  --model-profile economy \
  --no-history
```

TaskGen 的错误 SuccessSpec recovery 可用 CLI 的开发 fixture 走 fresh BBH `force_codegen`：
`--success-spec-fixture invalid_threshold` 必须与 `--mode force_codegen` 一起使用。该 fixture 只证明
invalid candidate → structured diagnosis → 最多一次 trusted official-equivalent fallback → compile；
它不是模型语义修复，也不能复用到正式论文结果。

`hammer_left_camera_contact_count` 只统计 actor pair `020_hammer` 与 `left_camera` 的 physical contact；
canonical aspect 是 `safety.hammer_left_camera_contact`，通用 `safety.unintended_contact` 仍是
unsupported。推荐在既有真实 BBH telemetry 上调用 trusted Tool 或 typed `event_count` 做 cached smoke；
结果为 0 也可通过 deterministic/oracle gate。不要把 robot grasp/support 接触一概算成失败，也不要
把此单项 proxy 命名为完整 unintended-contact、clearance 或 safety 指标。缓存 replay 的
`act_rollouts_started` 必须保持 0。

动态 PlanStep 的 `plan-only` 只能检查首轮范围。要证明 Sec. 3.2/Fig. 5 的 `Query + Y1:t` 数据流，
必须让当前 adaptive runtime 至少完成一轮真实 ACT N=1，并检查下一步 artifact 同时包含 Rule/VQA/
Evidence、coverage/navigation options、模型响应与最终裁决；之后才按 `1 → 3 → 5` 放大。
