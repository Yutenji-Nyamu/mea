# MEA 简明运行指引

本文面向第一次运行 MEA 的用户，给出从 RoboTwin 环境到 expert、ACT 和 Agent
入口的最短路径。MEA 是基于 RoboTwin 2.0 的完整 fork；实现边界和数据流另见
[当前架构与数据流](architecture_and_dataflow_zh.md)。

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

`both` 会在本次运行结束时比较两类 episode 的实际有序 seed；若不一致则把流水线标为
失败，避免静默混用。但它尚未由一个显式 seed manifest 驱动，也没有 Easy/Hard paired
统计，因此仍只适合 smoke test 和通路核验，不能直接当作论文级 paired 对照实验。

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
