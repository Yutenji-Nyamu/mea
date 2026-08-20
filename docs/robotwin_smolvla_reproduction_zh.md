# RoboTwin / SmolVLA 服务器复现 runbook

本文只保留可重复部署和运行所需的稳定步骤。2026-07-29 的逐命令流水、两次
0-action 失败、临时 runner 及哈希位于冷归档：
[`experiments/paper/robotwin_smolvla/history/20260729/`](../experiments/paper/robotwin_smolvla/history/20260729/README.md)。

所有下载、安装、模型加载和仿真均在 AutoDL/SeetaCloud 服务器完成；Windows
只保存代码和文档。本文不保存 SSH 密码、provider key 或 Hugging Face token。

## 1. 稳定边界与证据入口

- Checkpoint：`lerobot/smolvla_robotwin`，约 865 MiB。
- 单卡 RTX 4090 峰值 CUDA allocated 约 1.27 GiB。
- runner 不设任务 allowlist；`--task` 在 RoboTwin 的 `envs.<task>` 中运行时解析。

本 runbook 不复制会随运行改变的成功率、样本数或最新 batch 状态。当前方法证据见
[当前证据](evidence/current/README.md)，累计结论与缺口见
[论文 claim 与 gap](paper_claim_gap_zh.md)；历史部署 pilot 与逐命令故障记录见
[冷归档](../experiments/paper/robotwin_smolvla/history/20260729/README.md)。

上游资料：

- [LeRobot RoboTwin 文档](https://huggingface.co/docs/lerobot/main/robotwin)
- [`robotwin_unified` 数据集](https://huggingface.co/datasets/lerobot/robotwin_unified)
- [`lerobot/smolvla_robotwin`](https://huggingface.co/lerobot/smolvla_robotwin)

## 2. 固定目录和版本

```text
/root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin
/root/autodl-tmp/checkpoints/robotwin/SmolVLM2-500M-Video-Instruct-metadata
/root/autodl-tmp/envs/mea-libero
/root/autodl-tmp/envs/mea-robotwin-smolvla
/root/autodl-tmp/tmp/mea-smolvla-robotwin
```

仓库 runner：

```text
mea/robotwin/smolvla_rollout.py
experiments/paper/robotwin_smolvla/policy_server.py
experiments/paper/robotwin_smolvla/sim_client.py
```

已验证环境：

| process | Python | 关键包 |
| --- | --- | --- |
| SmolVLA policy | 3.12.13 | torch 2.11.0, transformers 5.5.4, lerobot 0.6.0 |
| RoboTwin simulator | 3.10.20 | torch 2.4.1, NumPy 1.26.4, gymnasium 0.29.1 |

LeRobot 0.6 要求 Python >=3.12，而 RoboTwin/CuRobo 保留 Python 3.10 ABI，
因此不要强行合并环境。

## 3. 下载 checkpoint

服务器直连 `huggingface.co` 曾超时；`hf-mirror.com` 可用。固定 revision：

```text
967623a0f38c7e1236c66b3893c830398d793ff7
```

主权重：

```text
model.safetensors
bytes=906712520
sha256=6fbaa809585cb10924351ce101ca2d576787a472cf8c34b5e2f10c84ef8a3134
```

下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_DISABLE_XET=1

/root/autodl-tmp/envs/mea-libero/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="lerobot/smolvla_robotwin",
    revision="967623a0f38c7e1236c66b3893c830398d793ff7",
    local_dir="/root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin",
))
PY
```

SmolVLA 只需要固定的 VLM metadata，不需要再次下载完整基础权重：

```bash
/root/autodl-tmp/envs/mea-libero/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    local_dir=(
        "/root/autodl-tmp/checkpoints/robotwin/"
        "SmolVLM2-500M-Video-Instruct-metadata"
    ),
    ignore_patterns=["*.safetensors", "*.bin", "*.onnx"],
))
PY
```

下载后用 `sha256sum` 校验 checkpoint 文件。需要抗 SSH 断连时，把上述 Python
命令放入 `nohup ... > download.log 2>&1 < /dev/null &`，并记录 PID、revision
和输出目录。

## 4. 隔离 simulator 环境

```bash
/root/miniconda3/bin/conda create -y \
  -p /root/autodl-tmp/envs/mea-robotwin-smolvla \
  --clone /root/autodl-tmp/conda/envs/RoboTwin
```

不要在该 Python 3.10 clone 中安装 `lerobot==0.6.0`；pip 会因
`Requires-Python >=3.12` 拒绝。policy 继续使用
`/root/autodl-tmp/envs/mea-libero`。

## 5. 进程间合同

```text
RoboTwin/Python 3.10
  -> 三相机 raw bytes + shape、14D state list
SmolVLA/Python 3.12
  -> 50 x 14 action list
RoboTwin take_action()
```

- server 只绑定 `127.0.0.1`。
- 不跨 NumPy 1.26/2.x pickle ndarray；图像用 bytes，state/action 用 Python list。
- 相机映射：head/left/right → camera1/2/3。
- checkpoint 合同：三张 `[3,256,256]` 图像、14D state/action、50-step chunk。
- config 中历史 `[6]` state 壳与 normalizer/runtime 的 14D 不一致；运行时保留
  14D，由模型 pad 到 32D，不修改 checkpoint。

## 6. 单任务运行

先在 MEA 仓库启动 policy server：

```bash
cd /path/to/mea
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/autodl-tmp/envs/mea-libero/bin/python \
  experiments/paper/robotwin_smolvla/policy_server.py \
  --checkpoint /root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin \
  --backbone-metadata \
    /root/autodl-tmp/checkpoints/robotwin/SmolVLM2-500M-Video-Instruct-metadata \
  --host 127.0.0.1 \
  --port 18771 \
  --seed 1000 \
  --ready-file /root/autodl-tmp/tmp/mea-smolvla-robotwin/server.ready.json \
  --max-clients 1
```

ready-file 出现后，在 RoboTwin 仓库启动 client：

```bash
cd /root/autodl-tmp/RoboTwin
PYTHONPATH=/root/autodl-tmp/RoboTwin \
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python \
  /path/to/mea/experiments/paper/robotwin_smolvla/sim_client.py \
  --host 127.0.0.1 \
  --port 18771 \
  --task beat_block_hammer \
  --seed 1000 \
  --output-dir /path/to/output/beat_block_hammer_seed1000
```

产物为 `result.json`、`initial_head.png`、`final_head.png`。上面是单 episode
独立 client，所以使用 `--max-clients 1`。生产 Agent 默认每个执行任务聚合
`M=5` 个 trial；若最多允许五个执行 round，同一 policy server 应使用
`--max-clients 25`。当前 client 在每次 reset 中显式发送 trial seed，server 在
`policy.reset()` 前按该 seed 重置 Torch/NumPy RNG；同一 evaluation 的 control 与
candidate 使用同一 seed 组，场景随机性和策略随机性都可配对比较。

上面的 external-first 路径仅适用于 unchanged official standalone client。运行 MEA
生成任务时，TaskGen 与 rollout 必须使用同一 RoboTwin fork：

```bash
MEA_REPO=/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime
PYTHONPATH="$MEA_REPO:/root/autodl-tmp/RoboTwin" \
  "$MEA_SIM_PYTHON" "$MEA_REPO/scripts/manipeval_agent.py" ...
```

生产入口会把 `MEA_REPO` 提升到 `sys.path[0]`；SmolVLA runner 在生成任务上再次校验
`envs` 来源，并在 simulator setup/首帧成功后才连接 policy server。

## 7. 常见失败

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| Hub 120s 无响应 | 服务器直连超时 | 固定 revision，改 `HF_ENDPOINT=https://hf-mirror.com` |
| `No module named envs` | 脚本路径不含 RoboTwin root | 设置 `PYTHONPATH=/root/autodl-tmp/RoboTwin` |
| `numpy._core.numeric` | 跨 NumPy 版本 pickle ndarray | 使用仓库 bytes/list 协议 |
| simulator 侧 `No module named sapien` | 错用 Python 3.12 policy 环境运行 official task | policy server 用 `mea-libero`；MethodRuntime/simulator 用 `mea-robotwin-smolvla` |
| TCP connect 探针后 server 退出或不再接 Agent | `--max-clients 1` 的唯一 client 被探针占用 | 只读取 ready-file、核对其中 PID，并用 `ss` 查看 LISTEN；不要建立 socket 连接 |
| client 退出后 server 等待 | `max-clients` 尚未满足 | 终止该 server，用新 port/ready-file 重启未完成项 |
| `create_actor()` 报不支持 `runtime_name`/`scale_multiplier`，server `request_count=0` | TaskGen 用 MEA fork 验证，rollout 却因 external-first `PYTHONPATH` 导入 upstream RoboTwin | 令 MEA repo-first；保留 simulator-source gate；policy socket 只在 setup 成功后连接 |
| simulator control 失败 | 真实 policy outcome | 记录失败并短路深入链，不把它当系统错误 |
| TaskGen expert 报 `target_pose=None` | 生成场景不可解，或该 seed 的 official expert 本身不可用 | 运行同 seed official expert 对照；后者失败时终止 TaskGen，不能放宽 checker |

系统错误不得计作 episode；每次运行都记录 task、scene seed、policy seed、client
顺序、checkpoint revision、runner commit、输出目录和 GPU 峰值。

## 8. 接入 MEA

SmolVLA 适合作为共享多任务 policy backend：

```text
Query → Plan Agent → runtime task binding → Proposal
      → 按需 TaskGen(scene/checker) → SmolVLA rollout
      → Rule/VQA Tool → Aggregate → next sub-aspect / Answer
```

所有可解析 RoboTwin task 均可先尝试 unchanged official control。仓库中的
`TaskSchema` 是人工 reviewed fast path，不是任务准入表：它提供稳定 actor role、
contact point、success contract 与 richer telemetry。缺少 reviewed `TaskSchema` 时，
系统先读取 official source，再用一次 fresh simulator reset 生成 run-local
`TaskContext`；该 probe 只接受 source-bound public root 下的直接 actor attribute，或
builtin `list/tuple/dict` 容器中的 typed `access_path`，并记录 runtime name、physics
timestep、action dimension 与 callable official `check_success()`；不会从名字或图像
猜测 role、contact point 或 success threshold。

runtime-derived `TaskContext` 已进入 generic TaskGen。Batch35 在无 reviewed schema 的
`press_stapler` 上完成一次真实 cold-start acceptance：provider-written scene/checker、
AST、fixture、render/VLM、expert、SmolVLA rollout 和 live Python Tool 均通过。该结果
仍只有单任务、单 seed，且因轮次预算停止而 `evidence_sufficient=false`；因此只证明
通用入口可执行，不代表跨任务生成成功率或策略泛化。
通用 scene/checker TaskGen、请求型 VQA 与 SmolVLA rollout 已接入同一
`RoundExecutor`；不要为每个任务增加手写 allowlist、Planner 分支或测试文件。

生产 `manipeval_agent.py` 已支持 `--policy-backend smolvla`：它先建立共享 runtime
task binding，再通过 `MethodRuntime/RoboTwinMethodBackend` 执行 SmolVLA round。
具有 reviewed `TaskSchema` 的任务直接记录 semantic telemetry，并回到同一 Rule/VQA
Tool、Aggregate、Plan Agent 与 Answer 编排。无 reviewed schema 的任务若 runtime
probe 通过，可缓存本次 run-local `TaskContext` 并测量其明确发布的 raw position
signals；probe 失败则只保存 official success、视频和限制。无论哪条路径，都不能
补造未验证的语义 role、contact 或任务阈值。

运行结果、未通过阶段和下一项方法验收只在
[当前证据](evidence/current/README.md)与
[论文 claim 与 gap](paper_claim_gap_zh.md)维护；本文只定义可重复部署、进程合同与
故障恢复。

## 9. 回滚

第 2 节列出的是依赖位置，不等于本批独占资源。`mea-libero` 是共享环境，禁止作为
本功能回滚目标；checkpoint、metadata 与 `mea-robotwin-smolvla` 仅在确认没有生产
或复现实验 caller 后按精确路径清理；临时 ready/log 必须先复制所需 provenance。
仓库接入只能用 Git 回滚，evaluation run 单独保留；不得连带删除原 RoboTwin、ACT、
DP3、LIBERO 环境或 MEA 工作树。
