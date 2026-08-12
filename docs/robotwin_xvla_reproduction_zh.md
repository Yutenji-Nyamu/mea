# RoboTwin / X-VLA 服务器复现 runbook

本文是按需读取的 cold reference。完整逐命令、结果、失败、恢复、版本和 hash 见
[`experiments/paper/robotwin_xvla/deployment_ledger.md`](../experiments/paper/robotwin_xvla/deployment_ledger.md)。
所有下载、安装、模型加载与推理均在 AutoDL/SeetaCloud 服务器完成；Windows 只保存代码和文档。

## 1. 固定边界

```text
source      /root/autodl-tmp/third_party/X-VLA
commit      6bc2513f5f1cbec715cc668b414392a6cae5c671
checkpoint  /root/autodl-tmp/checkpoints/robotwin/xvla_robotwin2
HF revision a157c580cfe6f9f445614490f3bec1b2f9ef9f18
policy env  /root/autodl-tmp/envs/mea-xvla
runtime     /root/autodl-tmp/tmp/mea-xvla-robotwin2
```

官方资料将 X-VLA 描述为 0.9B；RoboTwin2 checkpoint 使用每任务 50 demos，官方 client
列出 50 个任务，并报告 leaderboard 设置下 Easy/Hard 聚合成功率 70%/39%。这些是上游结果，
不是本项目复测结论，也不表示所有任务必然成功。

## 2. 网络与下载

服务器默认 GitHub/Hugging Face 路由曾在 TCP 连接阶段超时。本次只在相应子 shell 临时：

```bash
source /etc/network_turbo
```

随后固定 Git/HF revision；不持久修改代理、Git 或登录 shell。普通 Python wheel 应关闭该
临时加速，直连清华 PyPI；否则 wheel 下载会明显变慢。

## 3. 隔离环境

X-VLA 使用独立环境，不修改 SmolVLA、Hy-VLA 或 RoboTwin 环境：

```bash
/root/miniconda3/bin/conda create -y \
  -p /root/autodl-tmp/envs/mea-xvla \
  --clone /root/autodl-tmp/envs/mea-robotwin-smolvla

env PIP_DEFAULT_TIMEOUT=120 \
    PIP_CACHE_DIR=/root/autodl-tmp/pip-cache-xvla \
  /root/autodl-tmp/envs/mea-xvla/bin/python -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r /root/autodl-tmp/third_party/X-VLA/requirements.txt
```

## 4. 最小离线验收

下载和环境完成后，禁网加载本地 checkpoint，并生成一个 action chunk：

```bash
cd /root/autodl-tmp/mea-worktrees/evidence-refinement-runtime
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/autodl-tmp/envs/mea-xvla/bin/python \
  experiments/paper/robotwin_xvla/validate_install.py \
  --source /root/autodl-tmp/third_party/X-VLA \
  --checkpoint /root/autodl-tmp/checkpoints/robotwin/xvla_robotwin2 \
  --steps 1 \
  --output /root/autodl-tmp/tmp/mea-xvla-robotwin2/validation.json
```

该脚本只用三张合成零值图像和合成 proprio 验证 processor、模型加载及 action shape；它没有
复现官方四相机 observation preprocessing。验收记录加载/推理耗时、`[1,30,20]` action、finite
检查和 CUDA 峰值。真实 rollout 应另建
loopback policy backend，并复用共享 `MethodRuntime`；不要在本目录复制 Plan Agent、TaskGen、
ToolGen、Aggregate 或 Answer 外层。

当前实测已通过：load 约 2.24 秒，一步去噪 action inference 约 0.46 秒，输出
`[1,30,20]` 且全 finite；CUDA 峰值 allocated/reserved 分别约 3.72/3.92 GB。checkpoint
主权重为 3,519,068,172 bytes，SHA-256 为
`3f16a4b67a1d2675fc1cb0350c6d5617522e452f47359e729d74d76b8aa4835b`。

## 5. 科学边界

X-VLA 的用途是提供另一个小型、多任务 policy backend，并扩大任务选择与策略对照。部署成功
不等于 ManipEvalAgent 方法复现，也不应把一个 official N=1 rollout 外推成 50 任务表现。
