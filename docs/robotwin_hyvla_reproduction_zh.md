# RoboTwin / Hy-VLA 复现与 MEA 接入

这是一份按需读取的 cold runbook。完整的网络失败、安装命令、版本、hash 和原始结果见
[`experiments/paper/robotwin_hyvla/deployment_ledger.md`](../experiments/paper/robotwin_hyvla/deployment_ledger.md)。

## 已验证结论

服务器已独立部署官方 Hy-VLA RoboTwin 权重，并在 RTX 4090 上完成：

- 官方 observation encoder 与 wrapper 的离线加载和一次有限动作推理；
- `press_stapler / demo_clean / seed=10000` 的一次 official rollout；
- 24 步内 `eval_success=true`、官方 `check_success()=true`；
- rollout 41.15 秒，模型加载 164.76 秒，CUDA 峰值分配约 9.81 GB。
- 生产 v9 完成一次 official-control 方法轮：policy success 1.0、pipeline passed、
  official Rule Tool 精确复用、Agent 主动 stop、QueryContract 验证 evidence sufficient；
  随后的 provider-only cached finalization 以 0 rollout 完成 Answer。

这只证明 Hy-VLA 可作为第二个轻量多任务 policy backend 接入 RoboTwin，不代表 50
任务均成功，也不是论文中的多策略排名或完整生成式 MEA 方法证据。v9 只评估 unchanged
official control；没有生成新 scene/checker/Tool，也没有第二轮 evidence refinement。
早期 binding、admission 和 QueryContract 失败及其修复见冷流水账第 6 节。
可移动的 v9 方法证据见
[`evidence/supplements/2026-08-01/hyvla_v9_control`](evidence/supplements/2026-08-01/hyvla_v9_control/README.md)。

## 固定路径与版本

```text
source      /root/autodl-tmp/third_party/Hy-Embodied-0.5-VLA
commit      8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9
checkpoint  /root/autodl-tmp/checkpoints/robotwin/hyvla_robotwin
HF revision bd7bba6f5934ad62293a2a34f74760c6a3ef2ff8
policy env  /root/autodl-tmp/envs/mea-hyvla
runner      /root/autodl-tmp/tmp/batch33_hyvla_runner
```

`model.safetensors` 为 9,053,587,008 bytes，SHA-256 为
`3bd6c16225f905a298340489d519498d4e5ecf5bcdd28a5c1df63e29894fef60`。

## 最短验证

先确认 source、checkpoint、GPU 和独立环境，再离线加载：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/autodl-tmp/envs/mea-hyvla/bin/python \
  /root/autodl-tmp/tmp/batch33_hyvla_runner/validate_install.py \
  --source /root/autodl-tmp/third_party/Hy-Embodied-0.5-VLA \
  --checkpoint /root/autodl-tmp/checkpoints/robotwin/hyvla_robotwin \
  --instruction 'press the stapler' \
  --output /root/autodl-tmp/tmp/batch33_hyvla_validation.json
```

正式 rollout 使用同目录的 `policy_server.py` 与 `sim_client.py`。前者在 Hy-VLA
Python 3.12 环境加载官方 wrapper，后者在既有 RoboTwin Python 3.10 环境运行仿真；
两者只通过 `127.0.0.1` 传 observation/action。完整两条命令见冷流水账第 5 节。
生产 v9 的方法边界和 cached finalization 见冷流水账第 6 节；不要把这个 N=1
official-control 验收外推成多任务表现或生成式 TaskGen 证据。

## 网络与安装经验

- checkpoint 在服务器直接下载，未经过 Windows；固定 HF revision 后再保存 manifest。
- 大型 PyTorch/CUDA wheel 直连过慢；临时 `network_turbo` 配合清华 PyPI index 更稳定。
- `flash-attn` 与反复中断的 CUDA/Triton wheel 应严格按 `uv.lock` 版本和 hash 下载，
  再以 `uv pip --no-deps` 安装；不要改登录 shell、全局 Python 或既有 SmolVLA 环境。
- `imgaug` 对 `opencv-python` 的 metadata 报警来自上游 deliberate headless override；
  已验证导入和推理，不为消除报警而混装两个 OpenCV wheel。

## 在 MEA 中如何使用

Hy-VLA 只应实现 backend hook：policy identity、checkpoint、observation/action bridge 和
rollout。外层继续统一使用：

```text
Query → Plan Agent → Proposal → TaskGen / ToolGen
      → shared RoundExecutor → policy backend → Aggregate
      → next Proposal or validated stop → Answer
```

不要把 Planner、TaskGen、ToolGen 或任务名分支复制进本实验 adapter。后续最小推进是：

1. 发布 v9 的紧凑 official-control supplement，保留 Query、Rule reuse、Aggregate、
   stop、QueryContract、Answer 和 503→cached-finalization 边界；
2. 抽 10–15 个 official task 做 N=1 coverage，失败也原样记录；
3. 只在 control 可运行的任务上选择一个无手写 schema 的生成式 MEA 干净旗舰；
4. 再比较 SmolVLA 与 Hy-VLA，不把单 episode 成功外推为模型总体排名。
