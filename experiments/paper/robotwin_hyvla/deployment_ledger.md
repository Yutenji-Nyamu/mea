# Hy-VLA RoboTwin deployment ledger (cold reference)

Date: 2026-08-01 (Asia/Shanghai)

Purpose: preserve the exact server deployment decisions, commands, revisions,
failures, and first official rollout. This is cold context; routine MEA development
should read the short runbook instead. All downloads, imports, compilation, model
loading, and simulation were performed on the AutoDL/SeetaCloud server. Windows
only held these small source and documentation files. No credential or proxy value
is recorded.

## 1. Frozen server state

```text
GPU               NVIDIA GeForce RTX 4090, 24564 MiB
official source   /root/autodl-tmp/third_party/Hy-Embodied-0.5-VLA
source revision   8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9
checkpoint        /root/autodl-tmp/checkpoints/robotwin/hyvla_robotwin
HF revision       bd7bba6f5934ad62293a2a34f74760c6a3ef2ff8
policy env        /root/autodl-tmp/envs/mea-hyvla
RoboTwin env      /root/autodl-tmp/envs/mea-robotwin-smolvla
uv                /root/autodl-tmp/tools/uv/uv (0.12.1)
uv cache          /root/autodl-tmp/uv-cache
vendor wheels     /root/autodl-tmp/vendor_wheels
runner copy       /root/autodl-tmp/tmp/batch33_hyvla_runner
rollout artifacts /root/autodl-tmp/tmp/batch33_hyvla_press_stapler_seed10000
```

Post-run sizes were 9.0 GiB for the policy environment, 8.5 GiB for the checkpoint
directory, and 9.7 GiB for the uv cache. The existing RoboTwin and SmolVLA
environments were not modified.

Identity/capacity probe:

```bash
hostname
pwd
id -u
df -h /root/autodl-tmp
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
```

Observed host `autodl-container-ujxcycmw77-dd0e7d70`, `/root`, uid 0. After the
rollout no Hy-VLA runner process remained and GPU memory usage returned to 0 MiB.

## 2. Source and checkpoint

Reproducible source checkout:

```bash
git clone https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA.git \
  /root/autodl-tmp/third_party/Hy-Embodied-0.5-VLA
git -C /root/autodl-tmp/third_party/Hy-Embodied-0.5-VLA \
  checkout 8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9
git -C /root/autodl-tmp/third_party/Hy-Embodied-0.5-VLA rev-parse HEAD
```

The checkpoint was downloaded server-side with the preserved
`download_checkpoint.py` and `download_checkpoint_server.sh`: four workers, a
resolved immutable revision, and the temporary Hugging Face mirror route. The
actual 2026-08-01 invocation used the same Python source from
`/root/autodl-tmp/tmp/batch33_download_hyvla.py`; the repository copy is now its
durable authority.

```bash
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
/root/autodl-tmp/envs/mea-libero/bin/python \
  experiments/paper/robotwin_hyvla/download_checkpoint.py
```

The downloader resolved revision
`bd7bba6f5934ad62293a2a34f74760c6a3ef2ff8`, fetched 11/11 files in about
11 minutes, and wrote `mea_download_manifest.json`. Required files:

```text
model.safetensors  9,053,587,008 bytes
config.json                 3,509 bytes
norm_stats.pkl             10,363 bytes
all downloaded files  9,063,330,534 bytes
```

Integrity check:

```bash
sha256sum \
  /root/autodl-tmp/checkpoints/robotwin/hyvla_robotwin/model.safetensors \
  /root/autodl-tmp/checkpoints/robotwin/hyvla_robotwin/norm_stats.pkl
```

```text
3bd6c16225f905a298340489d519498d4e5ecf5bcdd28a5c1df63e29894fef60  model.safetensors
ce81158fb16dfcb16caa80ba33dd277d07b81e40de96d959616447d614feae41  norm_stats.pkl
```

## 3. Isolated Python environment

The login environment had no suitable standalone `uv`; it was installed to an
explicit path without editing shell startup files:

```bash
mkdir -p /root/autodl-tmp/tools/uv
curl -L --fail --silent --show-error --connect-timeout 10 --max-time 90 \
  https://astral.sh/uv/install.sh \
  -o /root/autodl-tmp/tmp/uv-installer.sh
sha256sum /root/autodl-tmp/tmp/uv-installer.sh
UV_INSTALL_DIR=/root/autodl-tmp/tools/uv UV_NO_MODIFY_PATH=1 \
  sh /root/autodl-tmp/tmp/uv-installer.sh
/root/autodl-tmp/tools/uv/uv --version
```

Installer SHA-256:
`d3f5412d38c99f9d024901843bf98206f0d2c6dbe64df40d0b740e2751ca62c1`.

The official lock was used with Python 3.12.3 and a dedicated environment:

```bash
export UV_PROJECT_ENVIRONMENT=/root/autodl-tmp/envs/mea-hyvla
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/root/autodl-tmp/tools/uv-python
source /etc/network_turbo
/root/autodl-tmp/tools/uv/uv sync \
  --frozen --python 3.12 \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-install-package flash-attn \
  --project /root/autodl-tmp/third_party/Hy-Embodied-0.5-VLA
```

Network/recovery sequence:

1. Direct PyPI downloads were correct but too slow for the CUDA wheel set.
2. `network_turbo` plus default PyPI reset the PEP-517 `hatchling` tunnel.
3. `network_turbo` plus the Tsinghua default index resolved ordinary/build
   dependencies.
4. `flash-attn` and six repeatedly interrupted CUDA/Triton wheels were downloaded
   at the exact `uv.lock` versions, hash-checked, then installed locally without
   dependency re-resolution:

```bash
/root/autodl-tmp/tools/uv/uv pip install \
  --python /root/autodl-tmp/envs/mea-hyvla/bin/python \
  --no-deps /root/autodl-tmp/vendor_wheels/*.whl
```

The seven local wheels were `flash-attn==2.7.4.post1`,
`nvidia-cuda-nvrtc-cu12==12.8.61`, `nvidia-cudnn-cu12==9.7.1.26`,
`nvidia-cusolver-cu12==11.7.2.55`, `nvidia-cusparse-cu12==12.5.7.53`,
`nvidia-cusparselt-cu12==0.6.3`, and `triton==3.3.1`. Their combined directory
size is about 2.0 GiB; the observed hashes matched `uv.lock`.

Key final versions:

```text
Python 3.12.3                  numpy 1.26.4
torch 2.7.1+cu128              torchvision 0.22.1
transformers 4.57.0            timm 1.0.21
flash-attn 2.7.4.post1         CUDA runtime 12.8
```

`uv pip check` reports one known upstream metadata mismatch: `imgaug` declares a
dependency on `opencv-python`, while the official project intentionally overrides
it with `opencv-python-headless`. The official project comment documents this;
imports and inference succeed, so it is recorded rather than patched.

## 4. Offline validation

All validation used local files only:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/autodl-tmp/envs/mea-hyvla/bin/python \
  /root/autodl-tmp/tmp/batch33_hyvla_runner/validate_install.py \
  --source /root/autodl-tmp/third_party/Hy-Embodied-0.5-VLA \
  --checkpoint /root/autodl-tmp/checkpoints/robotwin/hyvla_robotwin \
  --instruction 'press the stapler' \
  --output /root/autodl-tmp/tmp/batch33_hyvla_validation.json
```

An earlier `--encode-only` pass verified the official three-camera and dual-EEF
mapping: three `[1,3,480,640]` image tensors, state `[1,32]`, and finite values.
The full validation then loaded the official wrapper in 159.80 s and produced one
finite `(16,)` action in 1.35 s. Configuration was chunk 40, execution cache 7,
image history 6 at interval 5, and video encoder enabled. Peak CUDA allocation was
9,813,333,504 bytes; peak reservation was 9,971,957,760 bytes.

The first attempt wrapped the command with `/usr/bin/time`, which is absent on this
image; it failed before Python/model loading. The script's own monotonic timings
and CUDA counters replaced that optional utility.

## 5. Official N=1 RoboTwin rollout

The official policy environment (Python 3.12) and existing RoboTwin simulator
environment (Python 3.10) cannot safely be merged. A loopback-only two-process
transport keeps both upstream dependency sets intact. The policy process uses the
official `robotwin_eval.deploy_policy.encode_obs`, `build_policy`, and
`HyVLAPolicyWrapper.get_action`. The simulator process uses unchanged RoboTwin
`setup_demo`, `get_instruction`, `get_obs`, `take_action(action_type="ee")`, and
`check_success()`.

Policy process:

```bash
RUNNER=/root/autodl-tmp/tmp/batch33_hyvla_runner
OUT=/root/autodl-tmp/tmp/batch33_hyvla_press_stapler_seed10000
mkdir -p "$OUT"
PYTHONPATH="$RUNNER" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /root/autodl-tmp/envs/mea-hyvla/bin/python "$RUNNER/policy_server.py" \
  --source /root/autodl-tmp/third_party/Hy-Embodied-0.5-VLA \
  --checkpoint /root/autodl-tmp/checkpoints/robotwin/hyvla_robotwin \
  --seed 10000 --host 127.0.0.1 --port 18781 --max-clients 1 \
  --ready-file "$OUT/server.ready.json" \
  --summary-file "$OUT/server.summary.json" \
  >"$OUT/server.log" 2>&1 &
POLICY_PID=$!
```

After `server.ready.json` appeared, the simulator client ran from the canonical
RoboTwin source directory:

```bash
cd /root/autodl-tmp/RoboTwin
PYTHONPATH="/root/autodl-tmp/tmp/batch33_hyvla_runner:$PWD" \
  /root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python \
  /root/autodl-tmp/tmp/batch33_hyvla_runner/sim_client.py \
  --host 127.0.0.1 --port 18781 \
  --task press_stapler --seed 10000 \
  --output-dir /root/autodl-tmp/tmp/batch33_hyvla_press_stapler_seed10000 \
  > /root/autodl-tmp/tmp/batch33_hyvla_press_stapler_seed10000/client.log 2>&1
wait "$POLICY_PID"
```

Accepted `result.json`:

```text
task/config/seed    press_stapler / demo_clean / 10000
success authority  eval_success=true, official check_success()=true
actions            24 / 400
network forwards   4 (20 requests served from the official action cache)
rollout wall        41.1463 s, excluding model load
model load          164.7577 s
forward latency     1.3313, 0.9162, 0.8919, 0.8759 s
CUDA peak           9,813,333,504 allocated; 9,974,054,912 reserved bytes
```

Artifact hashes:

```text
4128fe4a2d3c13d0dc70750d7b7ee240a9ed28d980a804b7393c587c6f36bc25  result.json
fb32783932d217792e86eeb2b1ad14ad8f8a7b71c44feb3b8a772c94783c0055  initial_head.png
ee10e796d67de13df0af8282cd46eba8f489cd21b1bd5731c5dc443f81940ed7  final_head.png
```

SAPIEN emitted its existing Vulkan-ICD warning, RoboTwin printed
`missing pytorch3d`, and Warp emitted a deprecation warning. Rendering, rollout,
and official success all completed; these warnings were therefore non-blocking.

## 6. Production MEA integration attempts

The standalone official rollout above succeeded, but the first three production
MEA invocations all stopped before connecting to the policy server. They are
method-admission failures, not Hy-VLA inference or task failures. The exact v3
launcher remains on the server at
`/root/autodl-tmp/tmp/batch33_hyvla_control_live_v3.sh`; v1 and v2 remain beside
it as `batch33_hyvla_control_live.sh` and
`batch33_hyvla_control_live_v2.sh`. The shared command below is copied from the
v2 launcher; v1/v3 differences are recorded in the table instead of duplicating
the whole script.

```bash
PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin" CUDA_VISIBLE_DEVICES=0 \
UIUI_API_KEY="$UIUI_API_KEY" "$sim_python" scripts/manipeval_agent.py \
  --request "$query" --repo-root "$repo" \
  --evaluation-id "$evaluation_id" --benchmark robotwin --auto-route \
  --bound-task-name press_stapler --policy-backend hyvla \
  --execution-backend act --hyvla-checkpoint "$checkpoint" \
  --hyvla-source "$source_dir" \
  --hyvla-python-env /root/autodl-tmp/envs/mea-hyvla \
  --hyvla-port "$port" --start-seed 10000 --num-episodes 1 \
  --max-agent-rounds 1 --max-reflections 1 \
  --telemetry-profile balanced_v1 --model-profile balanced --gpu 0 \
  --no-history --base-url https://api.uiuihao.com/v1
```

| evaluation id | confirmed configuration | result | diagnosis and response |
| --- | --- | --- | --- |
| **eval_20260801_batch33_hyvla_plan_agent_control_v1** | press_stapler, Hy-VLA external backend; model load 164.323 s | PlanningContextError: schema-less policy binding requires a positive policy.physics_timestep_seconds; no policy connection and 0 rollout | Runtime binding metadata was incomplete. Commit **73d43ed** added physics_timestep_seconds=0.004 and action_chunk_size=6; this was a binding fix, not a model change. |
| **eval_20260801_batch33_hyvla_plan_agent_control_v2** | Query: “Does this Hy-VLA policy complete the unchanged official press_stapler task in RoboTwin? Run exactly one official control episode and answer only from that evidence.”; model load 165.023 s | Provider correctly returned scene/checker/VQA=false and official Rule reuse, but schema-less target resolution still classified it generation_required_no_registered_candidate / execution_authorized=false; completed_without_execution, 0 rollout | The resolver incorrectly required an old `target.aspects` entry after typed needs had already authorized unchanged official execution. The current fix binds `control_template_id(target)` directly and is covered by a schema-less regression. |
| **eval_20260801_batch33_hyvla_plan_agent_control_v3** | Broad weakness Query, press_stapler, max_agent_rounds=1; model load 165.77 s | unsupported_candidate_domain, completed_without_execution; no policy connection and 0 rollout | A broad candidate was not materialized into an executable Proposal. No follow-up rollout was run, so production Hy-VLA acceptance remains open. |

All three attempts paid model-load time before the method admission failure was
known. A future launcher should complete route/binding/admission before starting
the external Hy-VLA server whenever the selected round does not yet require policy
inference. None of these attempts changes the accepted standalone N=1 result in
section 5.

## 7. Upstream integration boundaries

- The official quick-start/deploy examples contain a stale checkpoint-name
  assumption; the pinned absolute checkpoint path above avoids it.
- `scripts/eval_robotwin_test.sh` contains a top-level Bash `local`, which is invalid
  outside a function. This adapter does not modify upstream source.
- The official examples assume policy and simulator can share one environment;
  dependency isolation is the only reason for the loopback transport.
- This is one successful official episode, not evidence of the published 50-task
  aggregate, sample efficiency, or policy ranking.
- It is not yet a complete MEA round: there is no generated Proposal/scene/checker,
  Tool/VQA/Aggregate, or evidence-conditioned next Plan Agent decision in this
  pilot. The three production attempts in section 6 also produced zero rollout.
  The production integration should add only a backend binding and reuse the
  shared method runtime.

## 8. Durability boundary

This file is sufficient to reproduce the pinned source checkout, checkpoint
snapshot, isolated locked environment, offline validation, and standalone N=1
rollout. It is not a byte-for-byte terminal transcript:

- the original checkpoint downloader remains at
  `/root/autodl-tmp/tmp/batch33_download_hyvla.py`; its exact source is now also
  preserved as `download_checkpoint.py` in this directory;
- the exact per-wheel interrupted download/retry commands and individual wheel
  hashes were not copied into the repository; the durable authority is the pinned
  uv.lock versions plus the recorded successful hash check;
- the three original production launchers remain in server tmp rather than Git;
  section 6 preserves the exact shared Agent command, immutable evaluation IDs,
  per-attempt differences, outcomes, and artifact paths.

These omissions should not be filled with reconstructed history. A future
deployment should copy its generated downloader, bounded network log, and final
package/hash manifest into this cold experiment directory before temporary server
files are removed.
