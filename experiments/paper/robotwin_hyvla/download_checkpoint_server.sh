#!/usr/bin/env bash
set -euo pipefail

# Run only on the AutoDL server from the MEA repository root.
source /etc/network_turbo >/dev/null 2>&1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
exec /root/autodl-tmp/envs/mea-libero/bin/python \
  experiments/paper/robotwin_hyvla/download_checkpoint.py
