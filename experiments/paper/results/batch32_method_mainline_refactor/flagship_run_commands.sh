#!/usr/bin/env bash
set -euo pipefail

# Reproduction template for the completed v18 configuration. Use a fresh
# EVALUATION_ID and LOG_DIR; the original immutable run is not overwritten.
: "${UIUI_API_KEY:?inject UIUI_API_KEY into this process only}"
: "${EVALUATION_ID:?choose a fresh eval_* id}"
: "${LOG_DIR:?choose a fresh server log directory}"

repo=/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime
sim_python=/root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python
policy_python=/root/autodl-tmp/envs/mea-libero/bin/python
checkpoint=/root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin
metadata=/root/autodl-tmp/checkpoints/robotwin/SmolVLM2-500M-Video-Instruct-metadata
port=18783
seed=100401
ready_file="$LOG_DIR/policy.ready.json"
query='Relative to the official grab task, does there exist a newly generated executable scene challenge that exposes a terminal alignment weakness in this policy? After observing official-control evidence, let the Plan Agent choose the most informative supported scene change without an aspect or template from me. To avoid a trivial perturbation, the chosen geometric scene change must displace the manipulated roller by at least 0.05 m while remaining expert-solvable; the Plan Agent chooses the axis and exact magnitude. Define experimental success as the official task goal AND both terminal TCPs being within 0.025 m of their corresponding roller contact points, using only current simulator point positions; do not require episode history, accumulated contact, or a trajectory-derived success threshold. Independently report one scalar metric computed from the rollout trajectory that diagnoses the chosen hypothesis, but treat that scalar strictly as diagnostic evidence and never as the terminal success outcome.'

cd "$repo"
test -z "$(git status --short)"
test ! -e "$repo/mea/evaluation_runs/$EVALUATION_ID"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$policy_python" experiments/paper/robotwin_smolvla/policy_server.py \
  --checkpoint "$checkpoint" \
  --backbone-metadata "$metadata" \
  --host 127.0.0.1 \
  --port "$port" \
  --seed "$seed" \
  --ready-file "$ready_file" \
  --max-clients 3 \
  >"$LOG_DIR/policy_server.log" 2>&1 &
policy_pid=$!
trap 'kill "$policy_pid" 2>/dev/null || true; wait "$policy_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  test -f "$ready_file" && break
  kill -0 "$policy_pid"
  sleep 1
done
test -f "$ready_file"

PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin" \
CUDA_VISIBLE_DEVICES=0 \
UIUI_API_KEY="$UIUI_API_KEY" \
"$sim_python" scripts/manipeval_agent.py \
  --request "$query" \
  --repo-root "$repo" \
  --evaluation-id "$EVALUATION_ID" \
  --benchmark robotwin \
  --auto-route \
  --bound-task-name grab_roller \
  --task-name grab_roller \
  --policy-backend smolvla \
  --execution-backend act \
  --smolvla-checkpoint "$checkpoint" \
  --smolvla-port "$port" \
  --start-seed "$seed" \
  --num-episodes 1 \
  --max-agent-rounds 3 \
  --max-reflections 1 \
  --telemetry-profile balanced_v1 \
  --model-profile balanced \
  --gpu 0 \
  --no-history \
  --base-url https://api.uiuihao.com/v1 \
  >"$LOG_DIR/agent_live.log" 2>&1
