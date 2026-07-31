#!/usr/bin/env bash
set -euo pipefail

: "${UIUI_API_KEY:?inject UIUI_API_KEY into this process only}"

repo=/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime
sim_python=/root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python
policy_python=/root/autodl-tmp/envs/mea-libero/bin/python
checkpoint=/root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin
metadata=/root/autodl-tmp/checkpoints/robotwin/SmolVLM2-500M-Video-Instruct-metadata
log_dir=/root/autodl-tmp/mea-run-logs/batch32_clean_flagship_v1
ready_file="$log_dir/policy.ready.json"
port=18783
seed=100401
commit=b75ee8434aaa1500310fb0063388fbfd5daea145
query='Does there exist a newly generated executable scene challenge that exposes a trajectory weakness in this policy? Let the Plan Agent choose the scene change. Define success as completing the official task goal AND satisfying one additional boolean condition read directly from current simulator object/contact state, never from a trajectory-derived threshold. Independently report one scalar metric computed from the rollout trajectory that decides the hypothesis.'

cd "$repo"
test "$(git rev-parse HEAD)" = "$commit"
test -z "$(git status --short)"
test ! -e "$repo/mea/evaluation_runs/eval_20260731_batch32_clean_flagship_plan_v1"
test ! -e "$repo/mea/evaluation_runs/eval_20260731_batch32_clean_flagship_live_v1"
test -z "$(ss -H -ltn "sport = :$port")"
mkdir -p "$log_dir"

# Gate 1: provider-backed plan-only. This does not start simulator or policy.
PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin" \
UIUI_API_KEY="$UIUI_API_KEY" \
"$sim_python" scripts/manipeval_agent.py \
  --request "$query" \
  --repo-root "$repo" \
  --evaluation-id eval_20260731_batch32_clean_flagship_plan_v1 \
  --benchmark robotwin \
  --auto-route \
  --bound-task-name grab_roller \
  --task-name grab_roller \
  --task-module envs.grab_roller \
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
  --plan-only \
  2>&1 | tee "$log_dir/plan_only.log"

# Stop here for manual inspection if the plan exposes an aspect/template
# itinerary, freezes round 2 before evidence, or requests an unverifiable
# checker relation.

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
"$policy_python" \
  experiments/paper/robotwin_smolvla/policy_server.py \
  --checkpoint "$checkpoint" \
  --backbone-metadata "$metadata" \
  --host 127.0.0.1 \
  --port "$port" \
  --seed "$seed" \
  --ready-file "$ready_file" \
  --max-clients 3 \
  >"$log_dir/policy_server.log" 2>&1 &
policy_pid=$!
trap 'kill "$policy_pid" 2>/dev/null || true; wait "$policy_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  test -f "$ready_file" && break
  kill -0 "$policy_pid"
  sleep 1
done
test -f "$ready_file"
ss -H -ltn "sport = :$port"

PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin" \
CUDA_VISIBLE_DEVICES=0 \
UIUI_API_KEY="$UIUI_API_KEY" \
"$sim_python" scripts/manipeval_agent.py \
  --request "$query" \
  --repo-root "$repo" \
  --evaluation-id eval_20260731_batch32_clean_flagship_live_v1 \
  --benchmark robotwin \
  --auto-route \
  --bound-task-name grab_roller \
  --task-name grab_roller \
  --task-module envs.grab_roller \
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
  2>&1 | tee "$log_dir/agent_live.log"
