# MEA

MEA is an agent-orchestrated evaluation workflow for natural-language manipulation-policy evaluation requests.
The current repository is a limited functional research prototype over two RoboTwin task families and a small
trusted capability catalog; it is not a reproduction of the paper-scale statistics, policy coverage, or
open-world TaskGen claims.

The end-to-end Plan Agent entry point is `scripts/manipeval_agent.py`:

```bash
export UIUI_API_KEY='...'
python scripts/manipeval_agent.py \
  --repo-root "$PWD" \
  --request 'How well does the click_bell ACT policy generalize across properties of the operated bell?' \
  --auto-route \
  --bound-task-name click_bell \
  --proposal-mode bounded_each_round \
  --plan-only \
  --no-history
```

The plan-only command checks routing and the first bounded proposal without starting simulation or ACT; it is
not policy-performance evidence. See the running guide before paying for live rollouts.

The current clean-head live acceptance completed two evidence-conditioned ACT rounds: position evidence led the
public planner to switch to an official instance. Both sampled rollouts succeeded, but the run stopped at a
two-round hard cap; untested variants and unsupported properties remain explicit. The next method gap is
query-conditioned evidence sufficiency, not another claim that two successful samples establish generalization.
See the [2026-07-23 development log](docs/development_log_20260723_reviewed_partial_route_clean_head_zh.md) and
[compact v4 evidence bundle](docs/evidence_runs/eval_20260723_batch17_clean_head_click_live_n1_v4/).

The inner TaskGen entry point is `scripts/manipeval_taskgen.py`, and the parameterized ACT evaluation entry point is `policy/ACT/eval_mea.sh`.

Start from the [Chinese documentation index](docs/index_zh.md), or go directly to the concise [setup and running guide](docs/running_guide_zh.md) for environment, assets, checkpoints, entry points, and artifacts.

The original implementation was difficult to navigate. This repository reorganizes it as a clean, structured refactor for study and teaching.
