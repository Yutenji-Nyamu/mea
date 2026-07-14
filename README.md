# MEA

MEA is an agent-orchestrated evaluation workflow that supports natural-language evaluation requests.

The end-to-end Plan Agent entry point is `scripts/manipeval_agent.py`:

```bash
export UIUI_API_KEY='...'
python scripts/manipeval_agent.py \
  --request 'Evaluate ACT in a scene with a blue block.'
```

The inner TaskGen entry point is `scripts/manipeval_taskgen.py`, and the parameterized ACT evaluation entry point is `policy/ACT/eval_mea.sh`.

Each completed agent run also produces one evidence-grounded `evaluation_report.md` for the user.

The original implementation was difficult to navigate. This repository reorganizes it as a clean, structured refactor for study and teaching. The original RoboTwin documentation is retained in `README_RoboTwin.md`.
