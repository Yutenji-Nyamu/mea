# MEA

MEA is an agent-orchestrated evaluation workflow that supports natural-language evaluation requests.

The end-to-end Plan Agent entry point is `scripts/manipeval_agent.py`:

```bash
export UIUI_API_KEY='...'
python scripts/manipeval_agent.py \
  --request 'Evaluate ACT with a blue block and varied object positions.'
```

The inner TaskGen entry point is `scripts/manipeval_taskgen.py`, and the parameterized ACT evaluation entry point is `policy/ACT/eval_mea.sh`.

The original implementation was difficult to navigate. This repository reorganizes it as a clean, structured refactor for study and teaching.
