# MEA

MEA is an agent-orchestrated evaluation workflow that supports natural-language evaluation requests.

The main entry point is `scripts/manipeval_taskgen.py`:

```bash
export UIUI_API_KEY='...'
python scripts/manipeval_taskgen.py \
  --request 'Change the red block in beat_block_hammer to blue while keeping all other behavior unchanged.' \
  --mode force_codegen --probe --vision-check --expert --run-act
```

The parameterized ACT evaluation entry point is `policy/ACT/eval_mea.sh`.

The original implementation was difficult to navigate. This repository reorganizes it as a clean, structured refactor for study and teaching. The original RoboTwin documentation is retained in `README_RoboTwin.md`.
