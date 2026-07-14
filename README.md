# MEA

MEA is an agent-orchestrated evaluation workflow that supports natural-language evaluation requests.

The end-to-end Plan Agent entry point is `scripts/manipeval_agent.py`:

```bash
export UIUI_API_KEY='...'
python scripts/manipeval_agent.py \
  --request 'Evaluate ACT with a blue block and varied object positions.'
```

The inner TaskGen entry point is `scripts/manipeval_taskgen.py`, and the parameterized ACT evaluation entry point is `policy/ACT/eval_mea.sh`.

The current bounded multi-round example evaluates a blue block for one episode,
feeds the observations back to the Plan Agent, then evaluates two official
position samples. Exact simulator poses and one evidence-grounded
`evaluation_report.md` are recorded for every run.
Generated scenes use bounded Visual Self-Reflection: render, diagnose, repair the complete `load_actors()`, and revalidate before policy execution.

The original implementation was difficult to navigate. This repository reorganizes it as a clean, structured refactor for study and teaching. The original RoboTwin documentation is retained in `README_RoboTwin.md`.
