# Test suites

The default suite covers the paper-method mainline:

```bash
python -m pytest -q
```

The source-level inventory is currently about 250 default test functions.
Repeated operator/schema wordings, legacy aliases, and ablation-specific cases
remain in explicit cold suites; server-side `--collect-only` is authoritative.

The remaining suites stay explicit so that paper experiments and compatibility
coverage do not dominate normal method development:

```bash
# Mainline only (same as the default).
python -m pytest -q tests/mainline

# Extended and compatibility coverage retained from earlier development.
python -m pytest -q tests/manipeval

# Paper result/publishing protocols.
python -m pytest -q tests/experiments/paper

# Everything currently retained, including explicit experiment/compat suites.
python -m pytest -q tests
```

Run these commands on the AutoDL/SeetaCloud server, not on the Windows PC.
See `experiments/paper/README.md` and `compat/README.md` for the intended
ownership of future moves out of `tests/manipeval`.
