# batch20 integrated paper-claim evidence

This directory contains only the compact, hash-bound summary files for the
batch20 paper-claim experiments; it is not an archival snapshot of every raw
rollout artifact. Large runtime artifacts remain server-side under the
`mea/evaluation_runs`, `mea/generated_tasks`, and `mea/protocol_runs` paths
recorded in `evidence_manifest.json`.

The manifest supports three bounded mechanism claims:

- a finite-domain Fig. 2/5 query-to-feedback loop;
- one provider-generated Fig. 3 scene-plus-checker rollout case;
- one Fig. 4 generated, validated, registered, and exactly reused Tool.

It deliberately does **not** mark Tables 1–2, Table 3, Tables 6–8, Table 9, or
Fig. 6 as reproduced. In particular, the efficiency toy saved real resources
but changed the weakness-axis conclusion, and the ACT/DP3 pilot completed only
five of six policy rollouts.

See
[`../../development_log_20260723_batch20_integrated_claims_zh.md`](../../development_log_20260723_batch20_integrated_claims_zh.md)
for the experiment narrative and limitations.
