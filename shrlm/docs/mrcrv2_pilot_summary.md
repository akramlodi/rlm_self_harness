# MRCRv2 pilot summary (template — not yet run)

**Status: no live run has happened yet.** Every numeric field below is `TBD`.
This file is a skeleton, structured to mirror `experiment_kimi/POST_MORTEM.md`
(the closest existing precedent in this repo — no completed "OOLONG pilot
summary" exists to copy verbatim instead). Fill it in from the real persisted
output of the two runs below; never hand-write a number here.

Environment module: `shrlm/environments/mrcrv2.py`. Config profile:
`configs/experiment_mrcrv2.toml`. Smoke script: `examples/mrcrv2_smoke.py`.

---

## 1. Smoke test (short split, H0 unoptimized)

Run: `uv run python -m examples.mrcrv2_smoke --live --config configs/experiment_mrcrv2.toml --n-short 50 --n-long 10`

| | |
|---|---|
| Environment | MRCRv2 short (~65,536 tokens, 2 needles), seed 0 |
| Harness | H0 (unoptimized, registry incumbent) |
| Runner | Kimi-K2.5 via azure_foundry, thinking on |
| Runs | TBD |

| Metric | Value |
|---|---:|
| Total cost | TBD |
| Mean cost/run | TBD |
| Mean input tokens/run | TBD |
| Mean output tokens/run | TBD |
| Verifier pass rate | TBD |
| Sub-verifier coverage (% runs with ≥1 checkable sub-verdict) | TBD |
| Single-pass runs (no sub-calls, `derive_failing_level == NO_RECURSION`) | TBD |
| Decomposed runs (≥1 sub-call) | TBD |

Long-split (2,000,000 tokens, 8 needles) instances generated for sizing only —
not executed in the smoke test.

---

## 2. 1-2 round pilot (weakness mining → proposal → validation → promotion)

Run: `uv run python examples/run_experiment.py --config configs/experiment_mrcrv2.toml --out-dir ./experiment_mrcrv2`

| | |
|---|---|
| Config identity | TBD |
| Splits | held-in 24 (mined + validated), held-out 12 (validated only) |
| Loop | m=2, v=3, k=4, t=2, patience=3, `initial_harness = "H0*"` |
| Promotion rule | pass-count delta ≥ 0 on both splits, > 0 on at least one; cost within [0.5, 1.25]× |

Cost by stage (from `stage_usage.jsonl`):

| Stage | Cost | Share |
|---|---:|---:|
| mining | TBD | TBD |
| attribution | TBD | TBD |
| proposal | TBD | TBD |
| validation | TBD | TBD |
| **total** | **TBD** | |

### Per-round results

| Round | Held-in pass count | Held-out pass count | Candidates proposed | Promoted? | Surface touched |
|---|---:|---:|---:|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |

### Single-vs-decomposed, before/after

| | Before (H0\*, round 1 baseline) | After (final incumbent) |
|---|---:|---:|
| Single-pass runs | TBD | TBD |
| Decomposed runs | TBD | TBD |

---

## 3. Notes

- Out of scope for this pass (per the original task): the full multi-round
  optimization run and final held-out freeze, the alignment-drift probe,
  leave-one-edit-out ablations, and any change to the nine declared harness
  surfaces or the promotion rule itself.
- Both runs above spend real money against the configured `azure_foundry` /
  Kimi-K2.5 backend; each requires its own explicit go/no-go before running,
  per this repo's own live-spend gating convention
  (`examples/experiment_smoke.py --live`).
