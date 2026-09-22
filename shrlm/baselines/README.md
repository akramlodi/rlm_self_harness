# Evaluation baselines

This package contains baseline-specific evaluation integrations.

## B1 / H₀

The sparse starting harness used by the Self-Harness optimization loop.

Implementation: `shrlm/rlm_harness.py`

## H₀*

The shipped upstream RLM reference harness reconstructed through the same
editable surfaces as H₀.

Implementation: `shrlm/rlm_harness.py`

## λ-RLM

A separate hand-designed inference method using a typed functional runtime.
For OOLONG-Pairs, the released source is missing the paper's specialized
Algorithm 5, so the local implementation reconstructs its bounded
SPLIT→MAP→PARSE→FILTER→CROSS path and identifies it explicitly as a paper
reconstruction. Other tasks continue through the pinned upstream runtime.
Each pairwise run persists its raw batch responses, retry rejections, parsed
record labels, label counts, and bounded call totals under
`metadata.pairwise_audit` in the run trace.

- Byte-identical upstream implementation: `shrlm/baselines/upstream/lambda_rlm.py`
- OOLONG-Pairs paper reconstruction: `shrlm/baselines/paper_lambda_rlm.py`
- Local evaluation adapter: `shrlm/baselines/lambda_rlm.py`
- License and provenance: `third_party/lambda-RLM/`

For a small matched live comparison against the author-style H₀* RLM harness
on four byte-identical long OOLONG-Pairs instances:

```bash
uv run python examples/lambda_rlm_oolong_pairs_long_smoke.py --live --compare-h0-star \
    --n 4 --task-ids 1,6,11,16 --out-dir ./lambda_vs_h0_star_long_smoke
```

With the shipped `$0.50` per-run cap, the configured ceiling is `$4.00`. The
command persists each method separately and writes the paired aggregate to
`comparison.json`. Add `--compare-b1` to include the sparse H₀ starting
harness too; four instances across all three conditions carry `$6.00` in
configured per-run caps.

Before a long H₀* run, isolate its execution on one short OOLONG-Pairs
instance:

```bash
uv run python examples/lambda_rlm_oolong_pairs_long_smoke.py --live \
    --conditions h0_star --context-length short --n 1 --task-ids 1 \
    --out-dir ./h0_star_oolong_short_sanity
```

`--conditions` also accepts comma-separated matched selections such as
`h0_star,lambda_rlm`; `--context-length` accepts `short` or `long`.

## SH-RLM

SH-RLM is the system under test, not a baseline. Its final harness is produced
dynamically by optimization and frozen under the experiment output directory.
