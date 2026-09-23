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

## OBLIQ-Bench Math

Not a baseline method -- a dataset. OBLIQ-Bench's `math` subset
(https://huggingface.co/datasets/dianetc/OBLIQ-Bench, "Analogue Queries")
tests long-context ranking: each query is a math competition problem, and the
gold set is every other corpus problem whose solution shares the same latent
proof technique, scored NDCG@10.

- Environment (loader, prompt builder, `ObliqBenchMathVerifier`):
  `shrlm/environments/obliq_bench_math.py`
- Requires the `obliq_bench` extra (`uv pip install -e ".[obliq_bench]"`).

This is not wired into the self-harness optimization loop; run a baseline
harness (default `H0*`, the RLM authors' own unmodified prompt -- see "H₀\*"
above) against a small live sample directly:

```bash
uv run python examples/obliq_bench_math_smoke.py --live --n 3 \
    --out-dir ./obliq_math_smoke
```

Each instance's prompt carries the full ~277k-token math corpus by default.
Sanity-check the wiring on a cheap subsampled pool first:

```bash
uv run python examples/obliq_bench_math_smoke.py --live --n 1 \
    --candidate-pool-size 200 --out-dir ./obliq_math_pool_smoke
```

Run the pinned generic λ-RLM combinator pipeline against the same environment
and verifier with:

```bash
uv run python examples/obliq_bench_math_smoke.py --live \
    --method lambda_rlm --query-ids q01522 --candidate-pool-size 200 \
    --config configs/experiment.toml --max-budget 0.10 --max-timeout 900 \
    --out-dir ./experiment_obliq_math_lambda_sanity
```

OBLIQ has no OOLONG task id, so this condition uses the pinned upstream
SPLIT→MAP→REDUCE runtime, not the OOLONG-specific Algorithm 5 reconstruction.
Its method-facing question explicitly identifies the QA task and carries the
closed-corpus ID-ranking contract; output is scored by the same
`ObliqBenchMathVerifier`.

Run a matched comparison by loading one instance set once and executing each
selected method against it:

```bash
uv run python examples/obliq_bench_math_smoke.py --live \
    --methods 'H0,H0*,lambda_rlm' \
    --query-ids q00201,q00076,q00096,q00408,q02550,q00355,q00999,q00211,q00239,q00077 \
    --candidate-pool-size 200 \
    --config configs/experiment_obliq_bench_math_DeepSeekV4Flash.toml \
    --max-budget 0.10 --max-timeout 900 \
    --out-dir ./experiment_obliq_math_azure_matched_10_v1
```

The ten fixed queries above span gold-set sizes 1, 2, 3, 4, 5, 6, 7, 8, 13,
and 37. Each method writes to its own subdirectory, while `comparison.json`
records paired per-instance NDCG, outcome, cost, runtime, and aggregate metrics.
Add `H0*R` to `--methods` only when explicitly testing recursion policy; it is
a local diagnostic variant rather than a peer reference baseline.

`--method` accepts one of `H0`, `H0*`, `H0*R`, or `lambda_rlm`; `--methods`
accepts a comma-separated matched selection. The older `--harness`
alias continues to accept `H0`, `H0*`, or `H0*R` (a locally-authored, non-reference
variant that makes `rlm_query` legible -- not one of this repo's documented
baselines, so it is opt-in, not the default). `--query-ids` selects specific
queries (comma-separated) instead of a seeded `--n`-sized sample. Backend and
pricing come from `configs/experiment_obliq_bench_math_DeepSeekV4Flash.toml`
(DeepSeek-V4-Flash via Azure Foundry) unless `--config` points elsewhere.

## SH-RLM

SH-RLM is the system under test, not a baseline. Its final harness is produced
dynamically by optimization and frozen under the experiment output directory.
