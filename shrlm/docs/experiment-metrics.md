# Experiment metrics: coverage and gaps

The analyses in the paper are only as good as what the optimization loop wrote down while it ran. Once the full loop has run, nothing is recoverable that was not persisted — traces are large enough that we discard them, and a 15-round run is not something we repeat because a column was missing.

**Owner's job:** establish, before the full run, that every metric the analysis needs is captured. The smoke is how you exercise that end to end for pennies.

## Running the smoke

Setup once:

```bash
uv pip install -e ".[graphwalks,oolong]"
echo 'AZURE_API_KEY=...' >> .env                # Azure AI Foundry key
echo 'AZURE_FOUNDRY_ENDPOINT=https://<resource>.services.ai.azure.com' >> .env
```

The configured backend is `azure_foundry` serving `Kimi-K2.5`; the config's `model` value must equal your Foundry deployment name (the catalog default is `Kimi-K2.5`). Before any live spend, verify the deployment's pay-as-you-go rates in the Azure portal match `[pricing.list_price]` in `configs/experiment.toml`, then attest them:

```bash
export SHRLM_VERIFIED_PRICING='0.6/3.0'        # USD per 1M input/output tokens as verified
```

Then, in order:

```bash
uv run python examples/experiment_smoke.py --probe                    # ~$0.01, one call
uv run pytest tests/experiment/test_smoke_mock.py -q                  # $0, offline
SHRLM_RUN_LIVE=1 uv run pytest tests/experiment/test_smoke_live.py tests/clients/test_azure_foundry.py -q   # < $0.25
uv run python examples/experiment_smoke.py --live --out-dir ./smoke_$(date +%m%d)
```

The probe confirms token usage (cost is synthesized client-side from tokens × configured pricing — Azure returns no cost field), sampling-arg acceptance, and that instant (non-thinking) mode is honored, before anything expensive. The live pytest tier proves the real deployment inside one driver round. The full live tier runs one shrunk optimization round plus evaluation; the cumulative ceiling across all live tiers is proven under $5 before any spend. Use a fresh `--out-dir`: backends, decoding, caps, and the promotion band are identity keys, so a directory built under different values (including pre-switch OpenRouter directories) refuses to resume.

Parameters live in `configs/experiment.toml`. The `[smoke]` table overrides scale counts only; `examples/experiment_smoke.py` carries its own tighter caps for the live tier.

## Where the run writes its numbers

| Artifact | Grain |
|---|---|
| `eval/eval_summary.json` | condition × test set |
| `**/runs.jsonl` | one line per run |
| `stage_usage.jsonl` | one record per pipeline stage attempt |
| `opt/round_NN/mining/` | mined failure records, attributions, evidence bundle |
| `report.json` | measured totals and the cost projection |

Trace bodies under `**/runs/` hold the full per-turn detail. They are hundreds of MB per run at long context, are gitignored, and should be treated as transient — anything the analysis needs has to reach a durable artifact before they are discarded.

## What must be tracked

Verified present today: verifier accuracy per test set, tokens, cost, wall time, and sub-call counts at set granularity.

The following are **not** captured, or not captured at the grain the analysis needs. Each is a work item.

**From the metrics section of the proposal**

- Maximum recursion depth reached per run. Currently only exists as an input cap, never as a measurement.
- Root- versus child-level failure attribution across evaluation runs. The attribution machinery runs during mining on held-in instances only; evaluation traces are never sub-verified.
- Whole-input sub-call collapse rate — how often the harness routes essentially the whole input to a single child instead of decomposing.
- Frequency of each mined failure pattern before versus after optimization.
- Bootstrap confidence intervals and the paired significance test. Parameters are configured; nothing computes them.

**Harness-evolution metrics (new)**

- Fraction of the declared surfaces the promoted lineage actually modifies — ten under the current contract, read per round from the persisted harness rather than from a literal (a round persisted before S10 declared nine).
- Harness complexity growth: lines, characters, and tokens introduced by the promoted edits, tracked across the lineage rather than only at the endpoints.
- Harness performance at each optimization step on held-out short.

## Definition of done

A script that consumes a completed run directory and emits every table and figure the paper needs, demonstrated end to end against a smoke run. Anything it cannot produce from persisted artifacts is a gap to close **before** the full loop starts.


## OOLONG-Pairs feedback used by optimization

Attribution digests and held-in proposal evidence expose the recorded precision,
recall, F1, and missing/extra counts. These are diagnostics at the verifier's
saved three-decimal precision; exact set equality still determines success.
Representative held-in examples include at most three missing and three extra
pairs. Runtime, resource, provider content-filter and format failures have
unavailable pair counts; a partial or redirected submission is never rescored
as an answer.

Prior-round proposal history reads baseline and combined-subject links from the
ledger and saved run manifests, without opening held-out trace bodies. It reports
exact passes, cost, runtime-error count, all-attempt mean F1, and missing/extra
totals with their measured-answer denominator. Explicit runtime, resource,
content-filter and format failures contribute zero to all-attempt F1. Missing
ordinary legacy metrics make that mean unavailable (`null`); zero attempts also
have no mean. A correct empty answer contributes F1=1 and zero pair errors.

History includes each batch member's predicted effect and one shared verdict.
Only aggregate validation diagnostics and recognized structural exception text
reach the proposer. Held-out task text, pair IDs, answers, code and arbitrary
exception payloads are excluded. These are read-only projections; existing
bundles, verdicts, summaries, ledgers and completed experiments are not rewritten.
Keep the current round's held-in traces available until its proposal stage is
sealed, so the diagnosis excerpts can be verified.
