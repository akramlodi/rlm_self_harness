# Experiment metrics: coverage and gaps

The analyses in the paper are only as good as what the optimization loop wrote down while it ran. Once the full loop has run, nothing is recoverable that was not persisted — traces are large enough that we discard them, and a 15-round run is not something we repeat because a column was missing.

**Owner's job:** establish, before the full run, that every metric the analysis needs is captured. The smoke is how you exercise that end to end for pennies.

## Running the smoke

Setup once:

```bash
uv pip install -e ".[graphwalks,oolong]"
echo 'OPENROUTER_API_KEY=sk-or-...' >> .env
```

Then, in order:

```bash
uv run python examples/experiment_smoke.py --probe                    # ~$0.002
uv run pytest tests/experiment/test_smoke_mock.py -q                  # $0, offline
uv run python examples/experiment_smoke.py --live --out-dir ./smoke_$(date +%m%d)
```

The probe confirms cost reporting and sampling args before anything expensive. The live tier runs one shrunk optimization round plus evaluation of two conditions across both environments at both lengths — roughly $0.20 and 75 minutes. Use a fresh `--out-dir`: caps and the promotion band are identity keys, so a directory built under different values refuses to resume.

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
