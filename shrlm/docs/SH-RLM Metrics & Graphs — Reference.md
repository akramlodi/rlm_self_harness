# SH-RLM Metrics & Graphs — Reference

## 1. What the proposal specifies

### 1.1 Optimization phase (source environment, propose/validate/promote loop)

**Primary loop mechanics (section 3.3, section 4 cost model)**
- Per round: held-in traces mined, `K` candidates proposed, each targeting exactly one declared surface (S1–S9)
- Per candidate: held-in delta (Δin) and held-out delta (Δho) in pass count vs. incumbent
- Promotion rule: accept only if Δin ≥ −τ_reg, Δho ≥ −τ_reg, and max(Δin, Δho) > τ_imp, and mean sub-call count/cost within a preregistered band vs. incumbent
- Structural rejections tracked separately from validation-gate rejections (stale base harness, zero/multiple surfaces changed, harness-invariant violation)
- Merged-edit re-evaluation when multiple accepted candidates touch disjoint surfaces in the same round; if the merge fails, nothing is promoted that round
- Stopping criterion: fixed max rounds `T`, or early stop after N consecutive rounds without a promotion
- Round cost vs. the `R_round = m·n_in + v(K+1)(n_in+n_ho)` estimate

**Mechanism-level analysis (section 3.5, called out as distinct from raw accuracy)**
- Frequency of each mined failure pattern (signature `ϕ = (verifier_cause, failing_level, causal_status, agent_mechanism)`), compared before optimization vs. after
- Whole-input sub-call collapse rate — a run that delegates most/all of the input to one child or performs no meaningful decomposition
- Root-vs-child failure attribution share, using sub-verifier-grounded labels — tests whether optimization repaired errors or just relocated them down the call tree

**Graphs / figures explicitly described**
- **Figure 1**: the initial harness H₀ shown as a 9-surface grid (S1–S9), most surfaces empty/generic at the start — a static snapshot, not a time series
- **Figure 2**: one optimization round as a flow diagram (mining → proposal → validation), with a side grid showing S1–S9 where *tinted cells = surfaces edited in earlier rounds* and *the solid cell = the newly merged edit this round*. This is the closest thing the proposal has to a "harness evolution / levers touched" visual, though it's presented as a single-round schematic rather than an explicit time-series plot.
- No explicit "accuracy vs. round" line chart is specified in the text, but section 3.5's ask to report "frequency of each mined failure pattern before and after optimization" and section 5's Week 10 plan to "compute verifier accuracy... as results arrive" implies a quality-over-time analysis is expected, even though no figure is mocked up for it in the proposal itself.

### 1.2 Evaluation phase (frozen harness, four test sets, vs. baselines)

**Primary metric**
- Verifier accuracy per condition (B1, H1, SH-RLM, optionally F1) × per test set (source-short, source-long, target-short, target-long)
- Bootstrap confidence intervals over task instances
- Paired per-instance test: SH-RLM vs. B1, and SH-RLM vs. H1

**Secondary efficiency metrics** (same condition × test-set grid)
- Total input/output tokens
- Recursive-call count
- Maximum recursion depth
- Accuracy per million tokens

**Mechanism-level metrics, carried into evaluation**
- Whole-input sub-call collapse rate, per condition × split
- Root-vs-child failure attribution share, per condition × split

**Integrity / audit requirements**
- `harness_hash` re-verified against the freeze-time hash
- `instances_sha256` / `split_file` proving the exact frozen test split was used, byte-identical across conditions
- `usage_lower_bound` flag and `skipped_run_ids` surfaced wherever the spend breaker tripped, rather than silently averaged into the headline numbers

**Graphs / figures**
- No explicit evaluation-phase figure is mocked up in the proposal text. Week 10–11 of the timeline describes computing and reporting the above metrics (accuracy with CIs, paired comparisons, token efficiency, recursion depth, call counts, sub-call collapse) but leaves the specific chart forms to the results-writing stage.

---

## 2. What is currently tracked in the codebase

### 2.1 Data already captured (per Claude Code's architecture summary)

**Per-run** (`mining.py`, `driver.py` → `runs.jsonl`)
- `run_id`, `instance_id`, `attempt`, `passed`, `cost`, `timestamp`
- `verdict`: cause, gold vs. produced, detail
- `trace_path` + `trace_sha256`

**Per-failure** (`attribution.py`, `digest.py`, `taxonomy.py` → `records.jsonl`)
- `signature`: verifier_cause, failing_level, agent_mechanism, causal_status
- `detail`: symptom summary, evidence node ids
- `stats`: n_iterations, n_nodes, n_rlm_children, max_observed_depth, collapse_ratio, suspected_lost_subcalls, trace_integrity, block_attribution_reliable
- `digest_sha256`, `level_grounded`, `attribution_failed`, `attribution_error_kind`

**Per-pattern** (`clustering.py`, `bundle.py` → `bundle.json`)
- `signature`, support/instance_support, actionability score, `grounded_fraction`, `below_support_floor`
- `shared_symptoms`, `verifier_evidence`, `representatives`
- `integrity`: known_substrate_biases, n_resource_terminated, n_transport_errors, mean_digest_coverage
- `config`: model, prompt version, taxonomy version, seed, harness_hash

**Per-candidate/round** (`proposal.py`, `validation.py`, `promotion.py`, `costs.py`)
- `proposal.json` + generated `surfaces.py` per candidate, including an explicit top-level `"surface"` field (e.g. `"S2"`)
- `decision.json`: baseline vs. candidate, promoted (bool), promoted_harness_hash, constituent_ids, excluded candidates and why
- `promotions.jsonl`: append-only ledger; each row now also carries a `surface` field (see §2.3), plus numeric `rule.heldin` / `rule.heldout` (baseline_pass_count, candidate_pass_count, delta, n_runs) and `band` (e.g. mean_cost baseline/candidate/within) for every scored candidate, promoted or not
- `CandidateSpendBreaker`: over_budget / RESOURCE_TERMINATED flags (lower bound, not silently dropped)

**Cross-cutting** (`usage.py` → `stage_usage.jsonl`)
- stage, round_index, attempt_index, cost, input_tokens, output_tokens, cache_hits, wall_seconds, resumed, `lower_bound` flag

**Evaluation loop** (`shrlm/experiment/evaluation.py` → `eval_summary.json`)
- pass_count, pass_rate, n_runs, n_instances, n_resource_terminated
- total_cost, mean_cost, input_tokens, output_tokens, wall_seconds
- mean_sub_calls / total_sub_calls
- `instances_sha256` + split_file, `harness_hash` (re-verified), `usage_lower_bound`, `skipped_run_ids`
- `report.py` / `scenarios.py` roll this into `report.json` (extrapolated cost/time projections)

**Audit** (`audit.py`) — verifies the whole hash chain is internally consistent (no orphaned references); doesn't track new data itself.

### 2.2 Gaps identified against the proposal, and how each was resolved

| Gap | Status | Resolution |
|---|---|---|
| Target surface not present on decision-layer records (`promotions.jsonl`), only on `proposal.json` | **Fixed** | `CandidateDecision` gained a `surface: str \| None = None` field, serialized in `to_dict()`. `score_candidate`, `decide_subject`, and `assess_round` now thread the surface through from the proposal loader. Applies to all future rounds; historical `promotions.jsonl` rows written before the fix still lack `surface` and would need a one-off backfill (join by `subject_id` against `proposal.json`) if those older rounds need to be included in analysis. |
| Losing-candidate Δin/Δho not distinguishable from a category label | **Confirmed already present, no code change needed** | `promotions.jsonl` rows already carry the full numeric `rule.heldin`/`rule.heldout` (baseline_pass_count, candidate_pass_count, delta, n_runs) for every scored candidate, not just the promoted one. Only exception: candidates rejected upstream of scoring (e.g. `over_budget`) have `rule: null`, since there's nothing to score against. |
| Mechanism-frequency (failure pattern signature) diff between two points in time (e.g. round 0 vs. freeze) | **Built** | `shrlm/experiment/pattern_frequency_diff.py` (see §2.3, Analysis — Mechanism-frequency diff). Confirmed as an analysis gap, not a capture gap: `bundle.json` already carried everything needed (`patterns[].signature`, `instance_support`, `totals.n_runs`, `config.round_index`); this is a downstream script over existing artifacts, no new instrumentation. |
| `surface = None` on merged-harness decision rows (a merge spans >1 surface, so a single value would misrepresent it) | **By design, not a gap** | Merge rows correctly stay `surface = None`; each constituent candidate retains its own individual surface on its own decision row instead. |

### 2.3 Graphs and analysis scripts implemented, and the decisions behind them

#### Graph A — Harness complexity / levers touched over time

**Aggregator:** `shrlm/experiment/surface_activity.py`, run via `python -m shrlm.experiment.surface_activity <out_dir>`. Reads `promotions.jsonl` in round order, writes `surface_activity.csv` and `surface_activity_unattributed.csv`.

**Decision — "attempted" vs. "actually changed" (levers touched):** Track **both**, as two separate counts per `(surface, round)`:
- `attempted_count` — increments for every row with a non-null `surface`, regardless of decision outcome. Represents where the proposer is spending search effort.
- `promoted_count` — increments only for rows that represent a real harness change: `decision == "promoted"` **or** (`decision == "accepted"` and the row is a merge constituent, i.e. `merge.role == "constituent"`). This second clause was a fix applied after integration testing surfaced that a merged promotion's constituent surfaces (e.g. S2 and S3 both winning and composing) were being missed — the ledger's `"promoted"` row is the synthetic merged subject, which has no single surface, while the actual constituent surfaces sit on rows marked `"accepted"`, not `"promoted"`. Verified post-fix: a merge round with S2+S3 constituents now correctly shows `cumulative_surfaces_promoted = 2` after that round (was 0 before the fix); `attempted_count` was confirmed unaffected by the same check, since it was already unconditional on decision outcome.

Cumulative series (`cumulative_surfaces_attempted`, `cumulative_surfaces_promoted`) are derived by counting distinct surfaces with a non-zero count up through each round.

Rows with `surface = None` (loader-gate rejections, the synthetic merged record itself) are excluded from surface-level counts but logged separately per round as `unattributed_rows_by_round`, so the exclusion is visible rather than silent.

**Plotting:** `shrlm/experiment/plot_surface_activity.py`, run via `python -m shrlm.experiment.plot_surface_activity <out_dir>`, writes `<out_dir>/plots/surface_activity.png` (static, matplotlib).
- **Panel A**: dashed "attempted" vs. solid "promoted" step lines, deliberately the **same color/hue** rather than two distinct colors — since promoted is always a subset of attempted each round, same-hue communicates the containment relationship rather than implying two independent series. Y-axis capped with a horizontal reference line at y=9 ("all surfaces"), end-of-line value labels.
- **Panel B**: two S1–S9 × round heatmaps (attempted, promoted) sharing a single color scale/colorbar so the two are directly comparable. Zero-activity cells render fully blank (masked to background), not a pale tint, so untouched surfaces are visually distinct from barely-touched ones — this was a bug caught and fixed during visual review (zero cells originally rendered as a visible pale blue). All 9 surfaces are always shown as rows even if some have zero activity across every round, to keep the "7 of 9 surfaces start empty" framing visible.
- A footnote reports total unattributed rows and flags any round exceeding 25% unattributed.
- A second bug caught during review: heatmap x-axis originally showed fractional round ticks (1.0, 1.5, 2.0…) instead of integers; fixed.

#### Graph B — Harness quality while looping

**Aggregator:** `shrlm/experiment/incumbent_quality.py`, run via `python -m shrlm.experiment.incumbent_quality <out_dir>`. Reads `promotions.jsonl` in round order, writes two CSVs:
- `incumbent_quality.csv` — one row per round: `round_index, heldin_pass_rate, heldout_pass_rate, incumbent_changed (bool), annotation (nullable)`. Incumbent state starts at H₀'s baseline values (from round 1's `rule.heldin/heldout.baseline_pass_count`) and updates only when a candidate is promoted that round; otherwise carries forward flat.
- `incumbent_quality_candidates.csv` — one row per scored candidate per round (added during this work; not present in the original aggregator), using each candidate's own `candidate_pass_count` rather than the shared incumbent baseline. Needed for the optional per-round scatter overlay.

**Decision — pass rate vs. raw pass count:** Use **pass rate** (normalized by n_runs), not raw pass count, since held-in and held-out splits have different sizes (e.g. n_in=24 vs. n_ho=40 in the proposal's example numbers) and raw counts aren't directly comparable across the two lines.

**Decision — handling of `rule: null` rows** (candidates rejected upstream of scoring, e.g. `over_budget`, no delta to speak of): **skip with optional annotation** — the incumbent line stays flat through that round (correct, since the incumbent genuinely didn't change), and a separate annotation record is emitted with the rejection reason, for optional plotting as a marker rather than being silently dropped.

**Plotting:** `shrlm/experiment/plot_incumbent_quality.py`, run via `python -m shrlm.experiment.plot_incumbent_quality <out_dir> [--show-candidates]`, writes `<out_dir>/plots/incumbent_quality.png` (static, matplotlib).
- Held-in (blue/circle) vs. held-out (orange/square) step lines.
- Black ring markers at rounds where `incumbent_changed == True`.
- Small triangle ticks below the axis at annotated (`rule: null`) rounds, keyed to a numbered footnote list with the rejection reason text (static output, so no hover tooltip — footnote list instead).
- `--show-candidates` flag (off by default) overlays every scored candidate's own pass rate as faint background points, from `incumbent_quality_candidates.csv`.

#### Analysis — Mechanism-frequency diff (no graph yet)

**Script:** `shrlm/experiment/pattern_frequency_diff.py`, run via `python -m shrlm.experiment.pattern_frequency_diff <bundle_1> <bundle_2> [<bundle_3> ...] --out <out_dir> [--labels <label_1> <label_2> ...]`. Direct code-level counterpart to the proposal's "report the frequency of each mined failure pattern before and after optimization" (section 3.5) — a downstream script over already-captured `bundle.json` artifacts, not a new instrumentation point.

**Join key:** full outer join of two bundles' `patterns[]` on the four-tuple failure signature (`verifier_cause`, `failing_level`, `causal_status`, `agent_mechanism` — `FailureSignature.key()`). Every signature present in either bundle gets one row.

**Decision — the rate denominator, stated honestly rather than assumed exact:** `bundle.json` has no distinct-instance count to normalize `patterns[].instance_support` against — `MiningTotals.n_runs` is a *run* count (`len(runs)` mined that round), not an instance count. `support_rate = instance_support / totals.n_runs`. When mining used one attempt per instance (the common case) the two coincide and this is an exact per-instance rate; under repeated attempts per instance it's a proportional, round-comparable figure rather than an exact fraction-of-instances-affected — documented in the script's module docstring rather than silently presented as precise.

**Status classification** (`RATE_EPSILON = 1e-9` on the rate delta, to absorb floating-point noise without collapsing genuinely small real changes into "unchanged"):
- `resolved` — present (rate > 0) before, absent after
- `new` — absent before, present after
- `persisted_improved` / `persisted_worsened` / `persisted_unchanged` — present in both; rate decreased / increased / unchanged

`grounded_fraction` and `below_support_floor` are carried through as before/after columns, unfiltered — metadata on the row, never a gate on inclusion. `delta_pct` is null when `support_rate_before == 0` (percent change is undefined for a `new` pattern), not zero or clipped.

**Multi-bundle handling:** more than two bundle paths diffs every consecutive pair *and* a first-vs-last pair, so the two-point before/after comparison is always available regardless of how many intermediate bundles are passed. A bundle with zero patterns (e.g. an all-pass mining round) needs no special-casing — the outer join naturally makes every signature from the other side fully `resolved` or fully `new`.

**Naming:** `<out_dir>/pattern_frequency_diff_<label_a>_vs_<label_b>.csv` plus a same-named `.json` summary (status counts + `total_signatures`, also printed to stdout). Label precedence: `config.round_index` (the normal case) → explicit `--labels` entry → the bundle's parent directory name (`round_NN` in the standard layout) → positional fallback; a label collision across bundles gets a positional suffix so filenames never clobber each other.

**Verified** against `examples/mining_rounds/round_00/bundle.json` plus synthetic variants exercising every status value, the empty-patterns edge case, missing `config.round_index`, the `--labels` fallback, and identical-bundle `persisted_unchanged`.

### 2.4 Still open / not yet built

- **Whole-input sub-call collapse rate** and **root-vs-child failure attribution share**, as explicit computed metrics/series (per round for the optimization phase, per condition×split for evaluation) — the underlying fields (`collapse_ratio`, `failing_level`, sub-verifier labels) are present in `records.jsonl`, but no aggregation/plotting step has been built for either yet.
- **Mechanism-frequency diff plotting** — `pattern_frequency_diff.py` produces the CSV/JSON table only; no chart form has been requested or built for it yet (unlike Graphs A and B, which both have a plotting script).
- **Historical `promotions.jsonl` backfill** for `surface` — only needed if pre-fix rounds are to be included in Graph A; not yet done, and not required for rounds going forward.
- **Evaluation-phase graphs** (accuracy per condition × test set with CIs, token efficiency, recursion depth, accuracy-per-million-tokens) — not yet scoped or built in this conversation; the proposal doesn't mock up specific chart forms for these either, so form is still to be decided.