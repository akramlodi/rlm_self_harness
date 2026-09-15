# SH-RLM Metrics & Graphs — Reference

## 1. What the proposal specifies

### 1.1 Optimization phase (source environment, propose/validate/promote loop)

**Primary loop mechanics (section 3.3, section 4 cost model)**
- Per round: held-in traces mined, `K` candidates proposed, each targeting exactly one declared surface (S1–S10)
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
- **Figure 1**: the initial harness H₀ shown as a 10-surface grid (S1–S10), most surfaces empty/generic at the start — a static snapshot, not a time series
- **Figure 2**: one optimization round as a flow diagram (mining → proposal → validation), with a side grid showing S1–S10 where *tinted cells = surfaces edited in earlier rounds* and *the solid cell = the newly merged edit this round*. This is the closest thing the proposal has to a "harness evolution / levers touched" visual, though it's presented as a single-round schematic rather than an explicit time-series plot.
- No explicit "accuracy vs. round" line chart is specified in the text, but section 3.5's ask to report "frequency of each mined failure pattern before and after optimization" and section 5's Week 10 plan to "compute verifier accuracy... as results arrive" implies a quality-over-time analysis is expected, even though no figure is mocked up for it in the proposal itself.

### 1.2 Evaluation phase (frozen harness, four test sets, vs. baselines)

**Primary metric**
- Verifier accuracy per condition (B1, H₀\*, λ-RLM, SH-RLM, optionally F1) × per test set (source-short, source-long, target-short, target-long)
- Bootstrap confidence intervals over task instances
- Paired per-instance tests: SH-RLM vs. B1, H₀\*, and λ-RLM

**Secondary efficiency metrics** (same condition × test-set grid)
- Total input/output tokens
- Recursive-call count
- Maximum recursion depth
- Accuracy per million tokens

**Mechanism-level metrics, carried into evaluation**
- Whole-input sub-call collapse rate, per condition × split
- Root-vs-child failure attribution share, per condition × split

**Integrity / audit requirements**
- `method_kind` + `method_hash` identify every evaluated inference method; harness-backed methods still re-verify the underlying harness against its freeze-time hash
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
- `metrics.skill_load_count` — S10 loader invocations per run, summed from the per-block `skill_loads` events the REPL records beside its sub-call events; the run's `run_metadata.skill_index` names what was available. Aggregated as `total_skill_loads` / `mean_skill_loads` in each validation split summary, beside the sub-call counts — audit only; the promotion rule does not read them, so a never-loaded S10 promotion remains a possible (and preregistered-band-accepted) false positive

**Per-failure** (`attribution.py`, `digest.py`, `taxonomy.py` → `records.jsonl`)
- `signature`: verifier_cause, failing_level, agent_mechanism, causal_status
- `detail`: symptom summary, evidence node ids
- `stats`: n_iterations, n_nodes, n_rlm_children, max_observed_depth, collapse_ratio, suspected_lost_subcalls, trace_integrity, block_attribution_reliable
- `digest_sha256`, `level_grounded`, `attribution_failed`, `attribution_error_kind`
- the digest itself is versioned (`DIGEST_VERSION`, `1.2.0` since S10): its Run header carries an `available_skills:` / `loaded_skills:` pair whenever the run's harness installed a skill index, which is the observable the S10 mechanism `unconsulted_procedure` is defined against

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
| Target surface not present on decision-layer records (`promotions.jsonl`), only on `proposal.json` | **Fixed** | `CandidateDecision` gained a `surface: str \| None = None` field, serialized in `to_dict()`. `score_candidate`, `decide_subject`, and `assess_round` now thread the surface through from the proposal loader. Applies to all future rounds. Historical rows written before the fix lack the key entirely, and current-format rows whose candidate was rejected by the proposal loader serialize as `surface: null`; both are recovered at *analysis* time by `rounds.resolve_surface`, which joins `subject_id` against the round's persisted `proposal.json` and marks the result `surface_source = backfilled` (see §3.4). The ledger bytes themselves are never rewritten. |
| Losing-candidate Δin/Δho not distinguishable from a category label | **Confirmed already present, no code change needed** | `promotions.jsonl` rows already carry the full numeric `rule.heldin`/`rule.heldout` (baseline_pass_count, candidate_pass_count, delta, n_runs) for every scored candidate, not just the promoted one. Only exception: candidates rejected upstream of scoring (e.g. `over_budget`) have `rule: null`, since there's nothing to score against. |
| Mechanism-frequency (failure pattern signature) diff between two points in time (e.g. round 0 vs. freeze) | **Built** | `shrlm/experiment/pattern_frequency_diff.py` (see §2.3, Analysis — Mechanism-frequency diff). Confirmed as an analysis gap, not a capture gap: `bundle.json` already carried everything needed (`patterns[].signature`, `instance_support`, `totals.n_runs`, `config.round_index`); this is a downstream script over existing artifacts, no new instrumentation. |
| `surface = None` on merged-harness decision rows (a merge spans >1 surface, so a single value would misrepresent it) | **By design, not a gap** | Merge rows correctly stay `surface = None`; each constituent candidate retains its own individual surface on its own decision row instead. |

### 2.3 Graphs and analysis scripts implemented, and the decisions behind them

#### Graph A — Harness complexity / levers touched over time

**Aggregator:** `shrlm/experiment/surface_activity.py`, run via `python -m shrlm.experiment.surface_activity <out_dir> [--out DIR]`. Reads `promotions.jsonl` in round order for every ledgered round the shared discovery finds, and writes `surface_activity.csv` and `surface_activity_unattributed.csv` **into a freshly allocated analysis snapshot** — `<out_dir>/analysis/<UTC-stamp>/`, or `<DIR>/<UTC-stamp>/` when `--out` moves the snapshot parent. `--out` relocates where snapshots accumulate; it never turns the stamping off, and it never names the file directory directly. The CLI prints the two absolute paths it wrote. §3 covers the snapshot layout and why a rerun no longer overwrites anything.

**Decision — "attempted" vs. "actually changed" (levers touched):** Track **both**, as two separate counts per `(surface, round)`:
- `attempted_count` — increments for every row with a non-null `surface`, regardless of decision outcome. Represents where the proposer is spending search effort.
- `promoted_count` — increments only for rows that represent a real harness change: `decision == "promoted"` **or** (`decision == "accepted"` and the row is a merge constituent, i.e. `merge.role == "constituent"`). This second clause was a fix applied after integration testing surfaced that a merged promotion's constituent surfaces (e.g. S2 and S3 both winning and composing) were being missed — the ledger's `"promoted"` row is the synthetic merged subject, which has no single surface, while the actual constituent surfaces sit on rows marked `"accepted"`, not `"promoted"`. Verified post-fix: a merge round with S2+S3 constituents now correctly shows `cumulative_surfaces_promoted = 2` after that round (was 0 before the fix); `attempted_count` was confirmed unaffected by the same check, since it was already unconditional on decision outcome.

Cumulative series (`cumulative_surfaces_attempted`, `cumulative_surfaces_promoted`) are derived by counting distinct surfaces with a non-zero count up through each round.

**Row categories, and where `surface = None` now goes.** The table is dense over `(round, category)` for every discovered round: the ten canonical surfaces plus an eleventh `merged` category, zero-filled where nothing happened. The merged harness's own re-evaluation record is counted in `merged` rather than dropped — it spans several surfaces by construction, so a null surface is *correct* for it, not missing data, and folding it into "unattributed" would have reported a real outcome as a measurement failure. `merged` sits beside the ten, never among them: it never joins the distinct-surface cumulative counts and never appears in the S1–S10 heatmap. The surface rows come from the declaration (`CANONICAL_SURFACES`), so S10 is a row in every round; whether a given round's harness *declared* it is a per-cell fact — `surface_source = undeclared` for a round persisted under the nine-surface contract — read from that round's `harness.json`, never from the current code (R13, KTD6).

A row whose ledger `surface` is absent or null and is *not* the merged record is first backfilled from the round's persisted proposal (§3.4). Only what survives that — a loader rejection whose proposal artifact is no longer on disk — is genuinely unattributed, and those are tallied per round in `surface_activity_unattributed.csv` (`round_index, unattributed_count, total_rows`), with one row per ledgered round whether or not anything was excluded. The CLI prints a warning for any round over 25% unattributed.

**Columns the CSV carries** (`surface_activity.csv`): `round_index, surface, surface_source, attempted_count, promoted_count, cumulative_surfaces_attempted, cumulative_surfaces_promoted, round_complete, runs_complete`. `surface_source` is per *cell*, not per ledger row: a cell mixing recorded and recovered rows reports `backfilled` — the weaker of the two claims — because a reader comparing that count against a purely recorded one has to know part of it was reconstructed. `round_complete` / `runs_complete` are the KTD9 tristates (§3.3).

**Plotting:** `shrlm/experiment/plot_surface_activity.py`, run via `python -m shrlm.experiment.plot_surface_activity <out_dir> [--snapshot STAMP]`, reads the latest published analysis snapshot (or the pinned one) and writes `<out_dir>/analysis/<stamp>/plots/surface_activity.png` (static, matplotlib). `plots/` is the one re-renderable part of a published snapshot; the figure's footer carries the experiment identity, the snapshot stamp, and the render time.
- **Panel A**: dashed "attempted" vs. solid "promoted" step lines, deliberately the **same color/hue** rather than two distinct colors — since promoted is always a subset of attempted each round, same-hue communicates the containment relationship rather than implying two independent series. The "all declared surfaces" reference is stepped per round at the size of that round's own declared set (ten under the current contract, nine for a round persisted before S10; no segment where the round's harness could not be read), and the y-limit, ticks, and axis label derive from the largest declared count rather than from a literal — the figure never draws a ten-surface ceiling over a round that only ever had nine. End-of-line value labels.
- **Panel B**: two S1–S10 × round heatmaps (attempted, promoted) sharing a single color scale/colorbar so the two are directly comparable. Zero-activity cells render fully blank (masked to background), not a pale tint, so untouched surfaces are visually distinct from barely-touched ones — this was a bug caught and fixed during visual review (zero cells originally rendered as a visible pale blue). All ten surfaces are always shown as rows even if some have zero activity across every round, to keep the "8 of 10 surfaces start empty" framing visible; three zero states render differently — declared-and-untouched is the blank cell, a surface the round's harness never declared (a pre-S10 round's S10 cell) is a crosshatched grey cell, and a cell whose round's harness could not be read is a dotted-outline cell with a `?`, never folded into either of the other two.
- A footnote reports total unattributed rows and flags any round exceeding 25% unattributed.
- A second bug caught during review: heatmap x-axis originally showed fractional round ticks (1.0, 1.5, 2.0…) instead of integers; fixed.

#### Graph B — Harness quality while looping

**Aggregator:** `shrlm/experiment/incumbent_quality.py`, run via `python -m shrlm.experiment.incumbent_quality <out_dir> [--out DIR]`. Reads `promotions.jsonl` in round order and writes two CSVs into a freshly allocated analysis snapshot (`--out` moves the snapshot parent, exactly as for `surface_activity`):
- `incumbent_quality.csv` — one row per round: `round_index, heldin_pass_rate, heldout_pass_rate, incumbent_changed (bool), annotation (nullable), round_complete, runs_complete`. Incumbent state starts at H₀'s baseline values (from round 1's `rule.heldin/heldout.baseline_pass_count`) and updates only when a candidate is promoted that round; otherwise carries forward flat. The two trailing completeness columns are the KTD9 tristates from discovery (§3.3), appended after the metrics so a pre-change reader's column positions are unmoved.
- `incumbent_quality_candidates.csv` — one row per scored candidate per round (added during this work; not present in the original aggregator), using each candidate's own `candidate_pass_count` rather than the shared incumbent baseline. Needed for the optional per-round scatter overlay. Columns: `round_index, subject_id, decision, surface, surface_source, heldin_pass_rate, heldout_pass_rate`. Its `surface` is resolved through the **same `rounds.resolve_surface`** `surface_activity.csv` uses — ledger value verbatim when there is one, the round's persisted proposal when there is not, the `merged` category for the merged harness's own record — so the two tables in one snapshot never place the same candidate on different surfaces. `surface_source` (`ledger` | `backfilled` | `merged` | `none`) says which of those spoke; an empty `surface` with `surface_source=none` means nothing on disk could place the row.

**Decision — pass rate vs. raw pass count:** Use **pass rate** (normalized by n_runs), not raw pass count, since held-in and held-out splits have different sizes (e.g. n_in=24 vs. n_ho=40 in the proposal's example numbers) and raw counts aren't directly comparable across the two lines.

**Decision — handling of `rule: null` rows** (candidates rejected upstream of scoring, e.g. `over_budget`, no delta to speak of): **skip with optional annotation** — the incumbent line stays flat through that round (correct, since the incumbent genuinely didn't change), and a separate annotation record is emitted with the rejection reason, for optional plotting as a marker rather than being silently dropped.

**Plotting:** `shrlm/experiment/plot_incumbent_quality.py`, run via `python -m shrlm.experiment.plot_incumbent_quality <out_dir> [--snapshot STAMP] [--show-candidates]`, reads the latest published analysis snapshot (or the pinned one) and writes `<out_dir>/analysis/<stamp>/plots/incumbent_quality.png` (static, matplotlib), with the same identity / snapshot / render-time footer.
- Held-in (blue/circle) vs. held-out (orange/square) step lines.
- Black ring markers at rounds where `incumbent_changed == True`.
- Small triangle ticks below the axis at annotated (`rule: null`) rounds, keyed to a numbered footnote list with the rejection reason text (static output, so no hover tooltip — footnote list instead).
- `--show-candidates` flag (off by default) overlays every scored candidate's own pass rate as faint background points, from `incumbent_quality_candidates.csv`.

#### Analysis — Mechanism-frequency diff (no graph yet)

**Script:** `shrlm/experiment/pattern_frequency_diff.py`, run via `python -m shrlm.experiment.pattern_frequency_diff <out_dir> <bundle_1> <bundle_2> [<bundle_3> ...] [--out DIR] [--labels <label_1> <label_2> ...]`. Direct code-level counterpart to the proposal's "report the frequency of each mined failure pattern before and after optimization" (section 3.5) — a downstream script over already-captured `bundle.json` artifacts, not a new instrumentation point.

> **Breaking CLI change.** The experiment directory is now a **required first positional**, and `--out` no longer names the directory the CSVs land in — it overrides the *snapshot parent*, defaulting to `<out_dir>/analysis`. An old invocation (`... <bundle_1> <bundle_2> --out results/`) will now read `<bundle_1>` as the experiment directory and fail or mis-anchor. The argument was made required rather than optional because the bundles carry every number this analysis reads but none of them answer "which experiment is this about": `out_dir` is what anchors the snapshot location and the `identity_hash` lookup, and a diff table with no answer to that question is the exact failure mode the snapshot layer exists to close. The bundle-path arguments themselves are unchanged.

**Completeness table, always written first:** `pattern_frequency_diff_bundles.csv` — one row per bundle handed to the script, whether or not it was compared: `label, bundle_path, round_index, round_complete, evidence_complete, taxonomy_version, included, exclusion_reason`. A bundle is matched to a discovered round by the path the loop itself would have written it to (`<mining round>/bundle.json`), **not** by the `config.round_index` the bundle carries — a bundle copied out of the tree, or one from a different experiment, records a round index just as convincingly as an in-tree one, and trusting it would attach another experiment's completeness to it. A bundle whose round is *known* not evidence-complete is excluded from every comparison, with the reason spelled out in the row (a frequency comparison against a round with missing evidence reads the missing evidence as change); a bundle whose round could not be matched at all is reported `unknown` and still included. Nothing is ever silently dropped — the completeness table is written even when fewer than two bundles remain comparable and no pair is diffed at all, because an otherwise-empty snapshot would say only that the analysis ran.

**Taxonomy-version boundary.** `TAXONOMY_VERSION` was bumped from `2.0.0` to `3.0.0` when S10 was declared, because the version *is* the surface contract and the mechanism vocabulary is this join's key. A bundle whose `config.taxonomy_version` differs from the running code's is excluded, with the exclusion reason naming both versions, and the JSON summary reports every version seen with its bundle count (`taxonomy_versions_seen`); diffing across the boundary is an explicit opt-in (`--taxonomy-version`), never an accident. An unversioned bundle (a hand-built fixture — mining always stamps the version) is unknown, not different, and is compared. The consequence is deliberate: a diff spanning the S10 change reports nothing rather than something misleading, so the committed `examples/mining_rounds/round_00/bundle.json` (stamped `2.0.0`) does not diff against a post-S10 bundle by default.

**Join key:** full outer join of two bundles' `patterns[]` on the four-tuple failure signature (`verifier_cause`, `failing_level`, `causal_status`, `agent_mechanism` — `FailureSignature.key()`). Every signature present in either bundle gets one row.

**Decision — the rate denominator, stated honestly rather than assumed exact:** `bundle.json` has no distinct-instance count to normalize `patterns[].instance_support` against — `MiningTotals.n_runs` is a *run* count (`len(runs)` mined that round), not an instance count. `support_rate = instance_support / totals.n_runs`. When mining used one attempt per instance (the common case) the two coincide and this is an exact per-instance rate; under repeated attempts per instance it's a proportional, round-comparable figure rather than an exact fraction-of-instances-affected — documented in the script's module docstring rather than silently presented as precise.

**Status classification** (`RATE_EPSILON = 1e-9` on the rate delta, to absorb floating-point noise without collapsing genuinely small real changes into "unchanged"):
- `resolved` — present (rate > 0) before, absent after
- `new` — absent before, present after
- `persisted_improved` / `persisted_worsened` / `persisted_unchanged` — present in both; rate decreased / increased / unchanged

`grounded_fraction` and `below_support_floor` are carried through as before/after columns, unfiltered — metadata on the row, never a gate on inclusion. `delta_pct` is null when `support_rate_before == 0` (percent change is undefined for a `new` pattern), not zero or clipped.

**Multi-bundle handling:** more than two bundle paths diffs every consecutive pair *and* a first-vs-last pair, so the two-point before/after comparison is always available regardless of how many intermediate bundles are passed. A bundle with zero patterns (e.g. an all-pass mining round) needs no special-casing — the outer join naturally makes every signature from the other side fully `resolved` or fully `new`.

**Naming:** `<snapshot>/pattern_frequency_diff_<label_a>_vs_<label_b>.csv` plus a same-named `.json` summary (status counts + `total_signatures`, also printed to stdout; the JSON additionally carries the snapshot's `identity_hash` and `created_at`, so a summary lifted out of its directory still says which experiment and when). Snapshot stamping is what makes these label-derived filenames safe: two runs over re-mined bundles carrying the same round indices would otherwise land on identical names. Label precedence: `config.round_index` (the normal case) → explicit `--labels` entry → the bundle's parent directory name (`round_NN` in the standard layout) → positional fallback; a label collision across bundles gets a positional suffix so filenames never clobber each other.

**Verified** against `examples/mining_rounds/round_00/bundle.json` plus synthetic variants exercising every status value, the empty-patterns edge case, missing `config.round_index`, the `--labels` fallback, and identical-bundle `persisted_unchanged`.

#### Analysis — Collapse rate & root/child attribution, both phases (no graph yet)

**Script:** `shrlm/experiment/collapse_and_attribution.py`, run via `python -m shrlm.experiment.collapse_and_attribution <out_dir> --phase optimization|evaluation [--out DIR]`. `--phase` is required. `--out` kept its spelling but changed meaning with the snapshot layer: it is now the snapshot *parent* (default `<out_dir>/analysis`), not the directory the CSV is written into — the CSV always lands inside a timestamped snapshot below it. Direct code-level counterpart to proposal sections 3.5 (optimization) and 1.2 (evaluation's "same metrics, carried into evaluation").

**Investigation, before building anything:**
1. *Is `collapse_ratio` available for passing runs?* No, but only by mining-stage convention, not a structural limit — `mining.py`'s `record_failure` returns before walking a passing run's trace (`if verdict.passed: return`), but `walker.walk()` itself needs only trajectory metadata, persisted identically for every run (pass or fail) at `runs/<run_id>.json`. So it's re-derivable from existing trace files, not newly instrumented.
2. *Does `evaluation.py` produce comparable per-run attribution?* No — confirmed gap, `eval_summary.json` is pure `split_aggregate` pass/fail-and-cost output; `evaluation.py` never calls `mining`/`walker`/`attribution`/`grounding`. But eval traces persist through the identical `driver.run_round` path mining uses, and critically, `failing_level` is **fully deterministic whenever a `SubVerifier` exists** (`grounding.py`'s `apply_sub_verifier` needs no LLM call — only `agent_mechanism`/`causal_status`, not computed here, need the attributor). So eval-phase root/child attribution is achievable without any new LLM spend.
3. *Environment coverage:* only GraphWalks has a registered `SubVerifier`; OOLONG-Pairs deliberately has none (its own module docstring: "child correctness here is a semantic-labeling judgment no deterministic check can recompute"). Optimization mining is GraphWalks-only by construction, so this only bites the evaluation phase's OOLONG-Pairs test sets.

**Decision (user-confirmed, both recommended):** re-walk every run's persisted trace directly via `shrlm.optimization.driver.load_round` (the same rehydration `split_aggregate` already uses) for **both** phases, rather than reading `records.jsonl`'s failures-only figures for optimization. This makes `collapse_rate` a true outcome-independent rate (matching the proposal's own framing — not conditioned on failure) and makes optimization and evaluation numbers structurally comparable, using the identical mechanism. It was cross-validated against `records.jsonl`'s own already-computed `failing_level` for the same real mining round (`examples/mining_rounds/round_00`) and matched exactly.

**Denominators, deliberately different per metric:** `collapse_rate` is over every walkable run, pass or fail (a harness-behavior metric). `root_failure_share` / `child_failure_share` / `ungrounded_share` are over **failures only** (a passing run has no failure to locate). A run whose trace can't be walked at all (no trajectory metadata) is excluded from every numerator and denominator and counted separately in `n_unwalkable`, never silently folded into a rate.

**One addition beyond the four named columns:** `FailingLevel` has a fourth member, `NO_RECURSION` (grounded, but a failure with no sub-calls to attribute to either level) — reporting only root/child/ungrounded would silently drop it from view, so a `no_recursion_failure_share` column was added to keep the four shares exhaustive over `n_failures`. Also carries `sub_verifier_available` (bool) per group, so `ungrounded_share = 1.0` for a structural reason (no `SubVerifier` for that environment) reads differently from `ungrounded_share = 1.0` from individually-ungrounded failures.

**Output** (into the allocated snapshot): `collapse_and_attribution_optimization.csv` — one row per round: `round_index, environment, n_runs, n_walked, n_unwalkable, n_failures, collapse_rate, root_failure_share, child_failure_share, no_recursion_failure_share, ungrounded_share, sub_verifier_available`, then the round completeness columns `n_expected_runs, n_missing_runs, runs_complete, evidence_complete, round_complete`. Or `collapse_and_attribution_evaluation.csv` — the same metrics keyed by `condition_id, test_set_id, environment, length` instead of `round_index`, followed by the eval-set completeness columns `outcome, n_skipped, n_expected_runs, n_missing_runs, runs_complete, complete`.

Completeness is appended *after* the metric columns deliberately, so the metric columns keep the positions a pre-change reader (or script) knows. `n_expected_runs` / `n_missing_runs` stay numeric and write an **empty cell** when the expected count is unknowable; the `runs_complete` flag beside them is what says which kind of blank it is (`unknown`, not `false`). See §3.3.

**Verified** against a real mining round (`examples/mining_rounds/round_00`, re-walked and cross-checked against its own `records.jsonl`) and a real live-mocked `run_evaluation` call (`tests/experiment/test_evaluation.py`'s harness) producing a genuine `eval/` tree and `eval_summary.json`, plus the fully-skipped-test-set and no-`SubVerifier`-environment edge cases.

### 2.4 Still open / not yet built

- **Collapse rate / attribution plotting** — `collapse_and_attribution.py` produces the CSV table only; no chart form requested or built yet.
- **Mechanism-frequency diff plotting** — `pattern_frequency_diff.py` produces the CSV/JSON table only; no chart form has been requested or built for it yet (unlike Graphs A and B, which both have a plotting script).
- ~~**Historical `promotions.jsonl` backfill** for `surface`~~ — **done, at analysis time rather than on disk.** `rounds.resolve_surface` recovers the surface from the round's persisted `proposal.json` and reports it as `surface_source = backfilled` (§3.4). The ledger bytes are deliberately never rewritten: the experiment's persisted evidence is not something an analysis gets to edit.
- **Snapshot retention** — analysis snapshots accumulate one per executed round for the life of an experiment, with no pruning policy (§3.7).
- **Evaluation-phase graphs** (accuracy per condition × test set with CIs, token efficiency, recursion depth, accuracy-per-million-tokens) — not yet scoped or built in this conversation; the proposal doesn't mock up specific chart forms for these either, so form is still to be decided.

---

## 3. Analysis snapshots, provenance, and completeness

Everything in §2.3 used to write a fixed-name CSV straight into the experiment directory. Two consequences, both fatal to the reproducibility claim the study rests on: a rerun silently destroyed the previous run's numbers, and a CSV sitting on disk carried nothing saying which experiment it came from, when it was produced, what code produced it, or which bytes it was derived from. A reader holding `surface_activity.csv` could not tell a figure built from a finished round from one built from the same round re-mined after a crash. The three modules below — `shrlm/experiment/rounds.py` (what exists and how complete it is), `shrlm/experiment/analysis_io.py` (where output goes and what it is stamped with), and `shrlm/experiment/plot_style.py` (how a figure reports its own provenance) — exist to close that.

`report.py` reads the same inventory, so no third view of the on-disk layout survives anywhere in the package.

### 3.1 Snapshot layout

```
<out_dir>/analysis/20260818T164800Z/
    surface_activity.csv
    surface_activity_unattributed.csv
    incumbent_quality.csv
    incumbent_quality_candidates.csv
    collapse_and_attribution_optimization.csv
    pattern_frequency_diff_bundles.csv
    pattern_frequency_diff_<a>_vs_<b>.csv / .json
    provenance.json      # identity, revision, tools, source hashes
    published.json       # written LAST; absent means "do not trust this"
    plots/               # deliberately re-renderable
```

**One snapshot per invocation *batch*, not per tool.** The caller — a CLI run, or the orchestrator's post-round hook — allocates one stamped directory and passes it to every aggregation it runs, so the CSVs a reader compares against each other are guaranteed to have come from a single pass over a single tree. A tool that allocated its own would leave four directories per invocation, each a different instant, with nothing saying they belonged together. (This is also the source of the gotcha in §3.7.)

**Allocation is a race, and is treated as one.** The stamped directory is created with `mkdir(exist_ok=False)` — the exclusive create *is* the lock. A stamp already taken (two invocations inside the same UTC second, which the post-round hook makes entirely reachable) retries with a monotonic suffix `-2`, `-3`, … up to 1000 before giving up. Checking `exists()` first and then creating would leave exactly the window in which two batches decide the same directory is free and then write over each other. Because a collision suffix sorts after the bare stamp it collided with, directory-name order stays chronological order.

**Published means finished.** `published.json` is written last, after every tool in the batch has finished *and* `provenance.json` is on disk. "Latest" resolution — what a plot CLI does by default — only ever returns a **published** directory, so a crashed or partially-failed batch can never be picked up as "the latest results". It stays on disk, inspectable, but never selected. A batch with a failed tool still writes its provenance (that record *is* the audit trail for the failure) and simply withholds the marker.

**Never-overwrite, and the one deliberate exception.** The shared writers refuse to replace an existing file with different bytes; a byte-identical rewrite is a no-op, a diverging one raises `AnalysisOutputError`. `plots/` is exempt by construction — it is written through neither writer — because a frozen snapshot's figures must stay re-renderable by improved plotting code without the frozen data underneath them ever moving. The figure's footer carries its own render time so a re-rendered PNG never misrepresents when it was drawn.

### 3.2 `provenance.json`

| Field | Meaning |
|---|---|
| `format` | `shrlm-analysis-provenance/v1` |
| `identity_hash` | The experiment identity (see below) |
| `identity_source` | `config` (read verbatim from `<out_dir>/config.json`) or `derived` |
| `legacy_derived` | `true` exactly when `identity_source == "derived"` |
| `created_at` | ISO-8601 UTC instant the batch was allocated |
| `analysis_revision` | The repository commit the analysis ran from, or `null` |
| `rounds` | Sorted optimization round indices this batch actually analyzed |
| `eval_sets` | Sorted `<condition_id>/<set_id>` pairs this batch analyzed |
| `tools` | One entry per tool in the batch: `{name, ok, error}` |
| `sources` | One entry per consumed artifact: `{path, kind, sha256}` (+ `n_files` for a directory) |

**`sources` is the load-bearing part of the reproducibility claim.** Experiment identity plus round number does not distinguish two snapshots built from a round that was re-mined or resumed; the content hashes of the consumed artifacts do. Paths are recorded relative to the experiment directory first (the ordinary case, and what makes two provenance records of the same tree comparable), relative to the repository next (a bundle passed in from a checked-in fixture), absolute only when the artifact lives outside both — an absolute path would be machine-specific and would make the same tree hash differently on two machines. An artifact that was *not* on disk is recorded with a `null` hash rather than omitted: "this analysis looked for it and it was not there" is part of how the numbers came out, and dropping the entry would make the absence invisible. A directory of per-run traces is recorded as **one** entry whose hash covers every file inside it, so provenance detects any byte that moved without growing to thousands of lines.

**`identity_source`.** `identity_hash` is taken *verbatim* from `<out_dir>/config.json` rather than re-derived from the TOML: the question provenance answers is "which experiment produced this", and that is what the experiment directory itself claims — a claim that stays true even if the profile in `configs/experiment.toml` has since drifted. A pre-orchestrator tree has no such file; rather than failing the analysis (an old tree is exactly the thing worth analyzing), the identity is derived deterministically from the source hashes and flagged `identity_source: derived` / `legacy_derived: true`, so a reader can always tell a real experiment identity from a stand-in. Two analyses of an unchanged legacy tree agree on what to call it; an analysis of a changed one does not.

**`analysis_revision`** is best-effort `git rev-parse HEAD`. No git, no repository, or a git failure records `null` — an unknown revision is worth recording as unknown, and is never worth failing an analysis over.

`published.json` carries `format: shrlm-analysis-snapshot/v1`, `identity_hash`, `created_at`, `published_at`, and `n_tools`. JSON aggregation outputs (the frequency-diff summaries) are additionally stamped with the snapshot's `identity_hash` and `created_at`, extending `eval_summary.json`'s existing precedent rather than inventing a second convention.

### 3.3 Completeness columns

One authoritative truth table, defined in `rounds.py` and consumed by every analysis, so no module re-decides what "complete" means:

| Field | `true` | `false` | `unknown` |
|---|---|---|---|
| `round_complete` | `round.json` present, well-formed, and recording this round's index | round directory discovered without a usable `round.json` | never — marker presence is always knowable |
| `evidence_complete` | `evidence_complete.json` present, recording *this* round and the `bundle.json` beside it, *and* its recorded counts match the **parsed** `records.jsonl` / `attributions.jsonl` | marker absent, naming another round or another bundle, counts disagreeing, or an audit file that does not parse | never |
| `runs_complete` | `n_expected` known and `n_present >= n_expected` | `n_expected` known and `n_present < n_expected` | `n_expected` unresolvable |
| eval set `complete` | outcome is `completed`, no skipped runs recorded, **and** `runs_complete` is true | outcome is anything else, any skipped runs recorded, or `runs_complete` is false | the set is absent from `eval_summary.json`, or its `runs_complete` is unknown |

**`unknown` is always its own value and never collapses into false** — an unmeasurable round must never be presented as a failed one. That is why these cells are written as the literal strings `true` / `false` / `unknown` rather than left to Python's rendering: `None` writes as an *empty* cell, and an empty cell in a completeness column is indistinguishable from a column the writer forgot.

The `evidence_complete` re-check is not belt-and-braces: `write_bundle` publishes `bundle.json` *before* `records.jsonl` and `attributions.jsonl`, so a marker written over a truncated audit file is a real crash window. Discovery re-runs the same check the orchestrator makes before writing the marker — same identity (the round index and the bundle id the marker records), same parse (`read_jsonl`, not a line count, so a torn final line cannot pass as an intact record) — because an analysis reads trees the current process did not write, including trees where a marker was copied in from elsewhere.

The eval-set verdict folds the run counts in for the same reason: `eval_summary.json` is a **self-report** written from the state the evaluation held in memory, so a set whose runs fall short on disk can still record `outcome: completed` with no skipped runs. `complete` is therefore the three-valued AND of the summary's verdict and `runs_complete` — either being false makes the set incomplete, an unknown planned count leaves it unknown — and `outcome`, `n_skipped`, `n_expected_runs`, `n_missing_runs`, and `runs_complete` all travel beside it so a reader can see which half moved.

**Expected run counts, and why nothing is inferred from the runs.** `n_expected = len(instances.jsonl) × repetitions`, with `repetitions` read from the experiment config **per stage** — `loop.m` for mining, `loop.v` for validation, `operational.eval_repetitions` for evaluation — because the config assigns those separately and one fixed multiplier would misreport whichever stage it does not match. Resolving the config means reading the profile name from `<out_dir>/config.json`, re-loading that profile from `configs/experiment.toml`, and checking the reload still hashes to the identity the directory recorded; a drifted TOML therefore yields "unknown" rather than counts from a configuration this experiment never ran under. There is deliberately **no** fallback that infers the attempt count from observed run ids: the manifest records only completed runs, so a round truncated after attempt 2 of 3 would infer an expected count of 2 and report itself complete — manufacturing exactly the false completeness this layer exists to remove.

A no-op round (round marker present, no ledger, because the round's proposal stage produced no candidate) is `round_complete = true` with `has_ledger = false`. It is a real outcome, not missing data, and it now appears in the round accounting instead of vanishing from it.

### 3.4 `surface_source`

Every output that tallies surfaces reports where each attribution came from, because a count of S3 touches means something different when the S3 came out of a proposal artifact than when the ledger recorded it. Six values:

| Value | Meaning |
|---|---|
| `ledger` | The `promotions.jsonl` row recorded a non-null `surface`; taken verbatim |
| `backfilled` | The ledger value was missing or null; recovered by joining `subject_id` to the round's persisted `proposal.json` |
| `merged` | The merged harness's own record (`subject_id == "merged"`) — its own category, never one of the ten surfaces |
| `none` | Nothing on disk could place the row: no ledger surface, and no proposal artifact left to recover one from |
| `undeclared` | The round's own `harness.json` was read and does not declare this surface (a round persisted before S10 existed declares nine; its S10 cell is this) — an absent surface, not a missing attribution, so never tallied unattributed |
| `unknown` | The round's `harness.json` is absent or unreadable, so nothing can be said about what it declared; the cell's count is still reported, and unknown never collapses into `undeclared` or `none` |

Resolution order is fixed: `merged` short-circuits first (backfilling it would invent a single-surface claim the experiment never made), then the round's declared set decides whether the cell exists at all (`unknown` when the harness could not be read, `undeclared` when it was read and lacks the surface), then a non-null ledger value wins (the ledger is the evidence; a proposal never overrides it), then the proposal join fires.

The backfill triggers on a `surface` key that is **missing OR present-and-null**, and the second spelling is the load-bearing one: `CandidateDecision.to_dict` always emits the key, so every loader rejection the current loop writes serializes as `surface: null`. A backfill keyed on key-absence alone would only ever have fired for pre-`surface` ledgers and would have silently missed every current-format rejection.

### 3.5 The post-round auto-run

The orchestrator refreshes the analysis snapshots itself, so mid-study outputs never go stale:

- **When.** Once after every **executed** round. During a resume, replayed rounds do *not* each fire a refresh — the replay phase refreshes **once, when it catches up** (at the first executed round, or at loop exit if the resume executed nothing at all). Firing per replayed round would run one full pass over the tree for each already-finished round before any new work happened, and only the last of those passes would carry current information.
- **What runs.** `collapse_and_attribution_optimization`, `incumbent_quality`, `surface_activity` always; `pattern_frequency_diff` once at least two rounds have persisted a `bundle.json`; `collapse_and_attribution_evaluation` only once `<out_dir>/eval/eval_summary.json` exists. Both conditionals are **silent** when they do not fire — the evaluation summary is written by the evaluation runner and is absent for the whole optimization run, and a first round has nothing to diff against; neither is a fault worth a warning. Aggregations only: the plots need the optional `plotting` extra.
- **Isolation, in two layers.** Each aggregation runs under its own guard — a failure is recorded against that tool's name in provenance and the batch carries on, so one broken aggregation does not cost the others their output. The whole hook then sits under a second blanket guard: a broken aggregation, an unreadable artifact, an unallocatable snapshot, even an `ImportError` raised while loading the analysis modules ends in a warning on stderr and a return. The experiment's outcome must never depend on whether its analyses ran.
- **What it touches.** Only `analysis/`. No new persisted experiment state, and nothing in the loop ever reads `analysis/` back.
- **What a failure looks like.** The batch publishes unconditionally in the sense that `provenance.json` is always written; the completion marker is withheld when a tool failed, and stderr names the failed tools and the unpublished directory. Nothing checks that record at end of run — a systematic failure (a missing optional dependency, say) would produce a study with a warning on every round and no usable snapshots, with the provenance records as the only audit trail.

### 3.6 Rendering a figure from a snapshot

```
python -m shrlm.experiment.plot_surface_activity  <out_dir> [--snapshot STAMP_OR_PATH]
python -m shrlm.experiment.plot_incumbent_quality <out_dir> [--snapshot STAMP_OR_PATH] [--show-candidates]
```

Both take the **experiment directory**, resolve the **latest published** snapshot by default, and write into that snapshot's `plots/`. `--snapshot` pins one and accepts either a stamp name inside `analysis/` or a path to a snapshot directory anywhere — both are things a person legitimately has in hand (the name an aggregation printed, or the path they just `ls`-ed). A missing companion CSV is a one-line stderr message and a non-zero exit, not a traceback: it is a user-facing mistake ("run the aggregation that writes it, or pin a snapshot that has it"), not a bug.

**Partial rounds are marked on the figure, not only in the CSV** — the PNG is the artifact a reader draws conclusions from, so completeness that stopped at the CSV would leave the objective unmet at the surface that matters. A round whose `round_complete` or `runs_complete` is anything other than the literal `true` renders with a hollow/distinct marker, gets a `*` on its x-tick label, and is named in a caption reporting how many of the plotted rounds are not confirmed complete and why (missing round marker, or a short-or-unknown run count). Anything that is not `true` — `false`, `unknown`, or a column an older CSV never wrote — counts as not-confirmed-complete: an unmarked partial round would be read as a finished one, while a conservatively marked complete round only costs a footnote.

**Every figure carries a provenance footer**: `experiment <identity, first 12 chars> | snapshot <stamp> | rendered <UTC stamp>`, plus `| UNPUBLISHED SNAPSHOT` when the pinned directory has no completion marker. All three fields, because each answers a question the other two cannot — which experiment, which frozen numbers, and (since `plots/` is re-renderable over an immutable snapshot) which plotting run drew this particular PNG.

Graph A's unattributed footnote counts **rounds that actually excluded something**, not every round in the table: the table holds one entry per round whether or not anything was excluded, and counting its length overstated the reach of the exclusions (three rounds "affected" when one row in one round was dropped).

### 3.7 Operating notes

**Rendering both graphs by hand: pin a snapshot, or run the aggregations as one batch.** Because a snapshot is per *invocation batch*, running `surface_activity` and then `incumbent_quality` as two separate commands produces **two** snapshots — each holding only its own CSVs. Each plot CLI then resolves "latest published" independently, and the one that resolves to the other tool's snapshot will not find its companion table (it fails cleanly, per §3.6, rather than plotting anything wrong). The default path just works for a batch that ran both aggregations — which is exactly what the post-round hook does — so this only bites a person driving the CLIs by hand. Either pin the same `--snapshot` for both figures, or plot from a hook-produced snapshot. Note also that a snapshot written under a relocated `--out` parent is invisible to the plot CLIs' default "latest" resolution (they look under `<out_dir>/analysis`); pass its path to `--snapshot`.

**Finish in-flight experiments before upgrading.** A round that was ledgered but not yet marked, resumed *after* the ledger `surface` field landed, can trip validation's non-clobber ledger check. This is not caused by the analysis layer — it is a consequence of the ledger record shape changing mid-experiment — and the mitigation (a ledger format bump) is deferred. Let a running experiment complete, then upgrade.

**Snapshots accumulate; nothing prunes them.** One snapshot per executed round, for the life of an experiment, with no retention policy yet. Two practical consequences: `report.py`'s disk-footprint number walks the whole experiment directory and therefore **includes** the accumulated `analysis/` snapshots (only `report.json` itself is excluded), so the measured-bytes and projected-bytes figures are inflated by analysis output relative to a pure evidence footprint; and provenance records each snapshot's tool list and outcomes but *not* what triggered it, so a snapshot published deliberately by hand is not distinguishable from the machine-generated series around it. Note the cost shape too: the aggregations re-read all prior rounds on every invocation and the source manifest adds one read per consumed artifact, so the auto-run is O(rounds²) file reads over an experiment — fine at tens of rounds, worth revisiting past roughly a hundred.
