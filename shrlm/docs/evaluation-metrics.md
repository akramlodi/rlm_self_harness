# Optimization loop (shrlm/optimization/ + shrlm/experiment/orchestrator.py)
1. Mining (mining.py, driver.py) — per-run execution
Every run against the current harness writes a line to runs.jsonl:

run_id, instance_id, attempt, passed, cost, timestamp
verdict — cause, gold vs produced, detail (why it failed)
trace_path + trace_sha256 (content-addressed pointer to the full run trace in runs/)
2. Attribution (attribution.py, digest.py, taxonomy.py) — why each failure happened
Per failed run, records.jsonl captures a closed-vocabulary failure signature:

signature: verifier_cause, failing_level, agent_mechanism, causal_status
detail: free-text symptom summary, evidence node ids
stats: n_iterations, n_nodes, n_rlm_children, max_observed_depth, collapse_ratio, suspected_lost_subcalls, trace_integrity, block_attribution_reliable
digest_sha256 — the exact rendered trace excerpt the attributor model saw (content-addressed in digests/), so any label is re-derivable/auditable
level_grounded, attribution_failed, attribution_error_kind — confidence/failure of the attribution itself
3. Clustering / bundling (clustering.py, bundle.py) — pattern mining across runs
bundle.json aggregates records into patterns:

signature, support/instance_support, actionability score, grounded_fraction, below_support_floor
shared_symptoms, verifier_evidence, representatives (example instance ids)
integrity: known_substrate_biases (explicitly documented measurement blind spots, e.g. "batched sub-calls under-count syntax errors"), n_resource_terminated, n_transport_errors, mean_digest_coverage
config: full pinned config (model, prompt version, taxonomy version, seed, harness_hash) so the bundle is reproducible/attributable to an exact harness+prompt version
4. Proposal / Validation (proposal.py, validation.py, promotion.py, costs.py)

proposal.json per candidate + generated surfaces.py (the actual edited harness code)
decision.json — the promotion verdict: baseline vs candidate links, promoted (bool), promoted_harness_hash, constituent_ids, excluded candidates and why
promotions.jsonl — append-only ledger of every promotion decision ever made
Spend governance: CandidateSpendBreaker tracks cumulative $ per candidate against candidate_budget, marking runs over_budget/RESOURCE_TERMINATED (a lower bound, not silently dropped)
5. Cross-cutting: stage_usage.jsonl (R5, via usage.py)
One record per stage attempt, every stage (mining/attribution/proposal/validation/eval):
stage, round_index, attempt_index, cost, input_tokens, output_tokens, cache_hits, wall_seconds, resumed, lower_bound flag (flagged whenever usage might be undercounted, e.g. a crashed attempt or terminated run).

6. audit.py doesn't track new data — it verifies the whole chain above is self-consistent (hashes match, no orphaned references), so results can be trusted after the fact.

Evaluation loop (shrlm/experiment/evaluation.py)
Runs each condition (b1 = baseline H0, sh_rlm = frozen optimized harness) against each test set (graphwalks_short/long, oolong_pairs_short/long — the short/long split you asked about), reusing the same runs.jsonl/stage_usage.jsonl machinery as mining. Aggregated into eval/eval_summary.json per condition × test set:

pass_count, pass_rate, n_runs, n_instances, n_resource_terminated
total_cost, mean_cost, input_tokens, output_tokens, wall_seconds
mean_sub_calls/total_sub_calls (recursion usage)
instances_sha256 + split_file (proof it ran on the exact frozen split, byte-identical across conditions — R8)
harness_hash (proof it ran the exact frozen harness, re-verified against the freeze-time hash)
usage_lower_bound flag, skipped_run_ids if the spend breaker tripped
Plus report.py/scenarios.py roll measured usage into report.json — extrapolated token/cost/time projections (pessimistic/point scenarios) per pricing tier or GPU profile, used to answer "was optimization worth it and what would scaling this cost."

The throughline: everything is hash-addressed (harness_hash, trace_sha256, digest_sha256, instances_sha256) so any later number in a report can be walked back to the exact harness, trace, and prompt that produced it — that's what audit.py mechanically checks.

Analysis layer (shrlm/experiment/rounds.py, analysis_io.py, the four aggregation CLIs)
Nothing above is an analysis; the loops only persist evidence. Everything downstream — the four aggregations, the two plot scripts, and report.py — reads that evidence through two shared modules, and that indirection is the point rather than a tidiness preference:

rounds.py is the one inventory of what an experiment directory holds. discover_rounds(out_dir) returns per-round records (round index, the loop's own mining/validation/proposals paths, marker and ledger presence) plus the evaluation test-set inventory (condition × test set, outcome, skipped run ids). Path construction goes exclusively through the loop's own layout functions and constants, and round_* directories are enumerated in exactly one function, so an analysis can never be reading a tree the loop no longer writes. It also owns the completeness truth table (round_complete, evidence_complete, runs_complete, eval-set complete) and resolve_surface, which recovers a ledger row's surface from the round's persisted proposal.json when the ledger's own value is missing or null.

Completeness is three-valued, and unknown is never false. n_expected = len(instances.jsonl) × repetitions, with repetitions read per stage from the experiment config (loop.m mining, loop.v validation, operational.eval_repetitions evaluation). If the config cannot be resolved and re-verified against the directory's recorded identity_hash, n_expected is null and the row reports count_unknown — deliberately, with no fallback that infers the attempt count from observed run ids, because the manifest records only completed runs and a round truncated at attempt 2 of 3 would infer an expected count of 2 and declare itself complete.

analysis_io.py is the one output layer. Every batch of analyses writes into a fresh <out_dir>/analysis/<UTC-stamp>/ snapshot carrying provenance.json (identity_hash + identity_source, created_at, analysis_revision, analyzed rounds and eval sets, per-tool success/failure, and a sha256 manifest of every consumed artifact) and, written last, published.json. The source hashes are the load-bearing part of the reproducibility claim: experiment identity plus round number does not distinguish two snapshots built from a round that was re-mined or resumed, and the content hashes do.

The orchestrator refreshes those snapshots itself, best-effort, after every executed round (and once at catch-up when a resume replays finished rounds), so mid-study outputs never go stale. The hook writes only under analysis/, is never read back by the loop, and cannot fail, block, or alter a round.

See `SH-RLM Metrics & Graphs — Reference.md` §3 for the snapshot layout, the provenance field list, the completeness columns each output carries, `surface_source`, and how to reproduce a figure from a pinned snapshot.

