# **Experimental Setup**

This section describes the two-stage experimental procedure used to produce and evaluate a self-optimized recursive language model harness (SH-RLM): (1) a **Self-Harness optimization stage**, in which a frozen-weight RLM iteratively edits its own harness against held-in and held-out data drawn from a single *source* environment; and (2) a **fixed-weight evaluation stage**, in which the frozen final harness is compared against fixed baselines across both the source environment and a held-out *target* environment, at both in-distribution and substantially longer input lengths. All hyperparameters marked `[placeholder]` are set via a pilot calibration run (Section 3.9) prior to the main experiment and are reported with their final values in the results section.

## **3.0 Backbone and harness invariants**

All conditions (B1, H1, SH-RLM) share a single frozen backbone model, `Qwen3-30B-A3B-Instruct-2507`, with an identical decoding configuration (`[temperature, placeholder]`, `[top-p, placeholder]`, `[max output tokens, placeholder]`) used for both root and recursive sub-calls. No condition updates model weights; only the harness code surrounding the model differs between conditions.

The RLM architecture is defined by three invariant properties that no harness edit is permitted to alter:

1. The input prompt is held as a variable in a code-execution (REPL) environment rather than passed directly in context.  
2. The root model issues sub-calls to fresh copies of itself programmatically, over slices or transformations of that variable.  
3. Sub-call outputs are accumulated into variables and composed programmatically into the final answer.

Nine harness surfaces remain editable throughout optimization, each corresponding to a declared position in the RLM turn loop:

| Surface | Governs |
| ----- | ----- |
| S1 — REPL contract | The factual contract given to the root model: available names, the answer protocol, per-turn REPL-block and stdout conventions, truncation behavior. |
| S2 — Decomposition instruction | Turns 1–2: how the root probes the input and whether/how it plans a decomposition. |
| S3 — Execution instruction | Per-turn discipline: what to print, when to offload to a sub-call vs. read directly, how to aggregate results. |
| S4 — Verification instruction | What the root checks before marking an answer ready. |
| S5 — Recovery instruction | What the root does when a sub-call errors or returns something unusable. |
| S6 — Runtime policy | Numeric limits and switches: chars per prompt, batch width, max calls per turn/total, recursion depth, retry-on-error, sub-output validation. |
| S7 — Metadata function | What carries across turns — the harness's memory of prior calls. |
| S8 — REPL helpers | Harness-local functions and data injected into the REPL namespace (chunkers, batch wrappers, etc.). |
| S9 — Answer middleware | Programmatic inspection of the detected final answer, with the ability to redirect (suppress and continue) rather than accept it. |

At initialization (harness B1) all nine surfaces are set to minimal/unmodified defaults; a candidate edit targets exactly one surface per proposal.

## **3.1 Environments and data splits**

Two long-context environments are used: **GraphWalks** and **OOLONG-Pairs**. **GraphWalks is the source environment**(used for optimization) and **OOLONG-Pairs is the target environment** (held out entirely from optimization, used only for evaluation).

Each environment provides a *short* split (comparable to typical single-context lengths) and a *long* split, constructed to be approximately `8–32×` longer than the short split (`[construction method, placeholder — not specified in the source plan]`).

The source environment's short split is partitioned into three disjoint subsets:

* **Held-in** (`n_in = 24` instances): used for weakness mining and as one arm of proposal validation.  
* **Held-out** (`n_ho = 40` instances): used only for proposal validation, never for mining. Provides an unbiased check that a candidate edit generalizes beyond the failures that motivated it.  
* **Test** (`40` short-test instances; the matching long-test set is `150` instances — the same 40/150 sizes are used for the target environment's short/long test sets, Section 3.5): withheld entirely until the harness is frozen (Section 3.3).

The source-long split, and both splits of the target environment, are held out from optimization in their entirety and are used exclusively in the evaluation stage. No harness edit, promotion decision, or hyperparameter choice may be conditioned on any instance from these four sets.

| Split | Environment | Length | Size | Role during optimization | Role during evaluation |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Held-in | Source | Short | 24 | Weakness mining; validation | — |
| Held-out | Source | Short | 40 | Validation only | — |
| Test (source-short) | Source | Short | 40 | Untouched | Evaluated |
| Test (source-long) | Source | Long | 150 | Untouched | Evaluated |
| Test (target-short) | Target | Short | 40 | Untouched | Evaluated |
| Test (target-long) | Target | Long | 150 | Untouched | Evaluated |

## **3.2 Self-Harness optimization procedure**

Starting from the unmodified harness B1, optimization proceeds for up to `T = 15` rounds, or until `[patience, placeholder — not given a numeric value in the source plan; described only as "several consecutive rounds without a promotion"]` consecutive rounds produce no promoted edit, whichever occurs first. Each round consists of three stages:

**(1) Weakness mining.** The current harness is run on all `n_in` held-in instances, `m = 2` times each, using an auxiliary sub-verifier to score both the final answer and intermediate sub-call outputs. Each failure is recorded as a structured record capturing the verifier's stated cause, the level (root vs. child) at which it occurred, its causal status relative to the final error, and the harness mechanism implicated. Failure records are clustered into a small number of dominant patterns, which form the evidence bundle passed to proposal.

**(2) Harness proposal.** Conditioned on the evidence bundle, the model proposes `K = 4` candidate edits per round, each a minimal, targeted modification to exactly one of the nine editable surfaces (S1–S9, Section 3.0), addressing one mined failure pattern.

**(3) Proposal validation.** Each of the `K` candidates, together with the current (pre-round) harness as a baseline, is evaluated on held-in ∪ held-out (`n_in + n_ho` instances), `v = 4` times each. A candidate is **promoted** if it produces no accuracy regression exceeding a preregistered threshold `τ = [placeholder — set from pilot variance measurements, Section 3.9; a value of zero recovers the strict non-regression rule of Zhang et al.]` (calibrated against baseline run-to-run variance measured in the pilot, Section 3.9) and its per-instance cost falls within a preregistered band `[cost_min, cost_max] = [placeholder — set from pilot, Section 3.9]`. If multiple compatible candidates are promoted in the same round, the merged harness (all promoted edits applied jointly) is additionally re-evaluated before being accepted as the round's output; if the merge itself regresses, the round promotes nothing rather than falling back to individually accepted edits.

Total per-round evaluation volume is `(n_in · m) + (n_in + n_ho) · (K + 1) · v` runs. At `n_in = 24`, `n_ho = 40`, `m = 2`, `v = 4`, `K = 4`, this is `1,328` runs per round and `19,920` short runs in total over `T = 15` rounds.

Algorithm: Self-Harness optimization  
Input: harness H\_0 \= B1, held-in D\_in, held-out D\_ho, T, patience  
H ← H\_0; rounds\_without\_promotion ← 0  
for t \= 1 … T:  
    traces ← run(H, D\_in, repetitions=m)              \# mining  
    failures ← sub\_verify(traces); patterns ← cluster(failures)  
    candidates ← propose(H, patterns, K)               \# proposal  
    results ← { validate(c, D\_in ∪ D\_ho, repetitions=v) : c ∈ candidates ∪ {H} }  
    promoted ← { c ∈ candidates : no\_regression(results\[c\], results\[H\], τ)  
                                    and cost(c) ∈ \[cost\_min, cost\_max\] }  
    if promoted ≠ ∅:  
        H' ← merge(promoted)  
        H ← H' if not regressed(H', results) else H   \# merge failure promotes nothing this round  
        rounds\_without\_promotion ← 0  
    else:  
        rounds\_without\_promotion \+= 1  
        if rounds\_without\_promotion ≥ patience: break  
return H   \# frozen as SH-RLM

## **3.3 Freezing the harness**

Once the stopping criterion in Section 3.2 is met, the harness is frozen and receives no further edits. This frozen harness is denoted **SH-RLM**. No test split (source-short, source-long, target-short, or target-long) is accessed prior to this point; freezing occurs strictly before any evaluation-stage computation.

## **3.4 Baselines**

Three harness conditions share the frozen backbone and decoding configuration described in Section 3.0:

* **B1** — the unmodified reference RLM harness (all nine editable surfaces at their initial/minimal state), unchanged throughout.  
* **H1 (λ-RLM)** — a hand-engineered harness from prior work, held fixed and used as an upper-bound reference for manually designed harness engineering.  
* **SH-RLM** — the frozen output of the optimization procedure in Sections 3.2–3.3.

An optional fourth condition, **F1**, is a weight-fine-tuned model on the same backbone and source-short training distribution, evaluated where a published checkpoint is available (Appendix A); F1 differs from the other three in that it modifies weights rather than harness code, and is reported separately rather than as a primary comparison.

## **3.5 Fixed-weight evaluation**

With weights and harness both frozen, B1, H1, and SH-RLM are each evaluated on all four test sets (source-short, source-long, target-short, target-long), with `3` seeded repetitions per instance per condition to support confidence-interval estimation. No harness or weight changes occur during this stage for any condition.

The **primary comparisons** are SH-RLM vs. B1 and SH-RLM vs. H1 on **source-long, target-short, and target-long** — the three settings in which nothing about the optimization procedure directly targeted the evaluated length or environment, making improvement here evidence of a transferable strategy rather than a memorized fix. Source-short is reported as a sanity check but is not treated as a primary result, since it is drawn from the same distribution used for weakness mining.

| Test set | Environment | Length | Comparison role |
| ----- | ----- | ----- | ----- |
| Source-short | Source | Short | Secondary (in-distribution sanity check) |
| Source-long | Source | Long | Primary (length generalization) |
| Target-short | Target | Short | Primary (task generalization) |
| Target-long | Target | Long | Primary (joint length \+ task generalization) |

## **3.6 Metrics**

**Primary metric.** Verifier-scored task accuracy per test set, reported with bootstrap confidence intervals (`[n_bootstrap, placeholder]` resamples) and a paired significance test (`[test type, placeholder — e.g. paired bootstrap / Wilcoxon signed-rank]`) between B1 and SH-RLM and between H1 and SH-RLM on each test set.

**Secondary (efficiency) metrics.** Total tokens per instance, sub-call count, maximum recursion depth, and accuracy per million tokens, reported per condition per test set.

**Mechanism-level (trace) metrics.** Using the sub-verifiers introduced during weakness mining: (i) root- vs. child-level failure attribution per test set and condition; (ii) whole-input sub-call collapse rate (fraction of runs in which the harness routes the entire input to a single sub-call rather than decomposing it); (iii) frequency of each mined failure pattern (Section 3.2) before vs. after optimization, to assess whether promoted edits reduced the specific mechanisms they targeted.

## **3.7 Ablations**

* **Leave-one-edit-out.** For each promoted edit in the SH-RLM lineage, that edit alone is removed from the final harness and the affected test conditions are re-evaluated, to attribute accuracy changes to individual edits rather than the harness as a whole.  
* **Sub-verification ablation (contingent on compute budget, Appendix B).** The full optimization procedure (Section 3.2) is re-run with sub-verifier signal withheld from weakness mining (root-level verification only), and the resulting harness is compared against SH-RLM on all four test sets. This isolates the contribution of mechanism-level (child-call) failure attribution to the quality of mined weaknesses and, downstream, to final harness performance.

## **3.8 Compute budget**

Total optimization cost is `(n_in · m) + (n_in + n_ho) · (K+1) · v` runs per round, times up to `T` rounds. Total evaluation cost is `3 conditions × 2 environments × (40 short-test + 150 long-test instances) × 3 repetitions` \= `720` short runs and `2,700` long runs. `[Report final measured token totals, GPU-hours, and cost here after the Section 3.9 pilot and main run.]` The source plan's initial estimate: optimization ≈`2.4×10⁹` tokens; evaluation ≈`9×10⁷` tokens for the 720 short runs plus ≈`3.2×10⁹` tokens for the 2,700 long runs (long runs dominate, since long instances are 8–32× larger and involve repeated context re-reads during recursion); project total ≈`5–6×10⁹` tokens, ≈`650` H100-hours served locally, or ≈`$1,200–$2,000` at current hosted open-weights inference rates. The sub-verification ablation (Section 3.7), where run, is budgeted as a separate full optimization pass (≈`2.4×10⁹` tokens) and is not included in the primary total above.

## **3.9 Pilot calibration (Section 3.0 reference run)**

Prior to the main experiment, a small-scale pilot is run on both environments (short and long splits) to measure, under the actual backbone and decoding configuration: (i) tokens per run and passes-over-context multiplier during recursion; (ii) baseline run-to-run accuracy variance, used to set the regression threshold `τ`; (iii) typical sub-call cost, used to set the cost band `[cost_min, cost_max]`. `n_in`, `n_ho`, `m`, `v`, `K`, `T`, and the test-set sizes are already fixed per the source plan (`24`, `40`, `2`, `4`, `4`, `15`, and `40`/`150` respectively); the pilot instead fixes patience, `τ`, and `[cost_min, cost_max]`, none of which are given numeric values in the source plan. Final values replace the remaining placeholders above before the optimization stage begins; no further adjustment is made once optimization starts.

## **3.10 Calibrated hyperparameters**

| \# | Parameter | Section | Status | Value | Notes |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 1 | Temperature | 3.0 | **Set** | `0.7` | Qwen3-30B-A3B-Instruct-2507 model-card default (non-thinking mode). Applies to root \+ sub-calls identically per the invariant. |
| 2 | Top-p / top-k / min-p | 3.0 | **Set** | `top_p=0.8, top_k=20, min_p=0` | Same source as above. `top_k`/`min_p`are not currently wired through the repo's Gemini client — needs a code change or extra\_body routing via a vLLM endpoint if used. |
| 3 | Max output tokens (root \+ sub-call) | 3.0 | **Needs pilot** | Start at `4096`/turn; range `2048–8192` | Watch pilot traces for truncated REPL blocks/answers to tune up, or wasted budget to tune down. |
| 4 | Long-split construction method | 3.1 | **Needs implementation decision** | OOLONG-Pairs: use existing `ALL_CONTEXT_LENGTHS`, e.g. short=4096 (8x) or short=8192 (32x) vs. long=32768/262144 — no new code. GraphWalks: no upstream long file exists; choose between (a) concatenating multiple instances with a target-graph query, or (b) regenerating larger native graphs via GraphWalks' own generation method — neither implemented in-repo yet. | GraphWalks is the source environment and blocks source-long (a primary comparison) until resolved. |
| 5 | Patience (rounds w/o promotion) | 3.2 | **Reasoned guess** | `3` rounds; range `3–5` | Self-Harness (Zhang et al.) ran 15/18/21 rounds across three model families with uneven promotion spacing — raise toward 5 if pilot rounds show frequent 2-round rejection streaks. |
| 6 | τ (regression threshold) | 3.2 | **Needs pilot measurement** | `~1–2 × SD` of B1 pass-count variance across repeated held-in/held-out runs | Consider adopting proposal's two-threshold form (`τ_reg` for non-regression, `τ_imp` for minimum improvement) rather than the single `τ` here — decide before the pilot, since it changes what's measured. `acceptance_inputs()` in `shrlm/runner.py` may already hardcode a default worth checking. |
| 7 | Cost band `[cost_min, cost_max]` | 3.2 | **Needs pilot measurement** | `[0.5×, 2×]` of B1's mean `HarnessRun.metrics["cost"]`on held-in | Read directly from existing `HarnessRun`/`UsageSummary` logging — no new instrumentation needed. |
| 8 | n\_bootstrap | 3.6 | **Set** | `10,000` | Conventional; not data-dependent. |
| 9 | Paired significance test | 3.6 | **Set** | Paired bootstrap over instance-aggregated pass rate | Matches a binary/pass-fail verifier metric and reuses the same CI machinery as the primary-metric bootstrap. Fall back to Wilcoxon signed-rank only if you want a non-resampling nonparametric test. |

| Step | Stage | Purpose |
| ----- | ----- | ----- |
| 1 | Pilot calibration | Fix decoding config, τ, cost band, patience, and max-output-tokens via small-scale runs on both environments before any optimization starts (§3.9) |
| 2 | Initial RLM baseline (B1) reproduction | Smoke-test the unmodified harness end-to-end on a small sample; establish starting performance |
| 3 | Self-Harness loop validation | Run the full mining → proposal → validation loop on a held-out dev subset (excluded from the real experiment) to confirm it works and is reproducible from saved configs |
| 4 | Self-Harness optimization | Run the full loop on source held-in/held-out; mine failures, propose edits, validate and promote across rounds |
| 5 | Freeze harness (→ SH-RLM) | Lock the final harness once the round cap or patience criterion is hit; no further edits permitted after this point |
| 6 | H1 (λ-RLM) baseline integration | Bring the hand-engineered harness online as the upper-bound reference for manual harness engineering |
| 7 | Source-short evaluation | Improvement on the training environment (in-distribution sanity check — secondary, not primary) |
| 8 | Source-long evaluation | Length generalization within the source environment (primary) |
| 9 | Target-short evaluation | Task/environment generalization without a length shift (primary) |
| 10 | Target-long evaluation | Joint task \+ length generalization (primary) |
| 11 | F1 fine-tuned baseline (optional) | Reference point bounding how much of the gap is model capability vs. orchestration, if a checkpoint/compute is available |
| 12 | Leave-one-edit-out ablation | Remove each promoted edit individually and re-evaluate, to attribute gains to specific edits rather than the harness as a whole |
| 13 | Sub-verification ablation (contingent) | Re-run optimization with child-level verifier signal withheld, to isolate its contribution to mined-weakness quality |
| 14 | Statistical \+ trace analysis | Compute CIs, paired significance tests, failure-pattern frequency before/after, whole-input collapse rate, root/child failure attribution |
| 15 | Alignment drift probe (optional) | Score refusal/harmful-answer rates across the promoted-harness lineage to check for unintended compliance drift |


# Evaluation Plan

This specifies the two evaluation components for SH-RLM: (1) quality of final results relative to baselines, and (2) quality relative to cost.

---

## 1. Eval of Quality vs. Baseline

### 1.1 Purpose

Determine whether the Self-Harnessed RLM (SH-RLM) produces higher-quality outputs than (a) an unmodified reference harness and (b) a hand-engineered harness, and whether any gains survive a length shift and a change of task environment.

### 1.2 Conditions compared

| Condition | Description | Role |
|---|---|---|
| **B1 — Initial RLM (H₀)** | Unmodified reference harness; every editable surface at its sparse default | Zero-intervention baseline; starting point of optimization |
| **H1 — λ-RLM** | Hand-designed harness (typed functional runtime, bounded leaf sub-problems), unmodified | Expert-engineered baseline; tests SH-RLM against human harness engineering |
| **SH-RLM** | Frozen harness produced by the propose-validate-promote optimization loop | System under test |
| **F1 — Fine-tuned RLM** (secondary, reported separately) | Same backbone, RL-trained inside the initial harness on the source-short split | Reference point only — not weight-matched to the other three, so excluded from primary comparisons |

### 1.3 Test sets

Each condition (excluding F1's separate budget) is evaluated on four untouched categories, held out from optimization:

1. **Source-short** — measures improvement at the optimization length (in-distribution check).
2. **Source-long** — measures length generalization (8–32× longer inputs than seen during optimization).
3. **Target-short** — measures cross-environment transfer with no length shift.
4. **Target-long** — measures cross-environment transfer *and* a length shift simultaneously (hardest test).

Source and target are two different task environments (e.g., GraphWalks and OOLONG-Pairs) with matched short/long splits and deterministic verifiers. Only the source environment is touched during optimization; the target environment is never seen until final evaluation.

### 1.4 Primary metric

- **Verifier accuracy**, reported separately for each of the four test sets, for each of the three primary conditions (B1, H1, SH-RLM).

### 1.5 Primary comparisons

- **SH-RLM vs. B1** on source-long, target-short, and target-long — tests whether self-discovered harness improvements generalize beyond the conditions they were mined from.
- **SH-RLM vs. H1** on the same three sets — tests whether a self-discovered harness is competitive with hand-designed harness engineering.

Interpretation of each test set:
- Source-long improvement → learned edits survive a length shift within the same environment.
- Target-short improvement → edits capture a transferable compositional strategy, not a source-specific rule.
- Target-long improvement → transfer holds even when environment and length change together.

### 1.6 Statistical treatment

- Results averaged over repeated seeded runs.
- **Bootstrap confidence intervals** computed over task instances.
- **Paired per-instance significance test** between B1 and SH-RLM (exact test and repetition-to-instance aggregation rule to be preregistered).

### 1.7 Supporting / diagnostic evals

These don't stand alone as "quality" numbers but explain *why* quality differs between conditions:

- **Failure pattern frequency, pre- vs. post-optimization** — are the specific failure mechanisms SH-RLM targeted actually reduced?
- **Whole-input sub-call collapse rate** — how often a run delegates almost the entire input to a single child or skips meaningful decomposition (a specific, known failure mode).
- **Root vs. child failure attribution** (via sub-verifiers) — for each condition/split, what share of failures originate at the root (bad aggregation of correct children) vs. at a child (a bad local sub-answer faithfully aggregated). Shows whether optimization repaired errors or just relocated them in the call tree.

### 1.8 Ablations that inform the baseline comparison

- **Sub-verification ablation**: repeat optimization with the sub-verifier signal withheld from weakness mining (proposer sees only root outcomes and traces); compare resulting frozen harness against SH-RLM on all four test sets. Tests whether checkable child-level evidence is what makes mined failure attributions actionable.
- **Leave-one-edit-out**: remove each promoted edit individually from the final harness and re-evaluate affected test conditions. Identifies which edits are responsible for the measured gains vs. sampling noise.

---

## 2. Eval of Quality Relative to Cost

### 2.1 Purpose

Establish whether SH-RLM's quality gains are worth the compute/token cost they require — both the one-time cost of running the optimization loop and the per-inference cost of the resulting harness — and how that cost compares to the alternative of fine-tuning weights (F1).

### 2.2 Secondary efficiency metrics (per condition, per test set)

| Metric | What it captures |
|---|---|
| Total input + output tokens | Raw token cost of a run |
| Recursive-call count | How many sub-calls a run issues |
| Maximum recursion depth | How deep the recursion tree goes |
| **Accuracy per million tokens** | Direct quality-for-cost ratio; the core cost-efficiency metric |

### 2.3 Cost accounting (from feasibility budget)

- **Optimization cost**: driven by weakness-mining runs plus validation runs across held-in/held-out splits and candidate harnesses per round, capped at a preregistered number of rounds (early-stopped after several rounds without a promotion). Estimated at roughly 2.4×10⁹ tokens for the full optimization run.
- **Evaluation cost**: dominated by long-instance runs (8–32× larger inputs), estimated at roughly 3.2×10⁹ tokens for final evaluation across all four test sets, vs. ~9×10⁷ tokens for the short-test runs.
- **Total project cost**: approximately 5–6×10⁹ tokens end-to-end (optimization + source evaluation + target evaluation), roughly 650 H100-hours if served locally, or approximately $1,200–$2,000 at current hosted open-weights inference rates.
- **Sub-verification ablation** adds one further full optimization run (~2.4×10⁹ tokens), budgeted as contingent.
- A **pilot run** (small sample, both environments) precedes full optimization specifically to measure tokens per run, recursive-call counts, and effective passes over the stored context — these pilot measurements fix the final test sizes and token budget before optimization begins, and can trigger a preregistered reduction in long-test sample size if costs run high.

### 2.4 Cost comparison against the fine-tuning alternative (F1)

F1 (RL fine-tuning the same backbone inside the initial harness) is budgeted **separately** from the harness-optimization cost above, since weight-training and harness-optimization costs are not directly comparable (F1 requires 8×H100 nodes and dedicated training compute). F1 is therefore used as a **reference point**, not a matched cost-quality comparison:
- If a published F1 checkpoint is available, it's evaluated directly rather than retrained, and this is disclosed.
- If neither checkpoint nor training compute is available, F1 is dropped, and its absence is reported alongside published figures for the same environment/backbone.

This framing lets the write-up show, qualitatively, how much *quality* a much more expensive fine-tuning approach buys relative to a comparatively cheap harness-optimization approach — even though it's not a strict apples-to-apples cost comparison.

### 2.5 How to present this in the paper

- Report accuracy-per-million-tokens alongside raw verifier accuracy for every condition/test-set pair, so the reader sees quality and cost side by side rather than in separate tables.
- Consider a cost-quality plot (tokens or $ on x-axis, verifier accuracy on y-axis) with B1, H1, SH-RLM (and F1, marked as a non-matched reference) as separate points/series — this is the natural way to visualize a Pareto-style tradeoff.
- Call out explicitly that optimization cost is a **one-time, amortized** cost (paid once to produce the frozen harness) whereas the efficiency metrics per test run reflect the **marginal, per-inference** cost of using that harness going forward — these are different cost regimes and shouldn't be conflated in the writeup.

---

## Quick Reference: Metric → Section Mapping

| Metric | Belongs to |
|---|---|
| Verifier accuracy (4 test sets) | Quality vs. Baseline |
| Bootstrap CIs, paired tests | Quality vs. Baseline |
| Failure pattern frequency pre/post | Quality vs. Baseline (diagnostic) |
| Sub-call collapse rate | Quality vs. Baseline (diagnostic) |
| Root vs. child failure share | Quality vs. Baseline (diagnostic) |
| Total tokens, call count, recursion depth | Quality vs. Cost |
| Accuracy per million tokens | Quality vs. Cost |
| Optimization cost (tokens, $, H100-hrs) | Quality vs. Cost |
| F1 fine-tuning cost comparison | Quality vs. Cost (reference only) |
