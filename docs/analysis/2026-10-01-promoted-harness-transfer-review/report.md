# Promoted harness edits and short-test transfer

Review date: October 1, 2026. Experiment: `experiment_oolong_pairs_dsv4f_20260928_174735_3rounds`.

The promoted harness did not gain a new exactly solved test task. Its five tasks with at least one exact success were a subset of the original harness's six. Exact successes fell from 8/30 to 7/30; mean F1 was essentially unchanged, 0.7866 to 0.7828, while recorded inference cost increased 2.11 times. There were useful partial improvements, especially on T13/W10 and T11/W10, but losses elsewhere offset them.

The traces explain several concrete failures. Some edits target errors that had already been repaired, one introduces an incorrect mathematical invariant, and several checks verify completeness without verifying the actual task predicate. The output-format guard demonstrably repaired one answer, but the iteration-limit fallback bypassed it twice. These findings explain why the intended mechanisms are not established by the promotion scores. They do not establish each edit's isolated causal effect on the full test result.

## Comparison and provenance

The comparison uses the exact saved initial harness before any promotion, `H0*R`, and the final saved harness after round 9. The experiment continued through round 12, stopping after three rounds without promotion. Seven single-edit promotions changed six surfaces: S2, S3, S4 twice, S5, S9, and S10. There were no intervening promotions after round 9.

|Measurement|Original, before promotions|Final promoted harness|
|---|---:|---:|
|Task instances / attempts per instance|10 / 3|10 / 3|
|Exact successful attempts|8/30 (26.7%)|7/30 (23.3%)|
|Tasks with at least one exact success|6/10|5/10|
|Tasks successful on all three attempts|0|0|
|Mean F1, including declared failure zeros|0.7866|0.7828|
|Malformed final answers|0|2|
|Recorded inference cost|$2.4034|$5.0820|
|Mean root trace iterations, including fallback|14.93|18.60|
|Total input / output tokens|11,270,598 / 513,773|23,773,590 / 1,107,794|

All 60 saved trace hashes were verified. Instance records, prompts, and gold pairs are identical between conditions. Each condition contains exactly one attempt numbered 1–3 for each task. Attempt numbers are execution identifiers, not matched random seeds. F1 averages use the saved verifier's three-decimal measurements and its declared zero for malformed answers. Costs are recorded model inference costs, not monetary valuations of elapsed time.

The two conditions ran sequentially, final first and original second, with three workers each. This is a comparison of repeated executions, not a randomized interleaved trial. Ten tasks and three attempts each do not establish a statistically reliable performance regression from a difference of one exact success. The justified conclusion is that this test did not demonstrate a gain.

## Which tasks changed?

`T` identifies the task predicate; `W` identifies the underlying short context window. Exact columns count successes out of three. Role A/B descriptions allow either ordering of the two users; output pairs are unordered and canonicalized.

|Task|Required condition|Original exact|Final exact|Original mean F1|Final mean F1|
|---|---|---:|---:|---:|---:|
|T03/W9|Both users: description or abbreviation|0|0|0.9210|0.9093|
|T05/W10|Both: entity or numeric; all entity dates before March 15, 2023|1|1|0.9733|0.9337|
|T06/W10|Both: location or abbreviation|2|2|0.9840|0.9920|
|T07/W10|Both: description or numeric; all numeric dates after February 1, 2023|2|2|0.9583|0.9793|
|T11/W10|A: entity and abbreviation; B: exactly one entity|0|0|0.5077|0.6077|
|T12/W9|A: at least two numeric; B: location and human|1|1|0.6770|0.6450|
|T12/W10|Same T12 predicate, different context; gold answer empty|1|1|0.3333|0.3333|
|T13/W10|A: exactly one description; B: abbreviation and entity|0|0|0.6693|0.8833|
|T17/W9|A: exactly one numeric; B: location and description|1|0|0.9647|0.7967|
|T20/W9|A: numeric and human; B: location, entity, and exactly one abbreviation|0|0|0.8773|0.7473|

Thus, the final harness retained exact successes on **T05/W10, T06/W10, T07/W10, T12/W9, and T12/W10**, and lost the original's one success on **T17/W9**. It added no newly solved instance. The exact counts on the other nine tasks were unchanged. Which attempt succeeded sometimes changed, but that is not evidence of moving competence to a new task.

Partial performance did move: F1 improved on four tasks, declined on five, and was unchanged on one. The strongest gains were T13/W10 (+0.2140) and T11/W10 (+0.1000); the largest losses were T17/W9 (−0.1680) and T20/W9 (−0.1300).

[The task catalog](task-catalog.md) contains every full instance ID, verbatim task question, gold-pair count, exact outcome, F1, missing/extra counts when available, and links to all 60 attempts. [The CSV](task-comparison.csv) contains the task-level comparison and individual attempt F1 values.

## What each promoted edit was supposed to do

These are contemporaneous incumbent-versus-candidate held-out measurements, ten tasks with one execution each. They are not cumulative improvements from a common baseline. The exact replacement text, code, and skill body, together with each proposal's claimed mechanism, are in [the promoted-edit appendix](promoted-edits.md).

|Round / surface|Intended intervention|Validation exact, incumbent → candidate|Validation F1, incumbent → candidate|Trace-based assessment|
|---|---|---:|---:|---|
|1 / S2 sub-call guidance|Check merged coverage; rerun missing chunks|0 → 1|0.6408 → 0.7492|The cited run had already repaired the merge and covered all 188 records. Does not establish the claimed coverage repair.|
|2 / S4 verification guidance|Spot-check category assignments and reclassify suspect items|1 → 2|0.6000 → 0.7505|Spot-checking occurred in test execution, but source diagnosis omitted a missing OR branch and actual formatting error. Semantic errors persisted.|
|3 / S3 aggregation guidance|Generate both role directions; deduplicate; verify pair count|2 → 3|0.5926 → 0.7419|Reverse generation was already redundant. The new expected-count instruction is wrong for overlapping eligibility sets and caused false discrepancies.|
|4 / S5 recovery guidance|Normalize mixed ID types after TypeError and verify count|0 → 2|0.6025 → 0.7092|Source had already normalized the IDs. The count check cannot certify the eligibility sets, and no root TypeError occurred in either short-test condition.|
|6 / S9 answer middleware|Redirect answers lacking a pair or explicit empty marker|4 → 5|0.7148 → 0.9098|One direct test repair is confirmed. Both final-harness malformed answers bypassed this guard through iteration-limit fallback.|
|7 / S10 coverage skill|Provide a reusable source-to-classification coverage procedure|3 → 3|0.8324 → 0.8154|No skill loads in its candidate validation; no root loads in the short test. Source's alleged missing-user count is false.|
|9 / S4 excluded-user check|Sample excluded users to find misclassification|2 → 2|0.7882 → 0.7941|Executed in test traces, but cannot target the falsely included user named in the proposal's rationale.|

### Round 1: coverage guidance did not repair the cited mechanism

The proposal said the root processed only chunks 0–3, abandoned chunks 4–6 after a parsing/type failure, and misleadingly reported zero missing labels. The complete source trace contradicts this: iteration 14 repairs the malformed fourth-index result; iteration 15 resets `all_labels = [None] * len(data)`, loops over **all** results, merges with chunk offsets, and prints `Missing labels: 0`. The category counts sum to 188.

That does not prove every label is correct, but it refutes the proposal's specific claim that the remaining chunks were never processed. The final answer still missed pairs, so the unresolved issue was downstream of mere assignment coverage or within the labels/associations themselves. The edit repeats a behavior already demonstrated in the cited failure. It might encourage coverage on other executions, but its validation gain is not evidence that it repaired this run's alleged missing chunks.

Source: [repaired coverage](trace-evidence.md#r01-repaired-coverage), held-in T02/W10 attempt 2.

### Round 2: classification checking was enacted, but the diagnosis missed the predicate

The source run classified all 188 items. It then selected only users with a description label, even though T03 asks for description **or abbreviation**, and formatted pairs without parentheses. The proposal instead attributed the small output to widespread misclassification and described `wrong_format` as being due to incompleteness. The code establishes omitted predicate logic and a formatting violation; it does not establish that the many missing answer users all required new labels.

The promoted S4 instruction asks for 5–10 samples from each category or the small categories, with reclassification until the spot-check passes. The final harness did perform such checks. There are qualified positive outcome signals, particularly T13/W10 and the two high-F1 T11/W10 attempts. However, reviewing labels does not verify a missing OR clause, preserve multiplicity, or ensure the final answer is submitted. The inspection is also performed by the same model without independent record-level label verification. A new confident label is not necessarily a better label.

Source: [the omitted predicate and format](trace-evidence.md#r02-omitted-predicate). A concrete rechecking regression is discussed below.

### Round 3: a redundant intervention introduced an incorrect invariant

The proposal claimed the missing 44 pairs were lost because the program only generated `A × B`. The source already canonicalized each pair with `tuple(sorted((u1, u2)))` and inserted it into a set. Under the task's unordered-pair semantics, also generating `B × A` adds **no new pair**. It cannot make an absent eligible user appear in either set.

The promoted instruction additionally requires the deduplicated count to equal `|A||B|` minus self-pairs. For overlap `s = |A ∩ B|`, the correct count is:

`|A||B| − s − s(s−1)/2`.

It must remove both self-pairs and the duplicate unordered pairs whose two users lie in the intersection. With `A = B = {1, 2}`, there is one pair, while the promoted instruction expects two. An exhaustive offline check of all 256 pairs of subsets of a four-element universe found 67 failures of the promoted formula; the corrected formula and the redundancy of the reverse product held throughout.

This defect was enacted, not merely latent. T12/W9 attempt 2 generated 29 unique pairs and expected 32; T13/W10 attempt 3 generated 33 and expected 34, then repeated enumeration to resolve the apparent gap. Other traces eventually corrected the mathematics themselves. The false invariant consumes reasoning and checking effort without identifying missing gold pairs.

Sources: [already canonicalized source](trace-evidence.md#r03-already-canonicalized), [29 versus 32](trace-evidence.md#test-wrong-count-and-fallback), [33 versus 34](trace-evidence.md#test-rechecks-degrade), and [offline checks](verified-measurements.json).

### Round 4: recovery guidance addresses an error already recovered

The proposal itself acknowledges that the source normalized mixed string/integer IDs and regenerated the pairs. The claimed drop from 37 to 27 users follows that normalization: the code converts every existing ID to an integer and deduplicates the set, without dropping classification records. Collapsing string/integer representations of the same ID is not evidence of missing input.

The remaining failure was 84 missing gold pairs. Its proposed post-recovery count check would only verify consistency with the current eligibility sets. Even a correct count formula cannot establish that those sets contain all the right users; this edit also repeats the incorrect overlapping-set formula from S3.

Neither condition's short-test root traces contain a `TypeError`, so the edit's stated root-level trigger was not exercised there. Recursive effects or general prompt influence cannot be excluded, but no observed test repair establishes the intended normalization benefit.

Source: [already recovered error](trace-evidence.md#r04-already-recovered).

### Round 6: the answer guard helped once, but does not guard every return path

This is the clearest directly observed intended success. On T12/W10 attempt 3, the root correctly concluded there were no valid pairs but submitted an empty string. S9 redirected it; the next iteration submitted `No valid pairs found.` and passed. This is an observed formatting repair, although the original also solved this instance once and there was no net gain in exact-task coverage.

Both malformed final-harness answers followed 30 REPL iterations and the extra fallback completion. The runtime applies S9 when a REPL block signals a final answer, but its `_default_answer` path returns directly after the iteration loop. The two fallback completions were unexecuted REPL code rather than answer pairs, and never passed through S9. This boundary is in the existing runtime, not newly introduced by the promoted code.

The runs had already printed complete intermediate pair lists: T12/W9 attempt 2 had F1 0.8276 at iteration 17, and T11/W10 attempt 2 had F1 0.6111 at iteration 25. Their eventual malformed responses scored zero. These are intermediate-answer diagnostics, not replacement official test scores or guarantees that an earlier stopping policy would always improve performance.

S9 is also only a presence check: it accepts any text containing a pair or the specific empty marker. It does not establish completeness or semantic correctness. Its regex is narrower than the verifier's whitespace-tolerant parser; that is a compatibility risk, not the observed cause of these two failures.

Sources: [successful S9 redirect](trace-evidence.md#test-s9-repair); the two fallback traces in the evidence appendix; frozen runtime `rlm/core/rlm.py`, normal middleware around lines 585–608 and direct fallback return around lines 669–688.

### Round 7: the coverage skill was not demonstrated to execute its intended repair

The proposal alleged that the source had only 58 users compared with “gold's 82+.” The entire W10 input has **58** distinct users. Its source trace also prints 188 refined classifications against an expected 188 before rebuilding those users. The claimed population deficit is therefore not a defensible explanation of missing answer pairs.

There were zero recorded skill loads across the ten candidate-validation executions that promoted S10. In the 30 final-harness test executions there were two loads, both in nested children, and none at the root. Both child blocks printed only the first 500 characters of the skill. These load events do not demonstrate application of the root merge-checking procedure. Coverage checks can still happen without loading the skill; the narrower finding is that the added skill's intended execution benefit is unestablished.

The skill also has latent correctness problems: its nonempty-line count can include a header and task text rather than only records; text-keyed mappings collapse duplicate records; and its suggested fallback index is local to a chunk, so merging multiple chunks without an offset or composite key collides. These are static counterexamples, not attributed causes of the short-test results, since root use was absent.

Sources: [full source population](trace-evidence.md#r07-full-population), the exact S10 body in [promoted edits](promoted-edits.md), and the canonical tree-load events in [verified measurements](verified-measurements.json). Tree traversal follows `rlm_calls` once and avoids duplicate observation records.

### Round 9: checking excluded users cannot catch the cited included user

The proposal describes user 48114 as falsely included in the target set, producing 32 extra pairs. It proposes sampling users **outside** that set and relabeling any who should have qualified. That can discover false exclusions; it cannot inspect this falsely included user. The direction of the claimed repair is reversed.

Test execution did include excluded-user checks. On T17/W9 attempt 1, the program even inspected an excluded user with a numeric record, but the implemented condition had already changed to numeric **and location**. Since the check used the mistaken predicate, it did not detect the mistake. More inspection against the same incorrect condition does not validate the task.

Sources: round 9 rationale and replacement in [promoted edits](promoted-edits.md), and [T17's predicate and excluded-user check](trace-evidence.md#test-lost-predicate).

## Why the improvements did not transfer

### The lost exact task exposes a predicate failure, not missing input

On T17/W9, original F1 across attempts was `[1.000, 0.931, 0.963]`; final F1 was `[0.518, 0.978, 0.894]`. The first final-harness attempt received 188 labels, merged per-user labels into sets, and used **numeric plus location** for role A. The actual role A is **exactly one numeric record**. Set conversion destroys the multiplicity needed to express that condition. It printed 54 pairs, versus 116 in the gold answer, while still doing coverage, spot checks, excluded-user checks, and the erroneous S3 count comparison.

For a controlled offline reconstruction, I retained the exact four saved child outputs, preserved their per-user label lists, and changed only the aggregation predicate to the requested one. This produced 114 pairs, two missing and zero extra, with F1 **0.9913**, versus the saved attempt's **0.518**. No model was rerun and no labels were corrected. It still was not exact, so residual classification/association errors matter, but most of this attempt's loss is explained by a concrete aggregation error.

The original successful attempt implemented the count condition correctly. This contrast establishes the operation that went wrong; it does not prove a particular promoted surface caused the model to choose it. Source and reconstruction are in [trace evidence](trace-evidence.md#test-lost-predicate) and [reproduce.py](reproduce.py).

### Rechecking does not necessarily improve the intermediate result

On T13/W10 attempt 3, iteration 10 printed 33 pairs with four missing, no extra, and F1 0.9429. Subsequent label rechecks, updates, and a reversal changed the final result to 29 pairs with eight missing and F1 0.8788. The run also wrote to a hard-coded label index before locating the actual record index. This is evidence that additional verification work can introduce or preserve errors; it is not a proof that every S4 check is harmful.

Another T13 attempt checked whether a description label was present rather than counting exactly one. Its final verification reused that weakened condition and declared all 50 produced pairs valid. This is a further visible predicate mismatch, although the trace alone does not prove the omitted count check changed membership for that particular set of assigned labels.

Classification remains unstable, but repeated label inspection and pair counting do not replace verification that the computation enforces the task's quantifiers, alternatives, dates, and asymmetric roles.

### Promotion measurements do not establish a durable monotonic improvement

The same saved harness scored **3/10 as round 3's candidate and 0/10 as round 4's baseline**. Another identical harness scored **5/10 as round 6's candidate and 3/10 as round 7's baseline**. The saved harness hashes match in each comparison. These are direct demonstrations of variability in this evaluation pipeline, even with temperature zero configured.

With one execution per held-out task, selecting a candidate that ties or beats its freshly measured incumbent does not show that its intended mechanism was fixed or that the gain will persist. Reusing a small held-out set for selection also leaves room for selection effects. This review does not estimate the size of those effects or claim that every gain was noise.

Rounds 7 and 9 correctly followed the configured exact-pass tie policy. Round 7's F1 decline is not a policy violation, and F1 need not become a universal promotion gate. The issue is interpretation: a permitted promotion is not evidence of the skill's execution or mechanism repair. F1 remains useful diagnostic information alongside exact passes.

### The short-test split is not a new underlying-record challenge

Held-in, held-out, and short-test task instances are disjoint, but the short conditions use the same two source record collections, with 188 records each. The results measure performance on held-out task instances over those collections. The observed failure cannot simply be explained as encountering entirely unfamiliar underlying records; the harness must carry the classification and aggregation procedure across different predicates.

This is a description of the chosen evaluation scope, not a recommendation to redesign the split. Larger record collections remain a separate long-evaluation question. This review makes no claim about long-test performance or unseen-source-data generalization.

## Small, general corrections suggested by the evidence

1. **Reject a proposed invariant that fails a tiny counterexample.** For claims about sets, ordering, counts, identities, or aggregation, check the stated rule on the smallest relevant case before relying on it. The overlap example would have rejected the S3/S5 count rule without another expensive task run.
2. **Judge the operation remaining after repair.** The diagnosis and proposal must identify the last relevant operation, distinguish its observed effect from uncertainty, and explain how the replacement changes that operation. Already repaired parsing and already canonicalized pairs are not unresolved mechanisms.
3. **Ask whether the computation still enforces the task's actual conditions.** Preserve the information those conditions need: multiplicity, identity, ordering, dates, and role membership where applicable. Checking the size of a result or sampling labels against the existing code is insufficient when the code implements another predicate. This is a general proposer/verification lever, not an OOLONG recipe.
4. **Bound checking and cover every final-answer return path.** A check needs a termination condition and a usable answer before the iteration budget expires. The runtime's fallback should respect the same applicable output contract as normal submission; simply adding further redirects after exhaustion would not solve the budget problem.
5. **Report activation separately from promotion.** An added recovery rule or skill can pass validation without its trigger occurring. Keep the promotion policy, but record mechanism status as observed, not observed, or not assessable. Do not credit an unloaded skill with a demonstrated repair.

These are recommendations only. No harness, optimizer, runtime, evaluation, or split was changed during this review.

## Review coverage and limits

The review covered all seven promoted replacements, their claimed source failures, the validation outcomes and lineage, all 60 short-test result records, and canonical trace-tree activation events. Detailed trace reading concentrated on the cited mining failures, the lost exact task, malformed final answers, contradictory count checks, and rechecking regressions. It was not an exhaustive semantic adjudication of every child label in every execution.

Correctness, reliability, performance, evaluation validity, and instruction activation were inspected sequentially under the supplied AGENTS tool mapping. No independent reviewer or cross-model review was run; the configured Git-diff route did not scope these saved generated artifacts. Findings were checked against source traces and offline counterexamples. This is an experiment-artifact review, not a merge-readiness verdict on the current branch.

No new paid calls, task executions, or surface ablations were launched. Per-edit test contributions remain unmeasured. The report distinguishes confirmed trace behavior, offline counterexamples/reconstructions, and explanations consistent with the evidence.

Reproduce numeric checks from the repository root with:

```bash
.venv/bin/python docs/analysis/2026-10-01-promoted-harness-transfer-review/reproduce.py
```

Saved artifacts: [measurements](verified-measurements.json), [all attempts](run-audit.json), [task catalog](task-catalog.md), [task comparison CSV](task-comparison.csv), [exact promoted edits](promoted-edits.md), [selected trace evidence](trace-evidence.md), and [scope/provenance](scope.json).

Initial harness hash: `0ca3510ded8a0ce5095c2e3338c4d8606fbc819650c20b65079b405ae1ecac2b`.

Final harness hash: `f81e4432b48978029e9e9546704d6b40a8a4e5674dc7da6c5ddbfc46425469d4`.

Frozen source and checkout reviewed: `e323f4328e0aa37b2cb5ff67cabd4729d1b0916c`.
