# What the OOLONG-Pairs experiment changed in its harness

Experiment: `experiment_oolong_pairs_dsv4f_20260916_131837`  
Report date: September 23, 2026  
Scope: all eight completed optimization rounds, all materialized proposals, and all distinct submitted edits that never reached validation. The long-context comparison was stopped at the user’s request after three edited attempts and is reported separately.

Navigation: [validation results](#round-by-round-validation) · [promoted changes](#the-promoted-edits-in-plain-language) · [unpromoted changes](#what-was-tried-but-not-promoted) · [exact promoted edits](#appendix-a-promoted-edits) · [exact tested unpromoted edits](#appendix-b-tested-unpromoted-edits) · [exact untested drafts](#appendix-c-unmaterialized-drafts).

## What changed

The experiment promoted **five edits in three rounds**, changing only two surfaces: **S2, the instructions for decomposing work and asking children for results; and S4, the instructions for checking those results before answering**. The final harness asks for more complete classification outputs, checks user coverage, spot-checks labels, repairs incorrect result keys, and asks the root to rerun every chunk after changing a classification prompt.

Each promotion improved the exact-pass count by one on that round's ten held-out attempts, and each also improved mean F1. These are observed validation gains, not proof that every added instruction helped. Two promotions were bundles, and the execution traces show both useful checking behavior and cases where the model succeeded after departing from the new procedure.

**Eleven other materialized edits were evaluated but not promoted.** Some lowered both exact accuracy and F1. Others improved F1 while losing exact matches: round 6 improved **0.6194 → 0.7193**, and round 8 improved **0.6128 → 0.6951**. Those outcomes deserve a different interpretation from an unequivocally harmful edit. They still failed the configured promotion rule, and their mechanisms are not established by one attempt per task.

There were also **11 distinct submitted edit payloads that never became validated candidates**, after deduplicating identical resubmissions within a round. These were rejected for proposal-contract, routing, or collision reasons. They have no validation score. Exact replacement text/code/policies for the five promoted edits, the 11 tested unpromoted edits, and the 11 untested drafts appear in Appendices A–C.

## How to read the results

The experiment mined weaknesses from **20 held-in short-context tasks, two attempts each**. Promotion used a separate, fixed **ten-task held-out short-context set, one attempt per task per harness**. The long-context test pool was reserved for final evaluation. The saved configuration specifies approximately **8k-token optimization contexts and 262k-token long-test contexts**. Results on these pools must not be pooled or presented as the same evaluation.

In each round, the current incumbent was run again alongside the proposed candidate or merged batch. Promotion required a strictly positive change in held-out exact-pass count, with no permitted count regression. Candidate mean cost could be up to **3× the fresh baseline mean cost**. All eight measured candidates satisfied the recorded cost band; every rejection in this experiment was decided by the exact-pass rule. F1 was diagnostic and did not override that gate. [Saved configuration](../../../experiment_oolong_pairs_dsv4f_20260916_131837/eval/.launch/20260922T202236Z/experiment.toml), [promotion implementation](../../../shrlm/optimization/promotion.py).

F1 gives partial credit when a produced pair set overlaps the expected set. An exact pass requires the complete correct set. This report uses the verifier's **saved, three-decimal F1**, averaged over all ten attempts; a malformed or terminated attempt with no scored answer contributes **zero**. Unfinished attempts are not treated as zero. The audit also reports F1 restricted to scored answers, and missing/extra-pair diagnostics, so changes in answer availability are visible. This is a macro average across tasks, not a pooled pair-level F1. [Metric extraction](../../../shrlm/environments/oolong_pairs.py), [complete audit](audit.json).

### Round-by-round validation

The cost column is the total recorded cost of each ten-attempt subject, not the whole optimization round. Costs with a terminated run can be lower bounds. Each row compares its own freshly measured baseline and candidate; the rows are not a cumulative learning curve.

| Round | Tested surfaces | Exact passes, baseline → candidate | Mean F1, baseline → candidate | F1 change | F1 wins / ties / losses | Recorded cost, baseline → candidate | Decision |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| 1 | S4 + S9 | 2/10 → 2/10 | 0.7085 → 0.5429 | −0.1656 | 2 / 2 / 6 | $0.6963 → $1.1387 | Rejected: no exact gain |
| 2 | S2 + S4 + S3 | 1/10 → 0/10 | 0.7156 → 0.6098 | −0.1058 | 3 / 2 / 5 | $0.6607 → $1.8009 | Rejected: exact regression |
| 3 | S4 | 2/10 → 3/10 | 0.5980 → 0.6791 | +0.0811 | 4 / 3 / 3 | $0.5860 → $1.1288 | **Promoted** |
| 4 | S4 + S2 | 1/10 → 2/10 | 0.6532 → 0.7250 | +0.0718 | 5 / 2 / 3 | $1.5660 → $2.1114 | **Promoted together** |
| 5 | S2 + S4 | 1/10 → 2/10 | 0.6859 → 0.7354 | +0.0495 | 6 / 3 / 1 | $2.5308 → $3.7616† | **Promoted together** |
| 6 | S6 | 1/10 → 0/10 | 0.6194 → 0.7193 | +0.0999 | 3 / 6 / 1 | $2.3966 → $3.5096 | Rejected despite F1 gain |
| 7 | S4 + S3 | 1/10 → 1/10 | 0.6355 → 0.5916 | −0.0439 | 5 / 3 / 2 | $2.6134 → $2.0649 | Rejected: no exact gain |
| 8 | S6 + S4 + S3 | 3/10 → 1/10 | 0.6128 → 0.6951 | +0.0823 | 3 / 4 / 3 | $2.3542 → $4.0138 | Rejected despite F1 gain |

† Round 5's candidate had one resource termination. F1 win/tie/loss counts compare matching task IDs, using the same failed-answer-zero convention as the means. All figures are reconstructed from the persisted validation manifests and checked against the decision ledgers. [Machine-readable table](rounds.csv), [round and per-attempt audit](audit.json).

Optimization stopped after round 8 because rounds 6–8 supplied three consecutive rounds without a promotion, matching `patience = 3`. The final harness is the round-5 batch, not the round-8 candidate.

## The promoted edits, in plain language

### Round 3: check that every input user survives the merge — S4

**Edit:** `r03-c01-s4`. S4 was previously empty. The replacement tells the root to compare the set of users in the parsed input with the set of users represented in merged child labels, identify missing IDs, rerun their chunks, and compute the final answer only after coverage is confirmed.

This introduces a concrete check at a useful boundary: a child can return valid-looking output that silently omits users. Comparing sets is also stronger than comparing only counts, as the rejected round-1 wording did: equal counts can conceal different identities. However, having one label for every user still does not prove that every question was labeled or that the labels are correct.

**Measured result:** exact passes rose 2 → 3 and F1 rose 0.5980 → 0.6791, at 1.93× the recorded mean cost. Task `t09/w10` improved from 13 missing pairs, no extras, and F1 0.923 to an exact answer. No previously exact answer was lost, although other partial-credit scores regressed.

**What the trace supports:** the successful `t09/w10` candidate checked that all 188 records had classifications, then inspected suspect labels and reclassified batches with more explicit category definitions. Its first coverage check already reported no missing record IDs. Thus, the final improvement is consistent with more deliberate verification and semantic correction; it does **not** establish that recovering skipped records was the decisive cause. [Recorded operations: round 3, iterations 9, 11, and 13](trace-evidence.json).

### Round 4: check label meaning and demand complete child outputs — S4 + S2

**Edits:** `r04-c01-s4` adds label spot-checks after the coverage check. It asks the root to inspect 3–5 questions per child chunk, compare labels with the six category definitions, and rerun a chunk with a clearer prompt when a sampled label looks wrong. `r04-c02-s2` changes the child-prompt example to emphasize classifying every line and returning a complete mapping for all users in a chunk.

These address two different failure possibilities: information can disappear during classification or merging, and information can be present but semantically wrong. The combined idea is stronger than treating every missing output pair as evidence of missing input coverage.

**Measured result:** exact passes rose 1 → 2 and F1 rose 0.6532 → 0.7250, at 1.35× recorded mean cost. Two tasks became exact, while one previously exact task regressed. In `t10/w9`, ten extra pairs disappeared; in `t09/w10`, fourteen extra pairs disappeared. This is evidence of improved pair precision in those attempts, not merely improved coverage.

**What the trace supports:** the successful `t10/w9` candidate inspected abbreviation/numeric classifications and the relevant dates before submitting. That is consistent with the new semantic-checking instruction, but there is no independent ablation separating S2 from S4. The model's own label inspection also is not an independent ground-truth label verifier. [Recorded operations: round 4, iterations 9–10](trace-evidence.json).

**A remaining contradiction:** the S2 worked example still asks for a JSON object of **label counts**, then additionally asks for a mapping of **all users**. Its parsing code still adds values into per-label totals. These are different return contracts. The promotion did not resolve that inconsistency; the exact replacement in Appendix A makes it visible.

### Round 5: verify result keys and propagate corrected prompts — S2 + S4

**Edits:** `r05-c01-s2` adds explicit instructions to return all items rather than only qualifying ones, check whether child keys are line indices or user IDs, and rerun a child that used the wrong key format. `r05-c02-s4` adds a stronger consistency rule: if any chunk is rerun with a corrected prompt, rerun all other chunks with the same prompt and rebuild the merged result.

The intended benefit is to prevent a locally repaired chunk from being combined with stale classifications from the rest of the dataset. This is a plausible source of inconsistent results. The tradeoff is potentially substantial extra work, especially on long inputs or when a small sample triggers repeated global reruns.

**Measured result:** exact passes rose 1 → 2 and F1 rose 0.6859 → 0.7354, at 1.49× recorded mean cost. Six tasks improved in F1, three tied, and one fell from exact to a failed unscored answer. Two tasks lost their extra pairs and became exact: `t14/w10` went from six extras to zero, and `t17/w10` from seventeen to zero. The candidate also incurred one resource termination, so its scored-answer-only F1 of **0.9193** must not replace the all-attempt mean of **0.7354**.

**What actually happened in one success:** on `t14/w10`, the root found 187 labels for 188 questions, identified the user with a missing label, and attempted to rerun all chunks. One corrected result came back as label counts in a Python-style dictionary rather than the requested user-to-label-list JSON. The root then returned to original chunks 0–2 and combined them with a newly repaired chunk 3, achieving complete counts and an exact final answer. The final merge therefore mixed prompt versions despite the new S4 prohibition. This is direct evidence that coverage/key repair can be useful, and a counterexample to claiming that consistent global reruns caused every success. The conflicting S2 return contracts are a plausible contributor to the failed rerun, but the trace alone cannot prove that causation. [Recorded operations: round 5, iterations 6, 7, 8, and 10](trace-evidence.json).

## What was tried but not promoted

The table below describes every materialized unpromoted edit. A batch rejection applies to the combination. It does not prove that each constituent would fail if evaluated alone.

| Round and candidate | Surface | Proposed change | Observed outcome and interpretation |
| --- | --- | --- | --- |
| R1 `r01-c01-s4` | S4 | Check counts of represented users and rerun chunks with missing users. | Shared result: 2 → 2 exact, F1 −0.1656. Counts alone are a weaker coverage test than checking the identities and per-record multiplicity. |
| R1 `r01-c02-s9` | S9 | Recognize alternative pair formats and normalize them. | Same rejected batch. Already recognizable pairs pass through unchanged; unrecognizable answers also pass through. The normalization branch calls `redirect`, which sends a nudge to the model rather than replacing the accepted answer directly. It cannot repair wrong classifications or predicates. |
| R2 `r02-c01-s2` | S2 | Append a generic post-merge coverage check to the decomposition instructions. | Shared result: 1 → 0 exact, F1 −0.1058. More coverage language did not establish better semantic classification or preserve task-specific information in the worked example. |
| R2 `r02-c02-s4` | S4 | Add another post-merge coverage check. | Same rejected batch; overlaps the S2 addition. All ten candidate attempts had scored pair answers, yet none was exact. Avoiding malformed output did not guarantee correct pairs. |
| R2 `r02-c03-s3` | S3 | Require newline-separated pairs and a strict final-output layout. | Same rejected batch. It emphasizes printing, although final submission uses `answer["content"]`. It also imposes a narrower layout than the verifier requires. |
| R6 `r06-c01-s6` | S6 | Enable two retries for syntax-classified sub-call errors. | Exact 1 → 0, but F1 +0.0999. The proposed explanation promises timeout recovery and smaller chunks, which this policy does not implement. See the F1 discussion below. |
| R7 `r07-c01-s4` | S4 | Check the number of labels per user against the number of questions, in addition to existing verification. | Shared result: 1 → 1 exact, F1 −0.0439. The proposal's own incumbent description says the observed execution already checked total and per-user counts; repeating that check is not a demonstrated fix for that execution. Equal counts also cannot establish label accuracy or exact record identity. |
| R7 `r07-c03-s3` | S3 | Request one label per question and append labels when merging, instead of overwriting per-user results. | Same rejected batch. Preserving multiplicity is a useful general principle, but its independent effect was not measured. The conflicting S2 counts example remained in place. |
| R8 `r08-c01-s6` | S6 | Reintroduce the same two-retry syntax policy as round 6. | Shared result: 3 → 1 exact, F1 +0.0823. This is the same policy payload, now bundled with two other edits; it is not evidence that a new timeout strategy was tested. |
| R8 `r08-c02-s4` | S4 | Compare parsed-record count with the number of data-bearing input lines and revisit parsing if records were missed. | Same rejected batch. Reasonable for demonstrated parsing omissions, but the rationale's assertion about a skipped header is not sufficient to show that an actual data record was lost. |
| R8 `r08-c03-s3` | S3 | Demand a canonical newline pair list with an example. | Same rejected batch. The verifier accepts both bracketed pair lists and newline pairs, so forbidding alternative accepted layouts does not directly solve semantic pair errors. |

These interpretations combine the submitted explanations, implemented edits, measured outcomes, and runtime contract. The explanations inside proposals are model hypotheses, not established diagnoses. [All proposal explanations and exact before/after surfaces](audit.json), [S9 redirect behavior](../../../rlm/core/rlm.py), [accepted pair formats](../../../shrlm/environments/oolong_pairs.py).

### The rejected F1 gains are informative—but what improved matters

**Round 6:** three tasks improved in F1, six tied, and one regressed. The largest improvement was `t05/w9`: a malformed baseline answer became a scored answer with F1 **0.796**, 27 missing pairs, and 13 extras. Meanwhile `t10/w9` went from exact to F1 **0.900** with ten extra pairs. The all-attempt F1 increase therefore includes recovery from an unusable answer; it is not evidence that every existing classification became more accurate. Among scored answers only, the mean went 0.7743 → 0.7193, but those means have different denominators (eight versus ten answers).

There is a further causal limitation. The runtime's S6 retry code retries **syntax-classified error results**, and repeats the **same prompt**. It does not split chunks or retry timeouts through these settings. Inspection of the saved candidate traces found **no positive recorded retry count** in either round 6 or round 8. That is an observation about the persisted instrumentation, not proof that every internal event is captured. It provides no support for attributing the F1 gains to activated syntax retries. The round-7 draft with `retry_on_syntax_error = false` would disable this retry path even though its explanation claimed one retry opportunity. [Runtime implementation](../../../rlm/environments/local_repl.py), [saved retry-metric audit](runtime-retry-audit.json).

**Round 8:** the candidate lost two exact passes through relatively small pair errors: `t04/w9` went **1.000 → 0.952** with 21 extras, and `t10/w9` went **1.000 → 0.900** with ten extras. Larger partial-credit gains elsewhere outweighed those losses in mean F1. This is why the batch can be rejected by exact match and still look directionally better on overlap. It cost about **1.70×** its fresh baseline, and S6/S4/S3 were tested together, so the result does not identify an effective individual edit.

The appropriate record is “rejected by the exact-pass gate, with improved partial-credit quality; mechanism and repeatability unresolved.” It would be misleading to call either direction exhausted solely because it was not promoted—or to call the F1 increase a reliable improvement under `v = 1`.

### Drafts that never reached validation

These are proposal-quality failures, not measured harness regressions. Valid sibling candidates were sometimes retained even when an overall response was marked unaccepted. Across fourteen proposal responses, nine had an `accepted: false` flag; that does not mean nine empty rounds. No materialization failures were recorded in this experiment.

| Appendix C item | Round | Surface / idea | Why it did not become a measured candidate |
| --- | --- | --- | --- |
| C1 | 2 | S9 format normalization | S9 was ineligible for the assigned `incomplete_coverage` mechanism. |
| C2 | 2 | S2 revised complete-mapping instructions | Repair selection reason exceeded the 600-character limit; the earlier valid S2 candidate was retained. |
| C3 | 2 | S4 revised coverage instruction | Repair tried to replace an already occupied pattern/surface. |
| C4 | 3 | S2 smaller batches, at most 10–15 simultaneous sub-calls | S2 was ineligible for the assigned `iteration_budget_exhaustion` mechanism. |
| C5 | 3 | S5 split and retry failed chunks | S5 was ineligible for that mechanism. |
| C6 | 3 | S10 batching skill | The repair selected S10 twice; neither skill became a candidate. |
| C7 | 3 | S10 failure-recovery skill | Same S10 collision. |
| C8 | 4 | S9 format normalization | Initial reason too long; repaired version still used S9 for an ineligible coverage mechanism. Identical edit payload appeared twice. |
| C9 | 7 | S6 one retry, syntax retry disabled | Selection reason exceeded the limit in both submissions. Independently, its claimed retry behavior disagrees with the runtime. |
| C10 | 8 | S2 smaller classification chunks and split-on-timeout guidance | S2 was ineligible for the assigned exhaustion mechanism; repair switched to S6 instead. |
| C11 | 8 | S1 parse all data-bearing lines rather than using a fixed offset | S1 was ineligible for the assigned coverage mechanism; repair switched to S4. |

Some of these interventions may be worth reconsidering, particularly task-derived chunk sizing and preserving per-record information. Their current status is **untested**, not “shown not to work.” The S5/S10 recovery drafts also explicitly permit continuing after failed pieces; for an exhaustive pair task, continuing with incomplete evidence could produce a partial answer. That tradeoff would need to be addressed before treating them as a complete solution.

## What can reasonably be concluded

1. **Verification became more explicit, and the promoted validation batches improved both exact counts and F1.** Coverage, key checks, semantic spot-checking, and repairs are concrete behaviors visible in successful traces.
2. **The strongest claims are at the batch level.** There is no per-edit ablation for rounds 4 or 5, and successful executions do not consistently follow every promoted rule. “Promoted” is a recorded selection outcome, not a causal certificate.
3. **The final harness still contains incompatible child-output instructions.** S2 mixes label-count aggregation with complete user mappings. An observed global rerun returned label counts where user mappings were requested. This is a specific remaining problem, not evidence that adding more coverage text will solve it.
4. **Coverage, semantic correctness, and predicate correctness remain distinct.** User-ID presence cannot verify one label per question; per-user counts cannot verify record identity or label meaning; correct labels do not by themselves guarantee correct date, multiplicity, or asymmetric-role predicates.
5. **Extra checking spends resources.** The three promoted rounds used about 1.93×, 1.35×, and 1.49× their respective fresh baseline mean costs. Those ratios should not be multiplied into a causal end-to-end cost estimate, because baselines are newly executed and noisy. The instruction to rerun all chunks is a particular scaling concern for long inputs.
6. **F1 captures changes that exact match misses.** Rounds 6 and 8 contain useful partial-credit signals despite rejection; round 7 shows that five task-level F1 wins can still be outweighed by two larger losses. The per-task evidence matters alongside the average.

The final frozen harness was evaluated afresh as the baseline in rounds 6, 7, and 8. Its exact score varied **1/10, 1/10, 3/10**, and its all-attempt F1 varied **0.6194, 0.6355, 0.6128**. This variation, despite the fixed model configuration and unchanged harness, is a practical reason not to treat one extra exact pass on ten tasks as established generalization. Repeated use of the held-out set for selection also makes the independent final test important.

## Long-context comparison: separate and stopped

At the user's requested stop, the original harness had completed **48 attempts across 16 long-context tasks**, with three attempts per task: **0 exact passes, 45 resource/time terminations, three incorrect scored answers**, and at least **$30.0970** recorded cost. Mean recorded F1 was **0.0058** when failed runs counted as zero. Three additional in-flight baseline attempts were manually canceled and excluded from the matched comparison; their unrecorded spend is not zero.

The edited harness launched on those same 16 tasks with three workers and the same model, snapshot, limits, and three attempts per task. It was bounded to 48 attempts, but the user stopped it on September 23 at 15:28 UTC (10:28 a.m. Chicago), after **three completed edited attempts on one task**: two resource/time terminations and one malformed answer. The matching baseline attempts were all resource/time terminations. Both harnesses therefore have **0/3 exact passes and mean F1 0.0000** on the completed matched attempts. The edited condition recorded at least **$1.6737**, excluding unrecorded spend from three canceled in-flight attempts. All workers, the supervisor, and monitoring have stopped; completed artifacts are preserved. **This report makes no general long-context improvement claim from this one-task comparison.** Its current matched results and provenance are available in the [paired evaluation report](../../../experiment_oolong_pairs_dsv4f_20260916_131837/eval/paired_20260923T140326Z/comparison.md) and [launch/output guide](../../../experiment_oolong_pairs_dsv4f_20260916_131837/eval/paired_20260923T140326Z/README.md). These test results are not used here to create new harness edits.

## Artifact coverage and reproduction

- [audit.json](audit.json): all 16 materialized proposals, exact before/after surface values, source paths and hashes, 11 unmaterialized draft payloads, 14 proposal-response records, and all 160 held-out validation attempts paired within their rounds.
- [rounds.csv](rounds.csv): the numerical table above.
- [trace-evidence.json](trace-evidence.json): complete selected root operations and outputs, with original trace paths, hashes, and iteration/block coordinates.
- [runtime-retry-audit.json](runtime-retry-audit.json): recorded retry observations for the ten candidate traces in each of rounds 6 and 8. Metric-object counts include nested persisted representations and are not unique-call totals.
- [extract_audit.py](extract_audit.py): rebuilds the proposal inventory, metrics, and exact-edit appendices from saved artifacts without model calls. Run from the repository with `uv run python docs/analysis/oolong-pairs-2026-09-23/extract_audit.py`.

The initial harness hash is `0ca3510ded8a0ce5095c2e3338c4d8606fbc819650c20b65079b405ae1ecac2b`; the final hash is `1714e1cdf98fad7f700c3109d99cfb753d14e37a0eeb2e84712feed57fb10cc3`. Comparing their saved envelopes confirms that only S2 and S4 changed. This report is a research record for later paper updates; it does not modify the paper.

<!-- GENERATED APPENDICES -->

## Exact-edit appendices

The replacements below are complete, including unchanged material retained by the proposer. A proposal replaces its named surface on that round's incumbent; it is not an instruction to append the entire block. Each item links to its exact edit JSON and, for materialized candidates, a unified before/after diff of the stored surface. Those files and `audit.json` preserve whitespace and terminal newlines that may be hard to distinguish in Markdown.

Text edits use the recorded `literal-text/v1` contract: the displayed instruction is literal model-facing text. The host escapes braces once for template storage; stored-surface diffs therefore show doubled literal braces where needed. Code and policy payloads are shown separately according to their edit kind. No proposed instruction below has been corrected or rewritten for this report.

- [Appendix A: five promoted edits](#appendix-a-promoted-edits)
- [Appendix B: eleven tested, unpromoted edits](#appendix-b-tested-unpromoted-edits)
- [Appendix C: eleven unmaterialized drafts](#appendix-c-unmaterialized-drafts)

## Appendix A: Promoted edits

### A1. r03-c01-s4 — S4

Round 3; stored field `S4_verification_instruction`. Validation subject: `r03-c01-s4`. Outcome: promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `0ca3510ded8a0ce5095c2e3338c4d8606fbc819650c20b65079b405ae1ecac2b`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_03/proposals/r03-c01-s4/proposal.json) · [Exact edit JSON](edits/r03-c01-s4.edit.json) · [Stored-surface diff](edits/r03-c01-s4.patch)

Full replacement instruction:

````text
After merging all child results into a single label mapping, verify coverage:

1. Collect the set of all user IDs from the parsed input.
2. Collect the set of user IDs that have at least one label in the merged results.
3. If the second set is missing any user from the first set, print the missing user IDs and re-run the affected chunks with corrected prompts (e.g., instruct the child to return labels for every user in its chunk, not just qualifying ones).
4. Only after coverage is confirmed, proceed to compute the final answer.

This check catches cases where a child returned results keyed by line index instead of user ID, or where a user's questions were split across chunks and the merge lost some entries.
````

### A2. r04-c01-s4 — S4

Round 4; stored field `S4_verification_instruction`. Validation subject: `merged`. Outcome: promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `13efc32c8ac7aad100fef40cfa92922de567217597512e814d003bf22f5950d1`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_04/proposals/r04-c01-s4/proposal.json) · [Exact edit JSON](edits/r04-c01-s4.edit.json) · [Stored-surface diff](edits/r04-c01-s4.patch)

Full replacement instruction:

````text
After merging all child results into a single label mapping, verify coverage:

1. Collect the set of all user IDs from the parsed input.
2. Collect the set of user IDs that have at least one label in the merged results.
3. If the second set is missing any user from the first set, print the missing user IDs and re-run the affected chunks with corrected prompts (e.g., instruct the child to return labels for every user in its chunk, not just qualifying ones).
4. Only after coverage is confirmed, proceed to verify label accuracy.

To verify label accuracy, spot-check each child's work: for a random sample of 3-5 questions per child chunk, print the original question text alongside the label the child assigned. Verify that the label matches the category definitions (abbreviation, entity, human being, numeric value, location, description and abstract concept). If any sample is misclassified, re-run that chunk with a corrected prompt that includes the category definitions more explicitly and instructs the child to classify each question carefully.

5. Only after both coverage and label accuracy are confirmed, proceed to compute the final answer.

This check catches cases where a child returned results keyed by line index instead of user ID, where a user's questions were split across chunks and the merge lost some entries, or where the child systematically misclassified questions.
````

### A3. r04-c02-s2 — S2

Round 4; stored field `S2_decomposition_instruction`. Validation subject: `merged`. Outcome: promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `13efc32c8ac7aad100fef40cfa92922de567217597512e814d003bf22f5950d1`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_04/proposals/r04-c02-s2/proposal.json) · [Exact edit JSON](edits/r04-c02-s2.edit.json) · [Stored-surface diff](edits/r04-c02-s2.patch)

Full replacement instruction:

````text
Choosing the sub-call for each sub-task:

- `llm_query` is a single bare completion with no REPL. Use it only when the
  sub-answer is a direct read of the text you hand it: extract a field, label
  ONE item, summarize ONE passage, answer a question a single visible passage
  settles.
- `rlm_query` is a child RLM with a REPL. Use it whenever the sub-task must
  COMPUTE over many items - count, tally frequencies, compare two counts,
  find the most/least common label, filter-then-count, anything phrased as
  "how many", "which is more common", or "the total number of". A bare
  completion asked to count hundreds of items counts in its head and is wrong
  past a few dozen; a child RLM labels items in a loop and counts in Python.
- For counting tasks, the chunk you hand a child is sized by what the child can
  enumerate exactly, not by context capacity: a few hundred lines per child is
  right, and a child returning a small JSON of counts is the ideal unit. Do not
  pack a counting chunk to the flat-call capacity ceiling.

Pattern for label statistics over a long line-oriented `context`
(split into chunks, have each child classify every line and return counts,
merge in Python):

```repl
import json
lines = [ln for ln in context.split("\n") if ln.strip()]
CHUNK = 300
chunks = [lines[i:i + CHUNK] for i in range(0, len(lines), CHUNK)]
labels = ["positive", "negative"]  # the label set stated in the task
prompts = [
    "Classify EVERY line below into exactly one of " + json.dumps(labels)
    + ". Do it line by line in your REPL, keep a running count per label, "
    + "and finish by setting answer['content'] to a JSON object mapping each "
    + "label to its count (integers, no other keys).\n\n"
    + "IMPORTANT: You must classify EVERY line in your chunk. Do not skip any line. "
    + "If a line's user ID appears in multiple chunks, still classify it. "
    + "Return a complete mapping for ALL users in your chunk.\n\n"
    + "\n".join(chunk)
    for chunk in chunks
]
results = rlm_query_batched(prompts)
totals = {label: 0 for label in labels}
for raw in results:
    for label, n in json.loads(raw).items():
        totals[label] += int(n)
print(totals)
```

Then decide the answer from `totals` in Python and print it before setting
`answer["ready"] = True`. If a child's result does not parse, print it, fix the
prompt, and re-run that chunk - do not fall back to counting by eye.
````

### A4. r05-c01-s2 — S2

Round 5; stored field `S2_decomposition_instruction`. Validation subject: `merged`. Outcome: promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `50de298a8fa910be808cd85f1ca83a00d72c9f1a91eb39d9568e863fba2a380e`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_05/proposals/r05-c01-s2/proposal.json) · [Exact edit JSON](edits/r05-c01-s2.edit.json) · [Stored-surface diff](edits/r05-c01-s2.patch)

Full replacement instruction:

````text
Choosing the sub-call for each sub-task:

- `llm_query` is a single bare completion with no REPL. Use it only when the
  sub-answer is a direct read of the text you hand it: extract a field, label
  ONE item, summarize ONE passage, answer a question a single visible passage
  settles.
- `rlm_query` is a child RLM with a REPL. Use it whenever the sub-task must
  COMPUTE over many items - count, tally frequencies, compare two counts,
  find the most/least common label, filter-then-count, anything phrased as
  "how many", "which is more common", or "the total number of". A bare
  completion asked to count hundreds of items counts in its head and is wrong
  past a few dozen; a child RLM labels items in a loop and counts in Python.
- For counting tasks, the chunk you hand a child is sized by what the child can
  enumerate exactly, not by context capacity: a few hundred lines per child is
  right, and a child returning a small JSON of counts is the ideal unit. Do not
  pack a counting chunk to the flat-call capacity ceiling.

Pattern for label statistics over a long line-oriented `context`
(split into chunks, have each child classify every line and return counts,
merge in Python):

```repl
import json
lines = [ln for ln in context.split("\n") if ln.strip()]
CHUNK = 300
chunks = [lines[i:i + CHUNK] for i in range(0, len(lines), CHUNK)]
labels = ["positive", "negative"]  # the label set stated in the task
prompts = [
    "Classify EVERY line below into exactly one of " + json.dumps(labels)
    + ". Do it line by line in your REPL, keep a running count per label, "
    + "and finish by setting answer['content'] to a JSON object mapping each "
    + "label to its count (integers, no other keys).\n\n"
    + "IMPORTANT: You must classify EVERY line in your chunk. Do not skip any line. "
    + "If a line's user ID appears in multiple chunks, still classify it. "
    + "Return a complete mapping for ALL users in your chunk.\n\n"
    + "\n".join(chunk)
    for chunk in chunks
]
results = rlm_query_batched(prompts)
totals = {label: 0 for label in labels}
for raw in results:
    for label, n in json.loads(raw).items():
        totals[label] += int(n)
print(totals)
```

Then decide the answer from `totals` in Python and print it before setting
`answer["ready"] = True`. If a child's result does not parse, print it, fix the
prompt, and re-run that chunk - do not fall back to counting by eye.

IMPORTANT: When instructing children to classify items, explicitly tell them to
return a complete mapping for ALL items in their chunk, not just those that
match a certain criterion. Children may return results keyed by line index
instead of user ID; verify the key format before merging. If a child's result
uses line indices, re-run with a corrected prompt specifying the correct key.
````

### A5. r05-c02-s4 — S4

Round 5; stored field `S4_verification_instruction`. Validation subject: `merged`. Outcome: promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `50de298a8fa910be808cd85f1ca83a00d72c9f1a91eb39d9568e863fba2a380e`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_05/proposals/r05-c02-s4/proposal.json) · [Exact edit JSON](edits/r05-c02-s4.edit.json) · [Stored-surface diff](edits/r05-c02-s4.patch)

Full replacement instruction:

````text
After merging all child results into a single label mapping, verify coverage:

1. Collect the set of all user IDs from the parsed input.
2. Collect the set of user IDs that have at least one label in the merged results.
3. If the second set is missing any user from the first set, print the missing user IDs and re-run the affected chunks with corrected prompts (e.g., instruct the child to return labels for every user in its chunk, not just qualifying ones).
4. Only after coverage is confirmed, proceed to verify label accuracy.

To verify label accuracy, spot-check each child's work: for a random sample of 3-5 questions per child chunk, print the original question text alongside the label the child assigned. Verify that the label matches the category definitions (abbreviation, entity, human being, numeric value, location, description and abstract concept). If any sample is misclassified, re-run that chunk with a corrected prompt that includes the category definitions more explicitly and instructs the child to classify each question carefully.

IMPORTANT: If you re-run ANY chunk with a corrected prompt, you MUST re-run ALL other chunks with the same corrected prompt as well. Do not mix old and new classifications from different prompt versions. After re-running all chunks, re-merge and re-verify coverage and accuracy from scratch.

5. Only after both coverage and label accuracy are confirmed, proceed to compute the final answer.

This check catches cases where a child returned results keyed by line index instead of user ID, where a user's questions were split across chunks and the merge lost some entries, or where the child systematically misclassified questions.
````

## Appendix B: Tested, unpromoted edits

### B1. r01-c01-s4 — S4

Round 1; stored field `S4_verification_instruction`. Validation subject: `merged`. Outcome: not promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `0ca3510ded8a0ce5095c2e3338c4d8606fbc819650c20b65079b405ae1ecac2b`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_01/proposals/r01-c01-s4/proposal.json) · [Exact edit JSON](edits/r01-c01-s4.edit.json) · [Stored-surface diff](edits/r01-c01-s4.patch)

Full replacement instruction:

````text
After merging all child results into a single user-to-labels mapping, verify that every user from the parsed data is represented. Count distinct users in the parsed data and distinct keys in the merged mapping. If the counts differ, print the missing user IDs, identify which chunk(s) should have covered them, fix the prompt (e.g., ensure the child returns results keyed by user ID, not by line index), and re-run those chunks. Do not proceed to pair generation until every parsed user has at least one label entry.
````

### B2. r01-c02-s9 — S9

Round 1; stored field `S9_answer_middleware`. Validation subject: `merged`. Outcome: not promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `0ca3510ded8a0ce5095c2e3338c4d8606fbc819650c20b65079b405ae1ecac2b`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_01/proposals/r01-c02-s9/proposal.json) · [Exact edit JSON](edits/r01-c02-s9.edit.json) · [Stored-surface diff](edits/r01-c02-s9.patch)

Edit metadata:

````json
{
  "def_name": "accept_answer",
  "kind": "code"
}
````

Complete function source:

````python
def accept_answer(answer: str, repl_inventory: dict[str, tuple[str, int]]) -> AnswerDecision:
    """Accept detected answer; normalize pair format if needed."""
    import re
    # If answer is the explicit empty marker, accept unchanged
    if answer.strip() == "No valid pairs found.":
        return AnswerDecision.accept(answer)
    # Try to detect pairs in various formats and normalize
    # Pattern: (num, num) already canonical
    canonical = re.findall(r'\((\d+),\s*(\d+)\)', answer)
    if canonical:
        # Already has canonical pairs; accept as-is
        return AnswerDecision.accept(answer)
    # Pattern: num, num on separate lines or comma-separated
    lines = [l.strip() for l in answer.strip().split('\n') if l.strip()]
    pairs = []
    for line in lines:
        # Try "num, num"
        m = re.match(r'(\d+),\s*(\d+)', line)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            pairs.append(f"({min(a,b)}, {max(a,b)})")
            continue
        # Try "num num" (space separated)
        m = re.match(r'(\d+)\s+(\d+)', line)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            pairs.append(f"({min(a,b)}, {max(a,b)})")
            continue
    if pairs:
        normalized = '\n'.join(pairs)
        return AnswerDecision.redirect(normalized)
    # No recognizable pairs; accept as-is (may be empty or other format)
    return AnswerDecision.accept(answer)
````

### B3. r02-c01-s2 — S2

Round 2; stored field `S2_decomposition_instruction`. Validation subject: `merged`. Outcome: not promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `0ca3510ded8a0ce5095c2e3338c4d8606fbc819650c20b65079b405ae1ecac2b`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_02/proposals/r02-c01-s2/proposal.json) · [Exact edit JSON](edits/r02-c01-s2.edit.json) · [Stored-surface diff](edits/r02-c01-s2.patch)

Full replacement instruction:

````text
Choosing the sub-call for each sub-task:

- `llm_query` is a single bare completion with no REPL. Use it only when the
  sub-answer is a direct read of the text you hand it: extract a field, label
  ONE item, summarize ONE passage, answer a question a single visible passage
  settles.
- `rlm_query` is a child RLM with a REPL. Use it whenever the sub-task must
  COMPUTE over many items - count, tally frequencies, compare two counts,
  find the most/least common label, filter-then-count, anything phrased as
  "how many", "which is more common", or "the total number of". A bare
  completion asked to count hundreds of items counts in its head and is wrong
  past a few dozen; a child RLM labels items in a loop and counts in Python.
- For counting tasks, the chunk you hand a child is sized by what the child can
  enumerate exactly, not by context capacity: a few hundred lines per child is
  right, and a child returning a small JSON of counts is the ideal unit. Do not
  pack a counting chunk to the flat-call capacity ceiling.

Pattern for label statistics over a long line-oriented `context`
(split into chunks, have each child classify every line and return counts,
merge in Python):

```repl
import json
lines = [ln for ln in context.split("\n") if ln.strip()]
CHUNK = 300
chunks = [lines[i:i + CHUNK] for i in range(0, len(lines), CHUNK)]
labels = ["positive", "negative"]  # the label set stated in the task
prompts = [
    "Classify EVERY line below into exactly one of " + json.dumps(labels)
    + ". Do it line by line in your REPL, keep a running count per label, "
    + "and finish by setting answer['content'] to a JSON object mapping each "
    + "label to its count (integers, no other keys).\n\n" + "\n".join(chunk)
    for chunk in chunks
]
results = rlm_query_batched(prompts)
totals = {label: 0 for label in labels}
for raw in results:
    for label, n in json.loads(raw).items():
        totals[label] += int(n)
print(totals)
```

Then decide the answer from `totals` in Python and print it before setting
`answer["ready"] = True`. If a child's result does not parse, print it, fix the
prompt, and re-run that chunk - do not fall back to counting by eye.

**Coverage check after merging**: After merging results from all chunks, verify
that every input item is represented. For example, if the input is a list of
records keyed by user ID, check that the number of distinct users in the merged
result equals the number of distinct users in the original data. If there is a
mismatch, print the missing IDs, identify which chunks should have covered them,
and re-run those chunks with corrected prompts before proceeding.
````

### B4. r02-c02-s4 — S4

Round 2; stored field `S4_verification_instruction`. Validation subject: `merged`. Outcome: not promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `0ca3510ded8a0ce5095c2e3338c4d8606fbc819650c20b65079b405ae1ecac2b`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_02/proposals/r02-c02-s4/proposal.json) · [Exact edit JSON](edits/r02-c02-s4.edit.json) · [Stored-surface diff](edits/r02-c02-s4.patch)

Full replacement instruction:

````text
After merging results from all child calls, verify that every input item is represented in the merged data. For example, if the input is a list of records keyed by user ID, check that the number of distinct users in the merged result equals the number of distinct users in the original parsed data. If there is a mismatch, print the missing IDs, identify which chunks should have covered them, and re-run those chunks with corrected prompts before proceeding to generate the final answer.
````

### B5. r02-c03-s3 — S3

Round 2; stored field `S3_execution_instruction`. Validation subject: `merged`. Outcome: not promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `0ca3510ded8a0ce5095c2e3338c4d8606fbc819650c20b65079b405ae1ecac2b`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_02/proposals/r02-c03-s3/proposal.json) · [Exact edit JSON](edits/r02-c03-s3.edit.json) · [Stored-surface diff](edits/r02-c03-s3.patch)

Full replacement instruction:

````text
When printing the final answer, output each pair on its own line in the format (user_id_1, user_id_2) with the lower ID first. Do not include any additional text, explanations, or summary after the pair list. The last line of output must be a pair in this format. If there are no valid pairs, print exactly "No valid pairs found." as the final output.
````

### B6. r06-c01-s6 — S6

Round 6; stored field `S6_runtime_policy`. Validation subject: `r06-c01-s6`. Outcome: not promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `1714e1cdf98fad7f700c3109d99cfb753d14e37a0eeb2e84712feed57fb10cc3`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_06/proposals/r06-c01-s6/proposal.json) · [Exact edit JSON](edits/r06-c01-s6.edit.json) · [Stored-surface diff](edits/r06-c01-s6.patch)

Complete policy edit, including unset fields:

````json
{
  "kind": "policy",
  "runtime_policy": {
    "enabled": true,
    "max_batch_width": null,
    "max_depth": null,
    "max_prompt_chars": null,
    "max_retries": 2,
    "retry_on_syntax_error": true,
    "validate_sub_output": null
  }
}
````

### B7. r07-c01-s4 — S4

Round 7; stored field `S4_verification_instruction`. Validation subject: `merged`. Outcome: not promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `1714e1cdf98fad7f700c3109d99cfb753d14e37a0eeb2e84712feed57fb10cc3`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_07/proposals/r07-c01-s4/proposal.json) · [Exact edit JSON](edits/r07-c01-s4.edit.json) · [Stored-surface diff](edits/r07-c01-s4.patch)

Full replacement instruction:

````text
After merging all child results into a single label mapping, verify coverage:

1. Collect the set of all user IDs from the parsed input.
2. Collect the set of user IDs that have at least one label in the merged results.
3. If the second set is missing any user from the first set, print the missing user IDs and re-run the affected chunks with corrected prompts (e.g., instruct the child to return labels for every user in its chunk, not just qualifying ones).
4. Only after coverage is confirmed, proceed to verify per-question label completeness.

To verify per-question label completeness: for each user, compare the number of labels assigned to that user against the number of questions that user has in the original data. If any user has fewer labels than questions, that user's questions were likely overwritten during merge because the child returned per-user aggregates instead of per-question labels. Print the affected user IDs and re-run the affected chunks with corrected prompts that instruct the child to return one label per question (e.g., a list of {question_index, label} pairs or a list of labels in order).

To verify label accuracy, spot-check each child's work: for a random sample of 3-5 questions per child chunk, print the original question text alongside the label the child assigned. Verify that the label matches the category definitions (abbreviation, entity, human being, numeric value, location, description and abstract concept). If any sample is misclassified, re-run that chunk with a corrected prompt that includes the category definitions more explicitly and instructs the child to classify each question carefully.

IMPORTANT: If you re-run ANY chunk with a corrected prompt, you MUST re-run ALL other chunks with the same corrected prompt as well. Do not mix old and new classifications from different prompt versions. After re-running all chunks, re-merge and re-verify coverage and accuracy from scratch.

5. Only after both coverage and label accuracy are confirmed, proceed to compute the final answer.

This check catches cases where a child returned results keyed by line index instead of user ID, where a user's questions were split across chunks and the merge lost some entries, where the child systematically misclassified questions, or where the child returned per-user aggregates instead of per-question labels.
````

### B8. r07-c03-s3 — S3

Round 7; stored field `S3_execution_instruction`. Validation subject: `merged`. Outcome: not promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `1714e1cdf98fad7f700c3109d99cfb753d14e37a0eeb2e84712feed57fb10cc3`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_07/proposals/r07-c03-s3/proposal.json) · [Exact edit JSON](edits/r07-c03-s3.edit.json) · [Stored-surface diff](edits/r07-c03-s3.patch)

Full replacement instruction:

````text
When instructing children to classify items, explicitly tell them to return one label per question, not one label per user. A user with multiple questions should have multiple labels in the result. The child can return a list of {question_index, label} pairs, a list of labels in the same order as the questions in the chunk, or a JSON object mapping user_id to a list of labels (one per question in order).

After receiving child results, merge by appending each question's label to the user's label list, not by overwriting. If a child returns a per-user aggregate (one label per user), the merge will lose labels for users with multiple questions. To detect this, verify that the number of labels per user equals the number of questions that user has in the chunk.

Example of instructing a child to return per-question labels:

```repl
prompt = "Classify EACH question below. Return a JSON list of {question_index, label} pairs, one per question, in the same order as the questions. Do not group by user.\n\n" + "\n".join(chunk)
```
````

### B9. r08-c01-s6 — S6

Round 8; stored field `S6_runtime_policy`. Validation subject: `merged`. Outcome: not promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `1714e1cdf98fad7f700c3109d99cfb753d14e37a0eeb2e84712feed57fb10cc3`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_08/proposals/r08-c01-s6/proposal.json) · [Exact edit JSON](edits/r08-c01-s6.edit.json) · [Stored-surface diff](edits/r08-c01-s6.patch)

Complete policy edit, including unset fields:

````json
{
  "kind": "policy",
  "runtime_policy": {
    "enabled": true,
    "max_batch_width": null,
    "max_depth": null,
    "max_prompt_chars": null,
    "max_retries": 2,
    "retry_on_syntax_error": true,
    "validate_sub_output": null
  }
}
````

### B10. r08-c02-s4 — S4

Round 8; stored field `S4_verification_instruction`. Validation subject: `merged`. Outcome: not promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `1714e1cdf98fad7f700c3109d99cfb753d14e37a0eeb2e84712feed57fb10cc3`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_08/proposals/r08-c02-s4/proposal.json) · [Exact edit JSON](edits/r08-c02-s4.edit.json) · [Stored-surface diff](edits/r08-c02-s4.patch)

Full replacement instruction:

````text
After merging all child results into a single label mapping, verify coverage:

1. Collect the set of all user IDs from the parsed input.
2. Collect the set of user IDs that have at least one label in the merged results.
3. If the second set is missing any user from the first set, print the missing user IDs and re-run the affected chunks with corrected prompts (e.g., instruct the child to return labels for every user in its chunk, not just qualifying ones).
4. Only after coverage is confirmed, proceed to verify label accuracy.

IMPORTANT: Verify that the number of parsed records matches the total number of data-bearing lines in the original input. If the parsed count is less than the total input lines, print the discrepancy and re-parse the original input to find the missing lines. Do not assume that the first N lines of context are the only data lines - headers, blank lines, or instructions may precede the actual data.

To verify label accuracy, spot-check each child's work: for a random sample of 3-5 questions per child chunk, print the original question text alongside the label the child assigned. Verify that the label matches the category definitions (abbreviation, entity, human being, numeric value, location, description and abstract concept). If any sample is misclassified, re-run that chunk with a corrected prompt that includes the category definitions more explicitly and instructs the child to classify each question carefully.

IMPORTANT: If you re-run ANY chunk with a corrected prompt, you MUST re-run ALL other chunks with the same corrected prompt as well. Do not mix old and new classifications from different prompt versions. After re-running all chunks, re-merge and re-verify coverage and accuracy from scratch.

5. Only after both coverage and label accuracy are confirmed, proceed to compute the final answer.

This check catches cases where a child returned results keyed by line index instead of user ID, where a user's questions were split across chunks and the merge lost some entries, where the child systematically misclassified questions, or where the root's initial parsing of context skipped data-bearing lines.
````

### B11. r08-c03-s3 — S3

Round 8; stored field `S3_execution_instruction`. Validation subject: `merged`. Outcome: not promoted. The validation score belongs to this subject, including its siblings when bundled.

Base harness: `1714e1cdf98fad7f700c3109d99cfb753d14e37a0eeb2e84712feed57fb10cc3`.

[Saved proposal](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_08/proposals/r08-c03-s3/proposal.json) · [Exact edit JSON](edits/r08-c03-s3.edit.json) · [Stored-surface diff](edits/r08-c03-s3.patch)

Full replacement instruction:

````text
When you have computed the final set of qualifying pairs, output them in the canonical format: one pair per line as (user_id_1, user_id_2), with lower user ID first. Do not add any extra text, separators, or formatting after the pair list. If no pairs qualify, output exactly: No valid pairs found.

Example of correct output:
(10149, 13574)
(10149, 14630)
(13574, 14630)

Do not use commas, spaces, or brackets as separators between pairs. Each pair must be on its own line with parentheses and a comma between the two IDs.
````

## Appendix C: Unmaterialized drafts

Each distinct edit payload is listed once per round. Repeated submissions of the same payload are listed together, even if the explanation changed. Violations below are the saved response-level messages and can mention sibling candidates. These drafts were never run through held-out validation.

### C1. Round 2 — S9

[Saved responses](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_02/proposals_complete.json) · [Exact edit JSON](edits/C01-r02-s9.edit.json)

Submission attempt 1, candidate position 3. Saved response violation:

````text
pattern 5 S9 proposal: pattern_index 5: surface 'S9' is not eligible for mechanism 'incomplete_coverage'; choose one of ['S2', 'S4', 'S3', 'S10']
````

Edit metadata:

````json
{
  "kind": "code"
}
````

Complete function source:

````python
def accept_answer(answer: str, repl_inventory: dict[str, tuple[str, int]]) -> AnswerDecision:
    """Accept valid answers; normalize pair format if needed."""
    import re
    
    # Accept empty marker as-is
    if answer.strip() == "No valid pairs found.":
        return AnswerDecision.accept(answer)
    
    # Check if answer already has canonical format: lines like (a, b)
    canonical_pattern = re.compile(r'\(\s*\d+\s*,\s*\d+\s*\)')
    if canonical_pattern.search(answer):
        return AnswerDecision.accept(answer)
    
    # Try to detect pairs in other formats and normalize
    # Format: a,b or a b (one pair per line)
    line_pairs = []
    for line in answer.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Try comma-separated integers
        nums = re.findall(r'\d+', line)
        if len(nums) == 2:
            line_pairs.append((int(nums[0]), int(nums[1])))
    
    if line_pairs:
        # Normalize: lower ID first, sort by first then second
        normalized = []
        for a, b in line_pairs:
            normalized.append((min(a, b), max(a, b)))
        normalized.sort()
        result = '\n'.join(f'({a}, {b})' for a, b in normalized)
        return AnswerDecision.accept(result)
    
    # If no pairs detected, accept as-is
    return AnswerDecision.accept(answer)
````

### C2. Round 2 — S2

[Saved responses](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_02/proposals_complete.json) · [Exact edit JSON](edits/C02-r02-s2.edit.json)

Submission attempt 2, candidate position 1. Saved response violation:

````text
pattern 0 S2 proposal: selection reason must contain 1-600 characters; pattern 1 S4 proposal: repair cannot replace an occupied pattern or surface
````

Full replacement instruction:

````text
Choosing the sub-call for each sub-task:

- `llm_query` is a single bare completion with no REPL. Use it only when the
  sub-answer is a direct read of the text you hand it: extract a field, label
  ONE item, summarize ONE passage, answer a question a single visible passage
  settles.
- `rlm_query` is a child RLM with a REPL. Use it whenever the sub-task must
  COMPUTE over many items - count, tally frequencies, compare two counts,
  find the most/least common label, filter-then-count, anything phrased as
  "how many", "which is more common", or "the total number of". A bare
  completion asked to count hundreds of items counts in its head and is wrong
  past a few dozen; a child RLM labels items in a loop and counts in Python.
- For counting tasks, the chunk you hand a child is sized by what the child can
  enumerate exactly, not by context capacity: a few hundred lines per child is
  right, and a child returning a small JSON of counts is the ideal unit. Do not
  pack a counting chunk to the flat-call capacity ceiling.

Pattern for label statistics over a long line-oriented `context`
(split into chunks, have each child classify EVERY line and return a complete
per-user label mapping, merge in Python):

```repl
import json
lines = [ln for ln in context.split("\n") if ln.strip()]
CHUNK = 300
chunks = [lines[i:i + CHUNK] for i in range(0, len(lines), CHUNK)]
labels = ["positive", "negative"]  # the label set stated in the task
prompts = [
    "Classify EVERY line below into exactly one of " + json.dumps(labels)
    + ". Do it line by line in your REPL, keep a running count per label, "
    + "and finish by setting answer['content'] to a JSON object mapping each "
    + "label to its count (integers, no other keys).\n\n" + "\n".join(chunk)
    for chunk in chunks
]
results = rlm_query_batched(prompts)
totals = {label: 0 for label in labels}
for raw in results:
    for label, n in json.loads(raw).items():
        totals[label] += int(n)
print(totals)
```

When the task requires identifying which items (e.g., users) satisfy a condition
based on their labels, have each child return a COMPLETE mapping from each item
to its label(s), not just the subset that match. The root then evaluates the
condition on the merged mapping. This ensures no item is missed because its
label was never reported.

Then decide the answer from `totals` in Python and print it before setting
`answer["ready"] = True`. If a child's result does not parse, print it, fix the
prompt, and re-run that chunk - do not fall back to counting by eye.
````

### C3. Round 2 — S4

[Saved responses](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_02/proposals_complete.json) · [Exact edit JSON](edits/C03-r02-s4.edit.json)

Submission attempt 2, candidate position 2. Saved response violation:

````text
pattern 0 S2 proposal: selection reason must contain 1-600 characters; pattern 1 S4 proposal: repair cannot replace an occupied pattern or surface
````

Full replacement instruction:

````text
After merging results from all child calls, verify that every item from the original data is represented in the merged result. For user-pair tasks: check that the set of users with labels equals the set of all users in the parsed data. If there is a mismatch, print the missing user IDs, identify which chunk(s) should have covered them, and re-run those chunks with corrected prompts. Do not proceed to compute the final answer until coverage is complete.
````

### C4. Round 3 — S2

[Saved responses](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_03/proposals_complete.json) · [Exact edit JSON](edits/C04-r03-s2.edit.json)

Submission attempt 1, candidate position 2. Saved response violation:

````text
pattern 8 S2 proposal: pattern_index 8: surface 'S2' is not eligible for mechanism 'iteration_budget_exhaustion'; choose one of ['S6', 'S3', 'S1', 'S10']; pattern 9 S5 proposal: pattern_index 9: surface 'S5' is not eligible for mechanism 'iteration_budget_exhaustion'; choose one of ['S6', 'S3', 'S1', 'S10']
````

Full replacement instruction:

````text
Choosing the sub-call for each sub-task:

- `llm_query` is a single bare completion with no REPL. Use it only when the
  sub-answer is a direct read of the text you hand it: extract a field, label
  ONE item, summarize ONE passage, answer a question a single visible passage
  settles.
- `rlm_query` is a child RLM with a REPL. Use it whenever the sub-task must
  COMPUTE over many items - count, tally frequencies, compare two counts,
  find the most/least common label, filter-then-count, anything phrased as
  "how many", "which is more common", or "the total number of". A bare
  completion asked to count hundreds of items counts in its head and is wrong
  past a few dozen; a child RLM labels items in a loop and counts in Python.
- For counting tasks, the chunk you hand a child is sized by what the child can
  enumerate exactly, not by context capacity: a few hundred lines per child is
  right, and a child returning a small JSON of counts is the ideal unit. Do not
  pack a counting chunk to the flat-call capacity ceiling.
- Limit concurrent RLM sub-calls: launch at most 10-15 per batch. If you have
  many chunks, process them in smaller batches and accumulate results
  incrementally. This prevents launching too many expensive sub-calls that
  collectively exhaust the time budget.

Pattern for label statistics over a long line-oriented `context`
(split into chunks, have each child classify every line and return counts,
merge in Python):

```repl
import json
lines = [ln for ln in context.split("\n") if ln.strip()]
CHUNK = 300
chunks = [lines[i:i + CHUNK] for i in range(0, len(lines), CHUNK)]
labels = ["positive", "negative"]  # the label set stated in the task
prompts = [
    "Classify EVERY line below into exactly one of " + json.dumps(labels)
    + ". Do it line by line in your REPL, keep a running count per label, "
    + "and finish by setting answer['content'] to a JSON object mapping each "
    + "label to its count (integers, no other keys).\n\n" + "\n".join(chunk)
    for chunk in chunks
]
results = rlm_query_batched(prompts)
totals = {label: 0 for label in labels}
for raw in results:
    for label, n in json.loads(raw).items():
        totals[label] += int(n)
print(totals)
```

Then decide the answer from `totals` in Python and print it before setting
`answer["ready"] = True`. If a child's result does not parse, print it, fix the
prompt, and re-run that chunk - do not fall back to counting by eye.
````

### C5. Round 3 — S5

[Saved responses](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_03/proposals_complete.json) · [Exact edit JSON](edits/C05-r03-s5.edit.json)

Submission attempt 1, candidate position 3. Saved response violation:

````text
pattern 8 S2 proposal: pattern_index 8: surface 'S2' is not eligible for mechanism 'iteration_budget_exhaustion'; choose one of ['S6', 'S3', 'S1', 'S10']; pattern 9 S5 proposal: pattern_index 9: surface 'S5' is not eligible for mechanism 'iteration_budget_exhaustion'; choose one of ['S6', 'S3', 'S1', 'S10']
````

Full replacement instruction:

````text
When a child RLM sub-call returns an error (timeout, parse failure, or unusable result):

1. Print the error message and the chunk that was sent to the child.
2. Split that chunk into smaller pieces (e.g., half the original size, or at most 150 lines per piece).
3. Retry each piece as a separate RLM sub-call.
4. If a piece still errors, split further and retry again (at most 2 retries per piece).
5. If all retries fail, print the error and the piece content, then continue with the results you have — do not let a single failed sub-call block the entire run.

Do not fall back to manual classification or guessing. The goal is to recover gracefully and continue accumulating results.
````

### C6. Round 3 — S10

[Saved responses](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_03/proposals_complete.json) · [Exact edit JSON](edits/C06-r03-s10.edit.json)

Submission attempt 2, candidate position 1. Saved response violation:

````text
surface S10 was selected more than once; surface S10 was selected more than once
````

Complete skill metadata:

````json
{
  "kind": "skills",
  "name": "batch_rlm_calls",
  "description": "Use when you have many RLM sub-calls to launch; limits batch size and accumulates results incrementally."
}
````

Complete skill body:

````markdown
## Use When
- You have more than 10 items to process via rlm_query_batched
- Each item requires a separate RLM sub-call (e.g., classifying per-user questions)

## Don't Use When
- You have fewer than 10 items (just launch them all at once)
- Items can be combined into a single sub-call (e.g., all lines in one chunk)

## Steps
1. Split your items into batches of at most 10.
2. For each batch, launch rlm_query_batched and collect results.
3. Process results immediately: parse, validate, and accumulate into your data structure.
4. Print a progress summary after each batch (e.g., "Batch X done: Y items processed, Z remaining").
5. If a result is an error or unparseable, print the error and the item index, then continue with the next batch.
6. After all batches, verify you have results for all items before proceeding.

## Pitfalls
- Launching all sub-calls at once can exhaust the time budget before any complete.
- Processing results incrementally lets you detect issues early and adjust.
- Do not skip validation: a failed sub-call produces no result, and you must retry or note the gap.

## Verify
- After all batches, assert that the number of results equals the number of items.
- If any item is missing, print its index and retry it individually with a smaller chunk.
````

### C7. Round 3 — S10

[Saved responses](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_03/proposals_complete.json) · [Exact edit JSON](edits/C07-r03-s10.edit.json)

Submission attempt 2, candidate position 2. Saved response violation:

````text
surface S10 was selected more than once; surface S10 was selected more than once
````

Complete skill metadata:

````json
{
  "kind": "skills",
  "name": "recover_rlm_error",
  "description": "Use when a child RLM sub-call returns an error; splits the chunk and retries each piece separately."
}
````

Complete skill body:

````markdown
## Use When
- A child RLM sub-call returns an error (timeout, parse failure, or unusable result)
- The chunk sent to the child was large (more than 50 items or 200 lines)

## Don't Use When
- The chunk was already small (fewer than 10 items) — retry once as-is, then skip
- The error is a transient API error (rate limit, etc.) — the harness retries those automatically

## Steps
1. Print the error message and the chunk that was sent to the child.
2. Split that chunk into smaller pieces: at most half the original size, or at most 50 items per piece.
3. Retry each piece as a separate RLM sub-call.
4. If a piece still errors, split further (at most 25 items) and retry again.
5. If all retries fail for a piece, print the error and the piece content, then continue with the results you have.
6. Do not let a single failed sub-call block the entire run — accumulate whatever results you got and proceed.

## Pitfalls
- A timeout on a large chunk often means the child had too many items to process.
- Splitting reduces each child's workload and increases the chance of completion.
- Do not retry indefinitely: at most 2 retries per piece.
- If you skip a piece, note which items were lost so you can report partial results.

## Verify
- After retries, check that you have results for all pieces that succeeded.
- Print a summary: "X pieces succeeded, Y pieces failed after retries."
````

### C8. Round 4 — S9

[Saved responses](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_04/proposals_complete.json) · [Exact edit JSON](edits/C08-r04-s9.edit.json)

Submission attempt 1, candidate position 3. Saved response violation:

````text
pattern 0 S4 proposal: selection reason must contain 1-600 characters; pattern 1 S2 proposal: selection reason must contain 1-600 characters; pattern 9 S9 proposal: selection reason must contain 1-600 characters
````

Submission attempt 2, candidate position 3. Saved response violation:

````text
pattern 9 S9 proposal: pattern_index 9: surface 'S9' is not eligible for mechanism 'incomplete_coverage'; choose one of ['S2', 'S4', 'S3', 'S10']
````

Edit metadata:

````json
{
  "kind": "code"
}
````

Complete function source:

````python
def answer_middleware(answer, repl_inventory):
    import re
    
    # Accept the empty marker unchanged
    if answer.strip() == "No valid pairs found.":
        return AnswerDecision.accept(answer)
    
    # Check if answer already contains canonical (a, b) pairs
    canonical_pattern = re.compile(r'\(\s*\d+\s*,\s*\d+\s*\)')
    if canonical_pattern.search(answer):
        return AnswerDecision.accept(answer)
    
    # Try to extract pairs from non-canonical formats
    # Format: comma-separated integers like "16357, 18063"
    comma_pair_pattern = re.compile(r'(\d+)\s*,\s*(\d+)')
    comma_matches = comma_pair_pattern.findall(answer)
    if comma_matches:
        pairs = []
        for a, b in comma_matches:
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            pairs.append(f"({a}, {b})")
        if pairs:
            return AnswerDecision.redirect("\n".join(pairs))
    
    # Format: space-separated integers like "16357 18063"
    space_pair_pattern = re.compile(r'(\d+)\s+(\d+)')
    space_matches = space_pair_pattern.findall(answer)
    if space_matches:
        pairs = []
        for a, b in space_matches:
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            pairs.append(f"({a}, {b})")
        if pairs:
            return AnswerDecision.redirect("\n".join(pairs))
    
    # Accept as-is if nothing matched (e.g., empty or unrecognizable)
    return AnswerDecision.accept(answer)
````

### C9. Round 7 — S6

[Saved responses](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_07/proposals_complete.json) · [Exact edit JSON](edits/C09-r07-s6.edit.json)

Submission attempt 1, candidate position 2. Saved response violation:

````text
pattern 3 S6 proposal: selection reason must contain 1-600 characters
````

Submission attempt 2, candidate position 1. Saved response violation:

````text
selection reason must contain 1-600 characters
````

Complete policy edit, including unset fields:

````json
{
  "kind": "policy",
  "runtime_policy": {
    "enabled": true,
    "max_batch_width": null,
    "max_depth": null,
    "max_prompt_chars": null,
    "max_retries": 1,
    "retry_on_syntax_error": false,
    "validate_sub_output": null
  }
}
````

### C10. Round 8 — S2

[Saved responses](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_08/proposals_complete.json) · [Exact edit JSON](edits/C10-r08-s2.edit.json)

Submission attempt 1, candidate position 1. Saved response violation:

````text
pattern 1 S2 proposal: pattern_index 1: surface 'S2' is not eligible for mechanism 'iteration_budget_exhaustion'; choose one of ['S6', 'S3', 'S1', 'S10']; pattern 0 S1 proposal: pattern_index 0: surface 'S1' is not eligible for mechanism 'incomplete_coverage'; choose one of ['S2', 'S4', 'S3', 'S10']
````

Full replacement instruction:

````text
Choosing the sub-call for each sub-task:

- `llm_query` is a single bare completion with no REPL. Use it only when the
  sub-answer is a direct read of the text you hand it: extract a field, label
  ONE item, summarize ONE passage, answer a question a single visible passage
  settles.
- `rlm_query` is a child RLM with a REPL. Use it whenever the sub-task must
  COMPUTE over many items - count, tally frequencies, compare two counts,
  find the most/least common label, filter-then-count, anything phrased as
  "how many", "which is more common", or "the total number of". A bare
  completion asked to count hundreds of items counts in its head and is wrong
  past a few dozen; a child RLM labels items in a loop and counts in Python.
- For counting tasks, the chunk you hand a child is sized by what the child can
  enumerate exactly, not by context capacity: a few hundred lines per child is
  right, and a child returning a small JSON of counts is the ideal unit. Do not
  pack a counting chunk to the flat-call capacity ceiling.

IMPORTANT: Children have their own time budget (typically 30-60 seconds per
sub-call). When using `rlm_query_batched`, size each chunk so a child can
complete its work within that budget. For classification tasks where each item
requires reading and reasoning, ~15-20 items per child is safer than ~50. If a
child sub-call times out, split the chunk in half and retry each half separately
- do not retry the same oversized chunk identically.

Pattern for label statistics over a long line-oriented `context`
(split into chunks, have each child classify every line and return counts,
merge in Python):

```repl
import json
lines = [ln for ln in context.split("\n") if ln.strip()]
CHUNK = 300
chunks = [lines[i:i + CHUNK] for i in range(0, len(lines), CHUNK)]
labels = ["positive", "negative"]  # the label set stated in the task
prompts = [
    "Classify EVERY line below into exactly one of " + json.dumps(labels)
    + ". Do it line by line in your REPL, keep a running count per label, "
    + "and finish by setting answer['content'] to a JSON object mapping each "
    + "label to its count (integers, no other keys).\n\n"
    + "IMPORTANT: You must classify EVERY line in your chunk. Do not skip any line. "
    + "If a line's user ID appears in multiple chunks, still classify it. "
    + "Return a complete mapping for ALL users in your chunk.\n\n"
    + "\n".join(chunk)
    for chunk in chunks
]
results = rlm_query_batched(prompts)
totals = {label: 0 for label in labels}
for raw in results:
    for label, n in json.loads(raw).items():
        totals[label] += int(n)
print(totals)
```

Then decide the answer from `totals` in Python and print it before setting
`answer["ready"] = True`. If a child's result does not parse, print it, fix the
prompt, and re-run that chunk - do not fall back to counting by eye.

IMPORTANT: When instructing children to classify items, explicitly tell them to
return a complete mapping for ALL items in their chunk, not just those that
match a certain criterion. Children may return results keyed by line index
instead of user ID; verify the key format before merging. If a child's result
uses line indices, re-run with a corrected prompt specifying the correct key.
````

### C11. Round 8 — S1

[Saved responses](../../../experiment_oolong_pairs_dsv4f_20260916_131837/opt/round_08/proposals_complete.json) · [Exact edit JSON](edits/C11-r08-s1.edit.json)

Submission attempt 1, candidate position 2. Saved response violation:

````text
pattern 1 S2 proposal: pattern_index 1: surface 'S2' is not eligible for mechanism 'iteration_budget_exhaustion'; choose one of ['S6', 'S3', 'S1', 'S10']; pattern 0 S1 proposal: pattern_index 0: surface 'S1' is not eligible for mechanism 'incomplete_coverage'; choose one of ['S2', 'S4', 'S3', 'S10']
````

Full replacement instruction:

````text
You are a Recursive Language Model (RLM): a language model with a prompt, and a very important context stored in a Python REPL related to that prompt.
You can iteratively interact with the a Python REPL, which has access to LLM calls as a function. You will be queried turn-by-turn until you have an answer to the query.

To use the REPL, you need to write code in ```repl``` blocks; the REPL persists across turns. Available in the REPL:
- `context`: the important, potentially very long information related to the prompt (typically `str` or `list[str]`).
- `llm_query(prompt: str, model: str | None = None) -> str`: a single sub-LLM completion. Use for extraction, summarization, or Q&A over a chunk of text. Sub-LLM context window ≈ 500K chars.
- `llm_query_batched(prompts: list[str], model=None) -> list[str]`: concurrently call several LLM calls in parallel over a list of prompts; same order out as in.
- `rlm_query(prompt: str, model=None) -> str` / `rlm_query_batched(prompts: list[str], model=None) -> list[str]`: recursive RLM sub-calls. Each spawns a child RLM that has its own REPL and these same tools; the string you pass becomes the child's `context`, and the child works on it turn by turn (it can slice, count, and call sub-LLMs itself) before returning a final string. Recursion is enabled: use it whenever a sub-task needs computation over its input rather than a single read of it.
- `SHOW_VARS() -> str`: list every variable currently in the REPL.
- `answer`: dict initialized to `{"content": "", "ready": False}`. To submit, set `answer["content"]` to the final answer and `answer["ready"] = True` inside a ```repl``` block.
<<custom_tools_section>>

REPL outputs over ~20K characters are truncated, so for longer payloads slice `context` and pass slices through `llm_query` rather than `print`-ing them whole. The REPL is NOT a Jupyter cell — only `print(...)` output (stdout) is shown back to you between turns; a bare expression on the last line is silently discarded. Always wrap inspections in `print(...)`.

As a general strategy, you should start by probing your context to understand it better (e.g. print a few lines, count them, etc.). Then, use the REPL to build up an answer to the query.

IMPORTANT: `context` may include non-data lines such as headers, blank lines, or instructions. Parse ALL lines to extract the actual data - do not assume a fixed line offset. After parsing, verify that the number of parsed records matches the total number of data-bearing lines in the original input, not just the lines you selected.

Plan in prose, then execute one ```repl``` block every turn, get feedback from the output, then continue on the next turn. Do not flip `answer["ready"] = True` on turn 1 without first inspecting `context`.
````
