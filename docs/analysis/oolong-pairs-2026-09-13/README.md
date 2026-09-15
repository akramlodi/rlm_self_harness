# OOLONG-Pairs optimization audit

The runs do not establish performance saturation. They show a loop repeatedly spending its opportunities on invalid answer middleware, incomplete proposal context, and diagnoses that do not reliably identify the failing operation. Several useful changes can fit inside the existing mining → proposal → combined held-out validation loop.

The strongest immediate intervention is to make S9 obey the actual answer contract and test its acceptance branch before paid validation. Increasing the round limit or accepting weaker candidates would leave the most clearly demonstrated problems intact.

## Evidence and scope

This audit covers all **14 completed rounds, 580 persisted attempts, 17 written proposals, and 282 mining failure records** across three directories. There are 380 mining attempts and 200 validation attempts. I inspected promotion ledgers and cached proposal responses, reconstructed the latest run's five proposer prompts (all five hashes match their saved markers), inspected selected underlying execution traces, and ran five synthetic cases against each of the eight proposed S9 implementations in short-lived local processes. No provider calls or experiment reruns were used.

I also re-scored recorded, redirected root submissions using the existing verifier and saved instances. These are diagnostic counterfactuals: saved outcomes, promotion decisions, and historical artifacts were not changed. The oldest run lacks the redirect events used for this audit, so its apparent zero redirects are not evidence that middleware never redirected.

The runs differ in held-in size, held-out membership, cost constraints, and code version. Their scores are not controlled estimates of one implementation outperforming another. Within-round comparisons and explicit contract failures provide stronger evidence.

| Run directory | Completed rounds | Promotions | Why it stopped |
|---|---:|---|---|
| `experiment_oolong_pairs_dsv4f` | 6 | R1 S4; R2 S9; R3 S4 | R4–R6 wrote no candidates: each final proposal reproduced an existing surface. Three such rounds exhausted patience. |
| `experiment_oolong_pairs_dsv4f_20260911_1458` | 3 | None | Three rounds without promotion, including a better, cheaper batch rejected by the cost floor. |
| `experiment_oolong_pairs_dsv4f_20260911_213930` | 5 | R2 S2+S9 | R3–R5 each evaluated a broken S9 candidate; after containment was fixed, all three completed and exhausted patience. |

Detailed machine-readable evidence: [round metrics](rounds.csv), [artifact audit](artifact-audit.json), [S9 probes](s9-contract-probes.json), and [correct submissions that were redirected](correct-answers-redirected.json). F1 in the CSV is averaged from the verifier's persisted, three-decimal detail; attempts without a scored answer contribute zero.

## What happened in the latest run

| Round | Candidate outcome | Held-out baseline → candidate | Preventable loss |
|---|---|---|---|
| 1 | S9 rejected before evaluation | Not evaluated | Calls nonexistent `AnswerDecision.reject`. |
| 2 | S2+S9 promoted | 0/10 → 2/10 | A real improvement, although the promoted S9 imposes an unnecessary bracketed-list contract. |
| 3 | S2 rejected; S9 evaluated alone | 2/10 → 0/10 | S2 contains unescaped template braces. Eight S9 attempts crash because inventory metadata is treated as raw context. |
| 4 | S2+S9 rejected | 1/10 → 0/10 | Five attempts crash on `AnswerDecision.accept()` without its answer argument. The combined evaluation cannot establish whether S2 helped. |
| 5 | S2 is a no-op; S9 rejected | 2/10 → 0/10 | Eight attempts repeat the missing-argument error. S2 is dropped without a repair because another proposal materialized. |

Thus **21 of the last 30 candidate attempts failed on runtime errors**, and the other nine failed answer formatting. These 0/10 results are strong evidence of broken candidates, rather than a measurement of the best achievable harness.

Sources: the latest run's [round 3 ledger](../../../experiment_oolong_pairs_dsv4f_20260911_213930/opt/round_03/validation/round_03/promotions.jsonl), [round 4 ledger](../../../experiment_oolong_pairs_dsv4f_20260911_213930/opt/round_04/validation/round_04/promotions.jsonl), [round 5 ledger](../../../experiment_oolong_pairs_dsv4f_20260911_213930/opt/round_05/validation/round_05/promotions.jsonl), and [round 5 proposal marker](../../../experiment_oolong_pairs_dsv4f_20260911_213930/opt/round_05/proposals_complete.json).

## Small fixes, in priority order

### 1. Give S9 the actual contract, including valid answers it must allow

**Change:** Add a short proposer contract showing `AnswerDecision.accept(answer)`, `AnswerDecision.redirect(message)`, and an inventory example such as `{"context": ("str", 19006)}`; state that inventory contains types and lengths, not values. State explicitly that newline-separated integer pairs and `No valid pairs found.` are valid output forms, that global list sorting is not required by the verifier, and that some users correctly appear in no pair.

The current verifier configuration tells the proposer `extraction_rule=user-id-pair-tuples` and `gold_ordering=sorted`, but does not explain the accepted empty marker or distinguish canonical display order from a requirement on model output. The evidence renderer then shows canonical bracketed pair sets, which makes a bracketed-list requirement look natural. The promoted S9 redirects valid newline answers and valid empty answers; an earlier promoted S9 incorrectly requires a complete graph over all IDs appearing in the answer, although several task predicates require different roles for the two users.

This is measurable harm: in the latest run, **39 attempts had an exact-correct submitted answer redirected; four ultimately failed**. In the three-round run, seven did, and three ultimately failed. One earlier held-out attempt submitted the same correct newline answer five times and was redirected each time. These observations do not imply every downstream result can be improved by exactly those counts, but they prove that middleware is rejecting valid solutions.

Sources: [verifier/parser](../../../shrlm/environments/oolong_pairs.py), [promoted latest S9](../../../experiment_oolong_pairs_dsv4f_20260911_213930/opt/round_02/proposals/r02-c02-s9/surfaces.py), [redirect audit](correct-answers-redirected.json). Confidence: **high**.

### 2. Probe both callable branches and repair gate failures before sealing proposals

**Change:** Extend the existing isolated S9 check with a valid pair list, newline pairs, an explicit empty answer, and malformed input, using realistic redacted inventory. Run the existing candidate gate before closing the proposal stage and give a failing surface one bounded repair attempt while retaining the valid proposals and the one-edit-per-surface rule.

The current probe passes only `"boundedness probe answer"`; a middleware that redirects that string never exercises its acceptance branch. A single valid-list test with realistic inventory reproduced all three latest runtime-broken S9 candidates offline. Existing gate failures, such as S2 brace errors, arrive after the proposal retry loop has ended; no-op retries also happen only when the whole batch fails to materialize, leaving the round-5 partial failure unrepaired.

This does not require individual held-out validation or extra repetitions. The repair costs a bounded proposal call and local checks, and valid edits still receive one combined held-out decision. Checking for a Python exception alone is insufficient: the valid-format fixtures must also assert acceptance where the middleware has no valid reason to veto.

Sources: [single middleware probe](../../../shrlm/runner.py), [materialization and retry boundaries](../../../shrlm/optimization/proposal.py), [synthetic probe results](s9-contract-probes.json). Confidence: **high**.

### 3. Show the full current surface once, instead of repeatedly showing its prefix

**Change:** Render each eligible surface once in a compact current-harness section and include its complete implementation. Keep per-pattern blocks focused on evidence, rather than repeating a 1,500-character prefix of each eligible surface while requesting a full replacement.

The latest proposer prompts are **62,860–86,811 characters**, yet the current S9 is truncated: its rendered value is 2,154 characters in rounds 3–5. The missing tail includes existing duplicate/order checks and `accept(answer)`; round 5 proposes “adding duplicate detection” that was already present and replaces the valid acceptance call with the broken zero-argument form. Round 2's S2 proposal even describes appending after “the truncated portion,” although the output format replaces the whole surface.

This improves edit fidelity while potentially reducing prompt size. It also addresses a remaining source of no-ops without changing edit quotas or introducing a patch language.

Sources: [`CURRENT_VALUE_RENDER_MAX_CHARS` and pattern renderer](../../../shrlm/optimization/proposal.py), latest [round 5 S9 proposal](../../../experiment_oolong_pairs_dsv4f_20260911_213930/opt/round_05/proposals/r05-c02-s9/proposal.json). Confidence: **high** for missing context; its effect on future generation needs measurement.

### 4. Pass concrete diagnosis and task semantics through to the proposer

**Change:** Add one representative `symptom_summary`, the relevant task instruction without gold data, and compact missing/extra counts or pair examples to each selected pattern. Include a short successful execution example from held-in mining, rather than identifying passing behavior only by opaque instance IDs.

The miner already stores useful explanations, evidence node IDs, and precision/recall/F1 in failure records. The proposer receives a mechanism name, generic mechanism definition, median execution statistics, instance IDs, and large produced/gold strings; the saved causal explanation is omitted. In the latest run, **56 of 115 rendered evidence entries are truncated before “expected” even appears**. Passing behavior is rendered as IDs only.

A concrete distinction—“every row was processed, but a label was wrong,” “a response was joined to the wrong row,” or “a date predicate was reversed”—would support different, narrower edits. Use held-in evidence for this diagnosis; do not start mining held-out answers.

Sources: [clustering evidence construction](../../../shrlm/optimization/clustering.py), [pattern and passing-behavior rendering](../../../shrlm/optimization/proposal.py), latest [round 5 failure records](../../../experiment_oolong_pairs_dsv4f_20260911_213930/opt/round_05/mining/round_05/records.jsonl). Confidence: **high** that information is lost; future proposal quality is an empirical outcome.

### 5. Stop treating semantic mistakes as proven aggregation bugs

**Change:** Add two attribution rules: do not label missing coverage when every record was processed, and do not label lossy aggregation when the merge correctly reflects the available classifications. Prefer S3/S4 over S9 for actual aggregation or predicate problems; reserve S9 for defects detectable from its answer and redacted inventory.

All **282 mining failures are ungrounded at the child-verifier level**, because OOLONG-Pairs intentionally has no sub-verifier. Nevertheless, 137 records are classified as `lossy_aggregation`, whose definition asserts that child results were correct; it routes primarily to S9. This often turns uncertain upstream classification mistakes into confident requests for answer completeness checks.

There is a concrete incorrect attribution: latest round 5, `oolong-t16-w9-1dcaca72ce800990__a01`, is described as failing to require both labels for user A, but iteration 17 explicitly uses `description_count >= 1 and human_count >= 1`. Another record describes classification errors while assigning `lossy_aggregation`; its root produces all 528 combinations of the 33 selected users, so checking pair-list completeness cannot repair incorrect user selection. These examples justify improving attribution instructions and exposing the supporting code; they do not prove every record in those clusters is mislabeled.

Sources: [taxonomy meaning and routing](../../../shrlm/optimization/taxonomy.py), [task-16 trace](../../../experiment_oolong_pairs_dsv4f_20260911_213930/opt/round_05/mining/round_05/runs/oolong-t16-w9-1dcaca72ce800990__a01.json), [task-1 trace](../../../experiment_oolong_pairs_dsv4f_20260911_213930/opt/round_05/mining/round_05/runs/oolong-t01-w9-ab9956978b367bac__a02.json). Confidence: **high** for the demonstrated mismatches, **moderate** for the proposed routing change.

### 6. Tell the next proposer why the previous candidate actually failed

**Change:** Add runtime-error counts and the first exception type/message to prior-edit history, along with each combined batch's constituent effects or concise edit descriptions. Distinguish a failed evaluation caused by a coding error from an evaluated behavioral hypothesis that did not help.

The reconstructed round-5 prompt contains the round-3 and round-4 pass-count/cost rejection reasons, but neither the tuple-versus-string error nor the missing `answer` argument. The history renderer skips bundled constituent records, so their predicted effects disappear too. Round 5 then repeats the exact missing-argument error from round 4.

The diagnostics are already persisted; this needs a small history-rendering change, not a new memory system or another mining pass. For merged evaluations, report only what is observed—“the batch failed with this S9 exception”—rather than attributing the joint score to each edit independently.

Sources: [history loading](../../../shrlm/experiment/orchestrator.py), [history rendering](../../../shrlm/optimization/proposal.py), latest [round 4 candidate trace manifest](../../../experiment_oolong_pairs_dsv4f_20260911_213930/opt/round_04/validation/round_04/merged/heldout/round_00/runs.jsonl). Confidence: **high**.

### 7. Make patience count unsuccessful valid experiments, with a separate invalid-proposal limit

**Change:** After bounded proposal repair, a round with no valid challenger should consume an invalid-proposal allowance rather than the same patience counter used for evaluated non-improvements. Keep the existing maximum-round/spend limits and a small consecutive-invalid limit so an unproductive proposer cannot run indefinitely.

The six-round experiment consumed its last three rounds on no-op proposals and did no validation in those rounds. The latest run spent its final three rounds on S9 implementations whose acceptance branch was broken. Both are poor evidence that the space of useful harness edits is exhausted.

Treat this as a follow-up to preflight repair, not the first fix: excluding invalid rounds from patience without improving proposal quality merely spends more on invalid rounds. Do not simply discard failed paid attempts or remove them from scoring; containment and full denominators should remain intact.

Source: [unconditional no-promotion counter](../../../shrlm/experiment/orchestrator.py). Confidence: **high** about the stopping mechanism; the best invalid-proposal allowance remains a design choice.

## The scoring question: useful signal, but not the first rescue

Exact-set success is much harsher than partial pair accuracy. Latest round 2's baseline was **0/10 exact passes with mean recorded F1 0.682**; 45 of the latest run's mining failures had recorded F1 at least 0.900 but below 1.000. The original RLM paper reports OOLONG-Pairs F1, whereas this implementation uses exact-set equality for its promotion pass count. [Primary paper, §2.1](https://arxiv.org/html/2512.24601v1#S2.SS1)

First expose mean F1, missing-pair counts, and extra-pair counts in diagnosis and history. A small optional follow-up is to use mean F1 as a tie-breaker when exact pass count is unchanged and the cost band is satisfied, while continuing to report exact success separately. That changes the promotion protocol and belongs in a fresh, explicitly identified experiment; it should not silently reinterpret completed runs.

There is **no observed saved rejection that this tie-breaker clearly rescues**: the better/cheaper batch below already improved exact success, and the latest failures are runtime/format failures. So dense scoring is a plausible way to expose future incremental improvements, not an explanation that excuses the current bad candidates.

Noise also matters with `v=1`: in the six-round run, the promoted round-2 harness scored **5/10**, then the same harness hash on byte-identical held-out instances scored **0/10** as the round-3 baseline. This is a direct observation of run-to-run variability, not a confidence interval or proof that a particular promotion was wrong. Keep the requested single-repeat protocol, but retain paired per-instance outcomes and dense diagnostics to make decisions interpretable.

## Useful work already done, and one concrete proposal direction

- **Keep the cost lower bound at zero.** In the three-round run, the S2+S3 batch improved exact success from 2/10 to 4/10 and mean recorded F1 from 0.784 to 0.920 while costing only 37.8% of baseline. Its only rejection reason was being below the old `0.5` cost floor. The latest config's `[0.0, 1.25]` already fixes this. [Ledger](../../../experiment_oolong_pairs_dsv4f_20260911_1458/opt/round_02/validation/round_02/promotions.jsonl)
- **Keep runtime containment and no-op-only re-asks.** They already address the original crashes and the older total-no-op behavior. Remaining gaps are callable preflight, partial-batch repair, and informative feedback; a second redesign of those mechanisms is unnecessary.
- **Keep one edit per surface and one combined held-out evaluation.** A broken S9 should be repaired or excluded by the local gate before it can spoil the measurement of an S2 edit. There is no evidence here that individual paid validation of every edit is necessary.

A promising S2/S3 proposal can be described in three sentences: “Classify each input record once and return its stable record ID with the label. Check ID coverage and label validity, aggregate counts and dates by user, then apply the actual task predicate and construct qualifying pairs in Python. Do not ask children to independently enumerate partial pair sets, or assume every user must occur in the answer.” This is a focused procedural instruction, not a new optimization architecture; it is supported by the earlier better/cheaper S2+S3 batch, though it still requires prospective validation against the current incumbent.

I would implement items **1–3 and 6 first**, then improve the evidence and routing in **4–5**. Only after those changes produce valid, differentiated challengers would I alter patience or the promotion tie-breaker. Success should mean more valid behavioral hypotheses evaluated and better held-out outcomes—not merely a larger round number.
