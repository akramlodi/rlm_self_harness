# OOLONG-Pairs split decision and optional alternative

Date: 2026-09-28. Updated after author clarification. No experiment configuration, loader, or optimizer behavior has been changed.

**Keep the current split for the forthcoming study: 20 short mining instances, 10 short validation instances, 10 short test instances, and 40 long evaluation instances.** Task instances are disjoint across the short roles, while their short-context records are shared. Describe short-test results as performance on held-out task instances. Use the separate long evaluation to assess performance on larger record collections, without claiming entirely unseen underlying question texts.

The [paper audit, finding #7](2026-09-28-paper-optimization-accuracy-audit.md) treats this as a description and interpretation requirement, not a requirement to redesign the split. A stronger split becomes necessary only if unseen-source-data generalization becomes a central claim. The earlier recommendation to separate short contexts is superseded for the upcoming experiment; it remains below as an optional alternative with its tradeoffs and supporting evidence.

## Previously considered allocation — not selected

One alternative would assign one complete short window to mining and the other to validation, with long contexts for final evaluation. It would separate the full record collections seen in mining and validation but leave no context-disjoint short test. If selected for a future study, the context assignment and task subset would be chosen before inspecting outcomes and persisted in the split manifest. Window A/B are roles, not hardcoded IDs that the optimizer needs to know.

| Role | Data | Task count | Attempts per task |
|---|---|---:|---:|
| Mining | Short window A | All 20 task predicates | 2 |
| Validation | Short window B | 10 fixed, balanced task predicates | 1 |
| Final long evaluation | Both supplied long windows | 20 predicates per window, 40 total | 3 per evaluated harness |
| Context-disjoint short test | Not available under this alternative | None | None |

Under that alternative, validation predicates could cover symmetric/asymmetric and counting/date conditions, independent of observed success. The other ten questions on window B would not supply a context-disjoint test: validation already adapts the harness to that record collection. They could still support evaluation on held-out task instances, which is also the accepted scope of the retained current short test.

The alternative's optimization task-attempt budget would remain `20*2 + 2*10*1 = 60` per round with a candidate. The promotion gate, batch composition, optional verifier metrics, and mining/proposal loop would not need to change. This allocation is not needed to support the current held-out-task-instance claim.

Implementing the alternative would require an explicit group-based split policy and handling of an omitted short test. Changing only `seed` or launching in a new output directory would not implement it. The existing loader shuffles `(window, task)` rows and the splitter slices them, so merely changing counts would not guarantee the assignment above. No such implementation is requested.

## If unseen-source-data generalization becomes a central claim

For a future claim requiring context-disjoint short evaluation, at least one additional short context would be needed, preferably several, with distinct context groups assigned to mining, validation, and final test before selecting task questions. Multiple questions from one context would stay in its assigned role. The existing 20/10/10 task budgets could remain; they need not equal the number of available questions. A claim about unseen underlying question texts would additionally require separating that source material, as discussed below.

At the current pinned revision, the upstream `test` split does not contain `trec_coarse`: it contains other source datasets. Increasing `max_scan`, switching to that split, or relabeling user IDs therefore does not produce more independent TREC-coarse windows. Other context lengths exist, but substituting lengths changes the data distribution and does not automatically solve source-text overlap.

A separate dataset-construction option is to form additional short contexts from an independently reserved labeled-record pool and recompute answers with the existing task predicates. Reserve the final-evaluation data first. If contexts are derived from a common parent document or overlapping source records, keep that family in one split. Do not take development chunks from a long context still counted as untouched final-evaluation input. This is a benchmark-construction change and should be described as such, not passed off as an unchanged canonical split.

## What group separation establishes

The two existing short windows each have 188 records. They have **zero identical `(user_id, date, question_text)` records and zero shared user IDs**, but **43 identical question texts** appear in both. Thus even context-separated mining and validation reuse some semantic classification inputs with different dates/users. The complete record collections and pair-computation problems differ; completely novel classification text is a stronger claim this allocation does not establish.

The two saved long windows likewise have no exact record-tuple overlap with either short window, but share many question texts with them. They can assess performance on different, longer record collections. They do not establish evaluation on entirely unseen underlying question texts, and 40 long task instances still represent only two long contexts. These facts concern the evaluation's scope, not proof of answer memorization.

If novel source-question generalization becomes a central claim, partition by normalized question identity before constructing contexts. That is a distinct, stronger experiment than preventing identical complete contexts from appearing in mining and validation; it should not be silently implied by a new shuffle.

If such a stronger split were needed, the task-agnostic rule would be: **the environment identifies the source-data group relevant to the claim, and the splitter assigns groups to roles before sampling task instances.** This would not require an OOLONG-specific optimization algorithm or different promotion criteria.

## Evidence

- [Pinned dataset inventory and overlap counts](2026-09-28-paper-optimization-audit/split-options-evidence.json). All 41 cached test shards and nine cached validation shards were scanned using metadata columns; no dataset download or model call was needed for the inventory. Overlap was computed from saved short/long prompts.
- [Environment loader](../../shrlm/environments/oolong_pairs.py): `collect_windows` filters TREC-coarse windows; `load_oolong_pairs` shuffles window-task rows; `parse_entries` defines record extraction.
- [Split materialization](../../shrlm/experiment/splits.py): partitions instance rows rather than complete contexts.
- [Official dataset card](https://huggingface.co/datasets/oolongbench/oolong-synth/blob/main/README.md): describes the published validation/test files and separate context/question fields. Per-source window counts above come from the pinned local data, not an inference from the card's aggregate row counts.

The current decision requires accurate paper wording and interpretation only; it leaves the split policy unchanged. Any previously inspected examples retain their exposure history; a fresh output directory does not make them previously unseen to the research process.
