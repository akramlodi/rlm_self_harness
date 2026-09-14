---
title: Proposal quality changes for a future paper update
date: 2026-09-14
type: docs
---

# Proposal quality changes for a future paper update

This note records the motivation and implemented methodological changes following `experiment_oolong_pairs_dsv4f_20260913_132607`.
The changes below are implemented in this follow-up and verified with offline tests and read-only replay of saved artifacts. No experiment using the new optimization loop has been run.
The repository's paper is behind the implementation; this note preserves material for a later paper update without editing that draft.

## Observations motivating the changes

The experiment launched from commit `3d8c8ef4a9002a84f0305abf3af27f2456c8f1a6` and completed four rounds.
Round 1 promoted a combined S2/S9 edit, improving held-out exact passes from 0/10 to 1/10.
Rounds 2 and 4 yielded no usable edits; their saved responses explicitly described proposed edits as unchanged or no-ops.
Round 3 evaluated one S2 candidate and rejected it after exact passes fell from 2/10 to 1/10.
The earlier investigation of three OOLONG Pairs experiments is in [the September 13 investigation](oolong-pairs-2026-09-13/README.md).

One round-4 diagnosis confused missing output pairs with incomplete input coverage.
For run `oolong-t02-w9-fcf921f0019d1bd5__a01`, it cited missing pairs as evidence that classification probably omitted questions.
The trace instead reports `Total classified: 188` and `Missing indices: []`.
That check establishes coverage of the parsed records represented by those IDs; it does not establish correct labels or prove that parsing retained every original input record.
The diagnosis needed evidence about the operation it blamed.

The promoted S2 also contains two procedures that need clearer scope.
Its worked example asks children for aggregate label counts, while a later paragraph asks for per-record labels joined by stable IDs.
Counts can suffice for simple label totals, but they discard the record/user/date associations needed for other predicates.
An effective edit could replace or explicitly scope that example instead of appending another coverage reminder.

Round 3 provides a separate reason to improve history:

| Recorded held-out measure | Incumbent | Candidate |
|---|---:|---:|
| Exact passes | 2/10 | 1/10 |
| Mean F1 across all attempts | 0.6388 | 0.7561 |
| Attempts with recorded pair metrics | 10/10 | 9/10 |
| Malformed answers | 0 | 1 |
| Missing pairs over scored attempts | 366 | 196 |
| Extra pairs over scored attempts | 98 | 136 |

The candidate's malformed attempt contributes zero to the all-attempt F1 mean.
Missing/extra totals have different denominators and must retain those denominators when reported.
The means use the verifier's saved, rounded diagnostic values.
The secondary improvement makes the direction potentially promising, while the lower exact-pass count and increased extra pairs remain contrary evidence.
With `v=1`, these observations do not establish a reliable improvement or identify the edit as its cause.

## Methodological changes

1. **Require evidence for the diagnosed operation.** Attribution should distinguish omitted input, incorrect or uncertain intermediate labels, and incorrect aggregation or predicate evaluation. It should cite the relevant operation and state what was not verified. A complete record-ID coverage check must not be presented as proof of semantic correctness.

2. **Show the proposer the relevant execution.** Resolve the existing call-node citations to the child call, its caller, and nearby code that parses or consumes its result. Prefer an explicitly cited operation when available. Allocate the existing excerpt budget to this evidence instead of always showing the first and last root blocks.

3. **Describe the behavioral difference before generating replacement text.** Each candidate should state the incumbent behavior, the observed remaining failure, and the concrete behavior changed by the edit. Repeating an existing instruction more emphatically is insufficient justification. Deterministic validation can check structure and literal no-ops; it cannot prove the semantic novelty or effectiveness of arbitrary prose.

4. **Allow the smallest effective replacement.** Proposal guidance should prefer replacing or clearly scoping an obsolete worked example when it conflicts with the desired return contract. For record classification, one consistent procedure should preserve record IDs, validate coverage and label vocabulary, retain user/date metadata, and compute the task predicate in Python. Examples use synthetic records and task-derived schemas, never remembered answers or instance-specific solutions.

5. **Match the edit surface to the operation and permit bounded retargeting.** Aggregation and predicate mistakes should favor execution/verification instructions, S3/S4. S9 can address defects visible in the answer and its documented redacted inventory; it cannot inspect hidden records or verify classifications. The existing repair attempt should be allowed to move a failed member to another eligible, unoccupied surface for the same pattern while retaining valid members.

6. **Separate failed promotion from evidence of partial progress.** Future proposal history should explicitly mark a rejected candidate or combined batch as potentially promising when a comparable, recorded secondary quality measure improves. Keep the rejection reason and regressions beside that annotation. The annotation invites a materially different refinement; it does not permit replaying the same rejected edit or relaxing promotion.

## Generalization beyond F1

Weakness mining already separates a closed failure signature from task-specific detail.
`Verdict.detail` stores verifier-authored observations, while `AttributionDetail` stores the diagnosis and citations outside the clustering key.
These are distinct channels: verifier measurements should not be replaced by an attributor's opinion about progress.

OOLONG Pairs and GraphWalks currently write precision, recall, F1, missing counts, and extra counts into `Verdict.detail`.
OOLONG writes `score`, `exact`, and answer kind instead.
Before this follow-up the proposal-history adapter understood only OOLONG Pairs diagnostics; preserving arbitrary detail does not itself define which direction is better, how to aggregate it, or what an unscored attempt means.

The extension uses small trusted interpretations of the existing verifier formats, feeding a common history comparison.
Each supported interpretation supplies the quality measure's name, direction, aggregation, and missing-score policy.
For the current bounded F1/score measures, known unscored terminal failures contribute zero to the all-attempt mean; missing legacy measurements remain unknown.
Tasks without a supported comparable measure retain their verdict and diagnostic context, with progress marked unassessed rather than invented.
Supporting another detail format requires an explicit interpretation, not guessing from arbitrary numbers in text.

Held-in diagnosis may use bounded verifier detail and trace evidence.
Held-out history exports aggregate measurements and fixed explanations only; raw verifier detail can contain answers and must not be forwarded.
Baseline and candidate comparisons use the same evaluated instance/attempt set and verifier contract.
When several edits were validated together, the evidence belongs to the batch and cannot be credited to each member independently.

## Experimental interpretation to retain in the paper

These changes target evidence quality, proposal specificity, and the treatment of unsuccessful attempts.
They preserve one edit per surface, combined candidate validation, held-out-only promotion evaluation, and `v=1`.
The exact-pass and cost gates remain the promotion criteria.
There is no new validation arm, model judge, or unbounded repair loop.

Any later paper claim that these changes produce more promotions or sustain more rounds needs a new experiment.
Record the implementation commit, prompt/taxonomy versions, proposal rejection reasons, repair retargets, usable-candidate counts, and both exact and secondary outcomes for that experiment.
Do not present the round-3 diagnostic gain as a demonstrated gain from the proposed changes.

## Evidence and implementation references

- [Round decisions and proposal artifacts](../../experiment_oolong_pairs_dsv4f_20260913_132607/opt/): each `round_XX/round.json` and `round_XX/work/proposal_result.json`.
- [Round-4 failure records](../../experiment_oolong_pairs_dsv4f_20260913_132607/opt/round_04/mining/round_04/records.jsonl) and the linked, hashed trace for `oolong-t02-w9-fcf921f0019d1bd5__a01`.
- [Round-3 baseline verdicts](../../experiment_oolong_pairs_dsv4f_20260913_132607/opt/round_03/validation/round_03/baseline/heldout/round_00/runs.jsonl) and [candidate verdicts](../../experiment_oolong_pairs_dsv4f_20260913_132607/opt/round_03/validation/round_03/r03-c01-s2/heldout/round_00/runs.jsonl).
- [Promoted harness](../../experiment_oolong_pairs_dsv4f_20260913_132607/sh_rlm/harness.json), including `S2_decomposition_instruction`.
- [Weakness-mining types](../../shrlm/optimization/types.py), [digest](../../shrlm/optimization/digest.py), [attribution](../../shrlm/optimization/attribution.py), [proposal evidence/history](../../shrlm/optimization/proposal_evidence.py), and [surface routing](../../shrlm/optimization/taxonomy.py).

Experiment directories may remain local artifacts; the numerical summary and named run IDs above preserve the observations when reading this note outside that workspace.

## Implementation verification

The new reader reproduces the recorded round-3 all-attempt means (0.6388 and 0.7561), marks the rejected candidate potentially promising, and retains exact passes, costs, malformed counts and the rise in extra pairs. The revised excerpt selector includes the round-4 `Missing indices: []` observation using the existing child citations. These checks read the saved experiment without rewriting it. Synthetic tests cover the same cases, non-F1 scores, unknown legacy measurements, incompatible comparisons, retained proposal bytes and bounded retargeting.

Live contracts are attribution prompt 1.3.0 / validator 1.1.0, taxonomy 3.2.0, digest 1.4.0, proposal prompt/validator 2.1.0, evidence selector 2.0.0 and diagnostic history 1.0.0. The additive proposal envelope remains v1. New live responses require the new fields; completed historical artifacts remain readable.
