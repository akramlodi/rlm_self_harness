# Three-round mining and proposal assessment, September 24–25, 2026

The experiment completed three rounds and promoted **four edits, versus two in the previous experiment's first three rounds**. Two of the new promotions came solely from accepting an exact-pass tie. Both experiments produced one batch that improved exact passes. Proposal admission and diagnosis formatting improved, but this run does not establish improved semantic proposal quality.

## Setup and completion

- Experiment: `experiment_oolong_pairs_dsv4f_20260924_161551_3rounds`.
- Runtime commit: `ad2ac00147d7e350bdaed12eb67ae941a36f08d6`, based on main `5e454117`.
- Environment/model: OOLONG-pairs / DeepSeek-V4-Flash, starting from H0*R.
- Each round: 20 held-in tasks × two mining attempts; one combined candidate evaluated against the incumbent on ten held-out tasks × one attempt. No held-in promotion validation.
- Three mining workers; five configured validation subject workers, with one run worker per subject. The two validation subjects were the incumbent and combined candidate.
- Promotion: held-out exact-pass delta **>= 0**, including zero-pass ties, with the existing resource bands retained. Mean cost may be up to 3× incumbent cost. F1 remains diagnostic.
- Protocol: `heldout-batch/v2`, with a new experiment identity so strict-gate results cannot silently resume under the inclusive rule.
- Started September 24 at 16:23 Central; completed September 25 at 10:25 Central, exit code 0. Exactly 180 task attempts persisted: 120 mining and 60 validation. No fourth round or separate final evaluation ran.
- Recorded total stage cost: **$29.30085**, using $0.19/$0.51 per million input/output tokens. The $29.14777 run total overlaps this total and must not be added to it.

All 199 frozen source files and the config matched their launch hashes after completion. The final harness hash is `e9ae3b12d7c46da61d4f2974a5b7eccc0de8f47a2de95e9f4e4cd49d509f58d6`.

The comparison experiment is `experiment_oolong_pairs_dsv4f_20260923_163355`. Its first three rounds used byte-identical held-in and held-out splits. The only TOML value changed for this launch was the round limit, from 15 to 3; the source also includes the merged mining/proposal and observation changes plus the inclusive gate. This is a descriptive comparison, not a controlled ablation of individual changes.

## Validation outcomes

All scores below compare that round's current incumbent with its combined proposed edits. They are fresh measurements, not a carried-forward quality curve.

| Round | Proposed surfaces | Exact passes | Mean F1 | Candidate/incumbent mean cost | Decision |
|---|---|---|---|---|---|
| 1 | S8 + S10 | 1/10 → 2/10 | 0.7318 → 0.7602 | 1.460× | Promoted; also qualifies under the old strict gate |
| 2 | S2 + S3 | 2/10 → 2/10 | 0.7908 → 0.6810 | 0.900× | Promoted solely by the new inclusive gate |
| 3 | S4 + S10 | 3/10 → 1/10 | 0.6805 → 0.5882 | 0.902× | Rejected under both gates |

F1 uses the verifier's declared all-attempt mean, including zero for malformed answers. Malformed-answer counts for incumbent/candidate were 0/1, 1/2, and 1/1 respectively. No validation run ended in a runtime or resource-limit failure.

The paired exact gains/losses were 2/1 in round 1, 2/2 in round 2, and 1/3 in round 3. A tie therefore did not mean the same tasks still passed. The round-3 candidate's F1 improved on two tasks, worsened on six, and tied on two.

The final harness retains **S8 and S10 from round 1, and S2 and S3 from round 2**. The round-3 S4 instruction and additional S10 skill were not retained. Batch validation cannot identify an individual edit's effect. With `v=1`, these observations also do not establish stable gains or regressions.

The old-rule comparisons above score each actual batch against its actual measured incumbent. They are not a replay of the trajectory that would have occurred if round 2 had been rejected.

## What improved in the optimization process

| First three rounds | Previous experiment | This experiment |
|---|---:|---:|
| Admitted edits | 5 | 6 |
| Distinct surfaces receiving admitted edits | 3: S2, S3, S4 | 5: S2, S3, S4, S8, S10 |
| Empty proposal rounds | 0 | 0 |
| Promoted batches / constituent edits | 1 / 2 | 2 / 4 |
| Batches qualifying under the strict rule | 1 | 1 |
| Promotions requiring the inclusive rule | 0 | 1 batch / 2 edits |
| Accepted diagnosis responses / all responses | 71/179 (39.7%) | 79/97 (81.4%) |
| Records obtaining an accepted diagnosis | 71/89 | 79/81 |
| Missing `coverage_basis` refusals | 97 | 0 |
| Proposal responses, including repairs | 6 | 5 |

The diagnosis-schema examples appear to have addressed the repeated missing-field failure: it disappeared in all three rounds. Valid references and unique surfaces reached materialization without collision or escaping failures. Round 2 admitted both edits on the first response. Round 1 retained its valid S8 sibling during repair and admitted S10 afterward.

Round 3 exercised the revision guard: initial S3 and S2 candidates omitted required predecessor revisions and were rejected. Repair chose S4 and S10 instead. That demonstrates working enforcement and retargeting, but not successful substantive revision. Moving related behavior to another surface can avoid the surface/mechanism predecessor requirement.

History included both completed rounds in the third proposal prompt, including round 2's F1 decline. Evidence stayed within 32,000 characters. Each round expanded two mechanisms, but rounds 2 and 3 had four actionable mechanisms; the timeout packet in round 3 was omitted for budget. Initial system prompts grew from 52,743 to 61,501 to 69,708 characters as the incumbent and history grew.

Saved model-call observations were present throughout mining and proposal stages. Inspected provider responses reported that separate reasoning text was not returned; saving observations does not manufacture a provider reasoning field.

## What still limits proposal quality

The following findings were recorded from held-in traces and reconstructed, hash-matching proposal prompts before inspecting each round's validation result.

**Round 1's S8/S10 batch:** The helper and skill encouraged classification checks, but their cited mechanisms were not established. The S8 helper did not flag its own cited input when locally replayed with its defaults. The S10 rationale attributed missing pairs to a user exclusion although the representative had zero missing pairs and two extras. Other diagnoses still inferred input loss from answer errors or preview truncation despite complete record counts. The batch's favorable validation result does not validate those explanations.

**Round 2's S2 edit:** Replacing the count-only worked example with per-instance output was a concrete text change. However, the witness had already repaired its return format and parsed all 188 classifications. Its pair builder already excluded self-pairs. A mistaken checker rejected overlapping group membership, then the root explicitly corrected that checker; the selected evidence omitted the correction. The proposal repeated completed work and confused potentially overlapping roles with mutually exclusive groups. Its example also retained a conflicting count-return sentence and referenced an undefined `extract_user_id` helper.

**Round 2's S3 edit:** The cited helper expected lists of regex patterns but received natural-language strings, which it iterated character by character. A local replay reproduced its all-`entity` suggestions. The proposal reacted to the suggestion distribution without correcting the argument contract, and treated no suggestions as verification passing. The root had already recognized the helper's bad behavior, but those observations were absent from the packet. Repeating classification does not necessarily fix the failing helper call.

**Round 3's S4 edit:** Requiring an explicit empty-result marker addresses a real submission defect. Its representative deliberately submitted an empty string after concluding no users qualified; the packet showed only the preceding profile printout. For that witness the empty-set conclusion was also semantically wrong, so formatting alone cannot solve it. Another member of the same pattern did have an empty expected answer, making the format correction directly relevant there. The diagnosis's “forgot to combine counts” explanation was less precise than the actual submission operation.

**Round 3's S10 edit:** The skill adds chunked classification and spot-checking, but the root already inspected labels repeatedly. Those inspections were omitted while final answer submission remained in the packet. The claim that “In what year did Thatcher gain power?” should be `entity` instead of `numeric value` had no verified intermediate-label evidence. Chunks of 100–300 items can still place all 188 items in one child, and the skill recommends label sets without explicitly retaining counts for an “exactly one” predicate. The new skill was loaded eight times across candidate execution trees; that does not demonstrate correct execution of its procedure.

The mining measurements also qualify the promotion count: exact passes across rounds were **13/40, 12/40, 14/40**, while F1 was **0.8133, 0.7965, 0.7708**. Mining cost rose **$4.39 → $6.78 → $7.61**. Round 3 had five malformed answers and one timeout. Three malformed answers followed exhausted execution: after 30 executed turns, the final non-executing call returned more code. Diagnosis assigned two of those to skipped verification or incomplete coverage; the third remained unattributed after invalid-coordinate repairs.

## Bounded, task-agnostic follow-ups suggested by the observations

These are findings for a subsequent change, not modifications made during this experiment.

1. **Anchor diagnoses in observed operations.** Separate an observed output error from an inferred input or label error. If intermediate labels are unverified, preserve that uncertainty instead of deriving a specific “correct” label from the final answer mismatch.
2. **Include the operation's inputs and latest relevant correction.** Prefer helper-argument construction, a repaired check, and the actual submission over answer-formatting repetition or large child payloads. Complete code alone is insufficient when its input contract or later correction is missing.
3. **Compare substantive interventions across surfaces.** Retargeting may be appropriate, but moving the same instruction from S2 to S10 should still explain what execution would change. A new surface is not itself a new mechanism.
4. **Expose termination and final-call semantics.** Preserve whether execution was exhausted and whether final-call code could run. That supports a bounded finalization or recovery intervention instead of blaming coverage for an unexecuted answer.
5. **Check proposed helpers against their cited input.** A small local fixture can establish whether the helper changes the demonstrated case and accepts the actual argument shape. It cannot establish semantic correctness for all tasks, but it can reject a concrete mismatch cheaply.

## Artifacts and verification

All experiment paths below are relative to `experiment_oolong_pairs_dsv4f_20260924_161551_3rounds/`:

- `final-assessment.md`: copy of this report.
- `round-01-assessment.md`, `round-02-assessment.md`, `round-03-assessment.md`: detailed trace reviews, written before each validation result and then supplemented with outcomes.
- `round-0N-proposer-prompt.txt`: exact reconstructed initial proposer prompts, verified against persisted SHA256 values.
- `round-0N-paired-validation.json`: per-task exact/F1/cost comparisons.
- `round-01-s8-synthetic-check.json`, `round-02-helper-contract-check.json`: local helper checks; no extra model calls.
- `latest-audit.json`, `baseline-audit.json`, `monitor-results.md`: cumulative metrics, comparison extraction, and assistant checkpoints.
- `opt/round_0N/proposals/*/proposal.json`: exact admitted edits and explanations; `proposals_complete.json` also retains rejected attempts and repairs.
- `analysis/20260925T152540Z/`: final built-in CSV/JSON analysis publication, including surface activity, incumbent/candidate quality, attribution, and pattern differences.
- `sh_rlm/harness.json`: final frozen harness.
- `.launch/launch.json`, `.launch/source/`, `status.json`: source/config hashes, prices, and successful exit status.

The inclusive-gate implementation's final focused test rerun passed 51 tests; scoped Ruff and whitespace checks passed. The earlier combined targeted run passed 169 tests and identified four fixtures assuming ties must reject; those fixtures were corrected and included in the passing rerun. A full repository test suite was not run for this change.
