# Appendix A — all edit outcomes and their implications for the optimizer

This appendix interprets all **16 materialized edits** and **11 distinct unmaterialized draft payloads**. Exact text, policies, code, original explanations, and before/after diffs remain in the [original report](../../../docs/analysis/oolong-pairs-2026-09-23/report.md) and [audit](../../../docs/analysis/oolong-pairs-2026-09-23/audit.json). A row's proposed alternative is a research hypothesis, not an evaluated improvement.

## Promoted edits

| Candidate | What was selected | What the result supports and does not support | General lesson for mining/proposal generation |
| --- | --- | --- | --- |
| `r03-c01-s4` | Explicit user-set coverage and repair. | Its single-edit validation improved exact 2→3 and F1 0.5980→0.6791. A successful trace already had full record coverage before reclassification, so skipped records are not an established explanation of that success. | Specify the invariant and its universe, then observe whether the changed check detects the claimed failure. Keep semantic error hypotheses separate. |
| `r04-c01-s4` | Add sampled semantic checks after coverage. | The S4+S2 batch improved exact 1→2 and F1 0.6532→0.7250. Label inspection appears in a successful execution, but self-inspection is not an independent label oracle. | Mine uncertain judgments separately from missing records. Require a bounded, relevant verification method and acknowledge what it cannot establish. |
| `r04-c02-s2` | Require complete child mappings in an existing counts example. | Shares round 4's outcome; no separate causal score. The example still requests counts and parses them as counts. | Check consistency of a proposed producer format with its retained consumer. Prefer replacing or scoping an example over adding incompatible requirements. |
| `r05-c01-s2` | Require all items and verify whether output keys denote users or record positions. | The S2+S4 batch improved exact 1→2 and F1 0.6859→0.7354. Key/coverage repair is visible, but the counts-versus-mappings conflict persists. | Make identity and schema assumptions explicit; consider a deterministic parser/join helper when that operation recurs. |
| `r05-c02-s4` | Rerun every chunk after correcting one prompt. | Shares round 5's outcome. One successful trace eventually mixed older chunks with a repaired chunk, contrary to the added rule. | Distinguish local repair from global invalidation. Ask which outputs depend on the changed assumption before multiplying work. |

The promoted examples suggest useful directions, but not a universal “add more verification” policy. Each promoted batch spent more than its fresh baseline; the cost ratios were approximately 1.93×, 1.35×, and 1.49×. Those ratios cannot be multiplied into an end-to-end estimate. [Round metrics](../../../docs/analysis/oolong-pairs-2026-09-23/rounds.csv).

## Tested but unpromoted edits

| Candidate | Outcome context | Why its proposed mechanism was weak or unresolved | Better general proposal question |
| --- | --- | --- | --- |
| `r01-c01-s4` | S4+S9: exact 2→2, F1 0.7085→0.5429. | Count equality is weaker than identity coverage and does not establish semantic correctness. Its individual contribution was not isolated. | Which invariant fails in the cited operation, and can a tiny counterexample distinguish it from a weaker count check? |
| `r01-c02-s9` | Same batch. No expanded evidence for its pattern. | The reason guesses a format defect; canonical pairs and unrecognizable text pass through. The normalization branch redirects rather than accepting transformed output. | Is there a demonstrated answer-visible defect? What exact `AnswerDecision` does the candidate return on it and on valid alternatives? |
| `r02-c01-s2` | S2+S4+S3: exact 1→0, F1 0.7156→0.6098. No expansion for its pattern. | Repeats a broad coverage remedy; does not demonstrate where information was actually lost. | What input or intermediate item disappeared, at which operation, and why is S2 the point that can prevent it? |
| `r02-c02-s4` | Same batch. | Substantially overlaps the S2 coverage addition. The whole batch produced scored answers but none exact. | Does this check enforce a different invariant, or duplicate another sibling? |
| `r02-c03-s3` | Same batch. No expansion for its pattern. | Focuses on output layout, including printing, rather than the actual submission operation and demonstrated failure. | Is the output missing, malformed, or semantically wrong? Fix the operation that produces the defect. |
| `r06-c01-s6` | Exact 1→0, F1 0.6194→0.7193. | A syntax retry is explained as timeout recovery and chunk splitting. No positive persisted retry counts were found. Higher F1 does not validate the claimed mechanism. | Which runtime branch activates? Does a stubbed timeout take that branch, and what changes in the retry input? |
| `r07-c01-s4` | S4+S3: exact 1→1, F1 0.6355→0.5916. | Its incumbent description already includes total/per-user count checks. More count checks cannot independently prove record identity or label correctness. | What new observable behavior would occur at the cited operation? If existing checks pass, what counterexample still fails? |
| `r07-c03-s3` | Same batch. No expansion for its pattern. | Preserving multiplicity and avoiding overwrites is plausible, but the actual overwrite must be shown. The conflicting S2 return contract remains. | Would an explicit merge procedure or a parameterized helper prevent the demonstrated loss while preserving ordering/multiplicity requirements? |
| `r08-c01-s6` | S6+S4+S3: exact 3→1, F1 0.6128→0.6951. | Repeats round 6's exact policy payload; it does not cap model-written retry loops or retry timeouts. | What substantive revision to the previously tested intervention is being made? |
| `r08-c02-s4` | Same batch. | Checking original-input coverage can help, but the rationale claims a data-bearing header was skipped without establishing that fact. | What is a record according to the actual input contract, and which observed data-bearing unit was excluded? |
| `r08-c03-s3` | Same batch. No expansion for its pattern. | A formatting instruction is used despite evidence suggesting the answer contains intermediate classifications rather than the required final result. | Did the model serialize the wrong object or serialize the right object incorrectly? Those require different interventions. |

The round-6 and round-8 F1 gains should remain qualified positive signals in history, with exact regressions and cost attached. The absence of a promotion is not sufficient to blacklist a mechanism; nor is a dense-metric gain sufficient to replay an inactive or misdescribed edit. [Original report's F1 analysis](../../../docs/analysis/oolong-pairs-2026-09-23/report.md).

## Drafts excluded before evaluation

| Original appendix ID | Surface and proposal | Recorded rejection | Implication |
| --- | --- | --- | --- |
| C1 | R2 S9 format normalization | S9 ineligible for `incomplete_coverage`. | This is not automatically a bad gate: semantic coverage is outside S9's visibility. Require the actual operation and answer-visible defect before rerouting. |
| C2 | R2 S2 revised mappings | Selection reason over 600 characters. | Repair the explanation field while preserving valid edit content; reason length should not require regenerating the entire intervention. |
| C3 | R2 S4 coverage revision | Occupied pattern/surface during repair. | Retained siblings need an explicit occupancy list. Repair should resolve the rejected slot, not replace a retained one. |
| C4 | R3 S2 smaller simultaneous batches | S2 ineligible for exhaustion. | Distinguish chunk decomposition from scheduling. Decomposition can justify S2; execution scheduling often belongs in S3. Do not route solely by the terminal resource label. |
| C5 | R3 S5 split/retry failed chunks | S5 ineligible for exhaustion. | Recovery is plausible when errors and their consumers are observed. Make that route conditional on evidence; do not claim every exhaustion is repairable after the fact. |
| C6 | R3 S10 batch-processing procedure | Two S10 selections. | A ranked selection should choose one supported procedure. Isolated record validation passes; usefulness and the appropriateness of its hardcoded sizes remain untested. |
| C7 | R3 S10 recovery procedure | Same S10 collision. | Could instead compete with C6 or become a coherent single procedure only if the evidence supports that combined intervention. Continuing after missing pieces must respect the task's completeness requirement. |
| C8 | R4 S9 normalization | Long selection reason, then ineligible coverage route. | Shortening the reason alone would not make the surface capable. Contract repair and mechanism repair are different. |
| C9 | R7 S6 one retry with syntax retry disabled | Long selection reason twice. | Even after repairing the reason, the claimed retry is absent: this configuration yields zero runtime retries. |
| C10 | R8 S2 smaller chunks and split-on-timeout | S2 ineligible for exhaustion. | A capacity-aware decomposition hypothesis deserves inspection where supported. It should not be converted into a syntax retry merely because S6 is eligible. |
| C11 | R8 S1 parse all data-bearing lines | S1 ineligible for coverage. | Keeping parsing strategy out of the factual REPL contract is appropriate. Consider S2/S3/S8 only after establishing an actual parsing loss. |

Untested drafts have no performance result. The purpose of examining them is to distinguish avoidable authoring failures, legitimate scope restrictions, and routes that exclude potentially useful intervention classes. No paid candidate execution was performed for this investigation.
