# Executive summary

**The optimizer is not giving all useful intervention types a fair, well-informed opportunity.** Improving that process is more promising than forcing a wider distribution of surface names.

The review covers all five promoted edits, eleven tested-unpromoted edits, eleven distinct untested drafts, and 239 failed held-in attempts. Only S2 and S4 were promoted. Their three selected batches improved both exact matches and F1, but the successful traces do not establish that every added rule was responsible. The final harness also retains incompatible child-output contracts. [Full analysis](full_report.md), [edit-by-edit assessment](appendices/edit-by-edit.md).

New findings:

- **S8 was never eligible.** Deterministic merge/parsing helpers are excluded by the current routes for those mechanisms.
- **S5 and S7 never received expanded evidence.** They had eligible patterns, but only coverage and exhaustion mechanisms were expanded during this experiment.
- **All 233 successful attributions were labeled causal**, with no child-verifier grounding. Some coverage diagnoses instead describe uncertain or incorrect semantic judgments.
- **Seven of sixteen tested edits targeted unexpanded patterns.** Their compact summaries could not substitute for complete relevant operations.
- **History reached 47,342 characters**, within a 100,283-character round-8 system prompt.
- **S6 proposals misdescribed runtime behavior.** Local probes confirm syntax retries repeat the same prompt, timeout strings do not trigger them, and a batch-width cap refuses rather than splits work.

[Reproducible census](data/census.json), [surface table](data/surface-funnel.csv), [runtime probes](data/runtime-probes.json).

The recommended changes are:

1. **Capability contracts and operation-supported routing.** Show what each surface observes, when it runs, what it changes, and what it cannot do. Permit S8 parsing/merging and S5 recovery where the operation supports them; retain genuine scope restrictions.
2. **Smaller complete evidence packets and bounded history.** Allocate several relevant operations before expanding surrounding context, preserve distinct-instance ranking, and keep a full archive outside the prompt.
3. **Observation-based mining.** Separate evidence from inference and unresolved failure from successful recovery. Give semantic judgment and task-condition errors explicit homes in the vocabulary.
4. **Choose intervention form before replacement text.** Compare a short instruction, executable helper, and conditional skill. Require a plausible call/load path for helpers and skills.
5. **Behavioral witnesses and merged-contract checks.** A tiny fixture should demonstrate the claimed code/policy difference and preserve a valid case. Check that retained examples and sibling edits agree on intermediate formats.
6. **History that requires a substantive revision.** Record activation and combined outcomes alongside exact/dense quality and cost. A rejected F1 gain can be promising; it does not justify replaying the same inactive policy.

The [prompt-template appendix](appendices/prompt-templates.md) supplies concrete task-agnostic wording and examples, including a multiplicity-preserving S8 helper, scoped S5/S10 recovery, and capability checks for S6/S7/S9. These are illustrative, not deployed candidates or claims of measured improvement.

Start with capability contracts, S8/S5 routing, and evidence allocation; evaluate those against stored held-in bundles from several environments under equal proposal budgets. Preserve one edit per surface, merged validation, held-out-only promotion, and current repeat settings. Measure the chain from available opportunity to activated behavior and task outcome, rather than optimizing surface diversity alone.

No experiment was restarted, no production optimizer was changed, and no paid model calls were made. The repository paper was not edited. The [source notes](sources/README.md) explain the primary research grounding and limitations.
