---
title: Proposal quality changes for a future paper update
date: 2026-09-14
type: docs
---

# Proposal quality changes for a future paper update


This note records the motivation and implemented methodological changes following `experiment_oolong_pairs_dsv4f_20260913_132607`.
The September 14 changes below were implemented and verified with offline tests and read-only replay of saved artifacts. At that note's writing time, no experiment using them had been run; the subsequent experiment is reviewed in the September 15 section at the end.
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

The September 14 iteration used attribution prompt 1.3.0 / validator 1.1.0, taxonomy 3.2.0, digest 1.4.0, proposal prompt/validator 2.1.0, evidence selector 2.0.0 and diagnostic history 1.0.0. The additive proposal envelope remains v1. New live responses require the new fields; completed historical artifacts remain readable.

## Next round of proposer improvements: September 15

The five changes in this section are **implemented and checked offline**. Their effect on live proposal quality and promotions remains unmeasured.
The [September 15 investigation](2026-09-15-task-agnostic-proposer-review.md) and [machine-readable audit](2026-09-15-proposer-review-audit.json) record the evidence.
The implementation plan is [Task-agnostic proposer improvements](../plans/2026-09-15-1643-fix-task-agnostic-proposer-plan.md).

The subsequent run, `experiment_oolong_pairs_dsv4f_20260914_111506`, completed three rounds with zero promotions.
Five of six S2 submissions failed template formatting, and both round-1 responses repeated two competing S3 edits.
Its admitted S2 intervention checked coverage while the cited failure concerned semantic labels; another contender proposed recovery that the trace had already performed.
Reconstructed prompts were approximately 158–174k characters, with roughly 35k of legacy answer dumps per round alongside newer diagnostics.
These observations identify proposal construction and reasoning problems; they do not establish the effect of the remedies.

1. **Let the host encode instruction templates.** The proposer writes literal text, and the materializer encodes it once under an explicit versioned contract. Current surfaces are displayed in the same representation, with a distinct marker for the live tool slot. Formatting preflight remains, while saved harnesses retain their existing representation and hashes. The model still needs valid response JSON; this change removes the additional template-escaping layer.

2. **Compare the edit with the observed execution.** Refine the existing explanation fields to name the unresolved operation and the action or value the edit changes there. Ask whether perfect compliance with the edit could still produce the demonstrated failure for the same reason. Preserve uncertainty about labels and parsing, and distinguish a recovered intermediate error from an unresolved failure. These are proposal reasoning requirements, not a claim that deterministic checks can prove semantic relevance.

3. **Choose interventions before writing replacements.** Within the existing proposer call, select one evidence-supported mechanism per surface, then generate matching replacements. Validate that declared selection and give collision repair the conflicting contenders and occupied surfaces explicitly. Repair chooses or withdraws contenders while retaining valid siblings; it does not combine unrelated ideas to evade the one-edit-per-surface rule.

4. **Derive procedures from the task's information needs.** Replace the detailed OOLONG record-ID recipe in the generic proposer with questions about information that must survive intermediate steps and conditions the final computation must enforce. Counts, dates, identity, ordering, provenance, units, and asymmetric roles are broadly useful considerations. For example, a set cannot preserve the multiplicity needed by an “exactly one” predicate. Keep authoritative environment and surface contracts, including the OOLONG Pairs answer format.

5. **Provide fewer complete evidence examples.** Retain a compact pattern inventory, then expand a few distinct mechanisms under one budget for rendered evidence. Prefer complete relevant code and a supported successful or recovered contrast; omit an oversized operation explicitly instead of cutting through it. Replace repeated raw answers with trusted diagnostics and bounded examples, retain bounded original evidence for unknown formats, and count serialization overhead in the budget. Full current surfaces and existing history remain available and separately accounted.

### Interpretation and future measurements

This iteration targets usable, distinct, causally relevant proposals across tasks.
It preserves held-in-only proposal evidence, aggregate-only held-out history, potentially-promising annotations, combined held-out validation, the existing promotion and cost gates, and `v=1`.
It adds no separate selection call, model judge, validation arm, or unbounded repair loop.
Held-out traces discussed in the investigation remain audit evidence and must not be forwarded as instance-specific live proposal context.

The implementation uses `literal-text/v1` and `proposal-selection/v1`, with proposal
prompt/validator and evidence selector `3.0.0`. Diagnostic history stays `1.0.0`;
stored harness v2 and proposal v1 envelopes remain unchanged. Live contract
changes refuse unfinished replay before another model call. Use a new experiment
directory for a later live evaluation.

Offline reconstruction used the three saved held-in mining rounds and the same
aggregate history from `experiment_oolong_pairs_dsv4f_20260914_111506`. The reader
verified trace hashes; the watched mining bundles, manifests, records, instances
and harness files retained their hashes. No model calls were made.

| Round | Previous system prompt | Revised system prompt | Evidence characters | Expanded pattern indices |
|---|---:|---:|---:|---|
| 1 | 157,881 | 50,376 | 31,336 | 0, 3 |
| 2 | 169,363 | 55,964 | 31,000 | 0 |
| 3 | 173,945 | 56,964 | 30,227 | 0, 1 |

Counts include serialized text; the system-prompt totals include complete surfaces
and history. Repair adds its own user-message overhead, recorded per attempt in
the proposal audit. The 32,000-character cap applies only to evidence. Each saved
round retained its full inventory; one or two patterns fit with complete operations
and a passing contrast. This verifies size reduction, not better model reasoning.

Regression checks cover literal braces/f-strings, marker collision and no-op
round trips, stored candidate loading, selection matching, duplicate keys,
collision repair with retained siblings, zero-call replay, rendered budget
boundaries, alternate representatives, intact operations/questions, relevant
contrasts, and held-out payload exclusion. Known-zero failed runs retain their
bounded verifier error detail and explicit failure cause instead of losing that
information when a zero score is available. The existing generic history checks for potentially promising directions remain
in place.
A later live evaluation should measure usable proposals per call, formatting and collision rejection rates, relevance to the cited unresolved operation, and exact and secondary validation outcomes.
More promotions or longer runs remain hypotheses until measured; cross-run comparisons and `v=1` alone do not establish causality.
The separate suggestions for a new history-revision requirement and executable-example probes are deferred.
The paper draft is unchanged.

### September 15 implementation and verification

Implementation commit: `ee0b4999` (`fix/task-agnostic-proposer`), based on main
`df2580f3`. The paper and experiment configuration were not edited.

- The final focused proposer/evidence/initial-harness checks passed: **206 tests**.
  The repaired end-to-end experiment and smoke checks passed: **33 tests**.
- The full suite finished with **2,363 passed, 7 skipped, 21 deselected and 11
  failures**. One failure was a stale prompt-format assertion, subsequently
  updated and retested. Rechecking all failures left **10 failures and 1 pass**.
  All ten remaining failures reproduced on an isolated `df2580f3` checkout:
  one async test lacking its pytest plugin in this environment, five existing
  config/split expectation mismatches, and four missing smoke-artifact fixtures.
- Changed-file Ruff, formatting and configured pre-commit hooks passed. The
  type hook is advisory (`--exit-zero`). Repository-wide lint/hooks still fail
  on existing example names and historical generated experiment modules;
  incidental hook rewrites outside this change were restored.
- The code-review and simplification checks ran sequentially in the main agent,
  as required by this repository's AGENTS.md. No independent reviewer or
  cross-model corroboration is claimed. Review found and corrected the
  known-zero/error-detail loss described above; no actionable findings remain.

These checks verify construction, accounting and replay behavior. They do not
measure whether the new prompts yield more useful edits or promotions.

---

## September 23 capability, evidence, and history changes

This follow-up is **implemented and tested offline; experiment performance is not yet evaluated**. The [capability-aware proposer plan](../plans/2026-09-23-1119-fix-capability-aware-proposer-plan.md) covers the four areas selected after the [September 23 investigation](../../RESEARCH/meta-harness-optimization-2026-09-23/full_report.md). Earlier implementation records above remain historical descriptions of those iterations.

The latest investigation covered five promoted edits, eleven evaluated but unpromoted edits, eleven untested drafts, and 239 failed held-in attempts. S8 was never eligible, S5/S7 never received expanded evidence, and seven evaluated edits targeted unexpanded patterns. By round 8, prior history occupied 47,342 characters. Several S6 rationales described behavior that their runtime fields do not implement.

The implemented changes are:

1. **Describe real surface capabilities.** Share accurate descriptions of inputs, triggers, effects, scope, and limitations between attribution and proposal generation. Distinguish syntax retries from timeout recovery, refusal limits from scheduling, and installed helpers or skills from their actual use.
2. **Admit supported helpers and recovery instructions.** Add narrow S8 routes for parsing and aggregation, and S5 routes for visible recoverable operations associated with exhaustion or execution faults. Require relevant admitted held-in evidence; preserve existing surface boundaries and one edit per surface.
3. **Allocate complete evidence across mechanisms.** Reserve space for several core operation packets before optional context and contrasts. Respect distinct-instance support, preserve citation coordinates, and make code omissions explicit in attribution as well as proposal evidence. Keep the current proposer evidence cap.
4. **Learn from observed behavior and qualified quality signals.** Bound rendered history while retaining the archive. Separate claimed effects, trustworthy activation observations, and measured batch outcomes. Require an explicit revision relationship; duplicate rejection operates on the evaluated harness, so an unchanged member of a changed batch is not assigned an individual failure.

Metric definitions and structured measurements now come from the verifier, including direction, aggregation, precision, and treatment of unscored failures. Generic evidence and history consume this contract; strict legacy interpretation lives with the environments. Tests cover the real `oolong_synth` and `oolong_real` configurations as well as a custom lower-is-better metric with a nonzero terminal penalty. Unsupported or incomparable metrics remain `not_assessed`; missing measurements are not silently assigned zero.

The saved round-6 F1 gain of 0.6194→0.7193 accompanies an exact decline of 1/10→0/10. Round 8's gain of 0.6128→0.6951 accompanies 3/10→1/10 exact. Preserve both as potentially promising batch observations with their costs and failure counts. They do not establish reliable improvement under `v=1`, prove activation, or isolate a constituent edit's contribution.

Activation reporting initially uses existing syntax-retry, answer-redirect, and named skill-load events. Missing telemetry and uninstrumented instruction/helper behavior remain unassessed; absence of a record is not automatically zero. This is a bounded observational extension, not a semantic compliance judge.

The protocol preserves combined held-out-only validation, existing exact/cost gates, `v=1`, and held-in-only task evidence. It adds no paid critic or validation arm. Later reporting should distinguish route availability, evidence shown, candidate admission, activation coverage, and batch outcomes. The paper and experiment configuration remain unchanged.

### Implementation details relevant to future reporting

- Shared surface descriptions identify what each surface can observe and change. Newly eligible S8/S5 choices require admitted operation references; evidence omitted for size cannot authorize an alternate route. The host checks structural support, not the truth of the proposer's causal explanation.
- Proposer evidence remains capped at 32,000 characters, with at most four expanded patterns. Distinct-instance support and actionability determine ordering, distinct mechanisms receive complete core packets before optional context, and code blocks are included whole or explicitly omitted. The attribution digest applies whole-block selection within its 12,000-character default and prioritizes late failures before filling space with omission coordinates.
- Rendered history has a separate 12,000-character cap. The full host-side attempt index remains available for revision checks. Local proposal failures retain known rationale without inventing effective content or measurements; repair receives a bounded predecessor description when needed.
- Duplicate evaluation is checked after loader admission, against the complete effective harness and incumbent. A sibling rejected by the loader cannot hide a duplicate. An unchanged constituent in a changed batch remains eligible with an explicit revised joint hypothesis. No individual causal credit or blame is assigned from a shared batch outcome.
- Validation summaries persist existing root syntax-retry/answer-redirect observations and named skill loads across canonical call edges. History reads these aggregates without reopening held-out traces. An observed event does not establish that the intended procedure was followed; missing coverage and unsupported surfaces remain unassessed.
- Live proposal selection is versioned `proposal-selection/v2`; literal-text authoring remains `literal-text/v1`. Validation summaries use v3 and retain v1/v2 readers. Completed artifacts are read without mutation; incompatible unfinished contracts fail before paid work.

### Read-only historical check

Reading all eight rounds of `experiment_oolong_pairs_dsv4f_20260916_131837` preserved round 6's F1 0.6194→0.7193 with exact 1→0 and round 8's F1 0.6128→0.6951 with exact 3→1. Both are qualified positive batch signals, with historical activation unassessed. The resulting history occupied 8,730 characters. SHA-256 checks confirmed all 80 inspected saved artifacts were unchanged. This replay check made no model calls and did not inspect held-out trace bodies.

Review and simplification ran sequentially in the main agent under this repository's AGENTS.md. No independent reviewer or cross-model corroboration is claimed. Review corrected late-operation crowding in the digest and missing predecessor/rationale context in refused-attempt history. Offline checks establish construction, accounting, admission, and replay behavior; promotion rate and general task-performance effects remain unmeasured.

### September 23 verification

- Focused checks passed: 248 proposal/evidence/history/digest tests, 22 batch-validation tests, and 83 evidence/validation tests. These include a wide trace with a late failure, unsupported recovery routes, loader rejection exposing an otherwise hidden duplicate, changed batches reusing a constituent, nested event accounting, custom metric definitions, and legacy metric adapters.
- The full suite recorded **2,404 passed, 7 skipped, 21 deselected, and 12 failures**. Two stale test expectations for history/revision were corrected; both affected tests subsequently passed. All ten other failures reproduced on an untouched archive of starting commit `f5cf78de`: one missing async-test plugin, five existing configuration/split expectations, and four missing smoke-artifact fixtures.
- Changed-file Ruff, formatting, and configured pre-commit hooks passed. Repository-wide lint and hooks still flag existing example/training code and historical generated modules. All incidental hook rewrites to unrelated files were restored. The configured type hook is advisory (`--exit-zero`); standalone type checking still reports existing repository errors.
- No experiment was launched or resumed, and no performance or promotion-rate gain is claimed from these software checks.


## September 23 follow-up: surface ownership and diagnosis evidence

These changes are **implemented and checked offline; their effect on promotions and task performance has not been evaluated**. The [proposal admission and evidence plan](../plans/2026-09-23-1534-fix-proposal-admission-and-evidence-plan.md) follows the stopped experiment `experiment_oolong_pairs_dsv4f_20260923_123738`, launched from commit `0298b1ec`.

Round 1 produced three S2 proposals; repair moved all three to S4. Both batches were rejected for collisions, with no usable edit or validation. Its evidence prompt expanded one unattributed provider error and three coverage signatures, leaving aggregation and misaligned-unit mechanisms unexpanded. Mining scored 11/40 exact and mean F1 0.7589; these are incumbent mining results, not an edit comparison. The user stopped the experiment during round 2 after nine additional persisted attempts. Recorded stage spend was at least $4.44617; unfinished calls may add unrecorded spend. No promotion occurred.

The implemented changes are intentionally small:

1. **Make surface ownership a host decision.** The first candidate passing the existing local gates owns its surface and pattern for the round. Later collisions cannot remove it, and repair sees only eligible free surfaces. An invalid candidate does not reserve anything. This keeps the current single response and bounded repair; it cannot prevent duplicate text from being generated inside one model response.
2. **Search for compact evidence across signatures of a mechanism.** Choose a smaller complete representative before spending the budget on a large example or repeated mechanism. Keep unattributed patterns in the inventory but do not expand them or allow them to authorize proposals. Preserve actionable `other` mechanisms and the existing evidence-based S8/S5 routes; introduce no surface quotas.
3. **Show execution results and require a coverage-loss observation.** The two inspected filter diagnoses claimed records had been dropped, but their saved inputs and traces show all 188 record lines retained. Their attribution digests omitted the corresponding stdout counts. Pair selected code with bounded outputs and require the diagnosis to identify a missing input unit/range or a same-scope discrepancy. Missing answers and absent checks alone remain insufficient. Preserve a compact coverage assessment; unsupported or contradicted claims become unattributed without an extra model call.

These rules use task-independent concepts: operation, input universe, processed subset, observed omission, and uncertainty. They do not add an OOLONG parser or assume that complete coverage establishes correct labels. The host can validate evidence references and declared status, but it cannot prove the semantic truth of a model-written witness.

The protocol retains combined held-out-only validation, `v=1`, and existing exact/cost gates. Offline regressions can verify retained candidates, evidence diversity, output visibility, and honest status propagation. Better proposals, more promotions, and improved exact or dense-quality performance still require a subsequent experiment. The paper is unchanged.

Local evidence: `round-01-assessment.md`, `round-01-core-packet-audit.json`, the saved proposal responses, and the linked inputs/traces/digests under the experiment directory. The plan carries the findings independently of those ignored artifacts.


### Implementation and offline checks

The host now admits proposals in returned candidate order, retaining the first edit that passes spec validation, materialization, revision checks, and harness preflight for each surface and pattern. Attempt audits record admitted owners and individually refused collisions. Failed candidates reserve nothing; repair sees occupied surfaces and eligible free alternatives, and cannot replace retained owners. Empty repair preserves survivors. This guarantees admission exclusivity, not that the first contender is the best one.

Evidence allocation searches every available signature and representative within each actionable mechanism for the smallest complete serialized core. Mechanisms keep their existing support priority. Repeats follow the distinct-mechanism pass, then optional context and contrasts. Unattributed rows remain visible without eligible surfaces; an explicit zero actionability or missing resolvable operation keeps a packet out of expansion. Actionable `other` and evidence-supported S8/S5 routes remain available.

Attribution now packs each admitted whole code block with up to 500 characters per stdout/stderr stream, including labels and truncation markers. A live `incomplete_coverage` response must supply a bounded input scope, loss observation, counterevidence, and assessment status. `observed_loss` requires an operation or child-call citation. `not_established` and `contradicted` are accepted without re-asking and persist as `other`/`unattributed`, preserving the hypothesis and assessment. Historical details without the field retain their serialization and are described as coverage basis not assessed when projected into evidence. This validates declared evidence structure, not the truth of a model's causal judgment.

The changed contracts are proposer prompt/validator 4.1.0, evidence selector 4.1.0, attribution prompt 1.4.0, attribution validator 1.2.0, and digest 1.6.0. No additional paid stage, validation arm, or model repair was added. Experiment settings and the paper were not changed.

A [saved round-1 reconstruction](2026-09-23-proposal-admission-offline-audit.json) expands patterns `[2, 3, 1]`: coverage, misaligned unit, then another coverage signature. Previously it expanded `[7, 0, 1, 2]`: an unattributed provider error and three coverage signatures. The new evidence block is 31,115/32,000 characters, with two distinct actionable mechanisms shown. Aggregation remains inventory-only because its complete core does not fit alongside the higher-priority mechanisms. Existing historical coverage claims were not re-attributed or retroactively marked supported.

The reconstruction also identifies a limit: the early record-count operations still do not fit in either inspected long diagnosis digest. Pairing preserves outputs of selected operations, but cannot guarantee that every useful count appears. Their reconstructed lengths are 11,995 and 11,974 characters. The stricter coverage assessment and uncertainty path remain necessary; offline packing alone does not establish better diagnosis quality.

Review and simplification ran sequentially in the main agent under AGENTS.md; no independent or cross-model corroboration is claimed. Regression checks cover retained collisions, failures at each admission gate, unique IDs and replay, cross-signature representative selection, unattributed eligibility, incomplete evidence, coverage normalization and persistence, legacy details, and output-source accounting. The experiment remains stopped.


Validation: 320 focused proposal/evidence/digest/attribution/mining tests, 13 type/serialization tests, 98 orchestration/mock-smoke tests, and six smoke-analysis consumer tests passed (437 focused/integration checks total). The orchestration fixture now supplies the new live coverage declaration and a concrete root-operation citation. Changed-file Ruff, formatting, and all configured pre-commit hooks pass. Repository-wide Ruff/pre-commit still report the two pre-existing ambiguous-variable errors in `examples/oolong_pairs/tasks.py`; their automatic edits were confined to an isolated snapshot. Standalone type checking has 556 diagnostics on both the revised tree and starting commit `0298b1ec`, with no new diagnostics after normalizing shifted line numbers.


The full-suite run recorded 2,405 passed, 7 skipped, 21 deselected, 13 failures, and 23 setup errors. Three failures and the setup errors came from the shared orchestration fixture still emitting the superseded coverage shape; the affected pipeline checks were rerun after updating it. The ten unrelated failures were replayed on an isolated source snapshot of starting commit `0298b1ec`, and all ten reproduced: one missing async-test plugin, five configuration/split expectations, and four absent smoke-artifact fixtures. The full suite was not rerun after that fixture-only correction.

For a later authorized experiment, inspect proposal attempt `admissions`, occupied/free-surface repair context, evidence `expanded_mechanisms` and omission reasons, and saved `coverage_basis` declarations. Count retained usable edits and assess batch exact/dense-quality/cost results under the existing gates. A broader surface distribution or an asserted coverage witness alone is not evidence of improved task performance. No experiment was started as part of this implementation.
