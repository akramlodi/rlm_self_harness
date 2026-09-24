# Making harness optimization discover better interventions across surfaces

September 23, 2026. Research recommendations, not implemented changes.

This investigation covers every promoted, tested-unpromoted, and unmaterialized edit in `experiment_oolong_pairs_dsv4f_20260916_131837`, its 239 held-in failure records, and the current optimization code. Ten relevant optimizer and runtime files match the frozen evaluation snapshot byte for byte, as recorded in the census. The earlier [complete edit report](../../docs/analysis/oolong-pairs-2026-09-23/report.md) contains the exact replacements; [Appendix A](appendices/edit-by-edit.md) gives this investigation's interpretation of all 27 distinct edit payloads. [Reproducible census](data/census.json).

## Main finding

**The optimizer needs a better connection between an observed operation, an editable surface's actual capabilities, and a testable change. Asking for more varied surface names will not establish that connection.** The current process gives coverage-related text edits repeated opportunities, gives some other surfaces no detailed evidence, and permits explanations that attribute capabilities to runtime settings they do not have.

This is not evidence that S2 and S4 are inherently the only useful surfaces. It is also not evidence that every unused surface would improve this task. S8 was never eligible for any mined pattern; S5 and S7 had eligible patterns but none received expanded evidence; S10 was eligible throughout but its two drafts collided before evaluation. Those are different problems and require different remedies. [Surface census](data/surface-funnel.csv), [routing implementation](../../shrlm/optimization/taxonomy.py).

I recommend six bounded changes, in order:

1. Give the proposer accurate surface capability contracts and route by the operation being changed.
2. Allocate evidence to several complete operations, and bound history separately.
3. Make mining distinguish observation, uncertainty, persistent failure, and successful recovery.
4. Compare instruction, helper, and reusable-procedure interventions before writing replacements.
5. Test the claimed behavioral difference and check the merged batch for incompatible contracts.
6. Record what a previous intervention actually changed, alongside its measured quality and cost.

These preserve one edit per surface, a single merged candidate, held-out-only promotion, and the existing validation repetition settings. They do not require a new search architecture or another paid critic for every proposal. Several can use existing artifacts and deterministic local checks.

## What the experiment establishes

Five edits were promoted: S4 in round 3; S4 and S2 together in round 4; S2 and S4 together in round 5. Eleven other edits reached validation; eleven distinct draft payloads did not. Only S2 and S4 differ between the original and final harness. [Original report and exact appendices](../../docs/analysis/oolong-pairs-2026-09-23/report.md).

| Round | Tested surfaces | Exact passes | Mean F1 | Selection |
| --- | --- | --- | --- | --- |
| 1 | S4 + S9 | 2 → 2 | 0.7085 → 0.5429 | Rejected |
| 2 | S2 + S4 + S3 | 1 → 0 | 0.7156 → 0.6098 | Rejected |
| 3 | S4 | 2 → 3 | 0.5980 → 0.6791 | Promoted |
| 4 | S4 + S2 | 1 → 2 | 0.6532 → 0.7250 | Promoted |
| 5 | S2 + S4 | 1 → 2 | 0.6859 → 0.7354 | Promoted |
| 6 | S6 | 1 → 0 | 0.6194 → 0.7193 | Rejected |
| 7 | S4 + S3 | 1 → 1 | 0.6355 → 0.5916 | Rejected |
| 8 | S6 + S4 + S3 | 3 → 1 | 0.6128 → 0.6951 | Rejected |

Each row contains ten attempts per condition on that round's fresh baseline and candidate. F1 is the mean of recorded three-decimal diagnostics, with known terminal failures without scored answers contributing zero. These are separate comparisons, not a cumulative learning curve. All measured batches passed the cost band. [Numerical source](../../docs/analysis/oolong-pairs-2026-09-23/rounds.csv).

The promoted changes made checking and repair more explicit, but their causal story is mixed. A successful round-3 trace already had complete record coverage before semantic reclassification improved its answer. A successful round-5 trace attempted the newly prescribed global rerun, encountered a malformed result, and ultimately combined old chunks with one repaired chunk. Therefore neither “coverage fixed everything” nor “rerunning all chunks caused the gain” is justified. [Verified trace excerpts](../../docs/analysis/oolong-pairs-2026-09-23/trace-evidence.json).

The final incumbent scored 1/10, 1/10, and 3/10 exact on fresh baseline runs in rounds 6–8. Its F1 was 0.6194, 0.6355, and 0.6128. A single extra exact pass under `v=1` is useful selection feedback but weak causal evidence. Bundled scores cannot identify the contribution of an individual edit. [Round audit](../../docs/analysis/oolong-pairs-2026-09-23/audit.json).

## Where opportunities disappear

### 1. Surface access is unequal before generation

“Eligible rounds” means at least one mined pattern allowed the surface. “Expanded rounds” means at least one such pattern received a full evidence expansion; it does not mean the selected proposal for that surface used that pattern. Draft counts deduplicate identical payloads within a round.

| Surface | Eligible rounds / 8 | Expanded rounds / 8 | Tested edits | Promoted edits | Untested drafts |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 factual REPL contract | 5 | 4 | 0 | 0 | 1 |
| S2 decomposition | 8 | 7 | 3 | 2 | 3 |
| S3 execution | 8 | 8 | 3 | 0 | 0 |
| S4 verification | 8 | 7 | 7 | 3 | 1 |
| S5 recovery | 4 | 0 | 0 | 0 | 1 |
| S6 runtime policy | 5 | 4 | 2 | 0 | 1 |
| S7 execution-result metadata | 3 | 0 | 0 | 0 | 0 |
| S8 REPL helpers | 0 | 0 | 0 | 0 | 0 |
| S9 answer middleware | 3 | 0 | 1 | 0 | 2 |
| S10 skills | 8 | 8 | 0 | 0 | 2 |

The table rules out a useful interpretation of “S8 failed to improve performance”: it never had a legal opportunity. The current map permits S8 for `repl_execution_fault` and `other`, but not for `lossy_aggregation` or `unparsed_child_output`, although deterministic merge and parsing helpers can address those operations. No attributed run used either S8-eligible mechanism. [Census](data/census.json), [mechanism-to-surface map](../../shrlm/optimization/taxonomy.py).

Conversely, S3 and S10 did have access to expanded patterns. Their problem cannot be explained solely by evidence starvation. S3 mostly became formatting advice; S10's two proposed procedures were submitted together and rejected for occupying the same surface. Both S10 drafts pass isolated skill-record validation, but that establishes format validity only, not useful behavior. [Edit analysis](appendices/edit-by-edit.md).

### 2. Mining reports certainty without distinguishing uncertainty

Across eight rounds, there were 239 failed held-in attempts. Six had no successful attribution. **All 233 attributed failures were labeled `causal`; none had child-verifier grounding.** Of the 239 records, 169 were labeled `incomplete_coverage`, 26 `misaligned_unit`, and 19 `iteration_budget_exhaustion`. There were no `other`, `contributing`, or `correlated` attributed signatures. [Saved record inventory](data/mining-records.json).

Lack of child grounding does not make an observed root overwrite or skipped slice uncertain: code can establish those directly. But uniform causal labels cannot help distinguish a proven overwrite from a conjecture about unverified classifications. For example, round 1's held-in run `oolong-t01-w9-ab9956978b367bac__a01` is classified as `incomplete_coverage` while its symptom summary says incorrect categories admitted extra users; its verification limits explicitly say classifications were not checked. That record does not establish a skipped-record mechanism. [Record inventory](data/mining-records.json), [attribution prompt and validation](../../shrlm/optimization/attribution.py).

This matters downstream: the enum controls eligible surfaces, grouping, and reading order. The existing prompt already says to use `other` for semantic failures outside the vocabulary, but that path was not used. Another reminder alone is unlikely to solve the issue. The closed vocabulary lacks a clear home for incorrect semantic judgments and incorrectly implemented task conditions, and the structural validator verifies coordinates rather than the truth of the diagnosis. [Taxonomy](../../shrlm/optimization/taxonomy.py), [attribution validation](../../shrlm/optimization/attribution.py).

### 3. Evidence became shorter, but not sufficiently distributed

The latest code already preserves complete code operations and caps the evidence section at 32,000 characters. The remaining problem is the size of the unit it admits: a context can carry six operations, and a contrast can add six more. A large first example plus its contrast can consume nearly the whole allocation. [Evidence selection and packing](../../shrlm/optimization/proposal_evidence.py).

| Round | Expanded patterns | Expanded mechanisms | System-prompt characters | History characters |
| --- | ---: | --- | ---: | ---: |
| 1 | 1 | coverage | 45,584 | 60 |
| 2 | 1 | coverage | 53,960 | 6,547 |
| 3 | 3 | coverage, exhaustion, exhaustion | 65,728 | 16,154 |
| 4 | 1 | coverage | 67,656 | 21,298 |
| 5 | 1 | coverage | 84,022 | 28,454 |
| 6 | 1 | exhaustion | 84,171 | 33,735 |
| 7 | 2 | coverage, exhaustion | 94,880 | 38,842 |
| 8 | 2 | coverage, exhaustion | 100,283 | 47,342 |

Only coverage and exhaustion ever received expanded evidence. Seven of sixteen materialized candidates targeted patterns outside those expansions. They still saw compact inventory entries, so “no evidence at all” would be inaccurate; they lacked the detailed operation packet the new design was meant to supply. Notably, every tested S3 candidate and the tested S9 candidate came from an unexpanded pattern. [Reconstructed counts and history lengths](data/census.json).

There is also an ordering inconsistency: clustering prioritizes distinct-instance support, but the evidence packer sorts by run-level support. Repeated failures of the same task can therefore regain priority at the point where detailed evidence is allocated. Representative selection and alternatives also rely heavily on stable IDs, rather than choosing the clearest discriminating operation. [Clustering](../../shrlm/optimization/clustering.py), [evidence packer](../../shrlm/optimization/proposal_evidence.py).

### 4. Surface descriptions hide important operational distinctions

S6 is described as “every number and switch,” and its authoring format lists keys with an enable flag. It does not explain the actual semantics of individual controls. The rejected S6 proposals promised timeout recovery, splitting chunks, or limiting model-written retry loops. The implementation does none of those through `max_retries`. [Proposer formats](../../shrlm/optimization/proposal.py), [runtime implementation](../../rlm/environments/local_repl.py).

The local probes reproduce the discrepancy without a model or provider call:

- A timeout result triggers zero syntax retries.
- A syntax error triggers two additional calls with the identical prompt under the proposed policy.
- `max_batch_width` refuses an oversized batch; it does not queue or split the work.
- Disabling `retry_on_syntax_error` gives zero retries even when `max_retries` is set.
- The saved round-1 S9 “normalization” emits `redirect(normalized_text)`, asking the model for another turn rather than accepting a transformed answer directly.

[Probe source and results](data/runtime-probes.json). These tests establish API behavior, not downstream task performance.

Surface reach also needs field-level precision. S6's local enforcement policy, S7, and S9 are not passed into child RLM constructors. However, S6 `max_depth` is mapped to the RLM's depth scalar, which children inherit. The blanket root-only surface label misses that distinction. S1–S5's shared prompt reaches children; a root-oriented instruction can therefore affect recursive subproblems unless scoped. [Runner](../../shrlm/runner.py), [child construction](../../rlm/core/rlm.py), [reach annotations](../../shrlm/optimization/taxonomy.py).

## Six changes I would make

### 1. Use accurate capability contracts to select surfaces

**Small change:** replace vague surface summaries with short, versioned contracts stating inputs, trigger, actual effect, scope, and unavailable capabilities. Use the existing mechanism map as a preference, while admitting additional operation-supported routes through explicit capability checks. Start with S8 for deterministic parsing/merging and S5 for observed failed-call recovery; preserve the factual boundary of S1 and the limited visibility of S7/S9.

The proposer should answer: “Can this surface run at the failing point, see the necessary information, and perform the claimed change?” For S6 it must name a field and its actual branch; for S8 it must identify where the helper will be called; for S10 it must supply a retrieval trigger. See the [capability cards and prompt templates](appendices/prompt-templates.md).

Do not simply allow every surface for every mechanism. Reject a semantic validator in S9 that requires hidden records, a parser algorithm placed in the factual S1 contract, or a child-memory fix placed in root-only S7. These are genuine capability mismatches. Routing should stop those while permitting a merge helper for a demonstrated overwrite.

This directly addresses the S6 rationale failures and the absence of S8 opportunities. It also avoids interpreting a misplaced S1 or S9 draft as an invitation to relax legitimate surface boundaries.

### 2. Pack several minimal, complete operation packets

**Small change:** admit a failing operation and the smallest necessary producer/consumer context first; add surrounding operations and contrasts only after allocating space to other distinct, supported mechanisms. Rank with distinct-instance support consistently. Place a separate bound on rendered history and retain a compact index of every older attempt outside the expanded portion.

For example, allocate evidence across two or three inspectable mechanisms before letting one example consume twelve operations. The numbers are design starting points to test, not a universal quota. Preserve complete statements and dependencies; if a necessary operation cannot fit, mark it unavailable rather than cutting code mid-operation or pretending its summary is sufficient.

A candidate should require a usable operation packet or an explicitly weaker hypothesis with a bounded test. If its pattern was omitted, use the existing selection/repair opportunity to bring that packet into the same budget or withdraw the candidate. Do not let detailed-looking replacement text conceal absent evidence.

Apply the same completeness rule upstream to attribution. Its current digest still head/tail-clips execution code, even though the newer proposer evidence preserves complete operations. A detailed proposer packet cannot retroactively correct a diagnosis made from an omitted middle section. Select fewer complete relevant operations within the existing digest budget, and record when the necessary operation is unavailable. [Attribution digest](../../shrlm/optimization/digest.py).

Use surface coverage as an audit and an evidence tie-breaker: if two equally supported opportunities compete, prefer one whose relevant operation has not yet been inspected. This explores more surfaces without forcing proposals or lowering the promotion bar.

### 3. Mine what remained wrong, and what recovery already accomplished

**Small change:** make the attribution output separate the observed operation from its hypothesized consequence, including the checked universe, unresolved state, and what remains unverified. Add clear generic mechanism choices for semantic judgment error and task-condition implementation error. Preserve `other` and uncertainty states; an incorrect answer alone must not establish which upstream operation caused it.

A useful compact record is: `operation → observed violation → downstream consequence → recovery status → verification limit`. Existing citation fields can carry much of this. Keep diagnosis separate from proposing remedies; the attributor reports whether a result was discarded or reused, not which edit to generate.

Also extract bounded opportunity records from existing held-in successes and recoveries: repeatedly written parsing code, successful repair sequences, or repeated expensive rework. These are reusable procedures or efficiency opportunities, not newly invented failures. They can suggest S8/S10 without reclassifying a correct run as wrong or paying for another rollout. Structural extraction can nominate examples; semantic claims still need evidence.

This fixes a failure-only blind spot: `record_failure` exits immediately for a passing run. Passing traces currently serve as contrasts, but their successful procedures do not become independently selectable opportunities. Use only held-in traces for such distillation. [Mining entry point](../../shrlm/optimization/mining.py), [passing evidence construction](../../shrlm/optimization/proposal_evidence.py).

### 4. Compare intervention forms before writing text

**Small change:** for a supported operation, have the existing selection stage compare a short instruction, a deterministic helper, and a conditional reusable procedure, then choose the smallest effective form. Produce replacement text only for the winner. Do not require all forms to be viable or fill all candidate slots.

An S4 reminder to preserve multiplicity, an S3 merge procedure, and an S8 helper enforcing that merge are alternative implementations of one idea. The helper becomes attractive when the same deterministic operation recurs and can be tested; an S10 procedure becomes attractive when the operation requires several decisions, checks, and recovery steps. One-line global advice still belongs in S2/S3, not a skill.

Clarify S10's current prohibition against restating decomposition/execution guidance: avoid duplicating always-on instructions, while allowing a scoped procedure that necessarily contains execution steps. The current wording is broad enough to discourage useful procedures. Make the skill description say when to load it and what inputs it needs, not merely name a general topic. [Current skill contract](../../shrlm/optimization/skill_edit.py).

For each form, require an adoption path. A helper advertised through its docstring may be usable without an additional S3 edit; a skill needs a recognizable trigger. Adding an unused tool or unloaded skill cannot be credited with a claimed effect. Tool descriptions are derived from callable docstrings in the current runtime. [Tool registration](../../rlm/environments/base_env.py).

There is an additional constraint worth revisiting later: the validator forbids multiple edits to the same **pattern**, as well as the same surface. This can block a helper plus an essential call-site instruction for one mechanism, even though the user-required rule is one edit per **surface**. If adoption cannot be achieved within one surface, consider a narrowly justified pair on different surfaces, still merged and evaluated once. This requires updating retained-member repair semantics too; it is not merely deleting a duplicate check. It is lower priority than making single-surface proposals effective. [Batch validation](../../shrlm/optimization/proposal.py).

### 5. Make each claimed change produce a behavioral witness

**Small change:** extend the existing local preflight with a tiny relevant input that distinguishes incumbent and candidate behavior, plus an unchanged valid case. Check actual hook output for code/policy edits; for instruction edits, require an explicit operation-level example and verify executable examples where possible. After merging, check that sibling edits and retained examples agree on their producer/consumer contracts.

This would catch a syntax retry claiming to split timeout chunks and a supposed direct normalizer returning a redirect. A shuffled, duplicated, or incomplete synthetic input can test an S8 merge helper. S7 can be tested with a long output followed by an error or summary, while checking its declared bound. None of these tests establishes semantic label correctness or replaces held-out evaluation.

The S2 changes demonstrate why checking only isolated new text is insufficient: the promoted example simultaneously requests label counts and complete user mappings, while retaining a parser that sums counts. A useful revision should replace or explicitly scope that example. The current prompt already requests this; a producer/consumer compatibility check makes the requirement concrete.

The same applies to proposed global repairs: state which previously completed work is invalidated and why. Rerun all pieces only when the changed assumption affects all pieces. Otherwise preserve valid work and repair the affected subset. This is general dependency reasoning, not an OOLONG solution recipe.

Use the existing bounded isolation for executable fixtures. LLM-authored examples remain hypotheses; passing a self-authored test is evidence of an implementation difference, not proof of overall quality. Keep valid siblings during repair, and make a collision repair choose one contender or withdraw them. For a field-length rejection, let repair correct that field without regenerating the valid replacement payload; several otherwise inspectable drafts failed on the 600-character selection-reason limit.

### 6. Make history about tested behavior, not repeated persuasive descriptions

**Small change:** give each attempted intervention a compact record of its operation, actual change, evaluation unit, outcome, and reason a retry would be different. Add available activation observations, such as retries triggered or skills loaded, without attributing a batch score to a member. Require a resubmission to name the closest prior attempt and the substantive revision.

The current prompt already displays diagnostic progress and explicitly forbids replaying the same edit. Nevertheless, rounds 6 and 8 repeat the same S6 policy. Retelling the claimed timeout strategy in a growing history is less useful than recording “syntax retry policy; timeout branch unaffected; rejected; no positive persisted retry count.” Persist the full archive, but expand only the most relevant predecessors and recent constraints in the prompt.

Retain partial-credit gains as **promising but unconfirmed**, with contrary exact/cost outcomes. Round 6's F1 increase largely includes recovery from an unscored answer; it does not demonstrate that S6's retry mechanism helped. Round 8's F1 increase accompanies two lost exact passes and a merged candidate. Their history should invite a changed, evidence-backed hypothesis, not a duplicate policy or a blanket blacklist. [Original F1 analysis](../../docs/analysis/oolong-pairs-2026-09-23/report.md).

The existing diagnostic comparison already checks metric name, direction, aggregation, missing-value policy, and comparable definitions before labeling progress. Preserve that behavior. It currently recognizes three environments in one function; for a new task, have the verifier supply a versioned metric descriptor rather than teach the generic proposer to parse an F1 string. When no comparable dense metric exists, report exact outcomes and execution diagnostics with `not_assessed`; do not invent a surrogate quality score. [Quality diagnostics](../../shrlm/optimization/proposal_evidence.py).

## What useful edits beyond S2/S4 could look like

These are illustrative intervention classes, not proposed incumbent replacements or predictions of improvement. Each requires corresponding held-in evidence. They intentionally omit benchmark categories, record layouts, task IDs, and answer-specific recipes.

| Surface | Example intervention | Evidence needed | General proposer question |
| --- | --- | --- | --- |
| S1 | Clarify that returned tool errors are values that must be inspected, or correct a misunderstood submission API. | The execution demonstrably misunderstood a factual API contract. | Which true environment fact was misunderstood? Keep strategy elsewhere. |
| S3 | Retain multiplicity, provenance, or ordering until the task's conditions have been applied; aggregate only after proving those details are no longer needed. | A specific lossy transformation occurs before a condition that needs that information. | What information is irreversibly removed here, and which remaining operation needs it? |
| S5 | Preserve valid sub-results, distinguish malformed output from semantic disagreement, and retry only affected work after changing the cause. | Failed/recovered calls and a visible consumer of their results. | What failed, what has already recovered, what must change before retry, and what work remains valid? |
| S6 | Set an actual guard on a demonstrated harmful root call pattern, with a known refusal-handling path; use syntax retries only for observed syntax-classified failures. | The guard/retry branch can activate at the right scope and has a measured failure mode to address. | What exact runtime branch changes, and what happens immediately afterward? |
| S7 | Preserve the error or final diagnostic tail of long execution output within the existing metadata bound, while retaining a useful beginning. | Truncation hid information needed on a later root turn. | Which visible bytes must survive, and can this hook actually observe them? |
| S8 | Add a pure `merge_grouped_values(parts)` helper that concatenates values for repeated keys instead of overwriting them. | Recurrent deterministic merge errors or costly repeated merge implementation. | Can a small, task-parameterized function enforce this operation and be used at the failing call site? |
| S9 | Normalize an unambiguous presentation defect by accepting transformed content, or issue a precise repair nudge for a stable answer-visible contract. | A real answer-visible defect and an environment-wide contract; valid alternatives and empty answers are accounted for. | Can correctness of this transformation be established from the answer alone? |
| S10 | Add a conditional procedure for diagnosing and repairing a failed batch: preserve successes, isolate the failure class, revise the affected operation, verify completeness before consuming results. | A reusable sequence supported by successful held-in recovery; a clear loading trigger. | Which multi-step procedure is repeatedly rediscovered, and when should it be loaded rather than run globally? |

The S8 helper applies to document extraction, grouped measurements, graph adjacency accumulation, and classification outputs; it need not encode a task predicate. It must not silently deduplicate values when multiplicity matters. S7 cannot read unprinted REPL values, S9 cannot validate hidden semantic classifications, and S6 caps are not adaptive schedulers. Scope these examples to actual capabilities rather than making them sound broadly powerful. [Harness surfaces](../../shrlm/rlm_harness.py), [runtime](../../rlm/environments/local_repl.py).

S1 and S9 should probably change less often than S3, S5, S8, or S10: factual contracts and answer middleware are narrow responsibilities. A productive optimizer need not promote every surface. The desired result is that supported operational interventions can compete fairly and demonstrate value.

## What external research contributes

**GEPA** uses module-specific execution feedback and rotates its target module, giving components explicit opportunities for updates. Its paper also separates accessible training feedback from restricted validation content. The useful lesson here is to allocate attention to editable components and their actual intermediate behavior. Borrowing that principle does not require adopting its candidate population or Pareto search. Its results do not prove that rotating these ten heterogeneous surfaces would help, so I recommend rotating evidence inspection opportunities rather than forcing edits. Agrawal et al., July 25, 2025, §3.2 and §4, printed pp. 6–7. [Primary paper](https://arxiv.org/pdf/2507.19457v1).

**TextGrad** frames feedback as criticism directed at individual variables in a computation graph, with natural-language constraints and optimization history. This supports focusing a proposal on the operation and component it can change. It does not make an LLM's causal diagnosis reliable; the present all-causal attribution distribution is precisely why observed facts and inferred consequences should be separated. Yuksekgonul et al., June 11, 2024, §2 and Appendix B. [Primary paper](https://arxiv.org/html/2406.07496v1).

**AgentOptimizer** updates agent functions using execution history, performance, and the current function set, with add/revise/remove actions rather than replacing everything. Its prompt asks for functions useful on future problems. This is concrete precedent for treating reusable executable functions as optimization outputs, which makes S8 worth opening to appropriate operations. Its benchmark results are not evidence that a particular helper here will be used or improve accuracy. Zhang et al., February 17, 2024, §2.1 and Appendix D.1. [Primary paper, original version](https://arxiv.org/html/2402.11359v1).

The recommendations above are primarily supported by this repository's artifacts and code. These papers provide design precedents, not a justification for importing their full algorithms. [Source notes and quality assessment](sources/README.md).

## How to test the improvements cheaply and generally

First, evaluate proposal quality against frozen **held-in** bundles from OOLONG-Pairs, GraphWalks, and OOLONG under equal proposal-call/token budgets. Include synthetic operation fixtures for capabilities not represented in those bundles. Do not replay existing held-out task details as proposal input. The earlier cross-experiment investigation is useful evidence of recurring errors, but the old experiments differ in models and gates and are not a controlled comparison. [Earlier investigation](../../docs/analysis/2026-09-15-task-agnostic-proposer-review.md).

Start with deterministic checks that cost no model calls: reproduce surface eligibility, evidence allocation, history size, valid fixture behavior, and policy activation. Then compare the current proposer with the proposed capability/routing/evidence changes using the same stored bundles. Add diagnosis/procedure/history changes only after the first comparison shows better grounded candidates. The research itself made no paid model calls.

Measure:

- Fraction of candidates supported by a complete, relevant operation packet.
- Agreement between claimed effect and actual hook behavior.
- Unsupported causal claims, valid-but-redundant edits, contradictory producer/consumer contracts, and repeats without a substantive revision.
- Valid candidates per proposal dollar, including retained siblings after repair.
- For each surface: opportunity observed → evidence shown → candidate admitted → behavior activated → batch promoted. Use opportunity-conditioned rates, not raw diversity counts.
- Actual exact outcomes, available dense quality metrics, completion rate, cost, and time under the unchanged promotion protocol.

The last two stages require execution. A better prompt can improve proposal validity without improving task outcomes. A merged promotion still only establishes a batch result. If individual causal attribution later matters, a separate, deliberately small ablation is needed; do not add per-edit validation to the production loop by default.

The long-context evaluation stopped after only three completed edited attempts on one task. It cannot establish a general baseline-versus-edited difference. It does reveal that the short optimization workload and deployment-scale workload need separate consideration. For future research, use a disjoint development stress set or synthetic size multipliers to test whether a proposed repair multiplies work; do not silently reuse final-test records for mining. The original final-test boundary has already been inspected in this audit, so any future claim involving design informed by it should identify a fresh independent evaluation. [Stopped evaluation record](../../experiment_oolong_pairs_dsv4f_20260916_131837/eval/paired_20260923T140326Z/README.md).

## Implementation boundaries

| Change | Existing files to touch if approved for implementation | Budget impact |
| --- | --- | --- |
| Capability contracts and justified surface routes | `taxonomy.py`, `proposal.py`; contracts verified against `runner.py` and runtime | Small replacement prompt blocks; no required extra model call |
| Operation-level evidence allocation and bounded history | `proposal_evidence.py`, `clustering.py`, history rendering in `proposal.py` | Same or smaller prompt budget |
| Observed/inferred/recovered diagnosis and semantic mechanisms | `attribution.py`, `taxonomy.py`, `types.py`, `digest.py` | Replace diagnosis fields; keep bounded evidence |
| Procedure/helper alternatives and adoption | Existing selection instructions in `proposal.py`, clarify `skill_edit.py` contract | Same response, alternatives kept short |
| Behavioral fixtures and batch consistency | Existing candidate preflight and merged validation setup | Bounded local CPU; do not add paid validation arms |
| Compact behavior history and generic metrics | `proposal_evidence.py`, `proposal.py`, verifier diagnostic interface | Lower history growth; optional aggregate instrumentation |

Version changed prompts, taxonomy, evidence selection, and candidate contracts so caches cannot mix old and new behavior. The current code already has these versioning mechanisms. Preserve host-owned template escaping, local structural checks, valid-sibling retention, and the existing F1/score progress annotations; those are foundations to improve, not missing features to implement again.

The highest-value first slice is **accurate capability contracts, operation-supported S8/S5 routing, and evidence allocation that exposes their opportunities**. Then make those candidates demonstrate an actual behavior change. That is a more plausible route to useful promotions beyond S2/S4 than surface quotas, longer patience, or another layer of generic reminders.
