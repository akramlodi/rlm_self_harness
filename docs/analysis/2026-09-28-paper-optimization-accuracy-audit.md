# Paper accuracy audit: mining, proposal, and validation

Date: 2026-09-28. Reviewed branch: `main`. Reviewed revision: `070d1a8493d79999ce4a5b01b19a2d0dc85b7637`.

**Verdict: the active drafts need substantive corrections before they can describe the implemented method or its experiments accurately.** The largest discrepancies concern the unit of validation, the promotion rule, the initial harness, the availability of child verification, and the independence and status of the experimental evidence. These are methodological differences, not terminology changes.

Here, the meta-harness means the optimizer that mines failures, proposes changes, and decides promotion; the candidate harness is the runtime configuration it edits.

This is a report-only review. No paper or implementation file was changed. The correction inventory covers every `.tex` file under `paper/`; it distinguishes current implementation, historical experiments, proposed future experiments, and archived/template documents. Findings are grouped by consequence, with all affected locations listed together. Line numbers refer to the reviewed revision.

The requested split between a short main description and a detailed appendix is appropriate. Joint validation, the verifier-defined promotion criterion, the actual starting harness, and adaptive use of validation belong in the main text. Prompt schemas, budgets, repair rules, runtime capabilities, and artifact formats belong in each active paper's appendix. Remove child-verifier descriptions from the papers as specified in #4; describe final-answer verification and model-inferred trace diagnoses directly.

## Author clarification: scope of the paper revision

The paper is to describe the current implementation and the forthcoming experiments. Historical launches and their version differences remain audit evidence, not a required narrative in the paper. Record the implementation/configuration used by the new study; do not add an account of previous development launches unless their results are actually reported.

- **Remove child-verifier-related method text, figures/legends, algorithm steps, claimed metrics, and ablations from the current papers.** Keep final-answer verification and ordinary root/child execution-trace descriptions. This is a paper correction, not a request to remove optional verifier integrations from the repository.
- **Keep joint batch promotion.** Lack of an individual causal estimate for each constituent is an accepted property of this design, not a request for per-edit validation or extra ablations. Describe the batch as the measured unit.
- **Keep the inclusive verifier-pass gate without adding an F1 requirement.** The general method uses the environment verifier's pass/fail outcome, with exact pair match as the OOLONG-Pairs instance of that contract. Optional verifier-defined dense metrics support diagnosis/history/reporting when available; F1 is only one example. Environments without a comparable dense metric remain `not_assessed` for that diagnostic. A falling secondary metric does not invalidate a promotion that satisfies the configured gate.
- **Keep the current split and describe its evaluation scope accurately.** Mining, validation, and short test contain disjoint task instances on shared short-context records. Describe short-test results as performance on held-out task instances. Use the separate long evaluation to assess performance on larger record collections, without claiming entirely unseen underlying question texts. Finding #7 is a description and interpretation requirement, not a requirement to redesign the split. A stronger split becomes necessary only if unseen-source-data generalization becomes a central claim.

These instructions govern the correction recommendations below. Scientific qualification of a result is separate from proposing a change to the optimization rule; this audit requests neither new promotion criteria nor a split redesign. The earlier context-separated split recommendation is superseded for the forthcoming study and retained only as an optional alternative in the linked split note.

## Review evidence and reproducible inventory

- [Structured findings](2026-09-28-paper-optimization-audit/findings.json): the same 29 numbered findings with source anchors and verification scope.
- [Machine-readable evidence](2026-09-28-paper-optimization-audit/evidence.json): all nine LaTeX files and hashes, ten saved OOLONG-Pairs experiment directories, frozen configurations where available, completed rounds, promotion decisions, measured subjects, recomputed recorded F1, and context hashes across splits.
- [Evidence collector](2026-09-28-paper-optimization-audit/collect_evidence.py): read-only reconstruction of that inventory; no model calls. Run with `.venv/bin/python docs/analysis/2026-09-28-paper-optimization-audit/collect_evidence.py` from this checkout. It expects the local experiment artifacts to exist.
- [LaTeX build checks](2026-09-28-paper-optimization-audit/latex-build-results.json): attempted builds and their actual failure points. Static include/reference checks supplement builds that stopped early.
- Existing investigations used as historical context: [September 14 methodology notes](2026-09-14-proposal-quality-methodology-notes.md), [September 23 edit report](oolong-pairs-2026-09-23/report.md), [September 25 three-round assessment](2026-09-25-three-round-inclusive-promotion-assessment.md), and [September 28 offline verification](2026-09-28-grounded-mining-proposal-verification.md). Numeric examples below were checked against saved manifests and decisions where stated, rather than accepted solely from these reports.

### Triage Groups

| Group | Findings | Recommended correction | Why |
|---|---|---|---|
| Actual algorithm | #1-4, #12-23 | Update the shared method account, then diagrams, pseudocode, and surface appendix in both active drafts | The prose currently describes a different optimizer and overstates what its checks establish |
| Experimental interpretation | #5-10, #24-26 | Describe the upcoming study's current protocol and split/evaluation definitions; use historical results only if explicitly reported | The new paper need not recount development launches, but must describe what its own evaluation measures |
| Cost and reproducibility | #11, #27-29 | Recompute budgets from the selected protocol; repair paper assembly and finish the submission checklist | Old run counts and incomplete document assembly would mislead reviewers |

All are proposed follow-up corrections, not authorization to change the papers in this review. Selecting the final study, transfer environment, and treatment of historical test exposure requires author judgment; documenting the verified current behavior does not.

## Verified current protocol

This table describes `configs/experiment_oolong_pairs_DeepSeekV4Flash.toml` at the reviewed revision. It is not a claim that every historical run used these settings, or that these settings are universal across environments.

| Component | Implemented/configured behavior | Primary evidence |
|---|---|---|
| Starting harness | `H0*R`: upstream-style prompt plus orchestrator addendum, a revised recursive-call contract, and a hand-authored decomposition example | Config:46; `shrlm/rlm_harness.py:609`, `:634`, `:696` |
| Dataset | Pinned OOLONG-Pairs reconstruction from `oolongbench/oolong-synth`, `trec_coarse`, revision `f0d59eaf0febf130664cfceb710436c8e3216b2b`; 20 task predicates per context window | Config:106; `shrlm/environments/oolong_pairs.py:1`, `:835` |
| Split sizes | 20 short mining instances, 10 short validation instances, 10 short test instances, 40 long test instances; partition seed 0 | Config:22; saved split manifest |
| Lengths | Upstream nominal context lengths 8,192 and 262,144, a nominal 32-fold difference; reconstructed prompts have different actual lengths | Config:112; environment module:22 |
| Mining | Two attempts per mining instance; failing runs may be diagnosed, with environment-owned failure exclusions | Config:36; `optimization/mining.py:67`, `:158`, `:239` |
| Child verification | No OOLONG-Pairs sub-verifier. No-recursion remains a structural observation; recursive causal attribution is otherwise model inferred | `experiment/orchestrator.py:299`; `optimization/grounding.py:75` |
| Proposal | At most four admitted edits, at most one per pattern and surface; fewer or zero are valid. All are based on the same incumbent | Config:38; `optimization/proposal.py:602`, `:1584`, `:2047` |
| Validation | Fresh incumbent and one composed candidate, held-out only, one attempt per instance. A single admitted edit is that candidate | `optimization/validation.py:971`, `:1022`; protocol `heldout-batch/v2` |
| Promotion | With both thresholds zero, candidate exact pass count must be at least incumbent count. Mean cost must be in `[0, 3]` times incumbent mean. Subcall-count band is unconstrained in this profile | Config:48; `optimization/promotion.py:293` |
| Dense metrics | Optional verifier-defined metrics inform diagnosis/history and reporting; they are not promotion conditions. OOLONG-Pairs supplies F1. Absent, unsupported, or incomparable metrics remain `not_assessed` | `optimization/proposal_evidence.py:48`, `:101`; `environments/diagnostics.py:9` |
| Stopping | At most three rounds; stop after three consecutive rounds without promotion. Freeze the final incumbent | Config:41; `experiment/orchestrator.py:832` |
| Final evaluation | Three attempts per task per condition, separate from `v=1`. Attempts have no per-attempt seed plumbing | Config:244; `experiment/evaluation.py:46` |
| Model/decoding | Azure Foundry deployment `DeepSeek-V4-Flash` for runner, attributor, proposer; temperature 0, top-p 0.95, output cap 8,192; default provider reasoning mode enabled | Config:13, :151, :171 |
| Runtime caps | 30 iterations, maximum depth 3, $2 per-run budget, 3,600-second timeout, $100 cumulative validation-subject budget | Config:64 |
| Concurrency | Mining run workers 3; validation subject workers 5 but at most two evaluated subjects, each with one run worker. Final evaluation exposes a separate run-worker argument | Config:244-283; `experiment/evaluation.py:58` |

Do not copy stale comments from this TOML into the paper: comments still mention 15-round, repeated dual-split/per-candidate scheduling and old cost bands. The parsed values, implementation, frozen run config, and manifests are the authorities.

### P1 -- High: corrections needed for scientific accuracy

#### #1. Validation is joint, held-out-only, and performed after composition

**Paper locations:** `proposal.tex:245`, `:350`, `:358-377`; `neurIPS_short_paper.tex:251`, `:343-371`; `proposal-old.tex:253`, `:382`, `:390-409`; `Figure_2.tex:173`, `:245-272`; `Figure_2B.tex:173`, `:245-259`.

The active text says the incumbent and **each candidate** are evaluated on **both mining and held-out data**, individually accepted candidates are merged, and the merge is evaluated again. The implemented loop instead loads/admission-checks edits, combines all usable disjoint edits, and evaluates only the incumbent and that combined candidate on held-out data. A singleton is evaluated once as the candidate. There is no per-edit tournament or subsequent merge-validation leg. No usable edit means no validation runs.

Evidence: `shrlm/optimization/validation.py:104-136`, `:913-1077`, especially `plan_batch` at :971 and subject selection at :1022. Constituent ledger rows at :1042 are `bundled`, with no independent scores; the composed subject owns the measured decision. `optimization/promotion.py` composes non-overlapping surface changes before scoring.

**Consequence:** the present papers misstate both the algorithm and the cost, and imply evidence of individual edit efficacy that the experiment did not collect. A promoted batch supports neither "every constituent improved performance" nor "this particular surface caused the gain."

Joint batch promotion is the intended, accepted design. Correct the attribution of measured outcomes in the paper; do not add individual edit evaluations to address this finding.

**Correction:** main text: "We compose admitted edits on distinct surfaces and compare the resulting harness with a freshly evaluated incumbent on validation tasks." Appendix: composition, singleton/empty cases, constituent accounting, and protocol versions. Change diagram arrows to **edits -> compose -> paired validation -> promote/reject batch**. Change "mining + validation" to "held-out validation"; remove the extra post-promotion merge loop.

#### #2. Describe the inclusive verifier-pass gate without requiring a secondary metric

**Paper locations:** `proposal.tex:124`, `:245`; `neurIPS_short_paper.tex:150`, `:251`; `proposal-old.tex:253`; `Figure_2.tex:264-267`; all three algorithm blocks.

Current scoring rejects `delta < tau_improvement`, not `delta <= tau_improvement`. At zero thresholds, a tie in verifier passes qualifies, including a possible 0/10 versus 0/10 tie. The general contract is the environment verifier's pass/fail outcome; OOLONG-Pairs defines a pass as exact pair-set equality. There is no secondary-metric gate, per-task non-regression constraint, or significance test. The papers' `max(delta_in, delta_ho) > tau_imp` is wrong on both the splits and the inequality. This finding concerns inaccurate documentation of the intended rule, not a defect caused by the absence of an F1 constraint.

Evidence: `optimization/promotion.py:293-345`; config:48-62. Let `Delta = verifier_passes(candidate) - verifier_passes(incumbent)` on the same `n_ho * v` attempts. The implemented accuracy conditions are `Delta >= -tau_regression` and `Delta >= tau_improvement`; the configured zero thresholds reduce these to `Delta >= 0`. The cost/subcall bands are inclusive baseline-relative mean bounds. The current subcall band is unbounded; a cost ceiling of 3 permits substantial cost increases.

**Illustration of the intended rule, not an implementation error:** September 24 experiment round 2 promoted S2+S3 at 2/10 versus 2/10 exact passes while recorded mean F1 fell **0.7908 to 0.6810**. Its cost ratio was approximately 0.900. This is consistent with the configured pass-count/resource gate. "Non-regression" should therefore name the measured aggregate verifier-pass count rather than imply preservation of every diagnostic metric. The example can remain in this audit without appearing in the new paper.

**Correction:** describe an inclusive gate on aggregate environment-verifier passes in the main text, then identify exact pair match as the OOLONG-Pairs pass definition in its experimental details. Appendix: arithmetic, denominator, resource bands, and optional dense-metric definitions. Preserve the distinction between observed pass counts and statistical evidence. Do not require F1, invent a substitute dense score for environments without one, or imply that every diagnostic metric must be non-decreasing. Historical decisions need their original interpretation only if their results are included.

#### #3. The experiment starts from a hand-guided `H0*R`, not the sparse displayed `H0`

**Paper locations:** `proposal.tex:192`, `:215-226`, `:284-299`, `:358`, `:548-570`; `neurIPS_short_paper.tex:202`, `:225-234`, `:268`, `:343`; `proposal-old.tex:194`, `:216-234`, `:263`; `Figure_1.tex:74-89`; `Figure_A1.tex:22-24`.

The sparse baseline is described as the optimizer's starting point, with most operational guidance recovered from traces. The actual profile starts at `H0*R`. This already contains a revised recursive-call instruction, an orchestrator addendum, and a substantial S2 example: when to use recursive versus bare calls, classifying lines, batches of 300, returning label counts, merging counts in Python, and recovery from parsing failure.

Evidence: config:46; `shrlm/rlm_harness.py:593-710`. The hand-authored H0*R change was motivated by earlier OOLONG observations, as its source comment at :609 records. Additionally, `taxonomy.py:134-145`, `:391-512`, and `proposal.py:587-775` supply human-designed mechanisms, eligible surfaces, runtime capabilities, and procedural guidance. The sentence that discovery is attributable to the loop "rather than to the designer" is too strong.

**Consequence:** a comparison against H0 conflates automated optimization with the starting harness's manual improvements. It also understates the prior structure supplied to the search. "Task agnostic" can describe the current optimizer's reusable interfaces and general reasoning instructions; it does not mean that initialization, environment adapters, or all historical prompts were task independent.

**Correction:** identify the actual initial harness in the main experimental paragraph and qualify the discovery claim. Retain H0 as a reference baseline only. Appendix: exact H0/H0*/H0*R differences, rendered prompt and serialization hashes, the orchestrator addendum, and human-authored taxonomy/proposer priors. Use the evaluation runner's `initial` condition for the exact starting-versus-final comparison (`experiment/evaluation.py:425-442`), alongside any broader baseline grid. The Figure 1/A1 captions must say which reference they show, rather than imply that listing is the run's full initial prompt.

#### #4. Remove child-verifier-related descriptions from the current papers

**Paper locations:** `proposal.tex:152`, `:237-242`, `:261`, `:269`, `:330-342`, `:362`, `:611-623`; `neurIPS_short_paper.tex:184`, `:246`, `:274`, `:354`, `:411-423`; `proposal-old.tex:239-243`, `:347-359`, `:549-565`; `Figure_2.tex:133`, `:257-265`; `Figure_2B.tex:133`.

The papers state that each subcall is checked, both environments have sub-verifiers, and attribution can distinguish correctly computed children from incorrect root aggregation. **OOLONG-Pairs has no sub-verifier configured.** Semantic labels returned by children are not independently established as correct. Deterministic verification applies to the final answer set; it does not establish the causal reason that set is wrong.

Evidence: `experiment/orchestrator.py:271-301`, especially `sub_verifier=None` at :299; `environments/oolong_pairs.py:46-50`, `:1096-1098`; `optimization/grounding.py:44-95`.

Even in environments with a sub-verifier, the current aggregation deserves care: any checked `False` produces `CHILD`; at least one checked `True` with no checked `False` produces `ROOT`, **even if other children are uncheckable**. `ROOT` therefore does not prove that every child was correct. A child error also does not prove it caused the final failure or that the root had no independent defect. No descendants yields structurally grounded `NO_RECURSION`, which must be distinguished from the two causal locations.

**Correction requested by the author: remove this child-verifier account, rather than expand it in the appendix.** Across each active paper and any historical draft maintained as current:

1. Remove claims that every subcall is independently checked, that both environments supply child verifiers, or that root-versus-child attribution proves which component was semantically correct.
2. Remove child-verifier operations from algorithm blocks, correctness coloring/legends from call-tree figures, and child-accuracy conditions from promotion captions. Keep the call tree and operation evidence themselves.
3. Remove child-verifier ablation subsections and any automatic with/without-child-label mining pass. Remove child-verifier-derived root/child correctness metrics from the proposed evaluation; trace-based model diagnoses can still be described explicitly as diagnoses.
4. Replace the method account with: "The environment verifier evaluates the final answer. The attributor uses execution traces to infer failure mechanisms and records uncertainty." Keep final-answer scoring and the distinction between observed operations and inferred causality.

The source details above remain audit evidence explaining the removal. They are not proposed appendix material. Optional integrations present elsewhere in the codebase need not be removed to make this paper edit; if a future reported study actively uses them, its actual protocol would need a separate accurate account.

**Implementation caveat to correct or disclose:** `grounding.py:57-62` can overstate ROOT under partial checkability; `proposal.py:592` still calls every pattern "verifier-grounded." These should not be quoted as evidence that the stronger paper claim is true. Changing the rule would be a separately versioned implementation change, not an undocumented reinterpretation of old runs.

#### #5. The environment roles, split sizes, and benchmark description disagree with the actual profile

**Paper locations:** `proposal.tex:258-277`, `:304-306`, `:384-390`; `neurIPS_short_paper.tex:260-264`, `:274`; `proposal-old.tex:324-337`, `:366-377`.

The short paper still says GraphWalks is the optimization source and OOLONG-Pairs is an unseen transfer target. The active pairs profile optimizes directly on OOLONG-Pairs and materializes only its short/long splits. It does not materialize an additional transfer environment. The expanded draft now names OOLONG-Pairs as source but still gives incompatible 24/40 split sizes, a 40/150 test plan, and elsewhere 10/10 optimization and 20/40 tests.

Evidence: `experiment/splits.py:137-205`; config:22-46, :106-120. Current sizes are **20 mining / 10 validation / 10 short test / 40 long test**. The pinned pool has only two windows per chosen length and 20 task definitions, hence 40 available instances at each length. A 150-instance long test cannot be obtained from that configuration. Source and target are profile-specific; GraphWalks and OOLONG-synth configurations must not be silently substituted into this study.

OOLONG-Pairs here is a reconstruction from OOLONG-synth data with task predicates and a deterministic user-pair scorer. The outputs are pairs of **user IDs**, not necessarily pairs of individual records. The implementation documents extraction/empty-answer behavior changes; "unmodified verifier" is inaccurate (`environments/oolong_pairs.py:1-50`). The 8,192/262,144 lengths are upstream window labels; stripping labels and adding the task changes the actual prompt length. Report that definition rather than implying a measured 32-fold tokenizer-identical input increase.

**Correction:** one short main-text table must name actual source, actual test conditions, and actual counts. Appendix: dataset revision, task IDs, reconstruction, canonical pair extraction, malformed versus empty answers, thresholds, partition seed, and measured prompt-length summaries. A new transfer environment can remain explicitly planned; no transfer result is established by the current pairs-only run.

#### #6. Validation is adaptively reused and its summaries influence later proposals

**Paper locations:** `proposal.tex:242`, `:275-277`, `:304`, `:489-492`; `neurIPS_short_paper.tex:248`, `:262`, `:274`; `proposal-old.tex:248`, `:373-377`, `:491-493`.

"Held-out" correctly means not part of mining execution, but it does not mean independent of optimization decisions. The same validation instances are reused across rounds. Promotion depends on them, and later proposer history includes aggregate exact scores, comparable quality changes, costs, and observed activation summaries. The text's "unbiased" wording is inappropriate for repeatedly selected validation scores.

Evidence: `optimization/validation.py:104-136`; `optimization/proposal_evidence.py:907-1050`; `optimization/history.py:122-299`; `proposal.py:901-918`. History does **not** forward held-out trace bodies or raw gold/answer dumps. It does expose aggregate diagnostics and sanitized failure summaries, so "never mined" must not become "never affects proposal generation."

**Correction:** main text: "Validation tasks are excluded from weakness mining but reused for adaptive selection; aggregate outcomes inform proposal history." Appendix: precise allowed feedback, comparable-metric checks, per-round reuse, and test-access policy. Reserve claims about final generalization for a separate frozen-harness evaluation, with exposure qualifications in #7 and #9.

#### #7. Retain the split and describe task-level separation accurately

**Paper locations:** `proposal.tex:275-277`, `:304-306`, `:321-325`; `neurIPS_short_paper.tex:262`, `:274`; `proposal-old.tex:345-346`, `:373-377`.

**Agreed disposition: keep the current split.** This is a paper description and interpretation requirement. Its priority concerns inaccurate claims about what is held out, not a defect in the split or a requirement to redesign the experiment.

An OOLONG-Pairs task instance combines **a context window of input records and one task question/predicate over those records**. It is not a fresh record collection for each question. Here there are two short windows, each with 188 records. The loader combines each window with 20 task questions, producing 40 task instances. It shuffles these `(window, task)` combinations and partitions them 20/10/10; it does not assign whole windows to separate roles.

The saved September 24 split files have distinct instance IDs, but **all three short roles reuse exactly the same two input corpora**. The audit stripped the exact task question from each saved prompt and hashed the remaining input prefix:

| Short corpus hash prefix | Mining tasks | Validation tasks | Short test tasks |
|---|---:|---:|---:|
| `5f067945b676` (window 9) | 10 | 6 | 4 |
| `e939371361d4` (window 10) | 10 | 4 | 6 |

For example, three actual instances use the same 188 records from window 9:

- Mining task 1 asks for user pairs where both users have at least one numeric-value or location instance.
- Validation task 9 asks for user pairs satisfying entity/location conditions plus a date restriction on location instances.
- Short-test task 12 asks for asymmetric user roles: one user has at least two numeric-value instances, while the other has location and human-being instances.

The requested answers and predicates differ. The underlying question texts to classify, user IDs, and dates are identical across these three inputs. The split therefore withholds the particular `(records, question)` combination, while the optimizer has already observed those records under other questions during mining. Task-question templates can also recur on the other window; these roles do not guarantee entirely unseen predicate families either.

Evidence: `evidence.json/latest_split_corpora`; `experiment/splits.py:268-297`; `environments/oolong_pairs.py` loader shuffles window-task instances before partitioning. Long corpora have different hashes; reusing window IDs alone would not prove identical short and long contents, and this audit makes no such claim.

**Interpretation:** short-test results measure performance on held-out task instances whose short-context records are shared with mining and validation. Validation uses its own disjoint task instances for adaptive selection, as described in #6. This is a defensible evaluation scope; the overlap does not establish answer or label memorization. Report task and context counts separately, and account for the shared contexts when interpreting uncertainty rather than treating 40 tasks as 40 independent record collections.

The separate long evaluation assesses performance on different, larger record collections. The two long windows each contain 6,374 records and together supply 40 task instances. They share underlying question texts with the short windows, despite having no identical complete `(user_id, date, question_text)` records. Accordingly, describe the long evaluation as evaluation on larger record collections; do not claim entirely unseen underlying question texts or count 40 task instances as 40 independent long contexts. This defines the measurement's scope without requiring a different split.

**Optional alternative, not the current recommendation:** the [split decision and alternatives note](2026-09-28-oolong-pairs-split-recommendation.md) retains the previously considered context-separated allocation and its data constraints. The two short windows have 43 question texts in common despite zero identical complete records between them; assigning different windows to different roles would therefore still not establish entirely novel classification inputs. Keep this distinct from the current cross-role reuse of each complete short window. A stronger split becomes necessary only if unseen-source-data generalization becomes a central claim, and must separate the source unit relevant to that claim. No split changes are requested by this audit.

**Correction:** main text: "Mining, validation, and short test contain disjoint task instances on shared short-context records. We report short-test performance on held-out task instances and separately evaluate performance on larger record collections; underlying question texts may recur across these collections." Appendix: retain the 20/10/10 allocation, task-versus-context counts, overlap table and hashes, and distinguish complete-record overlap from repeated question text. Keep the current split; do not make a redesign a prerequisite for the paper or forthcoming experiment.

#### #8. The current method and historical runs must not be presented as one preregistered experiment

**Paper locations:** `proposal.tex:245`, `:251`, `:267`, `:321-325`, `:474-492`; `neurIPS_short_paper.tex:251`, `:276`; `proposal-old.tex:253`, `:257`, `:345-346`, `:482-493`.

There have been substantial changes to evidence selection, prompt contracts, proposal admission/repair, history, split sizes, cost bands, and promotion thresholds. These were often motivated by inspection of preceding experiments. Freezing a config before a particular run is valuable, but is not evidence that the full evolving research protocol was preregistered before development outcomes were observed.

The latest three-round experiment's launch commit was **`ad2ac00147`** on September 24. Its recorded results do not test the September 28 changes in **`323dbceb`**, which added better input-definition/correction context and revised prompts. Those later changes have offline replay and test evidence, documented in the September 28 verification report; they have not acquired live three-round results merely because the papers were edited afterward.

**Correction:** choose and name the implementation revision for each reported study. Separate exploratory development from any future confirmatory protocol. Appendix: launch commit, source snapshot/hash, config and split hashes, harness hashes, prompt/digest/taxonomy versions, and whether a run was interrupted or resumed with changes. Use "frozen before this run" where that is the evidence; retain "preregistered" only with an actual dated preregistration and a record of deviations.

**Current paper scope:** describe the upcoming experiment's implementation/configuration; an account of previous launches and their version differences is not required. The historical discussion in this finding applies only if those earlier outcomes are reported or used as evidence in the paper.

The often-cited diagnosis response acceptance change **39.7% to 81.4%** measures schema/admission outcomes, not promotion quality or verified causal accuracy. The prior assessment compares the first three rounds of two different launches, with **71/179** versus **79/97** diagnosis responses admitted. Promotion changed from strict to inclusive at the same time as other optimizer changes. Strict-gate-eligible promoted batches were one in each three-round sequence. This is an uncontrolled development comparison and cannot identify the effect of one prompt change.

#### #9. Generalization and long-evaluation claims outrun the completed measurements

**Paper locations:** `proposal.tex:124-126`, `:194-198`, `:304-325`, `:489-492`; `neurIPS_short_paper.tex:150-160`, `:204-208`, `:274-285`; `proposal-old.tex:343-359`, `:491-493`.

Both active drafts retain "Evidence of" generalization/transfer contribution language while saying results are forthcoming. Conditional hypotheses are appropriate in a proposal; achieved-result wording is not supported by the saved experiments reviewed here.

The September 16 experiment's long evaluation was stopped:

- Starting harness: **48 completed attempts on 16 tasks**, three attempts each; 0 exact passes. Forty-five were resource-terminated and three were scored incorrect; recorded all-attempt mean F1 was approximately **0.0058**.
- Edited harness: **3 completed attempts on one task** from that matched subset; two resource-terminated and one malformed. The matching three initial-harness attempts were all resource-terminated. Both matched arms had **0/3 exact passes and mean F1 0**.
- In-flight canceled work is not a completed attempt and is not silently added as a zero. Unrecorded usage can make spend a lower bound.

Evidence: `experiment_oolong_pairs_dsv4f_20260916_131837/eval/initial/oolong_pairs_long/round_00/runs.jsonl`; `eval/paired_20260923T140326Z/sh_rlm/oolong_pairs_long/round_00/runs.jsonl`; that paired directory's `comparison.md`; September 23 report:139-155.

**Correction:** do not claim a completed baseline-versus-edited long benchmark, a demonstrated long-context gain, or cross-environment transfer from this evidence. Disclose the stopped pilot, resource censoring, and matched comparison size if used. Later experiments should not call these same previously inspected long instances "never opened" without a precise time/study qualifier. The final short/long claims must name their actual condition, completed denominator, and harness hash.

#### #10. "Repeated seeded runs" and the proposed uncertainty account are not implemented as written

**Paper locations:** `proposal.tex:321-325`; `neurIPS_short_paper.tex:274`; `proposal-old.tex:345-346`; algorithm/repetition descriptions in all drafts.

Dataset partitioning has a seed. Individual model attempts do not have per-attempt seeds wired through evaluation. The configured decoder is temperature zero, so these should not be described as controlled seeded stochastic replicates. Repeated requests may still vary, but this is not a common-random-seed paired design.

Evidence: `experiment/evaluation.py:46-60`; `experiment/config.py:39`; config:13-20, :244. Current validation uses **one** attempt; final evaluation is configured for **three**. Searches of the experiment code did not find an implemented bootstrap or McNemar-style analysis corresponding to the papers' promises.

**Correction:** state the two repetition counts separately. Describe the seed as controlling sampling/partitioning only. A bootstrap or paired significance test may remain explicitly planned until actually run and reported; specify the paired task and dependence unit, the treatment of repeated attempts, missing runs, and the two-window limitation. Three attempts do not triple the number of independently sampled tasks or corpora. No significance claim should be inferred from `v=1` promotion decisions.

#### #11. Cost formulas and budget tables describe the obsolete optimizer

**Paper locations:** `proposal.tex:350`, `:384-418`; `proposal-old.tex:382`, `:443-471`; validation diagrams and pseudocode in all drafts. The short paper needs an accurate appendix formula if it includes optimization cost claims.

The old expression `n_in*m + (n_in+n_ho)*(K+1)*v`, plus additional expected merge checks, no longer describes the implemented loop. The current number of scheduled task attempts in an ordinary completed round with an evaluable batch is:

```text
mining attempts     = m * n_in
validation attempts = 2 * v * n_ho
round total         = m * n_in + 2 * v * n_ho
```

With the current pairs profile that is **40 + 20 = 60**, independent of the number of edits in the batch. A round without an evaluable candidate, including a skipped duplicate rejected batch, has no new validation leg. Crashes, breakers, interruptions, and resumed cached work need actual-manifest accounting.

The September 24 three-round run recorded **180 task attempts: 120 mining and 60 validation**. A hypothetical completed four-condition evaluation on this profile's 10 short + 40 long tasks, three attempts each, would add **600** attempts; initial-versus-final only would add **300**. These are scheduling counts, not measured evaluation results. Default evaluation has four conditions (`evaluation.py:437`), whereas the legacy report knob `eval_conditions=3` is a projection input, not the runner's condition selector. A fifth `initial` condition is available when requested. Explicitly list the selected conditions when computing a budget.

**Correction:** replace the 1,328/19,920 and 5,250/6,330-run projections, or label them as dated projections of an abandoned design. Remove `p_merge=0.5` as a live validation cost factor. Recompute monetary estimates with relevant calibration, not just rescaled counts: initial harness, length, recursion, output cap, failure incidence, and cost band all affect usage. Keep the user-confirmed $0.19/M input and $0.51/M output as dated experiment accounting rates, not a verified current market quote.

Include attribution/proposal/retry calls in total optimization cost. Distinguish per-task costs, full stage usage, circuit-breaker reservations, and lower-bound telemetry; do not add task usage to a stage total that already includes it. A baseline subject exhausting its cumulative budget makes the comparison unscorable; it is not a rejected candidate. Root task attempts can spawn many model calls, so attempts are not interchangeable with request counts. The $2 cap also should not be advertised as exact complete billing in the presence of unreported/canceled requests.

### P2 -- Moderate: implementation and reporting detail that must be corrected or supplied

#### #12. Diagnosis now uses an explicit evidence/uncertainty contract

**Paper locations:** `proposal.tex:237-242` (including the obsolete commented paragraph at :238); `neurIPS_short_paper.tex:246`; `proposal-old.tex:239-243`; all algorithm `sub_verify -> cluster` shorthand.

The current account omits the model attribution step and its admission rules. After final-answer verification, the attributor receives a bounded digest with cited operations, nearby inputs/consumers/corrections, and uncertainty guidance. It must distinguish observed operations from a causal explanation. Missing output elements do not establish skipped input; absent evidence is not proof that an action was absent; recovered errors must not be described as unresolved without support.

For `incomplete_coverage`, `coverage_basis` records `status`, `input_scope`, `loss_observation`, and `counterevidence`. Status is `observed_loss`, `not_established`, or `contradicted`. Non-established/contradicted coverage claims normalize to `OTHER`/`UNATTRIBUTED`; they may therefore count as accepted responses while supplying no actionable coverage diagnosis. Host validation checks vocabulary, shape, bounded fields, and resolvable references, not the truth of the causal inference.

Evidence: `optimization/attribution.py:45-54`, `:110-165`, `:541-690`, particularly :578-580. Default attribution response attempts are bounded at three; operational transport retries are a distinct layer. Rejected/unattributed outcomes remain visible in saved records. Passing runs are not sent through failure attribution (`mining.py:158`). Some environment-owned failures are excluded from diagnosis, though retained as outcomes.

**Correction:** main text needs only "structured, evidence-referenced diagnoses with explicit uncertainty." Appendix: schema, normalization, rejection/retry handling, observation versus inference, and excluded outcome types. Remove child-verifier steps rather than carrying the obsolete `sub_verify` shorthand into the new algorithm. Never equate higher response acceptance with more correct causal diagnoses. Do not include an OOLONG-specific worked explanation as a definition of the general attribution method.

#### #13. The mechanism vocabulary, ranking, and eligible routes are human-defined and many-to-many

**Paper locations:** `proposal.tex:215-224`, `:239-242`; `neurIPS_short_paper.tex:225-234`, `:246-248`; `proposal-old.tex:216-220`, `:239-241`; surface discussion/captions in all drafts.

The "one-to-one" mechanism-to-surface description is obsolete. There is a primary route plus additional eligible routes and conditional support requirements. For example, lossy aggregation is routed to S3/S4 and, with suitable operation evidence, S8; it is not routed to S9 as a semantic oracle. Supported S8 routes require relevant complete operation references; supported S5 routes for execution/budget failures require an error and a subsequent complete operation. That opportunity test does not prove that recovery succeeded or that the route will improve the task.

Evidence: `taxonomy.py:134-145`, `:391-512`; `proposal_evidence.py:429-445`. `UNATTRIBUTED` yields no eligible surfaces. `OTHER` is not itself a verified causal mechanism and should not be conflated with a freely actionable unattributed summary.

Clustering still uses the four-part signature. Ranking first uses **distinct-instance support**, then actionability, then a deterministic signature tie-break. Attempt support is separately stored. Minimum support two is a flag, not automatic deletion. The three representatives and actionability weights are fixed heuristics: causal status 0.4, grounded fraction 0.3, known-mechanism indicator 0.2, normalized free-text homogeneity 0.1 (`clustering.py:34-113`, `:231-269`, `:339-352`). "Homogeneity" here is based on distinct normalized description strings, not a learned semantic clustering score. Actionability is a reading-order heuristic, not a calibrated probability of success.

**Correction:** a small appendix table should list the versioned mechanism/routing contract and explain the ranking. Main text should acknowledge predefined mechanisms and capabilities. Describe support-based S5/S8 eligibility as structural admission, with semantic relevance still judged by the model. Preserve the distinction between attempt and unique-instance counts in reported mining statistics.

#### #14. The proposer sees selected bounded evidence, not the entire saved trace bundle

**Paper locations:** `proposal.tex:237-242`; `neurIPS_short_paper.tex:246-248`; `proposal-old.tex:239-248`; `Figure_2.tex:126-169`, `:257-263`; corresponding regions of `Figure_2B.tex`.

Current selection builds a compact inventory, expands a limited number of actionable mechanisms, and retains complete relevant code operations where they fit. It prefers distinct mechanisms, then additional signatures and context; it can add one passing contrast sharing operation names. Relevant child calls, parsing/combination/filtering operations, preceding input definitions, and nearby downstream corrections replace the old first/last-root-block heuristic.

Evidence: `optimization/digest.py:41-45`, `:137-186`; `proposal_evidence.py:174-250`, `:449-689`. Defaults are **12,000 characters for the attribution digest**, **32,000 for proposal evidence**, and **12,000 for compact history**. Expanded evidence is capped by `min(k, 4)`; inventory-only patterns cannot authorize a proposal. The budget can still leave only one actionable mechanism. Required whole operations that do not fit can be omitted; outputs/prose may be shortened. This does not guarantee complete causal context or balanced coverage of every surface.

The 32,000-character limit is **not a total proposer-prompt cap**. System instructions, current surface text, environment contracts, and history are additional. The latest assessed live prompts were about 52.7K, 61.5K, and 69.7K characters; older 158-174K prompts belong to an earlier selector. The September 28 replay even traded an additional pattern for more complete context, so "more mechanisms shown" is not a general claim supported by that change.

**Correction:** appendix: inventory/admitted distinction, expansion order, whole-operation policy, truncation/omission markers, contrast selection, budgets, and versions. Main text can say "compact operation-level evidence." The presence of a successful/recovered contrast is evidence for comparison, not proof that the model's explanation is causal. Do not call ordinary assistant prose in the digest a captured private reasoning trace.

#### #15. Proposal selection, surface ownership, abstention, and repair materially constrain search

**Paper locations:** `proposal.tex:231-245`; `neurIPS_short_paper.tex:241-251`; `proposal-old.tex:234`, `:248-253`; `Figure_1.tex:88-89`; `Figure_2.tex:166`, `:245-249`; corresponding `Figure_2B.tex` locations.

A proposal is not an arbitrary independent edit drawn from every clustered pattern. The prompt exposes admitted pattern/surface/evidence choices, asks for selections first and replacements second **within one response**, and enforces at most one admitted edit per pattern and surface. The first candidate that passes admission owns its surface; a malformed/rejected candidate does not permanently consume it. The maximum is a ceiling, not a quota: one reasonable surface should yield one edit; no reasonable change should yield none.

The three explanation fields compare the actual incumbent, unresolved observed behavior, and the operation/input/trigger that would change. They challenge redundant instruction emphasis, moving the same instruction to another surface, and fixes for an already-recovered error. Minimality can require replacing a misleading worked example. These are instructions plus structural checks, not a deterministic proof of semantic novelty.

Response/schema retries are bounded by `DEFAULT_MAX_ATTEMPTS=8`; this is not eight full experiment validations. Once repair of rejected members is invoked, there is **one repair response**, limited to failed original patterns. It may choose another eligible unoccupied surface, must leave retained siblings unchanged, and may withdraw a failed member. It cannot introduce new patterns. All retained and repaired admissions remain subject to unique ownership.

Evidence: `optimization/proposal.py:136-186`, `:602-686`, `:780-821`, `:1584-1704`, `:1873-1913`, `:2047-2173`. No available choices can terminate proposal generation without a model call; completed checkpoint reuse likewise need not spend again.

**Correction:** main text: one edit per surface, up to a maximum, with abstention permitted. Appendix: selection/replacement order, evidence-reference contract, admission ownership, three behavioral fields, and repair scope. Do not describe separate selection and generation model stages unless such calls were actually made. Do not interpret multiple attempted submissions as independently validated candidates.

#### #16. Literal text, materialization, and preflight are missing from the reproducibility account

**Paper locations:** `proposal.tex:242-245`, surface appendix :506-537; `neurIPS_short_paper.tex:248-251`, :303-339; `proposal-old.tex:248-253`, :298-314; Figure 1 builder explanation.

Text proposals now use `literal-text/v1`: the model writes literal instructions and the host escapes formatting braces exactly once, restoring the designated live custom-tools marker. Earlier model-generated nested escaping caused repeated proposal failures. A paper about proposal success cannot omit this admission difference when comparing runs.

S1-S5 edits are full replacement texts. S6 is a policy dictionary. S7/S9 are functions with required interfaces; S7 retains the incumbent's declared output bound. S8 adds/replaces one named function in one selected root/child helper dictionary. S10 edits one named entry, including the supported removal form; the host merges it with the rest of the library. The harness interface can represent a larger space than this proposal contract exposes.

Evidence: `optimization/proposal.py:719-733`, `:1240-1345`; `shrlm/optimization/skill_edit.py:73-80`, `:168-247` skill merge/removal contract; `optimization/candidates.py:817-922`; `runner.py:362-665`.

The host verifies serialization/hash consistency, a genuine single-surface change, caps, reserved interfaces, prompt formatting, and selected smoke/answer fixtures. The OOLONG-specific S9 fixtures are an environment-admission detail (`candidates.py:718-721`). They are not hidden validation tasks and do not prove semantic correctness on arbitrary tasks. No-op serialization checks catch literal unchanged harnesses, not every instruction that would behave equivalently.

**Correction:** appendix: versioned proposal formats, materialization rules, actual editable payloads, and deterministic rejection checks. Do not present "passes preflight" as proof that an edit changes behavior, preserves every semantic invariant, or improves results. When publishing exact edits, distinguish literal model text, encoded template/source, and rendered runtime instruction.

#### #17. History now includes qualified dense-metric progress and substantive revision links

**Paper locations:** `proposal.tex:242`, `:307-309`; `neurIPS_short_paper.tex:248`, `:274`; `proposal-old.tex:248`, `:347-359`.

A binary list of rejected proposals is no longer an accurate history description. Archived history retains attempted edits, reasons, batch membership, measured outcomes, and revision relationships. The prompt receives a bounded summary: recent rounds, the current incumbent's promotion, relevant measured predecessors, and a relevant older promising direction when it fits. Referenced history that cannot be represented adequately may remove a choice from the current prompt. The full archive is not replayed verbatim.

"Potentially promising" is assigned to a rejected **measured** subject only when a verifier-defined dense metric is comparable and improves in its declared direction. It does not mean accepted, statistically reliable, or proven effective. It is not hard-coded to F1: definitions include identity/version, direction, aggregation, precision, and terminal-outcome values. Incomparable definitions, incomplete coverage, or unsupported metrics are `not_assessed`. Comparison checks include verifier/protocol/repetition compatibility and matching instance-attempt sets. A prior identical rejected batch under the same incumbent can be skipped before fresh validation.

Evidence: `optimization/history.py:11-299`; `proposal_evidence.py:48-120`, `:907-1050`; `validation.py:975-993`.

Preserve these real development examples with their qualifications:

| September 16 run | Measured subject | Exact passes | Recorded mean F1 | Interpretation |
|---|---|---|---|---|
| Round 6 | S6 singleton | 1/10 to 0/10 | 0.6194 to 0.7193 | Rejected; a qualified dense-metric signal, not an exact-match improvement |
| Round 8 | Combined candidate | 3/10 to 1/10 | 0.6128 to 0.6951 | Rejected; batch-level signal, not independent constituent credit |

**Correction:** main text can mention outcome-informed history; appendix should specify bounded selection, predecessor/revision rules, duplicate rejection avoidance, metric comparability, and examples of mixed outcomes. Do not call a direction exhausted merely because the exact gate rejected it, or call it successful because its F1 rose once.

F1 aggregation also needs precision: current OOLONG-Pairs records per-attempt F1 rounded to three decimals and averages over attempts, including verifier-declared zero outcomes. Exact pass is separate set equality, not `rounded_F1 == 1`. Missing/extra pair counts apply where the scorer could extract them; report coverage rather than silently treating absent counts as zero. This is macro averaging across attempts, not a pooled micro-F1 over all pairs.

#### #18. Saved behavior records activation, not verified compliance with the proposed behavior

**Paper locations:** `proposal.tex:242`, `:327-342`; `neurIPS_short_paper.tex:248`, `:274`; `proposal-old.tex:248`, `:347-359`; future surface-success analysis/captions.

Behavior observation currently summarizes existing instrumentation for S6 syntax retries, S9 answer redirects, and S10 skill loads. S6/S9 observation is root scoped; skill-load observation can cover the traced call tree. Missing observation coverage produces an unassessed result. S1-S5, S7, and S8 do not receive a general semantic-compliance detector.

Most importantly, `intended_behavior` is explicitly **`not_assessed`**. Loading a skill does not show that its procedure was followed; invoking a helper does not show that its inputs or task predicate were correct. The distinction is material to explanations of why a promoted edit worked.

Evidence: `optimization/behavior.py:24-110`; `optimization/history.py:122-224`; current validation summaries carry `behavior-observations/v1`. Older root-only skill counters and newer full-tree behavior counts have different scopes and must not be mixed without labels.

**Correction:** appendix: detector, observation scope, denominator, availability, and the difference between activation and intended behavior. Qualitative trace inspections may be presented separately as bounded examples with explicit evidence. Avoid statements that the system automatically confirms every proposal's intended behavioral change.

#### #19. Failure containment, incomplete subjects, and resume semantics affect the reported denominator

**Paper locations:** `proposal.tex:237`, `:245`, `:321-335`, `:410-418`; `neurIPS_short_paper.tex:246`, `:251`, `:274`; `proposal-old.tex:239`, `:253`, `:345-359`, `:443-471`.

Candidate-owned execution errors can now be recorded as failed attempts instead of crashing the entire experiment. Resource termination and content filtering are separately visible. This does not mean every exception is suppressed: configuration, operational/provider failures, cancellation, memory/I/O failures, and observation-persistence failures can still stop work. This distinction is needed to interpret the stopped experiments and the later failure-containment fix.

Evidence: `optimization/driver.py:969-1011` and surrounding exception handlers; `mining.py:67-90`; `validation.py:257-319`, `:403-470`, `:1030`; `experiment/evaluation.py:38-56`; `docs/experiment-reasoning.md`.

Completed failed attempts enter exact-pass denominators. Terminal quality values are verifier-defined (#17). A budget-stopped candidate with missing planned attempts is not silently scored as a completed all-failure candidate; a partial baseline cannot anchor a promotion comparison. Evaluation can report an over-budget/incomplete condition while retaining sibling results. Missing/canceled work and unwalkable traces must be counted separately from measured failures. Cached/resumed completed work must not count as fresh independent attempts or new spend.

**Correction:** one appendix paragraph and a small terminal-outcome table should define these cases, failure categories, denominator treatment, partial telemetry, and checkpoint identity checks. Any results table needs expected/completed/missing counts, not just pass percentage. Replace blanket "the experiment cannot crash" or "all calls have complete traces/costs" language if introduced during revision.

#### #20. The surface table overstates several capabilities and execution scopes

**Paper locations:** `proposal.tex:204-224`, `:506-537`; `neurIPS_short_paper.tex:214-234`, `:303-339`; `proposal-old.tex:208-220`, `:298-314`; `Figure_1.tex:74-85`; `Figure_A1.tex:22-24`.

Use actual runtime scope rather than aspirational examples:

| Surface/contract | Current paper problem | Correction and source |
|---|---|---|
| S1-S5 | S2 is labeled "Turns 1-2"; sequential surface descriptions can imply separate timed hooks | All five strings enter the shared system prompt. First-two-turn use is an instruction, not a host-enforced schedule. `rlm_harness.py:544-568` |
| S1 | "Only stdout" and "final answer exclusively through REPL" are too absolute | Ordinary interaction includes prompts and accumulated history. `rlm/core/rlm.py:614` formats iterations; :667-669 invokes a final-answer fallback after iteration exhaustion. Describe the ordinary REPL path and disclose exceptions |
| Root/child scope | Every child is implied to inherit the same REPL/harness | Actual recursive `rlm_query` children have their own REPL/harness context; bare `llm_query` and recursion-limit leaves do not. `taxonomy.py:134-145`; `rlm_harness.py:623-630` |
| S6 | "Max calls per turn/total," broad retry-on-error language | Implemented keys include enabled, prompt/batch caps, depth, syntax-error retry/count, and output validation. There is no general calls-per-turn/total policy field. Width/length caps refuse work rather than split it. Syntax retry repeats the same prompt, not arbitrary recovery. Root-local enforcement except inherited depth. `rlm_harness.py:212-229`; `taxonomy.py:140` |
| S7 | "Harness memory of prior calls" suggests full state/trajectory access | Receives bounded execution stdout and redacted variable type/length metadata; not hidden variable contents or full prior-call memory. Root formatter. `rlm_harness.py:232-260`; `taxonomy.py:141` |
| S8 | One builder per surface; helpers/data broadly editable | S8 covers two builder/dictionary fields. The current proposer edits one named function in one dictionary per candidate. Helpers affect behavior only when called; no automatic invocation. `rlm_harness.py:263`, `:275`; `proposal.py:1308-1312` |
| S9 | Semantic checker or rejection-only hook | Root hook sees answer text and redacted inventory. It may accept transformed text or redirect with a nudge; it cannot inspect hidden records/labels to establish semantic correctness. `taxonomy.py:143`; `runner.py:595` |
| S10 | Loader implies body is automatically read/executed | Shared index advertises skills; body matters only when loaded or forwarded. Text guidance is not executable S8 code. `taxonomy.py:144`; `runner.py:676-727` |
| "Single builder" | All ten surfaces claimed to map to one builder each | Replace with "a declared field or group of fields"; S8 demonstrably has two builders in Figure 1 itself |
| Number/sparsity | "Seven of ten" and old "seven of nine" counts | Name the actual empty/disabled components for the chosen initialization or omit the count. H0's sparsity does not describe H0*R |

**Correction:** keep the main surface taxonomy conceptual and short. Put a capability table like this in the appendix, including inputs visible to the hook, scope, invocation condition, and editable payload. Runtime constraints are authoritative; a model's proposed instruction cannot extend a terminated run, add host retries, access redacted state, or create a semantic oracle.

Provider/model selection and decoding/reasoning settings are experiment-owned, not editable surfaces. The optional audit examples at `proposal.tex:641-642` and the corresponding short/old appendices should not imply that a proposal can change provider reasoning mode through S1-S10. If the intended example is a textual instruction to reason differently, say so.

The three immutable invariants should be phrased as concrete scaffold constraints and tested pathways, not a claim that arbitrary generated code is formally proved safe or unable to print task data. The input initially lives in `context`; that does not forbid subsequent code from printing it into model-visible output.

#### #21. S10's token cost is not confined to the turn that loads a skill

**Paper locations:** `Figure_1.tex:82-85`; S10 rows at `proposal.tex:534-537`, `neurIPS_short_paper.tex:331-339`; any duplicated caption text.

The caption states that a procedure costs tokens only on the turn that loads it. If the loaded body is printed into execution output, it enters model-visible conversation history and can be present in later requests until truncation/compaction removes it. Forwarding it to a child also incurs input cost there. Calling the loader and merely storing its return value need not expose all its text to the model immediately. The advertised index has its own prompt cost.

Evidence: `runner.py:676-727`; `rlm/core/rlm.py:614` and accumulated `message_history`; the skill index is assembled in `rlm_harness.py`. S10 supports adding, replacing, and removing an entry; its existing removal description is **not** an error and should be retained. Current limits are eight entries, 40-character names, 200-character descriptions, 4,000-character bodies, and 16,000 total characters (`rlm_harness.py:342-346`).

**Correction:** caption: "Bodies are loaded on demand; their token cost depends on whether they are exposed in the conversation or forwarded to children." Appendix: index/body distinction, library merge/removal, caps, and persistent-history cost. Do not claim a measured token saving without usage evidence.

#### #22. Reasoning capture must distinguish provider-returned data from hidden reasoning and from optimizer input

**Paper locations:** `proposal.tex:237-242`, `:321-335`, reproducibility/timeline text :474-492; `neurIPS_short_paper.tex:246-248`, `:274`; `proposal-old.tex:239-243`.

The experiment now saves provider-returned reasoning/response observations across mining, validation, final evaluation, attribution, proposal/repair, recursive calls, compaction, and fallback paths where instrumented. These are separate hash-referenced artifacts. This is a substantial addition to saved outputs, but it does not mean full private chain of thought is always available.

Availability statuses include `returned`, `opaque_only`, `not_returned`, `unsupported_client`, `legacy_unavailable`, and `no_response`. A reasoning-enabled request can still yield no returned reasoning. Historical artifacts cannot be backfilled with reasoning that the provider never supplied. Reasoning token counts, when unavailable, remain unknown; they are not automatically zero or additional billable tokens beyond reported usage.

Evidence: [reasoning storage documentation](../experiment-reasoning.md), `rlm/core/llm_observation.py`, and `shrlm/optimization/llm_observation_store.py`, with their client/driver integrations. The September 25 assessment found the inspected live provider responses had `not_returned` reasoning. The saved observation bodies are not automatically fed into mining digests, proposer evidence, or history. Model-written interpretation paragraphs extracted from ordinary messages are not the same artifact.

**Correction:** appendix: what is saved, where, with which availability labels and identity hashes, and what actually enters each optimizer prompt. Main text need not discuss every storage format. Use "provider-returned reasoning, when available," not "complete CoT for every call." Separate observations from speculative reconstruction, and disclose legacy/missing coverage in any reasoning analysis.

#### #23. Pseudocode skips attribution and resets patience at the wrong event

**Paper locations:** `proposal.tex:358-377` (particularly :362-373); `neurIPS_short_paper.tex:343-371` (particularly :354-365); `proposal-old.tex:390-409` (particularly :399-405).

Beyond obsolete dual-split/per-edit evaluation (#1), the algorithm jumps from sub-verification to clustering without the structured model-attribution stage. It uses H0 irrespective of configured initialization and resets patience when individual acceptances exist, even if their merged candidate would fail. The implemented loop resets patience only on actual incumbent promotion. Empty proposals and rejected/unscorable progress must not be represented as improvements.

Evidence: `experiment/orchestrator.py:832-874`; `mining.py`; `validation.py:971-1077`. The last incumbent is frozen; the loop does not pick the best F1 trajectory point or retrospectively choose a test winner. With inclusive ties, promotion can reset patience without an exact-pass gain, so patience is no longer "rounds without accuracy improvement."

**Correction:** move one compact accurate algorithm to each paper's appendix and use the replacement skeleton later in this report. Main text needs only the three stages and stopping criterion. Describe a freeze after the stopping rule, followed by a separate evaluation, rather than adding test measurements to the selection loop.

#### #24. Analysis figures need explicit units, fresh measurements, and coverage

**Paper locations:** `proposal.tex:307-342`; `neurIPS_short_paper.tex:274`; `proposal-old.tex:347-359`; future results plots derived from analysis scripts.

Three existing script corrections change how results should be described:

- **Surface activity:** a surface receives promotion credit when its constituent belongs to the promoted batch, but the batch itself is not an eleventh surface. Attempted/submitted/admitted/evaluated/promoted counts are different populations. An edit counted as promoted still lacks independent measured efficacy. Evidence: `experiment/surface_activity.py:71-75` and aggregation logic.
- **Quality trajectory:** each round uses a fresh measured candidate score on promotion or fresh incumbent score on rejection. Old promotion-time scores are not carried forward as new observations. No new measurement should be plotted as a measured zero or silently replaced with an old value. Evidence: `experiment/incumbent_quality.py:3-10`, `:212-249`. The same harness can therefore have different measured accuracy across rounds; a non-monotone curve need not imply an unrecorded code change.
- **Environment labels and costs:** the report environment must come from the selected config/run identity, not the old GraphWalks label. Projection conditions and executed evaluation conditions must be distinguished (#11).

The "whole-input collapse rate" also needs its operational definition: a walkable run is flagged when the largest subcall/context-size ratio exceeds **0.8** (`optimization/clustering.py:43`; `experiment/collapse_and_attribution.py:363-397`). This is not automatically "no meaningful decomposition," nor the union of all forms of weak recursion. The ratio uses the largest descendant prompt character count divided by the root prompt character count (`optimization/walker.py:322-342`), not a token count or a semantic measure of task coverage. The denominator is walkable runs, passing and failing; unwalkable runs are reported separately. Failure-location shares use failing walkable runs, include no-recursion/ungrounded categories, and depend on sub-verifier coverage.

**Implementation caveat:** the standalone collapse/attribution analysis registry currently contains only GraphWalks, whereas the orchestrator also supports an OOLONG-synth sub-verifier (`collapse_and_attribution.py:140`; `orchestrator.py:271-301`). A future OOLONG-synth analysis cannot assume those two paths provide identical grounding. No such sub-verifier exists for OOLONG-Pairs in either path.

Under the author's revision scope, retain these details as internal audit evidence only. Remove child-verifier-derived accuracy/location-share metrics from the current papers. Runtime depth, subcall count, collapse behavior, and explicitly model-inferred diagnoses can still be described with their actual definitions.

**Correction:** appendix/captions: define each numerator, denominator, completeness status, and evidence scope. Keep efficiency behavior separate from causal success. Use the updated analysis scripts, pin their version, and do not splice incomparable older/newer outputs into one curve without disclosure.

#### #25. The proposed ablations require more precise counterfactuals

**Paper locations:** `proposal.tex:338-342`, `:611-629`; `neurIPS_short_paper.tex:411-429`; `proposal-old.tex:549-570`.

Withholding a sub-verifier cannot constitute a distinct OOLONG-Pairs condition when that environment already has no sub-verifier. Re-attributing identical saved traces can assess diagnostic changes without new harness executions, but it still involves model attribution calls and cannot establish the downstream effect on generated proposals/promotions unless those stages are rerun. The loop does not automatically mine both with and without child labels as the short/old drafts suggest.

"Leave one accepted edit out of the final harness" is ambiguous when successive edits replace the same surface. An overwritten edit is not independently present in the final text; subtracting it is not well-defined. Jointly promoted S2/S4 or S8/S10 changes can interact. A leave-one-surface-out evaluation, reverting a particular promotion against its historical predecessor, and replaying optimization without one edit are different experiments with different costs and interpretations.

**Correction:** remove the child-verifier ablation text from the main discussion and appendix, including the with/without-label comparison. Joint batch evaluation remains the accepted design; no new per-edit experiment is required by this audit. If the authors retain a separate optional edit-ablation proposal, keep it prospective and define its counterfactual, base/final hashes, treatment of later edits, calls to rerun, and supported claim. Otherwise remove that unexecuted proposal too. Do not report an individual edit's "marginal benefit" from a shared batch score or an undefined subtraction.

#### #26. Implementation status and future baseline descriptions are internally inconsistent

**Paper locations:** `proposal.tex:468-492`, `:573-605`; `neurIPS_short_paper.tex:276`, `:373-406`; `proposal-old.tex:475-493`, `:511-547`.

The timeline says only mining is operational and proposal/validation remain to be built, despite multiple completed optimization runs. At the same time, fine-tuning and broader baseline evaluations remain planned. These need separate status labels. A proposal should not mix a stale implementation schedule with descriptions of completed experiments.

The short paper names DeepSeek as the shared backbone at :258 but its fine-tuning appendix names **Qwen** at :382-384. Its hardware sentence at :399 contains the corrupted text `8\times\lambda-RLM00`. The expanded draft's DeepSeek fine-tuning appendix is still a prospective method; no evidence from a completed fine-tuning study was established by this audit. Do not imply that every supported hosted deployment is directly fine-tunable with the described recipe without specifying the actual accessible checkpoint and implementation.

The lambda-RLM comparison must retain its status as a pinned **paper reconstruction** where applicable; current evaluation records `method_kind`, reconstruction version, upstream repository/revision, and source hash (`evaluation.py:413-419`). "Unmodified released baseline" is not interchangeable with a reconstruction.

**Correction:** remove or date obsolete schedules; mark unexecuted baselines and analyses as planned. Resolve model/hardware placeholders and distinguish "same backbone for all fixed-weight arms" from an optional future training arm. These details belong in the relevant experimental appendix, not a longer optimizer narrative.

#### #27. Document assembly has real missing inputs/references, plus a local build limitation

**Paper locations:** `neurIPS_short_paper.tex:264`, `:341`; `proposal-old.tex:1-4`, `:222`, `:337`; `Figure_1.tex:79`; `Figure_2.tex` surface-appendix reference.

Static inspection found:

- `neurIPS_short_paper.tex:264` inputs **`figure_short_paper`**, but no `figure_short_paper.tex` exists in `paper/`.
- The short paper includes Figure 1, whose caption references `fig:full-repl-contract`; the corresponding full-contract figure/label is not included there.
- The old paper includes shared figures referencing `fig:full-repl-contract` and `app:surfaces`, neither defined in its expanded source.
- `proposal-old.tex` begins with three `\usepackage` calls before `\documentclass`, which causes an immediate LaTeX error.

Build attempts for the expanded and short active papers stopped earlier on this machine because the `phvr8t` font metric was unavailable. That is a **local toolchain limitation**, not proof of a defect in the papers' prose. The missing short-paper include and unresolved references were independently confirmed by static include expansion. A successful complete PDF build, bibliography pass, and visual inspection remain outstanding. There were no unresolved local labels in static expansion of the expanded proposal.

**Correction:** restore/rename the intended short figure and explicitly include the full-contract appendix if its reference remains. For the old paper, fix build order/references only if it is still intended to compile; otherwise mark it as archived and keep it out of the active build. Compile each active paper through the full toolchain after revision. Do not claim this review produced validated final PDFs.

#### #28. Archived and alternative LaTeX sources need an explicit maintenance policy

**Paper locations:** all of `proposal-old.tex`, particularly :194, :216, :234, :285-314, :432, :475; all of `Figure_2B.tex` and `Figure_A1.tex`; includes listed in #27.

The user asked for all LaTeX files, so the old and alternative sources cannot be ignored. The old paper still has a nine-surface account, while its shared Figure 1 now displays S10. It mixes old counts/models/design with current included figures. Figure 2B repeats the obsolete validation flow even though it is not currently included by either active paper. Figure A1 is also standalone/not currently included; the expanded proposal instead repeats the full-contract listing inline.

**Correction:** either maintain the old draft as a current third paper and apply all relevant findings, or clearly identify it as a historical artifact with its original protocol/date and prevent active shared fragments from silently changing its meaning. Do not retroactively rewrite historical experimental facts. Update or retire alternative figures consistently with the active method. Avoid two independently maintained full-contract listings unless a check keeps them synchronized.

This is separate from the scientific correction itself: leaving an unlabeled contradictory draft available as though it were current makes future reuse and submission error-prone.

#### #29. The project checklist is unfilled and absent from the active papers

**Paper locations:** `checklist.tex:34-244`; active paper endings; `neurips_2026.tex:488`.

`checklist.tex` contains **16 unanswered `answerTODO` entries** and matching justification placeholders. Only the vendor example `neurips_2026.tex` includes it. Neither active project paper currently does. The provided checklist asks directly about scope of claims, reproducibility, limitations, uncertainty, and compute resources, all affected by this audit.

**Correction:** complete truthful project-specific answers when preparing the relevant submission and include them where required by its chosen track/template. In particular, do not claim seeded evaluation, unseen transfer, complete long evaluation, independent contexts, or verified causal diagnoses in the checklist after removing those claims from the body. Confirm submission requirements separately if publishing; this audit did not verify current venue rules on the web.

`neurips_2026.tex` itself is the official example document, not an additional SH-RLM paper. It has no project optimizer description to correct; keep it as a template/reference rather than editing its sample scientific content to match this experiment.

## Suggested concise main-text description

These are proposed replacements, not edits already applied. The method paragraph should be the same in substance in both active papers. Numeric settings belong to a named study rather than a universal algorithm definition.

> We optimize a fixed-model harness in rounds. The environment verifier evaluates final answers on mining tasks, and an attributor uses failed execution traces to infer failure mechanisms with explicit uncertainty. Diagnoses use a predefined mechanism vocabulary and runtime capability descriptions. The proposer receives selected operation-level evidence, the current harness, and a bounded history of prior attempts. It proposes up to a configured maximum of edits, with at most one admitted edit per surface, and may abstain.
>
> Admitted edits are composed into one candidate before evaluation. The candidate and a freshly evaluated incumbent run on the same validation tasks, which are excluded from mining but reused for adaptive selection. In the current configuration, promotion requires at least as many environment-verifier passes as the incumbent and compliance with the configured resource bands. Optional verifier-defined dense metrics inform diagnosis and history when available; they are not promotion requirements. The combined candidate is the measured unit. Optimization stops at the round limit or after the configured number of rounds without promotion; the final incumbent is frozen for separate evaluation.

For the present pairs study, follow that with a compact experimental paragraph or table:

> The configured OOLONG-Pairs study starts from H0*R, which includes hand-authored recursion/decomposition guidance, using the same DeepSeek-V4-Flash deployment for execution, attribution, and proposal. It uses 20 mining tasks with two attempts each and 10 validation tasks with one attempt each; the run is capped at three rounds. The verifier requires exact pair-set equality for a pass; ties in pass count qualify and the allowed mean-cost ratio is 0-3. F1 is a supplementary diagnostic for this environment. Mining, validation, and short test contain disjoint task instances on shared short-context records. The 10 short-test tasks measure performance on held-out task instances; the separate 40-task long evaluation assesses performance on larger record collections, with underlying question texts allowed to recur. Final evaluation is configured for three attempts per task.

This wording deliberately does not claim unseen-task-family transfer, statistically established improvement, or completed long evaluation. Those require their own results, not a longer methods paragraph. If a different run is chosen as the paper's principal experiment, substitute its frozen parameters and gate; do not attach these settings to earlier results.

## Appendix organization for each active paper

Six compact subsections can cover the implementation without making the main paper a study of the optimizer's prompt engineering. A versioned artifact supplement can carry full prompt text and exact edit payloads.

| Appendix subsection | Minimum content | Findings covered |
|---|---|---|
| A. Runtime and initial harness | Capability/scope table; H0/H0*/H0*R differences; actual initial and final hashes; immutable scaffold; S10 lazy loading and limits | #3, #20-21 |
| B. Mining and evidence | Final-answer verification; trace-based diagnosis and causal uncertainty; coverage-basis normalization; clustering/ranking; exclusion rules; complete-operation selection and budgets. Omit child-verifier descriptions | #4, #12-14 |
| C. Proposal and history | Versioned selections/literal-text formats; one surface/pattern ownership; behavioral fields; repair/abstention; materialization/preflight; compact history and metric comparability | #15-18 |
| D. Validation and stopping | Joint held-out comparison; exact inequality and bands; singleton/empty/duplicate cases; error/missing-run treatment; frozen-incumbent rule; concise algorithm | #1-2, #19, #23 |
| E. Experimental protocol and results provenance | Named launch/revision/config/splits; retained split and task-level evaluation scope; complete-record versus question-text overlap; lengths; models/decoder; repetitions and concurrency; completed/partial denominators; adaptive development and test exposure; ablation definitions | #5-10, #25-26 |
| F. Measurement and artifacts | Exact and dense quality definitions; activation/collapse metrics; accounting/attempt formula; reasoning availability; analysis versions and artifact paths | #11, #17-19, #22, #24 |

The expanded proposal's existing surface, full-contract, and ablation appendices can be reorganized into this structure. The short paper needs its own sufficient appendix; a reader should not have to locate the other draft to discover its starting harness or actual gate. Shared LaTeX fragments can prevent inconsistent copies, provided each paper includes them explicitly and archives are handled separately.

### Current contract versions to record with the artifact supplement

These identify the reviewed implementation, not every historical launch. A paper reporting an earlier run should read its frozen source/artifacts instead of copying this table.

| Contract | Current value | Source |
|---|---|---|
| Taxonomy / capabilities | `3.3.0` / `surface-capabilities/v1` | `optimization/taxonomy.py:48-49` |
| Attribution prompt / validator | `1.6.0` / `1.2.0` | `optimization/attribution.py:45-51` |
| Attribution digest | `1.8.0` | `optimization/digest.py:41` |
| Proposal prompt / validator | `4.3.0` / `4.2.0` | `optimization/proposal.py:159-175` |
| Proposal response / literal text | `proposal-selection/v2` / `literal-text/v1` | `optimization/proposal.py:136-137` |
| Evidence selector / diagnostic history | `4.3.0` / `2.1.0` | `optimization/proposal_evidence.py:35-36` |
| Compact history schema | `proposal-history/v3` | `optimization/history.py:12` |
| Validation / summary | `heldout-batch/v2` / `shrlm-validation-summary/v3` | `optimization/validation.py:83-84` |
| Behavior observations | `behavior-observations/v1` | `optimization/behavior.py` and saved summaries |
| Provider observations | `rlm-llm-observation/v1` | `rlm/core/llm_observation.py` |

### Replacement algorithm skeleton

```text
Inputs:
  configured initial harness H; mining tasks M; validation tasks V
  mining repetitions m; validation repetitions v; edit ceiling k
  round ceiling T; no-promotion patience p; verifier-pass/cost/subcall gate settings

history = empty
rounds_without_promotion = 0
for round in 1..T:
    execute or resume H on M with m attempts per task
    retain root outcomes, failures, execution traces, and available observations
    infer failure mechanisms for eligible failed runs from structured trace evidence
    cluster diagnoses; prepare admitted bounded evidence and relevant history
    generate and admit up to k edits on distinct patterns and surfaces
        retain valid siblings; allow one bounded failed-member repair response
    if no evaluable batch, or identical rejected batch already measured for H:
        record the non-promotion reason without new validation runs
        promoted = false
    else:
        C = compose admitted edits onto H
        freshly execute or resume H and C on V with v attempts per task
        if the baseline comparison is unscorable:
            persist the failure and stop/raise through the experiment error path
        else:
            promoted = completed comparable candidate satisfies verifier-pass and resource gates
            record the measured batch decision and unscored constituent membership
            if promoted:
                H = C
    record proposal outcomes and eligible aggregate validation feedback in history
    rounds_without_promotion = 0 if promoted else rounds_without_promotion + 1
    if rounds_without_promotion >= p:
        break
freeze H
run separately specified evaluation conditions on their fixed task sets
```

This skeleton omits artifact/cache plumbing for readability. It must not imply that infrastructure failures automatically become ordinary failed tasks, that every proposed edit gets a score, or that a held-out trace is fed back into diagnosis. Resume reuses identified completed work; "fresh" means a newly scheduled round comparison rather than a score carried forward from a previous round.

## Experimental lineage and claims the artifacts support

This section is supporting audit evidence, not material required in the revised papers. The paper revision targets the current method and upcoming experiments. Include a historical result only if the authors choose to report it, with its actual protocol.

The inventory below counts completed round markers and their promoted batch decisions. In-progress rounds are excluded from completed counts. A no-validation round can be complete. These are **development launches, not independent replicates of one fixed method**. All launch snapshots found from September 11 onward use `H0*R`, `m=2`, `v=1`, and `k=4`; the earliest unsuffixed directory has no frozen TOML at the inspected launch path, so its configuration must not be inferred from today's file.

| Saved experiment suffix | Launch commit | Mining/validation/short-test sizes | Cost band | Gate version | Completed rounds | Promoted batches |
|---|---|---|---|---|---:|---:|
| Unsuffixed `experiment_oolong_pairs_dsv4f` | Not available in inspected launch snapshot | Not inferred | Not inferred | Recorded `heldout-batch/v1` where validation exists | 6 | 3 |
| `20260911_1458` | `a1f930ecca` | 10 / 10 / 20 | 0.5-1.25 | v1, strict | 3 | 0 |
| `20260911_213930` | `b3a8058284` | 20 / 10 / 10 | 0-1.25 | v1, strict | 5 | 1 |
| `20260913_132607` | `3d8c8ef4a9` | 20 / 10 / 10 | 0-1.25 | v1, strict | 4 | 1 |
| `20260914_111506` | `fcc62f6e5c` | 20 / 10 / 10 | 0-1.25 | v1, strict | 3 | 0 |
| `20260915_180003` | `f95d06d88c` | 20 / 10 / 10 | 0-1.25 | v1, strict | 3 | 0 |
| `20260916_131837` | `7aad3bc54b` | 20 / 10 / 10 | 0-3 | v1, strict | 8 | 3 |
| `20260923_123738` | `0298b1ec74` | 20 / 10 / 10 | 0-3 | No completed validation ledger | 1 | 0 |
| `20260923_163355` | `99812551d5` | 20 / 10 / 10 | 0-3 | v1, strict | 5 | 1 |
| `20260924_161551_3rounds` | `ad2ac00147` | 20 / 10 / 10 | 0-3 | v2, inclusive | 3 | 2 |

`heldout-batch/v1` and `/v2` share joint held-out-only evaluation; the version change relevant here is the inclusive exact threshold. Older generic/repository examples can use other validation protocols and repetition counts, so this table should not be extrapolated to every environment.

Some especially relevant implementation transitions, to describe as development history if needed:

| Revision/change | Meaning for the paper |
|---|---|
| `2857ece3` | Cheaper held-out batch validation invalidates the individual/dual-split method and cost equation |
| `bcb82829` and related failure handling | Run-local failures can become measured failed attempts rather than terminate the entire optimization |
| September 13-14 evidence/behavior/history changes | Pair diagnostics, operation evidence, behavioral-difference explanations, revised routes/repair, and qualified dense-metric history alter proposal admission |
| `ee0b4999` | Host-owned literal escaping, task-derived reasoning guidance, compact evidence, and selection contracts address a different proposer failure mode |
| `7aad3bc5` | Cost ceiling rises to 3; more admissible expensive candidates is a gate change, not proof of better proposal reasoning |
| `0298b1ec`, `99812551` | Capability descriptions, evidence-supported S8/S5 routes, metric/behavior history, surface ownership, and coverage admission alter search |
| September 24 choices/history/observation changes | Admitted choices and predecessor-aware history plus broader provider-response capture alter optimizer input and recorded evidence |
| `ad2ac001` | Inclusive exact-pass gate and three-round assessment; historical strict decisions remain strict |
| `323dbceb` | September 28 input-definition/nearby-correction evidence and revised diagnosis/proposal guidance; offline-tested after the latest live run |
| `070d1a84` | Latest LaTeX update; it is not the source revision of the earlier experiment |

### Latest measured batch outcomes

Recomputed from saved verdicts and validation summaries in the September 24 launch:

| Round | Admitted surfaces | Exact incumbent / candidate | Mean F1 incumbent / candidate | Candidate/incumbent cost | Actual decision | Would the old strict exact rule pass? |
|---|---|---|---|---:|---|---|
| 1 | S8 + S10 | 1/10 / 2/10 | 0.7318 / 0.7602 | 1.460 | Promoted batch | Yes |
| 2 | S2 + S3 | 2/10 / 2/10 | 0.7908 / 0.6810 | 0.900 | Promoted batch | No |
| 3 | S4 + S10 | 3/10 / 1/10 | 0.6805 / 0.5882 | 0.902 | Rejected batch | No |

These are six admitted constituent edits, three measured candidates, two promoted batches, and four promoted constituent edits on four distinct surfaces. They do not establish six independent treatment effects. The final harness is `e9ae3b12d7c46da61d4f2974a5b7eccc0de8f47a2de95e9f4e4cd49d509f58d6`; it retains round 1's S8/S10 and round 2's S2/S3. The round 3 incumbent remeasurement is not the promotion-time F1 from round 2.

The task-level and full-stage spend reports overlap; the September 25 assessment reports about $29.14777 in task attempts and $29.30085 in full stage usage. Do not sum them. No significance or final test conclusion follows from the above three adaptive comparisons.

## File-by-file correction map

This map catches duplicates and dormant fragments, so correcting the main validation paragraph does not leave the old method in a diagram or appendix. It covers all nine `.tex` files. Ranges are locations to inspect, not a request to replace all text indiscriminately.

| File | Locations | Required response |
|---|---|---|
| [proposal.tex](../../paper/proposal.tex) | 124-126, 152, 192-198 | Qualify non-regression, causal grounding, initialization, and achieved-result claims (#2-4, #9) |
| Same | 204-226 | Runtime scope, actual starting harness, human priors, builder/sparsity assertions; Figure 1 caption (#3, #13, #20-21) |
| Same | 231-245 | Structured attribution, evidence selection, choice/repair/history contracts; batch held-out inclusive gate (#1-2, #12-19) |
| Same | 251-277 | Frozen-versus-preregistered protocol, actual source/target, reconstruction/verifier, counts, shared contexts (#4-8) |
| Same | 280 | Shared Figure 2 must receive the same algorithm corrections (#1-4) |
| Same | 284-306 | Exact initial comparator and truthful test/environment conditions (#3, #5, #9) |
| Same | 307-342 | Dense metric aggregation, unseeded repeats, analysis denominators, causal/ablation limits (#10, #17-19, #22, #24-25) |
| Same | 350-377 | Attempt formula and full algorithm replacement (#1-2, #11, #23) |
| Same | 384-418 | Recompute/labeled-date budgets, condition counts, calibration, telemetry (#11) |
| Same | 468-492 | Current implementation status, actual development chronology, test exposure (#8-9, #26) |
| Same | 506-570 | Accurate capability table and actual initial prompt listing/caption (#3, #16, #20-21) |
| Same | 573-605, 611-629 | Distinguish planned training baseline; remove child-verifier ablation; define any retained optional edit ablation (#25-26) |
| Same | 639-655 | Keep optional safety/generalization audits explicitly optional and separate from the implemented selection gate; update referenced budgets/repetitions (#2, #11, #25-26) |
| [neurIPS_short_paper.tex](../../paper/neurIPS_short_paper.tex) | 150-160, 184, 202-208 | Same contribution/grounding/initialization/result qualifications (#2-4, #9) |
| Same | 214-251 | Same runtime, mining, proposal, history, and validation corrections; remove automatic "withheld sub-verifier" pass claim (#1-4, #12-23, #25) |
| Same | 258-276 | Resolve GraphWalks-source claim, split sizes, actual baseline, seeded-run wording, incomplete evaluation, stale status (#3, #5-10, #26) |
| Same | 264 | Missing figure input (#27) |
| Same | 280-285 | Keep hypothetical results conditional; do not convert them into achieved claims without results (#9) |
| Same | 303-341 | Capability table, shared Figure 1, missing full-contract reference (#20-21, #27) |
| Same | 343-371 | Replace obsolete algorithm (#1-2, #23) |
| Same | 373-406 | Resolve Qwen/DeepSeek mismatch, corrupted hardware identifier, and future baseline status (#26) |
| Same | 411-429, 431-455 | Remove child-verifier ablation; define any retained optional leave-one-out study; keep optional audits distinct from promotion (#25) |
| Same | Before document end | Add complete appropriate project checklist if required by the intended submission (#29) |
| [proposal-old.tex](../../paper/proposal-old.tex) | 1-4; document as a whole | Decide historical versus maintained status; fix preamble order if compiling (#27-28) |
| Same | 194, 208-253, 263-265 | Nine-surface/sparse-initialization claims, runtime limits, grounding, taxonomy, proposal and obsolete validation (#1-4, #12-23, #28) |
| Same | 284-337 | Model/placeholders, nine-surface table, source/target, shared figure assumptions (#5, #20, #26-28) |
| Same | 343-409 | Statistics, metric definitions, splits, attempted cost formula and pseudocode (#1-2, #5-11, #17-19, #23-25) |
| Same | 432, 443-493 | Nine-surface summary, old budgets/timeline, preregistration/test exposure (#8-11, #26, #28) |
| Same | 511-547, 549-570, 572-end | Future training/ablation/audit status and counterfactuals (#25-26) |
| [Figure_1.tex](../../paper/Figure_1.tex) | 18-72 | H0 listing may remain as H0; it is not the actual H0*R initial prompt (#3) |
| Same | 74-89 | Starting-point implication, one-builder assertion, reference inclusion, S10 cost/scope (#3, #20-21, #27) |
| [Figure_2.tex](../../paper/Figure_2.tex) | 126-136, 166-176, 238-272 | Remove child-verifier correctness labels/legend and child-accuracy gate; retain call-tree evidence; show distinct surfaces and composition before joint held-out validation with inclusive verifier-pass gate (#1-4, #14-15) |
| [Figure_2B.tex](../../paper/Figure_2B.tex) | Same body regions through 249; caption 254-259 | Same flow corrections, or explicitly retire the unused alternative (#1-4, #28) |
| [Figure_A1.tex](../../paper/Figure_A1.tex) | 1-26 | Keep accurate H0 reference label; do not imply actual initial prompt; resolve duplication/inclusion (#3, #27-28) |
| [checklist.tex](../../paper/checklist.tex) | 34-244 | Fill 16 answers/justifications consistent with corrected claims and actual evidence (#29) |
| [neurips_2026.tex](../../paper/neurips_2026.tex) | Entire file, especially 488 | Vendor example only; no SH-RLM method to update. Its inclusion of the checklist does not include it in either project paper (#29) |

## Claims that should be retained, with their current scope

The audit is not a request to redesign the optimizer or discard correct existing text:

- The fixed-model comparison, configurable runtime surfaces, separate mining/validation task roles, and frozen final harness are real parts of the design.
- Each admitted edit changes one surface; current admission additionally enforces one edit per surface and pattern per round. Joint evaluation does not contradict single-surface constituent edits.
- The gate counts environment-verifier passes, with exact pair match as the OOLONG-Pairs pass definition. Optional verifier-defined metrics support diagnosis/history/reporting; pair precision/recall/F1 are environment-specific examples, not requirements for other tasks.
- The current proposer asks task-derived questions about information preservation and task predicates. Authoritative environment output contracts remain environment-specific. Describe both without claiming either total task independence or answer memorization.
- S10 is an on-demand skill library and does support removal. Its invocation and token-cost claims need qualification, not removal of the surface.
- The optional safety/alignment checks are not a live promotion gate in this profile. Their prospective status can remain if consistently labeled.
- Versioned source/config/harness/split artifacts and deterministic admission checks materially improve auditability, but do not turn a model diagnosis into a causal proof.

### Coverage

This was a sequential review in the main agent, as the repository's supplied AGENTS instructions require for subagent-style work. It covered correctness, interface contracts, experimental/data integrity, statistical interpretation, error/accounting behavior, and maintainability of duplicated paper descriptions. The skill's mechanical finding checks accepted all 29 entries with no malformed, suppressed, or duplicate entries; this is a schema/evidence-quote check, not independent scientific validation. No independent reviewer, cross-model peer, or independent validator was run; direct source checks, saved-artifact recomputation, and static/build checks are the validation evidence. Cross-model/independent corroboration is therefore absent, not silently claimed. Candidate concerns were reconciled against actual code; for example, S10 removal is supported and is explicitly retained rather than reported as a defect.

The review inspected the current tree and relevant development history from before the September 9 held-out batch change, plus earlier initialization/runtime definitions necessary to interpret current claims. All nine LaTeX sources were inventoried, including active, archived, alternative, checklist, and vendor-template roles. It reviewed the mining/attribution/digest/grounding/clustering/taxonomy modules; proposal/evidence/history/materialization/admission; validation/promotion/driver; orchestrator/splits/config/evaluation; harness/runtime hooks; dense metrics, behavior observations, and relevant analysis/storage documentation.

The evidence collector reconstructed saved settings and outcomes from ten local OOLONG-Pairs experiment directories. It did not independently re-adjudicate every semantic classification in every child trace, replay paid experiments, verify external literature or current provider/venue policies, or establish statistical efficacy of the newest prompts. Such claims remain outside the evidence. The exact long-evaluation comparison is checked only over completed matched attempts. No claim is made that an absent source snapshot can be reconstructed from today's defaults.

Validation performed: read-only source and history inspection; raw decision/manifest checks and recorded-F1 recomputation; corpus-prefix hashing across splits; static LaTeX include/label expansion; attempted builds of both active papers and the old paper. A read-only synthetic check of `derive_failing_level` confirmed that one checked-correct child plus one unchecked child produces `ROOT`. The evidence collector was run successfully and passed focused Ruff checks. Application tests were not rerun because no implementation changed; existing test reports are cited as historical evidence, not newly reproduced test results. Builds did not complete: active papers encountered a missing local font metric, and the old paper failed preamble ordering. Full PDF rendering and bibliography validation remain unverified.

### Verdict

**Not ready to describe the implemented experiment accurately without revisions.** There are **29 consolidated findings: 11 P1 methodological/experimental corrections and 18 P2 implementation, analysis, or document-assembly corrections**. This is a publication-readiness judgment about the descriptions, not a request to revert the implemented optimizer or alter past results.

Fix the main-text account of joint held-out validation, inclusive verifier-pass counts, and H0*R initialization, and remove child-verifier descriptions. Keep the current split and describe short-test results as performance on held-out task instances with shared short-context records; describe the separate long evaluation as performance on larger record collections without claiming entirely unseen question texts. A stronger split is needed only if unseen-source-data generalization becomes a central claim. The earlier launch history and unfinished long evaluation need discussion only if their results are used. Put the operational contracts in the appendix, update every duplicate figure/algorithm, and compile the complete papers. No change to the split, batch promotion, or absence of a secondary-metric gate is requested.

### Actionable Findings

All items below are proposed manual follow-up corrections owned by the paper/code maintainer; none was applied. The author decisions are which study to present, whether to retain the historical draft, and which future evaluation/ablation to run. The other corrections can be written directly from the verified evidence.

| # | Priority | Anchor | Required response |
|---|---|---|---|
| 1 | P1 | `proposal.tex:245` and duplicate validation text/figures | Describe composition followed by one held-out batch comparison; remove individual and post-merge validations |
| 2 | P1 | `proposal.tex:245`, `Figure_2.tex:264` | State inclusive environment-verifier-pass gate and resource bands; keep dense metrics optional and outside promotion |
| 3 | P1 | `proposal.tex:215`, `:284`; `Figure_1.tex:74` | Identify H0*R and human-designed priors; compare final against the actual initial harness |
| 4 | P1 | `proposal.tex:269`; `neurIPS_short_paper.tex:246` | Remove child-verifier descriptions, labels, algorithm steps, metrics, and ablations; retain final-answer verification and trace-based diagnosis |
| 5 | P1 | `proposal.tex:275`; `neurIPS_short_paper.tex:260` | Correct source/target, reconstruction, split counts, and context-length definitions |
| 6 | P1 | `proposal.tex:275-277` | Describe adaptive validation and aggregate history feedback; remove "unbiased" implication |
| 7 | P1 | `proposal.tex:275`, `:321` | Keep the split; describe held-out short task instances on shared records and larger long collections without unseen-source claims |
| 8 | P1 | `proposal.tex:251`, `:474-492` | Record the upcoming study's current protocol; historical version comparisons matter only if earlier results are reported |
| 9 | P1 | `proposal.tex:194-198`, `:304-325` | Restrict achieved claims to completed measurements; disclose partial and exposed long evaluation |
| 10 | P1 | `proposal.tex:321-325` | Remove per-attempt seeded-run claim; distinguish planned statistics and repetition counts |
| 11 | P1 | `proposal.tex:350`, `:384-418` | Recompute attempts/costs from batch protocol and explicit evaluation conditions |
| 12 | P2 | `proposal.tex:237-242` | Document attribution schema, uncertainty, normalization, and error exclusions |
| 13 | P2 | `neurIPS_short_paper.tex:246` | Correct many-to-many routing; document support/ranking and supported S8/S5 evidence |
| 14 | P2 | `proposal.tex:242` | Document bounded operation evidence, inventory/admission, contrasts, and actual prompt budgets |
| 15 | P2 | `proposal.tex:242`; `Figure_2.tex:166` | Document ownership, selection-before-text within one response, abstention, and repair scope |
| 16 | P2 | `proposal.tex:242-245` | Document literal-text materialization and exact structural preflight limits |
| 17 | P2 | `proposal.tex:242`, `:307-309` | Document bounded revision history, generic metric comparability, qualified positive signals, and F1 aggregation |
| 18 | P2 | `proposal.tex:327-342` | Separate activation observations from unassessed intended behavior |
| 19 | P2 | `proposal.tex:245`, `:321-335` | Document failure containment, incomplete subjects, denominator and resume treatment |
| 20 | P2 | `proposal.tex:506-537` and duplicate surface tables | Correct runtime scope, S2 schedule, S6 keys, S7 visibility, S8 builders, and S9 powers |
| 21 | P2 | `Figure_1.tex:84-85` | Correct lazy-skill token accounting and history persistence |
| 22 | P2 | `proposal.tex:237-242` / new artifact appendix | Describe provider-returned reasoning availability and distinguish saved observations from prompt input |
| 23 | P2 | `proposal.tex:358-377` and duplicate algorithms | Add attribution; reset patience only on promotion; freeze final incumbent |
| 24 | P2 | `proposal.tex:307-342` | Define plot/metric units, fresh measurements, completeness and grounding scope |
| 25 | P2 | `proposal.tex:611-629` and duplicate ablations | Remove child-verifier ablations; define any retained optional edit counterfactual without requiring per-edit validation |
| 26 | P2 | `proposal.tex:468`; `neurIPS_short_paper.tex:382-399` | Update status; resolve model/hardware inconsistency; label prospective/reconstructed baselines |
| 27 | P2 | `neurIPS_short_paper.tex:264`; `proposal-old.tex:1` | Repair missing input/reference and preamble defects; then compile with a complete toolchain |
| 28 | P2 | `proposal-old.tex:194`; `Figure_2B.tex:173` | Establish archive/active-fragment policy and eliminate unlabeled contradictory copies |
| 29 | P2 | `checklist.tex:34-244` | Complete and include the appropriate project checklist with truthful evidence references |
