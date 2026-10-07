# Meta-harness improvements

Updated September 28, 2026. This summarizes the implemented optimization process at `070d1a8493d79999ce4a5b01b19a2d0dc85b7637` and the accompanying paper corrections. The meta-harness is the optimizer that mines failures, proposes changes, and decides promotion. The candidate harness is the runtime configuration it edits. This document describes what changed and why; it does not establish that each change independently improves performance.

## The current loop

Each round runs the incumbent on mining tasks, diagnoses eligible failures from execution evidence, proposes a bounded set of edits on distinct surfaces, and composes the admitted edits into one candidate. That candidate and a freshly evaluated incumbent run on the same validation tasks. One batch decision determines whether the incumbent is replaced. The final incumbent is frozen for separate evaluation.

The optimizer remains task agnostic: environments provide task instances, final-answer verifiers, answer contracts, and optional metric definitions. The general prompts reason about information preservation, operations, and task conditions. They do not contain a required OOLONG-Pairs solution recipe. Environment adapters and the initial harness can still contain task-informed choices; these should be disclosed separately.

## What changed

### Cheaper validation and joint promotion

- **Validation runs only on held-out validation tasks.** Mining tasks generate weaknesses and are no longer a second validation arm.
- **One attempt per validation task in the current profile (`v=1`).** Mining still uses two attempts, and final evaluation still uses three attempts per task per condition. These settings serve different purposes.
- **Compose first, validate once.** All usable edits are applied to the same incumbent before evaluation. There is no individual-edit tournament followed by an additional merge check. A singleton is the candidate; an empty batch schedules no validation.
- **Promote the batch as a unit.** Constituent edits have batch membership, not individual performance scores. Joint promotion does not identify which edit caused a gain.
- **Accept ties in verifier-pass count.** Both configured thresholds are zero, so the candidate must have at least as many held-out passes as the incumbent. The current mean-cost band is 0–3 times incumbent cost; subcall count is unconstrained. The general rule uses the environment verifier, with exact pair-set equality supplying OOLONG-Pairs passes. F1 is not required for promotion, and a tie may promote despite falling F1.
- **Remeasure the incumbent each round.** Compare against its score from the same round, not its earlier promotion-time score. Stop at the round ceiling or after the configured number of rounds without promotion; freeze the last incumbent, not the best historical F1 point.

An ordinary round with a candidate schedules `m * n_in + 2 * v * n_ho` task attempts. The current 20/10 profile schedules `2*20 + 2*1*10 = 60`, regardless of how many edits are composed. Root task attempts can contain many model calls, and attribution/proposal/repair calls add cost. No usable candidate or an eligible duplicate-batch skip removes the validation leg.

Sources: [validation](shrlm/optimization/validation.py), [promotion](shrlm/optimization/promotion.py), [orchestrator](shrlm/experiment/orchestrator.py).

### Better-supported diagnoses

- **Separate observed outcomes from inferred mechanisms.** The final-answer verifier establishes whether the answer passes. The attributor infers why it failed from execution traces and must cite supporting operations and retain uncertainty. Accepted diagnosis responses are not verified causal explanations.
- **Require evidence of input loss.** Missing output elements do not establish that input records were skipped. `incomplete_coverage` now requires a structured basis: status, input scope, loss observation, and counterevidence. Unsupported or contradicted coverage claims become `OTHER`/`UNATTRIBUTED` and cannot authorize a coverage edit.
- **Account for checks and recovery.** Evidence and prompts ask whether an apparent fault was repaired later and whether an existing check already addresses it. A code excerpt that omits a check is not proof that the check never happened. Incorrect labels must not automatically become an aggregation diagnosis, or vice versa.
- **Retain unattributed and excluded outcomes.** Some environment-owned failures are excluded from diagnosis but remain recorded. Time/budget exhaustion can still provide efficiency evidence when the exhausted resource is identified. Passing runs are not sent through failure attribution.
- **Rank support by distinct tasks.** Clustering uses the recorded outcome/location/causal-status/mechanism signature, with distinct-instance support ahead of a fixed actionability heuristic. Repeated attempts are recorded separately. Minimum support two is a flag, not automatic deletion of a plausible rare mechanism.

Sources: [mining](shrlm/optimization/mining.py), [attribution](shrlm/optimization/attribution.py), [clustering](shrlm/optimization/clustering.py), [taxonomy](shrlm/optimization/taxonomy.py).

### Fewer, more complete pieces of evidence

- **Select the relevant operation.** Evidence-node references locate the child call and nearby parsing, combining, filtering, or verification code. This replaces a first/last-root-block view that could show only an input preview and final submission.
- **Include local context and counterevidence.** Selection can bring in preceding input definitions, downstream consumers and corrections, and a passing contrast sharing operation names. An earlier error is interpreted in light of later execution.
- **Keep code operations whole.** Required operations that cannot fit are omitted with scope markers rather than presented as complete snippets cut through the operation. Outputs and prose can still be shortened. Omission is not evidence that the omitted operation was absent.
- **Allocate across mechanisms.** The proposer gets a compact inventory plus a small number of expanded, actionable patterns, preferring distinct mechanisms before more examples of the same mechanism. Inventory-only entries cannot authorize proposals.
- **Bound prompt components.** Defaults are 12,000 characters for an attribution digest, 32,000 for proposal evidence, and 12,000 for compact history. At most `min(k, 4)` patterns are expanded. These are separate component budgets, not a total prompt-size guarantee; current surfaces and other instructions also take space.

The selector can still show only one actionable mechanism when context is large or other diagnoses lack support. These changes improve the evidence available for judgment without guaranteeing diverse successful proposals.

Sources: [digest](shrlm/optimization/digest.py), [proposal evidence](shrlm/optimization/proposal_evidence.py).

### Proposals must describe a concrete change in execution

- **Compare against the actual incumbent.** Three explanation fields describe what it already does, what observed behavior remains wrong, and what changes at a particular operation/input/trigger. The prompt asks whether the same failure could still occur for the same reason if the edit were followed perfectly.
- **Require more than emphasis or relocation.** Repeating an instruction, moving it to another surface, or repairing an already-recovered error is insufficient without a substantive behavioral difference. Host checks can reject literal no-ops, but do not prove semantic novelty.
- **Replace misleading examples when appropriate.** “Minimal edit” means the smallest effective change. Replacing an obsolete worked example can be better than appending a conflicting instruction.
- **Select interventions before writing replacements.** A single response lists choices before edit payloads. This is not a separate model-planning stage.
- **Enforce surface ownership.** The first admitted intervention occupies its surface(s) and pattern for the round. Normally this is one edit; the October 4 update below permits a narrowly defined capability/caller pair on two distinct surfaces. Later candidates cannot reuse them; invalid candidates do not permanently occupy a surface. Four edits is a ceiling, not a quota. One reasonable surface should produce one edit; no supported intervention can produce none.
- **Keep valid siblings during repair.** One failed-member repair response can select another eligible unoccupied surface or withdraw the failed edit, limited to the failed original patterns. It cannot rewrite retained valid siblings or introduce unrelated patterns. Response/schema retries are separately bounded; they are not experiment validations.
- **Use task-derived reasoning.** Instead of prescribing record IDs and label-count recipes, the general prompt asks what must survive each step and which task conditions must be enforced. Identity, multiplicity, ordering, dates, provenance, and asymmetric roles are examples of information requirements, not an OOLONG-only algorithm.

Source: [proposal generation and admission](shrlm/optimization/proposal.py).

### Make all surfaces available within their actual capabilities

The goal is to admit useful edits beyond S2/S4 when evidence supports them, not to satisfy a surface-diversity quota. Mechanisms map to multiple eligible surfaces; capability descriptions are shared with diagnosis and proposal.

| Surfaces | What the optimizer can change | Key limitation |
|---|---|---|
| S1 | Environment/API contract text | Describe available capabilities, not invented APIs. |
| S2–S5 | Decomposition, execution, checking, and recovery instructions | All enter the shared root/recursive-child system prompt; instructions are not host-enforced timed hooks. |
| S6 | Runtime-policy dictionary | Root-local enforcement except inherited depth. Caps refuse work; syntax retries repeat the same prompt. No general timeout recovery or calls-per-turn field. |
| S7 | Execution-output formatter | Bounded stdout and redacted variable types/lengths, not hidden values or full call history. |
| S8 | One named root/child helper function | Runs only when invoked; identify the caller and deterministic input/output contract. |
| S9 | Answer middleware | Answer text and redacted inventory only; can transform/accept or redirect, but cannot determine hidden semantic correctness. |
| S10 | One named text skill, including removal | The index advertises a procedure; its body matters only if loaded or forwarded. Loading is not execution or compliance. |

Lossy aggregation routes first to S3/S4, with supported S8 available for a concrete operation. S9 is reserved for answer-visible defects. Supported S8 parsing routes require the cited operation and child-response evidence. Additional S5 routes for execution/budget failures require an observed error and a subsequent complete operation; that establishes an opportunity to act, not successful recovery. Bare completions and recursion-limit leaves do not have the recursive child's REPL capabilities.

Sources: [capabilities and routing](shrlm/optimization/taxonomy.py), [runtime harness](shrlm/rlm_harness.py), [runner](shrlm/runner.py).

### Let the host handle template syntax

Text proposals use `literal-text/v1`: the model writes literal instruction text and the host escapes template braces exactly once, preserving the designated live custom-tools marker. This removes a recurring nested JSON/Python/template burden from model repair.

Materialization still checks base identity, serialization, a genuine single-surface change, signatures, reserved names, resource bounds, formatting, and selected smoke/answer fixtures. S1–S5 replace text; S6 supplies a dictionary; S7/S9 supply functions; S8 and S10 merge one named entry. Passing preflight establishes structural admissibility, not task correctness. Exact-edit reports should distinguish literal text from encoded source and rendered instructions.

Sources: [proposal materialization](shrlm/optimization/proposal.py), [candidate checks](shrlm/optimization/candidates.py), [skill edits](shrlm/optimization/skill_edit.py).

### History retains qualified progress and substantive revisions

History records attempted edits, reasons, batch membership, measured outcomes, and revision links. The prompt receives a bounded selection of recent rounds, the incumbent's promotion, relevant predecessors, and an older promising direction when space allows. Revisited interventions must cite a relevant predecessor and describe a changed operation or joint hypothesis. An identical previously rejected batch under the same incumbent can be skipped before spending on validation again.

“Potentially promising” means a rejected **measured** subject improved a comparable verifier-defined dense metric in the declared direction. It is not a synonym for promotion or statistical confidence. Definitions carry metric identity/version, direction, aggregation, precision, and terminal values; comparisons check compatible protocols and matching instance-attempt sets. Tasks without comparable dense metrics remain `not_assessed`.

Two development examples illustrate why the distinction matters:

| September 16 experiment | Subject | Exact passes | Recorded mean F1 | Interpretation |
|---|---|---|---|---|
| Round 6 | S6 singleton | 1/10 → 0/10 | 0.6194 → 0.7193 | Rejected; qualified dense-metric improvement with exact-pass regression. |
| Round 8 | Combined candidate | 3/10 → 1/10 | 0.6128 → 0.6951 | Rejected; batch-level signal, not individual edit credit. |

These are historical motivating observations, not outcomes of the next study or evidence that the latest prompt changes caused a gain. The often-quoted diagnosis acceptance increase of 39.7% → 81.4% concerns response admission, not promotions or verified causal accuracy; concurrent development changes prevent isolating a single cause.

Activation summaries now accompany outcomes: root syntax retries, root answer redirects, and skill loads across the traced call tree, with coverage. `intended_behavior` remains `not_assessed`. A loaded skill or activated hook does not establish that the proposed procedure was followed correctly.

Sources: [history](shrlm/optimization/history.py), [metric comparisons](shrlm/optimization/proposal_evidence.py), [verifier metric definitions](shrlm/environments/diagnostics.py), [behavior observations](shrlm/optimization/behavior.py).

### Failed attempts, checkpoints, and reasoning observations

Candidate-owned execution failures can be recorded without crashing the whole experiment. Completed failures enter scoring denominators; missing/canceled attempts remain distinct. An incomplete baseline cannot anchor promotion. Operational/provider failures, configuration errors, cancellation, memory/I/O problems, and persistence errors can still stop work. Resume reuses matching completed work and preserves identity checks; it does not create new independent attempts or spend.

Provider-returned reasoning and response observations are saved across instrumented mining, validation, evaluation, attribution, proposal/repair, recursive calls, compaction, and fallback paths. They are separate, hash-referenced artifacts with availability labels such as `returned`, `not_returned`, `opaque_only`, and `legacy_unavailable`. They do not guarantee access to full private reasoning and are not automatically fed back into optimizer prompts. Unknown reasoning-token counts stay unknown; they are not charged twice. See [reasoning artifact locations and interpretation](docs/experiment-reasoning.md).

Sources: [driver](shrlm/optimization/driver.py), [evaluation](shrlm/experiment/evaluation.py), [observation storage](shrlm/optimization/llm_observation_store.py).

### Analysis reports now match the measured unit

Surface plots count constituent membership in combined promotions. Quality curves use each round's fresh candidate score on promotion or fresh incumbent score on rejection, without carrying older promotion-time scores forward as new observations. Cost reports derive environment labels from the configuration rather than hardcoding GraphWalks. Reporting distinguishes task attempts from provider calls, planned from completed counts, and actual usage from reservations/projections.

OOLONG-Pairs mean F1 averages recorded per-attempt values, rounded to three decimals, including verifier-defined terminal zeros. Exact pass still uses set equality. Missing/extra-pair counts are reported with availability, not silently imputed to zero. This metric is task specific; the optimizer's metric interface is general.

Sources: [surface activity](shrlm/experiment/surface_activity.py), [incumbent quality](shrlm/experiment/incumbent_quality.py), [OOLONG-Pairs scoring](shrlm/environments/oolong_pairs.py).

## Current study settings and interpretation

| Setting | Current OOLONG-Pairs profile |
|---|---|
| Initial harness | H0*R: upstream-style prompt, orchestrator addendum, explicit recursive-call contract, hand-authored S2 example. |
| Tasks | 20 mining / 10 validation / 10 short test / 40 long test. |
| Attempts | Mining 2; validation 1 per subject; final evaluation 3 per task per condition. |
| Loop | Up to 4 admitted edits on distinct surfaces; at most 3 rounds; patience 3. |
| Promotion | Candidate verifier passes ≥ incumbent; mean cost within 0–3×; no F1 gate. |
| Model | Azure Foundry DeepSeek-V4-Flash for runner, attributor, proposer; temperature 0, top-p 0.95, output cap 8,192. |
| Run limits | 30 iterations, depth 3, $2 usage budget, 3,600 seconds; $100 cumulative budget per validation subject. |
| Concurrency | 3 mining run workers; up to 2 actual validation subjects despite 5 configured subject slots, with 1 run worker each. Evaluation has a separate worker setting. |
| Accounting rates | Configured $0.19/M input and $0.51/M output; dated accounting inputs, not a current market quote. |

Source: [experiment configuration](configs/experiment_oolong_pairs_DeepSeekV4Flash.toml). Parsed values take precedence over stale explanatory comments.

**Keep the current split.** The short roles contain disjoint task instances on the same two 188-record windows. Short-test results measure performance on held-out task instances. The long test uses two different 6,374-record windows and measures performance on larger record collections. Underlying question texts recur across windows, even where complete `(user_id, date, question_text)` records do not. Do not claim entirely unseen question texts. A stronger split is needed only if unseen-source-data generalization becomes a central claim.

For the main comparison, explicitly request `initial` versus `sh_rlm`: `b1` is the sparse H0 reference, not the current H0*R starting harness. The default evaluation grid contains `b1`, `h0_star`, `lambda_rlm`, and `sh_rlm`, and omits `initial`. On all 50 short/long tasks, the initial/final comparison schedules 300 attempts; the default four-condition grid schedules 600. Lambda is a pinned paper reconstruction with provenance, not an unmodified released OOLONG-Pairs baseline. No new experiment was launched for this documentation update.

## Paper changes and maintenance

Both active manuscripts now include the same concise [method](paper/optimization_method.tex), [study protocol](paper/experiment_protocol.tex), and detailed [implementation appendix](paper/optimization_appendix.tex). Main text gives the three-stage loop, measured unit, gate, initialization, and evaluation scope. Appendices cover surface capabilities, diagnosis/evidence, proposal/repair/history, validation/stopping, dataset/scoring, accounting, and saved observations.

The revised papers remove child-verifier descriptions, correctness labels, algorithm steps, derived metrics, and ablations. This is a paper correction, not removal of optional code integrations. They also remove obsolete dual-split/per-edit validation equations, strict-improvement claims, stale schedules and cost projections, and claims of already-demonstrated length/transfer gains. Fine-tuning and optional follow-up studies remain prospective, without invented checkpoint or hardware commitments.

Entry points are [proposal.tex](paper/proposal.tex) and [neurIPS_short_paper.tex](paper/neurIPS_short_paper.tex). `proposal-old.tex` has been removed from the working tree; its historical content remains in git at the revision above. `Figure_2B.tex` delegates to the maintained `Figure_2.tex`. `Figure_A1.tex` is the single full sparse-H0 contract listing and is included in each appendix. The vendor example `neurips_2026.tex` and style file remain templates, not additional project manuscripts.

Results remain forthcoming. Before reporting the next study, attach its actual launch/config/split/harness identities, completed denominators, metrics, resource totals, and analysis version. The project checklist describes this draft's current status and must be refreshed when results and release artifacts exist. The paper describes the current method, not a retrospective narrative of every development launch.

The [paper revision verification](docs/analysis/2026-09-28-paper-revision-verification.md) maps the audit findings to the changes and records the successful builds and output locations.

## October 4: concrete diagnosis and coordinated capability proposals

Four changes address gaps found in the Self-Harness loop comparison while retaining the RLM's surface model:

1. **Preserve concrete diagnoses.** The proposer now receives the saved mechanism, level, and causal explanations alongside the cited operations and uncertainty. Each explanation is bounded; the evidence budget is unchanged. Broad taxonomy buckets do not turn one representative's explanation into a verified shared cause.
2. **Allow a capability and its caller together.** One pattern can optionally produce an S8 helper or S10 skill plus an S2/S3 instruction that invokes it. Both must already be eligible, use distinct surfaces, and consume two edit slots. Both survive or fail together through local checks, repair, publication, replay, and validation loading. Unrelated valid edits remain. This is a two-member exception, not a general dependency system.
3. **Describe benefits and protections as behavior patterns.** Existing rationale fields now ask which task conditions benefit, which operation changes, and which already-working behaviors should remain intact. Relevant passing held-in evidence gets space before optional snippets, and bounded protection descriptions survive history. These are predictions, without new subset scoring or guarantees of no regressions.
4. **Distinguish a repaired defect from a remaining defect.** Diagnosis adds one resolution assessment. Complete unresolved evidence is preferred; recovered-only patterns remain visible as contrast rather than actionable repair targets. Unknown and legacy evidence retain explicit uncertainty. Raw signature identities, verifier outcomes, and causal checks remain authoritative.

The [proposal interface](docs/harness-proposal-interface.md) documents the new response/persistence versions and legacy reading rules. The [verification report](docs/analysis/2026-10-04-grounded-coordinated-proposals-results.md) distinguishes constructed execution from live behavior. Synthetic helper and skill pairs execute correctly with scripted responses, including protected and renamed inputs. The small live counting probe answered all inputs correctly but did not invoke the supplied helper; its free-choice proposer chose a direct S3 fix. These observations do not establish better benchmark proposals or generalization.

**Paper follow-up:** add the retained diagnosis detail, optional atomic capability/caller pair, pattern-based intent/contrast, and defect-resolution selection to the implementation appendix, with a minimal main-text clarification where needed. Record schema changes as implementation details. Do not describe paired proposals as individual contribution estimates, protection descriptions as tested regression cases, or the synthetic checks as benchmark gains. This update leaves the LaTeX files unchanged. Regression-stage design, validation repeat counts, splits, promotion criteria, and experiment launches remain separate work.
