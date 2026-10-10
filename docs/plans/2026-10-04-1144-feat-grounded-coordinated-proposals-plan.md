---
title: "feat: Ground diagnosis and coordinate capability proposals - Plan"
type: feat
date: 2026-10-04T11:44:29-05:00
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-plan-bootstrap
execution: code
---

# feat: Ground diagnosis and coordinate capability proposals - Plan

## Goal Capsule

Improve the optimizer's ability to propose a concrete, executable response to an observed failure. Prioritize preserving diagnosis detail and allowing a capability plus its necessary caller to be proposed together. Keep target and protected behavior descriptions task agnostic, and make recovery status visible to evidence selection.

Deliver four changes to the existing mining/proposal path, deterministic integration tests, and a small optional Azure probe. Success means information reaches the proposer, coordinated edits survive or fail together, and constructed harnesses demonstrate actual invocation and behavior. It does not mean a higher promotion count proves better performance.

This plan explicitly excludes the optimization loop's proposed regression-testing stage. It does not change validation samples, repeat counts, promotion criteria, or experiment configuration.

## Product Contract

### Summary and problem frame

The [Self-Harness comparison](../analysis/2026-10-04-self-harness-loop-comparison/report.md) identified four relevant gaps in our implementation:

1. Diagnosis saves concrete mechanism, causal-status, and level explanations, but the proposer evidence projection drops them.
2. Admission enforces both one edit per surface and one edit per pattern. The second restriction can prevent an S8 helper or S10 procedure from being accompanied by its necessary invocation instruction.
3. Proposal intent does not consistently explain which task conditions should benefit and which already-working behaviors must remain intact. The review's case-scoring recommendation is deliberately narrowed here to pattern descriptions and evidence; scoring is deferred.
4. Whether a cited defect was repaired is left in prose. Repaired problems can remain attractive edit targets even when the remaining failure has a different or unknown cause.

The comparison's offline checks reproduce the first two gaps. The [transfer review](../analysis/2026-10-01-promoted-harness-transfer-review/report.md) also documents unused promoted capabilities and proposals aimed at already-repaired operations. These observations motivate the changes; they do not establish that the changes will improve benchmark scores.

### Requirements

| ID | Required behavior |
| --- | --- |
| R1 | Forward bounded saved diagnosis details into the evidence actually rendered for the proposer, alongside operation evidence and uncertainty. Label them as model assessments, not verified facts. |
| R2 | Permit an optional two-edit capability/caller pair for one evidenced failure pattern. Preserve one edit per surface and the existing combined batch evaluation. Never admit an orphaned pair member. |
| R3 | Describe expected benefit and protected behavior through task conditions and operations. Names, record IDs, and known answers must not become special-case repair rules. Use held-in evidence only. |
| R4 | Record `unresolved`, `recovered`, or `unknown` for the nominated defect. Prefer unresolved evidence; use recovered defects as contrast rather than repair targets. Preserve honest uncertainty and legacy readability. |
| R5 | Preserve current literal-text encoding, evidence budget, surface eligibility, preflight, valid unrelated edits, persistence/replay guarantees, verifier authority, and aggregate promotion semantics. |
| R6 | Verify these contracts with small deterministic harnesses and bounded optional live checks. Report invocation, observed behavior, and failures separately from model-generated claims. |

### Settled scope and non-goals

**Session-settled: user-directed.** R1 and R2 take priority. R3 concerns patterns of held-in behavior, not entity-specific memorization. The meta-harness remains task agnostic. Limited constructed tests are appropriate, even when the paired proposal is supplied by the test author.

**Deferred to follow-up work:** designing or restoring a held-in regression stage; generating regression banks; evaluating target/protected case subsets; changing `v`, split sizes, patience, exact-pass tie handling, or dense-metric gates; launching another full experiment. Existing software tests for the changes in this plan are in scope.

**Considered and not built:**

- A general dependency graph for edits: only a capability and one caller are needed. Reconsider if a demonstrated intervention requires more than two surfaces.
- New intent-scoring models or entity blacklists: neither establishes semantic generality. Use existing rationale fields, operation references, and renamed synthetic examples.
- New causal taxonomies, automatic clustering models, or multiple recovery fields: one resolution assessment and the existing evidence are enough to test the hypothesis.
- Production S8 invocation instrumentation: constructed helpers can expose invocation in test traces. Broader telemetry is separate work if real experiments require it.
- Extra repair attempts or a surface-diversity quota: retain the current repair allowance and evidence-based selection.

## Planning Contract

### Repository grounding

Reviewed local `main` at `e323f4328e0aa37b2cb5ff67cabd4729d1b0916c`; the comparison records upstream `Self-Harness` at `2720dbb3f52283684f4b85a1065d642df1779dd8`.

| Boundary | Existing implementation and consequence |
| --- | --- |
| Diagnosis → evidence | `AttributionDetail` in `shrlm/optimization/types.py` already stores three explanation fields. `load_proposal_evidence` in `proposal_evidence.py` omits them. |
| Selection → materialization | `proposal.py` reserves both surface and pattern; repair bookkeeping keys failures by pattern. A paired exception must change both admission and repair identity. |
| Published proposals → validation | `candidates.py:load_candidates` gates directories independently. Pair completeness must be checked here too, so a missing or rejected partner cannot leave an executable singleton. |
| Evidence allocation | `pack_evidence` chooses complete core packets under a 32,000-character budget, then extra operations, then a passing contrast. A useful contrast can lose its space to extras. |
| Proposal → history | `orchestrator.py:proposal_behavior` forwards behavior fields and predicted effect but drops `regression_risks`. `history.py` compacts the retained explanations. |

### Key technical decisions

**KTD1 — Preserve existing detail under the current budget. Governs R1, R5.** Add bounded `failing_level_detail`, `causal_status_detail`, and `agent_mechanism_detail` to each representative context. Start with a maximum of 600 characters per field, with explicit truncation markers. Keep complete cited operations, symptom, and verification limits; the global rendered budget remains authoritative. Missing legacy fields mean “not recorded.” For broad `other` clusters, explain that bucket support does not verify one representative's precise mechanism across all members.

**KTD2 — Bounded activation pairs. Governs R2, R5.** Each pair consists of exactly one S8 or S10 capability edit plus one S2 or S3 caller edit, both for the same pattern. Each surface must already be eligible with admitted evidence; do not broaden taxonomy routes merely to enable a pair. The proposer explains the shared intervention and why existing discovery/instructions do not already invoke the capability using the existing selection reason and behavior fields. A pair is optional, consumes two of `k` edit slots, and cannot coexist with another proposal on either surface or another intervention for that pattern. With fewer than two slots, it cannot be partially submitted. Distinct pairs may coexist if those same surface, pattern, and slot constraints permit them.

Use one optional `activation_pair` label on the model's selection and candidate entries. The host validates the two-member shape and stamps reciprocal partner candidate IDs into published pair metadata. Preserve the label and membership in frozen proposal results. Unpaired proposals retain their current fields and restriction to one edit per pattern. Update both the model response contract and the persisted proposal format: new writes use `shrlm-proposal/v2`, while readers continue accepting legacy unpaired v1 artifacts. Pair metadata is valid only in v2. Older loaders must reject the new format rather than ignore unknown pair fields. Update all format readers and relevant replay identities together; leave the serialized harness format unchanged.

**KTD3 — Pair admission is atomic across repair and reload. Governs R2, R5.** Materialize each member against the same incumbent under existing one-surface checks, but reserve surfaces and publish survivors only after both pass. On a member failure, the existing repair attempt must replace or withdraw the pair together; it may choose eligible, unoccupied surfaces while preserving capability/caller roles. Retain unrelated valid candidates. Key repair slots by stable member identity, not pattern alone. Do not silently downgrade a failed pair to a singleton.

At stage-three loading, check reciprocal membership, same pattern/incumbent, distinct surfaces, and allowed roles after individual gates. Missing, corrupt, mismatched, or rejected partners exclude both members before evaluation. Record explicit rejections and preserve unrelated candidates. Existing combined-harness preflight and batch validation still apply. History identifies both members as one intended intervention with a shared batch result; it does not assign individual causal credit.

**KTD4 — Reuse intent fields and reserve a useful contrast. Governs R3, R5.** Refine `predicted_effect` to name a task condition, changed operation, and expected benefit. Use `regression_risks` to name relevant already-working behavior, why the change should preserve it, and remaining uncertainty. Keep the existing three behavior fields. A suitable description is “preserve multiplicity when the predicate depends on counts”; an unsuitable one special-cases a named user or known answer. Existing case/operation references remain evidence provenance, never replacement-instruction constants.

Prefer a passing held-in attempt of the same relevant instance, otherwise an existing passing example sharing the relevant operation. After admitting complete failure cores, try one complete contrast before optional extra operations; retain the existing budget and mechanism limit. If no contrast fits or exists, record that absence. A passing answer does not prove intermediate correctness. Persist bounded risk/protection descriptions in history without claiming their predictions were tested. No new target-case manifest, subset scoring, or semantic judge is introduced.

**KTD5 — One defect-resolution field. Governs R4, R5.** Add a single live diagnosis field with values `unresolved`, `recovered`, and `unknown`, supported by existing operation references and explanation fields. Legacy absence is “not assessed,” not unresolved. Resolution concerns the cited defect, not whether the overall task passed. For example, a repaired parse failure and a later wrong computation are separate claims.

Use resolution in representative eligibility and ordering before evidence-slot allocation: exclude recovered-only patterns from actionable selection, prefer complete unresolved representatives over unknown/legacy representatives, and retain recovered operations as contrast when relevant. Do not replace an available unresolved representative merely because an uncertain one has fewer bytes. Keep raw records and signature identity unchanged; expose the resolution breakdown so total bucket frequency is not presented as unresolved support. A residual cost/failure claim must identify and cite the separate unresolved consequence; “there was an error earlier” is insufficient. Do not add a new attribution call or force uncertainty into a causal label.

**KTD6 — Bound verification and state what it proves. Governs R6.** Deterministic tests prove data flow, admission, execution plumbing, and expected synthetic behavior. A supplied paired proposal cannot prove autonomous discovery. A small free-choice live proposer call can provide an observation about discovery, but a miss is a reported result, not a reason to repeat until success.

Planning assumption: use a **$2.00 cumulative maximum** for optional Azure verification during implementation, not per script or retry. This is a conservative allowance chosen for the user's cost constraint, not a quoted user budget. Start offline; use at most eight tiny harness executions and three meta-model requests, further reduced as needed to fit the cost bound. Use the existing configured Azure model and pricing, one worker, bounded prompts/output/iterations/depth, and no full dataset run. All nested completions, transport/SDK retries, and repair calls count toward the bound. Reuse the conservative accounting approach in `examples/experiment_smoke.py`; do not launch that larger smoke script. If a pre-call upper bound cannot be established, skip the paid probe and record why. Post-call `max_budget` alone is not a hard spend ceiling.

### Admission flow

| Input | Admission and repair | Evaluation input |
| --- | --- | --- |
| Valid singleton | Existing checks | One surviving edit |
| Valid capability/caller pair | Check both, reserve both surfaces, persist reciprocal membership | Both edits in the combined batch |
| One invalid pair member | Repair or withdraw both; retain unrelated valid edits | Repaired complete pair or neither |
| Partner lost/rejected on reload | Reject surviving orphan with explicit reason | Neither member; unrelated edits remain |

## Implementation Units

### U1. Preserve concrete diagnosis details

**Goal:** Deliver the explanation already paid for to the proposer. **Requirements:** R1, R5. **Dependencies:** none; implement first.

**Files:** `shrlm/optimization/proposal_evidence.py`, `shrlm/optimization/proposal.py`; `tests/optimization/test_proposal_evidence.py`, `tests/optimization/test_proposal_context.py`.

**Approach:** Apply KTD1 in the shared context projection and rendered evidence, including alternatives. Update selector/prompt identities where inputs change. Make model attribution and broad-bucket limitations concise rather than adding another prompt section.

**Test scenarios:**

- A saved diagnosis contains its only concrete mechanism in `agent_mechanism_detail`; that detail reaches the actual proposer input with its source and cited operation intact.
- Missing legacy details remain explicitly unavailable; overlong details are bounded without splitting cited operations or exceeding the total budget.
- A broad `other` bucket does not relabel the representative explanation as a verified shared cause.

**Verification:** Check the final rendered prompt, not merely an intermediate dictionary. No additional model call is required.

### U2. Admit and persist a capability/caller pair

**Goal:** Make one evidenced intervention executable across two surfaces. **Requirements:** R2, R5. **Dependencies:** U1 for end-to-end evidence fixtures.

**Files:** `shrlm/optimization/proposal.py`, `shrlm/optimization/candidates.py`, `shrlm/optimization/history.py`, `shrlm/experiment/orchestrator.py`; `shrlm/optimization/validation.py` only if needed at the candidate-admission boundary; `tests/optimization/test_proposal.py`, `tests/optimization/test_candidates.py`, `tests/optimization/test_batch_validation.py`, `tests/optimization/test_history.py`, `tests/experiment/test_orchestrator.py`.

**Approach:**

1. Extend the response contract and selection parser per KTD2. Resolve the paired intervention before writing replacements, retaining current surface competition rules.
2. Apply KTD3 to materialization, occupancy, repair slot identity, frozen results, and publication. Reuse existing single-surface materializers.
3. Enforce reciprocal completeness on disk loading and include pair intent in existing history. Update affected versions/digests; accept legacy unpaired artifacts without inventing pair membership. Changed contracts must not silently reuse paid responses from an older contract.

**Test scenarios:**

- Eligible S8+S3 and S10+S2/S3 pairs for one pattern survive together; valid singletons still work.
- Duplicate surfaces, a third member, S3+S4 without a capability, mismatched patterns, missing evidence, and insufficient `k` are rejected with specific reasons.
- One failed member triggers pair repair or withdrawal; an unrelated valid edit remains byte-identical and retains its identity.
- Restart after a frozen result republishes the same pair and IDs. Missing/corrupt/rejected partner files cannot leave a singleton in validation.
- New paired envelopes fail the old v1 format contract; current readers still load genuine legacy singletons, and history reads both supported formats.
- A fake validation runner receives only the incumbent and combined surviving batch, with the existing sample/repeat policy and unchanged promotion result.

**Verification:** Demonstrate pair integrity from model response through persisted reload and batch planning. The existing unqualified same-pattern rejection test becomes explicit coverage of the narrow exception, not blanket acceptance.

### U3. Express benefits and protections as behavior patterns

**Goal:** Improve intent and contrast without encoding answer-specific fixes. **Requirements:** R3, R5. **Dependencies:** U1; integrate with U2 history fields.

**Files:** `shrlm/optimization/proposal.py`, `shrlm/optimization/proposal_evidence.py`, `shrlm/optimization/history.py`, `shrlm/experiment/orchestrator.py`; `tests/optimization/test_proposal_evidence.py`, `tests/optimization/test_proposal.py`, `tests/optimization/test_history.py`, `tests/experiment/test_orchestrator.py`.

**Approach:** Apply KTD4 to existing prompt fields, contrast ordering, and compact history. Keep opaque references for auditability. Preserve uncertainty when no passing comparison exists. Do not add scoring or admission claims that a host cannot substantiate from these fields.

**Test scenarios:**

- A relevant passing attempt gets space before redundant optional snippets while complete failure operations and budget limits remain intact.
- Absent or oversized passing evidence is explicitly reported; no heldout/test record is introduced into proposer evidence.
- Proposal intent and protected-pattern risks survive publication and history compaction with provenance, without becoming observed outcome claims.
- Constructed edits behave identically under renamed entities and reordered inputs where ordering is irrelevant; count-sensitive and count-insensitive conditions exercise both target and protected behavior.

**Verification:** Confirm the prompt asks for conditions and operations rather than case-specific corrections. Semantic generality is assessed through examples and execution, not an entity-name regex.

### U4. Distinguish recovered defects during evidence selection

**Goal:** Stop already-repaired operations from consuming actionable proposal slots. **Requirements:** R4, R5. **Dependencies:** U1 and U3 evidence paths.

**Files:** `shrlm/optimization/types.py`, `shrlm/optimization/attribution.py`, `shrlm/optimization/proposal_evidence.py`, `shrlm/optimization/proposal.py`; `tests/optimization/test_types.py`, `tests/optimization/test_attribution.py`, `tests/optimization/test_proposal_evidence.py`.

**Approach:** Apply KTD5 in diagnosis schema/prompt/validation and evidence allocation. Update the relevant prompt, validator, and selector versions. Preserve current causal-status and coverage-basis checks. Recovery annotations remain model assessments and cannot override verifier outcomes.

**Test scenarios:**

- An error followed by a successful repair is recovered even if a later operation fails; a separate unresolved defect remains eligible.
- A cheaper recovered or unknown representative does not displace an available complete unresolved representative of the pattern.
- A mixed bucket retains original records and reports its resolution breakdown; an all-recovered bucket remains visible but cannot consume an actionable slot ahead of another mechanism.
- Unknown and legacy diagnoses are not silently called unresolved or rejected solely for uncertainty; existing evidence/causal requirements still apply.
- Repeated recovery that consumes a budget is actionable only through a cited residual cost/failure mechanism. No extra diagnosis pass is introduced.

**Verification:** Inspect the selection presented to the proposer, not just the serialized enum; preserve the existing incomplete-coverage normalization behavior.

### U5. Demonstrate execution with small harnesses and document the contract

**Goal:** Verify that these changes can alter execution, with honest limits on conclusions. **Requirements:** R1–R6. **Dependencies:** U1–U4.

**Files:** new `tests/optimization/test_coordinated_harness.py`, `examples/meta_harness_probe.py`, and `docs/harness-proposal-interface.md` (referenced by code but absent in this checkout); existing `tests/optimization/test_proposal_evidence.py` as needed and `meta-harness-improvements.md`.

**Approach:**

1. Build two small task families independent of benchmark entities, such as multiplicity-sensitive aggregation and typed value normalization. Use explicit synthetic oracles and existing runner/materializer fixtures.
2. Exercise an S8 helper/caller pair and an S10 procedure/caller pair. Observe actual helper invocation or skill loading plus the resulting computation. Loading alone does not establish correct use. Keep invocation markers local to fixtures.
3. Add the opt-in probe under KTD6. Save harness hashes, supplied/generated proposals, call/usage records, observed invocation, expected/actual outputs, and failures to a new timestamped output directory. Summarize observations in a small analysis artifact.
4. Update the proposal interface and collaborator summary to describe implemented behavior and verification limits. Record any paper follow-up there; this work does not revise experimental claims or launch experiments.

**Test scenarios:**

- Hand-authored baseline, capability-only, caller-only, and paired harnesses go through real materialization. A scripted model makes the control flow deterministic; the combined harness demonstrably invokes the new capability and satisfies the constructed oracle.
- A protected input and a renamed/reordered variant preserve the intended invariant. A malformed member cannot run through the integration path.
- Offline/default probe mode makes no paid calls. The cost calculation rejects a probe whose nested/retry ceiling exceeds the allowance.
- Optional live execution uses one task family: four harness variants on one target-pattern input and one protected-pattern input, at most eight executions total. Both families and renamed variants retain offline coverage. Add at most two diagnosis requests plus one free-choice proposer request within the same allowance. Record a missed invocation, wrong answer, or absent pair as a failure/inconclusive observation without repeated sampling to obtain success.

**Verification:** The report separates structural admission, actual activation, correct synthetic output, and autonomous proposal generation. No result is presented as evidence of benchmark generalization or an individual edit's contribution to a historical batch.

## Verification Contract

During implementation, run the focused optimization tests covering attribution/types, proposal context/evidence, proposal admission, candidates, history, combined validation, and the new coordinated-harness fixtures, plus the affected experiment/orchestrator tests. Use `uv run pytest` for these tests and the repository's required checks before a PR. Run Ruff on changed files during development; avoid unrelated formatting churn. Broaden testing when interface changes or failures justify it.

The required outcomes are:

- Saved concrete diagnoses reach the actual prompt within the unchanged evidence budget.
- A valid pair is accepted, executed, persisted, and reloaded together; invalid or incomplete pairs never reach evaluation as singletons.
- Existing unrelated candidates, old unpaired artifacts, literal encoding, and replay behavior remain covered.
- Pattern intent and uncertainty survive history; recovery status affects selection rather than merely adding text.
- Constructed execution demonstrates invocation and expected behavior in two task families, including protected and renamed variants.
- Validation sample membership, repeat counts, acceptance criteria, and batch evaluation count remain unchanged.

Live verification is supplemental, bounded by KTD6, and must report actual cost and incomplete evidence honestly. An unavailable endpoint or unprovable spend ceiling does not invalidate deterministic wiring tests, but must be reported as “live behavior not assessed.” A live semantic failure becomes a recorded finding, not a hidden retry or automatic claim of success.

## Definition of Done

- R1–R6 have implementation and test evidence, with no regression-stage or experiment-config changes.
- Prompt/schema identities and persisted pair metadata support correct repair and restart; legacy artifacts remain readable where promised.
- The interface and collaborator documents match the final code, including the limits of model judgments and supplied test proposals.
- Verification results distinguish offline guarantees from live observations; optional Azure spend stays within the stated cumulative allowance.
- Remaining uncertainty is empirical: whether freer proposals and better evidence improve benchmark performance. No launch-blocking product question remains for this scoped implementation.

## Planning notes

No production code, tests, or paid calls were executed for this plan. The comparison's previously recorded 310 passing tests and offline reproductions are grounding evidence, not verification of the proposed changes. Implementation should preserve unrelated workspace changes and experiment artifacts.
