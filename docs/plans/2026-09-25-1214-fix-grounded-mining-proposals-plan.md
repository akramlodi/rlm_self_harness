---
title: Grounded Mining and Proposals - Plan
type: fix
date: 2026-09-25
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-plan-bootstrap
execution: code
---

# Grounded Mining and Proposals - Plan

## Goal Capsule

**Objective:** Spend fewer experiment rounds on edits whose rationale contradicts the trace or merely repeats behavior the harness already requests.
**Means:** Three local changes to diagnosis field guidance, evidence selection, and proposal instructions (KTD1–KTD3).
**Authority:** The user's task-agnostic, low-complexity constraint governs R4; requirements govern behavior and KTDs govern implementation.
**Execution:** Implement and verify U1–U3 on `plan/grounded-mining-proposals`. Stop at the verified change and report; publishing, merging, and another paid experiment require a subsequent request.

## Product Contract

### Summary

Teach the attributor to describe the observed failure before guessing its cause. Give it and the proposer nearby inputs, checks, and corrections. Ask the proposer to explain a changed operation or trigger even when it chooses another surface.

### Problem Frame

Diagnosis acceptance rose from **71/179 responses (39.7%) to 79/97 (81.4%)**. This is schema/reference acceptance before promotion, not edit acceptance. Missing `coverage_basis` refusals fell from 97 to zero. The inclusive exact-pass rule separately accounts for **two of four promoted edits**; both experiments had one batch that strictly improved exact passes. Later incumbents differ, so these are descriptive observations, not a causal ablation.

The [three-round assessment](../analysis/2026-09-25-three-round-inclusive-promotion-assessment.md) documents three remaining failures: unsupported intermediate-label claims, evidence missing later repairs, and familiar instructions repackaged on a different surface. Existing prompts already prohibit all three in general terms. Better evidence is the main intervention; the diagnosis change only consolidates how the existing fields express support and uncertainty. Its benefit remains unmeasured.

### Requirements

**Diagnosis and evidence**

- R1. Diagnosis instructions distinguish observed operations and outcomes from inferred causes, preserving uncertainty when intermediate correctness is unverified or later context is unavailable.
- R2. Evidence prioritizes the cited operation's direct inputs and nearby follow-up checks or corrections, with visible omissions when context cannot fit.

**Proposal behavior**

- R3. Proposal instructions require a concrete difference from both the observed execution and the incumbent behavior shown in context, regardless of target surface; a changed trigger or enforcement point can justify relocation.

**Complexity and experiment controls**

- R4. Keep this task-agnostic and within existing model calls, response fields, repair attempts, and character budgets; add no semantic judge, cross-surface matching system, or new optimizer stage.
- R5. Preserve combined held-out validation, the configured inclusive exact-pass threshold, cost bands, one edit per surface, and verifier-owned diagnostic metrics.

### Scope Boundaries

This change covers the three levers above and their regression coverage. Broader taxonomy/routing changes, forced surface diversity, finalization-runtime fixes, helper execution gates, and paper revisions are outside this plan.

## Planning Contract

- KTD1. **State observations before assigning a cause.** Consolidate the existing causal guidance into a short rule attached to the current fields: `operation_evidence` describes the visible action and result; `verification_limits` records what is unverified or contradicted by later evidence; `causal_status` reflects only the support those observations provide. Missing context is not evidence that an action never occurred. Preserve existing schema examples and add no worked failure narrative or benchmark-derived diagnosis example. This implements R1 through field guidance, without another field or validator, and follows R4's task-agnostic constraint.
- KTD2. **Prefer nearby context over distant name overlap.** In the existing selector, retain the cited operation and computational consumers, then prefer the next two complete operations after the last selected consumer over the current farthest related observation. Include a nearest prior definition for directly read argument names using the existing name analysis, without recursive dependency tracing. Include bounded ordinary root-response prose attached to selected iterations, labelled as the model's interpretation rather than verified fact. Enforce the existing 12,000-character digest and 32,000-character proposal-evidence budgets. Replace the six-snippet cutoff when it would split this bounded context group with whole-group budget admission and explicit omission, reusing `pack_evidence`; do not create another ranking framework. These selection hints support R2 but cannot prove recovery or semantic correctness.
- KTD3. **Compare behavior across the context already supplied.** Refine the existing three explanation fields and repair instructions to compare against all shown incumbent surfaces and compact history, rather than treating a new surface as a fresh idea. Require a changed operation, input, or invocation condition; identify an unavailable comparison as unverified. Keep the existing surface/mechanism revision gate unchanged. R3 is model guidance, not a claim that local validation can prove novelty.

## Implementation Units

### U1. Ground diagnosis in observations and counterevidence

**Goal:** Express R1 consistently through the existing diagnosis fields.
**Requirements:** R1, R4. **Dependencies:** None.
**Files:** `shrlm/optimization/attribution.py`, `tests/optimization/test_attribution.py`.
**Approach:** Apply KTD1 by replacing overlapping causal instructions with the short field guidance. Preserve the accepted uncertainty path and bounded operation citations. Bump the attribution prompt version, leaving response examples, schema, and validator behavior intact.
**Test scenarios:**

- Existing response examples continue to validate in grounded and ablated modes.
- A failed outcome whose cause is not established remains expressible as uncertain without inventing an intermediate failure.
- Existing observed-input-loss and accepted-uncertainty cases remain valid without extra re-asks.

**Verification:** Existing attribution tests cover contract compatibility; manual prompt review checks that field guidance expresses R1 without adding a failure-specific narrative. Assertions about prompt text do not establish model compliance.

### U2. Preserve inputs and follow-up checks in bounded evidence

**Goal:** Make the corrections currently omitted from evidence visible.
**Requirements:** R2, R4. **Dependencies:** None.
**Files:** `shrlm/optimization/digest.py`, `shrlm/optimization/proposal_evidence.py`, `tests/optimization/test_digest.py`, `tests/optimization/test_proposal_evidence.py`.
**Approach:** Apply KTD2 in the shared operation selector and both renderers. Retain complete code, exact source coordinates, existing omission reporting, and deterministic packing. Count added root prose in the rendered budget and avoid duplicating code embedded in that prose. Bump digest/evidence-selector versions. Keep child contract evidence where needed for existing routes.
**Test scenarios:**

- A faulty checker followed by a correction appears with that correction, ahead of a distant formatting block.
- A helper receives a variable holding the wrong argument shape; the nearest direct definition and call both appear without tracing an entire dependency graph.
- A child parse is repaired and followed by a label inspection; the packet shows the inspection without treating its conclusions as verified labels.
- A final profile print followed by submission retains the actual submission and its associated explanation.
- Oversized code, missing coordinates, and unparseable code produce explicit partial/unavailable context; outputs stay deterministic and within both budgets.
- Computational consumers still survive preview prints, and S8 child-contract/S5 recovery evidence remains available when supported.

**Verification:** Extend existing synthetic fixtures. Locally re-render witnesses `oolong-t14-w9-259b0efda7a2e2e8__a02` (checker correction), `oolong-t14-w9-259b0efda7a2e2e8__a01` (helper arguments), and `oolong-t11-w9-b4cbd11dd9c162ad__a01` (label checks) from `experiment_oolong_pairs_dsv4f_20260924_161551_3rounds`. Verify the missing operations now appear within the existing budgets. These traces are regression cases, not examples to insert into the standing diagnosis prompt. Use source coordinates and whole-operation assertions, not benchmark-specific selection rules. This replay makes no model calls and is not a required dependency for portable tests.

### U3. Require a changed execution when retargeting

**Goal:** Make surface relocation explain a substantive intervention.
**Requirements:** R3–R5. **Dependencies:** U1, U2.
**Files:** `shrlm/optimization/proposal.py`, `tests/optimization/test_proposal.py`, `tests/optimization/test_proposal_context.py`.
**Approach:** Apply KTD3 in initial and repair prompts. Replace duplicated guidance with one generic contrast: copying an already-followed check into a skill changes nothing; correcting the argument consumed by that check, or making a missed check run at the relevant point, changes execution. Tie `incumbent_behavior`, `observed_failure`, and `behavioral_change` to those distinctions within their current limits. Correct the stale “strictly more exact passes” wording to agree with R5. Bump the proposer prompt version without changing admission or history schemas.
**Test scenarios:**

- Retargeting from a prompt to a skill retains the comparison with the shown incumbent and prior attempt; moving the instruction alone is insufficient under the prompt contract.
- A different invocation/enforcement point is permitted when the explanation identifies how it changes the witnessed execution.
- Initial and repair prompts carry the same guidance while valid siblings, eligibility, and single-surface ownership remain intact.
- The prompt describes the inclusive promotion threshold without making dense-quality improvement a new gate.

**Verification:** Run existing proposal/context tests and inspect one reconstructed repair prompt containing a retargeted candidate. Use supplied candidate fixtures to verify context and contract preservation, not to claim that deterministic tests judge semantic novelty.

## Verification Contract

Run the focused attribution, digest, proposal-evidence, proposal, and proposal-context suites under `tests/optimization/` with the repository's existing `uv` environment, plus required Ruff/pre-commit checks from `AGENTS.md`. No paid experiment is part of implementation verification.

For a subsequent experiment, inspect the first admitted proposals against their held-in traces before reading validation results. Count unsupported causal claims, omitted observed checks/repairs, and proposals lacking a substantive execution difference. Report diagnosis response acceptance, proposal admission, and promotion separately. Compare promotion outcomes under the same rule; retain exact-pass and verifier-defined dense-metric changes together. Increased merge count alone does not establish success.

## Definition of Done

U1–U3 satisfy their scenarios, including visibility of the known missing operations in U2's replay. Explicit budget omissions pass the oversized-input edge case; they do not substitute for that regression proof. Rewritten prompts stay within R4 and describe R5 accurately. The implementation report distinguishes deterministic evidence/contract fixes from the still-unmeasured effect on proposal quality. Remove abandoned code and test artifacts, and leave experiment data and the paper untouched.
