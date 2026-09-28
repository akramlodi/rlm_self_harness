# Grounded mining and proposal changes: implementation verification

This implements the [grounded mining and proposals plan](../plans/2026-09-25-1214-fix-grounded-mining-proposals-plan.md) on `plan/grounded-mining-proposals`, starting from `ee77cdd8`. The changes improve what evidence the optimizer can see and how it is asked to interpret that evidence. Their effect on proposal quality and promotion remains unmeasured.

## Changes

- Diagnosis guidance now assigns a clear purpose to existing fields: describe the observed operation, describe what remains wrong after visible corrections, and state what is unverified. Missing context does not prove that an action never occurred. There is no new worked failure example or benchmark-derived diagnosis narrative.
- Evidence selection includes nearest direct input definitions, up to two computational consumers, and the next two complete operations after the last consumer. When no consumer is established, it uses the next two operations after the cited operation. These are bounded static and chronological hints; neither proves semantic correctness or recovery.
- Selected root iterations include bounded ordinary response prose, labelled as unverified interpretation. Fenced code is removed from that prose because complete code is shown separately. This does not introduce another reasoning-capture path.
- Proposal evidence admits complete context groups under the existing character budget instead of cutting them off after six snippets. One representative child prompt/return supplies contract context; explicitly cited additional children remain available. Code-block output excerpts are capped at 500 characters and root prose at 300; child prompt/return excerpts remain capped at 1,200.
- Initial and repair prompts require a changed operation, input, or invocation condition when comparing an edit with shown incumbent surfaces and prior attempts. Relocation can be justified by changed triggering or enforcement, but changing surfaces alone is insufficient. The quality guidance now accurately describes the configured inclusive exact-pass threshold.

The changes add no model calls, response fields, repair attempts, semantic judge, or optimizer stage. Digest and proposal-evidence budgets remain 12,000 and 32,000 characters. Validation, promotion, surface ownership, history schemas, and verifier-owned metrics are unchanged. Prompt versions are attribution `1.6.0` and proposal `4.3.0`; digest and evidence-selector versions are `1.8.0` and `4.3.0`.

## Saved-trace replay

Re-rendered the existing held-in evidence from `experiment_oolong_pairs_dsv4f_20260924_161551_3rounds` locally, without model calls or changes to experiment artifacts. Each witness was packed separately with its inventory entry to verify that the formerly missing operations remain visible within the existing proposal budget.

| Witness run | Previously omitted context now visible | Complete witness packet |
| --- | --- | --- |
| `oolong-t14-w9-259b0efda7a2e2e8__a02` | Corrected checker and follow-up, root iterations 13–14 | 15,836 characters |
| `oolong-t14-w9-259b0efda7a2e2e8__a01` | Helper argument definition at iteration 18, with the cited call and follow-up at 27 | 12,483 characters |
| `oolong-t11-w9-b4cbd11dd9c162ad__a01` | Later inspections at root iterations 7–8 | 18,817 characters |

The full-round replay reused the saved inventory and the existing `k=4` selection limit. Round 2 packed 25,695 characters and expanded pattern 0; previously it expanded patterns 0 and 5. Round 3 packed 29,565 characters and expanded patterns 3 and 11; previously it expanded patterns 3 and 8. All budgets include rendered metadata and truncation markers.

**Tradeoff:** more complete context leaves room for fewer mechanisms in round 2 and changes which mechanisms fit in round 3. The isolated witnesses establish visibility when selected; they do not imply that all three appear together in the default packets. No evidence budget or diversity quota was added to force a particular outcome.

## Verification

- Focused attribution, digest, proposal-evidence, proposal, and proposal-context suites: **312 passed**. Synthetic cases cover direct inputs, nearby corrections, print-only and unparseable origins, deterministic bounded packing, complete-code retention, uncertainty labels, and supported child/recovery routes.
- Reconstructed a mock repair that retargets an unchanged S3 candidate to S4 while retaining an independent S2 edit. Both initial and repair prompts contain the incumbent comparison guidance; the existing two-call repair and replay behavior are preserved.
- Scoped pre-commit checks passed. Repository-wide Ruff checks still report 10 lint violations and six formatting failures outside the changed files. Strict type checking of the four edited modules reports three existing diagnostics; the same three reproduce on `ee77cdd8`.
- Full non-live suite (`uv run --no-sync pytest -q -m 'not live'`): **2,511 passed, 7 skipped, 21 deselected, 9 failed** in 17m13s. All nine failures reproduce in an isolated copy of starting commit `ee77cdd8`: five concern existing experiment-config/split-size expectations, and four concern missing loader fixtures. No new failure was observed.
- Code review examined correctness, failure scenarios, project instructions, and R1–R5/U1–U3 coverage. Review was sequential in the main thread as required by this repository's agent mapping; it provides no independent reviewer corroboration.

Prompt review and deterministic tests establish contract and evidence-selection behavior. They do not establish that the model will make fewer unsupported causal claims or propose better interventions. A later experiment should assess those behaviors separately from diagnosis acceptance, proposal admission, and promotion, using the same promotion rule for comparisons.
