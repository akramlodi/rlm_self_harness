---
title: Repair Drifted Tests and Fixtures - Plan
type: fix
date: 2026-09-28
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-plan-bootstrap
execution: code
---

# Repair Drifted Tests and Fixtures - Plan

## Goal Capsule

**Objective:** Contributors can run the offline test suite from a clean checkout without false failures from independently tuned experiment profiles or missing local experiment outputs.

**Means:** Correct config expectations and commit a test-owned loader fixture (KTD1–KTD2).

**Authority:** The user's request to fix the nine failures governs scope; R1–R4 govern behavior, and KTD1–KTD2 govern implementation. The executor owns implementation and verification. Stop when the Definition of Done is met; investigate any newly demonstrated production defect before broadening this test-maintenance change.

## Product Contract

### Summary

Repair five config-related assertions and four tests that depend on a removed smoke-run artifact. Keep meaningful regression coverage while making the suite independent of local experiment directories.

### Problem Frame

The saved full non-live run finished with **9 failed, 2,511 passed, 7 skipped, and 21 deselected**. The same nine failures reproduced on the prior implementation baseline, as recorded in `docs/analysis/2026-09-28-grounded-mining-proposal-verification.md`.

| Failure group | Count | Evidence |
|---|---:|---|
| Every shipped profile must use five mining workers | 2 | The DeepSeek pairs profile declares three, for both full and smoke. |
| Pair allocation must be 10/10/20 | 1 | The profile now declares 20 held-in, 10 held-out, and 10 short-test tasks. |
| GPT-OSS must inherit the Kimi profile | 1 | Caps, split sizes, and validation repetitions differ. |
| Materialized pair pools must have old sizes | 1 | The test expects 10/10/20/40 instead of the configured 20/10/10/40. |
| Loader fixture exists under `experiment_smoke/` | 4 | Its proposal was untracked when experiment directories became local-only. |

Commit `671da8a91` introduced the current pairs allocation and worker count. Commit `67b14b48` removed tracked experiment directories; `.gitignore` deliberately excludes `experiment_*/`. These are stale test assumptions, with no demonstrated production defect behind the nine failures.

### Requirements

**Config coverage**

- R1. Config tests validate each profile's declared settings and the correspondence between configured splits and materialized pools, without imposing equality between independently tuned profiles.

**Fixture coverage**

- R2. Loader fixture tests work with tracked files alone and retain envelope, surface-key, hash, candidate-loader, rendered-module, and proposal-writer consistency checks.

**Change boundaries and completion**

- R3. Preserve production behavior, experiment configuration, historical example artifacts, and the experiment ignore policy; do not hide failures through skips, xfails, or weakened integrity checks.
- R4. Resolve all nine failures and pass the full non-live suite, with no new lint or formatting violations in changed code.

## Planning Contract

### Key Technical Decisions

- KTD1. **Separate loader contracts from profile snapshots.** For R1, compare each full/smoke profile's effective mining worker count with its raw TOML declaration, including a smoke override when present. Keep an explicit snapshot of the pairs preset's intended 20/10/10 allocation and 40-task short inventory. Replace whole-profile GPT-OSS/Kimi equality with GPT-OSS-owned assertions for its caps, splits, and repetitions. In split materialization tests, derive expected counts from the loaded input config while retaining the independently specified output-file set. This preserves checks for loading and materialization errors without coupling independently tuned profiles.
- KTD2. **Use a committed synthetic golden fixture.** For R2, generate one small S4 proposal through the current production writer and materializer during implementation, then commit its bytes as static test data. Store the expected rendered module as `surfaces.py.txt` so formatting or linting generated source cannot alter the golden comparison. Never regenerate expected output inside the tests that compare against it. The existing S2 validation example remains a separate loader fixture.

Implementation uses existing serialization and materialization APIs. No fixture-generation framework, new dependency, or production change is needed. U1 and U2 are independent and can be completed sequentially.

## Implementation Units

### U1. Repair config and split expectations

**Goal:** Restore the five config-related failures while preserving R1, R3, and R4.

**Files:** `tests/experiment/test_config.py`, `tests/experiment/test_splits.py`.

**Approach:** Apply KTD1 at the four existing test locations. Rename the blanket-five worker test and obsolete inheritance test to state their actual contracts. The GPT-OSS full profile currently owns caps of 1.0 budget and 1800 timeout, splits of 24/24/24/0, and four validation repetitions; its existing provider, pricing, decoding, and identity tests remain relevant. Retain finite-pool, environment-selection, and disabled-real-check assertions for pairs.

**Test scenarios:** Both full and smoke load the declared worker count, including the existing explicit smoke-override case. The pairs allocation exhausts the 40 short tasks and selects only its expected pool files. Each manifest entry matches its configured count. Independently changing Kimi caps or splits does not invalidate the GPT-OSS preset test.

**Verification:** The two experiment test files pass, including existing invalid-worker, override, identity, and split integrity cases.

### U2. Replace the local smoke-run dependency

**Goal:** Restore the four missing-fixture failures while preserving R2–R4.

**Files:** `tests/optimization/test_loader_gated_fixtures.py`; new `tests/optimization/data/loader_gated/r01-c01-s4/proposal.json` and `surfaces.py.txt`.

**Approach:** Apply KTD2 using `write_proposal` in `shrlm/optimization/proposal.py` and the existing materialization path. Use stable synthetic metadata and a single generic S4 instruction edit. Parameterize explicit proposal and expected-module paths so the existing `examples/validation_rounds/proposals/smoke-s2-restate/` artifact keeps its `.py` companion. Update smoke-specific names and document that the new fixture is synthetic. Do not copy old experiment bytes blindly: the current writer emits metadata absent from the removed artifact.

**Test scenarios:** Both tracked fixtures retain the v2 envelope and eleven surface keys, recomputable hashes, accepted candidate loads, and exact rendered-module comparisons. The synthetic S4 proposal matches current writer output byte for byte. Missing files or future serialization drift remain test failures.

**Verification:** All tests in `test_loader_gated_fixtures.py` pass from tracked files, alongside the existing candidate and proposal tests.

## Verification Contract

Run the three changed test files first. Then run `tests/optimization/test_candidates.py` and `tests/optimization/test_proposal.py` to confirm the fixture still exercises the existing gate and writer contracts.

Run the final suite with `uv run --no-sync pytest -q -m 'not live'` from a clean checkout containing the proposed tracked files and no ignored experiment directories. Reuse the installed development environment; do not launch paid model calls or experiments. Record the final counts and any unrelated failures rather than masking them.

Run the repository's Ruff and pre-commit checks. Require changed Python files to pass lint and formatting; report pre-existing repository-wide issues separately rather than expanding this patch into unrelated cleanup. Confirm the JSON and text fixture are tracked and the diff contains only the planned tests, fixture data, and documentation.

## Definition of Done

U1 and U2 pass their verification, all nine baseline failures are resolved, and R1–R4 hold in the clean-checkout run. The final diff contains no abandoned fixture-generation scripts or experimental changes. Record verification results concisely for review.
