---
title: Sub-Verifier Ablatability - Plan
type: fix
date: 2026-08-08
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Sub-Verifier Ablatability - Plan

## Goal Capsule

- **Objective:** Make the sub-verification ablation (proposal Appendix "Sub-verification") exercisable and proven: one persisted round can be mined both with and without the sub-verifier, both evidence bundles coexist and audit clean, and the mode difference is exactly the designed signal.
- **Authority:** The investigation findings in this plan's Problem Frame (verified against code); the ablation contract in `shrlm/optimization/grounding.py` / `mining.py` docstrings ("single ablation switch; nothing else varies"); `paper/proposal.tex` Appendix ablations.
- **Execution profile:** Offline only — MockLM and fixtures; no model calls, no dataset downloads. One unit at a time on branch `feat/weakness-mining` (extends PR #5).
- **Stop conditions:** No behavior change to grounded-signal design (excerpt selection stays verdict-aware — documented, not "fixed"); no re-generation of the committed example round; surface a blocker if a fix would change existing bundle ids for already-persisted legacy-layout rounds.

---

## Product Contract

### Summary

Parameterize where a bundle triplet lands so two modes' bundles coexist per round, make prompt persistence content-addressed so a second mining pass cannot break the first bundle's audit link, stop ablated digests from asserting a sub-verifier statistic that was never computed, and close the test gaps: a both-modes end-to-end round, grounded-mode attribution units, and config/identity assertions.

### Problem Frame

The ablation is the project's central causal claim (does checkable child-level evidence make mined attributions actionable?), but today it cannot run as a controlled comparison. The ablated mode is the only mode ever exercised end-to-end; no test constructs a miner with a sub-verifier. Structurally, one round directory holds exactly one bundle: the two modes hash to different bundle ids, so the second mine is refused by the non-clobber guard, and `overwrite` would replace rather than sit beside. A second mining pass also rewrites `attributor_prompt.txt`, retroactively breaking the first bundle's audit link. Two evidence leaks additionally violate "nothing else varies": ablated aggregate digests render `sub_verifier_failed=0` (indistinguishable from "ran and all passed"), and the fixture config hardcodes `sub_verifier_enabled: True` while the tests it feeds run ablated.

### Requirements

**Side-by-side execution**

- R1. The same persisted round can be mined with and without a sub-verifier; both bundle triplets persist under the round and both audit walks pass without cross-contamination.
- R2. A second mining pass never invalidates an earlier bundle's audit links (prompt files, digests, records).

**Honest mode separation**

- R3. An ablated digest never asserts a sub-verifier statistic that was not computed; "no sub-verifier ran" is distinguishable from "sub-verifier ran, all passed" in digest text.
- R4. `MiningConfig.sub_verifier_enabled` is true iff a sub-verifier was supplied, and flips the bundle id.
- R5. The excerpt-selection difference between modes (verdict-aware focus) is documented as designed signal in the module that owns it.

**Test coverage**

- R6. One MockLM round mined both ways end-to-end: divergent bundle ids, `level_grounded`/`n_ungrounded` correct per mode, both audits clean, multi-variant prompt files resolved.
- R7. Attribution mode gaps closed: grounded prompt omits the failing-level menu and field; `validate()` rejects a missing `failing_level` only when ungrounded; `cache_key` separates modes for the same digest.
- R8. The test fixture config no longer misdescribes the mode it runs in.

### Scope Boundaries

- The full-optimization ablation run itself (proposal Appendix) is stage-3/loop work — out.
- Re-generating `examples/mining_rounds/round_00` — out; legacy layout must keep auditing clean.
- Changing verdict-aware focus-excerpt selection — out (R5 documents it instead; neutralizing it would hide the grounding signal the ablation measures).
- The `paper/proposal.tex` §3.3.1 wording update runs alongside this plan but outside it (`paper/` is git-ignored on this branch).

---

## Planning Contract

### Key Technical Decisions

- KTD1. **Bundle destination becomes a parameter; the default is unchanged.** (session-settled: user-approved — chosen over a separate output directory per mode: keeps one audit chain per round.) `write_bundle`, `mine_round`, `run_audited_round`, and `audit_round` accept an optional bundle directory (default: the round root, exactly today's layout). Ablation callers pass `round_NN/bundles/<label>/` for the second mode. Legacy rounds — including the committed example — stay valid with zero migration; the audit resolves shared round artifacts (manifest, traces, digests, attributor prompt files, the attribution cache, harness, instances) from the round root and only the triplet (`bundle.json`, `records.jsonl`, `attributions.jsonl`) from the bundle directory. `attribution_cache_path` stays round-root-relative in both modes. Note: `write_bundle`'s existing `out_dir` is an experiment directory that appends `round_NN`; the new parameter names the triplet's directory directly, bypassing that join.
- KTD2. **Prompt persistence is always content-addressed.** The driver writes `attributor_prompt_<sha16>.txt` per variant and stops rewriting the unsuffixed file; the audit resolver already prefers the suffixed name and falls back to the legacy one, so old rounds resolve unchanged. Multiple passes and mixed variants coexist by construction.
- KTD3. **Ablated aggregate digests render `sub_verifier_failed=n/a`** when no verdicts were computed (any actual verdict set, even all-passing, keeps the count). `DIGEST_VERSION` bumps, which changes bundle ids via `MiningConfig.digest_version` only — it is not part of the attribution cache key. Cache invalidation for affected records happens through the digest bytes themselves: the `n/a` rendering changes ablated wide-tree digest text, hence those records' cache keys; grounded records' cached attributions deliberately survive the bump.
- KTD4. **`created_at` reuse keys on the bundle being rewritten**, not the round: `run_audited_round` reads the existing bundle in its own destination directory only, so each mode reproduces its own bundle byte-for-byte on re-invocation.

### Risks & Dependencies

- The `bundles/<label>/` layout adds a second valid shape the audit and any future tooling must know; the both-modes e2e test is the guard against drift.
- KTD3 changes ablated digest bytes and — via the `DIGEST_VERSION` pin in the mining config — the bundle id of any round mined after the change; grounded digest bytes and their cached attributions are untouched. Acceptable pre-release; the committed example round is untouched.

---

## Implementation Units

### U1. Parameterized bundle destination with dual-mode audit

- **Goal:** Two bundles per round, each auditable (R1, KTD1, KTD4).
- **Requirements:** R1; KTD1, KTD4.
- **Dependencies:** none.
- **Files:** `shrlm/optimization/bundle.py`, `shrlm/optimization/driver.py`, `shrlm/optimization/audit.py`, `tests/optimization/test_bundle.py`, `tests/optimization/test_audit.py`.
- **Approach:**
  1. Thread an optional bundle-directory parameter through `write_bundle` → `mine_round` → `run_audited_round` → `audit_round`, defaulting to the round root.
  2. Audit: triplet paths resolve against the bundle directory; shared round artifacts — including attributor prompt files and the attribution cache — against the round root; broken-link names unchanged.
  3. `run_audited_round` reuses `created_at` from the existing bundle in the destination directory only (KTD4).
- **Patterns to follow:** `round_dir` helper; note `write_bundle`'s existing `out_dir` is an experiment directory that internally appends `round_NN` — the new bundle-directory parameter names the triplet's directory directly (KTD1), a different semantic.
- **Test scenarios:**
  - Two `write_bundle` calls with different configs into root and `bundles/ablated/` → both persist; non-clobber guard untouched within each destination.
  - `audit_round` against the subdirectory bundle passes with all links resolving to round-root artifacts; against the root bundle unchanged.
  - Legacy layout (committed example round) still audits clean.
  - Re-invoking `run_audited_round` per destination is byte-idempotent for that destination.
- **Verification:** `uv run pytest tests/optimization -q` green; `python -m shrlm.optimization.audit examples/mining_rounds 0` still exits 0.

### U2. Content-addressed prompt persistence

- **Goal:** A second mining pass cannot break an earlier bundle's prompt link (R2, KTD2).
- **Requirements:** R2; KTD2.
- **Dependencies:** none (lands before or with U1).
- **Files:** `shrlm/optimization/driver.py`, `tests/optimization/test_driver.py`, `tests/optimization/test_audit.py`.
- **Approach:** `_persist_mining_artifacts` writes each rendered variant as `attributor_prompt_<sha16>.txt` unconditionally; never rewrites or removes existing prompt files; drop the single-variant unsuffixed branch (audit fallback keeps legacy rounds resolving).
- **Test scenarios:**
  - Mining the same round twice with different prompt variants leaves both files; the first bundle's audit still passes.
  - Suffixed-name resolution path exercised (previously untested branch).
  - Legacy round with only `attributor_prompt.txt` still resolves.
- **Verification:** audit link `attribution_prompt` proven for suffixed, legacy, and mixed cases.

### U3. Honest ablated digests

- **Goal:** No fabricated sub-verifier statistics in ablated mode (R3, R5, KTD3).
- **Requirements:** R3, R5; KTD3.
- **Dependencies:** none.
- **Files:** `shrlm/optimization/digest.py`, `shrlm/optimization/grounding.py` (docstring qualification), `tests/optimization/test_digest.py`.
- **Approach:** aggregate table renders `sub_verifier_failed=n/a` when the tree carries no computed verdicts; bump `DIGEST_VERSION`; document the R5 excerpt-selection design note in the digest module, and qualify the "nothing else varies" contract where it is stated (`grounding.py` / miner docstrings) to name the deliberately surfaced sub-verdict evidence.
- **Test scenarios:**
  - Wide ablated fixture → aggregate line shows `n/a`, never `0`.
  - Wide grounded fixture with zero failures → shows `0` (ran-and-passed stays distinguishable).
  - The same wide ablated fixture built before/after the change produces different digest sha and therefore different attribution cache keys; a grounded fixture's digest bytes and cache key are unchanged (DIGEST_VERSION reaches bundle ids only, via the config).
- **Verification:** digest tests green; no other digest bytes change for grounded fixtures.

### U4. Both-modes end-to-end ablation test

- **Goal:** The controlled comparison the ablation depends on, proven offline (R6, R4 partially).
- **Requirements:** R6, R4 (partial); exercises KTD1–KTD4 together.
- **Dependencies:** U1, U2, U3.
- **Files:** `tests/optimization/test_ablation.py` (new), reusing driver/audit test helpers and a stub SubVerifier over fixture trees.
- **Approach:** one MockLM round; one shared on-disk attribution cache at the round root (both bundles stamp the identical round-root-relative path); mine grounded (stub SubVerifier returning mixed verdicts) into the round root and ablated into `bundles/ablated/`; compare.
- **Test scenarios:**
  - Bundle ids diverge; `sub_verifier_enabled` True/False respectively.
  - The two persisted bundle configs differ in exactly `{sub_verifier_enabled}` — any future config field that accidentally varies with mode breaks this by construction.
  - Grounded mode: records with checkable verdicts are `level_grounded=True` with derived levels; ablated mode: every failure record ungrounded, `n_ungrounded == n_failures`.
  - Both audit walks pass; grounded round's mixed prompt variants (grounded + any UNDETERMINED-fallback ungrounded) all resolve as files; the shared cache path link audits in both walks.
  - For records the grounded pass actually grounded, the cache never serves a grounded response to the ablated pass or vice versa (call-count assertion scoped to grounded-variant records; an all-uncheckable record is byte-identical across modes and legitimately shares its cache entry — that hit is the cache working, not contamination; assert the sharing explicitly).
- **Verification:** this test is the plan's definition of "ablation exercisable".

### U5. Attribution-mode units and config identity

- **Goal:** Close the remaining mode-separation gaps (R4, R7, R8).
- **Requirements:** R4, R7, R8.
- **Dependencies:** none.
- **Files:** `tests/optimization/test_attribution.py`, `tests/optimization/test_bundle.py`, `tests/optimization/fixtures.py`.
- **Approach:** pure test additions plus the fixture correction; no production code expected — if an assertion fails, that failure is a finding to surface, not silently fix.
- **Test scenarios:**
  - `system_prompt(grounded=True)` contains no failing-level menu and no `failing_level` JSON field; `grounded=False` contains both.
  - `validate()` rejects a response missing `failing_level` when ungrounded; accepts the same response when grounded.
  - `cache_key` differs between modes for one digest.
  - `sub_verifier_enabled` flip alone changes `compute_bundle_id`.
  - `fixtures.make_config` defaults match the mode the fixture rounds actually run (`sub_verifier_enabled=False`), overridable.
- **Verification:** all new assertions green against current code, or divergences reported.

---

## Verification Contract

| Gate | Command | Applies to |
|---|---|---|
| Offline tests | `uv run pytest tests/optimization -q` | every unit |
| Lint/format | `uv run ruff check shrlm/ tests/` + `ruff format --check` | every unit |
| Legacy-round audit | `python -m shrlm.optimization.audit examples/mining_rounds 0` exits 0 | U1, U2 |

No live model runs required; the ablation e2e is MockLM-only by design.

## Definition of Done

- U1–U5 landed in dependency order; suite and lint green; committed example round still audits clean with no migration.
- The both-modes e2e test passes and pins: coexisting bundles, divergent ids, correct grounded/ungrounded accounting, clean dual audits, non-clobbering prompts.
- Ablated digests show `n/a`, never a fabricated `0`; excerpt-selection design note recorded (R5).
- No abandoned experimental code in the diff.
