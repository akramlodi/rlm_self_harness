---
title: Orchestrator Addendum as Starting Incumbent - Plan
type: feat
date: 2026-08-27
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Orchestrator Addendum as Starting Incumbent - Plan

## Goal Capsule

- **Objective:** A Self-Harness run on GraphWalks can start from a harness under which Kimi-K2.5 issues sub-calls, so the optimization loop mines and attributes decomposition behaviour instead of a runner that never recurses.
- **Means:** Make the loop's starting incumbent a config choice and point it at the registry's existing `H0*` harness, which carries the original RLM repo's `ORCHESTRATOR_ADDENDUM` (KTD1, KTD2).
- **Authority:** Product Contract requirements govern behaviour; Planning Contract KTDs govern mechanism; the frozen `H0*` definition in `shrlm/rlm_harness.py` is not edited by this plan.
- **Stop conditions:** Stop and report if the live recursion test shows zero sub-calls on every selected instance under `H0*` (KTD5). Stop if the config-key change cannot be made without editing `H0`'s surface text.
- **Execution profile:** Small, bounded code change plus tests; the live test spends real Azure money and is opt-in.
- **Tail ownership:** The implementer runs the offline suite and, with credentials, the live recursion test once, and reports the observed sub-call counts.

---

## Product Contract

### Summary

Add a `[loop] initial_harness` config key so an experiment starts from `H0*` (the shipped RLM system prompt plus `ORCHESTRATOR_ADDENDUM`) instead of the bare `H0`. Prove with offline tests that the assembled prompt carries the addendum and with an opt-in live test that Kimi-K2.5 issues at least one sub-call on the largest held-in GraphWalks instances.

### Problem Frame

`experiment_kimi/POST_MORTEM.md` §8 records that 0 of 5,104 runs issued an `rlm_query` or `llm_query`. The loop's starting harness is hard-coded as `INITIAL_INCUMBENT = "H0"` in `shrlm/experiment/orchestrator.py`. `H0`'s five prompt surfaces total six sentences and say nothing about when to delegate. The original RLM repo's decomposition guidance already exists in this tree as `ORCHESTRATOR_ADDENDUM` in `rlm/utils/prompts.py`, and the registry already exposes it as `H0*` (`shrlm/rlm_harness.py`), but nothing in the experiment can select it: the candidate gate in `shrlm/optimization/candidates.py` rejects any change to the `orchestrator` scalar, so the loop cannot discover it either. The failure taxonomy the attributor applies is a taxonomy of decomposition failures; with no sub-calls it has nothing to label.

### Requirements

**Harness selection**

- R1. An experiment config names its starting incumbent from the harness registry, with `H0` as the default so existing configs keep their meaning.
- R2. The starting-incumbent choice is part of the experiment identity, so a run that starts from `H0*` cannot resume into an out-dir created from `H0`.
- R3. `configs/experiment_kimiK25.toml` selects `H0*`.
- R4. The `H0` and `H0*` registry definitions are unchanged; their existing invariant tests in `tests/test_harness_surfaces.py` keep passing.

**Proof of recursion**

- R5. An offline test proves the effective system prompt for an `H0*`-started run contains `ORCHESTRATOR_ADDENDUM`, and that an `H0`-started run does not.
- R6. An offline test proves the loop and the candidate gate accept an `H0*` incumbent: a one-surface candidate on `H0*` materializes, and a candidate that flips `orchestrator` is rejected.
- R7. A live test, skipped unless the repo's live gate is open, runs `H0*` against a fixed handful of the largest held-in GraphWalks instances and asserts that at least one selected instance produced a sub-call.
- R8. The live test reports per-instance sub-call counts, verifier cause, and cost in its output regardless of pass or fail, and is bounded by an explicit per-run budget and timeout.

### Success Criteria

- The live test, run once by the implementer with credentials, reports a non-zero sub-call count on at least one selected instance. If it reports zero on all of them, that result is recorded under the follow-ups in `experiment_kimi/POST_MORTEM.md` rather than worked around (KTD5).

### Scope Boundaries

- The addendum is not copied into `H0`'s `S2`/`S3` text (KTD2).
- No S6 runtime-policy edit (`max_prompt_chars`) is made to force chunking; that is the next lever if `H0*` alone does not recurse.
- The verifier answer-format contract, the `extract_question` bug, the promotion-rule noise, the `format_tools_for_prompt` docstring drop, and the `MECHANISM_SURFACE` mapping from the post-mortem are out of scope.
- No new full experiment is launched by this plan.

#### Deferred to Follow-Up Work

- Enable S6 `max_prompt_chars` below the largest instance size if `H0*` does not recurse on 110k-char graphs.
- Re-run the Kimi experiment from `H0*` into a fresh out-dir once the post-mortem's verifier-format fix has landed, so measured pass rate reflects capability.

### Sources

- `rlm/utils/prompts.py` — `RLM_SYSTEM_PROMPT`, `ORCHESTRATOR_ADDENDUM`, `build_rlm_system_prompt(orchestrator=...)`.
- `shrlm/rlm_harness.py` — `H0`, `H0_STAR`, `HARNESSES`, `assemble_system_prompt`.
- `shrlm/runner.py` — `effective_system_prompt`, `build_harnessed_rlm`, `run_metrics` (`sub_call_count`).
- `shrlm/experiment/orchestrator.py` — `INITIAL_INCUMBENT`, the `incumbent = HARNESSES[...]` assignment in `run`.
- `shrlm/experiment/config.py` — `LoopConfig`, `IDENTITY_SECTIONS`, `identity_hash` (serializes each identity section with `asdict`).
- `shrlm/optimization/candidates.py` — the orchestrator-scalar gate (`GATE_SURFACE_DIFF`).
- `shrlm/optimization/walker.py` / `shrlm/optimization/types.py` — `walk`, `TreeStats.n_rlm_children`, `TreeStats.n_llm_leaves`.
- `tests/experiment/test_smoke_live.py`, `shrlm/experiment/live_gates.py` — `LIVE_FLAG`, `live_skip_reason`, azure credential keys.
- `experiment_kimi/opt/round_01/mining/round_01/instances.jsonl` — held-in instance ids and sizes.
- `experiment_kimi/POST_MORTEM.md` §8, §9.2.

---

## Planning Contract

### Key Technical Decisions

- KTD1. **Starting incumbent becomes `[loop] initial_harness`, validated against `HARNESSES`, default `"H0"`.** The key lives in `[loop]` because it changes what the loop does, and `loop` is already in `IDENTITY_SECTIONS`, which satisfies R2 without a new identity rule. Consequence: `identity_hash` serializes `LoopConfig` with `asdict`, so adding the field changes the hash of every existing config, and `experiment_kimi` / any other existing out-dir will refuse to resume. Accepted: `experiment_kimi` is complete and no other run is live. The config comment must say so.
- KTD2. **Use the registry's `H0*` unchanged rather than porting addendum text into `H0`'s `S2`/`S3`.** `H0*` is byte-pinned to the shipped RLM prompt by `tests/test_harness_surfaces.py`, and `H0` is pinned to contain none of the star clauses; porting text would break both and erase the H0-vs-H0* comparison the evaluation grid (`shrlm/experiment/evaluation.py`, `CONDITION_H0_STAR`) depends on. Trade-off accepted: under `H0*`, `S2`–`S5` start empty and the addendum itself stays proposer-immutable (it is appended at runtime by the `orchestrator` flag, which the candidate gate freezes).
- KTD3. **The live recursion test lives under `tests/experiment/` and reuses the existing live gate.** It follows `tests/experiment/test_smoke_live.py`: skipped unless `live_skip_reason(runner_backend="azure_foundry")` returns `None` (Azure credentials, `SHRLM_RUN_LIVE`, pricing attestation), built through `build_harnessed_rlm` / `execute_run` so it exercises the same path the experiment uses, and bounded by `max_budget` and `max_timeout` per run.
- KTD4. **Instance selection is fixed by id, not sampled.** The held-in split is content-addressed (`<problem_type>-<sha256(prompt)[:16]>`), so the four ~110k-char held-in instances can be named directly and loaded through `load_graphwalks` with the pinned revision from `configs/experiment_kimiK25.toml`. Candidates, largest first: `bfs-a1c463dbfd91f84c` (gold 181), `bfs-e1b3ed79dbe72c0f` (gold 72), `parents-7cfd78c8d93d5bfc` (gold 7), `bfs-2b1afb378791d043` (gold 1). These are the instances where the addendum's stated ~100K-characters-per-prompt ceiling is closest to binding.
- KTD5. **The recursion assertion is "≥ 1 sub-call on at least one selected instance", and a zero result is a reported finding, not a test to loosen.** The addendum tells the model to read directly when a regex over `context` pins the answer, so a faithful model may still solve a 110k-char graph in-REPL. The test asserts the disjunction across instances, prints every count, and the plan's follow-up (S6 per-prompt cap) is the next lever if it fails.
- KTD6. **Sub-calls are counted from the trace, two ways that must agree.** `run_metrics(completion)["sub_call_count"]` (per-turn trace metrics) and `walk(completion)` → `TreeStats.n_rlm_children + n_llm_leaves` (reconstructed tree) both count sub-calls; the test reads both so a discrepancy surfaces as a test failure rather than a silent accounting gap.

### Assumptions

- `H0*`'s empty `S2`–`S5` do not trip any runner check; `check_harness` already accepts `H0_STAR` in the existing surface tests.
- The Azure deployment is unchanged from the completed experiment (`AZURE_API_KEY`, `AZURE_FOUNDRY_ENDPOINT`, `SHRLM_VERIFIED_PRICING=0.6/3.0`).

### Sequencing

U1 → U2 → U3 → U4. U3 and U4 depend on U1 only; U2 depends on U1.

---

## Implementation Units

### U1. `[loop] initial_harness` config key and orchestrator wiring

**Goal:** The experiment starts from the harness the config names.

**Requirements:** R1, R2, R3, R4 (KTD1, KTD2)

**Dependencies:** none

**Files:**
- `shrlm/experiment/config.py` — `LoopConfig` gains `initial_harness: str`; the section constructor treats it as optional with default `"H0"`; validation rejects a value not in `HARNESSES`.
- `shrlm/experiment/orchestrator.py` — replace the `INITIAL_INCUMBENT` constant lookup in `run` with `HARNESSES[self.config.loop.initial_harness]`; keep the constant only if something else imports it, otherwise delete it.
- `configs/experiment.toml` — document the key under `[loop]` with the identity consequence from KTD1 (default `"H0"`).
- `configs/experiment_kimiK25.toml` — set `initial_harness = "H0*"`.
- `tests/experiment/test_config.py` — config-level scenarios below.
- `tests/experiment/test_orchestrator.py` — the incumbent-selection scenario below.

**Approach:**
1. Add the field with a default so every existing TOML loads unchanged; the smoke profile's overridable-key set does not include it.
2. Validate against the registry at load time so a typo fails before any spend, in the same style as the existing profile check.
3. Import `HARNESSES` from `shrlm/rlm_harness.py` into `config.py` for the check; `rlm_harness.py` imports nothing from `shrlm.experiment`, so there is no cycle.
4. Update the identity docstring in `config.py` to name the new key.

**Patterns to follow:** `_reject_unknown` / optional-key handling in `shrlm/experiment/config.py`; the profile check in `load_config`; comment style in `configs/experiment_kimiK25.toml`.

**Test scenarios:**
- Loading a config with no `initial_harness` yields `"H0"` and the orchestrator picks `HARNESSES["H0"]`.
- Loading `configs/experiment_kimiK25.toml` yields `"H0*"`.
- `initial_harness = "H9"` raises at load with a message naming the valid registry keys.
- `identity_hash` differs between a config with `"H0"` and the same config with `"H0*"`.
- A config with `initial_harness` omitted and one with `initial_harness = "H0"` written explicitly hash identically.
- The orchestrator, given a config naming `"H0*"`, starts round 1 with an incumbent whose `harness_hash` equals `harness_hash(H0_STAR)`; the existing `tests/experiment/test_smoke_mock.py` scaffold shows how to drive a round offline.
- All of `tests/test_harness_surfaces.py` still passes (R4).

**Verification:** The offline experiment and config suites pass; `uv run python -c` style introspection is not needed — the config test pins the behaviour.

### U2. Offline proof that `H0*` reaches the model and the loop accepts it

**Goal:** Without spend, show the addendum is in the effective prompt and the optimization loop can operate on an `H0*` incumbent.

**Requirements:** R5, R6 (KTD2)

**Dependencies:** U1

**Files:**
- `tests/experiment/test_initial_harness.py` (new) — prompt-assembly and gate scenarios.
- `tests/optimization/test_candidates.py` — reuse its candidate-materialization fixtures; no changes to the module itself.

**Approach:**
1. Build a `HarnessedRLM` for `H0_STAR` through `build_harnessed_rlm` with the mock backend the existing runner tests use, and read `effective_system_prompt`.
2. Materialize a candidate that edits `S2` on an `H0*` base and run it through the candidate gate; materialize one that sets `orchestrator: false` and confirm the `GATE_SURFACE_DIFF` rejection text.

**Patterns to follow:** `system_prompt_for` in `tests/test_harness_surfaces.py`; candidate materialization fixtures in `tests/optimization/test_candidates.py`.

**Test scenarios:**
- Effective prompt for `H0_STAR` contains `ORCHESTRATOR_ADDENDUM` and every `STAR_CLAUSES` entry; for `H0` it contains none of them.
- A one-surface `S2` candidate whose `base_harness_hash` is `harness_hash(H0_STAR)` passes the gate and its `changed_surfaces` is exactly `{"S2"}`.
- A candidate identical to `H0_STAR` except `orchestrator: false` is rejected with the reason naming the orchestrator scalar.
- A proposer pattern block rendered against an `H0*` incumbent shows `S2`'s current value as empty text rather than raising.

**Verification:** The new test module passes offline in the standard suite.

### U3. Fixed-instance fixture for the recursion test

**Goal:** A deterministic, small set of GraphWalks instances the live test runs against, loaded from the pinned dataset revision.

**Requirements:** R7 (KTD4)

**Dependencies:** U1

**Files:**
- `tests/experiment/recursion_instances.py` (new) — the instance-id list and a loader that filters `load_graphwalks` output to those ids.
- `tests/experiment/test_initial_harness.py` — offline scenario that the loader is deterministic when `fetch_rows` is monkeypatched with a fixture parquet, mirroring `tests/experiment/test_smoke_mock.py`'s `refuse_fetch` pattern.

**Approach:**
1. Hard-code the four ids from KTD4 with a comment giving prompt size and gold-set size for each.
2. Load with the `dataset_revision` and `max_chars` from `configs/experiment_kimiK25.toml` so the ids resolve to the same prompts the experiment saw; fail loudly if any id is missing.

**Patterns to follow:** `row_to_instance` id derivation in `shrlm/environments/graphwalks.py`; fixture-instance helpers in `tests/experiment/test_smoke_mock.py`.

**Test scenarios:**
- With a fixture parquet containing the four rows, the loader returns exactly four instances in id order.
- A missing id raises with the missing id in the message.
- Each returned instance's `id` recomputes from its `prompt` sha (guards against a revision drift silently changing the prompt).

**Verification:** Offline scenarios pass; the live loader path is exercised by U4.

### U4. Live recursion test

**Goal:** Prove, at bounded cost, that Kimi-K2.5 under `H0*` issues sub-calls on the selected instances.

**Requirements:** R7, R8 (KTD3, KTD5, KTD6)

**Dependencies:** U1, U3

**Files:**
- `tests/experiment/test_recursion_live.py` (new).

**Approach:**
1. Gate with `live_skip_reason(runner_backend="azure_foundry")` from `shrlm/experiment/live_gates.py`; passing the backend explicitly avoids `tests/experiment/test_smoke_live.py`'s detour through the smoke profile.
2. For each instance, build `H0_STAR` through `build_harnessed_rlm` with the azure backend kwargs the experiment uses (`sampling_args`, pricing from config), `max_iterations` from `[caps]`, and explicit `max_budget` / `max_timeout` at or below the smoke live constants in `examples/experiment_smoke.py`.
3. Run through `execute_run` with the GraphWalks verifier so cost accounting and the lower-bound flag behave as in the experiment.
4. Compute sub-calls both ways (KTD6) and print a one-line table: instance id, `sub_call_count`, `n_rlm_children`, `n_llm_leaves`, verifier cause, cost, iterations.
5. Assert the two counts agree per instance and that the sum across instances is ≥ 1.

**Execution note:** Run once with credentials and paste the printed table into the PR description; the number is the deliverable even when the assertion passes.

**Patterns to follow:** `tests/experiment/test_smoke_live.py` gating and budget reserve constants; `tests/test_e2e_depth.py` for the shape of a real-LLM test.

**Test scenarios:**
- Gate closed (no `LIVE_FLAG`, or missing Azure credentials, or pricing attestation mismatch): the test skips with the gate's reason.
- Gate open: each selected instance completes within `max_timeout` and under `max_budget`; a `resource_terminated` run is reported, not hidden.
- `run_metrics` `sub_call_count` equals `n_rlm_children + n_llm_leaves` from `walk` for every instance.
- At least one instance has a sub-call count ≥ 1 (the plan's stop condition if it fails).
- The printed table includes every selected instance even when an earlier one fails.

**Verification:** Skips cleanly in CI; with credentials, prints the table and passes or fails on the disjunction. Worst-case spend is `4 × max_budget`.

---

## Verification Contract

| Check | Command / gate | Applies to |
|---|---|---|
| Offline suite | `uv run pytest -q tests/experiment tests/optimization tests/test_harness_surfaces.py` | U1, U2, U3 |
| Lint | `uv run ruff check shrlm tests configs` and `uv run ruff format --check` | all |
| Identity pin | config test asserting `identity_hash` changed and the new value | U1 |
| Live recursion | `SHRLM_RUN_LIVE=1` with Azure credentials, `uv run pytest -q tests/experiment/test_recursion_live.py -s` | U4 |
| Pre-existing failures | The 11 failures present on `main` before this work — 5 in `tests/experiment/test_report.py`, 2 in `tests/experiment/test_smoke_mock.py::TestLiveSmokeGuards`, 4 in `tests/optimization/test_loader_gated_fixtures.py` — stay at 11; no new failures | all |

## Definition of Done

- `configs/experiment_kimiK25.toml` names `H0*`; loading any other shipped config still resolves `H0`.
- `tests/test_harness_surfaces.py` unchanged and green.
- New offline tests green; live test skips without the gate and, when run once with credentials, prints the per-instance table.
- The plan's stop condition is honoured: a zero-sub-call result is written up as a finding under `experiment_kimi/POST_MORTEM.md` follow-ups, not patched around.
- No experimental or dead-end code left in the diff; `INITIAL_INCUMBENT` removed if unused.
