---
title: Parallel Mining Runs - Plan
type: perf
date: 2026-09-02
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Parallel Mining Runs - Plan

## Goal Capsule

- **Objective:** Execute a round's held-in mining runs concurrently while preserving the evidence and proposal inputs produced by sequential mining.
- **Means:** Give mining its own worker-count setting, pass it to the concurrent governed-round dispatcher that validation already uses, and canonicalize the two manifest readers whose outputs currently depend on completion order.
- **Safety:** Children keep writing only their per-run artifacts. The parent remains the sole writer of the mining manifest and the sole owner of verification and spend accounting.
- **Delivery:** Four implementation units with deterministic offline tests. A small paid smoke is optional and is not part of Definition of Done.

---

## Product Contract

### Summary

Add `operational.mining_run_workers`. Values above 1 send mining runs through the existing process-based concurrent dispatcher. The default remains 1. Mining evidence and proposal inputs are read in canonical instance-and-attempt order so child completion order cannot change downstream results.

The OpenAI-compatible client already applies capped full-jitter exponential backoff independently inside each child process. This plan preserves and tests that behavior; it does not add retry logic or rate-limit accounting to the dispatcher.

### Problem Frame

Round 1 of `experiment_oolong_dsv4f` spent 13,304 seconds running 20 mining runs sequentially. The existing governed-round dispatcher already runs validation work concurrently with parent-only manifest writes, per-run child directories, deadlines, and spend accounting. Mining does not use it because `_mine_runs` never supplies `run_workers` or a child client factory.

Parallel completion changes manifest append order. `load_round` and `load_passing_behaviors` currently consume file order, so mining cannot safely reuse the dispatcher until those readers impose a stable order. Resume also needs to remain safe when an operator changes the worker count after a parent crash.

### Key Decisions

- Mining gets its own worker setting because mining and validation run in different stages and may need different provider-pressure limits.
- Manifest writes remain append-on-reap and parent-owned. Determinism is restored at the two mining read boundaries.
- A reservation-gate stop fails loudly and remains resumable. The orchestrator does not silently start a second sequential dispatch.
- Retry jitter stays in the LM client. The dispatcher adds no sleep or retry policy.

### Requirements

**Configuration**

- R1. `operational.mining_run_workers` is a positive integer. Its default is 1, it is identity-exempt, and smoke configuration may override it.
- R2. Profiles without the key continue to load with `mining_run_workers == 1` and the same identity hash.
- R3. The DeepSeek profile explicitly starts mining at 3 workers. Other profiles rely on the default unless separately tuned.

**Execution**

- R4. With `mining_run_workers > 1`, `_mine_runs` uses `run_governed_round`'s existing concurrent branch with no more than the configured number of live children.
- R5. The parent verifies each child result, appends its manifest line, and charges the shared breaker. Children never write shared round files.
- R6. With `mining_run_workers = 1`, mining keeps the existing in-process path and spawns no run child.

**Determinism**

- R7. `load_round` and `load_passing_behaviors` sort manifest entries by instance position from `instances.jsonl`, then attempt number.
- R8. On deterministic fixtures, worker counts 1 and 2 produce the same run ids, verdicts, evidence records, and proposal prompt after scrubbing clock-derived fields. Reading one persisted manifest in different line orders produces byte-identical downstream artifacts.

**Resume and failure**

- R9. Switching worker counts on an existing round is safe: complete orphan traces are adopted without another model call, and a still-live child from an earlier parent blocks the new invocation before any round artifact is rewritten.
- R10. If a parallel reservation gate leaves skipped runs while spend remains within `candidate_budget`, mining raises a resumable error with the skipped count and recommends retrying with fewer workers. It does not dispatch a sequential tail automatically.
- R11. If actual spend exceeds `candidate_budget`, mining raises `MiningBudgetExceededError` even when no run ids remain skipped. Existing crash and deadline outcomes remain persisted and charged by the shared dispatcher.

**Provider behavior**

- R12. Mining children use the configured OpenAI-compatible client's existing capped full-jitter exponential backoff for 429, 5xx, and connection failures. The dispatcher adds no retry loop.

### Success Criteria

- Offline orchestrator tests prove that mining runs with workers 2, never exceeds two live children, persists every expected run, and proceeds to evidence and proposal generation.
- Workers 1 and 2 produce equivalent outputs after scrubbing `created_at`, `execution_time`, `timestamp`, `trace_sha256`, and `wall_seconds`. Raw byte identity is required only when the same persisted manifest is reread in different line orders.
- Tests prove both budget outcomes: an under-budget reservation stop is resumable and does not launch a fallback, while actual overspend always raises the existing budget error.
- A deterministic client test proves that every transient retry requests a fresh full-jitter delay with the configured capped exponential ceiling.

### Scope Boundaries

- Attribution, proposal generation, validation, and the final evaluation grid are unchanged.
- No mining-specific dispatcher, thread pool, rate limiter, retry counter, or measurement API.
- No automatic sequential fallback after a parallel reservation stop.
- No clustering changes beyond the canonical order supplied by the two manifest readers.
- No requirement to run a paid benchmark before merging.

### Deferred Follow-Up

- Production measurement and retuning of `mining_run_workers` after observing provider behavior.
- Any change to retry attempts, backoff constants, or provider-specific rate limiting.
- Resumable per-round budget allocation.

### Risks

- Three mining workers increase peak provider request rate. The existing client jitter reduces synchronized retries, and the operational knob permits an immediate reduction without changing experiment identity.
- Moving orphan checks to the shared governed-round entry affects sequential validation and evaluation resumes. Existing governed-round and validation tests must remain green.
- Round 2 and later use generated-code incumbents. The orchestrator test must cover parallel mining with a promoted incumbent.

---

## Planning Contract

### Key Technical Decisions

- KTD1. **Reuse the existing dispatcher.** `_mine_runs` supplies `run_workers` and the mining slice of the existing dotted-path client factory to `RoundConfig`. The concurrent implementation in `run_governed_round` remains shared with validation.
- KTD2. **Keep the setting operational.** `mining_run_workers` lives beside the validation worker settings, is excluded from experiment identity, and does not flow through evaluation's `round_config_kwargs`.
- KTD3. **Canonicalize only the affected readers.** `load_round` and `load_passing_behaviors` sort by the position of `instance_id` in `instances.jsonl`, then `attempt`. The parent never rewrites the paid append-only manifest.
- KTD4. **Keep reservation stops explicit.** `_mine_runs` distinguishes actual overspend from under-budget skipped work. Actual overspend raises `MiningBudgetExceededError`; an under-budget gate stop raises `MiningDispatchStoppedError` with a lower-worker-count remedy. Neither condition depends on whether the other is present, and no second dispatch is started.
- KTD5. **Make orphan guards independent of this invocation's worker count.** After acquiring the split claim, `run_governed_round` refuses still-live children before `prepare_round` or any sidecar write. Complete orphan traces are then adopted before either execution branch can redispatch them.
- KTD6. **Keep jitter client-owned.** `rlm.clients.openai._rate_limit_backoff_seconds` remains the single retry-delay policy. Each subprocess starts a fresh interpreter and draws its own full-jitter delay; the dispatcher does not coordinate retries.

### High-Level Flow

```text
_mine_runs
  -> RoundConfig(run_workers=mining_run_workers, client_factory=mining factory)
  -> run_governed_round
       -> claim split
       -> refuse live children; adopt complete orphan traces
       -> workers == 1: existing sequential path
       -> workers > 1: existing bounded child dispatcher
  -> distinguish overspend from resumable reservation stop
  -> load_round / load_passing_behaviors in canonical order
  -> evidence and proposal generation
```

---

## Implementation Units

### U1. Add the mining worker setting

- **Goal:** Add and validate `operational.mining_run_workers` without changing experiment identity.
- **Requirements:** R1-R3.
- **Files:** `shrlm/experiment/config.py`, `configs/experiment_oolong_DeepSeekV4Flash.toml`, `tests/experiment/test_config.py`.
- **Work:**
  1. Add `mining_run_workers: int = 1` to `OperationalConfig` and validate it with `_require_positive_int`.
  2. Add the key to `SMOKE_SCALE_KEYS`, not `IDENTITY_OPERATIONAL_KEYS` or `round_config_kwargs`.
  3. Set the DeepSeek profile to 3. Leave other profiles keyless so they exercise the default.
- **Tests:** Defaulting, positive-integer validation, smoke override, identity-hash stability, and exclusion from evaluation kwargs.

### U2. Canonicalize mining manifest reads

- **Goal:** Remove completion-order dependence from evidence and proposal inputs.
- **Requirements:** R7, R8.
- **Files:** `shrlm/optimization/driver.py`, `shrlm/optimization/proposal.py`, `tests/optimization/test_driver.py`, `tests/optimization/test_proposal.py`, `tests/optimization/test_mining.py`.
- **Work:**
  1. Preserve ordered instances from `instances.jsonl` and sort manifest entries by instance position and attempt in `load_round`.
  2. Apply the same ordering in `load_passing_behaviors`.
  3. Keep existing errors for manifest entries that reference unknown instances.
- **Tests:** Reverse a multi-attempt (`loop.m = 2`) manifest and prove identical loaded order, evidence artifacts, passing-behavior rendering, and proposal prompt hash. Pin raw bytes when the same canonical manifest is reread.

### U3. Make resume guards branch-independent

- **Goal:** Prevent duplicate paid work when a crashed parallel round is resumed with a different worker count.
- **Requirements:** R9, existing dispatcher crash semantics.
- **Files:** `shrlm/optimization/costs.py`, `tests/optimization/test_run_worker.py`, `tests/optimization/test_costs.py`, `tests/optimization/test_validation_e2e.py`.
- **Work:**
  1. Check for still-live run children immediately after taking the split claim and before `prepare_round`, sidecar writes, or orphan adoption.
  2. Prepare the round and adopt complete orphan traces before choosing sequential or concurrent execution.
  3. Make the shared reservation error refer to the applicable run-worker setting rather than validation specifically.
- **Tests:** A parallel orphan resumed at workers 1 is adopted without a client call or a second charge. A live child blocks both branches before artifact mutation. Existing sequential, concurrent, and validation tests remain green.

### U4. Wire mining to the dispatcher

- **Goal:** Enable bounded mining concurrency and preserve clear budget behavior.
- **Requirements:** R4-R6, R10-R12.
- **Dependencies:** U1-U3.
- **Files:** `shrlm/experiment/orchestrator.py`, `tests/experiment/test_orchestrator.py`, `tests/clients/test_openai_transport.py`, concise worker-setting documentation in `shrlm/optimization/README.md` and relevant docstrings.
- **Work:**
  1. Pass `mining_run_workers` and the reserved mining client-factory arguments into mining's `RoundConfig`.
  2. After dispatch, check actual spend independently of skipped ids. Raise the existing budget error on overspend and a distinct resumable error on an under-budget reservation stop.
  3. Document the mining setting, child artifacts, and lower-worker-count resume remedy.
  4. Add a focused regression test for the existing full-jitter delay helper. Do not add logging, retry counting, or dispatcher sleeps.
- **Tests:**
  - Workers 2 execute through child clients with a maximum of two live children and complete the mining stage.
  - Workers 1 spawn no child and remain equivalent to current fixture output after clock scrubbing.
  - Workers 1 and 2 produce the same run ids, verdicts, evidence records, and proposal prompt hash.
  - A promoted generated-code incumbent completes mining with workers 2.
  - Under-budget skipped work raises the resumable stop without starting another dispatch; actual overspend raises `MiningBudgetExceededError`, including when every run completed.
  - Each transient retry draws a fresh value from `random.uniform(0, capped_exponential_ceiling)`.

---

## Verification Contract

| Gate | Command |
|---|---|
| Configuration | `uv run pytest tests/experiment/test_config.py` |
| Canonical readers | `uv run pytest tests/optimization/test_driver.py tests/optimization/test_proposal.py tests/optimization/test_mining.py` |
| Governed-round resume | `uv run pytest tests/optimization/test_run_worker.py tests/optimization/test_costs.py tests/optimization/test_validation_e2e.py` |
| Orchestrator and jitter | `uv run pytest tests/experiment/test_orchestrator.py tests/clients/test_openai_transport.py` |
| Full suite | `uv run pytest` |

An optional paid smoke may run the existing smoke experiment with `mining_run_workers = 2`. It is operational evidence, not a merge gate.

Fourteen suite failures predate this work on `feature/oolong`; the change must introduce no additional failures.

---

## Definition of Done

- U1-U4 are implemented.
- All targeted gates pass and the full suite adds no failures beyond the documented baseline.
- Worker-count equivalence, concurrency bounds, generated-incumbent mining, budget-stop classification, orphan adoption, and live-child refusal are covered by deterministic tests.
- Full-jitter backoff remains client-owned and has focused regression coverage.
- No measurement API, retry-log parser, automatic fallback, or unrelated clustering change is introduced.
