---
title: Parallel Validation Subjects - Plan
type: perf
date: 2026-08-24
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Parallel Validation Subjects - Plan

## Goal Capsule

- **Objective:** A full-profile validation round finishes in roughly one-fifth of today's wall clock, with every persisted artifact (manifests, summaries, ledger, decision) byte-identical to what the sequential path produces from the same scripted inputs.
- **Means:** Evaluate the baseline and each candidate in its own OS process (KTD1), driven by a config knob that defaults to sequential (KTD2).
- **Authority hierarchy:** Experimental reproducibility and resume safety (persist-first, R3 identity, R12 breakers) decide judgement calls; speed never trades against them.
- **Stop conditions:** (a) any change that would alter which runs a tripped breaker skips within one subject; (b) any change to `IDENTITY_SECTIONS` or `IDENTITY_OPERATIONAL_KEYS`; (c) the parallel path proving unable to reproduce a sequential ledger byte-for-byte on the scripted e2e fixture.
- **Execution profile:** One unit at a time; offline scripted tests per unit; one opt-in live smoke at the end.

---

## Product Contract

### Summary

Run the validation stage's independent subjects (baseline plus k candidates) concurrently as child processes, bounded by a new `operational.validation_workers` setting. Sequential behavior is preserved when the setting is 1, and every subject keeps its own directory, spend breaker, hard deadline, and resume semantics.

### Problem Frame

One full-profile round makes about 1,328 model runs, of which 1,280 are validation runs (5 subjects × 4 repetitions × 64 instances). Measured throughput on `experiment_qwen_0823` is ~70 s per run, so validation alone is ~25 h per round while mining, attribution, and proposal together take under an hour. `evaluate_validation_round` evaluates subjects strictly one after another even though each subject owns its directory, its `CandidateSpendBreaker`, and its harness, with no shared mutable state between them. Threads are not an option: the per-run hard deadline binds `SIGALRM`, which only works on the main thread (`shrlm/optimization/costs.py`, `_alarm_available`), and the shared `harnessed` RLM/logger inside `run_round` is not thread-safe.

### Key Decisions

- **Parallelize at the subject level with OS processes, not threads and not per-run.** (session-settled: user-directed — chosen over a thread pool and over per-run fan-out: threads lose the SIGALRM hard deadline and share the `harnessed` logger; per-run fan-out makes breaker skip sets order-dependent.) Governs R1, R2, R3.
- **A failing subject does not abort its siblings; the stage raises after all workers finish.** (session-settled: user-approved — chosen over fail-fast termination: persisted work from siblings is kept, so a resume only re-pays the failed subject.) Governs R6.
- **The worker count lives in the experiment TOML under `[operational]` and is excluded from the identity hash.** (session-settled: user-approved — chosen over a CLI flag or env var: it stays with the other operational knobs, and exclusion keeps existing out-dirs resumable.) Governs R4, R5.
- **Children rebuild harnesses from serialized envelopes rather than receiving pickled live objects.** (session-settled: user-approved — chosen over pickling: candidate surfaces are functions imported from generated modules and do not pickle; the envelope round-trip is the gate the loader already proves.) Governs R7.

### Requirements

**Concurrency**
- R1. `evaluate_validation_round` evaluates the baseline and every loaded candidate concurrently, at most `validation_workers` subjects at a time, each in its own child process.
- R2. Each subject's evaluation inside the child is the existing `evaluate_subject` call, unchanged in breaker semantics, split order, hard deadline, and persist-first resume.
- R3. The merged re-evaluation (`MERGED_SUBJECT_ID`) stays on the sequential in-process path.

**Configuration and identity**
- R4. A new `operational.validation_workers` integer (default 1, minimum 1) controls the worker cap; the smoke profile may override it.
- R5. `identity_hash` is unchanged for any existing config file; `validation_workers` is not an identity key.

**Failure and interruption**
- R6. When a child exits non-zero or produces no result, the stage waits for the remaining children, then raises an error naming every failed subject and its log path; already-persisted runs remain on disk.
- R7. A child refuses to run when the harness it rebuilds does not hash to the expected value.
- R8. A `KeyboardInterrupt` in the parent terminates every child; re-invoking the same command resumes each subject from its manifests.

**Equivalence and accounting**
- R9. For identical inputs, the parallel path produces the same `summary.json`, `promotions.jsonl`, and `decision.json` bytes as the sequential path, and `RoundEvaluation.candidates` keeps loader order regardless of completion order.
- R10. Stage usage in `stage_usage.jsonl` for the validation stage counts every run the children persisted, via the existing disk re-aggregation.
- R11. With `validation_workers = 1` no child process is spawned and every existing test passes unchanged.

### Success Criteria

- Full-profile validation stage wall clock drops from ~25 h to ~5-6 h at `validation_workers = 5` (measured on the next real round via `stage_usage.jsonl` `wall_seconds`).
- The scripted e2e round in `tests/optimization/test_validation_e2e.py` passes under both worker settings with identical ledger bytes.

### Scope Boundaries

- Mining runs (48 per round) stay sequential.
- Runs within one subject (across instances, attempts, or splits) stay sequential.
- Attribution and proposal stages are untouched.

#### Deferred to Follow-Up Work

- Per-run fan-out inside `run_round` (the ~30× option) — needs a process-based deadline and a breaker overshoot policy.
- Attribution fan-out — only worthwhile with a slow (thinking) attributor.
- Provider rate-limit backoff tuning once five subjects hit OpenRouter concurrently.

### Risks & Dependencies

- **Provider rate limits.** Five concurrent RLM loops multiply request rate; 429s surface as run errors persisted as failures. Mitigation: `validation_workers` is tunable down; deferred backoff work noted above.
- **Disk contention.** Each subject writes to its own directory; the only shared file is the parent's `stage_usage.jsonl`, written by the parent alone.
- **Environment leakage.** Children need the backend credential, so they inherit the parent environment rather than the gate's allowlist. The request file must never carry credentials (the driver's sensitive-kwarg scan already rejects them in `backend_kwargs`).

---

## Planning Contract

### Key Technical Decisions

- KTD1. **Child processes are spawned as `python -m shrlm.optimization.subject_worker <request.json>`, one per subject, managed with `subprocess.Popen` and a bounded slot count.** Mirrors `_run_gate_subprocess` in `shrlm/optimization/candidates.py`. No `multiprocessing` pool: no pickling, no fork-safety questions on macOS, and the child is the main thread so `SIGALRM` still binds. (session-settled: user-directed — chosen over a thread pool and over per-run fan-out: see Key Decisions.)
- KTD2. **`validation_workers` defaults to 1 and selects the unchanged in-process loop; only values above 1 enter the process dispatcher.** Keeps every existing test on the in-process `get_client` monkeypatch seam and makes the parallel path opt-in.
- KTD3. **The request file is self-contained JSON: subject id, harness envelope (`serialize_harness` output plus expected hash), split instances, caps, repetitions, backend name, backend kwargs, out dir, round index, verifier factory dotted path, and an optional test-only client factory dotted path with opaque JSON args.** The child rebuilds the harness with `materialize_harness` into a subject-local module file and refuses on hash mismatch (R7), following `rematerialize_harness_envelope`.
- KTD4. **The parent decides `CandidateRejection` before spawning.** `governed_limits` is pure, so the caps gate runs in the parent and only runnable subjects get a child; the child's result is therefore always a completed subject or a failure.
- KTD5. **The parent rebuilds `SubjectEvaluation` from disk with `load_summary`, not from child stdout.** The child's stdout carries one JSON status line; the summary the ledger consumes is the persisted `summary.json`, the same bytes the sequential path returns.
- KTD6. **Verifier and test client are passed as dotted import paths.** `EvaluationConfig` gains `verifier_factory` (dotted path to a zero-arg callable); the orchestrator supplies the `GraphWalksVerifier` path when no verifier was injected. A non-default injected verifier with `validation_workers > 1` and no factory is a configuration error raised before any spawn.
- KTD7. **Child stdout/stderr go to `<subject_dir>/worker.log`.** The failure error (R6) names the log path per failed subject.
- KTD8. **Results are placed by loader index.** A completion-order list would reorder the ledger; the dispatcher writes each result into a preallocated slot.

### High-Level Technical Design

```mermaid
sequenceDiagram
    participant O as orchestrator._validate
    participant V as validate_round / evaluate_validation_round
    participant D as SubjectDispatcher (workers>1)
    participant W as subject_worker (child)
    participant FS as subject dir on disk

    O->>V: EvaluationConfig(workers, verifier_factory)
    V->>V: load_candidates; caps gate per subject (KTD4)
    alt workers == 1
        V->>V: evaluate_subject(...) sequentially (unchanged)
    else workers > 1
        V->>D: subjects in loader order
        loop up to workers concurrent
            D->>FS: write worker_request.json
            D->>W: spawn python -m subject_worker
            W->>W: materialize_harness + hash check (KTD3)
            W->>W: evaluate_subject (breaker, SIGALRM, persist-first)
            W->>FS: summary.json, runs.jsonl, worker.log
            W-->>D: status line, exit code
        end
        D->>FS: load_summary per subject (KTD5)
        D-->>V: SubjectEvaluation | failure, by index (KTD8)
        V->>V: raise if any failed (R6)
    end
    V->>V: assess_round, plan_promotion, merge leg (sequential), ledger
    V-->>O: ValidationRound
    O->>FS: _validation_usage delta -> stage_usage.jsonl (R10)
```

### Assumptions

- `tests` is an importable package (`tests/__init__.py` exists), so test-only dotted paths such as a scripted client factory resolve inside the child.
- The child inherits the full parent environment (including `.env`-loaded keys via `rlm.clients`' `load_dotenv`).

---

## Implementation Units

### U1. Config knob and plumbing

- **Goal:** Add `operational.validation_workers` with validation, identity exemption, smoke overridability, and threading into `EvaluationConfig`.
- **Requirements:** R4, R5, R11.
- **Dependencies:** none.
- **Files:** `shrlm/experiment/config.py` (`OperationalConfig`, `evaluation_config_kwargs`, smoke-overridable set, module docstring), `configs/experiment.toml` (`[operational]` and `[smoke.operational]` comments/values), `shrlm/optimization/validation.py` (`EvaluationConfig` fields `workers`, `verifier_factory`), `tests/experiment/test_config.py`, `tests/optimization/test_validation.py`.
- **Approach:**
  1. `OperationalConfig.validation_workers: int = 1`, rejected below 1 with the same message style as `eval_repetitions`.
  2. Leave `IDENTITY_OPERATIONAL_KEYS` unchanged; add the key to the smoke-overridable set.
  3. `evaluation_config_kwargs` passes `workers`; `EvaluationConfig.__post_init__` rejects `workers < 1` and `workers > 1` without `verifier_factory`.
- **Patterns to follow:** `eval_repetitions` validation and docstring treatment in `shrlm/experiment/config.py`.
- **Test scenarios:**
  - Loading the shipped full and smoke profiles yields `validation_workers == 1` and identical `identity_hash` to before the field existed (pin the hash from the current file).
  - `validation_workers = 0` in a TOML raises a `ValueError` naming the key.
  - `[smoke.operational] validation_workers = 3` is accepted; a non-overridable key alongside it is still rejected.
  - `EvaluationConfig(workers=2)` without `verifier_factory` raises; with it, constructs.
- **Verification:** config tests pass; `identity_hash` of `configs/experiment.toml` unchanged.

### U2. Subject worker child entry

- **Goal:** A `python -m shrlm.optimization.subject_worker <request.json>` entry that rebuilds one subject and runs `evaluate_subject` in-process as the main thread.
- **Requirements:** R2, R7.
- **Dependencies:** U1.
- **Files:** `shrlm/optimization/subject_worker.py` (new), `shrlm/optimization/validation.py` (request/result schema constants, `write_subject_request` helper), `tests/optimization/test_subject_worker.py` (new), `tests/optimization/fixtures.py` (scripted client factory usable from a child).
- **Approach:**
  1. Define the request document per KTD3 and a one-line JSON result (`ok`, `subject_id`, `summary_path` or `error`).
  2. Child: load request, import `verifier_factory`, optionally install `client_factory` on `rlm.core.rlm.get_client`, `materialize_harness` into `<subject_dir>/subject_module_<hash16>.py`, compare `harness_hash` to expected, build `EvaluationConfig`, call `evaluate_subject`, print result line, exit 0; any exception prints an error line and exits 1.
  3. A `CandidateRejection` returned by `evaluate_subject` inside the child is an invariant violation (KTD4) and is reported as a failure.
- **Patterns to follow:** `run_subprocess_gate` / `__main__` block in `shrlm/optimization/candidates.py`; `rematerialize_harness_envelope` in `shrlm/experiment/orchestrator.py`.
- **Test scenarios:**
  - Happy path: request for `H0` with a scripted client factory over two instances × two splits produces `summary.json` with the scripted pass counts and exit 0.
  - Tampered envelope (edited instruction text, stale hash) exits 1 with a hash-mismatch message and creates no split directories.
  - Missing `verifier_factory` module exits 1 with an import error in the result line.
  - Resume: running the same request twice makes zero new model calls on the second run (scripted client asserts call count).
- **Verification:** worker tests pass; the generated module file lives under the subject directory.

### U3. Process dispatcher in `evaluate_validation_round`

- **Goal:** Route subjects through child processes when `workers > 1`, with bounded concurrency, loader-order results, sibling-completing failure policy, and interrupt propagation.
- **Requirements:** R1, R3, R6, R8, R9, R11.
- **Dependencies:** U2.
- **Files:** `shrlm/optimization/validation.py` (`evaluate_validation_round`, new `SubjectWorkerError`, dispatcher helper), `tests/optimization/test_validation.py`, `tests/optimization/test_validation_e2e.py`.
- **Approach:**
  1. Keep the existing sequential branch verbatim for `workers == 1`.
  2. Parallel branch: caps-gate every subject in the parent (KTD4); for runnable ones write the request, spawn with `subprocess.Popen` redirecting to `worker.log` (KTD7), at most `workers` alive; poll and refill.
  3. On child exit 0, `load_summary` → `SubjectEvaluation` into its index slot (KTD5, KTD8); on non-zero, record failure and keep going.
  4. After all exit: if any failed, raise `SubjectWorkerError` listing subject ids and log paths.
  5. `KeyboardInterrupt`: terminate then wait all children, re-raise.
  6. `validate_round` merge leg keeps calling `evaluate_subject` directly (R3).
- **Patterns to follow:** existing `evaluate_validation_round` id checks; `_run_gate_subprocess` timeout/stderr handling.
- **Test scenarios:**
  - `workers=3`, five subjects (baseline + 4) with scripted child clients: all five `summary.json` files exist, `RoundEvaluation.candidates` is in loader order, and at no point were more than three children alive (assert via a child-side lock-file counter).
  - Byte equivalence: run the e2e scripted round sequentially and with `workers=2` into two directories; `promotions.jsonl`, `decision.json`, and every `summary.json` compare equal.
  - One candidate's child exits 1 (scripted crash): the other subjects complete and persist, `SubjectWorkerError` names the failed id and its `worker.log`, and re-running with a fixed script resumes only the failed subject (zero new calls elsewhere).
  - A candidate rejected by the caps gate (S6 policy above caps) appears as `CandidateRejection` without any child spawned.
  - `workers=1` spawns nothing (monkeypatch `subprocess.Popen` to fail) and all pre-existing validation tests pass.
  - Simulated `KeyboardInterrupt` during polling terminates children (asserted via their exit status) and re-raises.
- **Verification:** validation unit and e2e suites pass under both worker settings.

### U4. Orchestrator wiring and usage accounting

- **Goal:** The experiment loop passes the worker count and verifier factory through and still meters validation usage correctly.
- **Requirements:** R10, R11.
- **Dependencies:** U3.
- **Files:** `shrlm/experiment/orchestrator.py` (`_validate`, `_Experiment` verifier-factory resolution, module docstring resume contract), `shrlm/experiment/config.py` (`evaluation_config_kwargs`), `tests/experiment/test_orchestrator.py`, `tests/experiment/test_smoke_mock.py`.
- **Approach:**
  1. `_validate` builds `EvaluationConfig` with `workers` and `verifier_factory`; the default `GraphWalksVerifier` maps to its dotted path, an injected verifier with `workers > 1` raises a configuration error before the stage meter opens.
  2. `_validation_usage` before/after delta is unchanged; confirm it captures child-persisted manifests.
- **Patterns to follow:** existing `_validate` meter block.
- **Test scenarios:**
  - Mock-backed smoke experiment with `validation_workers = 2` completes a round; `stage_usage.jsonl` validation record's `input_tokens`/`cost` equal the sum over every `runs.jsonl` under the validation round.
  - Injected non-default verifier plus `validation_workers = 2` raises before any directory is created.
  - Existing orchestrator tests pass with the default of 1.
- **Verification:** experiment test suite passes.

### U5. Documentation

- **Goal:** Record the knob, the process model, and the exposure math change.
- **Requirements:** R4.
- **Dependencies:** U4.
- **Files:** `shrlm/optimization/README.md`, `configs/experiment.toml` comments, `examples/run_experiment.py` module docstring, `shrlm/docs/` validation notes if present.
- **Approach:** Describe `validation_workers`, the `worker.log` location, the failure policy, and that worst-case spend is unchanged (same run count, same caps) while peak request rate scales with the worker count.
- **Test expectation:** none — documentation only.
- **Verification:** docs mention every new file and setting named in this plan.

---

## Verification Contract

| Check | Command | Applies to |
|---|---|---|
| Unit and e2e tests | `uv run pytest tests/optimization/test_validation.py tests/optimization/test_validation_e2e.py tests/optimization/test_subject_worker.py tests/experiment/test_config.py tests/experiment/test_orchestrator.py` | U1-U4 |
| Full suite | `uv run pytest` | all |
| Lint/format | `uv run ruff check . && uv run ruff format --check .` | all |
| Identity pin | `identity_hash(load_config("full"))` equals the value recorded before U1 | U1 |
| Live smoke (opt-in, spends money) | `uv run python examples/run_experiment.py --profile smoke --out-dir ./experiment_smoke_parallel` with `[smoke.operational] validation_workers = 3` | U4 |
| Speed exit criterion | next full round's validation `wall_seconds` in `stage_usage.jsonl` ≤ 0.25 × the sequential figure | Success Criteria |

---

## Definition of Done

- All units landed; `validation_workers = 1` path byte-identical to current behavior and all pre-existing tests green.
- Byte-equivalence test (U3) green under `workers = 2`.
- Failure, interrupt, and resume scenarios (U3) green.
- `identity_hash` of the shipped config unchanged.
- No abandoned dispatcher variants (thread pool, multiprocessing pool) left in the diff.
- Docs (U5) updated.
