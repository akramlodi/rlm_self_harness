---
title: Parallel Validation Runs - Plan
type: perf
date: 2026-08-24
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Parallel Validation Runs - Plan

## Goal Capsule

- **Objective:** A validation subject's runs stop being a serial queue, so one subject finishes in a fraction of the ~6 hours it takes today, and a run that hits a resource limit reports what it actually cost instead of losing it.
- **Means:** Publish the runtime's own usage total before the object holding it is destroyed (KTD1), then execute a subject's runs in bounded, concurrent child processes with the parent owning every shared file (KTD4, KTD6).
- **Authority hierarchy:** Experimental validity first — measured cost, pass rates, and promotion decisions must not become artifacts of how many runs happened to be in flight. Resume safety and the spend guarantee rank above wall clock. Speed never buys a weaker guarantee.
- **Stop conditions:** (a) the U8 measurement shows pass rate or mean cost moving with the worker count beyond sampling noise — stop before widening it further; (b) any design that requires a child process to append to `runs.jsonl`; (c) any change that would let the breaker charge a terminated run less than the spend the runtime actually recorded for it.
- **Execution profile:** Three ordered phases. The accounting correction lands and is verified before any fan-out code exists; the three sequential prerequisites (U3, U4, U10) land before the concurrent dispatcher; live measurement chooses the worker count.

---

## Product Contract

### Summary

Fan out the individual runs inside one validation subject across bounded child processes, after first correcting — in the runtime, at the source — how a resource-terminated run's usage is recorded. Only the validation stage opts in; mining, attribution, proposal, and the final evaluation grid keep today's strictly sequential path.

### Problem Frame

One validation subject is 256 runs (24 held-in and 40 held-out instances, 4 repetitions each). Measured on the live qwen experiment, a subject takes **5.9 hours** at a mean of 83 s per run, with a median of 44 s — the mean is twice the median, so a long tail dominates. Subject-level parallelism shipped in the predecessor plan and cut a round from ~29 h to ~6 h, but it only helps when a round produces several candidates: round 1 of both live experiments produced exactly one candidate, so two subjects ran and four of the five configured worker slots sat idle. The remaining time is entirely inside a subject, where `run_round` executes one `(instance, attempt)` pair at a time.

A second problem must be fixed first, because concurrency multiplies it. When a run hits a resource limit, the runtime discards the usage it accumulated. `RLM._spawn_completion_context` builds an `LMHandler` as a local, and that handler is the only object holding the run's true total; it goes out of scope as the exception propagates. `_partial_completion` therefore synthesizes an empty usage summary, salvaging a cost figure only from `BudgetExceededError.spent`. Every other termination persists `cost: null`, and `breaker_run_cost` then prices it at the `max_budget` ceiling.

The loss is large and it is not recoverable after the fact. In the live qwen experiment, run `bfs-84c18f2383c7c0c3__a04` really cost **$0.866138** (the exception's own figure) while the per-turn costs in its persisted trajectory sum to **$0.0158** — 1.8%. The reason is an ordering detail: `_check_iteration_limits` runs before the turn is priced and logged, so the terminating iteration — here an `llm_query` fan-out worth $0.850 — never reaches the trajectory at all. Any approach that estimates a terminated run's cost from what was logged is estimating from the wrong side of that boundary. The handler knew the real figure; nothing published it.

### Key Decisions

- **The accounting is corrected upstream in the runtime, not estimated downstream.** (session-settled: user-directed — chosen over reconstructing cost from the persisted trajectory, and over leaving the accounting alone: the trajectory is missing exactly the terminating turn where the money is, measured at 1.8% of true cost on the one run with ground truth.) Governs R1, R2, R3.
- **The accounting correction lands before any fan-out code.** (session-settled: user-directed — chosen over shipping fan-out first: concurrency raises the termination rate, so fanning out first would mean measuring the new path through a known-broken meter.) Governs R1, R2.
- **Fan-out is confined to the validation stage.** (session-settled: user-directed — chosen over fanning out every stage: the final evaluation grid is the experiment's published measurement, and concurrency-induced termination would corrupt the numbers being reported.) Governs R10, R11.
- **Bounded breaker overshoot from in-flight runs is acceptable.** (session-settled: user-directed — chosen over strictly serialized charging: serialized charging is the thing being removed.) Governs R8, R9.
- **The evaluation grid stays sequential.** (session-settled: user-directed — chosen over opting it in for speed: it is the measurement, and it is cheap to opt in later once the validity question is settled.) Governs R11.

### Requirements

**Termination accounting (lands first)**
- R1. A run terminated by any resource limit persists the usage the runtime actually recorded for it — calls, tokens, and cost — rather than an empty summary.
- R2. That figure is captured before the object holding it is destroyed, and covers every termination path, including a limit raised inside a client rather than by the iteration checks.
- R3. Spend the runtime never recorded locally keeps the run flagged as a lower bound, and a run with no recorded cost at all keeps today's ceiling pricing.
- R4. Populating usage on a terminated run does not change how downstream trace analysis classifies error nodes.
- R5. Each split's aggregate records the effective run-worker concurrency the split executed under, and how many of its runs were resource-terminated.

**Run-level fan-out**
- R6. A validation subject executes up to `validation_run_workers` of its runs concurrently, each in its own child process; `1` is the unchanged sequential path.
- R7. Dispatch order stays instance-major, attempt-minor, and a subject that stops early stops on a contiguous tail — the same shape a sequential run produces.
- R8. A subject that skipped any run never reports `completed`, at the level the promotion rule actually reads.
- R9. Dispatch stops before the breaker can exceed its budget by more than the worst observed single-run charge times the worker count.
- R10. The capability is opt-in per round; mining and the final evaluation grid pass the default and are byte-unaffected.
- R11. The merged re-evaluation uses the same dispatcher as every other subject.

**Safety and correctness**
- R12. No child process writes to `runs.jsonl`, `harness.json`, `instances.jsonl`, or a shared surface module; the parent owns every file shared within a split.
- R13. A run child writes its trace atomically, and the parent hashes it only after the child has exited.
- R14. Each run carries a wall-clock bound that survives the process boundary, and a child that ignores it is killed by the parent.
- R15. Every run that could have spent money is charged exactly once — including a child that crashed after spending, and a trace orphaned by a parent that died before reaping it.
- R16. A terminated or crashed run is attributed to the run that actually failed, never to another run.
- R17. A second invocation cannot dispatch runs into a split directory that already has live workers.
- R18. An interrupt terminates every descendant process, and a child whose parent dies exits on its own.
- R19. Persisted aggregates, the promotion ledger, and recomputed sub-call counts remain identical between the sequential and concurrent paths for any subject that does not stop early.

### Success Criteria

- One validation subject at the worker count U8 recommends completes in well under the 5.9 h sequential baseline, measured on comparable instances.
- At that worker count, pass rate and mean cost over a fixed instance set, paired by instance against worker count 1, stay inside the preregistered detectable effect from U8 — the evidence stop condition (a) needs in order to fire.
- The breaker's recorded `spent` for a subject is within a few percent of the provider-reported spend over that subject's wall-clock window, rather than dominated by ceiling-priced terminations.

### Scope Boundaries

- Mining, attribution, and proposal keep the sequential path.
- The final evaluation grid keeps the sequential path.
- The preregistered promotion rule, its thresholds, and the quantity the cost band compares are unchanged. Because the band is a ratio, giving terminated runs a real cost makes it sensitive to how much the two arms differ in termination count — recomputed on the live round, the candidate-to-baseline ratio moves 13.5% from five terminated runs out of 390, against a band edge at 1.25. The accounting correction does change that quantity's *value* — a terminated run gains a real cost where it previously contributed none — so a baseline and the candidates scored against it must always be priced under the same accounting.
- No client-side rate limiter; the worker count is the only concurrency control.

#### Deferred to Follow-Up Work

- **Recursive sub-call spend, which was missing from both the budget check and the reported cost.** *(Implemented here rather than deferred -- user-directed, after the measurement below.)* A child RLM builds its own handler and clients, so its spend never reached the parent's aggregate: `_cumulative_cost` was assigned from that aggregate on every budget check, clobbering the sub-call increments, and the max-depth branch returned before reaching them. Measured across the 86 live validation runs that use recursive sub-calls: **$0.914 of sub-call spend sat outside $1.529 recorded -- a median of 45% of a run's true cost, up to 96%.** The hidden fraction scales with how much a harness decomposes, so the cost band under-measured exactly the candidates this experiment exists to promote. It changes both what is reported and when runs terminate, so the fresh-out-dir requirement below covers it too, and it is stamped under the same accounting version as U1.
- A shared cross-process concurrency budget spanning subject workers and run workers.
- Opting the evaluation grid in, once the concurrency-versus-measurement question is settled.
- Recording the terminating turn in the trajectory by moving the limit check after the turn is logged — it would fix the per-turn view that trace analysis reads, at the cost of changing trajectory semantics for every run.

### Risks & Dependencies

- **The correction makes the breaker trip later.** Most terminated runs currently priced at the $0.50 ceiling will carry a real, smaller figure, so a candidate the breaker would have stopped may now keep running and keep spending. This is more accurate and financially less conservative at the same time. Re-derive `candidate_budget` before enabling it on a paid run.
- **Results before and after are not comparable, in every stage.** U1 lands in the shared runtime and the shared driver, so mining rounds and the final evaluation grid route through it exactly as validation does. A terminated run's cost changes from absent to real, which moves `total_cost` and `mean_cost` for any split, mining round, or evaluation set containing one. Since `summary.json` is written non-clobbering and refuses a diverging rewrite, a resumed directory would also mix accounting rules across subjects. **A fresh out-dir is required** — deleting affected summaries is not sufficient, because already-persisted runs keep their old manifest figures and would be compared against newly-priced ones.
- **Provider rate limiting is the real ceiling on concurrency.** The live experiments already log rate-limit retries at three concurrent runs. Total in-flight requests scale as `validation_workers × validation_run_workers × max_concurrent_subcalls`. The mitigation is U8's measurement and a conservative default, not a new mechanism.
- **Concurrency is a confound on measured cost.** A subject evaluated while siblings compete for one API key is not under identical conditions to one that ran alone. R5 makes the effective concurrency visible in the aggregate so the confound is auditable; U8 tests whether it is material.
- **`rlm/` is a fork with an exported patch file.** `patches/0001-harness-seams.patch` is regenerated by hand and is already stale (949 lines committed against 1997 in a fresh diff). Regenerating it alongside a runtime change absorbs that pre-existing drift; say so in the commit rather than letting it look like this change's doing.
- **Stage wall-clock is summed per-run execution time**, which overstates elapsed time under any fan-out. Pre-existing at subject level; run-level fan-out multiplies it. Out of scope, named here so the reported figure is not read as elapsed time.

---

## Planning Contract

### Key Technical Decisions

- KTD1. **The runtime publishes its usage total in the completion context's `finally`, before the handler is stopped.** That block runs during exception propagation, after every completed call has been recorded and before the only object holding the total goes out of scope. It is the single point where "the run is over" is true regardless of how it ended, and it covers every termination path — including a limit raised inside a client, which never passes through the iteration checks. The driver then reads that published figure instead of synthesizing an empty summary.
- KTD2. **The exception attributes stay as they are.** Attaching usage to each limit exception would be more self-describing, but it means touching five exception classes, five raise sites, and a signature change, and it still misses the client-raised path. One published field on the instance is the smaller and more complete seam. `BudgetExceededError.spent` remains the breaker's own trip figure and keeps its current meaning.
- KTD3. **`usage_lower_bound` stays true for terminated runs.** A request killed in flight, a response with a deficient body, a route that reports no cost, and recursive child-RLM spend (the published figure is the root completion's own aggregate, which is what successful runs already report) are all genuinely unrecorded. The correction removes the large, systematic loss; it does not make the figure exact, and the flag is what says so.
- KTD4. **Children never touch `runs.jsonl`.** A validation manifest line reaches 48 KB, and `_persist_run` appends through a buffered text handle, so a single line is written as several `write()` calls and two concurrent appenders interleave; `_load_manifest` then raises on the torn line rather than skipping it. The parent appends every manifest line on reap. This also keeps the spend breaker single-threaded and preserves the documented trace-then-manifest ordering.
- KTD5. **The parent runs the verifier.** The verifier is a live callable that does not cross a process boundary; keeping verification in the parent avoids threading a factory down another level, keeps verdict construction in one place, and lets a complete trace from a child that died before reporting be adopted rather than re-paid.
- KTD6. **The parent owns every shared file in the split, including the surface module.** Round preparation, `harness.json`, `instances.jsonl`, and one materialized surface module are written once by the parent before any dispatch; a child imports that module and never writes it. Letting each child rematerialize to a shared path would reintroduce exactly the interleaving hazard KTD4 removes.
- KTD7. **Each run child arms its own SIGALRM deadline, and the parent holds an independent per-child deadline as a backstop.** The alarm binds cleanly on a child's main thread, which is what made child processes the right unit in the predecessor plan; the parent's monotonic deadline covers a child that ignores or swallows the signal.
- KTD8. **Run children are signalled individually, never by process group.** They are deliberately not session leaders, so their pid is not a process-group id and a group-directed signal would either raise and be swallowed or hit an unrelated group. The parent calls terminate, then kill after the grace period, on each child directly. The predecessor's group-directed cleanup remains correct for subject children, which *are* session leaders; in the recommended shape there is no subject worker at all, so there is no group cleanup to inherit.
- KTD9. **Dispatch is reservation-gated, sized on the worst observed per-run charge rather than on `max_budget`.** A run is not capped at `max_budget`: the runtime checks the budget only between iterations, and a budget termination is charged its exception's figure verbatim — the live data has a single run charged **$0.866 against a $0.50 cap**, 1.73×. The gate therefore reserves that observed over-cap factor per in-flight run, and the realised worst case is `in-flight × (worst per-run charge − max_budget)`, not zero.
- KTD10. **A subject that skipped any run reports `over_budget` at the subject level, not just per split.** The promotion rule reads the subject summary's outcome, which today is derived solely from whether the breaker tripped; reservation can stop dispatch while spend is still inside the budget, so that derivation would silently mark a truncated subject `completed` and let the preregistered rule score a sample that never ran.
- KTD11. **A child that crashed without a usable trace has its run persisted as terminated and charged, never re-dispatched.** Retrying would leave the crashed attempt's spend uncharged and would punch a hole in the contiguous tail. Reporting keeps both properties: every attempt that could have spent money is charged exactly once, and the skipped set stays a tail.
- KTD12. **A split-level exclusive claim guards the directory regardless of worker count.** The predecessor's pid lock is per subject and only consulted when subject-level workers are enabled; the recommended shape here bypasses it entirely. The claim uses the repo's existing exclusive-create idiom and holds the live children's pids.
- KTD13. **`persist_interrupted_run` takes the run id it is recovering.** Its current heuristic synthesizes a terminated line for the round's first pending run, which under concurrency names a run that never executed and charges it.
- KTD14. **The per-run execution body is extracted once and shared.** The child cannot call `run_round` — that would write the shared files KTD6 reserves for the parent — so without extraction it would re-implement the timing window, the limit-exception handling, and the lower-bound flag, and the two copies would drift apart silently. Extraction also means the accounting correction lands in one place.
- KTD15. **Two knobs, documented as multiplying.** A shared cross-process concurrency budget is the better long-term answer and is deferred; until then `validation_workers × validation_run_workers` is the total, the shipped profile sets subject workers back to 1, and the guidance is to prefer run-level fan-out because it yields a two-level process tree and keeps each subject's breaker charging strictly ordered.

### High-Level Technical Design

```mermaid
sequenceDiagram
    participant RT as RLM completion context
    participant D as run_round / dispatcher (parent)
    participant FS as split directory
    participant C as run_worker child (xN)

    Note over RT: Phase A — accounting
    RT->>RT: limit raised; finally publishes handler usage total (KTD1)
    D->>FS: terminated run persists real calls, tokens, cost (R1)

    Note over D: Phase B — fan-out
    D->>FS: prepare round dir, harness.json, one surface module (KTD6)
    D->>FS: claim split lock (KTD12)
    D->>D: load manifest once, verify traces once, adopt orphan traces (R15)
    loop until pending and in-flight are empty
        alt slot free and reservation allows (KTD9)
            D->>FS: write run request
            D->>C: spawn (individually signalled, KTD8)
            C->>C: import surface module, arm SIGALRM (KTD7)
            C->>FS: write trace atomically, then result document (KTD4)
        end
        C-->>D: exit
        D->>FS: read trace + result
        D->>D: verify (KTD5), append manifest line, charge breaker
    end
    D->>D: any skipped run -> over_budget at subject level (KTD10)
    D->>FS: release split lock
```

### Assumptions

- The live per-run figures (83 s mean, 5.9 h per subject) are representative enough to size the measurement; U8 measures rather than assumes the worker count.
- Publishing usage on terminated completions does not disturb trace analysis that treats an empty usage summary as its error-node discriminator; U1 pins this with a test rather than assuming it.

---

## Implementation Units

### U1. Publish the runtime's usage total at termination

- **Goal:** A run that hits a resource limit persists what it actually cost, because the runtime hands over the total it already had.
- **Requirements:** R1, R2, R3, R4.
- **Dependencies:** none.
- **Files:** `rlm/core/rlm.py`, `shrlm/optimization/driver.py` (`_partial_completion` and its module docstring), `patches/0001-harness-seams.patch`, `tests/optimization/test_driver.py`, `tests/test_depth_metadata.py`, `tests/experiment/test_smoke_live.py`.
- **Approach:**
  1. Add a per-completion usage field to the RLM instance, reset with the other per-completion state so it cannot leak across completions on a reused instance.
  2. Set it from the handler's aggregate in the completion context's `finally`, before the handler is stopped, and expose it read-only.
  3. In the driver's limit handler, pass that published summary into `_partial_completion` and persist it verbatim instead of synthesizing an empty one; keep the lower-bound flag set (KTD3).
  4. Update the module and function docstrings that currently assert the usage is unrecoverable.
  5. Stamp an accounting-version marker on every manifest line and on `summary.json`, following the repo's existing versioned-format convention, and refuse a round or subject whose lines mix versions — so a reused directory fails loudly instead of relying on the fresh-out-dir instruction being remembered.
  6. Regenerate the exported patch file, noting in the commit that it also absorbs pre-existing drift.
- **Execution note:** start from a characterization test that pins today's empty-usage behavior, so the recovered figures are visible in the diff rather than inferred.
- **Patterns to follow:** the existing additive `cost_source` handling; the repo's fail-fast convention — no defensive fallback if the field is absent.
- **Test scenarios:**
  - A run terminated by a timeout persists the calls, tokens, and cost its client recorded, with the lower-bound flag still true.
  - A run terminated by a token limit raised inside the client — which never reaches the iteration checks — also persists its recorded usage.
  - A budget termination persists usage consistent with the exception's own figure.
  - A run that made no calls before terminating persists no cost, and the breaker still prices it at the ceiling.
  - The published figure is reset between completions on a reused instance, so a second run cannot inherit the first's usage.
  - Trace analysis still classifies error nodes correctly when a terminated completion carries usage.
  - A fixture reproducing the live 6-iteration budget termination persists a figure close to the exception's, not the trajectory's 1.8%.
- **Verification:** driver, depth-metadata, and live-smoke suites pass with their terminated-run expectations updated; the regenerated patch file matches a fresh diff.

### U2. Record concurrency and termination context in the aggregate

- **Goal:** A reader of a split's aggregate can tell what conditions produced it.
- **Requirements:** R5.
- **Dependencies:** U1.
- **Files:** `shrlm/optimization/validation.py` (`split_aggregate`), `shrlm/optimization/driver.py` (the sidecar writer), `tests/optimization/test_validation.py`.
- **Approach:** the round writes a small sidecar document beside its manifest recording the effective run-worker count it executed under — written by the sequential path too, where the value is 1 — and `split_aggregate` reads it from disk. A sidecar rather than a manifest field, because stamping it per line would break the concurrent-equivalence check that manifests differ only in line order. Leave `total_cost` and `mean_cost` definitions untouched — their *values* move because U1 gives terminated runs a real cost, and that is stated in Scope Boundaries rather than hidden by a redefinition.
- **Patterns to follow:** the existing aggregate keys and their placement.
- **Test scenarios:**
  - A split executed sequentially records concurrency 1. (The higher-worker-count case is asserted in U6, which is where a value above 1 first becomes reachable.)
  - The aggregate remains recomputable from disk alone.
  - A split containing a terminated run now contributes that run's real cost to `total_cost`.
- **Verification:** validation suite passes; a summary written before U1 and one after differ in the terminated run's contribution and the new key, and in nothing else.

### U3. Attribute an interrupted run to the run that failed

- **Goal:** Recovery from a hard-deadline escape names the run that actually hung — a prerequisite for more than one run in flight.
- **Requirements:** R16.
- **Dependencies:** none.
- **Files:** `shrlm/optimization/driver.py` (`persist_interrupted_run`), `shrlm/optimization/costs.py` (`_run_slice`), `tests/optimization/test_costs.py`, `tests/optimization/test_driver.py`.
- **Approach:**
  1. Give `persist_interrupted_run` an explicit run id (KTD13), defaulting to today's first-pending behavior so the sequential path is unchanged.
  2. Replace the manifest-length-delta recovery test with an explicit check of whether the named run already persisted.
- **Execution note:** behavior-preserving for the sequential path; the tests should show identical persisted bytes before and after.
- **Test scenarios:**
  - A sequential round whose in-flight run escapes the deadline persists exactly the line it persists today.
  - Recovery targeted at a named run synthesizes that run's line, not the first pending one.
  - Recovery for a run that already persisted is a no-op rather than a duplicate line.
- **Verification:** costs and driver suites pass unchanged apart from the new cases.

### U4. Claim the split directory exclusively

- **Goal:** Two invocations cannot dispatch runs into the same split, at any worker count.
- **Requirements:** R17.
- **Dependencies:** none.
- **Files:** `shrlm/optimization/costs.py`, `tests/optimization/test_costs.py`.
- **Approach:**
  1. On entry to the governed round, claim a lock directory inside the round directory using exclusive creation, recording the claiming pid.
  2. A claim held by a live pid refuses the round with a named error; a claim whose pid is gone is reclaimed.
  3. Release on every exit path, including exceptions.
- **Patterns to follow:** the repo's existing exclusive-create idiom; the live-pid and stale-pid handling written for subject workers.
- **Test scenarios:**
  - A second call against a split claimed by this live process is refused, naming the split and pid.
  - A claim left by a dead pid is reclaimed and the round proceeds.
  - The claim is released after a normal round, a raising round, and an interrupt.
  - A sequential round at the default worker count claims and releases — the guard is not gated on concurrency — and mining and the evaluation grid still complete unchanged.
- **Verification:** costs, orchestrator, and evaluation suites pass; no lock directory survives a completed round.

### U10. Extract the per-run execution body

- **Goal:** One implementation of "execute one run and describe its outcome", callable in-process and from a child.
- **Requirements:** R19.
- **Dependencies:** U1, U2.
- **Files:** `shrlm/optimization/driver.py`, `tests/optimization/test_driver.py`.
- **Approach:** per KTD14, extract two helpers from `run_round`. The first is the round preamble — config validation, round preparation, one-time trace verification, the credential check, and the instance-major pending list — returning the prepared path, the verified entries, and the pending pairs. The second is the per-run body: the timing window, the limit-exception handling, the partial-completion path, and the lower-bound flag. The per-run helper takes an optional verifier so the sequential path keeps verification *inside* the limit handler, where a deadline landing during verification is caught rather than escaping the round; the run child omits it, because the parent verifies (KTD5). `run_round` calls both; the dispatcher and the run worker call the same ones.
- **Execution note:** pure refactor — the sequential path must persist byte-identical manifests before and after.
- **Test scenarios:**
  - Every existing driver round test passes unchanged.
  - The helper called directly produces the same completion and flags the loop produced.
  - A deadline raised during verification is still caught by the per-run handler and persists a terminated line, not an escaped round.
  - The preamble helper returns the same pending list, in the same order, that the loop derived.
- **Verification:** driver suite passes; a scripted round's manifest is byte-identical to the pre-refactor bytes.

### U5. Run worker child entry

- **Goal:** One `(instance, attempt)` pair executes in its own process and reports back without touching shared state.
- **Requirements:** R12, R13, R14, R18.
- **Dependencies:** U3, U10.
- **Files:** `shrlm/optimization/run_worker.py` (new), `shrlm/optimization/candidates.py` (split materialization into a write step and an import-and-assemble step), `tests/optimization/test_run_worker.py` (new), `tests/optimization/subject_worker_support.py` (extend the child-side scripting seam to per-run scripts).
- **Approach:**
  1. Define a versioned request naming the split directory, run id, instance, attempt, the harness envelope and the hash the parent expects, backend and kwargs, merged limits, the parent's pid, and an optional test-only client factory with per-run arguments — the seam a child needs to install a scripted client, mirroring the subject worker's.
  2. Split materialization so the write and the import-and-assemble halves are separately callable; the child calls only the second half against the module the parent already wrote (KTD6), verifies the harness hash, arms its own SIGALRM deadline, and calls U10's shared helper.
  3. It writes its trace through a temporary file and an atomic replace, then a small result document recording success or the limit exception; it never verifies, never appends to the manifest, and never writes a shared file.
  4. It runs a parent-death watchdog and exits when its parent disappears.
- **Patterns to follow:** the subject worker's request and result protocol, format tags, hash-check-or-refuse, watchdog, and its documented exception to the fail-fast rule for reporting child failures as data.
- **Test scenarios:**
  - A scripted run writes a trace matching what the sequential path persists, plus a success result.
  - A run terminated by a limit reports that exception and still leaves a trace carrying its recorded usage.
  - A tampered harness envelope is refused before the run executes and leaves no trace.
  - The trace never appears at its final path in a partial state.
  - A child whose parent disappears exits within the watchdog interval.
  - The child writes no file outside its own per-run artifacts.
- **Verification:** run-worker suite passes; a child invoked directly produces artifacts the parent can consume.

### U6. Concurrent dispatcher inside the governed round

- **Goal:** The parent keeps up to N runs in flight, and every persisted byte is written by the parent exactly as the sequential path writes it.
- **Requirements:** R6, R7, R8, R9, R12, R15, R16, R18, R19.
- **Dependencies:** U1, U2, U3, U4, U5, U10.
- **Files:** `shrlm/optimization/costs.py`, `shrlm/optimization/validation.py` (`evaluate_subject`'s outcome derivation), `shrlm/optimization/driver.py`, `tests/optimization/test_costs.py`, `tests/optimization/test_validation.py`.
- **Approach:**
  1. Keep the sequential branch untouched when the worker count is 1.
  2. Otherwise, in the parent: validate the config and require the backend credential exactly as the sequential path does, prepare the round, materialize the surface module once, verify persisted traces once, and charge existing entries.
  3. Adopt orphan traces before dispatching — a complete, well-formed per-run trace whose run id is absent from the manifest is verified, appended, and charged, so a parent that died mid-flight does not re-pay for runs already made (R15).
  4. Fill slots from the pending list in instance-major order under the reservation gate (KTD9).
  5. On each reap: read the child's result and hash the trace the child already wrote — never rewrite it — run the verifier, append the manifest line, charge the breaker, release the reservation. A child that exited without a usable trace has its run persisted as terminated under its own run id and charged, never re-dispatched (KTD11).
  6. Enforce a per-child deadline; on expiry, signal that child individually (KTD8) and synthesize its terminated line through U3's run-id-targeted recovery. The concurrent branch does not run under the sequential path's per-slice alarm — that alarm is sized for one run and would fire mid-round across a subject taking hours — so the dispatcher holds its own monotonic deadlines, and every escape into interrupted-run recovery passes an explicit run id rather than the first-pending default.
  7. Compute the skipped set by run id; report `over_budget` whenever it is non-empty, and derive the *subject* summary's outcome from the splits' skipped sets as well as the breaker (KTD10).
  8. On any escape, terminate then kill every live child before propagating.
- **Execution note:** land the dispatcher behind the default of 1 first and prove the sequential path is byte-identical, then enable concurrency in tests.
- **Patterns to follow:** the subject dispatcher's slot loop, preallocated result slots, and poll seam — but not its group-directed signalling (KTD8).
- **Test scenarios:**
  - At worker count 1 no child process is spawned and every existing governed-round test passes unchanged.
  - At worker count 3 a round persists every run exactly once, and the manifest, aggregates, and sub-call counts match a sequential round of the same scripted responses.
  - Concurrency stays bounded at the configured count.
  - A child that crashes without a trace has its run persisted as terminated and charged, and is not re-dispatched.
  - A child that crashed after writing a complete trace has it adopted rather than re-paid.
  - Orphan traces left by a killed parent are adopted and charged on the next invocation.
  - A child that ignores its deadline is killed and recorded under its own run id.
  - A breaker trip leaves a contiguous tail skipped and reports `over_budget`.
  - Reservation that stops dispatch while under budget also reports `over_budget`, and the subject summary — not just the split — says so.
  - An interrupt terminates every live child and propagates.
  - Resume after a partial fan-out pays only for runs that were never made.
- **Verification:** costs and validation suites pass at both worker counts; a scripted round compared across counts differs only in manifest line order.

### U8. Measure, then choose the default

- **Goal:** The recommended worker count and the evidence for stop condition (a) come from observation, not assumption.
- **Requirements:** Success Criteria; evidence for stop condition (a).
- **Dependencies:** U6.
- **Files:** a purpose-sized live measurement harness under `examples/`, and the recorded numbers land in U9's documentation.
- **Approach:**
  1. Measure one subject over real graphwalks instances at the shipped caps — not the existing validation smoke, which runs four toy runs per subject at toy limits and cannot exercise more than four workers.
  2. Run each worker-count level against its own fresh out-dir; a reused directory replays persisted runs from disk and would measure a no-op.
  3. Record at each level: wall clock, pass count, mean cost, resource-terminated count, and rate-limit retry incidence, over enough runs that a change off the observed ~1.4% termination base rate is detectable.
  4. Preregister the detectable effect before measuring, tied to the promotion rule rather than to "noise": a mean-cost shift of 12.5% (half the cost band) and a pass-count shift of one run, since both promotion thresholds are zero. Pair the comparison by instance so the heavy tail cancels — per-run cost has a coefficient of variation above 4, so an unpaired one-subject comparison cannot resolve anything near those thresholds. State the per-level run count that achieves the stated power, and recommend the highest level meeting it.
- **Execution note:** this unit spends real money; state the per-level run count before starting, and record the observations in the repository rather than only in a transcript.
- **Test expectation:** none — this is a measurement; its output is the recommended default plus the observations behind it.
- **Verification:** a recommended `validation_run_workers` value, with the per-level table that justifies it and the pass-rate comparison stop condition (a) needs.

### U7. Configuration and validation wiring

- **Goal:** The knob exists, reaches every path that needs it, and every other stage is provably unaffected.
- **Requirements:** R6, R10, R11.
- **Dependencies:** U6, U8.
- **Files:** `shrlm/experiment/config.py`, `configs/experiment.toml` and the sibling shipped profiles, `shrlm/optimization/driver.py` (round config field), `shrlm/optimization/validation.py`, `shrlm/optimization/subject_worker.py`, `tests/experiment/test_config.py`, `tests/optimization/test_validation.py`, `tests/optimization/test_subject_worker.py`, `tests/experiment/test_orchestrator.py`.
- **Approach:**
  1. Add `validation_run_workers` to the operational section with a default of 1 and the same positive-integer validation as its siblings; keep it out of the identity hash and add it to the smoke-overridable set.
  2. Thread it to the round config through the evaluation config, and through the subject worker's request document — a subject child rebuilds its evaluation config field by field, so a knob that is not in the request silently defaults to 1 in every child.
  3. Set the shipped `validation_workers` to 1, per KTD15's guidance, so the shipped profile does not multiply the two knobs by default.
  4. Re-derive `candidate_budget` against the post-U1 charge model and record the figures behind it. Measured on the live baseline subject, ceiling pricing supplies $2.00 of $4.19 charged, so the same work charges roughly a third less and a candidate gets correspondingly more runs before tripping. `candidate_budget` is an identity key, so this moves the identity hash deliberately — the pin below covers the knob addition alone.
  4. Apply the knob to the merged re-evaluation as well (R11); mining and the evaluation grid keep the default.
- **Patterns to follow:** how `validation_workers` was added in the predecessor change, plus the request-document threading that knob did not need.
- **Test scenarios:**
  - The shipped profiles load with the new key defaulted, and adding the knob alone leaves the identity hash unchanged; the `candidate_budget` re-derivation moves it deliberately and is recorded.
  - A non-positive value is rejected naming the key.
  - The smoke profile may override it.
  - A subject worker child dispatches at the configured run-worker count rather than 1.
  - Mining and evaluation-grid rounds receive the default regardless of the setting.
  - The merged re-evaluation receives the configured value.
  - An end-to-end mock experiment above worker count 1 completes and meters the same totals its manifests hold.
- **Verification:** config, validation, subject-worker, and orchestrator suites pass; identity hash pinned.

### U9. Documentation

- **Goal:** The shipped prose describes what the code now does, including where the predecessor's claims no longer hold.
- **Requirements:** R10.
- **Dependencies:** U7.
- **Files:** `shrlm/optimization/README.md`, `configs/experiment.toml` comments, `examples/run_experiment.py` docstring, `docs/plans/2026-08-24-0942-perf-parallel-validation-subjects-plan.md` (its deferred-work entry).
- **Approach:**
  1. Document the accounting correction and what it changes about terminated runs, including that a fresh out-dir is required.
  2. Document the run-level dispatcher, the parent-owns-shared-files rule, the split claim, and the process topology.
  3. Correct the predecessor's statement that persisted artifacts are byte-identical to the sequential path: that holds for any subject that does not stop early, and a subject that stops early stops on a contiguous tail whose boundary depends on realised costs.
  4. Record that its deferred per-run fan-out item is implemented here, and that its stop condition on breaker skip sets was lifted deliberately, with the reasoning.
  5. Document the two knobs as multiplying, with U8's measured recommendation.
- **Test expectation:** none -- documentation only.
- **Verification:** every new file, knob, and behavior named in this plan appears in the prose.

---

## Verification Contract

| Check | Command | Applies to |
|---|---|---|
| Unit suites | `uv run pytest tests/optimization/test_driver.py tests/optimization/test_costs.py tests/optimization/test_run_worker.py tests/optimization/test_validation.py tests/optimization/test_subject_worker.py tests/experiment/test_config.py tests/experiment/test_orchestrator.py tests/test_depth_metadata.py` | U1-U10 |
| Full suite | `uv run pytest` | all |
| Lint and format | `uv run ruff check . && uv run ruff format --check .` | all |
| Pre-commit | `uv run pre-commit run --all-files` | U1 (first unit under the core-library checklist) |
| Patch export | a fresh `git diff` of the runtime package matches the committed patch file, and the regenerated patch still reapplies cleanly onto the recorded base commit | U1 |
| Identity pin | adding the knob alone leaves the shipped profile's identity hash unchanged; any `candidate_budget` change is a deliberate, recorded move | U7 |
| Refactor equivalence | a scripted round persists byte-identical manifests before and after the extraction | U10 |
| Sequential equivalence | at worker count 1, a scripted round persists byte-identical manifests, aggregates, and ledger to the post-U2 baseline | U6 |
| Concurrent equivalence | at worker count 3, aggregates and ledger match the sequential round, differing only in manifest line order | U6 |
| Accounting | a fixture reproducing the live budget termination persists a figure close to the exception's, not the trajectory's 1.8% | U1 |
| Live measurement (spends money) | the U8 harness, one fresh out-dir per worker-count level | U8 |
| Speed exit criterion | one subject at U8's recommended count completes in well under the 5.9 h sequential baseline | Success Criteria |
| Validity exit criterion | instance-paired pass rate and mean cost at the recommended count stay inside U8's preregistered detectable effect | Success Criteria |

---

## Definition of Done

- Phase order held: U1 and U2 landed and verified before any fan-out code existed.
- All units landed; worker count 1 is byte-identical to the post-accounting sequential path and every pre-existing test passes.
- Concurrent equivalence and the failure, deadline, interrupt, orphan-adoption, and resume scenarios in U6 are green.
- Adding the knob left the identity hash unchanged; any `candidate_budget` re-derivation is recorded with its figures. The shipped subject-worker count is 1.
- A recommended worker count is recorded with the per-level measurements behind it, including the pass-rate comparison.
- The exported patch file matches a fresh diff of the runtime package and reapplies cleanly onto its recorded base commit.
- No abandoned dispatcher variants, dead flags, or experimental code left in the diff.
- Documentation updated, including the corrections to the predecessor plan's claims and the fresh-out-dir requirement.
