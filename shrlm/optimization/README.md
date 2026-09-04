# `shrlm` — Self-Harness for Recursive Language Models

Fork research code, kept out of the vendored `rlm/` package tree. Everything here implements
`docs/Self Harnessing RLMs.md`; where the code and that document disagree, the document is the
specification.

## The loop this belongs to

A fixed-weight model improves its own *harness* — the scaffolding around the model — instead of its
weights. The same model executes tasks and proposes changes to the scaffolding. Each round has
three stages (§3.3 of the proposal):

| Stage | What it does | Status |
|---|---|---|
| **1. Weakness Mining** | Run the current harness on short held-in instances. Record verifier outcomes and recursive traces. Score each sub-call with the environment's synthesized sub-verifier. Convert failures to structured records and cluster them by signature → an **evidence bundle** `B_t`. | **implemented** (`shrlm/optimization/`) |
| **2. Harness Proposal** | Give the model the mined patterns, behaviors to preserve, and prior edit history; get back several minimal candidate edits, each targeting one pattern on one declared surface. | not implemented |
| **3. Proposal Validation** | Evaluate candidates on held-in plus a disjoint held-out split never shown to the proposer. Promote only on no meaningful accuracy regression with sub-call/cost in a preregistered band. Merged compatible edits are re-evaluated before promotion. | `validation.py`, `subject_worker.py`, `run_worker.py` |

The ten editable surfaces declared by the harness (`shrlm/rlm_harness.py`, `SURFACES`) are
enumerated in code as `EditableSurface` ([taxonomy.py:51](optimization/taxonomy.py#L51)), keyed by
surface id: S1 repl_contract, S2 decomposition_instruction, S3 execution_instruction,
S4 verification_instruction, S5 recovery_instruction, S6 runtime_policy, S7 metadata,
S8 repl_helpers+sub_repl_helpers, S9 answer_middleware, S10 skills. S10 is the skill library —
reusable procedures available across turns: only its name/description index is rendered into
the system prompt, and the runner installs a fixed `load_skill(name)` loader (scaffold, not an
S8 entry) that returns a body on demand. `SURFACE_REACH`
([taxonomy.py:118](optimization/taxonomy.py#L118)) annotates each surface as root-only or
child-reachable: S1–S5 travel with the system prompt, which child RLMs inherit, S8 propagates
via `sub_repl_helpers`, and S10 is child-reachable on both legs (the index travels with the
prompt and the loader is installed in the child REPL too), while the S6/S7/S9 seams apply at
the root only (residual finding C7).
The evaluator, external tools, and the three §3.1 invariants are off-limits.

## Why the package is laid out this way

`shrlm` is a top-level package, not `rlm.selfharness`. `rlm/` is upstream
([alexzhang13/rlm](https://github.com/alexzhang13/rlm)) and gets rebased; research code living
inside it would collide on every merge. The dependency runs **one way**: `shrlm` imports
`rlm.core.types.RLMChatCompletion` and `rlm.clients.base_lm.BaseLM`, and nothing in `rlm/` knows
`shrlm` exists.

Stage 1 sits flat under `shrlm/optimization/`. Stages 2 and 3 become sibling modules
(`proposal.py`, `validation.py`) when written.

The whole package is **stdlib-only** — no numpy, no sklearn, no embedding model. That is a
consequence of the clustering design below, not an accident.

## Module map

Read in this order; each module depends only on the ones above it.

### `taxonomy.py` — the closed vocabulary
Every label in the system, as enums: `VerifierCause`, `FailingLevel`, `CausalStatus`,
`AgentMechanism` (16 concrete mechanisms + `OTHER`), `EditableSurface` (the ten harness surfaces
S1–S10). Plus four tables: `MECHANISM_DOCS`, `MECHANISM_SURFACE` (mechanism → the one surface that
could fix it, [:339](optimization/taxonomy.py#L339); every one of the ten surfaces is reachable —
S10 through `unconsulted_procedure`, defined against the digest's `available_skills` /
`loaded_skills` lines and claimed only when neither the S6 budget-exhaustion nor the S3
depth-degradation mechanism independently explains the terminal failure), `SURFACE_REACH`, and
`CAUSAL_WEIGHT`. `TAXONOMY_VERSION` is `3.0.0` since S10 was declared (2.0.0 before); it stamps
every bundle, and the mechanism-frequency diff refuses to compare bundles written under a
different version unless explicitly told to.

*Why closed:* the vocabulary **is** the clustering key. An open vocabulary would make every failure
its own cluster and mining would return nothing.

*Why the prompt text is generated here:* `render_taxonomy_block()`
([:423](optimization/taxonomy.py#L423)) builds the attributor's label menu — including the
ten-surface table with reach annotations — from the enums themselves, so adding a mechanism cannot
leave the prompt describing a vocabulary the validator no longer accepts.
[`test_taxonomy.py`](../tests/optimization/test_taxonomy.py) asserts every mechanism is
documented and mapped to exactly one surface, and that the surface ids agree with
`shrlm.rlm_harness.SURFACES`.

*Deliberate omission:* `FailingLevel` has no `BOTH` member. A trace where root and children both
failed does not implicate one surface, so it is `UNDETERMINED` — an honest non-answer rather than a
label that would pollute a cluster with an unfixable case.

### `types.py` — the data model
`CallNode` / `IterationNode` / `CodeBlockNode` (the tree), `TreeStats`, `Verdict`, the `Verifier`
and `SubVerifier` protocols, `FailureSignature`, `FailureRecord` (= the proposal's `rᵢ`),
`FailurePattern` (= a cluster), `MiningConfig`, `IntegrityReport`, `EvidenceBundle` (= `B_t`).

*The load-bearing decision* is at [types.py:236](optimization/types.py#L236): `FailureSignature`
holds only the four enum values. All free text lives on a separate `AttributionDetail`, so two
records that describe the same failure in different words still cluster together. `key()`
([:250](optimization/types.py#L250)) is the 4-tuple of strings the clustering groups on.

### `walker.py` — trajectory → call tree
Turns the runtime's nested logging dicts into an addressable tree, and reports what the log could
not tell us. Three known defects in the logged format are handled explicitly rather than assumed
away ([walker.py:11-21](optimization/walker.py#L11-L21)):

- `rlm_calls` commingles plain `llm_query` calls with recursive `rlm_query` children; the runtime's
  discriminator (presence of `metadata`) has known false negatives → `TraceIntegrity` marks
  indeterminate nodes.
- Depth is not recorded anywhere (`run_metadata.max_depth` is a constant copied into every child),
  so depth is derived from nesting.
- A sub-call that raises is swallowed into a bare string and appends nothing to `rlm_calls`, so the
  node vanishes. It cannot be recovered, but `count_lost_subcalls`
  ([:264](optimization/walker.py#L264)) counts the evidence it left behind.

`TreeStats` carries `collapse_ratio`, `terminated_by_fallback`, `suspected_lost_subcalls`,
`block_attribution_reliable`, `n_indeterminate` — the mechanical facts every later stage reasons
over instead of re-reading the trace.

### `grounding.py` — the ablation switch
`derive_failing_level` ([:37](optimization/grounding.py#L37)): any wrong child ⇒ `CHILD`; all
children correct ⇒ `ROOT`; no descendants ⇒ `NO_RECURSION`; nothing checkable ⇒ `UNDETERMINED`.

*Why it is its own module:* this is the distinction the whole project rests on. A **root** failure
means correct sub-results were aggregated into a wrong answer; a **child** failure means a sub-call
returned a wrong local result the root faithfully combined. The two implicate different editable
surfaces. With a `SubVerifier` this is a checkable fact; without one it is a model's opinion, and
`FailureRecord.level_grounded` records which. Nothing else in the pipeline varies with that choice
— otherwise the Appendix-B ablation would be comparing two things at once
([test_grounding.py](../tests/optimization/test_grounding.py)).

### `digest.py` — bounded, deterministic trace view
An RLM trace can exceed a context window — that is the premise of the paradigm — so the attributor
cannot see the raw trajectory. Summarizing it with a model would insert a second uncontrolled
sampling step upstream of every mined pattern, so the compression is **entirely mechanical**:
header + root skeleton (50% of budget) + sub-call table + focused excerpts (50%), with the wide-tree
case degrading to a per-depth aggregate above 40 sub-calls.

Truncation is always announced in the text, and `TraceDigest.coverage` records how much survived —
the truncation policy is a hidden hyperparameter of every attribution and belongs in the results.
`test_digest.py` asserts a digest never leaks REPL locals.

### `attribution.py` — LLM labeling, validated
The model supplies **two of the four** signature components: `causal_status` and `agent_mechanism`.
`verifier_cause` comes from the verifier and `failing_level` from grounding. That split is what
makes an attribution more than a model's reading of a trace.

Output is validated against the enums, never parsed leniently
([validate, :234](optimization/attribution.py#L234)) — an off-vocabulary label would silently create
a singleton cluster, and cited `evidence_node_ids` must exist in the tree. A rejected response is
re-asked with the violation named (3 attempts); one that never validates is recorded as
**unattributed**, not coerced to `OTHER`.

`AttributionCache` ([:104](optimization/attribution.py#L104)) is a JSONL-backed replay cache keyed by
`(digest sha, prompt version, taxonomy version, grounded, attempt)`. Temperature zero against a
hosted API is not determinism; the cache is what makes a round re-runnable.

### `clustering.py` — the mining step
```python
groups.setdefault(record.signature.key(), []).append(record)   # clustering.py:174
```
Two runs cluster together **iff all four signature strings match exactly**. This is what the
proposal means by "clustered by the verifier-grounded signature φ(rᵢ)" — a `GROUP BY` on four
columns, not vector similarity.

*Why not embeddings:* the goal is not "which traces read alike," it is "which failures admit the
same harness edit." `MECHANISM_SURFACE` guarantees each cluster names the one surface stage 2 may
touch. Two traces can embed close together while differing on `failing_level` — exactly the cases
this design must keep apart. Exact matching also makes the bundle a deterministic function of its
records, bounds the cluster count, and gives every cluster a readable identity instead of "cluster
#7."

Cluster-level evidence (`shared_symptoms`, [:99](optimization/clustering.py#L99)) is computed
arithmetically from `TreeStats` — median sub-calls/iterations/depth, >80% collapse, fallback
termination — never by re-querying a model.

Two things here go beyond what the proposal text specifies, and are free parameters that need
writing up: `DEFAULT_MIN_SUPPORT = 2` (below-support patterns are **flagged, never dropped** — a
mechanism seen once may be real) and the 4-term `ACTIONABILITY_WEIGHTS`
([:29](optimization/clustering.py#L29)), recorded in the bundle config so a change is visible in the
artifact rather than only in the code. `compute_marginals` ([:213](optimization/clustering.py#L213))
gives backoff views (by cause / level / status / mechanism / surface) for rounds too sparse to
cluster.

### `bundle.py` — `B_t` assembly and serialization
Writes `round_NN/{bundle.json, records.jsonl, attributions.jsonl}`. `compute_bundle_id` hashes
config + instance ids and **excludes the timestamp**, so an identical round is identifiable as
identical.

`assert_no_prescription` ([:106](optimization/bundle.py#L106)) lints free-text fields for phrases
like "the fix is" / "we recommend". The bundle describes weaknesses; proposing edits is stage 2's
job. Structurally no field can hold an edit; the lint catches drift in prose. Kept short on purpose
— a longer marker list would start rejecting legitimate mechanism descriptions.

### `mining.py` — the orchestrator
`WeaknessMiner.record_failure` runs verify → walk → ground → digest → attribute per run;
`mine()` clusters, computes marginals and integrity, and returns a `MiningResult`.

The miner **does not execute the harness**. It consumes runs that already happened, which keeps the
experiment driver's concerns (splits, repetitions, budgets) out of mining and makes mining testable
without a live model. It also holds the package's only `try` — one bad attribution must not abort a
round.

### `validation.py` + `subject_worker.py` — stage-3 evaluation, optionally in parallel
`validate_round` is the whole validation stage (loader → evaluation → promotion → merged
re-evaluation → ledger). `evaluate_validation_round` evaluates the baseline and every loaded
candidate; with `operational.validation_workers > 1` (`configs/experiment.toml`) it does so
concurrently, one **child process per subject**, at most that many alive at once:

```
python -m shrlm.optimization.subject_worker <round>/<subject_id>/worker_request.json
```

Subjects share nothing — directory, spend breaker, SIGALRM hard deadline, and persist-first
manifests are all per subject — so for any subject that does not stop early the persisted
artifacts (`summary.json`, `promotions.jsonl`, `decision.json`) match the sequential path's; only
the wall clock changes. (An earlier version of this note claimed byte-identity unconditionally.
That is too strong: a subject the breaker or the reservation gate stops early stops on a
contiguous *tail* whose boundary depends on realised costs, so where it stops can differ between
two runs of the same configuration.) The
worker count is identity-exempt (it may change under an existing out-dir) and worst-case spend is
unchanged (same run count, same caps); peak request rate scales with it. Child processes are the
chosen mechanism because threads would lose the hard deadline (SIGALRM binds on the main thread
only) and share the runtime logger.

Per subject the parent writes `worker_request.json` (harness envelope + expected hash, splits,
caps, backend, verifier factory dotted path), redirects the child's stdout/stderr to `worker.log`,
and reads back `worker_result.json` plus the persisted `summary.json`. The child refuses a harness
that does not rematerialize to the expected hash. A crashed subject never aborts its siblings:
the parent waits for every child, then raises `SubjectWorkerError` naming each failed subject and
its log; re-running the same command resumes only the missing runs. Each subject directory also
carries `worker.pid` while its child is alive: a resume that finds a live pid refuses to spawn
(`SubjectWorkerBusyError`) rather than pay for the same runs twice, and every child exits on its
own when its parent disappears (a SIGKILLed parent cannot terminate anyone). The caps gate runs in the
parent, so a rejected candidate never gets a child, and the merged re-evaluation stays sequential
in-process. Tests script the children through the request's test-only `client_factory` seam
(`tests/optimization/subject_worker_support.py`).

### `run_worker.py` — one run per child process, inside a subject

The optimization loop also reuses this dispatcher for held-in mining when
`operational.mining_run_workers > 1`; `1` keeps mining in the parent process. Mining children write
only `mining/round_NN/run_workers/<run_id>/{request.json,run.log,result.json}` and their individual
trace. The parent alone verifies results, appends `runs.jsonl`, and charges the round breaker. If
the reservation gate leaves an under-budget tail, the loop raises `MiningDispatchStoppedError`:
all completed work is resumable, and the remedy is to re-run the same output directory with a
lower `mining_run_workers` value. It never starts a hidden sequential fallback. Actual cumulative
overspend remains `MiningBudgetExceededError`, even when every dispatched run completed.
All shipped experiment configs start mining at five workers. If the provider returns 429s, reduce
the setting to three and then one; one restores the sequential path.

Subject-level fan-out only helps when a round produces several candidates, and round 1 of both
live experiments produced exactly one — so two subjects ran and most worker slots sat idle. The
time that remains is all *inside* a subject: 256 runs at a mean of 83 s, about 5.9 h, executed one
at a time. `operational.validation_run_workers > 1` fans those out:

```
python -m shrlm.optimization.run_worker <round>/run_workers/<run_id>/request.json
```

**The parent owns every shared file.** It prepares the round, writes the one surface module every
child imports, verifies persisted traces once, appends every manifest line itself on reap, runs
the verifier, and charges the single breaker. A child's entire footprint is its own per-run
directory (`request.json`, `run.log`, `result.json`). This is not tidiness: a validation manifest
line reaches 48 KB and is appended through a buffered handle, so one line becomes several
`write()` calls and two concurrent appenders interleave into a torn line that `_load_manifest`
then refuses outright.

A child executes its run through the same `driver.execute_run` the sequential loop uses, so the
timing window, limit handling, and lower-bound flag cannot drift between the two paths. It
publishes its trace by atomic rename — the parent hashes those exact bytes and never rewrites
them — arms its own SIGALRM deadline, watches for its parent disappearing, and reports every
failure as data rather than an uncaught exception.

Failure handling is about paying exactly once. A complete trace whose manifest line never landed
(a parent killed mid-flight) is **adopted** — verified and recorded — rather than re-executed. A
child that left no usable trace is recorded as terminated under its own run id and charged, never
re-dispatched: retrying would leave the lost attempt's spend uncharged and punch a hole in the
contiguous tail. Overdue children are signalled **individually**, never by process group: run
children are deliberately not session leaders, so their pid is not a process-group id.

Dispatch is reservation-gated. A run is not capped at `max_budget` — the runtime checks the budget
only between iterations, and a budget termination is charged its exception's figure verbatim; the
live data has a run charged $0.866 against a $0.50 cap. The gate therefore reserves a multiple of
the cap per in-flight run, so it can stop dispatch while spend is still inside the budget. When it
does, the subject reports `over_budget` rather than `completed`: the promotion rule reads that
outcome, and a truncated sample scored as complete would be worse than a slow one.

**One dispatcher per split, enforced.** A governed round claims its round directory before it
dispatches anything, at *every* worker count — a second sequential invocation of one split does the
same damage as a concurrent one. The claim is a `.claim` directory published by a single atomic
rename, already containing its owner's pid, so it is never observable without an owner; eviction of
a dead owner is also a rename, so exactly one of several racing processes can perform it. A claim
held by a live process refuses the round by name (`SplitClaimedError`), and release only tears down
a claim this process still owns.

The claim names the *parent*, which is not sufficient on its own: a crashed parent's run children
outlive it by up to one watchdog interval. Each live child therefore records its pid beside its
request, and a new dispatcher refuses to start while any of them is still alive rather than
re-dispatching runs somebody else is still paying for.

**The two knobs multiply.** Total in-flight runs is `validation_workers × validation_run_workers`,
and provider rate limiting is the real ceiling (the live experiments already log retries at three
concurrent runs). The shipped profile keeps subject workers at 1 and prefers run-level fan-out: it
yields a two-level process tree and keeps each subject's breaker charging in a strict order.

### Cost accounting: a fresh out-dir is required

A run terminated by a resource limit used to persist an empty usage summary, so it read as free
and was priced at the per-run ceiling; recursive sub-call spend never reached the parent's total
at all (a median of 45% of a decomposing run's true cost, up to 96%). Both are now recorded at the
source. **This changes what a run costs**, which moves `total_cost`, `mean_cost`, and the
promotion rule's cost band for any split containing a terminated or decomposing run.

Figures produced before and after are therefore not comparable, and a baseline and the candidates
scored against it must always be priced the same way. Every manifest line and `summary.json`
carries an `accounting_version`; a round whose persisted runs predate the correction is refused
before any new run executes, `score_candidate` refuses two arms priced under different versions,
and an unrecognized version is refused by name. **Use a fresh out-dir** — deleting summaries is not
enough, because the old manifest lines are the problem.

## Dependency graph

```
taxonomy                     (imports only `shrlm.rlm_harness.SURFACES`)
    ↑
  types
    ↑        ↑        ↑
 walker  grounding  digest
    ↑        ↑        ↑
    └─── attribution ─┘
    ↑        ↑
    │   clustering    bundle
    └────────┴───────────┴──→ mining
```

Data flow for one failed run:

```
RLMChatCompletion
  → walk()            → CallNode tree + TreeStats
  → apply_sub_verifier() → per-child verdicts → FailingLevel   (grounded)
  → build_digest()    → bounded text + sha256 + coverage
  → attribute()       → FailureSignature + AttributionDetail   (LLM, validated)
  → FailureRecord
      ⋮ (many runs)
  → cluster_failures() → FailurePattern[]  (ranked by support, then actionability)
  → build_evidence_bundle() → EvidenceBundle → round_NN/
```
