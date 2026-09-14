# DeepMind MRCR v2 — Round 1 Postmortem

**Experiment:** `DeepMind_mrcrv2`  
**Round:** 1  
**Final harness:** `H0*`  
**Final harness hash:** `be733515a34beefc2b17e74111e40b19fad5b132fd8ed1b149e8b2f11283210d`  
**Experiment identity hash:** `99dbf6668b626bcc2bf4534502e4525e735aff022e4678908f7e7144af61ea95`  
**Round outcome:** No promotion  
**Analysis snapshot:** `20260913T170917Z`

---

## 1. Executive Summary

Round 1 of the `DeepMind_mrcrv2` experiment completed one optimization round and terminated at the configured `max_rounds=1` boundary. The incumbent harness, `H0*`, was retained; no candidate harness was promoted.

The optimization produced **2 candidate proposals**:

1. **`r01-c01-s6` — S6 proposal:** rejected at the `harness_check` gate before quality validation because the candidate's runtime constraint contradicted the capacity stated to the model.
2. **`r01-c02-s2` — S2 proposal:** passed into validation but was rejected on incumbent-quality criteria. Its measured held-in pass rate was **73.33%** and held-out pass rate **76.67%**, versus the incumbent `H0*` at **93.33% held-in** and **86.67% held-out**.

The mining phase successfully completed **20/20 runs**, with no unwalkable runs and a measured **0.0 collapse rate**. Three mining failures were recorded; the attribution analysis classified **100% of the observed failure share as `no_recursion_failure`**, with zero root-failure, child-failure, or ungrounded shares.

A subsequent raw-trace inspection of all 20 mining traces revealed an important behavioral characteristic of `H0*`: **none of the 20 runs made a sub-LLM or recursive RLM call**. Instead, the root model repeatedly used the Python REPL to inspect and search the full context, relying on context slicing, regular expressions, loops, and direct extraction. Across the 20 runs there were **299 root-model iterations, 3,202,192 input tokens, 49,202 output tokens, 304 detected context-slicing operations, 126 regex operations, and 200 loop operations**, with **0 subcalls**. Twelve of the 20 runs experienced at least one REPL-output truncation event. This makes the observed incumbent better characterized as a **programmatic retrieval strategy using the REPL** than as an actively recursive/sub-call-based RLM strategy for this benchmark.

The major operational problem was the validation worker for `r01-c02-s2`. It repeatedly expanded to approximately the full available physical memory and was killed by the Linux OOM killer. This occurred on progressively larger machines (approximately 8 GiB, 32 GiB, and 64 GiB configurations), with the final 64-GiB run reaching approximately **59.4 GiB RSS**. A 32-GiB swap file was subsequently added, but the final persisted validation result remained a lower-bound/partial result rather than a clean worker completion.

Therefore, the round is best characterized as **experimentally completed at the orchestrator level, but with a significant validation infrastructure limitation**. The optimization decision itself is clear for the tested candidate—`r01-c02-s2` performed worse than `H0*` on both reported quality metrics—but the validation execution should not be described as a completely clean end-to-end run.

---

## 2. Configuration

The round used:

| Parameter | Value |
|---|---:|
| `m` | 2 |
| `v` | 3 |
| `k` | 4 |
| `t` | 1 |
| `patience` | 3 |
| `environment` | `DeepMind_mrcrv2` |
| `initial_harness` | `H0*` |
| `tau_regression` | 0.0 |
| `tau_improvement` | 0.0 |
| `cost_band` | `[0.5, 1.25]` |
| `max_iterations` | 30 |
| `max_depth` | 3 |
| per-run `max_budget` | 4.0 |
| `max_timeout` | 3600 s |
| validation workers | 5 |
| validation run workers | 3 |
| evaluation repetitions | 3 |
| real-check interval | disabled (`0`) |

The experiment used verified pricing:

`$0.19 / $0.51`

for the configured model pricing.

---

## 3. Dataset and Mining Setup

The experiment used the released DeepMind MRCR v2 **8-needle 1–2M-token context bucket**.

The complete release contained:

- **310 rows**
- **3,056,474,852 bytes**
- 10 metadata columns

Two rows had parser mismatches and were excluded from the clean sampling pool:

- row 145: parser found 5 target turns instead of the expected 8
- row 154: parser found 10 target turns instead of the expected 8

Despite those mismatches, both rows located the official target answer perfectly (best-match ratio 1.0).

The resulting clean pool therefore contained **308 usable rows**, while all 310 rows remained retained in the cache/audit data.

Validation split requirements were satisfiable, including the configured long test set of 10 examples.

---

## 4. Mining Results

### 4.1 Completion

The mining stage completed all expected runs:

| Metric | Result |
|---|---:|
| Expected runs | 20 |
| Runs walked | 20 |
| Unwalkable runs | 0 |
| Failures | 3 |
| Collapse rate | 0.0 |
| Ungrounded share | 0.0 |

The analysis reported:

```text
n_runs = 20
n_walked = 20
n_unwalkable = 0
n_failures = 3
collapse_rate = 0.0
root_failure_share = 0.0
child_failure_share = 0.0
no_recursion_failure_share = 1.0
ungrounded_share = 0.0
```

### 4.2 Interpretation

There was **no measured recursive collapse** in the 20 mined runs.

All three observed failures were attributed to the `no_recursion_failure` category. The analysis therefore provides no evidence in this round that recursive execution itself was the cause of those failures.

The mining evidence was marked complete.

### 4.3 Raw Trace Behavioral Analysis

The raw JSON traces were subsequently inspected across **all 20 mining runs** to determine how the incumbent actually used the RLM environment. This analysis is important because the harness exposes recursive/sub-LLM tools, but availability of those tools does not imply that the model actually uses them.

The result was unambiguous:

| Trace behavior | Aggregate result |
|---|---:|
| Mining trace files | **20** |
| Root-model iterations | **299** |
| Root-model input tokens | **3,202,192** |
| Root-model output tokens | **49,202** |
| Sub-LLM calls (`llm_query*`) | **0** |
| Recursive RLM calls (`rlm_query*`) | **0** |
| Runs with any subcalls | **0/20** |
| Context-slicing operations detected in code | **304** |
| Regex operations detected in code | **126** |
| Loop operations detected in code | **200** |
| Runs with truncation events | **12/20** |
| Runs with an answer event | **17/20** |

The trace evidence shows that `H0*` generally solved these MRCR v2 instances by **directly manipulating and searching `context` in Python**, rather than delegating chunks of the context to sub-models. Typical operations included:

1. inspecting `len(context)` and slices such as `context[:N]` and `context[-N:]`;
2. locating example boundaries with regular expressions;
3. searching for `User:` / `Assistant:` markers;
4. searching for task-specific phrases and exact prompt patterns;
5. extracting the relevant assistant response directly from the context; and
6. setting the final answer once the target text had been located.

For example, the inspected `a01` trace used eight root-model iterations and reported `sub_call_count=0` throughout. It repeatedly sliced and searched the context, eventually locating the target story and submitting it. The all-run trace analysis confirms that this was not an isolated behavior: **0/20 runs used subcalls**.

This distinction is scientifically important. The incumbent should therefore not be characterized, based on this benchmark evidence, as a strategy that relies on recursive decomposition or sub-LLM delegation. Its observed strategy is better described as **LLM-guided programmatic retrieval over the persistent Python REPL context**. The model decides what Python code to execute, but the actual retrieval work is performed through ordinary Python operations such as slicing, regex matching, iteration, and string extraction.

There is also evidence of inefficiency in this approach. **12/20 runs experienced truncation events**, including traces in which the model attempted to print or inspect large portions of the context. Thus, although direct REPL retrieval was effective enough to produce a strong incumbent score, it sometimes spent iterations inspecting large amounts of text and triggered output truncation rather than using hierarchical/sub-call retrieval.

The trace analysis counts function names appearing in generated code separately from runtime `sub_call_count`; the latter is the stronger indicator for actual delegation. Here, both the code-level detection and the runtime metrics agree: **zero actual subcalls were observed across all 20 runs**.

---

## 5. Proposal Generation

The round generated **2 proposals**.

### Proposal 1 — `r01-c01-s6`

**Target surface:** S6  
**Surface source:** backfilled  
**Decision:** rejected at `harness_check`

The candidate attempted to impose:

```text
max_prompt_chars = 50000
```

(~50K characters).

However, the RLM prompt presented the sub-call capacity as approximately:

```text
~100K
```

The harness checker rejected the candidate because the runtime constraint contradicted the capacity statement made to the model.

Recorded rejection:

> The prompt states a sub-call capacity of `['~100K']` per prompt, but S6 enforces `max_prompt_chars=50000` (~50K). The model would be told a false fact about its own environment.

**Important:** this candidate was not rejected because of measured task performance. It was rejected for an internal environment/contract consistency violation before quality validation.

---

### Proposal 2 — `r01-c02-s2`

**Target surface:** S2  
**Surface source:** ledger  
**Decision:** rejected

This candidate passed the relevant proposal/harness pathway far enough to undergo quality validation.

Performance:

| Metric | `H0*` incumbent | `r01-c02-s2` |
|---|---:|---:|
| Held-in pass rate | **93.33%** | **73.33%** |
| Held-out pass rate | **86.67%** | **76.67%** |

Relative to `H0*`, the candidate changed:

- Held-in: **−20.00 percentage points**
- Held-out: **−10.00 percentage points**

Thus, the candidate was worse than the incumbent on **both** reported quality measures.

---

## 6. Incumbent Quality

The incumbent `H0*` remained unchanged.

Recorded incumbent quality:

| Metric | Result |
|---|---:|
| Held-in pass rate | **93.33%** |
| Held-out pass rate | **86.67%** |
| Incumbent changed | **False** |
| Round complete | **True** |
| Runs complete | `unknown` |

The incumbent-quality annotation additionally recorded the S6 candidate's upstream rejection:

```text
r01-c01-s6: rejected upstream at gate harness_check
```

because of the S6 prompt-capacity/runtime inconsistency described above.

---

## 7. Validation

### 7.1 Subjects

The validation stage contained two subject directories:

1. `baseline`
2. `r01-c02-s2`

The baseline validation completed successfully, with both `summary.json` and `worker_result.json` produced.

The `r01-c02-s2` worker did not complete cleanly.

Its final `worker_result.json` reported:

```json
{
  "error": "parent process 6808 died; worker exited",
  "format": "shrlm-subject-worker-result/v1",
  "ok": false,
  "subject_id": "r01-c02-s2"
}
```

Its `worker.log` was empty.

### 7.2 OOM failures

The same validation subject repeatedly encountered memory exhaustion.

Observed behavior across instance sizes:

| Approx. machine RAM | Observed outcome |
|---:|---|
| ~8 GiB | Python process killed around the available memory ceiling |
| ~32 GiB | Python validation worker reached roughly 25–29 GiB and was killed |
| ~64 GiB | Python validation worker reached approximately **59.4 GiB RSS** and was killed |

The final 64-GiB kernel report identified the killed process with approximately:

```text
anon-rss: 62358504 kB
```

which is approximately 59.4 GiB of resident anonymous memory.

This is notable because the corresponding `worker_request.json` was only approximately **398,436 KB (~389 MB)** on disk.

A **32-GiB swap file** was subsequently configured and made persistent through `/etc/fstab`. The machine then had approximately 61 GiB RAM plus 31 GiB usable swap.

The final resume attempt completed the orchestration path and persisted the lower-bound validation state, but did not turn the previously failed worker into a clean completed validation.

### 7.3 Validation cost and attempts

Recorded validation usage:

#### Attempt 1

- Input tokens: **7,794,494**
- Output tokens: **171,719**
- Cost: **$1.56853055**
- Wall time: **3,890.44 s**
- `lower_bound`: `true`

#### Attempt 2 / resumed validation

- Input tokens: **8,328,217**
- Output tokens: **198,037**
- Cost: **$1.68336010**
- Wall time: **4,479.93 s**
- `lower_bound`: `true`

#### Final resume

- Input tokens: **0**
- Output tokens: **0**
- Cost: **$0**
- Wall time: **758.46 s**
- `lower_bound`: `true`

The final zero-token resume therefore did not incur additional model API cost according to the recorded ledger.

---

## 8. Stage Usage and Cost

Recorded stage usage:

| Stage | Input tokens | Output tokens | Cost |
|---|---:|---:|---:|
| Mining | 2,370,256 | 39,493 | $0.47049007 |
| Mining resumes | 0 | 0 | $0 |
| Attribution | 11,532 | 311 | $0.00234969 |
| Proposal | 5,109 | 510 | $0.00123081 |
| Validation attempt 1 | 7,794,494 | 171,719 | $1.56853055 |
| Validation attempt 2 | 8,328,217 | 198,037 | $1.68336010 |
| Validation final resume | 0 | 0 | $0 |

**Total recorded API/model cost: approximately $3.73.**

The validation stage accounted for the overwhelming majority of recorded spend.

---

## 9. Surface Activity

The analysis reported:

| Surface | Attempted | Promoted |
|---|---:|---:|
| S1 | 0 | 0 |
| S2 | 1 | 0 |
| S3 | 0 | 0 |
| S4 | 0 | 0 |
| S5 | 0 | 0 |
| S6 | 1 | 0 |
| S7 | 0 | 0 |
| S8 | 0 | 0 |
| S9 | 0 | 0 |
| S10 | 0 | 0 |
| merged | 0 | 0 |

Cumulative:

- Surfaces attempted: **2**
- Surfaces promoted: **0**

The unattributed-surface analysis reported:

```text
unattributed_count = 0
total_rows = 2
```

So there were no unattributed proposal rows in the published surface activity analysis.

---

## 10. Final Optimization Decision

The round-level result was:

```json
{
  "format": "shrlm-experiment-round/v1",
  "has_ledger": true,
  "promoted": false,
  "promoted_harness_hash": null,
  "round": 1
}
```

Therefore:

- **No candidate was promoted.**
- `H0*` remained the incumbent.
- The optimization stopped after the configured single round.
- The final harness was frozen as `H0*`.

The frozen harness is stored as:

```text
experiment_DeepMind_mrcrv2/sh_rlm/harness.json
```

with hash:

```text
be733515a34beefc2b17e74111e40b19fad5b132fd8ed1b149e8b2f11283210d
```

---

## 11. Final Harness Characteristics

The final `H0*` harness retained the RLM S1 contract and default floor behavior.

Notable S1 behavior includes:

- persistent Python REPL context
- `context` as potentially very long information
- `llm_query`
- `llm_query_batched`
- recursive `rlm_query` / batched variants
- `SHOW_VARS()`
- explicit `answer` submission mechanism
- instruction to inspect context before answering
- 20K-character REPL-output truncation guidance

S6 runtime policy was disabled in the final harness:

```text
enabled = false
```

and therefore the final harness did not retain the rejected S6 candidate's 50K prompt-character enforcement.

The final S9 answer middleware remained identity/accept behavior.

---

## 12. Operational Timeline

### Mining

The initial mining attempt completed 20 runs in approximately:

**1,257 seconds (~21 minutes)**

at a recorded cost of approximately:

**$0.47**

A subsequent resume verified the mining stage without additional model cost.

### Attribution and proposal

Attribution completed in approximately:

**42 seconds**

with a recorded cost of approximately:

**$0.00235**

Proposal generation completed in approximately:

**4 seconds**

with a recorded cost of approximately:

**$0.00123**

### Validation

Validation was the dominant operational bottleneck.

The first validation attempt ran for approximately:

**65 minutes**

before encountering worker-level memory failure.

The second resumed validation attempt ran for approximately:

**75 minutes**

before the same class of memory failure.

The final resume required approximately:

**12.6 minutes**

and recorded no additional model-token cost.

---

## 13. What Went Well

1. **Mining completed fully:** 20/20 expected runs were walked.
2. **No unwalkable mining runs:** all 20 runs were available for analysis.
3. **No recursive collapse detected:** measured collapse rate was 0.0.
4. **Failure attribution was complete enough for the published analysis:** all observed failure share was classified as non-recursive.
5. **Proposal generation worked:** two distinct candidate surfaces were explored.
6. **Harness validation caught an internal inconsistency:** the S6 candidate was correctly blocked because its runtime constraint contradicted the model-facing capacity statement.
7. **The S2 candidate was measurably worse than H0\*:** both held-in and held-out quality were below the incumbent.
8. **Checkpoint/resume behavior worked sufficiently to preserve completed stages and avoid re-paying for already-completed mining work.**
9. **The final harness and analysis snapshot were successfully persisted.**
10. **The round-level decision was deterministic and conservative:** no inferior candidate was promoted.

---

## 14. What Went Wrong

### 14.1 Validation memory exhaustion

The largest technical failure was the memory growth of the `r01-c02-s2` validation worker.

The worker repeatedly expanded until it consumed nearly all available RAM. The fact that this occurred at approximately 8, 32, and 64 GiB machine sizes strongly suggests that the issue may involve substantial in-memory expansion, duplication, or accumulation rather than merely a fixed hardware requirement.

This should be treated as an **implementation/infrastructure issue requiring investigation**.

### 14.2 Validation did not complete cleanly

Although the orchestrator completed the round and generated the analysis snapshot, one candidate's validation worker remained unsuccessful.

Consequently, the validation evidence should be treated as **lower-bound/partial validation**, not as an entirely clean successful validation execution.

### 14.3 Validation was disproportionately expensive

The validation stage consumed approximately:

- **16.1 million input tokens**
- **369.8 thousand output tokens**
- **$3.25**

across its two paid attempts.

This was by far the largest component of the experiment's recorded model cost.

### 14.4 S6 proposal failed before empirical evaluation

The S6 candidate did not reach performance validation because its environment contract was internally inconsistent.

This means no empirical quality conclusion should be drawn about the underlying S6 idea from this round.

---

## 15. Scientific Interpretation

The strongest conclusions supported by Round 1 are:

### Supported

- `H0*` was a strong incumbent on the configured validation sample, with **93.33% held-in** and **86.67% held-out** pass rates.
- The tested S2 candidate did **not** improve on the incumbent; it scored **73.33% held-in** and **76.67% held-out**.
- The S2 candidate was therefore rejected and not promoted.
- The S6 candidate was rejected for an environment-contract inconsistency rather than performance.
- The 20-run mining set showed **0.0 collapse rate** and **0.0 ungrounded share**.
- Raw-trace inspection showed that **0/20 mining runs used any sub-LLM or recursive RLM calls**; the incumbent instead relied on root-model-guided Python/REPL retrieval using slicing, regex, loops, and direct extraction.
- The observed incumbent therefore represents a strong **programmatic retrieval baseline**, rather than a sub-call-heavy recursive strategy, on this MRCR v2 configuration.
- The observed mining failures were classified entirely as non-recursive failures.
- The raw traces provide direct behavioral evidence that the incumbent's effective strategy is Python-mediated context search rather than recursive delegation.

### Not fully supported

- A claim that the entire validation pipeline completed without infrastructure failure.
- A claim that the S2 validation was obtained from a completely clean, uninterrupted execution.
- A definitive conclusion about the quality of the untested S6 candidate.
- A claim that the validation worker's memory consumption is an inherent requirement of the task rather than an implementation problem.

---

## 16. Recommended Follow-Up

### Priority 1 — Investigate validation memory behavior

Before rerunning the full experiment, inspect the validation worker and identify where the approximately 389-MB request expands toward 60 GB of resident memory.

Areas to investigate include:

- loading the complete request into Python objects
- repeated copies of the long context
- message/history duplication
- serialization/deserialization copies
- retained API request/response objects
- accumulation of results in lists/dictionaries
- process-level duplication caused by validation concurrency
- interaction between `validation_workers=5` and `validation_run_workers=3`


### Priority 2 — Use the trace finding to guide future harness proposals

Future proposals should be evaluated against the fact that `H0*` already performs strong direct retrieval through Python. Simply adding recursive/sub-LLM instructions is not guaranteed to improve performance and may introduce additional token cost, latency, or failure modes. A useful proposal should demonstrate a concrete advantage over the incumbent's existing retrieval behavior, such as better localization, less redundant context inspection, more reliable handling of truncation, or a measurable reduction in expensive root-model iterations.

The trace finding also suggests a useful experimental distinction for future rounds: **REPL-based programmatic retrieval versus recursive delegated retrieval**. Measuring both subcall count and Python retrieval behavior would make it possible to determine whether a candidate genuinely changes the computational strategy rather than merely changing instructions while leaving the incumbent behavior intact.

### Priority 3 — Preserve the current Round 1 artifacts

The small final artifacts already copied locally are sufficient for the initial analysis:

```text
harness.json
round.json
stage_usage.jsonl
analysis/20260913T170917Z/
```

The raw experiment directory is approximately 61 GB and does not need to be downloaded wholesale for initial analysis.

### Priority 4 — Decide whether a rerun is scientifically necessary

Because the tested S2 candidate is clearly below the incumbent on both reported quality metrics, a rerun should not be undertaken merely to obtain another identical rejection.

A rerun becomes more justified if:

- the validation memory issue is fixed,
- clean validation is required for publication-quality evidence,
- or the S6 candidate needs to be redesigned and empirically evaluated.

---

## 17. Artifact Inventory

The principal Round 1 artifacts are:

```text
experiment_DeepMind_mrcrv2/
├── analysis/20260913T170917Z/
│   ├── collapse_and_attribution_optimization.csv
│   ├── incumbent_quality.csv
│   ├── incumbent_quality_candidates.csv
│   ├── surface_activity.csv
│   ├── surface_activity_unattributed.csv
│   ├── provenance.json
│   └── published.json
├── opt/round_01/
│   ├── round.json
│   ├── proposals_complete.json
│   ├── mining/
│   ├── proposals/
│   └── validation/
├── sh_rlm/
│   └── harness.json
├── splits/
└── stage_usage.jsonl
```

The provenance snapshot identifies the relevant source artifacts and records SHA-256 hashes for reproducibility.

---

## 18. Bottom Line

**Round 1 successfully completed its optimization workflow and retained `H0*` as the final harness.**

The round explored **2 proposals**:

- **S6:** rejected before validation due to an explicit model/runtime contract mismatch.
- **S2:** empirically rejected because it underperformed `H0*` on both held-in and held-out quality.

The mining evidence is clean with respect to recursive collapse: **20/20 runs walked, 0.0 collapse rate, 0.0 ungrounded share**, and the three observed failures were classified as non-recursive.

The raw traces also reveal that the incumbent's successful behavior is primarily **Python-mediated retrieval rather than recursive delegation**: all 20 mining runs used zero subcalls and instead relied on context slicing, regex searches, loops, and direct extraction. This is an important baseline characteristic for interpreting future harness improvements.

The principal weakness is **validation infrastructure**. The `r01-c02-s2` worker repeatedly exhausted memory, reaching approximately **59.4 GiB RSS on a 64-GiB instance**. The experiment's final orchestrator state was nevertheless persisted and the final harness was frozen.

Accordingly, the round should be reported as:

> **A completed one-round optimization experiment with no promotion, strong incumbent performance, rejection of the tested S2 candidate, pre-validation rejection of the S6 candidate for contract inconsistency, clean mining/collapse analysis, and a significant validation-worker memory failure that prevents treating the validation execution as fully clean.**
