# Postmortem: DeepSeek-V4-Flash RULER Long Evaluation

**Evaluation directory:** `experiment_ruler_DeepSeekV4Flash_long_eval/`  
**Frozen harness:** `a7d3ef0c5a8a1a999c20a5f5a95df52f3af128eb0ea3fa48a32ed4cb3fdd0ed7`  
**Provider:** Azure Foundry  
**Model:** `DeepSeek-V4-Flash`  
**Target context:** 1,048,576 tokens  
**Evaluation condition:** `sh_rlm` only

## Executive Summary

The long-only evaluator correctly selected exactly 12 RULER long-test instances. The selection assertion printed:

```text
Dispatched test sets: ['ruler_long']
Dispatched instance count: 12
```

However, the evaluation did **not** complete the full 12-instance set. The persisted artifacts contain 12 model-run records, but those records cover only 4 distinct instances with 3 repetitions each. Eight distinct long instances were never reached.

Among the 12 persisted attempts:

- 11 passed.
- 1 was resource-terminated by the 3,600-second per-run timeout.
- The raw attempt pass rate was `91.67%`.
- The distinct-instance coverage was only `4/12` (`33.33%`).
- The timeout occurred on a `common_words_extraction` instance.
- The process was manually interrupted during a later rerun while trying to execute a fresh attempt, after the already persisted attempts had been reused or inspected.
- No final `eval_summary.json` was written.

Therefore, this is a useful diagnostic of long-context behavior and cost, but it is **not a valid completed 12-instance long-set score**.

## What Was Evaluated

The frozen long split contains 12 distinct instances:

| Task family | Distinct long instances in split |
|---|---:|
| `common_words_extraction` | 2 |
| `frequent_words_extraction` | 2 |
| `niah_multikey` | 2 |
| `niah_multiquery` | 2 |
| `niah_multivalue` | 2 |
| `variable_tracking` | 2 |
| **Total** | **12** |

The persisted run records cover only:

| Task family | Distinct instances evaluated | Attempts |
|---|---:|---:|
| `common_words_extraction` | 2 | 6 |
| `niah_multikey` | 1 | 3 |
| `niah_multiquery` | 1 | 3 |
| **Total** | **4** | **12** |

The following task families had no persisted long-evaluation attempts:

- `frequent_words_extraction`
- `niah_multivalue`
- `variable_tracking`

This coverage gap is the most important limitation of the result.

## Performance

### Persisted attempts

| Metric | Result |
|---|---:|
| Persisted attempts | 12 |
| Passed attempts | 11 |
| Failed attempts | 1 |
| Raw attempt pass rate | 91.67% |
| Distinct instances covered | 4/12 |
| Distinct-instance coverage | 33.33% |
| Runtime errors | 0 |
| Resource terminations | 1 |

The one failed attempt was:

```text
Instance: ruler-common_words_extraction-1048576-36b01865b217de68
Attempt: 2
Cause: resource_terminated
Elapsed: 3882.19 seconds
Configured timeout: 3600 seconds
```

The model had generated an intermediate response saying that 44 prompts were too many for one batch and that it would process them in batches of approximately 20. The run timed out before producing a valid final answer. The same instance passed attempts 1 and 3, showing that the task was not deterministically impossible, but it was operationally unstable at this scale.

### By task family

| Task family | Attempts | Passes | Pass rate | Cost |
|---|---:|---:|---:|---:|
| NIAH multi-key | 3 | 3 | 100% | $0.01594 |
| NIAH multi-query | 3 | 3 | 100% | $0.01272 |
| Common words extraction | 6 | 5 | 83.33% | $3.32832 |
| **Total** | **12** | **11** | **91.67%** | **$3.35698** |

The result is dominated by common-words extraction. Retrieval tasks were cheap and reliable, while aggregation over a million-token word stream consumed almost all of the measured cost and produced the only timeout.

## Cost and Token Usage

Persisted attempt totals:

- Total synthesized cost: **$3.35697540**
- Input tokens: **15,182,004**
- Output tokens: **926,264**
- Aggregate recorded execution time: **8,197.41 seconds**
- Approximate aggregate execution time: **2.28 hours**

The average across all persisted attempts was approximately:

- **$0.27975 per attempt**
- **1,265,167 input tokens per attempt**
- **77,189 output tokens per attempt**
- **683.12 seconds per attempt**

These averages are misleading because the workloads are highly heterogeneous. The two common-words instances accounted for approximately **$3.32832**, or **99.15%** of the persisted cost. The NIAH multi-key and NIAH multi-query instances together cost only about **$0.02866**.

The most expensive attempts were common-words extraction:

- `$1.45017`, terminated after 3,882 seconds
- `$1.10489`, passed after 2,228 seconds
- `$0.76475`, passed after 1,818 seconds

Azure cost was synthesized from token counts using the configured rates:

```text
Input:  $0.19 per 1M tokens
Output: $0.51 per 1M tokens
```

The configured per-run cap was `$2.00`, and the failed attempt remained chargeable because it consumed resources before timeout. Its record correctly retained its cost and marked `usage_lower_bound = true`.

## Recursive Calls and Decomposition

The frozen promoted harness was configured with `max_depth = 3`. The persisted long traces show that recursion was used:

- Trace files: 12
- Root iterations: 85
- Total recorded subcalls: 92
- Maximum subcalls in one trace: 40
- Final answer events: 11
- Syntax errors: 0
- Configured maximum depth: 3

The long evaluation therefore exercised actual RLM decomposition rather than only direct root-local Python processing.

The recursive behavior was concentrated in the difficult common-words aggregation task. The NIAH traces were generally solved with few or no subcalls, while the common-words traces generated large batches of child work. The timeout trace explicitly attempted to create 44 prompts and then batch them in groups, which explains both the high subcall count and the extreme cost/time behavior.

The run used the frozen promoted harness, including the S3, S4, and S10 changes from the preceding optimization round. No new harness optimization or promotion occurred during this long evaluation.

## Operational Failure and Interruption

The first long-only process persisted 12 records but remained alive after the records stopped increasing. Its final summary was never written. The process held substantial network state and was idle during finalization, so it was interrupted with `Ctrl-C`.

The rerun correctly printed the 12-instance dispatch assertion, but it was also interrupted while attempting to execute a fresh governed run. The traceback ended in:

```text
CancellationError: Execution cancelled by user (Ctrl+C)
```

This means:

- The dispatch filter worked correctly.
- The persisted records remain available.
- The evaluation directory is not a clean completed evaluation artifact.
- `eval/eval_summary.json` must not be treated as an available aggregate.
- The raw `runs.jsonl` and individual trace files are still suitable for partial analysis.

No claim lock remained after the interrupted process exited.

## Interpretation

The long-context results reveal a clear task-dependent scaling pattern:

1. **Retrieval remains easy at 1M-token context.**
   NIAH multi-key and multi-query passed all persisted attempts at very low cost.

2. **Aggregation is the bottleneck.**
   Common-words extraction required millions of input tokens, many child calls, long execution times, and almost all measured spend.

3. **The harness decomposes, but decomposition can explode.**
   The model recognized the need to split the large aggregation job into child batches. That is qualitatively correct behavior, but one trace created 44 prompts and exceeded the one-hour run cap.

4. **The promoted S3/S4/S10 edits address answer correctness, not workload scheduling.**
   They helped preserve extraction correctness, but they do not impose a hard chunk-width, child-count, or wall-time policy. The long timeout exposes this remaining weakness.

5. **The 91.67% number is not a long-set benchmark score.**
   It is an attempt-level score over only 4 of 12 distinct instances, with three repetitions per reached instance. The unmeasured eight instances prevent a complete long-set conclusion.

## Comparison With the Short Optimization Round

The preceding short held-out validation used 36 instances and promoted the merged harness at 36/36 passes (100%), with mean cost approximately `$0.01095` per run and mean subcalls `1.25`.

The long evaluation is qualitatively different:

| Metric | Short held-out promoted harness | Persisted long attempts |
|---|---:|---:|
| Target context | 65,536 tokens | 1,048,576 tokens |
| Distinct instances | 36 | 4 of 12 reached |
| Attempts | 36 | 12 |
| Pass rate | 100% | 91.67% attempt-level |
| Mean cost | $0.01095/run | $0.27975/attempt |
| Mean subcalls | 1.25/run | 7.67/attempt |
| Runtime terminations | 0 | 1 |

The long-context workload increased mean cost by roughly 25.5x and mean subcalls by roughly 6.1x in the persisted sample. These ratios are approximate because the task mix and coverage are different, but they clearly show that the long aggregation regime is much more expensive and operationally fragile.

## Recommendations

1. **Do not report this as a completed 12-instance long evaluation.** Report it as a partial diagnostic: 4/12 distinct instances, 12 attempts, 11 passes, one timeout.

2. **Separate aggregation from retrieval in future budgets.** A single `$2` per-run cap is not enough to make common-words behavior predictable at 1M tokens; the task needs a chunking policy and a bounded child-call budget.

3. **Add or activate runtime scheduling guidance.** The harness should instruct the root to cap batch width, cap total child prompts, and aggregate child results incrementally rather than creating 44 child prompts in one turn.

4. **Evaluate each long instance once first.** Use `eval_repetitions = 1` for a coverage pass over all 12 distinct long instances before spending on repeated attempts.

5. **Use repetitions only after coverage.** Once all task families have one completed measurement, repeat the hard families with `v = 2` or `3` to estimate variance.

6. **Keep long-evaluation output separate.** The isolated directory was the correct operational choice and should be retained as partial evidence rather than overwritten.

7. **If a complete aggregate is required, use a fresh output directory with lower repetition and stronger timeout-aware chunking.** Do not delete this directory; it contains the only persisted evidence from the interrupted run.

## Artifact Index

- Raw long records: `eval/sh_rlm/ruler_long/round_00/runs.jsonl`
- Raw long traces: `eval/sh_rlm/ruler_long/round_00/runs/`
- Frozen harness copy: `sh_rlm/harness.json`
- Frozen long split: `splits/ruler_long_test.jsonl`
- Config identity: `config.json`
- Execution metadata: `eval/sh_rlm/ruler_long/round_00/execution.json`

## Bottom Line

The long evaluation proved that the promoted harness can solve 1M-token retrieval and some 1M-token aggregation instances, and it demonstrated real recursive decomposition. It also exposed the main remaining scaling failure: common-words aggregation can generate large child batches, consume over an hour, cost more than a dollar per attempt, and still terminate on timeout.

The persisted data supports a strong diagnostic conclusion about long-context cost and decomposition behavior, but not a statistically complete 12-instance performance claim because only 4 of the 12 unique long instances were actually evaluated.
