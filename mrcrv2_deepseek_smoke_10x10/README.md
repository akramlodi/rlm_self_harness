# MRCRv2 DeepSeek 10+10 Smoke Run

## Executive Summary

This directory contains a one-round MRCRv2 smoke experiment using Azure Foundry and `DeepSeek-V4-Flash`. Compared with the earlier 3-held-in/3-held-out smoke run, this run used 10 held-in and 10 held-out instances, giving 20 validation runs per harness.

The result is a useful negative result:

- The baseline passed **20/20 (100%)**.
- Candidate `r01-c01-s2` passed **16/20 (80%)**.
- Candidate `r01-c02-s4` passed **18/20 (90%)**.
- Neither candidate was promoted.
- The baseline remained the final and frozen harness.

The two candidates reduced model-call volume and cost substantially, but both regressed in accuracy. On this run, the baseline is the only harness supported by the validation evidence.

### Run facts

- Round: 1 of 1
- Profile: `smoke`
- Environment: synthetic MRCRv2
- Model: `DeepSeek-V4-Flash`
- Backend: Azure Foundry
- Short split: approximately 65,536 target tokens, 2 needles
- Held-in: 10 instances
- Held-out: 10 instances
- Mining repetitions: `m=1`
- Validation repetitions: `v=1`
- Candidate width: `k=2`
- Optimization rounds: `t=1`
- Long split: not evaluated
- Proposals generated: 2
- Materialization failures: 0
- Proposals promoted: 0
- Stop reason: `max_rounds`
- Final harness: baseline hash `be733515a34beefc2b17e74111e40b19fad5b132fd8ed1b149e8b2f11283210d`

## Experiment Flow

The run executed the normal self-harness pipeline:

1. Mining over the 10 held-in instances
2. Failure attribution
3. Proposal generation and materialization
4. Validation of the baseline and both candidates on held-in and held-out splits
5. Promotion decision

The experiment completed all stages. There were no skipped proposals, materialization failures, or incomplete validation subjects.

## Cost and Token Usage

The persisted stage ledger reports the following:

| Stage | Cost | Input tokens | Output tokens | Recorded stage time |
|---|---:|---:|---:|---:|
| Mining | $0.04298218 | 201,997 | 9,025 | 517.7 s |
| Attribution | $0.00122175 | 5,880 | 205 | 5.3 s |
| Proposal | $0.00080720 | 3,113 | 423 | 9.4 s |
| Validation | $0.36412906 | 1,744,996 | 63,882 | 881.5 s |
| **Total** | **$0.40914019** | **1,955,986** | **73,535** | **1,413.8 s** |

The stage times sum recorded work and should not be interpreted as strictly serial wall-clock time because validation used parallel workers. The run took approximately 24 minutes from the first mining ledger timestamp to the final validation ledger timestamp.

### Validation cost by harness

| Harness | Cost | Input tokens | Output tokens | Mean cost/run |
|---|---:|---:|---:|---:|
| Baseline | $0.20124056 | 977,381 | 30,467 | $0.01006203 |
| S2 candidate | $0.07526034 | 350,403 | 17,027 | $0.00376302 |
| S4 candidate | $0.08762816 | 417,212 | 16,388 | $0.00438141 |

Relative to baseline validation cost:

- S2 cost was approximately **62.6% lower**.
- S4 cost was approximately **56.5% lower**.

Those savings came with accuracy regressions, so they did not satisfy the promotion rule.

## Accuracy Results

Each subject ran once over 10 held-in and 10 held-out instances.

| Harness | Held-in | Held-out | Overall | Promotion result |
|---|---:|---:|---:|---|
| Baseline | 10/10 (100%) | 10/10 (100%) | 20/20 (100%) | Incumbent retained |
| `r01-c01-s2` | 8/10 (80%) | 8/10 (80%) | 16/20 (80%) | Rejected |
| `r01-c02-s4` | 10/10 (100%) | 8/10 (80%) | 18/20 (90%) | Rejected |

The decision artifact records:

- `n_candidates = 2`
- `plan = none`
- `promoted = false`
- `promoted_harness_hash = null`
- `promoted_subject_id = null`

### Interpretation

The baseline was perfect on both splits in this run. That means the candidates had no opportunity to improve the measured baseline score; the only acceptable outcome was non-regression. Both candidates failed that requirement:

- S2 lost two held-in and two held-out examples.
- S4 preserved held-in accuracy but lost two held-out examples.

The held-out regressions are especially important because held-out examples are not used for mining. They indicate that the candidate behavior did not generalize safely even within this small short-split evaluation.

## Recursion and Model-Call Behavior

The validation summaries report sub-call counts by split:

| Harness | Held-in sub-calls | Held-out sub-calls | Total sub-calls | Mean sub-calls/run |
|---|---:|---:|---:|---:|
| Baseline | 6 | 0 | 6 | 0.30 |
| S2 candidate | 4 | 0 | 4 | 0.20 |
| S4 candidate | 3 | 0 | 3 | 0.15 |

The underlying model-call totals provide a broader cost/behavior proxy:

| Harness | Total model calls | Mean calls/run | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Baseline | 174 | 8.70 | 3 | 20 |
| S2 candidate | 105 | 5.25 | 2 | 13 |
| S4 candidate | 112 | 5.60 | 2 | 11 |

The candidates did not simply preserve the baseline behavior at lower cost. They changed the execution strategy substantially, reducing recursive/sub-call activity and total model calls. That reduction correlates with the observed accuracy loss in this run.

There were no resource terminations in validation for the baseline or either candidate. The validation subjects all completed. The analysis snapshot reports the mining-side failures as `no_recursion_failure_share = 1.0`, but `sub_verifier_available = false`; therefore no sub-verifier coverage or grounded child-quality result was available for this run.

## Mining Findings

Mining ran 10 held-in instances:

- 8 mining runs passed
- 2 mining runs failed
- Mining cost: `$0.04298218`
- Mining usage: 201,997 input tokens and 9,025 output tokens

The two mined failures were:

### Failure 1: whole-input sub-call collapse

Instance: `mrcrv2-2n-2fe9527067914895`

The attribution described the behavior as:

> The root never decomposed the input or issued any sub-calls; instead it spent all 6 iterations exploring the raw context string as a data analysis problem, never extracting the requested assistant response.

Classification:

- Causal status: causal
- Agent mechanism: `whole_input_subcall_collapse`
- Failing level: `no_recursion`
- Verifier cause: `resource_terminated`

### Failure 2: premature termination / formatting mistake

Instance: `mrcrv2-2n-90f94733a86bf4fb`

The attribution described the behavior as:

> The root extracted the assistant response including the `assistant: ` prefix, which the verifier rejected as a wrong value because the gold answer expects only the content without the prefix.

Classification:

- Causal status: causal
- Agent mechanism: `premature_termination`
- Failing level: `no_recursion`
- Verifier cause: `wrong_value`

These two mining failures motivated two different proposed interventions: force decomposition, and require completion verification before finalization.

## Proposed Harnesses

Exactly two candidates were generated and materialized.

### Candidate 1: `r01-c01-s2`

Surface changed: `S2_decomposition_instruction`

Candidate harness hash:

`e152171526e29ddd0c37093ff3615953c0b63b2564706be12ec4df6103b52d71`

The proposal instructed the root:

> You have a large context. Do NOT delegate the entire context to a single sub-call. Instead, split the input into at least 2 roughly equal slices (e.g., by lines, by items, or by some natural boundary) and issue a separate sub-call for each slice. After all sub-calls return, combine their results into a final answer. If the context is small enough that splitting seems unnecessary, still split into at least 2 slices to ensure decomposition.

Predicted effect:

- Prevent the root from collapsing the entire input into one sub-call
- Force at least two sub-calls
- Enable recursion/decomposition on large inputs

Declared risk:

- Unnecessary splits on small contexts could waste resources, even when one sub-call would suffice

Observed result:

- Held-in: 8/10
- Held-out: 8/10
- Overall: 16/20
- Total sub-calls: 4
- Total model calls: 105
- Cost: `$0.07526034`
- Decision: rejected

The candidate did reduce call volume and cost, but the forced decomposition rule was too aggressive or otherwise disrupted answer production on four validation instances. The persisted validation artifacts record those runs as unsuccessful with no runtime error field; they do not provide a structured verifier-cause label for each candidate miss.

### Candidate 2: `r01-c02-s4`

Surface changed: `S4_verification_instruction`

Candidate harness hash:

`58d26841921e33f24eabe80a624a77c04d0f07138e495795c59fc7dba5138c91`

The proposal instructed the root:

> Before setting `answer["ready"] = True`, verify that all slices or sub-tasks you created have been processed and their results are included in the final answer. If any slice remains unvisited or any sub-call returned an error, do NOT mark the answer as ready. Instead, re-issue the sub-call or handle the error. Only commit when you have evidence that every part of the input has been covered.

Predicted effect:

- Prevent premature termination
- Require explicit completion checking
- Prevent finalization when a slice is unvisited or a sub-call failed

Declared risk:

- The model could get stuck in a loop if it cannot resolve a sub-call error

Observed result:

- Held-in: 10/10
- Held-out: 8/10
- Overall: 18/20
- Total sub-calls: 3
- Total model calls: 112
- Cost: `$0.08762816`
- Decision: rejected

S4 preserved held-in accuracy but still introduced two held-out failures. It was safer than S2 by accuracy, but not safe enough to replace a perfect baseline.

## Failure Summary

| Area | Baseline | S2 candidate | S4 candidate |
|---|---:|---:|---:|
| Validation runs | 20 | 20 | 20 |
| Passed | 20 | 16 | 18 |
| Failed | 0 | 4 | 2 |
| Resource terminations | 0 | 0 | 0 |
| Held-in pass rate | 100% | 80% | 100% |
| Held-out pass rate | 100% | 80% | 80% |
| Total model calls | 174 | 105 | 112 |
| Total sub-calls | 6 | 4 | 3 |
| Validation cost | $0.20124056 | $0.07526034 | $0.08762816 |
| Promoted | No change | No | No |

The main problem was not a timeout or provider failure during validation. It was behavioral regression: both proposed instructions changed how the root interacted with the context, and the changed behavior produced wrong or incomplete answers on held-out examples.

## What Improved

The candidates improved operational metrics:

- Lower total model-call count
- Lower input-token consumption
- Lower output-token consumption
- Lower validation cost
- Fewer recursive/sub-call operations

However, those are not improvements for this task unless accuracy is preserved. The promotion rule correctly rejected both candidates because each lost held-out accuracy relative to a perfect baseline.

The baseline itself also had a higher recursion rate and cost. That is a real optimization opportunity, but this round did not find a safe way to reduce it. The next candidate should target unnecessary work more narrowly, rather than requiring decomposition or completion checking globally.

## Important Limitations

1. **Small sample.** Twenty validation instances per harness is better than the earlier 3+3 run, but it is still a smoke test, not a definitive benchmark result.
2. **Short split only.** This run used approximately 65K target tokens and 2 needles. It did not evaluate the configured 2M-token/8-needle long split.
3. **Synthetic MRCRv2.** The environment generates instances locally and is not the official DeepMind `eval_hub/mrcr_v2` dataset.
4. **No sub-verifier coverage.** The analysis says `sub_verifier_available = false`; recursion counts are available from validation summaries, but grounded child-quality verdicts are not.
5. **No promoted change.** The frozen harness is the baseline, not either proposed candidate.
6. **Candidate failure causes are not fully structured.** Mining failures have detailed causal attribution. Candidate validation misses are recorded as failed runs with no runtime error, but the persisted summaries do not attach a per-run verifier-cause explanation.
7. **Pricing is synthesized.** Costs use the configured DeepSeek rates of `$0.19` per 1M input tokens and `$0.51` per 1M output tokens.

## Artifact Map

- `stage_usage.jsonl`: stage-level costs, tokens, and times
- `opt/round_01/mining/round_01/`: mining runs, records, attributions, and evidence
- `opt/round_01/proposals_complete.json`: candidate inventory and materialization status
- `opt/round_01/proposals/r01-c01-s2/proposal.json`: S2 proposal
- `opt/round_01/proposals/r01-c02-s4/proposal.json`: S4 proposal
- `opt/round_01/validation/round_01/baseline/summary.json`: baseline validation summary
- `opt/round_01/validation/round_01/r01-c01-s2/summary.json`: S2 validation summary
- `opt/round_01/validation/round_01/r01-c02-s4/summary.json`: S4 validation summary
- `opt/round_01/validation/round_01/decision.json`: promotion decision
- `analysis/`: generated analysis tables
- `sh_rlm/harness.json`: frozen baseline harness

## Bottom Line

This 10+10 smoke run was successful as an experiment and unsuccessful as a candidate-promotion round. The baseline solved all 20 validation instances. Both generated proposals reduced cost and recursion, but both reduced accuracy, especially on held-out data. The correct conclusion is to retain the baseline, treat unnecessary recursion as the optimization target, and design a narrower proposal before running a larger pilot or the long 2M/8-needle evaluation.
