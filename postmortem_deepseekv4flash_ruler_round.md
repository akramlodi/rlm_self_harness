# Postmortem: DeepSeek-V4-Flash RULER Optimization Round

**Experiment directory:** `experiment_ruler_DeepSeekV4Flash_round2/`  
**Round:** 1  
**Provider:** Azure Foundry  
**Model:** `DeepSeek-V4-Flash`  
**Environment:** RULER  
**Final status:** completed successfully; merged harness promoted

## Executive Summary

The one-round RULER optimization experiment completed end to end. The baseline harness achieved 34/36 held-out passes (94.44%). The merged candidate harness achieved 36/36 (100%), correcting the two baseline failures observed on the held-out set. It also reduced mean cost per run from $0.02774 to $0.01095 and reduced mean recursive subcalls from 2.97 to 1.25.

Three candidate edits were proposed and evaluated together as one merged harness:

1. S4 verification instruction for checking the final answer before submission.
2. S3 execution instruction for cleaning punctuation from extracted numbers.
3. S10 skill for extracting the value after a key-specific label instead of matching the first number in a line.

The merged harness was promoted. Because all three edits were evaluated as a bundle, this round establishes the effectiveness of the combined change, not the independent causal effect of each edit.

## Configuration and Scope

The RULER configuration used:

```toml
[splits]
n_in = 36
n_ho = 36
test_short = 36
test_long = 12

[loop]
m = 2
v = 1
k = 4
t = 1
environment = "ruler"

[promotion]
cost_band = [0.0, 3]
```

The RULER task families were:

- `niah_multikey`
- `niah_multivalue`
- `niah_multiquery`
- `variable_tracking`
- `common_words_extraction`
- `frequent_words_extraction`

The short pool was generated at 65,536 target tokens. The long test split was generated at 1,048,576 target tokens with 12 instances. The round's optimization promotion decision used the 36-instance held-out short split; the long split was materialized but was not part of this promotion decision.

The experiment used `H0*R` as the initial harness. The baseline harness hash was:

```text
0ca3510ded8a0ce5095c2e3338c4d8606fbc819650c20b65079b405ae1ecac2b
```

The promoted harness hash was:

```text
a7d3ef0c5a8a1a999c20a5f5a95df52f3af128eb0ea3fa48a32ed4cb3fdd0ed7
```

## Experiment Stages

### Mining

The mining stage ran two attempts per each of the 36 held-in instances, producing 72 mining runs. It consumed:

- Cost: `$2.26695933`
- Input tokens: `10,859,157`
- Output tokens: `399,450`
- Wall time: `4,802.19` seconds recorded across stage work

The mining stage produced evidence that the baseline had concrete, repairable answer-construction and extraction failures. That evidence was passed to attribution and proposal generation.

### Attribution

The attribution stage consumed:

- Cost: `$0.00277905`
- Input tokens: `12,345`
- Output tokens: `850`
- Wall time: `17.02` seconds

The attribution artifacts classified the failures as root-level, non-recursive failure modes. The report's `collapse_and_attribution_optimization.csv` records three optimization failures and identifies `no_recursion` as the failure level for those findings.

### Proposal Generation

Proposal generation consumed:

- Cost: `$0.00591707`
- Input tokens: `21,686`
- Output tokens: `3,523`
- Wall time: `41.00` seconds

Three candidates were produced:

| Candidate | Surface | Main change |
|---|---:|---|
| `r01-c01-s4` | S4 | Verify the complete computed answer before setting `answer["ready"] = True`. |
| `r01-c02-s3` | S3 | Strip trailing non-digit characters from extracted numeric values. |
| `r01-c03-s10` | S10 | Add a reusable skill for key-specific numeric extraction and verification. |

### Validation

Baseline and candidate validation each used the 36-instance held-out split with one repetition. Validation consumed:

- Cost: `$1.39274674`
- Input tokens: `6,727,579`
- Output tokens: `224,523`
- Wall time: `3,366.69` seconds

The three candidates were not promoted independently. They were merged into one subject and evaluated as a single combined harness.

## Performance Results

| Metric | Baseline | Promoted merged harness | Change |
|---|---:|---:|---:|
| Held-out passes | 34/36 | 36/36 | +2 |
| Pass rate | 94.44% | 100.00% | +5.56 percentage points |
| Total held-out cost | $0.99856458 | $0.39418216 | -60.6% |
| Mean cost per run | $0.02773790 | $0.01094950 | -60.6% |
| Total subcalls | 107 | 45 | -62 |
| Mean subcalls per run | 2.9722 | 1.25 | -58.0% |
| Runtime errors | 0 | 0 | unchanged |
| Resource terminations | 0 | 0 | unchanged |
| Skill loads | 0 | 1 | +1 |

The promoted harness was inside the configured cost band:

```text
Baseline mean cost:  0.027737905
Candidate mean cost: 0.010949504
Allowed band:       [0.0, 3.0] x baseline
Within band:        true
```

The subcall band was unconstrained, but the merged candidate nevertheless used substantially fewer subcalls.

### Total Experiment Cost

The recorded stage costs sum to approximately:

```text
Mining       $2.26695933
Attribution  $0.00277905
Proposal     $0.00591707
Validation   $1.39274674
--------------------------------
Total        $3.66840219
```

Azure cost was synthesized from token usage using the configured rates of `$0.19` per million input tokens and `$0.51` per million output tokens.

## Weaknesses Found

### 1. Premature or incomplete final answer assembly

The S4 proposal was based on a concrete trace in which the root computed four correct variables, then submitted only three:

```text
VAR_0_0, VAR_0_1, VAR_0_2
```

The missing value was `VAR_0_3`. The root had enough intermediate evidence to produce the correct result but did not compare the final answer against that result before marking it ready.

**Repair:** S4 now requires a final consistency check between `answer["content"]` and the complete computed result before `answer["ready"] = True`.

### 2. Trailing punctuation in numeric answers

The root sometimes extracted values such as:

```text
861194.
```

The RULER verifier expects the plain numeric value:

```text
861194
```

**Repair:** S3 now instructs the root to strip trailing punctuation and retain only the numeric value when extracting numbers.

### 3. Key-suffix confusion

RULER keys contain numeric suffixes, for example:

```text
cascade-4744
```

A broad regex such as `r"(\d+)"` matches `4744`, even though the actual magic number appears later in the line. This caused the root to return the key suffix rather than the target value.

**Repair:** S10 adds `extract_value_by_key`, which recommends a key-specific pattern such as:

```python
re.search(r"magic number for " + re.escape(key) + r" is: (\d+)", line)
```

The skill also requires checking that the extracted number is not the key suffix.

## Harness Changes

The final promoted harness is the merged result of three surface edits:

### S3: execution instruction

Added numeric cleanup guidance:

> When extracting numeric values from text, clean each extracted value by removing any trailing non-digit characters before including it in the answer.

### S4: verification instruction

Added final-answer verification guidance:

> Before flipping `answer["ready"] = True`, verify that `answer["content"]` matches the complete result you computed.

It specifically warns against accidentally truncating a complete list during final assembly.

### S10: reusable skill

Added the `extract_value_by_key` skill with:

- Use conditions for key/value extraction.
- A key-specific regex procedure.
- A fallback delimiter procedure.
- A check against key-suffix confusion.
- Pitfalls involving first-number matching and punctuation.
- Verification steps comparing the source line and extracted value.

No changes were made to S1, S2, S5, S6, S7, S8, or S9 in the promoted bundle.

## Promotions and Decision Logic

The decision artifact reports:

- Three candidate constituents.
- One merged validation subject.
- Baseline pass count: `34`.
- Merged candidate pass count: `36`.
- Held-out delta: `+2`.
- Promotion decision: `promoted`.

The promotion used zero regression and zero improvement thresholds, so the merged candidate needed to avoid regression and satisfy the configured resource rules. It passed the held-out comparison and stayed within the `[0.0, 3.0]` cost band.

The constituent records for S3, S4, and S10 are marked `bundled`; they were evaluated only as part of the merged subject. Consequently:

- The round proves the bundle improved the measured result.
- It does not prove S3 alone, S4 alone, or S10 alone would produce the same improvement.
- A future ablation should evaluate each edit separately if individual attribution matters.

## Recursive Calls and Decomposition

This round did exercise recursive behavior during held-out validation.

### Baseline

- Total subcalls: `107`
- Mean subcalls per run: `2.9722`
- Skill loads: `0`
- All 36 held-out runs completed without runtime errors.

### Promoted merged harness

- Total subcalls: `45`
- Mean subcalls per run: `1.25`
- Skill loads: `1`
- All 36 held-out runs completed without runtime errors.

The model therefore used child calls and decomposition in validation, but the promoted instructions made it more accurate while requiring fewer child calls. The cost reduction is consistent with the lower subcall count and reduced downstream work, although this round does not isolate the exact causal contribution of each surface.

The mining attribution report classified the observed failures as `no_recursion` failures. This means the failures being repaired were root-level mistakes made without sufficient recursive decomposition; it does not mean that the entire experiment had zero subcalls. The validation summaries clearly record recursive subcalls for both baseline and merged harnesses.

## What Was Not Measured

The long RULER split was materialized, but it was not included in the round's held-out promotion decision. This round therefore does not establish performance at the configured 1,048,576-token target.

The experiment also did not independently evaluate the three candidates. The merged result may reflect interactions among S3, S4, and S10.

The validation used one repetition per instance (`v = 1`). The 100% candidate result is strong for this held-out sample, but repeated evaluation would provide a more stable estimate of model variance.

## Recommendations

1. Run the promoted harness on the 12 long-test RULER instances.
2. Repeat the 36-instance held-out evaluation with `v >= 3` to measure variance.
3. Run ablations for S3, S4, and S10 separately.
4. Track pass rate and subcalls by RULER family, especially `common_words_extraction` and `frequent_words_extraction`.
5. Preserve the baseline and merged artifacts as the comparison pair; do not overwrite this experiment directory.
6. Consider reducing validation concurrency or adding provider-specific throttling when Azure rate limits dominate wall time.

## Artifact Index

- Round decision: `opt/round_01/validation/round_01/decision.json`
- Promotion ledger: `opt/round_01/validation/round_01/promotions.jsonl`
- Baseline summary: `opt/round_01/validation/round_01/baseline/summary.json`
- Merged summary: `opt/round_01/validation/round_01/merged/summary.json`
- Candidate proposals: `opt/round_01/proposals/r01-c01-s4/`, `r01-c02-s3/`, `r01-c03-s10/`
- Final harness: `sh_rlm/harness.json`
- Stage accounting: `stage_usage.jsonl`
- Analysis snapshot: `analysis/20260916T172322Z/`
