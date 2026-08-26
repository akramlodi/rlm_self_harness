# Live baseline smoke — H0\* and λ-RLM

A live (real API spend, $0.5226 total measured) run of `examples/experiment_smoke.py --live`,
exercising the full experiment pipeline — one optimization round (mining + validation) followed by
evaluation of all four conditions — against the `[smoke]` profile with the live-scale overrides
(`LIVE_*` constants in `examples/experiment_smoke.py`).

**Purpose**: prove the scaffold works end-to-end on a real backend, not to benchmark baseline
quality. Sample sizes here (1-2 instances per environment/length) are far too small to draw
accuracy conclusions — see [Caveats](#caveats--known-issues).

## Run configuration

| | |
|---|---|
| Profile | `smoke` |
| Model | `qwen/qwen3-30b-a3b-instruct-2507` via OpenRouter |
| Config identity hash | `6b1ac5aac58cacd381c48b4352ed7aae221c6ce0807d0365862ac8cb97b69120` |
| Code state | `integration/lamda-rlm-current-main` @ `557e806`, plus the hard-deadline fix below (uncommitted at run time) |
| Conditions | `b1` (H0, sparse starting harness) · `h0_star` (H0\*, shipped RLM reference) · `lambda_rlm` (λ-RLM, typed combinator runtime) · `sh_rlm` (this run's optimized harness) |
| Environments | GraphWalks (BFS / parent-finding over graphs) · OOLONG-Pairs (pairwise user classification over long contexts) |

## Headline result

**`SMOKE PASSED`** — pipeline plumbing is sound: all artifacts persisted, every run carries a
cost, all three model roles (runner/attributor/proposer) measurably exercised, at least one
uncapped long run landed per environment. Total measured spend **$0.5226** against the $5.00
ceiling.

**Accuracy: 6/24 (25%) exact-match passes.** Expected at this scale and task difficulty — see
[Why this is expected](#why-a-low-pass-rate-is-expected), not a regression.

## Full results

| Condition | Env | Length | Passed | Cause | Cost | Time (s) |
|---|---|---|---|---|---|---|
| b1 | graphwalks | short | ❌ | no_answer | $0.0004 | 10.4 |
| b1 | graphwalks | short | ✅ | — | $0.0027 | 62.4 |
| b1 | graphwalks | long | ❌ | wrong_format | $0.0047 | 94.8 |
| b1 | oolong_pairs | short | ❌ | wrong_format | $0.0012 | 44.9 |
| b1 | oolong_pairs | short | ❌ | wrong_format | $0.0029 | 75.9 |
| b1 | oolong_pairs | long | ❌ | wrong_format | $0.0419 | 548.2 |
| h0_star | graphwalks | short | ✅ | — | $0.0011 | 11.0 |
| h0_star | graphwalks | short | ✅ | — | $0.0023 | 38.8 |
| h0_star | graphwalks | long | ❌ | wrong_format | $0.1050 | 101.8 |
| h0_star | oolong_pairs | short | ❌ | mixed_set_error | $0.0030 | 52.4 |
| h0_star | oolong_pairs | short | ❌ | wrong_format | $0.0111 | 191.4 |
| h0_star | oolong_pairs | long | ❌ | wrong_format | $0.0842 | 259.4 |
| lambda_rlm | graphwalks | short | ✅ | — | $0.0002 | 6.7 |
| lambda_rlm | graphwalks | short | ✅ | — | $0.0008 | 67.4 |
| lambda_rlm | graphwalks | long | ❌ | wrong_format | $0.0237 | 57.7 |
| lambda_rlm | oolong_pairs | short | ❌ | mixed_set_error | $0.0007 | 8.0 |
| lambda_rlm | oolong_pairs | short | ❌ | wrong_format | $0.0019 | 131.4 |
| lambda_rlm | oolong_pairs | long | ❌ | resource_terminated | $0.0464 | 1830.4 |
| sh_rlm | graphwalks | short | ❌ | no_answer | $0.0004 | 8.3 |
| sh_rlm | graphwalks | short | ✅ | — | $0.0013 | 80.5 |
| sh_rlm | graphwalks | long | ❌ | no_answer | $0.0004 | 13.4 |
| sh_rlm | oolong_pairs | short | ❌ | spurious | $0.0023 | 38.8 |
| sh_rlm | oolong_pairs | short | ❌ | wrong_format | $0.0034 | 41.2 |
| sh_rlm | oolong_pairs | long | ❌ | mixed_set_error | $0.0026 | 87.3 |

### Per-condition summary

| Condition | Passed | Eval cost |
|---|---|---|
| `b1` | 1/6 | $0.0538 |
| `h0_star` | 2/6 | $0.2067 |
| `lambda_rlm` | 2/6 | $0.0738 |
| `sh_rlm` | 1/6 | $0.0103 |
| **Total** | **6/24** | **$0.3445** |

### Failure causes (18 failures)

| Cause | Count | Meaning |
|---|---|---|
| `wrong_format` | 10 | No parseable answer in the required syntax at all — not "wrong reasoning," no answer was extractable |
| `no_answer` | 3 | Harness ended without ever calling `answer["ready"] = True` |
| `mixed_set_error` | 3 | Parseable answer, but both missing and extra items vs. gold |
| `resource_terminated` | 1 | λ-RLM's OOLONG-long run hit the hard wall-clock deadline (working as intended, see below) |
| `spurious` | 1 | Parseable answer, only extra (no missing) items vs. gold |

## Key findings

### 1. The two behavioral clusters are exact, not approximate

Conditions don't just tie on pass *count* — they pass and fail the **identical instances**:

```
b1         -> FAIL PASS FAIL FAIL FAIL FAIL
sh_rlm     -> FAIL PASS FAIL FAIL FAIL FAIL   (identical to b1)
h0_star    -> PASS PASS FAIL FAIL FAIL FAIL
lambda_rlm -> PASS PASS FAIL FAIL FAIL FAIL   (identical to h0_star)
```

`sh_rlm` (this run's optimized harness) inherited `b1`'s exact weakness on the
`graphwalks-parents-short` instance, same cause (`no_answer`) — one heavily-shrunk optimization
round produced no measurable change here, which is expected given the smoke's tiny mining/validation
budget, not a regression.

`h0_star` and `lambda_rlm` are independent implementations (a harness-REPL prompt variant vs. a
separate typed-combinator runtime) that converge on the exact same capability gain over `b1` on
that one instance — suggestive that the gap is about instruction completeness, not
scaffold architecture, at least for this query type.

### 2. Nothing passes OOLONG-Pairs — 0/12 across both conditions that share the pattern

Every condition fails every OOLONG-Pairs instance (short and long). `wrong_format` dominates:
10 of 18 total failures. This is a real, worth-investigating gap, distinct from "the model reasoned
incorrectly" — the verifier (`shrlm/environments/oolong_pairs.py`) never found a parseable
`(user_id_1, user_id_2)` answer in the response at all in most of these cases.

### 3. The long OOLONG-Pairs smoke instance is a pathological outlier

The single sampled long instance (`oolong-t11-w9-cea0544da66c5117`) requires **21,464 exact gold
pairs** in its answer — not "tens to low thousands" as `oolong_pairs.py`'s own safety-ceiling
comment assumes is typical of a legitimate window. `sh_rlm`'s attempt produced a 519 KB answer and
still had ~20,701 missing / ~33,844 extra pairs. This single draw dominates the entire long-OOLONG
result; a second random seed drew an even larger outlier (86,234 pairs). This is a sampling problem,
not a baseline-quality problem — see [Caveats](#caveats--known-issues).

### 4. The hard-deadline backstop fired correctly

λ-RLM's long OOLONG run ran for exactly **1830.4s** (the computed hard deadline:
`max_timeout(1200) × 1.5 + 30`) before being cleanly terminated with `HardDeadlineExceeded`,
persisting a `resource_terminated`/`usage_lower_bound=true` entry rather than hanging.

This required a real fix during this session: the deadline's `SIGALRM`-raised exception was being
silently caught and turned into an in-band `"Error: ..."` string by broad `except Exception`
handlers in `rlm/environments/local_repl.py` (`execute_code`, `_llm_query_once`,
`_llm_query_batched`, `_rlm_query_once`) and `rlm/core/comms_utils.py`
(`send_lm_request`/`send_lm_request_batched`) — which `_call_with_retry` then classified as a
retryable "syntax error" and retried indefinitely with the one-shot alarm already spent, hanging
the run forever. Fixed by re-raising `TimeoutExceededError` before the broad catch at each site;
regression tests added in `tests/test_local_repl.py`, `tests/test_lm_handler.py`, and
`tests/baselines/test_lambda_governance.py` (the last drives the real `SIGALRM` mechanism
end-to-end, not a mocked one).

## Why a low pass rate is expected

- **The tasks are genuinely hard.** In the λ-RLM paper (arXiv:2603.20105), plain RLM averages only
  17.1% accuracy on the equivalent OOL-Pairs task *across all 9 benchmarked models*, including
  235B/405B-parameter ones. This run uses one mid-size model and a 1-2 instance sample.
- **`b1`/`sh_rlm` are the loop's sparse starting point**, not a finished system — the whole point of
  the Self-Harness optimization loop is to improve on `b1` over many rounds; this smoke ran exactly
  one, heavily shrunk for cost.
- **Exact-match verification, no partial credit for pass/fail.** `wrong_format` in particular means
  no answer was extractable in the required syntax, a stricter bar than "reasoned incorrectly."
- **Sample size 1-2 per condition** has no averaging protection against an unlucky (or lucky) draw.

## Caveats / known issues

- **OOLONG-Pairs long-instance sampling can draw pathological outliers.** Two consecutive random
  seeds both drew instances with tens of thousands of required gold pairs. A smoke-only
  gold-pair-cardinality preflight (reject and refuse to spend before any live call) was drafted
  during this session but **is not included in this run's code state** — it was reverted along with
  a set of prompt/format fixes so this run's results reflect the baseline pipeline behavior only.
  Worth adding before the next live run.
- **The `wrong_format` failure mode has a plausible root cause but no confirmed/shipped fix in this
  run.** `shrlm/environments/oolong_pairs.py`'s `build_prompt()` appends the task instruction after
  only a blank line, with no explicit structural marker separating "data to classify" from "the
  actual question" — a plausible explanation for why models (independently, across both the
  harness-REPL and λ-RLM paths) sometimes fold the instruction into the data being processed. A
  candidate fix (explicit final-answer format instructions + an empty-answer recovery nudge) was
  drafted and spot-tested; it eliminated the specific `wrong_format` failure on a retest but
  surfaced a different failure mode (the model looping without ever committing to a final answer,
  timing out at the soft 300s per-run budget instead). Not conclusive with one data point — needs a
  broader retest.
- **This run does not include those candidate prompt fixes.** It reflects only the hard-deadline
  fix described above.
