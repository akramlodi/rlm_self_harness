# Post-mortem: Kimi-K2.5 Self-Harness run (`experiment_kimi`)

Experiment identity `bf56cc0e…`, config `configs/experiment_kimiK25.toml`, all three roles
`azure_foundry / Kimi-K2.5` in default thinking mode. Five optimization rounds, stopped on
patience=3. 5,104 runs, $124.82, 33.5 h wall (13.9 h of it a restart loop, see §9).
Analysis snapshot: `analysis/20260827T220346Z/`.

Every number below was recomputed from the artifacts under `opt/` and `analysis/`; the
scripts are described in Appendix C so the figures can be regenerated.

---

## 0. TL;DR

The loop did not improve the harness because the signal it optimized was not task
capability. Five findings, in order of how much they explain:

1. **The pass-rate signal was a serialization artifact.** Ignoring quote characters and
   the literal `Final Answer:` prefix, Kimi-K2.5 produced the *exact gold node set* in
   86–91% of runs under every harness tested — including the untouched H0 baseline
   (0.91). The measured pass rate (0.375 → 0.646 held-in) tracks only whether those bytes
   crossed the verifier's regex. The single promotion (`r02-c03-s4`) traded one string
   bug (`['id']`) for another (missing `Final Answer:` prefix); correct answers kept
   failing.
2. **The promotion gate is noise-dominated.** With v=4, the pass-count delta between
   two measurements of the *same* harness has SD ≈ 6.3 (held-in, of 96) and ≈ 7.9
   (held-out, of 160). A candidate byte-identical to the incumbent is rejected by the
   preregistered `tau=0` rule **72%** of the time. The incumbent, re-measured in rounds
   3–5, scored 58, 64, 51 held-in passes.
3. **The evidence chain fed the attributor the wrong question.** `instances.jsonl`
   `question` is the prompt's worked *example* ("Perform a BFS from node abcd") for
   100% of instances — a `_QUESTION_RE` first-match bug. 49 of 120 attribution records
   (41%) blame the model for "hardcoding the wrong node" when the model used the node
   the real operation asked for. The proposer then chased a phantom.
4. **The proposer was funnelled into surfaces that could not fix the real bug.**
   Surface choice is hard-wired by mechanism (`MECHANISM_SURFACE`). In a regime with
   zero sub-calls, S3/S5/S7/S9/S10 are unreachable by construction; the loop spent 12 of
   14 candidates on S8 (×7) and S2 (×5), zero promotions. S9 — the surface designed for
   programmatic answer normalization, i.e. the actual fix — was never reachable.
5. **S8 was dead on arrival.** Helpers are advertised to the model as
   ``- `format_id_list`: A custom function`` (docstring dropped by
   `rlm/environments/base_env.py:format_tools_for_prompt`). The model called the helper
   in 2 of 96 runs. Every S8 candidate also inflated input tokens ~60%, tripping the
   cost band.

Underneath all of it: **the recursive language model never recursed.** 0 of 240 mining
runs and 0 of 4,864 validation runs issued an `rlm_query` or `llm_query` — except one
S2 candidate that forced 0.07 sub-calls/run and regressed. GraphWalks-short fits in the
REPL and is solved by a 10-line Python BFS. The failure taxonomy is a taxonomy of
decomposition failures applied to a task with no decomposition.

---

## 1. What was run

| | |
|---|---|
| Environment | GraphWalks ≤128k chars, `bfs` + `parents`, seed 0 |
| Splits | held-in 24 (mined + validated), held-out 40 (validated only) |
| Loop | m=2 mining reps → 48 mining runs/round; v=4 → 256 validation runs/subject; k≤4 |
| Promotion rule | pass-count delta ≥ 0 on **both** splits, > 0 on at least one; cost within [0.5, 1.25]× |
| Rounds | t=5, patience=3 |
| Runner/attributor/proposer | Kimi-K2.5, thinking on, temp 0.6, max_output 8192 |
| Prompt sizes | 2.7k – 110k chars, median 28k; gold sets 1–181 nodes |

Cost by stage (from `stage_usage.jsonl`):

| Stage | Cost | Share | Wall |
|---|---:|---:|---:|
| validation | $116.66 | 93.5% | 17.2 h |
| mining | $5.56 | 4.5% | 1.3 h |
| attribution | $2.46 | 2.0% | 14.9 h (13.9 h = round-5 restart loop) |
| proposal | $0.14 | 0.1% | 4 min |
| **total** | **$124.82** | | **33.5 h** |

The stage that decides everything (proposal) consumed 0.1% of spend and saw the least
information.

---

## 2. Headline results

### 2.1 Incumbent trajectory (as the analysis reports it vs. as measured)

| Round | `incumbent_quality.csv` held-in / held-out | Baseline actually re-measured that round (held-in/96, held-out/160) |
|---|---|---|
| 1 | 0.375 / 0.513 | 36 / 82 |
| 2 | **0.646 / 0.656** (promotion) | 32 / 86 (same H0 harness as round 1) |
| 3 | 0.646 / 0.656 (carried flat) | **58 / 113** |
| 4 | 0.646 / 0.656 (carried flat) | **64 / 103** |
| 5 | 0.646 / 0.656 (carried flat) | **51 / 106** |

`incumbent_quality.csv` carries the promotion-time value forward by design
(`shrlm/docs/SH-RLM Metrics & Graphs — Reference.md` §"incumbent_quality"). The
right-hand column is the same harness measured three more times: a 13-pass spread
on held-in. The figure the analysis draws is smoother than the experiment.

### 2.2 All 14 candidates

"Set-correct" = fraction of runs whose bracketed node set equals the gold set once
quotes and the `Final Answer:` prefix are ignored (Appendix C). It is what the model
actually computed.

| Cand. | Surface | Held-in Δ (of 96) | Held-out Δ (of 160) | Cost × | Measured pass | **Set-correct** | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| baseline r1 (H0) | — | 36 | 82 | 1.00 | 0.46 | **0.91** | |
| r01-c01-s2 | S2 | −6 | −10 | 1.47 | 0.40 | 0.88 | rejected |
| r01-c02-s8 | S8 | −2 | −11 | 1.08 | 0.41 | 0.89 | rejected |
| r01-c03-s1 | S1 | −15 | −36 | 1.22 | 0.26 | 0.79 | rejected |
| baseline r2 (H0) | — | 32 | 86 | 1.00 | 0.46 | 0.91 | |
| r02-c01-s8 | S8 | −5 | −15 | 1.16 | 0.38 | 0.83 | rejected |
| r02-c02-s2 | S2 | −10 | −3 | 1.04 | 0.41 | 0.86 | rejected |
| **r02-c03-s4** | S4 | **+30** | **+19** | 1.10 | 0.65 | 0.88 | **promoted** |
| baseline r3 (S4) | — | 58 | 113 | 1.00 | 0.67 | 0.90 | |
| r03-c01-s8 | S8 | −8 | −26 | 1.43 | 0.54 | 0.78 | rejected |
| r03-c02-s8 | S8 | −5 | −19 | 1.18 | 0.57 | 0.80 | rejected |
| r03-c03-s8 | S8 | −8 | −18 | 1.50 | 0.57 | 0.78 | rejected |
| r03-c04-s1 | S1 | **−58** | **−112** | 1.18 | **0.00** | 0.69 | rejected |
| baseline r4 (S4) | — | 64 | 103 | 1.00 | 0.65 | 0.86 | |
| r04-c01-s2 | S2 | −9 | −14 | 1.59 | 0.56 | 0.82 | rejected |
| r04-c02-s8 | S8 | −16 | −2 | 1.47 | 0.58 | 0.84 | rejected |
| baseline r5 (S4) | — | 51 | 106 | 1.00 | 0.61 | 0.86 | |
| r05-c01-s2 | S2 | +8 | −3 | 1.50 | 0.63 | 0.86 | rejected (held-out −3, cost) |
| r05-c02-s8 | S8 | −4 | −17 | 1.56 | 0.53 | 0.80 | rejected |

Read the two right-hand columns together. Under the metric the verifier reports, the
promotion was a +30/+19 event. Under what the model computed, H0 (0.91) was already
the best harness in the table and nothing that followed beat it.

### 2.3 Surface coverage

| Surface | Attempted | Promoted | Reachable in this regime? |
|---|---:|---:|---|
| S1 repl_contract | 2 | 0 | yes (`repl_contract_misuse`) |
| S2 decomposition | 5 | 0 | yes (`whole_input_subcall_collapse`) |
| S3 execution | 0 | 0 | **no** — only via depth/recursion mechanisms |
| S4 verification | 1 | 1 | yes (`skipped_verification`, `premature_termination`) |
| S5 recovery | 0 | 0 | **no** — only via `swallowed_subcall_error` |
| S6 runtime_policy | 0 | 0 | yes (`iteration_budget_exhaustion`), never proposed |
| S7 metadata | 0 | 0 | **no** — only via `unparsed_child_output` |
| S8 repl_helpers | 7 | 0 | yes (`repl_execution_fault`) |
| S9 answer_middleware | 0 | 0 | **no** — only via `lossy_aggregation` |
| S10 skills | 0 | 0 | **no** — only via `unconsulted_procedure` |

Six of ten surfaces were structurally unreachable given that no run ever made a
sub-call. Of the four reachable ones, the loop spent 12 of 14 candidates on the two
(S8, S2) that could not address the actual failure.

---

## 3. Finding 1 — the signal was a serialization artifact

### 3.1 The verifier's contract

`shrlm/environments/graphwalks.py:GraphWalksVerifier`: the **last line** of the
produced string must contain `Final Answer:` and a bracket pair; items are split on
commas and stripped of whitespace and brackets — **not quotes**. The RLM's final string
is `str(answer["content"])` (`rlm/environments/local_repl.py:772`). So:

| Model sets `answer["content"]` to | Response last line | Verdict |
|---|---|---|
| `"Final Answer: [a, b]"` | `Final Answer: [a, b]` | pass |
| `"Final Answer: ['a', 'b']"` | `Final Answer: ['a', 'b']` | `mixed_set_error` (every item extra *and* missing) |
| `["a", "b"]` (a list object) | `['a', 'b']` | `wrong_format` (no marker) |
| `"[a, b]"` | `[a, b]` | `wrong_format` (no marker) |

The harness (S1–S5) never states this contract. The model infers it from the task
prompt's own instruction and complies inconsistently. Nothing in the loop — not the
digest, not the pattern block, not the proposer prompt — shows the verifier's
extraction rule.

### 3.2 What the failures actually were

Across all 5,104 runs, failures whose bracketed set equals gold once quotes/prefix are
ignored:

| Subject (round) | Measured pass | Failures that were set-correct | "True" set-correct rate |
|---|---:|---:|---:|
| baseline H0 (r1) | 118/256 | **116** of 138 failures | 0.91 |
| baseline H0 (r2) | 118/256 | 114 of 138 | 0.91 |
| promoted S4 (r3 baseline) | 171/256 | 60 of 85 | 0.90 |
| promoted S4 (r4 baseline) | 167/256 | 52 of 89 | 0.86 |
| promoted S4 (r5 baseline) | 157/256 | 64 of 99 | 0.86 |
| r03-c04-s1 (the 0/96 candidate) | 1/256 | **175** of 255 | 0.69 |

Under H0, 116 of 138 failures (84%) were correct answers wearing quotes. Under the
promoted S4 harness the quote failures fell (105 → 30 on held-in+out) but
`wrong_format` rose (18 → 42): the model, told to "use the format [id1, id2] without
quotes", now often emitted exactly `[id1, id2]` with no `Final Answer:` prefix. Verifier
evidence in the round-3/4/5 bundles reads, literally:

```
bfs-4f5aebdd34c0be52:      expected '[1f0e3dad99]',              produced '[1f0e3dad99]'
bfs-2b1afb378791d043:      expected '[2c7f9ccb5a]',              produced '[2c7f9ccb5a]'
parents-672402b661de6f18:  expected '[7f6ffaa6bb, 8e296a067a]',  produced '[8e296a067a, 7f6ffaa6bb]'
```

These were shown to the proposer as `repl_contract_misuse` / `wrong_format` patterns.
The proposer's response (`r03-c04-s1`) told the model to set `answer["content"]` to a
string like `"[id1, id2]"` — which guarantees no `Final Answer:` prefix — and scored
0/96 and 1/160 while being set-correct in 175 of 256 runs. The loop's most catastrophic
candidate was a correct diagnosis of the symptom implemented against a contract it
could not see.

### 3.3 Consequence

The experiment measured **byte compliance with an unstated regex**, and the loop
optimized it. The "+30 held-in passes" of the promotion is real under that metric and
irrelevant to the question the paper asks. A one-line S9 middleware (`accept_answer`
already exists as an identity function) that normalizes the last line to
`Final Answer: [sorted, unquoted, ids]` would have moved every harness in the table to
≈0.86–0.91 measured pass rate before the first round ran.

---

## 4. Finding 2 — the promotion gate cannot distinguish candidates from noise

### 4.1 Instance-level variance

Same harness, rounds 3–5, per-instance pass rate over v=4 (held-in shown):

```
bfs-371b9355b377782d   0.25  1.00  0.75
bfs-a454cef56a3a9fbb   0.25  1.00  0.25
parents-57d125882fbbefee 1.00 1.00  0.25
parents-c1d49383a40d08e1 0.75 1.00  0.25
parents-7cfd78c8d93d5bfc 1.00 0.25  0.50
```

Of 64 instances: 2 always pass, 0 always fail, **27 swing by ≥ 0.50** between rounds
with no harness change. Each instance is a coin with p ≈ 0.6 (held-in) / 0.67
(held-out); v=4 flips per instance resolve nothing.

### 4.2 Simulation against the preregistered rule

Drawing two independent 96/160-run measurements from the observed per-instance rates
(20,000 trials):

| | |
|---|---|
| SD of held-in pass-count delta | 6.3 passes |
| SD of held-out pass-count delta | 7.9 passes |
| P(byte-identical candidate is **rejected** by `tau_regression=0` on either split) | **0.72** |
| True per-instance improvement needed for P(promoted) ≥ 0.9 | ≈ +0.10 absolute |

Twelve of thirteen rejections cite regression on at least one split. Several are
within one SD of zero (`r04-c02-s8` held-out −2; `r05-c01-s2` held-out −3 with held-in
+8). The rule is not wrong — strict non-regression is the Zhang et al. rule the config
cites — but at v=4 on 24+40 instances it is a noise detector with a 72% false-reject
rate and, symmetrically, would promote a null edit 28% of the time. The one promotion
(+30 / +19, ≈4σ / 2.4σ) is the only decision in the run that clears the noise floor,
and §3 explains why it is large.

### 4.3 Reporting

`incumbent_quality.csv` reports 0.646 / 0.656 for rounds 3–5 because it carries the
promotion value forward. The re-measured baselines were 0.604 / 0.706, 0.667 / 0.644,
0.531 / 0.663. The analysis should report the re-measurement (it is on disk in every
round's `promotions.jsonl` `rule.*.baseline_pass_count`) and draw the error bar.

---

## 5. Finding 3 — the attributor was told the wrong question

### 5.1 The bug

`shrlm/environments/graphwalks.py`:

```python
_QUESTION_RE = re.compile(r"^\s*(Perform a BFS\b.*|Find the parents\b.*)$", re.MULTILINE)
...
match = _QUESTION_RE.search(prompt)   # first match
```

Every GraphWalks prompt opens with a worked example whose first operation line is
`Perform a BFS from node abcd with depth 1.` The real operation is the *last*
`Operation:` block. Result across held-in (24), held-out (40), and every round:

```
distinct `question` values across 24 held-in instances:
  {'Perform a BFS from node abcd with depth 1.': 24}
question != real Operation line: 24 / 24   (11 of them are `parents` tasks)
```

The digest handed to the attributor (`## Run / question:`) therefore said "BFS from
abcd" for every failure, including parents lookups.

### 5.2 Effect on attribution

Attribution records whose symptom summary names the phantom node or blames a
"wrong/hardcoded start node":

| Round | Records | Mention `abcd` | "wrong/hardcoded node" |
|---|---:|---:|---:|
| 1 | 36 | 19 | 20 |
| 2 | 34 | 12 | 13 |
| 3 | 15 | 7 | 7 |
| 4 | 19 | 5 | 7 |
| 5 | 16 | 6 | 6 |
| **all** | **120** | **49 (41%)** | **53 (44%)** |

Representative (round 1, `bfs-4e79e2d22c5c1630`): *"The root extracted the wrong
start node identifier (17e62166fc instead of abcd)"*. The real operation was `BFS from
17e62166fc`; gold = the three children of 17e62166fc. The model read the task
correctly. What it actually did wrong in that trace — `context.split("The graph has
the following edges:")[1].split("Operation:")[0]` picked the *example* sub-graph
because the prompt reuses its delimiters — went unnamed.

These mislabelled records were clustered under `whole_input_subcall_collapse`,
`repl_contract_misuse`, and `other`, and drove the S2 and S1 proposals. Four of five
rounds' S2 candidates ("decompose instead of hardcoding") were responses to a failure
that did not exist.

### 5.3 Two further attribution-integrity problems

- **`failing_level=root` with zero sub-calls: 38 of 120 records (32%).** The taxonomy
  defines `root` as "every sub-call returned a correct local result"; with no
  sub-calls it is a category error. `grounding.derive_failing_level` would return
  `NO_RECURSION` deterministically, but `level_grounded` is `False` on all 120 records,
  so the LM chose freely and the validator accepted it. Because `failing_level` is part
  of the signature, this split otherwise-identical patterns (e.g.
  `repl_execution_fault/mixed_set_error` appears as both `root` and `no_recursion`),
  halving supports and pushing patterns under the floor.
- **`other` is unaddressable.** `MECHANISM_SURFACE` has no entry for
  `AgentMechanism.OTHER`, so `_addressable_patterns` drops it. In round 2 the largest
  cluster (support 10, `other/root/mixed_set_error` — mostly quoted-list failures) was
  invisible to the proposer.

### 5.4 Support

`min_support = 2` against 48 runs and 15–19 signatures per round means most patterns
are singletons or pairs. `pattern_frequency_diff_*` marks 60–80% of rows
`below_support_floor`. Round-over-round "resolved"/"new" statuses at 1/48 are coin
flips, and the diffs after round 2 are churn, not trend.

---

## 6. Finding 4 — the proposer could only reach the wrong surfaces

`shrlm/optimization/proposal.py` states the constraint: *"already assigned to the ONE
harness surface its mechanism implicates — you may not choose a different surface for
a pattern."* With every failure labelled by one of four mechanisms
(`repl_execution_fault → S8`, `whole_input_subcall_collapse → S2`,
`repl_contract_misuse → S1`, `iteration_budget_exhaustion → S6`), plus one
`skipped_verification → S4`, the reachable set was {S1, S2, S4, S6, S8}. The proposer
never selected S6.

What the proposer saw per pattern (`_render_pattern_block`): signature, mechanism
doc, support, `shared_symptoms`, truncated `verifier_evidence`, representative ids,
current surface text. Two problems:

- `shared_symptoms` was identical for every pattern in every round
  (`"median sub-calls per run: 0" … "no sub-calls issued at all in n/n runs"`). It
  carried no discriminating information.
- `verifier_evidence` is truncated at 2,000 chars at render time. Gold sets run to
  181 nodes; the `expected '[…]'` list alone can exceed the cap, so for the largest
  instances the proposer never saw the `produced` side — the one string that shows the
  quotes.

The proposer did see prior-round history with rejection reasons, and still proposed
the same list-formatting helper four times (`format_id_list`, `format_id_list`,
`submit_id_list`, `format_answer_list`) because it was the only edit the
`repl_execution_fault → S8` mapping permitted. The S2 edits ("you must decompose via
`rlm_query`") were the only permitted response to `whole_input_subcall_collapse`;
each forced sub-calls onto a task that fits in the REPL and each regressed.

k=4 was never binding. Candidate counts (3, 3, 4, 2, 2) were set by how many
addressable, above-floor patterns existed, not by the cap.

---

## 7. Finding 5 — S8 helpers were never usable

`rlm/environments/base_env.py:format_tools_for_prompt` renders a callable without a
`description` as ``- `name`: A custom function``. The harness passes S8 helpers as
bare callables; the docstring the proposer wrote is never shown. Verified in the
`r01-c02-s8` traces:

```
6. Custom tools and data available in the REPL:
- `format_id_list`: A custom function
```

Across 96 held-in runs: mentioned in the prompt of 94, **called in 2**. The same holds
for all seven S8 candidates. Their measured regressions come from two sources: (a) the
helper is inert, so pass rate is baseline ± noise; (b) input tokens rose ~60% (25.9k →
39–48k per run) — the tool line and/or the extra exploration it prompts — pushing mean
cost to 1.4–1.6× and tripping the 1.25 band independently of pass counts. Five of
seven S8 rejections cite cost.

---

## 8. Finding 6 — the RLM never recursed, and the task did not need it

| Metric | Value |
|---|---|
| Mining runs (5 rounds) using `rlm_query`/`llm_query` in any code block | **0 / 240** |
| Validation `mean_sub_calls` for baseline, every round | 0.0 |
| Only candidate with non-zero sub-calls | `r01-c01-s2`: 0.074/run, −6/−10, cost 1.47× |
| `no_recursion_failure_share` (`collapse_and_attribution_optimization.csv`) | 1.0 every round |
| `n_grounded` | 0 every round (nothing to sub-verify) |

Median run: 5 iterations, 5 code blocks, 12–20k input tokens, ~1.8k output tokens,
$0.02. The model parses edges with a regex, runs BFS/parents in Python, formats an
answer. Prompts top out at 110k chars — inside the context window — so decomposition
has nothing to buy. Every S2 edit that pushed the model toward sub-calls cost more and
scored lower. The environment is a poor test of a *recursive* harness; the taxonomy
(`incomplete_coverage`, `depth_degradation`, `lossy_aggregation`, …) has nothing to
label, and the attributor fell back on `whole_input_subcall_collapse` (whose doc
includes "or performed no meaningful decomposition at all") as a catch-all — 28 of
120 records — which routed to S2.

---

## 9. Finding 7 — model–harness incompatibilities the taxonomy cannot name

### 9.1 Native tool-call leakage

Kimi-K2.5 sometimes emits its function-calling tokens instead of a fenced block:

```
<|tool_calls_section_begin|><|tool_call_begin|>functions.repl:8<|tool_call_argument_begin|>{"code": "..."}
```

Nothing executes; the iteration is spent; the model repeats it. Mining: **14 of 240
runs, 0% pass**, all reaching the 31-iteration cap. The fallback then synthesizes an
answer from nothing, and the verifier evidence shows hallucinated *other tasks*:

> "Based on my analysis of the context containing 10 papers …"
> "Let me extract the Trivia Game questions …"
> "identified all documents mentioning 'quantum' …"

Validation: 2–8 such runs per 256-run subject. These are labelled
`iteration_budget_exhaustion → S6` and were never proposed on; S6 (a numeric policy
table) could not fix a tokenizer-level behaviour anyway. Iterations that produced no
code block at all: 31, 107, 92, 64, 93 per round (of ~250–300).

### 9.2 Delimiter collision with the in-prompt example

74 of 240 mining runs split the context on the **first** `Operation:` /
`The graph has the following edges:` occurrence, i.e. on the worked example. Many
recover in a later turn; the aggregate pass rate of those runs (0.55) is not below the
rest (0.48), so this is a hazard rather than the dominant cause — but it is the true
cause in the trace the attributor mislabelled in §5.2, and the kind of concrete,
reusable procedure an S10 skill exists to encode. S10 was unreachable.

### 9.3 Content filter

Round 5 stalled for 13.9 h and 102 restarts because one attribution digest was
deterministically refused by Azure's content filter and the round-close gate treated
it as a retryable transport failure. Fixed on 2026-08-27
(`AttributionErrorKind.CONTENT_FILTERED`, contained not fatal); documented here
because it is why round-5 attribution shows 50,200 s wall.

---

## 10. Why the S4 promotion happened, exactly

`r02-c03-s4` appended one sentence to S4: *"Verify that list outputs use the format
[id1, id2] without quotes around individual items."* It targeted a **singleton**
pattern (`skipped_verification`, support 1, below floor). It won because the sentence
happens to describe the verifier's item-parsing rule, which no other surface text
does. Held-in+out `mixed_set_error` fell 105 → 29; `wrong_format` rose 23 → 48;
set-correct rate was unchanged (0.91 → 0.88). It is the loop's only decision above
the noise floor and it is a documentation fix for an undocumented contract.

Rounds 3–5 then had no comparable lever: the remaining failures were the missing
prefix (`repl_contract_misuse → S1`, where the one attempt was catastrophic because
the proposer could not see the prefix requirement), quotes the S4 sentence hadn't
suppressed (`→ S8`, inert), and tool-call leakage (`→ S6`, never chosen).

---

## 11. What would have worked

Ordered by evidence, not by effort.

1. **Normalize the answer channel once, programmatically.** S9 `accept_answer`
   already receives the produced string. Rewrite the last line to
   `Final Answer: [sorted, deduplicated, quote-stripped ids]`. Expected effect from
   §3.2: every harness in the table moves to ≈0.86–0.91 measured; the H0 → S4 delta
   disappears. Alternatively state the contract in S1. Either way the loop should start
   from a harness whose measured pass rate reflects capability.
2. **Fix `extract_question`** to take the last `Operation:` block. One-line change;
   removes 41% of attribution text being about a phantom node.
3. **Make the promotion decision statistically honest.** Options, cheapest first:
   report the re-measured baseline and a CI; set `tau_regression` to ≥ 1 SD (≈ 6–8
   passes at v=4) or, better, compare paired per-instance outcomes; raise v (the
   per-instance p ≈ 0.6 needs ≈ 16 reps to resolve a 10-point effect at 80% power
   per instance); or run the baseline and candidates in the same batch so provider
   drift is shared.
4. **Ground `failing_level` from the tree.** `derive_failing_level` already returns
   `NO_RECURSION` for zero descendants; use it whenever the tree has no descendants,
   regardless of sub-verifier availability. Removes the 32% `root`-with-no-sub-calls
   split.
5. **Let the proposer see the contract and the produced string.** Add the verifier's
   extraction rule to the pattern block; show `produced` before `expected` and
   truncate each side separately so a 181-node gold list cannot hide a 20-char
   produced string.
6. **Fix `format_tools_for_prompt`** to render the helper's docstring, or have the
   harness pass `{"tool": fn, "description": doc}`. Until then every S8 edit is inert
   and should be reported as such rather than validated at $5 each.
7. **Decouple surface from mechanism, or map more mechanisms to more surfaces.**
   At minimum give `OTHER` a surface, let `repl_contract_misuse` reach S9, and let
   `iteration_budget_exhaustion` reach S1/S3 (where "emit a ```repl``` block, not a
   tool call" belongs).
8. **Handle native function-calling.** Either disable it on the Kimi deployment
   (`tool_choice: "none"` / no tools declared) or detect `<|tool_call` in a response
   and re-prompt. This is a runner fix, not a harness edit.
9. **Pick an environment where recursion is load-bearing.** GraphWalks-long
   (256k–1M chars) or OOLONG-Pairs at 262k tokens would at least produce sub-calls for
   the taxonomy to label. On the short split the "recursive" harness is a Python REPL
   with an unused function.

---

## 12. What this run does establish

- The persist-first, identity-hashed, replayable design held: 5,104 runs, three
  reboots, one 100-restart loop, one source-level containment patch mid-run, and every
  artifact is byte-accounted in `provenance.json`. That is a real result for the
  infrastructure.
- Kimi-K2.5 solves GraphWalks-short in-REPL with the exact gold set ≈ 90% of the time
  under a five-line harness. That number is the baseline the paper should quote, with
  the serialization failures reported separately as a verifier-interface effect.
- The loop can promote a real edit when the effect is ≈ 4σ (the S4 sentence). It
  cannot see effects of ≈ 1σ, which is where every other candidate lived.

---

## Appendix A — per-round mining summary

| Round | Failures / 48 | Mechanisms (attributor) | Verifier causes | Patterns (≥ floor) |
|---|---:|---|---|---:|
| 1 | 36 | collapse 10, other 10, exec_fault 8, contract 7, iter 1 | mixed 31, wrong_format 4, no_answer 1 | 12 (7) |
| 2 | 34 | other 13, exec_fault 8, collapse 6, contract 4, iter 2, skipped_verif 1 | mixed 27, wrong_format 5, no_answer 1, spurious 1 | 15 (6) |
| 3 | 15 | exec_fault 6, contract 4, collapse 2, iter 2, other 1 | mixed 7, wrong_format 5, no_answer 2, spurious 1 | 11 (4) |
| 4 | 19 | collapse 7, exec_fault 5, other 4, iter 2, contract 1 | mixed 10, wrong_format 9 | 12 (5) |
| 5 | 16 (1 unattributed: content filter) | exec_fault 4, collapse 3, iter 3, contract 3, other 2 | wrong_format 10, mixed 4, no_answer 1 | 9 (5) |

Instances failing in every round's mining under both harnesses: `parents-8f10703b6d7797b8`,
`bfs-4e79e2d22c5c1630`, `bfs-371b9355b377782d` — all three are format/prefix failures
on correct sets in at least one round.

## Appendix B — the promoted diff

```
S4_verification_instruction
- Check your candidate answer before setting `answer["ready"] = True`.
+ Check your candidate answer before setting `answer["ready"] = True`. Verify that list outputs use the format [id1, id2] without quotes around individual items.
```

Frozen harness: `sh_rlm/harness.json` (`f50714ced3a06982…`). All other surfaces
identical to H0.

## Appendix C — how the numbers were computed

All from the on-disk artifacts; no re-execution.

- **Measured pass counts, deltas, cost bands**: `opt/round_0N/validation/round_0N/promotions.jsonl`
  (`rule.heldin/heldout.*`, `band.mean_cost`).
- **Set-correct**: for every run in `opt/round_0N/{mining,validation}/…/runs.jsonl`
  with `passed=false` and cause in {`wrong_format`, `mixed_set_error`, `spurious`,
  `incomplete`}, take `verdict.produced`, find the last `[...]` group, split on
  commas, strip whitespace and the characters `'"[]` from each item, compare the set to
  `verdict.gold` parsed the same way. A run is "set-correct" if equal. (For
  `wrong_format`, `produced` is the raw response; for the others it is
  `serialize_nodes(pred_set)`, in which quote characters survive inside ids.)
- **Incumbent noise**: per-(split, instance) pass rate over the v=4 baseline runs in
  rounds 3, 4, 5; "swing" = max − min ≥ 0.5.
- **Simulation**: per-instance p pooled over rounds 3–5 baselines; two independent
  draws of 4 Bernoulli trials per instance for each split; 20,000 trials; the rule as
  written in `shrlm/optimization/promotion.py` (`delta < -tau_regression` on either
  split rejects; needs `delta > tau_improvement` on at least one to promote).
- **Question bug**: compare `instances.jsonl[].question` to the last
  `Operation:\n<line>` in `instances.jsonl[].prompt`.
- **Attributor mislabels**: regex over `records.jsonl[].detail.symptom_summary` (+
  `agent_mechanism_detail`) for `abcd`, and for
  `wrong|incorrect (start|target)? node|hardcod`.
- **Trace behaviour**: `opt/round_0N/mining/round_0N/runs/*.json` →
  `metadata.iterations[].code_blocks[].code` for `rlm_query|llm_query` calls and for
  `split("Operation:")[0]`-style first-delimiter splits; `iterations[].response` for
  `<|tool_call`; `usage_summary` for tokens.
- **S8 advertisement**: `r01-c02-s8/heldin/round_00/runs/*.json` grep for
  `format_id_list` in the system prompt vs. `format_id_list(` in code.
- **Cost**: `stage_usage.jsonl` summed by (`round_index`, `stage`).
