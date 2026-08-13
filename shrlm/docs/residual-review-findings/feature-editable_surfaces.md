# Residual code-review findings — `feature/editable_surfaces`

Review run `20260728-204913-cbeba795`, six reviewers (correctness, adversarial, testing, reliability, api-contract, project-standards) against base `f05aca7`.

Four findings were fixed in `b93e7f0`. The rest are recorded here rather than dropped. They are grouped by what they threaten, because roughly half are ordinary code issues and half would quietly corrupt the experiment's measurements — and the second group matters more than its severity labels suggest.

---

## A. Measurement fidelity — these corrupt ϕ or the cost gate

These are the ones to fix before the first optimization run. Every one of them makes a recorded number wrong rather than making the code crash, so nothing surfaces them at runtime; they surface at analysis time, after the compute is spent.

**A1. Batched sub-calls always record `syntax_error: False`.** (`rlm/environments/local_repl.py:471`, also `:565`, `:592` — adversarial, P2, confidence 75.) `_record_call` defaults the flag and the batched paths never pass the classification, so the same failure is counted on the single-call path and not on the batched one. Since batching is the behavior S6 exists to encourage, syntax-error rate will read artificially low exactly as a harness gets better at batching. Fix: pass `syntax_error=self._is_syntax_error(...)` at each `_record_call` in the batched paths.

**A2. Retried sub-calls are recorded once.** (`rlm/environments/local_repl.py:374-377` — adversarial, P2, 75.) `_call_with_retry` keeps only the last attempt's completion, so a retried call bills as one call. The cost-aware acceptance gate then under-prices exactly the policies that retry most, which biases promotion toward them. Fix: count attempts, either by recording one entry per attempt or by adding `trace_metrics["retries"]` into `run_metrics`' `sub_call_count`.

**A3. Trace metrics are always-on, which the "byte-identical when unconfigured" claim does not cover.** (`rlm/environments/local_repl.py:356`, `rlm/core/rlm.py:497` — api-contract, P1, 75.) An unconfigured run now serializes `trace_metrics` through `REPLResult.to_dict` and across the socket. This is arguably intended — KTD5 says the metrics are built once and consumed by mining — but the plan's hard requirement says byte-identical, and it is not. Resolve by *scoping the claim* (byte-identical **behavior**, additive telemetry) rather than by gating the metrics off, which would defeat their purpose. Decide deliberately and write it down.

**A4. `syntax_error` trace flag classifies on raw substrings with no `Error:` prefix gate**, unlike the retry classifier. (reliability, residual.) A legitimate model answer discussing a `SyntaxError` mislabels a turn. Low frequency, but it is a research metric.

---

## B. The invariant checks can pass while the invariant is violated

`shrlm/runner.py` is a verification mechanism, so a silent pass is its worst failure mode. All from the adversarial reviewer.

**B1. Policy *values* are never validated — only key names.** (`shrlm/runner.py:332`, P0, confidence 100.) `max_depth: 0` passes every check and deletes recursion outright; zero-valued caps likewise. This is the single highest-severity open finding: it is a one-value edit that removes the mechanism the experiment measures, and it validates cleanly. Fix: range-check the numeric fields (`max_depth >= 1`; caps `None` or positive int) with the same invariant-naming error style used elsewhere in the module.

**B2. The I1 boundedness probe tests one fixed namespace.** (`shrlm/runner.py:234`, P1, 75.) `probe_repl_state` always builds the same four names (`context_0`, `context`, `answer`, `buffer`). A metadata builder that branches on any other variable name — or that is stateful across calls — passes the probe and can still leak. Fix: probe several namespace shapes including an empty one and model-plausible extra names, and call twice at the same scale asserting identical output.

**B3. The redacted inventory carries model-controlled type names verbatim.** (`rlm/utils/parsing.py:74`, P1, 75.) `type(value).__name__` is copied unbounded, and a model can name a class anything. Fix: truncate the type name and bound the entry.

**B4. The stated-limits check is three exact-phrase regexes.** (`shrlm/runner.py:151`, P1, 75.) Any paraphrase of the truncation or capacity sentence slips past — and paraphrasing is precisely what an S2–S5 prompt edit does. The check therefore weakens exactly as the loop starts editing prose. Fix: scan for any numeric figure adjacent to capacity vocabulary and fail closed on unrecognized figures.

**B5. The S9 nudge is an unbounded text channel.** (`shrlm/runner.py:36`, P2, 75.) The module docstring claims S9 "has no channel through which to rewrite the root's program," but the nudge is arbitrary text injected as a user message. Either bound its length or soften the claim to what is actually enforced (inputs and return type).

---

## C. Runtime behavior

**C1. Unbounded S9 redirect loop.** (`rlm/core/rlm.py:545`, adversarial P1, 75.) A middleware that always redirects burns to `max_iterations` and falls through to `_default_answer`, which never consults the middleware. Fix: reject a redirect carrying no nudge, add a `max_redirects` counter that accepts the answer once exceeded, and surface the redirect count in `run_metrics`.

**C2. Refusal strings are indistinguishable from real answers and the model is never told the convention.** (`rlm/environments/local_repl.py:320`, reliability P1, 75.) Both batched paths return `list[str]`, so `SUBCALL_REFUSED:` is type-identical to content, and no prompt text explains it. Fix: extend `derive_capacity_sentence`'s policy-enabled branch to state the convention, leaving `DEFAULT_CAPACITY_SENTENCE` untouched so H0\* stays byte-identical.

**C3. A configured sub-output validator overwrites sub-call error text.** (`rlm/environments/local_repl.py:308`, correctness P2, 75.) A failed call's error message is replaced by the invalid marker, losing the diagnostic. Fix: return `text` unchanged when it already starts with an error or refusal prefix.

**C4. `max_prompt_chars` is enforced only on batched paths.** (`rlm/environments/local_repl.py:418`, correctness P2, 75.) `_llm_query` / `_rlm_query` skip it, while the derived capacity sentence reads as a universal per-call bound. Fix either the enforcement or the sentence.

**C5. Batched retries run serially.** (`rlm/environments/local_repl.py:489`, reliability P2, 75.) The initial pass uses a thread pool; the retry loop does not. Fix: reuse the same executor pattern.

**C6. S6 policy applies only to `environment_type == "local"`, silently.** (correctness + api-contract residual.) `build_harnessed_rlm(H, environment="docker")` passes every structural check and then enforces nothing. Either reject a non-local environment at construction or document it.

**C7. S7/S9/S6 do not propagate to child RLMs.** (correctness residual.) The `orchestrator` and `capacity_sentence` leak was fixed in `b93e7f0`; the seams themselves remain root-only. Behavior-neutral at H0 defaults, but a promoted S7/S9 edit applies only at depth 0. Decide whether that is intended and document it.

---

## D. Test coverage gaps

- No test drives more than one S9 redirect, nor the `max_iterations` → `_default_answer` terminal path.
- No test asserts a child RLM's system prompt under H0 (would have caught the orchestrator leak).
- No test asserts `patches/0001-harness-seams.patch` matches a fresh `git diff 72d6940..HEAD -- rlm/`. It went stale once already this session.
- No test covers a validator that raises, or a validator meeting an already-failed sub-call.
- No test exercises a batch large enough to exceed `max_concurrent_subcalls` together with caps or retries.
- `prepatch_format_iteration` is a hand-transcribed oracle rather than an import from the pre-patch ref; a future refactor could update both in lockstep and lose the baseline.
- `H0.runtime_policy` and the helper dicts are mutable nested fields on frozen dataclasses; no fixture guards against a test mutating them in place and leaking across the session.

---

## E. Standards tension (no action taken)

`AGENTS.md` states "fail fast, fail loud — no defensive programming or silent fallbacks" and "minimize branching." This work adds optional-kwarg branches and one `except Exception` in `build_repl_inventory`. The project-standards reviewer suppressed these as findings because both patterns have pervasive precedent in the same files (`rlm/core/rlm.py` already carries the `| None = None` idiom throughout; `local_repl.py:57` already does `except Exception: pass` for the same reason — guarding against arbitrary REPL objects). Recorded so the tension is visible rather than silently resolved.
