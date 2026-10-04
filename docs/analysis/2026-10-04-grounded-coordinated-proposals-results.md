# Grounded and coordinated proposals: implementation verification

Date: 2026-10-04. Branch: `feat/grounded-coordinated-proposals`. Base: `e323f4328e0aa37b2cb5ff67cabd4729d1b0916c`.

The [plan](../plans/2026-10-04-1144-feat-grounded-coordinated-proposals-plan.md) implements four changes: forward concrete diagnosis details, admit an optional atomic capability/caller pair, express benefits/protections as task-condition patterns with relevant passing evidence, and select evidence using a defect-resolution assessment. The [interface](../harness-proposal-interface.md) and [collaborator summary](../../meta-harness-improvements.md) describe the resulting behavior. Validation samples, repeats, gates, and experiment configuration are unchanged; no regression stage was added.

## Deterministic verification

The tests exercise saved mining artifacts through the final rendered proposer prompt; all-or-neither pair admission through materialization, repair, publication, frozen replay, disk loading, and combined validation; compact intent/history; and actual harness execution. Red tests first reproduced dropped diagnosis detail, rejected pairs/orphan survival, dropped protection history, omitted contrast, and missing resolution selection.

Two constructed task families use real materializers and the actual RLM/REPL runner with scripted model responses. The occurrence-counting family adds an S8 helper plus S3 caller; the unit-normalization family adds an S10 procedure plus S3 caller. The script deliberately gives the baseline an information-losing computation. Inputs and independent expected outputs are authored fixtures.

| Variant, in each family | Target-pattern input | Protected-pattern input | Capability activation |
| --- | --- | --- | --- |
| Baseline | Fails | Passes | None |
| Capability only | Fails | Passes | None |
| Caller only | Fails | Fails | Unavailable capability |
| Paired | Passes | Passes | Helper invoked / skill loaded and computation correct |

Both paired harnesses also pass renamed, reordered target inputs. These controls establish execution wiring and the constructed invariant, not discovery quality or benchmark generalization. The standalone offline probe made **zero provider requests**. The test module adds the renamed variants to its execution coverage.

The fake validation-runner check observes only the incumbent and the combined admitted pair on the existing held-out samples. Removing a partner excludes the pair before any validation run. Existing literal-text, route, collision, repair, replay, and legacy singleton tests remain applicable.

## One bounded live probe

Command:

```bash
uv run python -m examples.meta_harness_probe --live \
  --out-dir /tmp/rlm-meta-harness-probe-20261004-live
```

Azure DeepSeek-V4-Flash used the configured runner settings with a 4,096-token output cap, three RLM iterations, depth one, and 90-second run limit. The shared guard reserved every synchronous/asynchronous SDK request, including retries/nested calls, before dispatch; it capped requests at 32 and conservative configured-price reservations at $2. No failed observation was resampled.

Eight executions covered four harness variants on one target and one protected occurrence-counting input. **All eight answers were correct; none invoked the supplied helper.** The paired system prompt contained both its advertised helper and the explicit caller instruction, but the live model used its own computation. Thus correctness was already saturated on these trivial cases, and live activation was not demonstrated.

One additional free-choice proposer request produced an admitted **S3 singleton**, not a pair. A pair was optional and a direct fix was reasonable for this operation. This is one synthetic observation, not a pair-generation rate or evidence of poorer proposal quality.

Accounting: **18 SDK requests**, **$0.00411348** estimated from reported usage at configured input/output rates, and **$0.05647512** reserved ceiling. The estimate is not an Azure billing receipt. There were no live diagnosis requests and no benchmark experiment launch. S10 execution was tested offline only.

## Outputs and checks

- Offline report and execution trees: `/tmp/rlm-meta-harness-probe-20261004-offline/`.
- Live report and execution trees: `/tmp/rlm-meta-harness-probe-20261004-live/`.
- Each execution saves `trace.json`, `result.json`, and runner `logs/`; reports include harness hashes, expected/actual output, activation, usage, and error status.
- Supplied proposals and their materialization audits are under each family's `proposals/` and `work/`. The live free-choice response, saved observations, proposal, and audit are under `free-choice/`.
- Temporary output directories are local verification artifacts; this committed report preserves the conclusions and accounting. Re-running requires a new directory.

Focused checks during development passed 178 attribution/evidence/history/type/candidate tests and 276 proposal/batch/analysis/reasoning tests. The constructed-harness test module passed all three tests. Final suite and review results are recorded below when complete.

Repository-wide Ruff currently reports 24 issues in the pre-existing Self-Harness clone, example fixtures, and training code. Repository-wide type checking reports existing missing optional imports and typing debt; new probe type-narrowing issues were fixed. These checks do not authorize rewriting unrelated files. Final changed-file checks and the review distinguish introduced defects from those pre-existing diagnostics.

## Interpretation and paper follow-up

The implementation removes specific information/admission barriers and preserves the existing validation policy. It does not establish that this model will consistently invent useful capabilities, invoke them, or generalize better. The live missed activation is an empirical limitation worth monitoring in the next separately authorized experiment. Keep structural admission, observed activation, semantic correctness, and measured benchmark improvement separate.

Document the four method changes briefly in the eventual implementation appendix; avoid claiming a regression bank, guaranteed protection, individual credit for batch members, or benchmark gains from these constructed tests. No LaTeX changes are part of this implementation.
