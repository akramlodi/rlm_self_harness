# Experiment reasoning capture: implementation and validation

Implemented the [saved-reasoning plan](../plans/2026-09-24-1450-feat-save-experiment-reasoning-plan.md) on `fix/task-agnostic-proposer`, starting at `775e6221`. See the [output guide](../experiment-reasoning.md) for paths and a reader example.

New experiments save provider-returned reasoning as immutable, hash-verified files alongside run traces and meta attempts. The runtime associates responses with individual calls before normalization or execution, including concurrent children, rejected responses, repair attempts, and cache replay. Capture uses existing provider request settings. Reasoning is inspection data: it does not enter mining evidence, proposal history, scoring, or promotion decisions. Historical artifacts are not rewritten.

## Coverage

| Plan requirements | Implementation and verification |
|---|---|
| R1–R3: coverage and provider fidelity | Sync/async OpenAI-compatible, Azure, Portkey, Anthropic, and Gemini adapters preserve returned fields and report missing, opaque, unsupported, and legacy availability explicitly. Provider fixtures cover received-but-rejected responses. |
| R4: call association | Threaded sockets, asynchronous batch slots, recursive calls, compaction, and fallbacks use call-local observations. Duplicate prompts and out-of-order responses are tested. |
| R5–R6: behavioral and accounting parity | Paired fixtures compare requests, answers, verdicts, usage, and weakness digests; reasoning sentinels stay out of subsequent prompts. Existing proposal, validation, and experiment tests exercise unchanged decisions. |
| R7–R9: default persistence and failures | Experiment run paths, subprocess workers, Lambda adapters, attribution, and proposals bind recorders automatically. Tests cover partial execution, committed files after process termination, moves, hashes, and escaping references. |
| R10–R11: caches and history | Meta caches replay portable observations without provider calls. Legacy traces retain their bytes and hashes. Sealed-stage references are verified. Independent-run evidence comparisons exclude only observation metadata after verifying it. |
| R12: persistence boundary | Storage failures preserve available usage and escape transport retries and task-failure conversion. A regression test covers a worker storage failure coinciding with its deadline. |

## Validation results

All verification used offline fixtures; no paid experiment was launched.

- Full offline suite: `uv run --no-sync pytest -m 'not live' -q` — **2,486 passed, 16 failed, 7 skipped, 21 deselected** in 17m28s. Seven failures introduced or exposed by this change were then fixed: lightweight logger compatibility, two outdated artifact/test-double assumptions, and four test patch targets affected by the repository's module-reimport tests.
- Follow-up regression suite — **141 passed, 3 skipped**. This includes every affected regression, the canonical-manifest tests, runtime seams, all new runtime/store/meta/parity tests, and import tests placed before the new tests to reproduce the original order-sensitive condition.
- Provider capture and worker-deadline regression suite — **20 passed**. Earlier focused runs also covered client transports, recursion, workers, proposals, evaluation, and Lambda behavior.
- The remaining **nine failures reproduce on the unmodified starting commit** in an isolated checkout: two mining-worker profile expectations, the DeepSeek finite-inventory expectation, the GPT-OSS/Kimi profile inheritance expectation, selected-pool sizes, and four tests requiring the absent `experiment_smoke/opt/round_01/proposals/r01-c01-s4/proposal.json` fixture. These configuration and historical-fixture issues were left unchanged.
- Scoped Ruff checks and formatting pass; `git diff --check` passes. Full-repository Ruff reports 10 diagnostics versus 11 at the starting commit. Six existing files remain unformatted versus seven at the starting commit.
- Full `ty check` reports 534 diagnostics versus 538 at the starting commit; comparing diagnostic messages after removing line/column offsets finds **no new diagnostics**. New observation modules and tests also pass their scoped type check.
- The required mutating Ruff and pre-commit commands ran against a staged snapshot in an isolated checkout, protecting unrelated working-tree files. Pre-commit's type hook passed; Ruff hooks found existing issues in tracked archived experiment scripts and reformatted historical files. None of those automatic edits changed an owned implementation file or were copied into this branch.

The full suite was not rerun after the fixes; the 141-test follow-up verifies the affected paths. The three parity tests and final deadline regression were added after the original full-suite collection and are included in the focused results.

## Review and follow-up

`ce-simplify-code` and `ce-code-review` ran sequentially in the main thread under this repository's AGENTS.md mapping. The completed review covered correctness, project standards, testing, maintainability, API contracts, reliability, and adversarial failure scenarios. It identified the deadline/persistence race, which was fixed and tested. There are no unresolved review findings. This provides no independent-agent or cross-model review coverage.

Review receipt: `/tmp/compound-engineering-501/ce-code-review/reasoning-5a5e05974169/review.json`; caller fix dispositions are in adjacent `fixes.json`. Temporary full-suite, baseline, hook, and regression logs use `/tmp/rlm-reasoning-*.log`; this document preserves their conclusions beyond temporary-file cleanup.

For a later authorized fresh experiment, inspect its first completed mining run, first meta attempt, and first validation run. Each should resolve its observation references and distinguish returned reasoning from provider-unavailable data. The configured live provider's returned fields remain unverified. Missing/corrupt references or changed model requests/accounting are regression signals; stop and investigate before resuming the experiment. This check belongs to the agent or maintainer launching that experiment and does not require changing the experiment's scoring rules.

Implementation and documentation are committed locally. Publishing, merging, paid backfill, and launching an experiment remain separate actions. The existing analysis-report edits and `RESEARCH/` working files are excluded.
