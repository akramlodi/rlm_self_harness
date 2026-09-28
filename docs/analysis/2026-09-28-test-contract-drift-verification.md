# Test contract drift: implementation verification

Implements the [test repair plan](../plans/2026-09-28-1438-fix-test-contract-drift-plan.md). All nine failures recorded in the [previous verification report](2026-09-28-grounded-mining-proposal-verification.md) are resolved. Production code and experiment settings are unchanged.

## Changes

- Shipped-profile tests compare loaded mining worker counts with their TOML declarations, including smoke overrides. The pairs preset retains explicit inventory checks with its current 20/10/10 allocation.
- GPT-OSS tests assert that profile's own caps, splits, repetitions, environment, and starting harness. They no longer require equality with the independently tuned Kimi profile.
- Pair split tests compare manifest counts and actual file lengths with the input config, retaining the exact expected pool filenames.
- Loader tests use a committed synthetic S4 proposal instead of a missing local smoke-run artifact. The fixture was generated once through the production writer and candidate loader. Tests retain envelope, surface-key, hash, loader, rendered-module, and byte-for-byte proposal checks. The expected rendered module is text data; existing historical examples are unchanged.

## Verification

| Check | Result |
|---|---|
| Config and split suites | 143 passed |
| Loader fixture, candidate, and proposal suites | 252 passed |
| Full non-live suite in a clean local clone | **2,519 passed, 8 skipped, 21 deselected; zero failures**, 16m51s |
| Ruff lint, formatting, and pre-commit on changed files | Passed |
| Simplification and code review | No further changes or actionable findings |

The full run used `uv run --no-sync python -m pytest -q -m 'not live'` at implementation commit `d3efc9a3`, with the existing development environment. Imports were confirmed to resolve to the clone. No ignored local experiment outputs were copied, and no `experiment_smoke/` directory existed there. Tracked historical experiment directories remain part of the repository. Tests left the clone's tracked files unchanged.

The skips came from existing optional-dependency and live-integration conditions, including the unavailable IPython module during collection. No skips or xfails were introduced. No experiments or paid model calls were launched.

Repository-wide checks remain limited by existing issues outside the changed lines: Ruff reports 10 lint errors and six formatting failures; strict type checking reports 535 diagnostics. Repository-wide pre-commit also processes tracked generated artifacts and reports failures. It ran only in the disposable clone, and its automatic changes were discarded before the full tests. The patch does not include that unrelated cleanup.

Review used the code-review skill's local lite path, with project standards and plan requirements checked. Simplification ran sequentially in the main session as required by the repository's agent mapping; no independent reviewer corroboration is claimed.

Local execution logs are `/tmp/test-contract-drift-config-tests.log`, `/tmp/test-contract-drift-candidate-tests.log`, `/tmp/test-contract-drift-full-pytest.log`, `/tmp/test-contract-drift-ruff.log`, `/tmp/test-contract-drift-format.log`, `/tmp/test-contract-drift-all-precommit.log`, and `/tmp/test-contract-drift-typecheck.log`.
