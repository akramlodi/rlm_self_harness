# shrlm — Self-hosted Recursive Language Model harnesses

This package defines the RLM harness abstraction used by the optimization and experiment loop:
a small, auditable specification of the root model's environment (nine editable surfaces plus
an orchestrator flag), a runner that enforces mechanical invariants and constructs an RLM for
execution, and utilities to serialize and identity a harness so runs can be attributed
deterministically.

Contents
- `rlm_harness.py` — the nine editable surfaces (S1..S9), the Harness dataclass, two starting
  harnesses (`H0`, `H0*`), and helpers to assemble system prompts.
- `runner.py` — constructs a `HarnessedRLM` from a `Harness`, runs structural checks (I1–I3),
  derives prompt and capacity sentences, monitors behavioral metrics, and exposes helpers to
  aggregate run metrics and acceptance inputs.
- `harness_identity.py` — deterministic serialization and hashing of a `Harness` (so runs can be
  reproduced and compared), plus `write_harness_json` to write the harness envelope to disk.
- `docs/`, `environments/`, `experiment/`, `optimization/` — related project folders (see top-level
  docs for usage).

Why this module exists
- To separate editable design surfaces (S1..S9) from runtime mechanics and experiment-owned
  controls.
- To enforce three frozen invariants (I1: prompt-as-variable mitigation, I2: programmatic sub-calls,
  I3: answers emitted from variables) structurally where possible, and to monitor the rest.
- To make a harness fully reconstructible (serializable) so measured effects between runs are
  attributable to content changes, not to object identity or ephemeral state.

Key concepts
- Surfaces (S1–S9)
  - S1: repl_contract — root system prompt contract
  - S2: decomposition_instruction — turns 1–2 planning/decomposition
  - S3: execution_instruction — per-turn execution rules
  - S4: verification_instruction — pre-submission checks
  - S5: recovery_instruction — what to do when sub-calls fail
  - S6: runtime_policy — numeric limits and switches (only specific fields allowed)
  - S7: metadata — turn-to-turn memory (must declare `declared_bound`)
  - S8: repl_helpers / sub_repl_helpers — functions and values installed into sub/ root REPLs
  - S9: answer_middleware — programmatic acceptance/redirect logic
- Invariants
  - I1 (boundedness): S7 must not grow proportionally with prompt size (S7 must declare a positive `declared_bound`).
  - I2 (programmatic sub-calls): sub-calls must be issued by code, not verbalized as single actions.
  - I3 (outputs-in-variables): the final answer must be accumulated in a variable (not directly verbalized).
- Orchestrator flag: affects the effective system prompt (an addendum is appended when `orchestrator=True`).

Quick examples

Import the provided starting harnesses:
```python
from shrlm.rlm_harness import H0, H0_STAR, assemble_system_prompt
print(H0.name)        # "H0"
print(H0_STAR.name)   # "H0*"
print(assemble_system_prompt(H0))
```

Make a small, editable harness based on `H0` (use `dataclasses.replace` for frozen dataclasses):
```python
from dataclasses import replace
from shrlm.rlm_harness import H0

custom = replace(H0, name="my_harness", decomposition_instruction="Probe context and split into steps.")
print(custom.name)
```

Build a harnessed RLM and run one completion:
```python
from shrlm.runner import build_harnessed_rlm

# backend and backend_kwargs are experiment-owned; pass them as appropriate.
hr = build_harnessed_rlm(custom, backend="openai", backend_kwargs={"api_key": "..."}, log_dir="/tmp/rlm_logs")
run = hr.completion("Solve X by ...")
print(run.metrics)  # run.metrics includes turns, sub_call_count, cost, answer_from_variable, etc.
```

Serialize a harness and compute its content hash:
```python
from shrlm.harness_identity import write_harness_json, harness_hash

# write JSON envelope to disk
envelope = write_harness_json(custom, "my_harness.json")
print(envelope["hash"])n# or compute hash directly
print(harness_hash(custom))
```

Programmatic checks
- `shrlm.runner.check_harness(harness)` will raise `ValueError` on structural problems:
  - S6 contains unknown or experiment-owned keys,
  - S7 fails boundedness or omits `declared_bound`,
  - S1 does not state the truncation sentence that S7 declares,
  - S8 shadows required REPL plumbing names, or
  - S9 has the wrong signature or returns a non-AnswerDecision.
- `build_harnessed_rlm` runs `check_harness` before constructing the RLM; any structural violation is rejected early.

Notes and best practices
- S7 builders must set a `declared_bound` attribute (positive integer). The runner uses it to derive S1's truncation sentence and probes S7 for boundedness.
- Callables used in S7/S8/S9 must be statically definable so `inspect.getsource()` can recover their source; otherwise serialization fails with `HarnessSerializationError`. Dynamically generated callables require explicit provenance.
- Keep experiment-owned concerns (budget, tracing callbacks, backend routing) out of harness surfaces — these belong to experiment code, not to the harness.

Where to look next
- `rlm_harness.py` — read the surface builders and the `Harness` dataclass to see intended defaults and conventions.
- `runner.py` — read structural checks and `build_harnessed_rlm` for how harness fields are enforced and wired into the runtime.
- `harness_identity.py` — read serialization rules and hashing logic to understand what is recorded and hashed (name is informational and not included in the content hash).

If you want, I can:
- Commit this README to `shrlm/README.md` in the repository.
- Expand sections with more examples, tests, or a short tutorial that walks through constructing an edit and running the acceptance gate.
