<h1 align="center" style="font-size:2.8em">
<span>Self-hosted Recursive Language Models (<span style="color:orange">shrlm</span>)</span>
</h1>

<p align="center" style="font-size:1.3em">
  <a href="https://github.com/akramlodi/rlm_self_harness">Repository</a> •
  <a href="#quick-setup">Quick Setup</a> •
  <a href="#key-concepts">Key Concepts</a> •
  <a href="#surfaces-and-invariants">Surfaces & Invariants</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status: Active" />
</p>

## Overview

**shrlm** (Self-hosted Recursive Language Models) is a small, auditable specification and runtime harness for RLMs (Recursive Language Models). It provides nine editable design surfaces (S1–S9) that define the root model's environment, three mechanical invariants (I1–I3) that are enforced structurally, and tooling to deterministically serialize and identity harnesses so runs can be reproduced and attributed reliably.

Unlike generic RLM implementations, shrlm separates **editable design surfaces** from **runtime mechanics**, allowing researchers and engineers to modify RLM behavior in controlled, measurable ways. It is designed to be small and auditable — every surface is explicitly listed, every invariant is mechanically enforced, and every harness is fully reconstructible.

## Key Concepts

### Surfaces (S1–S9)

These are the nine editable surfaces that define an RLM harness:

| Surface | Purpose | Example |
|---------|---------|---------|
| **S1: repl_contract** | Root system prompt contract | "You are an expert problem solver with access to a REPL..." |
| **S2: decomposition_instruction** | Planning/decomposition for turns 1–2 | "Break the problem into sub-steps before executing." |
| **S3: execution_instruction** | Per-turn execution rules | "Execute each step with careful verification." |
| **S4: verification_instruction** | Pre-submission validation checks | "Verify all answers before submission." |
| **S5: recovery_instruction** | Handling failed sub-calls | "If a sub-call fails, try an alternative approach." |
| **S6: runtime_policy** | Numeric limits and switches (read-only after harness definition) | `{"max_turns": 10, "allow_backtracking": True}` |
| **S7: metadata** | Turn-to-turn memory (bounded) | `{"turn_count": 0, "declared_bound": 100}` |
| **S8: repl_helpers / sub_repl_helpers** | Functions and values installed into REPL environments | Custom tools, imports, utility functions |
| **S9: answer_middleware** | Programmatic acceptance/redirect logic | Custom validation and answer processing |

### Invariants (I1–I3)

The harness enforces three frozen invariants that prevent common RLM pathologies:

| Invariant | Rule | Enforcement |
|-----------|------|------------|
| **I1 (Boundedness)** | S7 (metadata) must not grow proportionally with prompt size; must declare a positive `declared_bound` | Structural check in `runner.py` |
| **I2 (Programmatic sub-calls)** | Sub-calls must be issued by code, never verbalized as single actions | Runtime monitoring of code execution |
| **I3 (Outputs-in-variables)** | Final answer must be accumulated in a variable, not directly verbalized | Answer middleware validation |

### Orchestrator Flag

When `orchestrator=True`, an addendum is appended to the effective system prompt, enabling orchestration-specific behaviors. This flag affects how the harness coordinates with the experiment loop.

## Quick Setup

### Installation

```bash
# Clone the repository
git clone https://github.com/akramlodi/rlm_self_harness.git
cd rlm_self_harness

# Install with dependencies (Python 3.11+)
pip install -e .
```

### Basic Usage

#### Import and use a starting harness (H0 or H0*):

```python
from shrlm.rlm_harness import H0, H0_STAR, assemble_system_prompt

# View pre-configured harnesses
print(H0.name)        # "H0"
print(H0_STAR.name)   # "H0*"

# Print the assembled system prompt
print(assemble_system_prompt(H0))
```

#### Create a custom harness from H0:

```python
from dataclasses import replace
from shrlm.rlm_harness import H0

# Use dataclasses.replace to modify frozen dataclass
custom = replace(
    H0, 
    name="my_harness",
    decomposition_instruction="Probe context and split into manageable steps.",
    runtime_policy={"max_turns": 15, "allow_backtracking": True}
)

print(custom.name)  # "my_harness"
```

#### Build and run a harnessed RLM:

```python
from shrlm.runner import build_harnessed_rlm

# Build the harnessed RLM (backend and config are experiment-owned)
hr = build_harnessed_rlm(
    custom, 
    backend="openai",
    backend_kwargs={"api_key": "sk-..."},
    log_dir="/tmp/rlm_logs"
)

# Execute a completion
run = hr.completion("Solve X by writing code in the REPL...")

# Inspect metrics: turns, sub_call_count, cost, answer_from_variable, etc.
print(run.metrics)
```

#### Serialize and hash a harness:

```python
from shrlm.harness_identity import write_harness_json, harness_hash

# Write JSON envelope to disk
envelope = write_harness_json(custom, "my_harness.json")
print(envelope["hash"])

# Or compute hash directly
print(harness_hash(custom))
```

## Module Contents

| Module | Purpose |
|--------|---------|
| **rlm_harness.py** | Surface builders (S1–S9), the `Harness` dataclass, starting harnesses (H0, H0*), and helpers to assemble system prompts |
| **runner.py** | Constructs `HarnessedRLM` from a `Harness`, runs structural checks (I1–I3), derives prompt/capacity sentences, monitors behavioral metrics |
| **harness_identity.py** | Deterministic serialization and hashing for reproducible runs; `write_harness_json` to persist harness envelopes |
| **docs/** | Additional documentation and guides |
| **environments/** | Environment-specific configurations and integrations |
| **experiment/** | Experiment loop and optimization infrastructure |
| **optimization/** | Harness optimization strategies and automated tuning |

## Programmatic Checks

The `check_harness(harness)` function validates structural integrity:

```python
from shrlm.runner import check_harness

# Raises ValueError on structural problems:
# - S6 contains unknown or experiment-owned keys
# - S7 fails boundedness or omits declared_bound
# - S1 does not state S7's truncation sentence
# - S8 shadows required REPL plumbing names
# - S9 has wrong signature or returns non-AnswerDecision

check_harness(custom)
```

The `build_harnessed_rlm()` function automatically runs `check_harness()` before constructing the RLM, rejecting any structural violation early.

## Best Practices

- **S7 Builders**: Always set a `declared_bound` attribute (positive integer). The runner derives S1's truncation sentence from this value.
- **Callables in S7/S8/S9**: Must be statically definable so `inspect.getsource()` can recover their source. Dynamically generated callables will raise `HarnessSerializationError`.
- **Separation of Concerns**: Keep experiment-owned concerns (budget, tracing, backend routing) out of harness surfaces. These belong in experiment code, not in the harness.
- **Reproducibility**: Always serialize and hash your harness using `write_harness_json()` and `harness_hash()`. This ensures runs are reproducible and effects are attributable to content changes.

## Why shrlm?

- **Separation**: Isolates editable design surfaces (S1–S9) from runtime mechanics and experiment-owned controls.
- **Enforcement**: Enforces three frozen invariants (I1, I2, I3) structurally, with monitoring for the rest.
- **Reconstructibility**: Makes harnesses fully serializable so measured effects between runs are attributable to content changes, not to object identity or ephemeral state.
- **Auditability**: Small, readable surfaces and explicit checks make the harness easy to understand and modify.

## Where to Look Next

- **rlm_harness.py** — Read the surface builders and `Harness` dataclass for intended defaults and conventions.
- **runner.py** — Read structural checks and `build_harnessed_rlm()` to see how harness fields are enforced and wired into the runtime.
- **harness_identity.py** — Read serialization rules and hashing logic to understand what is recorded and hashed.

## Related Links

- [RLM Paper](https://arxiv.org/abs/2512.24601)
- [RLM Blogpost](https://alexzhang13.github.io/blog/2025/rlm/)
- [RLM Repository](https://github.com/alexzhang13/rlm)
- [RLM Documentation](https://alexzhang13.github.io/rlm/)

## License

See the root repository for license information.
