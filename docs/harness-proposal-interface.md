# The Harness Proposal Interface (`shrlm-proposal/v1`)

This is the contract between **Stage 2 (Harness Proposal)** and **Stage 3
(Proposal Validation)**. Stage 2 produces candidate harness edits as artifacts
in this format; stage 3's loader (`shrlm/optimization/candidates.py`) enforces
every rule below and turns each conforming candidate into a runnable harness.
If you are building stage 2, this document plus
`docs/handoff-harness-proposal.md` is everything you need to target; the loader
is the executable version of this page, and its tests
(`tests/optimization/test_candidates.py`) double as worked examples.

Versioning: the `format` tag is the version. The loader rejects anything but
`shrlm-proposal/v1`; a future v2 will be additive. Unknown extra top-level
fields are tolerated and ignored, so you may carry your own bookkeeping.

## Directory layout

One candidate = one directory = one `proposal.json`:

```
<proposals_dir>/
├── <candidate_id>/
│   └── proposal.json      ← everything below
├── <candidate_id>/
│   └── proposal.json
└── ...
```

- The **directory name must equal the `candidate_id`** inside the file; the
  promotion ledger links candidates by id, so a mismatch is rejected, not
  repaired — at the text level (`schema` gate), before any candidate code runs
  or `surfaces.py` is written.
- `candidate_id` must be filesystem-safe: `[A-Za-z0-9][A-Za-z0-9._-]*`.
- The loader writes one generated file, `surfaces.py`, into the candidate's
  directory when it materializes callable surfaces. Do not create or edit that
  file yourself; treat the candidate directory as writable by stage 3.

## `proposal.json`

```json
{
  "format": "shrlm-proposal/v1",
  "candidate_id": "r00-c01-s4-verify",
  "base_harness_hash": "<sha256 of the incumbent harness>",
  "target_signature": {
    "verifier_cause": "incomplete",
    "failing_level": "root",
    "causal_status": "causal",
    "agent_mechanism": "skipped_verification"
  },
  "surface": "S4",
  "harness": {
    "format": "shrlm-harness/v1",
    "name": "r00-c01-s4-verify",
    "hash": "<sha256 of the serialization below>",
    "harness": { "name": "...", "orchestrator": false, "surfaces": { "...": "all nine" } }
  },
  "predicted_effect": "The root re-checks accumulated results against the frontier before flipping ready.",
  "regression_risks": ["One extra turn per run; may over-trigger on short tasks."],
  "provenance": {
    "model": "qwen/qwen3-30b-a3b-instruct-2507",
    "prompt_sha256": "<sha256 of the exact proposer prompt used>"
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `format` | string | Exactly `"shrlm-proposal/v1"`. |
| `candidate_id` | string | Stable identity; equals the directory name; filesystem-safe. |
| `base_harness_hash` | string | `harness_hash(...)` of the harness this edit starts from. Must be the current incumbent or the candidate is rejected. |
| `target_signature` | object | The four φ strings of the mined failure pattern this edit targets, verbatim from the bundle's `patterns[i].signature`. Each value must be in the closed vocabulary of `shrlm/optimization/taxonomy.py` (`VerifierCause`, `FailingLevel`, `CausalStatus`, `AgentMechanism`). |
| `surface` | string | The one declared surface the edit modifies: `"S1"`–`"S9"`. Must equal the surface the serialization actually differs on. |
| `harness` | object | A full `shrlm-harness/v1` envelope (below) — the *entire* candidate harness, not a delta. |
| `predicted_effect` | string | Non-empty. What behavior the edit is predicted to change (the paper's audit record). |
| `regression_risks` | list of strings | What the edit might break. May be empty, but must be present. |
| `provenance` | object | `model` (the fixed proposer model) and `prompt_sha256` (sha256 of the exact proposer prompt), both non-empty strings. |

## The harness payload: a full envelope, not a delta

The `harness` field is a `shrlm-harness/v1` envelope, exactly what
`shrlm.harness_identity.write_harness_json` produces: `format`, informational
`name`, content `hash`, and the full `harness` serialization of all nine
surfaces plus the `orchestrator` scalar. Build it in code — construct the edited
`Harness` with `dataclasses.replace(incumbent, <field>=...)` and serialize it
through `serialize_harness` / `hash_of_serialization`; never assemble the JSON
by hand.

Rules the loader enforces against the incumbent:

- **Exactly one surface differs.** The loader diffs the candidate serialization
  against the incumbent's, surface by surface. Byte-identical → rejected
  ("modifies no surface"); two or more surfaces changed → rejected naming all
  of them; the changed surface must be the declared `surface`. S8 counts as
  one surface even though it serializes as two keys (`S8_repl_helpers` and
  `S8_sub_repl_helpers`) — editing both dicts is still one edit.
- **`orchestrator` is not editable.** A candidate that flips it is rejected.
- **`name` is free.** It is informational and excluded from the hash; give the
  candidate harness a name matching the `candidate_id`.
- **The envelope hash must be true.** The loader recomputes the hash of the
  serialization; a mismatch is rejected as tampering.

## Callable surfaces (S7, S8 entries, S9)

Callables travel as source text (the `source` fields `serialize_harness`
emits). The loader materializes them **file-backed**: each source is written
into a generated module (`surfaces.py`) in the candidate's directory and
imported from there, so `inspect.getsource` recovers the exact source and the
harness re-serializes and re-hashes identically — this is what lets the
evaluation driver record the candidate's `harness.json` without serialization
errors. Consequences for candidate source:

- Each callable's source must be **exactly one undecorated top-level `def`**,
  ending in a newline. No module-level statements, no decorators, no lambdas,
  no `exec`-built functions.
- S7's `declared_bound` travels in the envelope's `S7_metadata` field (as
  `serialize_harness` records it), never as code; the loader reattaches it as
  the function attribute. Remember the runner's coupling: S1's truncation
  sentence must state the bound the active S7 builder declares, so an S7 edit
  that changes the bound cannot pass as a one-surface edit by design.
- The materialized harness must **re-serialize byte-identically** to the
  envelope. Serialize your candidate from live code with `serialize_harness`
  and this holds automatically.

### Allowed-import vocabulary

The generated module begins with a fixed preamble, and that preamble is the
entire set of names candidate source may reference beyond builtins
(`CANDIDATE_MODULE_PREAMBLE` in `shrlm/optimization/candidates.py`):

```python
import json
import math
import re
import textwrap
from typing import Any

from rlm.core.types import AnswerDecision
from rlm.utils.parsing import DEFAULT_MAX_CHARACTER_LENGTH, default_metadata_builder
```

Anything outside this vocabulary fails at import or probe time inside the gate
subprocess and comes back as a structured rejection. The namespace is a
legibility contract and defense-in-depth, not a security boundary — the REPL
already executes model-written code by design.

## What the loader does with a candidate (gate order)

Gates run in this order and stop at the first failure. Every failure is a
`CandidateRejection` value carrying `candidate_id`, the `gate` name, and the
violation — never an exception — so the promotion ledger records every
candidate, including the ones that never ran.

| # | Gate | Checked | Candidate code runs? |
|---|---|---|---|
| 1 | `schema` | Fields, types, closed vocabularies, filesystem-safe id | No |
| 2 | `envelope_hash` | Recomputed serialization hash equals the declared one | No |
| 3 | `base_hash` | `base_harness_hash` is the incumbent's hash | No |
| 4 | `surface_diff` | Exactly one surface differs, and it is the declared one; `orchestrator` untouched | No |
| 5 | `caps` | An *enabled* S6 policy may not exceed any experiment-owned cap (e.g. `max_depth`), and every capped value must be a positive finite number (NaN/inf rejected); comparison only — the tighten-only merge lives in the cost governor | No |
| 6 | `materialization` | Source parses, writes to `surfaces.py`, imports — in a subprocess under a wall-clock timeout | Subprocess only |
| 7 | `harness_check` | `shrlm.runner.check_harness`: invariant probes (I1 boundedness, S9 signature/return type, S6 field vocabulary, plumbing, stated limits) — same subprocess | Subprocess only |
| 8 | `round_trip` | The materialized harness re-serializes byte-identically to the envelope | Subprocess only |

The gate subprocess runs with a stripped environment: the child inherits only a
small allowlist (`PATH`, `PYTHONPATH`, `HOME`, `TMPDIR`, and locale variables),
and it re-scrubs itself after import so values re-loaded from a `.env` file are
dropped too — API keys and every other host secret are gone before candidate
code runs.

A candidate that survives is returned as a `LoadedCandidate`: the live
`Harness` (edited surface materialized, every unchanged surface the incumbent's
own object), the edited surface id, the envelope hash, and the parsed proposal
— ready for the evaluation driver. Load a whole proposals directory with
`load_candidates(proposals_dir, incumbent)`; it returns
`(loaded, rejections)` covering every candidate directory exactly once. Every
rejection from a directory load is keyed by the (unique) directory name, so
two malformed proposals declaring the same id cannot collide in the ledger; a
different declared id is preserved in the rejection reason.

## Practical notes for the stage-2 developer

- **Build candidates from live `Harness` values**, then serialize. If your
  proposer emits surface text, apply it with `dataclasses.replace` and let
  `serialize_harness` produce the envelope; hand-built JSON is how you get
  `envelope_hash` and `round_trip` rejections.
- **The incumbent moves.** `base_harness_hash` must name the harness the
  validation round is actually defending; read it from the round's
  `harness.json` (see the handoff doc) or from the promotion ledger.
- **Rejections are data.** Expect your proposals to be rejected sometimes and
  read the `gate`/`reason` pair; they are designed to be actionable
  ("candidate declares surface S2 but modifies S3").
- **K distinct candidates** (the handoff doc's contract) means K distinct
  candidate directories; the loader gates each independently.
