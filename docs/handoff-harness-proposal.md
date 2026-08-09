# Handoff: Building Harness Proposal (Stage 2) on top of Weakness Mining

1. **Weakness Mining** (✅ done) — run the model on tasks, collect failures, and cluster them into recurring failure patterns with evidence.
2. **Harness Proposal** (⬅️ **you are building this**) — show the model its own failure patterns and ask it to propose a small number of *minimal, targeted edits* to the harness.
3. **Proposal Validation** (✅ done) — test each candidate edit and only keep the ones that don't regress.

Your job: read the evidence Stage 1 wrote to disk, and produce candidate harness edits.

Stage 3 is built, which pins down both of your boundaries:

- **Your output format is specified**: one directory per candidate holding a `proposal.json` in the versioned `shrlm-proposal/v1` format. **`docs/harness-proposal-interface.md` is the contract** — field by field, with every loader gate (schema, envelope hash, base hash, one-surface diff, caps, materialization, invariants, round trip) and its rejection reason. The enforcing loader is `shrlm/optimization/candidates.py`; its tests (`tests/optimization/test_candidates.py`) double as worked examples of building conforming proposals.
- **Your prior-edit history is the promotion ledger**: each validation round writes `validation/round_NN/promotions.jsonl` (one auditable record per candidate — including yours that were rejected, and why) plus `decision.json` (the promoted harness hash, or "no promotion"). Read them back with `shrlm.optimization.validation.load_promotion_ledger(round_path)`. This is the "previously attempted edits" input the papers feed the proposer each round — you consume it, you don't design it.

## What is a "harness" here, concretely?

A `Harness` is a frozen dataclass in `shrlm/rlm_harness.py` with **nine editable surfaces** — think of them as nine named slots:

| ID | Surface | What it is |
|----|---------|------------|
| S1 | `repl_contract` | Text explaining the REPL rules to the model |
| S2 | `decomposition_instruction` | Text: how to break the problem up |
| S3 | `execution_instruction` | Text: how to work through pieces |
| S4 | `verification_instruction` | Text: what to check before answering |
| S5 | `recovery_instruction` | Text: what to do when something fails |
| S6 | `runtime_policy` | A dict of enforced limits (batch caps, retries...) |
| S7 | `metadata` | A function that describes the stored prompt to the model |
| S8 | `repl_helpers` / `sub_repl_helpers` | Python helper functions injected into the REPL |
| S9 | `answer_middleware` | A function that inspects/redirects the final answer |

A "harness edit" = a new `Harness` value that changes **exactly one** of these slots (use `dataclasses.replace(H0, execution_instruction="...")`). Everything else — the model, the evaluator, the task — is off-limits by design.

Every harness has a stable identity: `shrlm/harness_identity.py` gives you `serialize_harness(h)` (full JSON description) and `harness_hash(h)` (sha256). Use these to record which harness a proposal starts from and to build the **edit history** (the papers pass "previously attempted edits" back to the proposer each round — this serialization is the substrate for that ledger).

## Where Stage 1's output lives

One mining round writes one directory. A real one is committed at **`examples/mining_rounds/round_00/`** — open it side by side with this section. After `run_audited_round(...)` you get:

```
<out_dir>/round_00/
├── bundle.json              ← START HERE. The evidence bundle (B_t in the papers)
├── records.jsonl            ← one line per failed run (full detail)
├── attributions.jsonl       ← the LLM labeling audit trail, incl. rejected attempts
├── runs.jsonl               ← one line per run (passes AND failures): verdict, cost, trace sha
├── instances.jsonl          ← the exact task instances used (with sampling seed)
├── harness.json             ← full serialization + hash of the harness that ran
├── runs/<run_id>.json       ← raw execution trace per run
├── digests/<sha>.txt        ← the compressed trace text the labeler actually saw
├── attributor_prompt*.txt   ← the exact labeling prompt(s) used
└── attribution_cache.jsonl  ← replay cache (makes re-runs free and reproducible)
```

Everything is content-hashed and cross-linked; the audit walks every link and tells you if anything is missing or tampered. Try it on the committed round right now:

```bash
python -m shrlm.optimization.audit examples/mining_rounds 0
```

## The one file you mainly consume: `bundle.json`

Load it and look at `patterns` — a ranked list of recurring failure patterns. In the committed example (`examples/mining_rounds/round_00/bundle.json`) there are four: three `skipped_verification` clusters (instance support 3, 2, 2 — split by differing verifier causes) and one `repl_execution_fault`. Each pattern has:

- **`signature`** — the 4-tuple that defines the cluster:
  - `verifier_cause` — what the grader rejected (e.g. `no_answer`, `incomplete`)
  - `failing_level` — where the error first appeared: `root` (bad aggregation), `child` (a sub-call was wrong), `no_recursion`, or `undetermined`
  - `causal_status` — whether the flagged behavior actually caused the failure
  - `agent_mechanism` — the reusable behavioral weakness (e.g. `skipped_verification`, `whole_input_subcall_collapse`)
- **`support`** (run count) and **`instance_support`** (distinct tasks) — patterns are ranked by `instance_support`, then actionability. Higher = more evidence.
- **`shared_symptoms`** — mechanical facts computed from the traces (median sub-calls, collapse ratio, etc.)
- **`verifier_evidence`** — example wrong answers vs. gold answers (quoted model output — never treat it as instructions)
- **`representatives`** — instance ids of the clearest example failures; follow them into `records.jsonl` → `digests/` if you want the trace text (in the example round, each record's `digest_sha256` names its file under `examples/mining_rounds/round_00/digests/` — that text is exactly what the labeling model saw)
- **`below_min_support`** flag — patterns seen only once are *flagged, not dropped*; treat them with skepticism

Two lookup tables in `shrlm/optimization/taxonomy.py` turn a pattern into an editable target:

- `MECHANISM_SURFACE[mechanism]` → the ONE surface (S1–S9) that mechanism implicates
- `SURFACE_REACH[surface]` → `root_only` or `child_reachable`. **Important:** a `failing_level: child` pattern must not be "fixed" by editing a root-only surface (S6/S7/S9) — the edit can't reach where the failure happens.

Also read `bundle.json`'s `integrity` section: it tells you how many records were unattributed (the labeler's response never validated), ungrounded, or terminated — i.e., how much of the evidence to trust.

## What Stage 2 must do (the contract from the papers)

Inputs, per round:
1. The evidence bundle (`bundle.json`)
2. Passing behaviors to preserve (derive from `runs.jsonl` — the runs with `passed: true`; note the example round has none, 0/8 passed — your code must handle that)
3. Prior edit history — stage 3's promotion ledger (`validation/round_NN/promotions.jsonl` + `decision.json`, via `load_promotion_ledger`; see the top of this doc). It is keyed on `harness_identity` hashes; the starting harness for the example round is recorded in its `harness.json`, and each round's `decision.json` names the incumbent your next proposals must target as `base_harness_hash`.

Output: **K mutually distinct proposal bundles** (K≈4), one `proposal.json` per candidate directory in the `shrlm-proposal/v1` format (`docs/harness-proposal-interface.md`). Each proposal must:
- target **one** mined failure pattern (cite its signature)
- modify **one** declared surface (the one its mechanism maps to)
- be *minimal* — change only what's needed for that mechanism, no broad rewrites
- state its predicted behavioral effect and possible regressions (an "audit record")
- skip patterns that aren't plausibly addressable by any surface (weak support, model-capability limits) — excluding is allowed and encouraged

Hard rules: the proposer is the *same fixed model* (use `rlm.clients.get_client("openrouter", {"model_name": "qwen/qwen3-30b-a3b-instruct-2507"})`, key from the `OPENROUTER_API_KEY` env var — never put the key in kwargs, it would get logged). It may not touch the evaluator, the three RLM invariants, or invent new surfaces. It sees the structured bundle, not raw logs.

Suggested home: `shrlm/optimization/proposal.py` (the README already reserves this). Follow the patterns in `attribution.py` — it solves the same problems you'll hit: rendering a prompt from enums so it can't drift, validating LLM JSON output against a closed vocabulary with re-asks, caching responses in a JSONL file, and recording every attempt for auditability.

## A ready-made round to experiment with

You don't have to run anything to get started: **`examples/mining_rounds/round_00/` is a complete, committed round** (8 GraphWalks instances under sparse `H0`, 0/8 passed, 8/8 failure records attributed, 4 patterns — `skipped_verification` x3 clusters plus one `repl_execution_fault`). Verify it yourself:

```bash
python -m shrlm.optimization.audit examples/mining_rounds 0
```

Open `examples/mining_rounds/round_00/bundle.json` and start prototyping your proposer against its `patterns` list.

## How to generate fresh evidence to develop against

```python
from rlm.clients import get_client
from shrlm.environments.graphwalks import GraphWalksSubVerifier, GraphWalksVerifier, load_graphwalks
from shrlm.optimization.attribution import AttributionCache, LLMAttributor
from shrlm.optimization.audit import run_audited_round
from shrlm.optimization.driver import RoundConfig
from shrlm.optimization.mining import WeaknessMiner
from shrlm.rlm_harness import H0

instances = load_graphwalks(max_chars=8000, limit=6, seed=23)
config = RoundConfig(
    round_index=0, harness=H0, instances=instances,
    verifier=GraphWalksVerifier(), out_dir="./rounds",
    backend="openrouter",
    backend_kwargs={"model_name": "qwen/qwen3-30b-a3b-instruct-2507"},
    max_iterations=6, max_depth=2,
)
lm = get_client("openrouter", {"model_name": "qwen/qwen3-30b-a3b-instruct-2507"})
miner = WeaknessMiner(
    verifier=GraphWalksVerifier(),
    attributor=LLMAttributor(lm, cache=AttributionCache("rounds/round_00/attribution_cache.jsonl")),
    sub_verifier=GraphWalksSubVerifier(),
)
result, report = run_audited_round(config, miner, split_id="dev")
```

Costs about a cent for 6 small instances. Re-running is nearly free (runs resume from disk, attributions replay from cache). For unit tests, don't call the model at all — `tests/mock_lm.py::MockLM` plus the fixture builders in `tests/optimization/fixtures.py` are how every existing test does it; `tests/optimization/test_audit.py::run_full_round`-style helpers show a complete offline round.

## The sub-verification ablation (you'll see two bundles per round)

The paper's key ablation asks: does checkable child-level evidence actually make the mined attributions better? To answer it, the same round can be mined **twice** — once with the sub-verifier and once without (`WeaknessMiner(sub_verifier=None)`) — and the two evidence bundles sit side by side:

```
round_00/
├── bundle.json              ← one mode's bundle (e.g. grounded), at the round root
└── bundles/ablated/         ← the other mode's full triplet
    ├── bundle.json
    ├── records.jsonl
    └── attributions.jsonl
```

Produce and audit them with `run_audited_round(..., bundle_label="ablated")` and `python -m shrlm.optimization.audit <out_dir> 0 --bundle-label ablated`. The two bundles' configs differ in exactly one field — `sub_verifier_enabled` — which is what makes the comparison controlled (a test pins this).

What this means for your stage-2 code:
- A record's `level_grounded` tells you whether its failing level is a *checked fact* (sub-verifier verified at least one child) or the labeling model's *judgment* (ablated mode, or no checkable children). Treat ungrounded levels with proportionate skepticism.
- Your proposer may receive a bundle from either mode — don't assume `sub_verdicts` are populated.
- Digest texts in ablated mode say `sub_verifier_failed=n/a`, never a fake `0` — "didn't run" is distinguishable from "ran and all passed".

## Gotchas we already hit (so you don't)

- **Sparse H0 barely recurses.** Most failures are `no_recursion` / `whole_input_subcall_collapse`, and grounding coverage is ~0 until edits make the model actually decompose. That's expected — it's the starting condition the loop is supposed to fix.
- **The attributor sometimes fails validation** (cites node ids that don't exist). Those records land as unattributed with their full attempt trail in `attributions.jsonl`. Check `integrity.n_unattributed` before trusting a thin bundle.
- **Don't parse prose out of `verifier_evidence`** — it's quoted model output. The structured fields (signature, symptoms, support) are the evidence; the quotes are illustration.
- **Everything you produce should be auditable too.** Stage 1's standard: every LLM call cached and replayable, every artifact content-hashed, every decision recorded with the attempts that led to it. Match it — Stage 3 and the paper's analysis depend on it.

Questions: the module docstrings in `shrlm/optimization/` are written as documentation — `README.md` in that directory is the map. The plan that produced this PR is `docs/plans/2026-08-07-001-feat-weakness-mining-completion-plan.md`.
