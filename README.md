 
---

<h1 align="center" style="font-size:2.8em">
<span>Recursively self-harnessing recursive language models (<span style="color:orange">SH-RLM</span>s)</span>
</h1>


## Overview
Recursive language models (RLMs) let a fixed, bounded-context model answer queries over inputs far larger than its context window: the model treats its prompt
as an external variable, decomposes it in code, and recursively calls itself on the pieces [arXiv preprint](https://arxiv.org/abs/2512.24601). Whether recursion succeeds depends not on the model a[...]
the harness that governs how it decomposes the input, when it stops, and how it recovers from errors. Today that harness is supplied either by fine-tuning the model or by an expert hand-engineerin[...]


> [!NOTE]
> This repository contains inference code for SH-RLMs with support for various sandbox environments. Open-source contributions are welcome. This repository is maintained by the authors of the pape[...]

## Repository structure
This repository separates core runtime, harness specifications, training code, and supporting tools so you can find the part you need quickly.

Top-level layout (annotated):

```text
.gitattributes                       repository attributes
.github/                              GitHub workflows and CI configs
.gitignore                            ignore rules
AGENTS.md                             contributor and coding-agent guidelines
Makefile                              common tasks (install, lint, test, runtime examples)
pyproject.toml                        packaging, dependencies, and extras
uv.lock                               lockfile for the uv environment
configs/                              experiment TOMLs (one per model x environment arm; see below)
docs/                                 dated plans and analysis notes for the experiment work
examples/                             experiment drivers (run_experiment, run_evaluation, ...) and runtime demos
paper/                                paper source and assets
patches/                              auxiliary patches
rlm/                                  core RLM runtime package (clients, environments, logger, utils)
scripts/                              proposal-diff helper
shrlm/                                self-harness spec, environments, baselines, optimization loop, experiment orchestration
third_party/                          licence and provenance of vendored code (lambda-RLM)
training/                             training harness, environments, and examples (rlm-train, oolong env)
visualizer/                           web visualizer (Node.js + shadcn/ui) for run logs
tests/                                unit tests and test utilities (paid `live` tests are opt-in)
```

How it fits together: The `rlm/` package implements the runtime client, REPL environments, logging, and utilities used when running RLMs. The `shrlm/` package provides the self-harness specification (S1–S10), structural checks (I1–I3), and helpers to build and serialize harnesses; on top of that it holds the benchmark environments (`shrlm/environments/`), fixed baselines (`shrlm/baselines/`), the mine → attribute → propose → validate → promote loop (`shrlm/optimization/`), and the experiment orchestrator and analysis tools (`shrlm/experiment/`). The `training/` directory holds training harnesses and example environments (including `oolong`) that depend on the core packages. The `visualizer/` is a separate frontend that reads JSONL logs written by `RLMLogger` in order to show call graphs, code, and sub-call traces.

Where to look next:
- rlm/Readme.md — runtime usage and REPL environments
- shrlm/README.md — self-harness specification and examples
- shrlm/optimization/README.md — the optimization loop, stage by stage
- shrlm/baselines/README.md — H0, H0* and λ-RLM baselines
- pyproject.toml — packaging, extras for sandboxes (modal, prime, ipython)

## Self-Harness experiments

New experiment calls save provider-returned reasoning beside their traces and meta attempts. See [saved reasoning outputs](docs/experiment-reasoning.md) for locations, cache behavior, and availability meanings.

The Self-Harness experiment (the optimization loop in `shrlm/experiment/`) is driven by
`examples/run_experiment.py`, which loads one profile from one TOML in `configs/`.
Every experiment parameter lives in that TOML — the code hardcodes none of them — and the
loaded config is hashed into an **identity** that is stamped into the output directory.
Re-invoking the same command resumes an interrupted experiment exactly; a changed config
refuses to resume, so every distinct configuration gets its own fresh `--out-dir`.

### Environments and configurations

`[loop].environment` selects which benchmark the loop mines and validates on. Each is
implemented in `shrlm/environments/`:

| Environment | Task | Quality metric |
|---|---|---|
| `oolong_pairs` (default) | OOLONG-Pairs pair extraction at 8k/16k/32k; the 262k pool is test-only | pair-set F1 (`primary_quality`) |
| `oolong_synth` | OOLONG-synth aggregation; OOLONG-real is run periodically as a generalization check only | pass count |
| `obliq_bench_math` | OBLIQ-Bench Math retrieval (151 queries) | NDCG@10 (`primary_quality`) |

Every config runs **DeepSeek-V4-Flash** for all three roles (runner, attributor,
proposer), hosted either on OpenRouter or on Azure AI Foundry. Each TOML in `configs/` is
one environment × host arm:

| Config | Environment | Host | Notes |
|---|---|---|---|
| `experiment_oolong_pairs_blended_openrouter_DeepSeekV4Flash.toml` | oolong_pairs (8k+16k+32k) | OpenRouter | **default** (`config.CONFIG_PATH`) and current arm: 4 screened candidates/round, patience 3 |
| `experiment_oolong_pairs_blended_DeepSeekV4Flash.toml` | oolong_pairs (8k+16k+32k) | Azure | per-length τ, 3 validation reps, patience-based stop |
| `experiment_oolong_pairs_DeepSeekV4Flash.toml` | oolong_pairs (single length) | Azure | |
| `experiment_oolong_pairs_openrouter_DeepSeekV4Flash.toml` | oolong_pairs | OpenRouter | read by `examples/lambda_rlm_oolong_pairs_long_smoke.py` only |
| `experiment_oolong_DeepSeekV4Flash.toml` | oolong_synth | Azure | up to 15 rounds |
| `experiment_obliq_bench_math_DeepSeekV4Flash.toml` | obliq_bench_math | Azure | real mining loop |
| `experiment_obliq_bench_math_DeepSeekV4Flash.*.toml` | obliq_bench_math | Azure | frozen copies for the qualification and matched-20 baseline studies |
| `dataset_qualification_DeepSeekV4Flash.toml` | all | Azure | input to `examples/dataset_qualification.py`, not an optimization run |

Every config also carries a `smoke` profile: its `[smoke.*]` tables override **scale
counts only** (instance counts, repetitions, candidate width, rounds, spend caps); every
other semantic is identical to `full`. The header comment of each TOML records why it
differs from the file it was cloned from.

### Running an experiment

Append `--dry-run` first: it checks credentials, loads the config, and prints the
identity and per-role backends without creating anything or spending anything.

```bash
# Smoke: end-to-end plumbing check on the default config (small, but spends real money)
uv run python examples/run_experiment.py --profile smoke --out-dir ./experiment_smoke

# The default arm (blended OOLONG-Pairs on OpenRouter)
uv run python examples/run_experiment.py --out-dir ./experiment_oolong_pairs_openrouter

# Any other arm
uv run python examples/run_experiment.py --config configs/<file>.toml --out-dir ./<fresh_dir>
```

Interrupting with Ctrl-C loses nothing — re-run the same command and the experiment
resumes at its exact stage boundary. Two arms can run concurrently from the same
checkout as long as each has its own `--out-dir`.

### Credentials and pricing attestation

OpenRouter configs need `OPENROUTER_API_KEY` (a `.env` file at the repo root works);
OpenRouter reports the real per-response cost, so nothing else is required. Azure
configs instead need:

```bash
echo 'AZURE_API_KEY=...' >> .env
echo 'AZURE_FOUNDRY_ENDPOINT=https://<your-resource>.services.ai.azure.com' >> .env
# Verify the deployment's pay-as-you-go rates in the Azure portal match the
# config's [pricing.list_price] ($0.19 in / $0.51 out per 1M), then attest them --
# Azure returns no per-response cost, so cost is synthesized from these rates
# and the launcher refuses any live spend until the attestation matches:
export SHRLM_VERIFIED_PRICING='0.19/0.51'
```

The config's `model` value must equal your Foundry **deployment name**
(`DeepSeek-V4-Flash`). In every configuration the launcher refuses to start — and spends
nothing — if a required credential or attestation is missing.

Note for Azure Foundry: an empty completion arriving with `finish_reason='length'` is
classified once as token-limit exhaustion (`TokenLimitExceededError` ->
`resource_terminated`, spend kept) instead of entering the empty-content retry ladder —
a length cut is deterministic for the prompt, so a retry only re-bills the same outcome.
Empty completions with other finish reasons keep the retry ladder.

### Live pytest tiers

Paid pytest items are marked `live` and are **deselected** (never executed) unless
`SHRLM_RUN_LIVE=1` is exported and `CI` is unset; once selected, module-level skip
gates still name any missing credential or attestation in `-rs` output.
`SHRLM_EXPERIMENT_CONFIG=<path-to-toml>` points every live tier — the client live
class, the smoke live round, the OOLONG recursion tier, and
`examples/experiment_smoke.py` via `--config` — at that config; unset, each tier keeps
its own default (the default config's smoke profile, or
`configs/experiment_oolong_DeepSeekV4Flash.toml` for the recursion tier).

### Proposal quality and OOLONG-Pairs feedback

Proposal generation now checks edited candidates locally and can repair failed
members once while retaining valid siblings; repair can move the same failed
pattern to another eligible, unoccupied surface. Diagnoses cite operations and
verification limits, and the proposer sees the relevant caller and consumer code.
The proposer selects one intervention per surface before writing replacements.
Each proposal compares actual execution with the unresolved operation and explains
why the change addresses its cause. Task-derived reasoning asks which information
must survive and which conditions the final computation must enforce.
The host encodes literal instruction text once; models no longer escape template braces.
A shared capability description explains what each surface can observe, when it
runs, and what it can change. New S8 parsing/aggregation and S5 recovery routes
require admitted operations from held-in traces. A 32,000-character evidence
budget allocates complete core examples across distinct mechanisms before adding
context or contrasts; attribution also preserves complete code and coordinates.
Rejected candidates with comparable recorded quality gains remain visible as
potentially promising in a separately bounded 12,000-character history. Verifiers
declare their primary quality metric, direction, precision and terminal values;
missing or incomparable measurements remain unassessed. History separates claims,
observed retry/redirect/skill-load events, and measured batch outcomes. Revisited
interventions require a prior reference and revision explanation. Identical
rejected evaluated harnesses under the same incumbent are refused before paid calls.
Exact outcomes and regressions remain visible, and `v=1` provides no causal guarantee.

### Screening, validation and repair

Each round proposes up to `[loop].k` independent single-surface candidates instead of
one merged batch. When more than one is admitted, a **screening tournament** runs every
candidate at `[loop].v_screen` attempts (strictly fewer than `v`) on a small instance
subset, ranks them, and fully validates only the winner at `v` attempts on held-out data,
reusing its screening runs. `[operational].screening_workers` and
`screening_run_workers` control screening concurrency separately from validation.
If every screened candidate is rejected and `[reattribution].enabled` is set, the same
mining runs are re-attributed once (no new rollouts) and re-screened.

For `primary_quality` metrics with a per-length `[promotion.tau_regression]` table,
validation uses **τ early stopping**: context lengths run in order and a candidate
stops as soon as one length's regression veto fires. An unchanged incumbent's baseline
is reused from the previous round instead of being re-run.

With `[refinement].enabled`, a rejected candidate that improves quality on answered
instances while a listed terminal failure cause (by default `wrong_format`) gets worse
receives one bounded **repair generation**; the repaired descendant is judged against
the same incumbent on the same held-out split.

See the [proposal contract](shrlm/docs/harness-proposal-interface.md),
[metric definitions](shrlm/docs/experiment-metrics.md), and
[OOLONG-Pairs investigation](docs/analysis/oolong-pairs-2026-09-13/README.md).
The [methodology note](docs/analysis/2026-09-14-proposal-quality-methodology-notes.md)
records these changes for a future paper update.

### How the analysis populates

Nothing needs to be run to get analysis output. After **every executed round** the
orchestrator refreshes aggregation snapshots under the experiment directory:

```text
<out_dir>/
  config.json                  # profile + config identity hash for this experiment
  splits/                      # the seeded dataset partitions
  opt/round_NN/                # per-round artifacts: runs, manifests, proposals, ledgers
  analysis/<UTC-stamp>/        # one snapshot per analysis batch, e.g. 20260823T210400Z/
    surface_activity.csv
    incumbent_quality.csv
    ...
    provenance.json            # identity, code revision, source-artifact hashes
    published.json             # written LAST -- a snapshot without it is incomplete; ignore it
    plots/                     # figures; re-renderable without touching the frozen CSVs
```

Snapshots are append-only: a re-run allocates a new UTC-stamped directory rather than
overwriting the previous one, and only directories containing `published.json` are ever
selected as "latest" by the plotting tools. The post-round refresh is strictly one-way —
the optimization loop never reads `analysis/` back.

The same aggregations can be re-run by hand at any time (each allocates a fresh
snapshot): `python -m shrlm.experiment.surface_activity <out_dir>`,
`...incumbent_quality`, `...collapse_and_attribution`, `...pattern_frequency_diff`, and
the `plot_*` counterparts. The cost/time report is
`python -m shrlm.experiment.report <out_dir> --profile <profile> --config <toml>`.

Surface activity credits each bundled edit when its linked combined candidate
is promoted, so batch promotions appear on all affected surfaces. Incumbent
quality uses the current round's score of the retained harness: the candidate
score on promotion, otherwise the fresh baseline score. Unscored comparisons
stay blank; earlier scores are never carried forward. The cost report labels
optimization runs and selects their projection basis using `[loop].environment`
from the supplied config; evaluation results keep their recorded environments.

One caveat for any tool that resolves the config from an out-dir: the default TOML is
`config.CONFIG_PATH` (the blended OpenRouter OOLONG-Pairs arm), and a config that no
longer hashes to the identity recorded in `<out_dir>/config.json` resolves to "unknown"
(deliberately — no numbers are invented from a config the experiment never ran under).
For any other arm, always pass `--config configs/<file>.toml` where the tool accepts it.

## Baselines and frozen evaluation

The optimization loop starts from **H0** (sparse harness); **H0\*** (the upstream RLM
authors' hand-written harness) and **λ-RLM** (a typed functional runtime, vendored in
`shrlm/baselines/upstream/` with its licence under `third_party/lambda-RLM/`) are fixed comparison baselines. Details are in
[`shrlm/baselines/README.md`](shrlm/baselines/README.md).

- `examples/run_evaluation.py` evaluates saved harnesses (e.g. `initial`, `sh_rlm`) on
  frozen test splits with resumable workers; `--dry-run` prints run counts and caps.
- `examples/dataset_qualification.py --config configs/dataset_qualification_DeepSeekV4Flash.toml`
  runs H0, H0\* and λ-RLM on matched instances to compare datasets; it does not optimize.
- `examples/obliq_bench_math_smoke.py` and `examples/lambda_rlm_oolong_pairs_long_smoke.py`
  are narrow live smoke tests for one baseline × environment slice.
- `examples/calibrate_oolong_pairs_tau.py` measures the repeated-H0 noise floor used to
  set the per-length OOLONG-Pairs τ.

## RLM runtime

The `rlm/` package (runtime setup, REPL environments -- local, IPython, Docker, Modal,
Prime, Daytona, E2B -- model providers, trajectory logging, and the visualizer) is documented in
[`rlm/Readme.md`](rlm/Readme.md). The self-harness specification (surfaces S1-S10,
invariants I1-I3) is documented in [`shrlm/README.md`](shrlm/README.md).

## Relevant Reading
* **[Dec '25]** [Recursive Language Models arXiv](https://arxiv.org/abs/2512.24601)
* **[Oct '25]** [Recursive Language Models Blogpost](https://alexzhang13.github.io/blog/2025/rlm/)
* **[Jun '26]** [Self-Harness: Harnesses That Improve Themselves](https://arxiv.org/pdf/2606.09498)
  
A citation for SH-RLM will be added here once the paper is public; its source is in [`paper/`](paper/).
