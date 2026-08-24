 
---

<h1 align="center" style="font-size:2.8em">
<span>Recursively self-harnessing recursive language models (<span style="color:orange">SH-RLM</span>s)</span>
</h1>

<p align="center" style="font-size:1.3em">
  <a href="https://arxiv.org/abs/2512.24601">Full Paper</a> •
  <a href="https://alexzhang13.github.io/blog/2025/rlm/">Blogpost</a> •
  <a href="https://alexzhang13.github.io/rlm/">Documentation</a> •
  <a href="https://github.com/alexzhang13/rlm-minimal">RLM Minimal</a>
</p>

<p align="center">
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/style.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/style.yml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/test.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/test.yml/badge.svg" alt="Test" />
  </a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.24601">
    <img src="media/paper_preview.png" alt="Paper Preview" width="300"/>
  </a>
</p>

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
AGENTS.md                             agent-related documentation
Makefile                              common tasks (install, check, quickstart)
pyproject.toml                        packaging, dependencies, and extras
uv.lock                               lockfile for the uv environment
configs/                              optional configuration files
docs/                                 additional documentation
examples/                              example scripts and small demos
paper/                                 paper source and assets
patches/                               auxiliary patches
rlm/                                   core RLM runtime package (clients, environments, logger, utils)
shrlm/                                  self-harness implementation and spec (surfaces, runner, harness_identity, optimization)
training/                               training harness, environments, and examples (rlm-train, oolong env)
visualizer/                             web visualizer (Node.js + shadcn/ui) for run logs
tests/                                  unit tests and test utilities
```

How it fits together: The `rlm/` package implements the runtime client, REPL environments, logging, and utilities used when running RLMs. The `shrlm/` package provides the self-harness specification (S1–S10), structural checks (I1–I3), and helpers to build and serialize harnesses for reproducible experiments. The `training/` directory holds training harnesses and example environments (including `oolong`) that depend on the core packages. The `visualizer/` is a separate frontend that reads JSONL logs written by `RLMLogger` in order to show call graphs, code, and sub-call traces.

Where to look next:
- rlm/Readme.md — runtime usage and REPL environments
- shrlm/README.md — self-harness specification and examples
- pyproject.toml — packaging, extras for sandboxes (modal, prime, ipython)

## Self-Harness experiments

The Self-Harness experiment (the optimization loop in `shrlm/experiment/`) is driven by
`examples/run_experiment.py`, which loads one profile from one TOML in `configs/`.
Every experiment parameter lives in that TOML — the code hardcodes none of them — and the
loaded config is hashed into an **identity** that is stamped into the output directory.
Re-invoking the same command resumes an interrupted experiment exactly; a changed config
refuses to resume, so every distinct configuration gets its own fresh `--out-dir`.

All shipped configurations run on OpenRouter and require `OPENROUTER_API_KEY` in the
environment (a `.env` file at the repo root works). The launcher refuses to start — and
spends nothing — if a required credential is missing.

### The three configurations

| | Smoke | Qwen (main) | Ox |
|---|---|---|---|
| Config file | `configs/experiment.toml` | `configs/experiment.toml` | `configs/experiment_ox.toml` |
| Profile | `smoke` | `full` (default) | `full` (default) |
| Model (all 3 roles) | `qwen/qwen3-30b-a3b-instruct-2507` | `qwen/qwen3-30b-a3b-instruct-2507` | `stealth/ox-alpha` (free stealth model) |
| Scale | tiny: 3 held-in / 3 held-out instances, 1 round, 2 candidates | 24 / 40 instances, up to 3 rounds, 4 candidates | same instances as Qwen, up to **16 rounds** |
| Purpose | end-to-end plumbing check for cents | the 2026-08-23 comparison run | free-model comparison arm; bounded by wall clock and `patience`, not spend |

The `smoke` profile is the same TOML as the Qwen run with the `[smoke.*]` tables at the
bottom overriding **scale counts only** (instance counts, repetitions, candidate width,
rounds, spend caps); every other semantic is identical. The Ox config is a fork of the
Qwen TOML that changes the role models, `loop.t = 16`, `max_output_tokens` (ox-alpha is a
reasoning model; see the comment in `[decoding]`), zeroes the `[pricing]` tables (the
model bills $0), and gives the attribution/proposal caches `_ox`-suffixed paths so two
experiments running concurrently from this directory never append to the same cache file.

### Exact commands

Append `--dry-run` to any of these first: it checks credentials, loads the config, and
prints the identity and per-role backends without creating anything or spending anything.

```bash
# Smoke: end-to-end plumbing check (small, but spends real money)
uv run python examples/run_experiment.py --profile smoke --out-dir ./experiment_smoke

# Qwen: the main comparison run
uv run python examples/run_experiment.py --out-dir ./experiment_qwen_0823

# Ox: the stealth/ox-alpha free-model arm
uv run python examples/run_experiment.py --config configs/experiment_ox.toml --out-dir ./experiment_ox_0823
```

Interrupting with Ctrl-C loses nothing — re-run the same command and the experiment
resumes at its exact stage boundary. The Qwen and Ox experiments can run concurrently
from the same checkout: they share nothing but the (read-only) datasets and the API key.

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

One caveat for any tool that resolves the config from an out-dir: the default TOML is
`configs/experiment.toml`, and a config that no longer hashes to the identity recorded in
`<out_dir>/config.json` resolves to "unknown" (deliberately — no numbers are invented
from a config the experiment never ran under). For the Ox experiment, always pass
`--config configs/experiment_ox.toml` where the tool accepts it.

## RLM runtime

The `rlm/` package (runtime setup, REPL environments -- local, IPython, Docker, Modal,
Prime -- model providers, trajectory logging, and the visualizer) is documented in
[`rlm/Readme.md`](rlm/Readme.md). The self-harness specification (surfaces S1-S10,
invariants I1-I3) is documented in [`shrlm/README.md`](shrlm/README.md).

## Relevant Reading
* **[Dec '25]** [Recursive Language Models arXiv](https://arxiv.org/abs/2512.24601)
* **[Oct '25]** [Recursive Language Models Blogpost](https://alexzhang13.github.io/blog/2025/rlm/)
* **[Jun '26]** [Self-Harness: Harnesses That Improve Themselves](https://arxiv.org/pdf/2606.09498)
  
If you use this code or repository in your research, please cite:
