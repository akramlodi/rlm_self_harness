# Dataset inventory and split sizing

Row counts measured directly from the pinned Hugging Face snapshots cached
locally on 2026-08-19:

- GraphWalks: `openai/graphwalks` @ `f338bb265735a56a79f4b0f5def722c9c3268ead`
- OOLONG-Pairs: `oolongbench/oolong-synth` @ `f0d59eaf0febf130664cfceb710436c8e3216b2b`

Both revisions are the ones pinned in `configs/experiment.toml`.

## Raw row counts

| source file | raw rows |
|---|---|
| `graphwalks_128k_and_shorter.parquet` | 750 |
| `graphwalks_256k_to_1mil.parquet` | 400 |
| `oolong-synth` validation split | 1,300 (650 `trec_coarse`, 650 `spam`) |
| `oolong-synth` test split | 5,200 (0 `trec_coarse`) |

## Rows that pass our filters

| source file | raw rows | filter applied | eligible |
|---|---|---|---|
| `graphwalks_128k_and_shorter.parquet` | 750 | `prompt_chars <= 128,000`, `problem_type` in {bfs, parents} | **650** (300 bfs / 350 parents) |
| `graphwalks_256k_to_1mil.parquet` | 400 | `prompt_chars >= 0`, no upper cap, same types | **400** (200 bfs / 200 parents) |
| `oolong-synth` validation | 1,300 | `dataset == trec_coarse`, `context_len == 8192` | **40** (2 windows x 20 tasks) |
| `oolong-synth` validation | 1,300 | `dataset == trec_coarse`, `context_len == 262144` | **40** (2 windows x 20 tasks) |
| `oolong-synth` test | 5,200 | `dataset == trec_coarse` | **0** — subset absent from this split |

Notes:

- The GraphWalks "128k and shorter" file is loosely named: 100 of its 750 rows
  carry `prompt_chars` above 128,000 (max 219,451), and our configured
  `max_chars = 128000` excludes them.
- GraphWalks short and long come from separate files, so they never compete for
  rows. The two OOLONG context lengths are separate rows as well, so the pools
  are 40 and 40, not 40 shared.
- OOLONG-Pairs instances are (context window x task). Within one context length
  the `trec_coarse` subset carries 50 rows but only **2 distinct
  `context_window_id` values**, and our loader generates 20 tasks per window
  (`task_ids = [1..20]`), giving a hard ceiling of 40 instances per context
  length. `max_scan` is not the limiting factor: the whole validation split is
  1,300 rows and all of it is read.

## Suggested splits

| dataset | length | role | rows |
|---|---|---|---|
| GraphWalks | short | held_in | 24 |
| GraphWalks | short | held_out | 40 |
| GraphWalks | short | test | 40 |
| GraphWalks | long | test | 150 |
| OOLONG-Pairs | short | test | 40 |
| OOLONG-Pairs | long | test | **40** |

Only one value changes from the preregistered full profile: OOLONG-Pairs long
test drops from 150 to 40, which is the entire available pool. Every other size
is unchanged and sits well inside its pool — GraphWalks short draws 104 of 650,
GraphWalks long draws 150 of 400.

## Required config change

`configs/experiment.toml`, in `[environments.oolong_pairs]`:

```toml
n_long = 40 # target-long test size (upstream ceiling: 2 windows x 20 tasks)
```

Without this change the full profile fails at split materialization, before any
spend, with:

```
ValueError: insufficient distinct trec_coarse windows at context length(s) [262144]:
need 8 per length, but the first 50000 rows of oolongbench/oolong-synth:validation
ship only {262144: 2}
```

`n_short = 40` passes, but with no slack: it consumes the full pool of 2 windows
x 20 tasks.

## Cost impact

The full-experiment projection in `experiment_smoke/report_full_profile.md`
sized its long evaluation leg at 3 conditions x (150 GraphWalks + 150 OOLONG) x
3 repetitions = 2,700 runs. At the corrected OOLONG size that becomes
3 x (150 + 40) x 3 = **1,710 runs**, so the report needs a re-render; the point
estimate falls from $77.40 by roughly the 37% cut to that leg.

## If 150 target-long instances are required

The only lever is more tasks per window, and `TASK_TEXTS` in
`shrlm/environments/oolong_pairs.py` defines exactly 20. Reaching 150 would mean
authoring new task templates, which changes the benchmark. Alternatives that do
not help: raising `max_scan` (the pool is not scan-limited), switching to the
test split (no `trec_coarse` rows), or choosing a different context length
(every length has the same 2-window ceiling).
