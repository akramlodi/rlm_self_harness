
## Context

We have an existing `mrcrv2` environment at `/Users/mohammedakram/Desktop/rlm_self_harness/rlm_self_harness/shrlm/environments/mrcrv2.py` that generates its
own synthetic instances in-process (see `configs/experiment_mrcrv2.toml`). A smoke test on it (10
held-in / 10 held-out, 65K/2-needle short split) scored 100% accuracy — too easy to be a useful
optimization-loop environment.

**Leave `mrcrv2.py` and `configs/experiment_mrcrv2.toml` completely untouched.** We're adding a second,
separate environment that pulls DeepMind's actual released MRCR v2 data instead of generating our own,
so we can compare real difficulty against our synthetic version. Name it `DeepMind_mrcrv2` everywhere
(module name, `[loop] environment` key, config filename, cache paths) so the two never collide.

**Reference implementation to pull from:** `https://github.com/google-deepmind/eval_hub`, specifically
the `eval_hub/mrcr_v2/` subdirectory. Cite `https://arxiv.org/abs/2409.12640v2` (the Michelangelo paper)
per that repo's README if this shows up in any writeup.

**Mirror the structure of our own OOLONG environment** at `/Users/mohammedakram/Desktop/rlm_self_harness/rlm_self_harness/shrlm/environments/oolong.py` (or
wherever the base OOLONG module lives) for the plugin interface shape — loader, root verifier,
sub-verifier, split builder should have the same public methods/signatures as that module, not a novel shape.

## What DeepMind's real MRCR v2 actually looks like (confirmed from their README and run_evaluation.py)

- **Task**: a long sequence of user/assistant turns. Each user turn requests a piece of writing given a
  (format, topic, style) triplet; the assistant turn is a distinct response satisfying it. The same
  triplet gets asked multiple times across the conversation (that's the needle count — e.g. 8 needles =
  8 distinct assistant responses to variations of what looks like the same request). At the end, the
  model is asked to reproduce the *i*-th assistant response to one specific query, and must **first
  output a unique 12-character random hash string**, then the reproduced content.
- **Data format**: pre-packaged as CSV with exactly two columns, `queries` (the full flattened prompt)
  and `answer` (the 12-char hash concatenated with the target content). This is flattened text, not a
  structured object with explicit needle offsets.
- **Official metric** (`mrcr_v2_metric` in their `run_evaluation.py`) — **port this exactly, do not
  reimplement from description alone**:
  1. If prediction is empty/non-string → 0.0.
  2. Target's first 12 characters are the hash; the rest (stripped) is the reference text.
  3. Find the **last** occurrence of that 12-char hash in the prediction; if absent → 0.0.
  4. Take everything after that last hash occurrence in the prediction (stripped) as the candidate content.
  5. Score = `difflib.SequenceMatcher(a=target_ref, b=prediction_content).ratio()` — continuous in [0, 1],
     not binary. Decide and document how this maps to whatever pass/fail threshold our promotion rule
     expects (check the threshold convention `oolong_pairs` or `oolong` actually use for their own
     continuous scores, and match it — don't invent a new convention).
- **Noise rates** (from their README, useful for sanity-checking our own numbers aren't inflated): a
  model that reproduces a uniformly random *relevant* needle scores ~51% at 2-needle, ~27% at 4-needle,
  ~15% at 8-needle under this metric. If our pilot's 8-needle score comes in notably above ~15% but the
  decomposition log shows no real cross-checking happened, that's a signal the harness got lucky or
  partially credit-matched rather than actually resolving the count — flag this explicitly in the report,
  don't just report the raw score.
- **Context length**: selectable via their `download.sh` by `num_needles` and max context length; their
  README states published buckets include "upto_128K" (cumulative) and "at_1M" (pointwise), with the
  full release scaling up to 8M. **We have not confirmed there is an exact 2,097,152-token bucket.**

## Step 1 — Inspect and report back before building anything

Clone `google-deepmind/eval_hub`, look at `eval_hub/mrcr_v2/download.sh` and any generator/library code
in that directory (the README mentions test and visualization utilities for "context and needle position
distributions," implying there's a generator module with more structure than the flat CSV — find it).

Report back, in this order, before writing the environment module:

1. **Available context-length buckets for `num_needles=8`**, listed explicitly. If an exact 2M-token
   bucket doesn't exist, stop here and tell me the real options rather than picking one yourself — this
   is a deliberate checkpoint, not an oversight.
2. **Whether the generator/library code exposes structured per-instance metadata** (needle turn offsets,
   which triplet each needle answers, instance index) independent of the flattened CSV. If yes, describe
   the interface. If the only usable form is the flat `queries`/`answer` CSV, say so plainly.
3. Based on #2, **propose a sub-verification approach** and wait for my go-ahead before implementing it:
   - If structured metadata is available: build a real sub-verifier the way we discussed for the
     synthetic version — given a sub-call's slice offsets, compute which needles fall inside it, label
     the sub-call's local claim correct/incorrect/uncheckable.
   - If only the flat CSV is available: propose a concrete approximation (e.g., re-deriving approximate
     turn boundaries by locating each recorded needle's known target text within the flattened prompt at
     load time, since we do have the ground-truth answer text even without official offsets) and be
     explicit about what it can't catch.

**Do not proceed past this point until you've reported these three things and I've responded.**

## Step 2 — Build the environment (after Step 1 is confirmed)

- `[PATH_TO_ENVIRONMENTS_FOLDER]/deepmind_mrcrv2.py` — loader (real download, pinned to an actual commit
  SHA, not a placeholder), root verifier (exact port of `mrcr_v2_metric` above), sub-verifier (per the
  approach confirmed in Step 1), split builder for held-in/held-out (short pool) and test-long (8-needle
  pool at whatever bucket Step 1 confirmed).
- Confirm the runtime batch cap is set below the chosen long-split token count, same reasoning as before:
  we want genuine decomposition, not a single sub-call swallowing the whole context.
- Log single-pass-through vs. multi-chunk decomposition per run, same as our other environments.

## Step 3 — Create the new config file

Create `configs/experiment_DeepMind_mrcrv2.toml`, mirroring `configs/experiment_mrcrv2.toml` (attached
separately) almost exactly — same DeepSeek-V4-Flash backend tables, same `[decoding]`/`[promotion]`
shape, same inert environment tables kept for schema uniformity (`graphwalks`, `oolong_pairs`, `oolong`,
and the existing `mrcrv2` table itself, all untouched) — but:

- `[loop] environment = "DeepMind_mrcrv2"`, `initial_harness` unchanged.
- `[loop] t = 1` — this is one real, detailed round, not a 1-2 round pilot ceiling.
- `[splits] n_in = 10`, `n_ho = 10` (confirmed). Leave `test_short`/`test_long` for you to fill once Step
  1's real pool sizes are known — don't guess these.
- `[environments.DeepMind_mrcrv2]`: real dataset source info (repo, subpath, pinned commit SHA — no
  `TODO` placeholders left in the final file), `num_needles_short`/`num_needles_long`, the actual
  confirmed context-length bucket(s) from Step 1, and whatever generator/download parameters Step 1 surfaced.
- `[caps]`: treat `max_budget`/`max_timeout`/`candidate_budget` as unverified starting points scaled up
  from the synthetic `experiment_mrcrv2.toml` values (real DeepMind data may have longer/harder
  distractor writing than our synthetic generator) — say explicitly in a comment that these need
  correcting from this round's real observed spend, don't present them as calibrated.
- Update `attribution_cache_path` / `proposal_cache_path` to `..._deepmind_mrcrv2.jsonl`.

## Step 4 — Run the one round and report

Run the single real optimization round (weakness mining → harness proposal → proposal validation) on
the 10/10 held-in/held-out split at the confirmed 8-needle long-context bucket. Report:

- Root verifier score distribution (it's continuous — give me mean/median/range, not just a pass rate),
  compared against the ~15% 8-needle noise floor from the README.
- Sub-verifier coverage and correct/incorrect/uncheckable breakdown.
- Weakness clusters mined, candidate proposal(s), and the promote/reject decision on held-out.
- Single-pass-through vs. decomposed call count.
- Real $ cost for the round, from actual token usage.
- The exact command to reproduce this run from scratch.

If anything in Step 1 changes the shape of this plan materially (e.g. no bucket near 2M exists at all),
stop and tell me before spending on Step 4.