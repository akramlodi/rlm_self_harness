# DeepMind MRCR v2 one-round handover

Run this from a checkout containing the `DeepMind_mrcrv2` changes and the
Azure Foundry credentials required by the project. The run needs substantial
local storage: allow at least **50 GB free** before starting. It persists
hundreds-of-megabytes traces for 1--2M-token prompts.

## 1. Cache the released data

From the repository root:

```bash
mkdir -p .cache/shrlm/DeepMind_mrcrv2
curl --fail --location \
  'https://storage.googleapis.com/mrcr_v2/mrcr_v2p1_8needle_in_(1048576,2097152)_dynamic_fewshot_text_style_fast.csv' \
  --output '.cache/shrlm/DeepMind_mrcrv2/mrcr_v2p1_8needle_in_(1048576,2097152)_dynamic_fewshot_text_style_fast.csv'
```

The download is 3,056,474,852 bytes. If interrupted, add `--continue-at -`
to the `curl` command to resume rather than redownloading.

## 2. Verify pricing and run one round

First verify the Azure deployment's input/output prices in Azure Portal. The
configuration expects USD `$0.19/$0.51` per million input/output tokens. Only
after confirming those rates, attest them and start the resumable experiment:

```bash
export SHRLM_VERIFIED_PRICING='0.19/0.51'

PYTHONPATH="$PWD" uv run python examples/run_experiment.py \
  --config configs/experiment_DeepMind_mrcrv2.toml \
  --out-dir ./experiment_DeepMind_mrcrv2
```

This is the real DeepMind 8-needle `in_(1048576,2097152)` bucket. The loader
audits every released row and samples only its 308 parser-clean rows; the two
parser-mismatch rows stay in the cached source CSV but are excluded from splits.

Do not delete `experiment_DeepMind_mrcrv2` while a run is resumable. Reusing
the same `--out-dir` resumes incomplete stages; changing the configuration
requires a fresh output directory.
