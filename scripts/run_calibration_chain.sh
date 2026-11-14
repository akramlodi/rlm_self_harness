#!/bin/bash
# Chains the 2-round OBLIQ-Bench Math calibration probe: stage 1 (pool=200)
# then stage 2 (pool=500), each via examples/obliq_two_stage/run_calibration.py.
#
# Resumable by construction, not by any special-casing in this script:
#   - freeze_stage_splits() (examples/obliq_two_stage/freeze_splits.py) writes
#     byte-identical split files every time it's called -- deterministic from
#     partition.json -- so re-freezing on every invocation is a no-op once
#     the files already exist.
#   - run_experiment() is persist-first: re-invoking with the same --out-dir
#     replays already-completed rounds and resumes an interrupted round at
#     its stage boundary (the same guarantee scripts/ox_supervisor.sh relies
#     on) -- it never repeats paid work.
# So simply re-running this script after a crash, a kill, or a reboot
# continues exactly where it left off: stage 1 won't restart once its
# harness.json is frozen, and a stage already fully done exits near-instantly
# with zero new spend.
#
# Usage:
#   ./scripts/experiment_obliq_bench/run_calibration_chain.sh
#   nohup caffeinate -i ./scripts/experiment_obliq_bench/run_calibration_chain.sh \
#       >> logs/obliq_calibration_chain.log 2>&1 &
#
# After any interruption, rerun exactly the same way -- no flags, no cleanup.
# Do NOT run this concurrently with another invocation of the same chain
# (two processes writing rounds into the same --out-dir at once will race).

set -euo pipefail

PROJECT_DIR="/Users/jaloliddin/Projects/rlm_self_harness"
cd "$PROJECT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== calibration chain: stage 1 (pool=200) ==="
uv run python examples/obliq_two_stage/run_calibration.py --stage 1
log "=== calibration chain: stage 1 done, starting stage 2 (pool=500) ==="
uv run python examples/obliq_two_stage/run_calibration.py --stage 2
log "=== calibration chain: stage 2 done, calibration complete ==="
