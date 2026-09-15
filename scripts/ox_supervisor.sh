#!/bin/bash
# Supervisor for the OX experiment.
#
# Polls every POLL_SECONDS. While the run is alive it records a heartbeat.
# When the run exits it decides between three outcomes:
#
#   completed  -- the log contains the orchestrator's "Frozen harness:" line.
#                 Stop supervising; the experiment is done.
#   refused    -- the log contains a pre-flight "Declining to run:" line.
#                 A restart would be refused identically, so stop and alert.
#   crashed    -- anything else. Restart, subject to the guards below.
#
# Guards, in the order they are checked:
#
#   disk       -- the 2026-08-25 failure was ENOSPC (Errno 28), not OOM: workers
#                 died mid-write and one could not even write its own worker.log.
#                 Restarting into a full disk re-runs paid work that cannot
#                 persist, so a restart is refused below MIN_FREE_KB.
#   crash loop -- a run that dies within MIN_HEALTHY_SECONDS of launch counts as
#                 a fast failure. MAX_FAST_FAILURES consecutive fast failures
#                 stop the supervisor rather than burn budget in a tight loop.
#                 A run that survives longer resets the counter.
#
# The experiment is resumable (persist-first): re-invoking with the same
# --out-dir replays completed rounds and resumes an interrupted round at its
# stage boundary, so a restart never repeats persisted runs.
#
# Usage:
#   nohup ./scripts/ox_supervisor.sh > /dev/null 2>&1 &
#   tail -f experiment_ox.supervisor.log
#   ./scripts/ox_supervisor.sh --status
#   ./scripts/ox_supervisor.sh --stop

set -uo pipefail

PROJECT_DIR="/Users/williamstanford/RESEARCH/rlm_self_harness"
CONFIG="configs/experiment_ox.toml"
OUT_DIR="./experiment_ox_full"
RUN_LOG="experiment_ox.log"
SUP_LOG="experiment_ox.supervisor.log"
PID_FILE=".ox_supervisor.pid"

POLL_SECONDS=120
MIN_FREE_KB=$((15 * 1024 * 1024))   # 15 GB; refuse to restart below this
MIN_HEALTHY_SECONDS=300             # a run dying sooner is a "fast failure"
MAX_FAST_FAILURES=3

cd "$PROJECT_DIR" || exit 1

# Matches ONLY this experiment. Both the config AND the out-dir are needed as
# discriminators: a second run was launched from the same TOML against a
# different --out-dir, so the config filename alone is ambiguous. Never match on
# "run_experiment" alone, which would also catch the concurrent Kimi run.
PATTERN="run_experiment.py --config $CONFIG --out-dir $OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$SUP_LOG"; }

run_pid() { pgrep -f "$PATTERN" | head -1; }

free_kb() { df -k /System/Volumes/Data | tail -1 | awk '{print $4}'; }

runs_done() { find "$OUT_DIR" -path '*/runs/*.json' 2>/dev/null | wc -l | tr -d ' '; }

human_gb() { awk -v k="$1" 'BEGIN{printf "%.1f", k/1024/1024}'; }

case "${1:-}" in
  --stop)
    if [ -f "$PID_FILE" ]; then
      kill "$(cat "$PID_FILE")" 2>/dev/null && echo "supervisor stopped (pid $(cat "$PID_FILE"))"
      rm -f "$PID_FILE"
    else
      echo "no supervisor pid file"
    fi
    exit 0
    ;;
  --status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "supervisor: alive (pid $(cat "$PID_FILE"))"
    else
      echo "supervisor: not running"
    fi
    pid=$(run_pid)
    [ -n "$pid" ] && echo "experiment: alive (pid $pid)" || echo "experiment: NOT running"
    echo "runs persisted: $(runs_done)"
    echo "disk free: $(human_gb "$(free_kb)") GB"
    echo "--- last supervisor lines ---"
    tail -5 "$SUP_LOG" 2>/dev/null
    exit 0
    ;;
esac

# Refuse to run two supervisors against one experiment.
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "supervisor already running (pid $(cat "$PID_FILE"))" >&2
  exit 1
fi
echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

fast_failures=0
restarts=0
last_launch=$(date +%s)

log "supervisor started (pid $$), polling every ${POLL_SECONDS}s"
log "guards: min free ${MIN_FREE_KB}KB ($(human_gb $MIN_FREE_KB) GB), max $MAX_FAST_FAILURES fast failures"
log "watching: $PATTERN"

while true; do
  pid=$(run_pid)

  if [ -n "$pid" ]; then
    avail=$(free_kb)
    log "alive pid=$pid runs=$(runs_done) free=$(human_gb "$avail")GB restarts=$restarts"

    # Warn early: the disk filling while the run is alive is the failure that
    # already happened once, and it is silent until a worker dies mid-write.
    if [ "$avail" -lt "$MIN_FREE_KB" ]; then
      log "WARNING low disk: $(human_gb "$avail")GB free, below the $(human_gb $MIN_FREE_KB)GB floor -- ENOSPC killed this experiment on 2026-08-25"
    fi

    sleep "$POLL_SECONDS"
    continue
  fi

  # --- the run is gone; classify why -------------------------------------
  now=$(date +%s)
  uptime=$(( now - last_launch ))

  if tail -40 "$RUN_LOG" 2>/dev/null | grep -q "Frozen harness:"; then
    log "COMPLETE: experiment finished; $(tail -3 "$RUN_LOG" | tr '\n' ' ')"
    log "supervisor exiting (nothing left to supervise)"
    exit 0
  fi

  if tail -40 "$RUN_LOG" 2>/dev/null | grep -q "Declining to run:"; then
    log "STOP: pre-flight refused -- a restart would be refused identically"
    log "$(grep 'Declining to run:' "$RUN_LOG" | tail -1)"
    exit 1
  fi

  avail=$(free_kb)
  if [ "$avail" -lt "$MIN_FREE_KB" ]; then
    log "STOP: refusing restart -- only $(human_gb "$avail")GB free, floor is $(human_gb $MIN_FREE_KB)GB"
    log "the prior crash was ENOSPC; restarting now would spend money on runs that cannot persist. Free space, then relaunch."
    exit 1
  fi

  if [ "$uptime" -lt "$MIN_HEALTHY_SECONDS" ]; then
    fast_failures=$(( fast_failures + 1 ))
    log "fast failure $fast_failures/$MAX_FAST_FAILURES (died after ${uptime}s)"
    if [ "$fast_failures" -ge "$MAX_FAST_FAILURES" ]; then
      log "STOP: $MAX_FAST_FAILURES consecutive fast failures -- crash loop, not a transient fault"
      log "last log lines: $(tail -5 "$RUN_LOG" | tr '\n' ' | ')"
      exit 1
    fi
  else
    fast_failures=0
  fi

  restarts=$(( restarts + 1 ))
  log "RESTART #$restarts (previous run lasted ${uptime}s, $(runs_done) runs persisted, $(human_gb "$avail")GB free)"
  log "exit context: $(tail -3 "$RUN_LOG" | tr '\n' ' | ')"

  # Append, never truncate: the crash that caused the restart stays readable.
  nohup uv run python examples/run_experiment.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" >> "$RUN_LOG" 2>&1 &

  last_launch=$(date +%s)
  sleep 30
  newpid=$(run_pid)
  if [ -n "$newpid" ]; then
    log "restart ok, pid=$newpid"
  else
    log "restart did NOT produce a live process; will re-evaluate next poll"
  fi
  sleep "$POLL_SECONDS"
done
