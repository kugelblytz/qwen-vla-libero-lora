#!/usr/bin/env bash
# Keep one training run going across crashes. Arguments are passed to `pipeline.sh train-fast`
# (give --run explicitly; do not pass --resume). Each launch resumes from the newest COMPLETE
# checkpoint (train.py --resume latest), restoring adapter, optimizer, scheduler and RNG state.
# The same arguments every launch keep the schedule identical (same --passes = same max_steps).
#
#   exit 0   training reached its configured updates
#   exit 75  another process holds the run's training lock (nothing started)
#   exit 1   configuration invalid, or MAX_RESTARTS consecutive failures
#
# If the whole server/pod restarts, this script dies with it: start the same command again.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
WAIT_S="${WAIT_S:-120}"
RUN=""
prev=""
for arg in "$@"; do
  case "$arg" in --resume*) echo "Do not pass --resume; train_resilient.sh always uses --resume latest" >&2; exit 1 ;; esac
  [ "$prev" = "--run" ] && RUN="$arg"
  prev="$arg"
done
[ -n "$RUN" ] || { echo "Pass --run explicitly; without it train-fast uses the v5 dataset pointer" >&2; exit 1; }
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
newest() { ls -d "$RUN"/training-output/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1; }

echo "[$(stamp)] dry run"
bash "$HERE/pipeline.sh" train-fast "$@" --resume latest --dry-run || { echo "[$(stamp)] dry run failed; not starting" >&2; exit 1; }

failures=0
while true; do
  before="$(newest)"
  echo "[$(stamp)] launch from checkpoint ${before:-none} (consecutive failures so far: $failures)"
  bash "$HERE/pipeline.sh" train-fast "$@" --resume latest
  code=$?
  case "$code" in
    0)  echo "[$(stamp)] training complete"; exit 0 ;;
    75) echo "[$(stamp)] run is locked by another training process; stopping" >&2; exit 75 ;;
  esac
  after="$(newest)"
  if [ "${after:-0}" -gt "${before:-0}" ]; then failures=1; else failures=$((failures+1)); fi
  echo "[$(stamp)] trainer exited with $code"
  if [ "$failures" -ge "$MAX_RESTARTS" ]; then echo "[$(stamp)] giving up after $failures failures" >&2; exit 1; fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed "s/^/[$(stamp)] GPU memory: /"
  sleep "$WAIT_S"
done
