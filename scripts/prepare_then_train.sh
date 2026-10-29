#!/usr/bin/env bash
# Prepare the full dataset, then start the tested two-pass fast training configuration.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
mkdir -p .workspace
exec 9> .workspace/libero-object-prepare-then-train.lock
flock -n 9
RUN="$ROOT/.workspace/libero-object-1cm5deg-$(date -u +%Y%m%dT%H%M%S)-$$"
printf '%s\n' "$RUN" > .workspace/libero-object-active-run.txt
echo "Preparing full ten-task dataset: $RUN"
echo "Preparation log: $ROOT/.workspace/libero-object-preparation.log"
bash "$HERE/pipeline.sh" prepare --run "$RUN" > .workspace/libero-object-preparation.log 2>&1
echo "Preparation succeeded. Checking optimized training configuration."
bash "$HERE/pipeline.sh" train-fast --run "$RUN" --passes 2 --microbatch 32 --dry-run
echo "Starting two-pass training. Log: $ROOT/.workspace/libero-object-fast-training.log"
printf '%s\n' "$$" > .workspace/libero-object-training.pid
exec bash "$HERE/pipeline.sh" train-fast --run "$RUN" --passes 2 --microbatch 32 > .workspace/libero-object-fast-training.log 2>&1
