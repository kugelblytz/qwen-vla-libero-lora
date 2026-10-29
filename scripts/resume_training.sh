#!/usr/bin/env bash
# Resume the saved first pass of the interrupted two-pass experiment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
DATASET="$(cat "$ROOT/.workspace/libero-object-1cm5deg-dataset.txt")"
CHECKPOINT="$DATASET/training-output/checkpoint-1732"
exec bash "$HERE/pipeline.sh" train-fast --run "$DATASET" --passes 2 --microbatch 32 --resume "$CHECKPOINT" "$@"
