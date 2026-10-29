#!/usr/bin/env bash
# After a training run: held-out changepoint check over its adapter snapshots (frames), pick the best,
# then the frozen 100-attempt LIBERO evaluation of that adapter (episodes). Unattended and resumable.
#
#   sweep_then_evaluate.sh RUN NAME STEP [STEP ...]
#     RUN   training run directory (relative to the repository root or absolute)
#     NAME  prefix for result names, e.g. v6-2b
#     STEP  adapter snapshots to check, e.g. 2500 3000 3500 4000 4500 6928
#
# 1. Waits until RUN/training-output/all_results.json exists (written when training completes).
# 2. Frames: changepoint_sweep.sh into .workspace/changepoint-sweep-NAME/.
# 3. Selects the adapter with the highest accuracy at action changes (lower copy rate breaks ties),
#    written to .workspace/changepoint-sweep-NAME/selection.json.
# 4. Episodes: evaluate.py --attempts 10 --budget native (frozen protocol, no turn guard).
# Rerun the same command after a server stop: finished checks are skipped and an interrupted
# evaluation is resumed (an attempt cut off mid-way is moved aside first, kept as *.interrupted-*).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"  # repository root
S=scripts
RUN="$(realpath "$1")"; NAME="$2"; shift 2; STEPS=("$@")
OUT=.workspace/changepoint-sweep-$NAME
mkdir -p "$OUT"
stamp() { date -u +%H:%M:%S; }

echo "[$(stamp)] waiting for training to finish: $RUN"
# LLaMA-Factory writes all_results.json before its final validation, so also wait until no
# training process for this run is left (it holds the GPU until the very end).
until [ -f "$RUN/training-output/all_results.json" ] && ! pgrep -f "train_resilient.sh.*$(basename "$RUN")" > /dev/null \
      && ! pgrep -f "llamafactory.cli train .*$(basename "$RUN")" > /dev/null; do sleep 60; done
sleep 30
echo "[$(stamp)] training finished"

JOBS=()
for step in "${STEPS[@]}"; do
  [ -d "$RUN/training-output/adapter-$step" ] || { echo "[$(stamp)] missing adapter-$step; skipped"; continue; }
  JOBS+=("$NAME-$step|$RUN|$RUN/training-output/adapter-$step")
done
bash "$S/changepoint_sweep.sh" "$OUT" "${JOBS[@]}"

python3 - "$OUT" "$NAME" <<'PY'
import json, sys
from pathlib import Path
out, name = Path(sys.argv[1]), sys.argv[2]
rows = []
for d in sorted(out.glob(f'{name}-*/summary.json')):
    s = json.loads(d.read_text())
    rows.append(dict(adapter=Path(s['adapter']).name, step=int(d.parent.name.rsplit('-', 1)[1]),
                     changepoint_accuracy=s['changepoint_accuracy'], changepoint_copy_rate=s['changepoint_copy_rate'],
                     continue_accuracy=s['groups'].get('continue', {}).get('accuracy')))
assert rows, 'no changepoint results'
best = max(rows, key=lambda r: (r['changepoint_accuracy'], -r['changepoint_copy_rate']))
(out / 'selection.json').write_text(json.dumps(dict(rule='max changepoint_accuracy, then min copy rate',
                                                    selected=best, candidates=rows), indent=1) + '\n')
for r in rows: print(r)
print('selected', best['adapter'])
PY
[ -f "$OUT/selection.json" ] || { echo "[$(stamp)] no adapter could be checked; not evaluating. See $OUT/*.server.log"; exit 1; }
BEST="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['selected']['adapter'])" "$OUT/selection.json")"
ADAPTER="$RUN/training-output/$BEST"

for i in $(seq 1 24); do
  python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 8000)); s.close()' 2>/dev/null && break
  sleep 5
done
if [ -f "$OUT/evaluation.txt" ]; then
  EVAL="$(cat "$OUT/evaluation.txt")"
  for d in "$EVAL"/task-*-attempt-*; do
    [ -d "$d" ] && [ ! -f "$d/trajectory.mp4" ] && mv "$d" "$d.interrupted-$(date -u +%Y%m%dT%H%M%S)" && echo "[$(stamp)] moved aside $(basename "$d")"
  done
  echo "[$(stamp)] resuming evaluation $EVAL"
  bash "$S/pipeline.sh" evaluate --run "$RUN" --adapter "$ADAPTER" --attempts 10 --budget native --resume-evaluation "$EVAL"
else
  echo "[$(stamp)] evaluating $BEST"
  before="$(ls -d "$RUN"/eval-native-2* 2>/dev/null | tail -1)"
  bash "$S/pipeline.sh" evaluate --run "$RUN" --adapter "$ADAPTER" --attempts 10 --budget native &
  ev=$!
  for i in $(seq 1 60); do
    now="$(ls -d "$RUN"/eval-native-2* 2>/dev/null | tail -1)"
    [ -n "$now" ] && [ "$now" != "$before" ] && { echo "$now" > "$OUT/evaluation.txt"; break; }
    sleep 5
  done
  wait "$ev"
fi
echo "[$(stamp)] evaluation exited $?; directory $(cat "$OUT/evaluation.txt" 2>/dev/null)"
