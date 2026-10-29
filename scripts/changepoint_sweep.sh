#!/usr/bin/env bash
# Held-out changepoint check over several adapters, one served at a time (single GPU slice).
# Resumable: adapters with an existing summary.json are skipped, so rerun after a server stop.
#
#   changepoint_sweep.sh                          the 4B v5/v6 sweep recorded in results-v6.md
#   changepoint_sweep.sh OUT name|run|adapter ... any other set (OUT relative to the repository root)
# The serving profile comes from each run's dataset-manifest.json (4B or 2B).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"  # repository root
S=scripts
SERVE=scripts/sim/serve_qwen.sh
if [ $# -ge 2 ]; then
  OUT="$1"; shift; JOBS=("$@")
else
  OUT=.workspace/changepoint-sweep
  SOURCE=.workspace/libero-object-1cm5deg-20261009T203209-239977
  V6="$(cat .workspace/libero-object-v6-dataset.txt)"
  JOBS=(
    "v5-3464|$SOURCE|$SOURCE/training-output/checkpoint-3464"
    "v6-6928|$V6|$V6/training-output/adapter-6928"
    "v6-4000|$V6|$V6/training-output/adapter-4000"
    "v6-3500|$V6|$V6/training-output/adapter-3500"
    "v6-3000|$V6|$V6/training-output/adapter-3000"
  )
fi
mkdir -p "$OUT"
stamp() { date -u +%H:%M:%S; }
for job in "${JOBS[@]}"; do
  IFS='|' read -r name run adapter <<< "$job"
  adapter="$(realpath "$adapter")"
  if [ -f "$OUT/$name/summary.json" ]; then echo "[$(stamp)] $name already done"; continue; fi
  rm -rf "$OUT/$name"
  if curl -sf http://127.0.0.1:8000/health > /dev/null; then
    echo "[$(stamp)] another server already answers on port 8000; stop it first (ps -eww | grep -i vllm)"; exit 1
  fi
  # A just-stopped server can hold the port for a while; wait until it can be bound.
  for i in $(seq 1 24); do
    python3 -c 'import socket; s=socket.socket(); s.bind(("0.0.0.0", 8000)); s.close()' 2>/dev/null && break
    sleep 5
  done
  echo "[$(stamp)] $name: serving $adapter"
  profile="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('serve_profile','4b-libero-v5'))" "$run/dataset-manifest.json")"
  QWEN_PROFILE="$profile" LIBERO_ADAPTER="$adapter" PORT=8000 setsid bash "$SERVE" > "$OUT/$name.server.log" 2>&1 &
  server=$!
  ready=0
  for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null; then ready=1; break; fi
    kill -0 "$server" 2>/dev/null || break
    sleep 5
  done
  if [ "$ready" = 1 ]; then
    echo "[$(stamp)] $name: server ready, checking"
    bash "$S/pipeline.sh" check-changepoints --run "$run" --adapter "$adapter" --out "$OUT/$name" > "$OUT/$name.log" 2>&1
    echo "[$(stamp)] $name: check exited $?"
  else
    echo "[$(stamp)] $name: server did not become ready; see $OUT/$name.server.log"
  fi
  kill -- -"$server" 2>/dev/null; wait "$server" 2>/dev/null; sleep 15
done
echo "[$(stamp)] sweep finished"
