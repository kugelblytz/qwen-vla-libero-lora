#!/usr/bin/env bash
# Machine-relative launchers; uses the simulator and official trainer environments already set up in .workspace/
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
MODE="${1:?Use prepare, export-variant, train, train-fast, train-resilient, setup-fast, probe, benchmark, status, evaluate, check-controls, check-checkpoint or check-changepoints}"
shift
SIM="${SIM_PYTHON:-$ROOT/.workspace/LIBERO/.venv/bin/python}"
TRAIN="${TRAIN_PYTHON:-$ROOT/.workspace/lf-paper-venv/bin/python}"
PY="${PIPELINE_PYTHON:-python}"
case "$MODE" in
  prepare) exec "$PY" -u "$HERE/prepare.py" "$@" ;;
  export-variant) exec "$TRAIN" -u "$HERE/export_variant.py" "$@" ;;
  train) exec "$TRAIN" -u "$HERE/train.py" "$@" ;;
  train-fast) REQUIRE_FAST_KERNELS=1 exec "$ROOT/.workspace/lf-fast-venv/bin/python" -u "$HERE/train.py" "$@" ;;
  train-resilient) exec bash "$HERE/train_resilient.sh" "$@" ;;
  setup-fast) exec "$PY" -u "$HERE/setup_fast_kernels.py" "$@" ;;
  probe) exec "$TRAIN" -u "$HERE/train.py" --probe "$@" ;;
  benchmark) exec "$TRAIN" -u "$HERE/benchmark.py" "$@" ;;
  status) exec "$PY" -u "$HERE/status.py" "$@" ;;
  evaluate) exec "$PY" -u "$HERE/evaluate.py" "$@" ;;
  check-checkpoint) exec "$ROOT/.workspace/qwen-venv/bin/python" -u "$HERE/check_checkpoint.py" "$@" ;;
  check-changepoints) exec "$ROOT/.workspace/qwen-venv/bin/python" -u "$HERE/check_changepoints.py" "$@" ;;
  check-controls)
    exec "$PY" -u - "$SIM" "$HERE/check_controls.py" "$@" <<'PY'
import runpy,subprocess,sys
from pathlib import Path
contract=runpy.run_path(str(Path(sys.argv[2]).with_name('contract.py')))
raise SystemExit(subprocess.run([sys.argv[1],'-u',sys.argv[2],*sys.argv[3:]],
    env=contract['simulator_env'](Path.cwd())).returncode)
PY
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
