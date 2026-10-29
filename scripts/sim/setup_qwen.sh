#!/usr/bin/env bash
# Install the isolated serving stack; no model weights are downloaded.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNTIME_DIR="${VLA_WORKSPACE:-$PROJECT_DIR/.workspace}"
QWEN_VENV="${QWEN_VENV:-$RUNTIME_DIR/qwen-venv}"
UV_BIN="${UV_BIN:-uv}"
command -v "$UV_BIN" >/dev/null || { echo 'Install uv first: python -m pip install --user uv' >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || { echo 'This setup supports Linux CUDA machines.' >&2; exit 1; }
case "$(uname -m)" in
  x86_64) WHEEL_ARCH=x86_64 ;;
  aarch64) WHEEL_ARCH=aarch64 ;;
  *) echo 'No supported official wheel for this architecture.' >&2; exit 1 ;;
esac
command -v nvidia-smi >/dev/null || { echo 'A visible NVIDIA driver is required.' >&2; exit 1; }
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d ' ')"
DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
[[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] || { echo 'Cannot determine NVIDIA driver version.' >&2; exit 1; }
CUDA_PROFILE="${QWEN_CUDA_PROFILE:-auto}"
if [[ "$CUDA_PROFILE" == auto ]]; then
  if (( DRIVER_MAJOR >= 580 )); then CUDA_PROFILE=cu130
  elif (( DRIVER_MAJOR >= 525 )); then CUDA_PROFILE=cu129
  else echo 'Driver too old for the supported CUDA 12/13 wheels; update the driver.' >&2; exit 1
  fi
fi
case "$CUDA_PROFILE" in
  cu129) MIN_DRIVER=525; WHEEL_SUFFIX='+cu129' ;;
  cu130) MIN_DRIVER=580; WHEEL_SUFFIX='' ;;
  *) echo 'QWEN_CUDA_PROFILE must be auto, cu129, or cu130.' >&2; exit 1 ;;
esac
(( DRIVER_MAJOR >= MIN_DRIVER )) || { echo "Driver $DRIVER_VERSION does not support $CUDA_PROFILE." >&2; exit 1; }
WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0${WHEEL_SUFFIX}-cp38-abi3-manylinux_2_28_${WHEEL_ARCH}.whl"
LOCK="$PROJECT_DIR/serving/requirements-${CUDA_PROFILE}-${WHEEL_ARCH}.lock"
mkdir -p "$RUNTIME_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUNTIME_DIR/.cache/uv}"
if [[ ! -x "$QWEN_VENV/bin/python" ]]; then
  "$UV_BIN" venv --managed-python --python 3.11 "$QWEN_VENV"
fi
"$QWEN_VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use a separate Python 3.11 venv"'
# The checked-in lock is the tested profile. For other profiles resolve once and retain
# that machine's full lock locally rather than editing the reference artifacts.
if [[ ! -f "$LOCK" ]]; then
  LOCK="$RUNTIME_DIR/requirements-${CUDA_PROFILE}-${WHEEL_ARCH}.lock"
  if [[ ! -f "$LOCK" ]]; then
    "$UV_BIN" pip compile "$PROJECT_DIR/serving/requirements.in" \
    --python "$QWEN_VENV/bin/python" --torch-backend "$CUDA_PROFILE" \
    --output-file "$LOCK" --no-header --no-annotate --no-emit-index-url \
    <(printf 'vllm @ %s\n' "$WHEEL_URL")
  fi
fi
"$UV_BIN" pip sync --python "$QWEN_VENV/bin/python" --torch-backend "$CUDA_PROFILE" "$LOCK"
"$UV_BIN" pip check --python "$QWEN_VENV/bin/python"
"$QWEN_VENV/bin/python" "$SCRIPT_DIR/check_qwen_environment.py" \
  --output "$RUNTIME_DIR/qwen-environment.json"
printf 'Serving environment: %s\nCUDA profile: %s\nLock: %s\n' "$QWEN_VENV" "$CUDA_PROFILE" "$LOCK"
