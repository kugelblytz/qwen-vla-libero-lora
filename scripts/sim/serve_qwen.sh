#!/usr/bin/env bash
# Serve pinned local weights using the pinned upstream adapter interface.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNTIME_DIR="${VLA_WORKSPACE:-$PROJECT_DIR/.workspace}"
SHOW_HARNESS="${SHOW_HARNESS:-$RUNTIME_DIR/Show-Harness}"
case "${QWEN_PROFILE:-2b-sim}" in
  2b-sim) MODEL_ROOT=models; MODEL_MANIFEST=qwen-models.json; BASE_NAME=Qwen/Qwen3.5-2B; ADAPTER_FOLDER=qwen3_5_2b_sim; ADAPTER_NAME=qwen3_5_2b_showharness_sim ;;
  4b-real) MODEL_ROOT=models-4b; MODEL_MANIFEST=qwen-models-4b.json; BASE_NAME=Qwen/Qwen3.5-4B; ADAPTER_FOLDER=qwen3_5_4b; ADAPTER_NAME=qwen3_5_4b_showharness_ft ;;
  4b-libero-v5) MODEL_ROOT=models-4b; MODEL_MANIFEST=qwen-models-4b.json; BASE_NAME=Qwen/Qwen3.5-4B; ADAPTER_NAME=qwen3_5_4b_libero_v5; ADAPTER_FOLDER=unused; : "${LIBERO_ADAPTER:?Set LIBERO_ADAPTER to the trained v5 adapter directory}" ;;
  2b-libero) MODEL_ROOT=models; MODEL_MANIFEST=qwen-models.json; BASE_NAME=Qwen/Qwen3.5-2B; ADAPTER_NAME=qwen3_5_2b_libero; ADAPTER_FOLDER=unused; : "${LIBERO_ADAPTER:?Set LIBERO_ADAPTER to the trained 2B LIBERO adapter directory}" ;;
  *) echo 'Unknown QWEN_PROFILE' >&2; exit 1 ;;
esac
[[ "$(git -C "$SHOW_HARNESS" rev-parse HEAD)" == e9867e971c3a8c5da352e151006cac859923e85e ]] || {
  echo 'Check out the Show-Harness revision recorded in setup.md.' >&2; exit 1;
}
[[ -f "$RUNTIME_DIR/$MODEL_MANIFEST" ]] || { echo 'Run download_qwen.py first.' >&2; exit 1; }
export VLLM_VENV="${QWEN_VENV:-$RUNTIME_DIR/qwen-venv}"
export MODEL="$RUNTIME_DIR/$MODEL_ROOT/base"
if [[ "${QWEN_PROFILE:-2b-sim}" == 4b-libero-v5 || "${QWEN_PROFILE:-2b-sim}" == 2b-libero ]]; then
  export LORA="$ADAPTER_NAME=$LIBERO_ADAPTER"
else
  export LORA="$ADAPTER_NAME=$RUNTIME_DIR/$MODEL_ROOT/adapter/$ADAPTER_FOLDER"
fi
export FAMILY=qwen3_5
export CHAT_TEMPLATE="$SHOW_HARNESS/models/chat_templates/qwen3_5_nothink.jinja"
export HF_HOME="${HF_HOME:-$RUNTIME_DIR/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export HOST="${HOST:-127.0.0.1}"
export GPU_UTIL="${GPU_UTIL:-0.8}"
export MAX_LEN="${MAX_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export TEMPERATURE=0
[[ -f "$MODEL/config.json" && -f "${LORA#*=}/adapter_model.safetensors" && -f "$CHAT_TEMPLATE" ]] || {
  echo 'Missing model, adapter, or chat template.' >&2; exit 1;
}
# Refuse to reuse any occupied port, including another model server.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  "$VLLM_VENV/bin/python" - "${PORT:-8000}" "$HOST" <<'PYPORT'
import socket, sys
with socket.socket() as sock:
    sock.bind((sys.argv[2], int(sys.argv[1])))
PYPORT
fi
CMD=("$VLLM_VENV/bin/vllm" serve "$MODEL"
  --served-model-name "$BASE_NAME" --dtype bfloat16
  --enable-lora --max-lora-rank 64 --lora-modules "$LORA"
  --chat-template "$CHAT_TEMPLATE"
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_NUM_SEQS" --tensor-parallel-size "${TP:-1}"
  --gdn-prefill-backend "${GDN_PREFILL_BACKEND:-triton}"
  --attention-backend "${ATTENTION_BACKEND:-FLASH_ATTN}"
  --override-generation-config '{"temperature": 0, "top_p": 1.0, "top_k": -1}'
  --host "$HOST" --port "${PORT:-8000}")
[[ "$ENFORCE_EAGER" == 1 ]] && CMD+=(--enforce-eager)
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
exec "${CMD[@]}"
