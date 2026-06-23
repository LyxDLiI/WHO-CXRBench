#!/bin/bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-0528-Qwen3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

vllm serve "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --trust-remote-code
