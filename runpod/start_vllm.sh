#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/stepfun}"
MODEL_ID="${MODEL_ID:-stepfun-ai/Step-Audio-R1.1}"
MODEL_DIR="${MODEL_DIR:-/workspace/Step-Audio-R1.1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Step-Audio-R1.1}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-9999}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/Legalphoenix/stepfun.git "${REPO_DIR}"
else
  git -C "${REPO_DIR}" pull --ff-only
fi

python -m pip install --no-cache-dir -U huggingface_hub

export MODEL_ID MODEL_DIR
python -u "${REPO_DIR}/runpod/download_model.py"

exec vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --chat-template "${REPO_DIR}/runpod/chat_template.jinja" \
  --enable-log-requests \
  --interleave-mm-strings \
  --trust-remote-code
