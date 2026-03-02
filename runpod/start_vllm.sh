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
UI_PORT="${UI_PORT:-7860}"

STEP_AUDIO2_REPO_DIR="${STEP_AUDIO2_REPO_DIR:-/workspace/Step-Audio2}"
TOKEN2WAV_MODEL_ID="${TOKEN2WAV_MODEL_ID:-stepfun-ai/Step-Audio-2-mini}"
TOKEN2WAV_MODEL_DIR="${TOKEN2WAV_MODEL_DIR:-/workspace/Step-Audio-2-mini}"
TOKEN2WAV_DIR="${TOKEN2WAV_DIR:-${TOKEN2WAV_MODEL_DIR}/token2wav}"
TOKEN2WAV_PROMPT_WAV="${TOKEN2WAV_PROMPT_WAV:-${STEP_AUDIO2_REPO_DIR}/assets/default_female.wav}"
TOKEN2WAV_FLOAT16="${TOKEN2WAV_FLOAT16:-0}"

ensure_repo() {
  local repo_url="$1"
  local repo_dir="$2"
  local clone_mode="${3:-full}"

  if [[ -d "${repo_dir}/.git" ]]; then
    if ! git -C "${repo_dir}" pull --ff-only; then
      echo "Warning: git pull failed for ${repo_dir}; using existing checkout."
    fi
    return
  fi

  if [[ -d "${repo_dir}" ]] && [[ -n "$(ls -A "${repo_dir}" 2>/dev/null)" ]]; then
    echo "Using existing non-git directory at ${repo_dir}."
    return
  fi

  if [[ "${clone_mode}" == "depth1" ]]; then
    git clone --depth 1 "${repo_url}" "${repo_dir}"
  else
    git clone "${repo_url}" "${repo_dir}"
  fi
}

ensure_repo "https://github.com/Legalphoenix/stepfun.git" "${REPO_DIR}"
ensure_repo "https://github.com/stepfun-ai/Step-Audio2.git" "${STEP_AUDIO2_REPO_DIR}" "depth1"

python3 -m pip install --no-cache-dir \
  "huggingface_hub<1.0" \
  "gradio<6" \
  requests \
  pydub \
  onnxruntime \
  s3tokenizer \
  hyperpyyaml \
  soundfile

python3 - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("torchaudio") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "torchaudio"])
PY

export MODEL_ID MODEL_DIR
python3 -u "${REPO_DIR}/runpod/download_model.py"

export TOKEN2WAV_MODEL_ID TOKEN2WAV_MODEL_DIR
python3 -u "${REPO_DIR}/runpod/download_token2wav.py"

VLLM_HELP="$(vllm serve --help 2>/dev/null || true)"
VLLM_ARGS=(
  "${MODEL_DIR}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${API_HOST}"
  --port "${API_PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --chat-template "${REPO_DIR}/runpod/chat_template.jinja"
  --enable-log-requests
  --trust-remote-code
)
if [[ "${VLLM_HELP}" == *"--interleave-mm-strings"* ]]; then
  VLLM_ARGS+=(--interleave-mm-strings)
else
  echo "vLLM does not support --interleave-mm-strings in this image; continuing without it."
fi

vllm serve "${VLLM_ARGS[@]}" \
  &

VLLM_PID=$!

cleanup() {
  kill "${VLLM_PID}" >/dev/null 2>&1 || true
  kill "${UI_PID:-0}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

export OPENAI_API_URL="http://127.0.0.1:${API_PORT}/v1/chat/completions"
export UI_PORT
export SERVED_MODEL_NAME
export STEP_AUDIO2_REPO_DIR
export TOKEN2WAV_DIR
export TOKEN2WAV_PROMPT_WAV
export TOKEN2WAV_FLOAT16

python3 -u "${REPO_DIR}/runpod/gradio_app.py" &
UI_PID=$!

wait -n "${VLLM_PID}" "${UI_PID}"
EXIT_CODE=$?
cleanup
exit "${EXIT_CODE}"
