#!/bin/bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

if ! command -v streamlit >/dev/null 2>&1; then
  if python -c "import streamlit" >/dev/null 2>&1; then
    STREAMLIT=(python -m streamlit)
  elif command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx jupyter_env; then
    exec conda run --no-capture-output -n jupyter_env "$0"
  else
    echo "streamlit is not installed in the active environment." >&2
    echo "Activate the app environment or install requirements.txt." >&2
    exit 127
  fi
else
  STREAMLIT=(streamlit)
fi

export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# gpu selection: DROUGHTAPP_GPU_DEVICE, CUDA_VISIBLE_DEVICES, then 0
if [[ -n "${DROUGHTAPP_GPU_DEVICE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${DROUGHTAPP_GPU_DEVICE%%,*}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
else
  export CUDA_VISIBLE_DEVICES=0
fi
export TF_CPP_MIN_LOG_LEVEL=2
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_XLA_FLAGS=--tf_xla_auto_jit=2

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
"${STREAMLIT[@]}" run app.py \
  --server.port "${STREAMLIT_PORT:-8501}" \
  --server.address "${STREAMLIT_ADDRESS:-0.0.0.0}"
