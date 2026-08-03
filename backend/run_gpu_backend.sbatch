#!/bin/bash
#SBATCH --job-name=drought_backend
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/maandrew-models/droughtapp/streamlit_app/backend/logs/gpu_backend_%j.out
#SBATCH --error=/scratch/maandrew-models/droughtapp/streamlit_app/backend/logs/gpu_backend_%j.err

set -euo pipefail

APP_DIR=/scratch/maandrew-models/droughtapp/streamlit_app/backend
PORT="${BACKEND_PORT:-18000}"

cd "$APP_DIR"

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export DROUGHTAPP_GPU_DEVICE="${DROUGHTAPP_GPU_DEVICE:-0}"

echo "drought_backend_job_id=${SLURM_JOB_ID:-unknown}"
echo "drought_backend_node=$(hostname)"
echo "drought_backend_port=${PORT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "started_at=$(date -Is)"

.venv/bin/python - <<'PY'
import tensorflow as tf
print("tensorflow_gpus", tf.config.list_physical_devices("GPU"))
PY

exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
