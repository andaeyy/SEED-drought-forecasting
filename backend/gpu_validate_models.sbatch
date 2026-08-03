#!/bin/bash
#SBATCH --job-name=droughtapp_gpu_validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o gpu_validate_%j.out
#SBATCH -e gpu_validate_%j.err

set -euo pipefail

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-jupyter_env}"
set -u

if [[ -d "${SLURM_SUBMIT_DIR:-}/backend/app" ]]; then
  BACKEND_DIR="$(cd "${SLURM_SUBMIT_DIR}/backend" && pwd)"
elif [[ -d "${SLURM_SUBMIT_DIR:-}/app" ]]; then
  BACKEND_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$BACKEND_DIR"

if [[ -n "${DROUGHTAPP_GPU_DEVICE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${DROUGHTAPP_GPU_DEVICE%%,*}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
else
  export CUDA_VISIBLE_DEVICES=0
fi

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}}"
mkdir -p "$MPLCONFIGDIR"

python -u <<'PY'
import numpy as np
import tensorflow as tf

from app.adapt.config import BASE_DIR, ELM_ET_PATH, ELM_SM_PATH, FORCING_DIR, MODEL_GRID_PATH, TIMESCALES
from app.adapt.data_local import get_model_grid_from_local_forcing
from app.adapt.models import load_models, resolve_timescale_model_dir

print("CUDA_VISIBLE_DEVICES=", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
gpus = tf.config.list_physical_devices("GPU")
print("tensorflow_gpus=", gpus)
if not gpus:
    raise SystemExit("TensorFlow cannot see a GPU in this batch allocation.")

for name, spec in TIMESCALES.items():
    model_dir = resolve_timescale_model_dir(BASE_DIR, spec.parent_dirs, spec.best_arch_folder)
    bundle = load_models(model_dir)
    print(
        name,
        "model_dir=", model_dir,
        "sm_input=", bundle.sm_model.input_shape,
        "et_input=", bundle.et_model.input_shape,
        "sm_norm_keys=", sorted(bundle.sm_norms),
        "et_norm_keys=", sorted(bundle.et_norms),
    )

lat2d, lon2d = get_model_grid_from_local_forcing(
    FORCING_DIR,
    ELM_ET_PATH,
    ELM_SM_PATH,
    model_grid_path=MODEL_GRID_PATH,
)
lon_180 = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
print(
    "model_grid=",
    lat2d.shape,
    lon2d.shape,
    "lat=", float(np.nanmin(lat2d)), float(np.nanmax(lat2d)),
    "lon=", float(np.nanmin(lon_180)), float(np.nanmax(lon_180)),
)
print("GPU/model/grid validation complete.")
PY
