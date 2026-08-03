#!/bin/bash
#SBATCH --job-name=seed_app_validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --chdir=/scratch/maandrew-models/droughtapp/streamlit_app/backend
#SBATCH --output=/scratch/maandrew-models/droughtapp/streamlit_app/validation/logs/model_validate_%A_%a.out
#SBATCH --error=/scratch/maandrew-models/droughtapp/streamlit_app/validation/logs/model_validate_%A_%a.err

set -euo pipefail
module purge || true
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-jupyter_env}"
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/seed-app-mpl-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR"

OUTPUT_DIR="/scratch/maandrew-models/droughtapp/streamlit_app/validation/selected_models_${SLURM_ARRAY_JOB_ID}"
srun python -u validate_selected_models_gpu.py \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "$OUTPUT_DIR"
