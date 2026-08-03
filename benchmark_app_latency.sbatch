#!/bin/bash
#SBATCH --job-name=droughtapp_latency
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o latency_droughtapp_%j.out
#SBATCH -e latency_droughtapp_%j.err

set -euo pipefail

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate jupyter_env
set -u

APP_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/../.." && pwd)"
SCRIPT="${REPO_ROOT}/Emulator models/inference_latency.py"
ARTIFACTS="${APP_DIR}/model_artifacts"

run_one() {
  local label="$1"
  local model_dir="$2"
  local target="$3"
  local horizon="$4"
  local out_dir="${model_dir}/latency"

  mkdir -p "${out_dir}"
  echo "=== ${label}: ${target} ==="
  python -u "${SCRIPT}" \
    --model_path "${model_dir}/keras_convlstm_${target}_best.keras" \
    --batch_size 2 \
    --warmup_steps 20 \
    --timed_steps 300 \
    --horizon_days "${horizon}" \
    --ensemble_size 10 \
    --output_json "${out_dir}/latency_${target}.json"
}

run_one "Weekly Seq2seqconvlstm" "${ARTIFACTS}/Weekly/Seq2seqconvlstm" "sm" "7"
run_one "Weekly Seq2seqconvlstm" "${ARTIFACTS}/Weekly/Seq2seqconvlstm" "et" "7"
run_one "Monthly DEconvlstm" "${ARTIFACTS}/Monthly/DEconvlstm" "sm" "30"
run_one "Monthly DEconvlstm" "${ARTIFACTS}/Monthly/DEconvlstm" "et" "30"
run_one "Seasonal DEconvlstm" "${ARTIFACTS}/Seasonal/DEconvlstm" "sm" "90"
run_one "Seasonal DEconvlstm" "${ARTIFACTS}/Seasonal/DEconvlstm" "et" "90"

echo "Done."
