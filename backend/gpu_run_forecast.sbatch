#!/bin/bash
#SBATCH --job-name=droughtapp_forecast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -o gpu_forecast_%j.out
#SBATCH -e gpu_forecast_%j.err

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
APP_ROOT="$(cd "${BACKEND_DIR}/.." && pwd)"
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
export BATCH_OUTPUT_DIR="${BATCH_OUTPUT_DIR:-${APP_ROOT}/batch_outputs/${SLURM_JOB_ID:-local}}"
mkdir -p "$MPLCONFIGDIR" "$BATCH_OUTPUT_DIR"

# override with --export=ALL,TIMESCALE=Weekly,AS_OF_DAY=2026-06-26
export TIMESCALE="${TIMESCALE:-Weekly}"
export AS_OF_DAY="${AS_OF_DAY:-}"

python -u <<'PY'
from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

from app.core.forecast_jobs import get_forecast_job, submit_forecast_job, to_response
from app.core.map_features import forecast_result_to_geojson
from app.core.schemas import ForecastJobCreate

timescale = os.environ["TIMESCALE"]
as_of_day = os.environ["AS_OF_DAY"] or (date.today() - timedelta(days=3)).isoformat()
out_dir = Path(os.environ["BATCH_OUTPUT_DIR"])
poll_seconds = int(os.environ.get("POLL_SECONDS", "30"))

request = ForecastJobCreate(timescale=timescale, as_of_day=as_of_day)
record = submit_forecast_job(request)
print("submitted", record.job_id, record.status, "timescale=", timescale, "as_of_day=", as_of_day)

while not record.future.done():
    latest = get_forecast_job(record.job_id)
    print("waiting", latest.job_id, latest.status, "updated=", latest.updated_at.isoformat(), flush=True)
    time.sleep(poll_seconds)

record.future.result()
latest = get_forecast_job(record.job_id)
response = to_response(latest)
summary_path = out_dir / "forecast_job_response.json"
summary_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
print("status", response.status, "summary=", summary_path)

if response.status != "complete":
    raise SystemExit(f"Forecast did not complete: {response.error}")

geojson_path = out_dir / "drought-risk.geojson"
geojson_path.write_text(
    json.dumps(forecast_result_to_geojson(latest.result), separators=(",", ":")),
    encoding="utf-8",
)
print("geojson=", geojson_path)
print("forecast complete")
PY
