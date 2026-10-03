# SEED Drought Outlook App

This project serves Great Plains drought outlooks through a FastAPI backend and a Next.js frontend. The backend loads six independently selected TensorFlow checkpoints, retrieves NASA NLDAS forcings with `earthaccess`, predicts ET and soil moisture, and returns ET, SM, and ET/SM-derived drought-category endpoint maps to the browser.

The legacy Streamlit code remains for compatibility. New development should treat `backend/` as the API service and `frontend/` as the browser client.

## Active Model Deployment

The default deployment is `selected_2019_v20260731`. It contains one target-specific ET model and one target-specific SM model for each horizon:

| Horizon | ET model | SM model |
|---|---|---|
| 10 input days -> lead day 7 | Autoregressive ConvLSTM, run 001 | Encoder-decoder ConvLSTM, run 007 |
| 45 input days -> lead day 30 | Sequence-to-map ConvLSTM, run 029 | Encoder-decoder ConvLSTM, run 024 |
| 135 input days -> lead day 90 | Autoregressive ConvLSTM, run 034 | Encoder-decoder ConvLSTM, run 037 |

All checkpoints were selected using 2019 validation only and evaluated on the independent 2020 period. The models consume exactly `PRECTmms`, `TBOT`, `WIND`, `QBOT`, `PSRF`, `FSDS`, and `FLDS` in their verified training-time channel order. Outputs are single endpoint maps at lead day 7, 30, or 90, not full forecast sequences.

Install or re-verify the locked bundles without overwriting conflicting artifacts:

```bash
python scripts/install_selected_models.py --dry-run
python scripts/install_selected_models.py
```

## Evaluation and Systems Contribution

SEED combines target-specific spatiotemporal model selection with a versioned GPU inference service and an interactive geospatial client.

- **Chronological evaluation:** candidate checkpoints are selected using only 2019 validation; the selected choices are frozen before independent 2020 evaluation. The test period is not used to select architectures. ET and SM are selected separately at each of the 7-, 30-, and 90-day leads, producing the six ConvLSTM selections in [Active Model Deployment](#active-model-deployment).
- **Reproducibility contracts:** versioned bundles bind each target and horizon to a model ID, checkpoint, normalizer, ordered seven-channel input, fixed 10/45/135-day input window, and single lead-day endpoint output. The [installer](scripts/install_selected_models.py) checks the locked selection and source hashes and refuses to overwrite conflicting artifacts; the loader verifies checkpoint and normalizer SHA-256 hashes before inference.
- **Backend/frontend design:** FastAPI owns NASA NLDAS retrieval, preprocessing, TensorFlow GPU inference, and ET/SM-derived drought maps. Next.js provides forecast controls, map-layer inspection, and GeoJSON export. ET, SM, and drought layers share the same selected model pair for a forecast; Slurm provisions the GPU backend separately from the browser client.
- **Deterministic artifact validation:** [contract tests](backend/tests/test_selected_model_contract.py) check all six model identities, manifest schemas, artifact hashes, channel order, input windows, lead days, endpoint semantics, and API metadata. A separate [GPU parity gate](backend/validate_selected_models_gpu.py) compares deployed predictions with locked 2020 archives on three fixed dates, checks missing-value normalization and temporal alignment, and enforces a maximum absolute prediction difference of `5e-5` in physical units. This is an implementation-parity tolerance, not a forecast-accuracy result or a claim of bitwise deterministic GPU execution.
- **Latency methodology:** [benchmark_app_latency.sbatch](benchmark_app_latency.sbatch) requests one GPU and invokes model timing with 20 warm-up steps, 300 timed steps, batch size 2, ensemble size 10, and separate ET/SM JSON outputs for each horizon. It currently targets legacy Weekly/Monthly/Seasonal checkpoints and requires an external `Emulator models/inference_latency.py` script; it does not establish latency for `selected_2019_v20260731` or end-to-end API/browser latency.

**Verified legacy timing:** the committed Git LFS reports record the following `mean_ms` values under TensorFlow 2.18.1 and the protocol above:

| Lead | ET mean (ms) | SM mean (ms) | Reports |
|---|---:|---:|---|
| 7 days | 165.21 | 175.45 | [ET](model_artifacts/Weekly/Seq2seqconvlstm/latency/latency_et.json), [SM](model_artifacts/Weekly/Seq2seqconvlstm/latency/latency_sm.json) |
| 30 days | 719.17 | 758.41 | [ET](model_artifacts/Monthly/DEconvlstm/latency/latency_et.json), [SM](model_artifacts/Monthly/DEconvlstm/latency/latency_sm.json) |
| 90 days | 2029.65 | 2192.97 | [ET](model_artifacts/Seasonal/DEconvlstm/latency/latency_et.json), [SM](model_artifacts/Seasonal/DEconvlstm/latency/latency_sm.json) |

These are recorded model-timing values for batch size 2, not per-request browser latency. The reports identify one visible GPU but not its hardware model; legacy ET inputs have eight channels, unlike the current seven-channel deployment. The external timing implementation is needed to interpret ensemble handling and GPU synchronization. No current-deployment latency, RMSE, MAE, baseline comparison, or speedup is claimed here without verified evaluation outputs.

## Requirements

- Python 3.10 or newer
- Git LFS 3 or newer
- A CUDA-capable GPU with a TensorFlow-compatible CUDA/cuDNN runtime
- Node.js 20 or newer
- A free NASA Earthdata account: <https://urs.earthdata.nasa.gov/users/new>
- Local access to required model artifacts and any raw NetCDF fallback data

After cloning, materialize the versioned model bundles:

```bash
git lfs install
git lfs pull
```

The default `selected_2019_v20260731` deployment and the legacy compatibility
checkpoints are stored in Git LFS. The loader verifies every active checkpoint
and normalizer against its manifest SHA-256 before inference.

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Authenticate with NASA Earthdata once on the machine that will run inference:

```bash
python -c "import earthaccess; earthaccess.login(persist=True)"
```

Start the FastAPI development server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Before running forecasts, confirm TensorFlow sees the assigned GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The list must contain at least one GPU. The inference pipeline intentionally preserves the Python TensorFlow/Earthaccess flow and should not fall back to CPU for production forecasts.

### Slurm GPU service

Launch the API as a one-A40 service and read the assigned compute node:

```bash
sbatch backend/run_gpu_backend.sbatch
squeue -j <job-id> -o '%.18i %.20j %.9T %.10M %.9l %.20R'
```

The batch service listens on port `18000` of its assigned GPU node. The standard dashboard build expects the API on browser-local port `8000`. From the local workstation, open separate tunnels for the login-node frontend and compute-node backend:

```bash
ssh -N -L 3000:127.0.0.1:3000 maandrew@login1.coeus.rc.pdx.edu
ssh -N -J maandrew@login1.coeus.rc.pdx.edu -L 8000:127.0.0.1:18000 maandrew@<gpu-node>
```

Then open `http://localhost:3000`. The default retrieval windows are exactly 10, 45, and 135 historical days for weekly, monthly, and seasonal inference.

## Frontend Setup

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

The frontend defaults to the local API at `http://localhost:8000`. Open the landing page at `http://localhost:3000` and the forecast workspace at `http://localhost:3000/dashboard`.

One forecast run computes all three map layers from the same frozen ET/SM model pair. The dashboard layer tabs and GeoJSON download distinguish `ET`, `SM`, and the downstream `Drought` category explicitly.

## Environment Variables

Backend variables are documented in `backend/.env.example`:

- `DROUGHTAPP_GPU_DEVICE`: explicit GPU ID to use, for example `0`.
- `SEED_MODEL_VERSION`: versioned deployment directory, default `selected_2019_v20260731`.
- `CUDA_VISIBLE_DEVICES`: fallback GPU visibility setting when `DROUGHTAPP_GPU_DEVICE` is unset.
- `TF_CPP_MIN_LOG_LEVEL`: TensorFlow log verbosity, usually `2`.
- `TF_GPU_ALLOCATOR`: TensorFlow GPU allocator, usually `cuda_malloc_async`.
- `STREAMLIT_MODEL_ARTIFACTS_DIR`: path to the model artifact directory used by the preserved inference code.
- `MODEL_GRID_PATH`: path to the precomputed model grid `.npz`.
- `NLDAS_BASE_DIR`: base directory for local raw NLDAS/ELM data.
- `NLDAS_FORCING_DIR`: local yearly NLDAS forcing directory used for fallback grid reconstruction.
- `ELM_ET_PATH`: local ELM evapotranspiration NetCDF path.
- `ELM_SM_PATH`: local ELM soil-moisture NetCDF path.
- `NLDAS_CACHE_DIR`: local cache for NLDAS files downloaded through Earthaccess.
- `NLDAS_SHORTNAME`: NASA Earthdata collection short name, default `NLDAS_FORA0125_H`.
- `DEFAULT_HISTORY_DAYS` and horizon-specific variants: retrieval history used by the inference pipeline. Model tensors remain fixed at 10, 45, or 135 input days.
- `DEFAULT_COPULA_TAU`: drought-index threshold parameter, default `0.40`.
- `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`: optional for non-interactive deployments only. Prefer persisted `earthaccess.login(...)` credentials.

Frontend variables are documented in `frontend/.env.local.example`:

- `NEXT_PUBLIC_API_BASE_URL`: FastAPI base URL used by the Next.js client.

## Earthdata Login

`earthaccess` needs each developer or deployment user to authenticate with their own NASA Earthdata account. The recommended local setup is:

```bash
python -c "import earthaccess; earthaccess.login(persist=True)"
```

This stores credentials outside the repository for reuse by later downloads. Do not commit `.env` files, `.netrc`, Earthdata cookies, or any credential material.

## Artifact And Data Policy

Model artifacts under `model_artifacts/` are versioned with Git LFS. Do not
commit model checkpoints anywhere else. Do not commit NetCDF data, cache files,
local `.env` files, or credentials.

The ignore rules exclude common local outputs including:

- Keras and NumPy artifacts outside the versioned `model_artifacts/` tree
- NetCDF and geospatial data such as `*.nc`, `*.nc4`, `*.grib`, and `*.tif`
- `yearly/`, `droughtapp_cache/`, and `NLDAS_Cache/`
- backend virtualenv/cache directories and frontend dependency/build directories

Use Git LFS for additional deployment checkpoints. Keep raw training and
evaluation datasets in approved external storage rather than Git.

## Legacy Streamlit Run

The original Streamlit entrypoint can still be used during transition from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import earthaccess; earthaccess.login(persist=True)"
streamlit run app.py
```

Use `DROUGHTAPP_GPU_DEVICE` or `CUDA_VISIBLE_DEVICES` to select the GPU before launching Streamlit.

## Verification

Run the deterministic bundle and metadata tests:

```bash
cd backend
PYTHONPATH=. python -m unittest discover -s tests -v
```

Run responsive browser tests after starting the API and frontend:

```bash
cd frontend
PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright-browsers" npm run test:e2e
```

The A40 checkpoint/archive parity gate is submitted with:

```bash
cd backend
sbatch validate_selected_models_gpu.sbatch
```
