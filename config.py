from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Any


GPU_ENV_VAR = "DROUGHTAPP_GPU_DEVICE"
MODEL_WARMUP_ENV_VAR = "DROUGHTAPP_MODEL_WARMUP"
HISTORY_DAYS_USE_TIMESCALE_DEFAULT = 0


def _first_csv_value(value: str | None) -> str | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts[0] if parts else None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(names: List[str], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def _default_history_days(timescale: str, default: int) -> int:
    upper = timescale.upper()
    return _env_int(
        [
            f"{upper}_DEFAULT_HISTORY_DAYS",
            f"DEFAULT_HISTORY_DAYS_{upper}",
            "DEFAULT_HISTORY_DAYS",
        ],
        default,
    )


_requested_gpu = _first_csv_value(os.environ.get(GPU_ENV_VAR))
_visible_gpu = _first_csv_value(os.environ.get("CUDA_VISIBLE_DEVICES"))

if _requested_gpu is not None:
    GPU_DEVICE_ID = _requested_gpu
    GPU_CONFIG_SOURCE = GPU_ENV_VAR
elif _visible_gpu is not None:
    GPU_DEVICE_ID = _visible_gpu
    GPU_CONFIG_SOURCE = "CUDA_VISIBLE_DEVICES"
else:
    GPU_DEVICE_ID = "0"
    GPU_CONFIG_SOURCE = "default"

# sets GPU visibility before TensorFlow import
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE_ID
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


@dataclass(frozen=True)
class TimescaleSpec:
    name: str
    parent_dirs: List[str]
    best_arch_folder: str
    horizon_days: int
    default_history_days: int


APP_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(APP_DIR, "..", ".."))
BASE_DIR = os.environ.get("NLDAS_BASE_DIR", REPO_ROOT)
STREAMLIT_MODEL_ARTIFACTS_DIR = os.environ.get(
    "STREAMLIT_MODEL_ARTIFACTS_DIR",
    os.path.join(APP_DIR, "model_artifacts"),
)
ACTIVE_MODEL_VERSION = os.environ.get("SEED_MODEL_VERSION", "selected_2019_v20260731")
ACTIVE_MODEL_ROOT = os.path.join(STREAMLIT_MODEL_ARTIFACTS_DIR, ACTIVE_MODEL_VERSION)
MODEL_GRID_PATH = os.environ.get(
    "MODEL_GRID_PATH",
    os.path.join(STREAMLIT_MODEL_ARTIFACTS_DIR, "grid", "model_grid.npz"),
)
FORCING_DIR = os.environ.get("NLDAS_FORCING_DIR", os.path.join(BASE_DIR, "yearly"))

ELM_ET_PATH = os.environ.get("ELM_ET_PATH", os.path.join(BASE_DIR, "ELM_EVAPOTRANSPIRATION_2000_2019.nc"))
ELM_SM_PATH = os.environ.get("ELM_SM_PATH", os.path.join(BASE_DIR, "ELM_SM_2000-2019.nc"))

_DEFAULT_NLDAS_CACHE_DIR = os.path.join(BASE_DIR, "droughtapp", "droughtapp_cache")
NLDAS_CACHE_DIR = os.environ.get("NLDAS_CACHE_DIR", _DEFAULT_NLDAS_CACHE_DIR)
try:
    os.makedirs(NLDAS_CACHE_DIR, exist_ok=True)
except OSError:
    if "NLDAS_CACHE_DIR" in os.environ:
        raise
    NLDAS_CACHE_DIR = os.path.join(APP_DIR, "droughtapp_cache")
    os.makedirs(NLDAS_CACHE_DIR, exist_ok=True)
PROCESSED_DAILY_CACHE_DIR = os.environ.get(
    "PROCESSED_DAILY_CACHE_DIR",
    os.path.join(NLDAS_CACHE_DIR, "processed_daily"),
)
os.makedirs(PROCESSED_DAILY_CACHE_DIR, exist_ok=True)

# fallback for callers without a timescale
DEFAULT_HISTORY_DAYS = HISTORY_DAYS_USE_TIMESCALE_DEFAULT
DEFAULT_COPULA_TAU = float(os.environ.get("DEFAULT_COPULA_TAU", "0.40"))
MODEL_WARMUP_ENABLED = _env_bool(MODEL_WARMUP_ENV_VAR, True)


# models selected on 2019 validation
TIMESCALES: Dict[str, TimescaleSpec] = {
    "Weekly": TimescaleSpec(
        name="Weekly",
        parent_dirs=[os.path.join(ACTIVE_MODEL_ROOT, "Weekly")],
        best_arch_folder="target_specific",
        horizon_days=7,
        default_history_days=_default_history_days("Weekly", 10),
    ),
    "Monthly": TimescaleSpec(
        name="Monthly",
        parent_dirs=[os.path.join(ACTIVE_MODEL_ROOT, "Monthly")],
        best_arch_folder="target_specific",
        horizon_days=30,
        default_history_days=_default_history_days("Monthly", 45),
    ),
    "Seasonal": TimescaleSpec(
        name="Seasonal",
        parent_dirs=[os.path.join(ACTIVE_MODEL_ROOT, "Seasonal")],
        best_arch_folder="target_specific",
        horizon_days=90,
        default_history_days=_default_history_days("Seasonal", 135),
    ),
}

# legacy artifact aliases
ET_FILE_DEFAULT = ELM_ET_PATH
SM_FILE_DEFAULT = ELM_SM_PATH
APP_CACHE_DIR = NLDAS_CACHE_DIR

TIMESCALE_TO_HORIZON_DAYS = {k: v.horizon_days for k, v in TIMESCALES.items()}
TIMESCALE_TO_DEFAULT_HISTORY_DAYS = {k: v.default_history_days for k, v in TIMESCALES.items()}

DEFAULT_BUNDLE_KEY = "default"
BUNDLES: Dict[str, Dict[str, Any]] = {
    DEFAULT_BUNDLE_KEY: {
        "forcing_dir": FORCING_DIR,
        "et_path": ELM_ET_PATH,
        "sm_path": ELM_SM_PATH,
        "model_grid_path": MODEL_GRID_PATH,
        "cache_dir": NLDAS_CACHE_DIR,
        "processed_cache_dir": PROCESSED_DAILY_CACHE_DIR,
    }
}
