from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from functools import lru_cache
from threading import Lock
from time import perf_counter
from typing import Any, Dict, Tuple

import numpy as np

from .config import (
    DEFAULT_COPULA_TAU,
    GPU_CONFIG_SOURCE,
    GPU_DEVICE_ID,
    GPU_ENV_VAR,
    HISTORY_DAYS_USE_TIMESCALE_DEFAULT,
    MODEL_WARMUP_ENABLED,
    MODEL_WARMUP_ENV_VAR,
    TIMESCALES,
)

import tensorflow as tf
from .models import load_models
from .data_local import get_model_grid_from_local_forcing
from .data_earthaccess import (
    load_or_build_daily_forcing_on_target_grid,
)
from .drought_index import drought_from_zscores


def _require_tf_gpu() -> str:
    tf.config.set_soft_device_placement(False)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return "/GPU:0"
    raise RuntimeError(
        "TensorFlow cannot see a GPU. This app only runs on GPU. "
        "Start the app inside a GPU session, then set "
        f"{GPU_ENV_VAR} to the assigned GPU ID or rely on the session's "
        "CUDA_VISIBLE_DEVICES value. "
        f"Current selection: {GPU_DEVICE_ID!r} from {GPU_CONFIG_SOURCE}."
    )


def _enable_tf_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def _as_np_date(d: Any) -> np.datetime64:
    if isinstance(d, np.datetime64):
        return d.astype("datetime64[D]")
    if isinstance(d, Date):
        return np.datetime64(d.isoformat())
    if isinstance(d, str):
        return np.datetime64(d)
    raise TypeError(f"Unsupported date type: {type(d)}")


def _broadcast_mu_sd(mu: np.ndarray, sd: np.ndarray, C: int) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu).astype(np.float32)
    sd = np.asarray(sd).astype(np.float32)
    if mu.ndim == 1:
        mu = mu.reshape((1, 1, 1, -1))
    if sd.ndim == 1:
        sd = sd.reshape((1, 1, 1, -1))
    if mu.shape[-1] != C or sd.shape[-1] != C:
        raise ValueError(f"Norm channel mismatch. Expected C={C} but got mu={mu.shape}, sd={sd.shape}")
    return mu, sd


def _model_in_channels(model: tf.keras.Model) -> int:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[-1])


def _model_seq_len(model: tf.keras.Model) -> int:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[1])


_MODEL_WARMUP_LOCK = Lock()
_MODEL_WARMUP_STATUS: Dict[str, Dict[str, Any]] = {}


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000.0, 3)


def _call_lru_cached_with_timing(cached_func, *args):
    before = cached_func.cache_info()
    start = perf_counter()
    value = cached_func(*args)
    after = cached_func.cache_info()
    return value, {
        "elapsed_ms": _elapsed_ms(start),
        "cache_hit": bool(after.hits > before.hits),
    }


def _timescale_default_history_days(timescale: str, bundle: Dict[str, Any]) -> int:
    spec = TIMESCALES.get(timescale)
    if spec is not None:
        return int(spec.default_history_days)
    if bundle.get("default_history_days") is not None:
        return int(bundle["default_history_days"])
    return int(HISTORY_DAYS_USE_TIMESCALE_DEFAULT)


def _resolve_history_days(
    *,
    timescale: str,
    bundle: Dict[str, Any],
    history_days: int | None,
    min_needed: int,
) -> tuple[int, Dict[str, Any]]:
    default_history_days = _timescale_default_history_days(timescale, bundle)
    history_days_input = None if history_days is None else int(history_days)
    requested_history_days = (
        history_days_input
        if history_days_input is not None and history_days_input > HISTORY_DAYS_USE_TIMESCALE_DEFAULT
        else None
    )
    base_history_days = default_history_days if requested_history_days is None else requested_history_days
    used_history_days = max(int(base_history_days), int(min_needed))
    return used_history_days, {
        "history_days_input": history_days_input,
        "history_days_requested": requested_history_days,
        "history_days_default": int(default_history_days),
        "history_days_min_needed": int(min_needed),
        "history_days_used": int(used_history_days),
        "history_days_source": "timescale_default" if requested_history_days is None else "request",
        "history_days_min_protection_applied": bool(used_history_days > base_history_days),
    }


def _zeros_for_input_shape(shape: Any):
    dims = [1 if dim is None else int(dim) for dim in tuple(shape)]
    if not dims:
        raise ValueError(f"Unexpected empty model input shape: {shape}")
    dims[0] = 1
    return tf.zeros(tuple(dims), dtype=tf.float32)


def _dummy_model_input(model: tf.keras.Model):
    shape = model.input_shape
    first_shape = shape[0] if isinstance(shape, list) and shape else None
    if isinstance(shape, list) and (isinstance(first_shape, (list, tuple)) or hasattr(first_shape, "as_list")):
        return [_zeros_for_input_shape(item) for item in shape]
    return _zeros_for_input_shape(shape)


def _materialize_tf_output(value: Any) -> None:
    for tensor in tf.nest.flatten(value):
        if hasattr(tensor, "numpy"):
            tensor.numpy()
        else:
            np.asarray(tensor)


def _warmup_models_once(model_dir: str, loaded: Any) -> Dict[str, Any]:
    if not MODEL_WARMUP_ENABLED:
        return {
            "enabled": False,
            "env_var": MODEL_WARMUP_ENV_VAR,
            "attempted": False,
            "cache_hit": False,
            "status": "disabled",
            "elapsed_ms": 0.0,
        }

    with _MODEL_WARMUP_LOCK:
        cached = _MODEL_WARMUP_STATUS.get(model_dir)
        if cached is not None:
            result = dict(cached)
            result["attempted"] = False
            result["cache_hit"] = True
            return result

        start = perf_counter()
        status: Dict[str, Any] = {
            "enabled": True,
            "env_var": MODEL_WARMUP_ENV_VAR,
            "attempted": False,
            "cache_hit": False,
            "status": "not_started",
            "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
        }

        if not status["gpu_devices"]:
            status.update(
                {
                    "status": "skipped_no_gpu",
                    "elapsed_ms": _elapsed_ms(start),
                }
            )
            _MODEL_WARMUP_STATUS[model_dir] = dict(status)
            return status

        model_timings: Dict[str, float] = {}
        try:
            status["attempted"] = True
            with tf.device("/GPU:0"):
                sm_start = perf_counter()
                _materialize_tf_output(loaded.sm_model(_dummy_model_input(loaded.sm_model), training=False))
                model_timings["sm_model"] = _elapsed_ms(sm_start)

                et_start = perf_counter()
                _materialize_tf_output(loaded.et_model(_dummy_model_input(loaded.et_model), training=False))
                model_timings["et_model"] = _elapsed_ms(et_start)
            status.update(
                {
                    "status": "ok",
                    "elapsed_ms": _elapsed_ms(start),
                    "timings_ms": model_timings,
                }
            )
        except Exception as exc:
            status.update(
                {
                    "status": "failed",
                    "elapsed_ms": _elapsed_ms(start),
                    "timings_ms": model_timings,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        _MODEL_WARMUP_STATUS[model_dir] = dict(status)
        return status


@lru_cache(maxsize=8)
def _get_loaded_models(model_dir: str):
    return load_models(model_dir)


@lru_cache(maxsize=8)
def _get_model_grid_cached(forcing_dir: str, et_path: str, sm_path: str, model_grid_path: str | None):
    return get_model_grid_from_local_forcing(forcing_dir, et_path, sm_path, model_grid_path=model_grid_path)


def _load_earthaccess_daily_forcing(
    *,
    start_day: np.datetime64,
    as_of: np.datetime64,
    cache_dir: str | None,
    processed_cache_dir: str | None,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    X_phys, day_keys, used = load_or_build_daily_forcing_on_target_grid(
        start_day=start_day,
        end_day=as_of,
        cache_dir=cache_dir,
        lat2d=lat2d,
        lon2d=lon2d,
        processed_cache_dir=processed_cache_dir,
    )
    used = dict(used)
    used["timings_ms"] = {
        "earthaccess_search_download": sum(
            float(item.get("earthaccess_search_time_seconds", 0.0))
            + float(item.get("earthaccess_download_time_seconds", 0.0))
            for item in used.get("raw_hourly_fetches", [])
        )
        * 1000.0,
        "xarray_open": float(used.get("xarray_open_time_seconds", 0.0)) * 1000.0,
        "daily_resample_regrid": float(used.get("daily_resample_regrid_time_seconds", 0.0)) * 1000.0,
    }
    return X_phys, day_keys, used


def _predict_single(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    with tf.device(_require_tf_gpu()):
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        y = model(x_tf, training=False)
    y = np.asarray(y)

    if y.ndim == 5:
        y = y[:, -1]
    if y.ndim == 4:
        y = y[0, :, :, 0]
    elif y.ndim == 3:
        y = y[0]
    else:
        raise ValueError(f"Unexpected model output shape: {y.shape}")
    return y.astype(np.float32)


def _build_sm_pred_feature_window(
    *,
    sm_model: tf.keras.Model,
    X_base_norm: np.ndarray,
    sm_y_mu: np.ndarray,
    sm_y_sd: np.ndarray,
    et_in_mu_feat: float,
    seq_len_sm: int,
    horizon: int,
    et_start_idx: int,
    et_end_idx: int,
) -> np.ndarray:
    t_indices = np.arange(et_start_idx, et_end_idx + 1, dtype=np.int64)
    end_indices = t_indices - horizon

    valid = end_indices >= (seq_len_sm - 1)

    H, W = X_base_norm.shape[1], X_base_norm.shape[2]
    sm_feat = np.full((t_indices.size, H, W), float(et_in_mu_feat), dtype=np.float32)

    if not np.any(valid):
        return sm_feat

    starts = end_indices[valid] - (seq_len_sm - 1)
    ends = end_indices[valid]

    xb_list = []
    for s, e in zip(starts, ends):
        xb_list.append(X_base_norm[s : e + 1])
    xb = np.stack(xb_list, axis=0).astype(np.float32)

    with tf.device(_require_tf_gpu()):
        x_tf = tf.convert_to_tensor(xb, dtype=tf.float32)
        y = sm_model(x_tf, training=False)
    y = np.asarray(y)

    if y.ndim == 5:
        y = y[:, -1]
    if y.ndim == 4:
        y = y[..., 0]
    elif y.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected SM model output shape: {y.shape}")

    sm_y_mu = np.asarray(sm_y_mu).astype(np.float32).reshape(-1)[0]
    sm_y_sd = np.asarray(sm_y_sd).astype(np.float32).reshape(-1)[0]
    y_phys = y.astype(np.float32) * sm_y_sd + sm_y_mu

    sm_feat[valid] = y_phys
    return sm_feat


@dataclass
class ForecastResult:
    target_day: np.datetime64
    horizon_days: int
    lat2d: np.ndarray
    lon2d: np.ndarray
    sm_pred: np.ndarray
    et_pred: np.ndarray
    z_sm: np.ndarray
    z_et: np.ndarray
    pdry: np.ndarray
    category: np.ndarray
    debug: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def run_forecast(
    *,
    bundle: Dict[str, Any],
    timescale: str,
    as_of_day: Any,
    history_days: int | None = None,
    tau: float = DEFAULT_COPULA_TAU,
    drought_sensitivity: float = 1.0,
) -> ForecastResult:
    """predicts endpoint ET, SM, and drought from historical NLDAS forcing"""
    forecast_start = perf_counter()
    timings_ms: Dict[str, float] = {}
    cache_hits: Dict[str, bool] = {}
    _enable_tf_gpu_memory_growth()

    model_dir = bundle["model_dir"]
    forcing_dir = bundle.get("forcing_dir")
    et_path = bundle.get("et_path")
    sm_path = bundle.get("sm_path")
    model_grid_path = bundle.get("model_grid_path")
    cache_dir = bundle.get("cache_dir")
    processed_cache_dir = bundle.get("processed_cache_dir")

    if forcing_dir is None or et_path is None or sm_path is None:
        raise ValueError("bundle must include forcing_dir, et_path, sm_path")

    as_of = _as_np_date(as_of_day)

    loaded, model_load_meta = _call_lru_cached_with_timing(_get_loaded_models, model_dir)
    timings_ms["model_load"] = model_load_meta["elapsed_ms"]
    cache_hits["model_load"] = model_load_meta["cache_hit"]
    model_warmup = _warmup_models_once(model_dir, loaded)
    tf_device = _require_tf_gpu()

    sm_model = loaded.sm_model
    et_model = loaded.et_model
    sm_norms = loaded.sm_norms
    et_norms = loaded.et_norms
    model_manifest = loaded.manifest

    seq_len_sm = _model_seq_len(sm_model)
    seq_len_et = _model_seq_len(et_model)

    C_sm = _model_in_channels(sm_model)
    C_et = _model_in_channels(et_model)

    if C_sm != 7:
        raise ValueError(f"SM model expected 7 channels but got {C_sm}")
    if C_et != 7:
        raise ValueError(f"Selected ET model expected 7 NLDAS channels but got {C_et}")

    horizon_days = int(bundle["horizon_days"])

    # gives each target model its own historical window
    min_needed = max(seq_len_et, seq_len_sm)
    hist, history_debug = _resolve_history_days(
        timescale=timescale,
        bundle=bundle,
        history_days=history_days,
        min_needed=min_needed,
    )

    (lat2d, lon2d), grid_load_meta = _call_lru_cached_with_timing(
        _get_model_grid_cached,
        forcing_dir,
        et_path,
        sm_path,
        model_grid_path,
    )
    timings_ms["grid_load"] = grid_load_meta["elapsed_ms"]
    cache_hits["grid_load"] = grid_load_meta["cache_hit"]

    start_day = (as_of - np.timedelta64(hist - 1, "D")).astype("datetime64[D]")
    forcing_start = perf_counter()
    X_base_phys, day_keys, used = _load_earthaccess_daily_forcing(
        start_day=start_day,
        as_of=as_of,
        cache_dir=cache_dir,
        processed_cache_dir=processed_cache_dir,
        lat2d=lat2d,
        lon2d=lon2d,
    )
    timings_ms["forcing_load"] = _elapsed_ms(forcing_start)
    forcing_source = "earthaccess_hourly_resampled_daily"
    forcing_meta: Dict[str, Any] = {"earthaccess_used": used}
    if isinstance(used, dict) and isinstance(used.get("timings_ms"), dict):
        forcing_meta["timings_ms"] = used["timings_ms"]
        timings_ms["earthaccess_search_download"] = round(
            float(used["timings_ms"].get("earthaccess_search_download", 0.0)),
            3,
        )
        timings_ms["xarray_open"] = round(float(used["timings_ms"].get("xarray_open", 0.0)), 3)
        timings_ms["daily_resample_regrid"] = round(
            float(used["timings_ms"].get("daily_resample_regrid", 0.0)),
            3,
        )
    if isinstance(used, dict):
        cache_hits["processed_daily"] = bool(used.get("processed_daily_cache_hit"))
        cache_hits["processed_daily_partial"] = bool(used.get("processed_daily_cache_partial_hit"))
        if used.get("raw_hourly_cache_hit") is not None:
            cache_hits["raw_hourly"] = bool(used.get("raw_hourly_cache_hit"))

    if day_keys.size == 0:
        raise RuntimeError("No daily forcing days returned for  requested range.")

    idx_asof = np.where(day_keys == as_of)[0]
    if idx_asof.size == 0:
        raise RuntimeError(
            f"as_of_day {as_of} not present in Earthaccess daily forcing. "
            f"Available range: {day_keys[0]} .. {day_keys[-1]}"
        )
    idx_asof = int(idx_asof[-1])

    sm_in_mu, sm_in_sd = _broadcast_mu_sd(sm_norms["input_mu"], sm_norms["input_sd"], 7)
    sm_y_mu = sm_norms["target_mu"]
    sm_y_sd = sm_norms["target_sd"]

    X_base_norm_for_sm = (X_base_phys - sm_in_mu) / sm_in_sd
    X_base_norm_for_sm = np.nan_to_num(X_base_norm_for_sm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    sm_start = idx_asof - (seq_len_sm - 1)
    if sm_start < 0:
        raise RuntimeError(f"Not enough history for SM model: need {seq_len_sm} days before {as_of}")

    xb_sm = X_base_norm_for_sm[sm_start : idx_asof + 1][None, ...]
    sm_predict_start = perf_counter()
    sm_pred_norm = _predict_single(sm_model, xb_sm)
    timings_ms["sm_predict"] = _elapsed_ms(sm_predict_start)
    sm_y_mu0 = np.asarray(sm_y_mu).reshape(-1)[0]
    sm_y_sd0 = np.asarray(sm_y_sd).reshape(-1)[0]
    sm_pred_phys = sm_pred_norm * float(sm_y_sd0) + float(sm_y_mu0)

    target_day = (as_of + np.timedelta64(horizon_days, "D")).astype("datetime64[D]")

    et_start = idx_asof - (seq_len_et - 1)
    if et_start < 0:
        raise RuntimeError(f"Not enough history for ET model: need {seq_len_et} days before {as_of}")

    X_for_et_phys = X_base_phys[et_start : idx_asof + 1]

    et_in_mu = et_norms["input_mu"]
    et_in_sd = et_norms["input_sd"]
    et_t_mu = et_norms["target_mu"]
    et_t_sd = et_norms["target_sd"]

    et_feature_start = perf_counter()
    X_et_phys = X_for_et_phys
    et_in_mu_b, et_in_sd_b = _broadcast_mu_sd(et_in_mu, et_in_sd, 7)
    timings_ms["et_feature_build"] = _elapsed_ms(et_feature_start)

    X_et_norm = (X_et_phys - et_in_mu_b) / et_in_sd_b
    X_et_norm = np.nan_to_num(X_et_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    xb_et = X_et_norm[None, ...]

    et_predict_start = perf_counter()
    et_pred_norm = _predict_single(et_model, xb_et)
    timings_ms["et_predict"] = _elapsed_ms(et_predict_start)

    et_t_mu0 = np.asarray(et_t_mu).reshape(-1)[0]
    et_t_sd0 = np.asarray(et_t_sd).reshape(-1)[0]
    et_pred_phys = et_pred_norm * float(et_t_sd0) + float(et_t_mu0)

    drought_index_start = perf_counter()
    z_sm = (sm_pred_phys - float(sm_y_mu0)) / float(sm_y_sd0)
    z_et = (et_pred_phys - float(et_t_mu0)) / float(et_t_sd0)

    pdry, cat = drought_from_zscores(
        z_et=z_et,
        z_sm=z_sm,
        tau=tau,
        sensitivity=drought_sensitivity,
    )
    timings_ms["drought_index"] = _elapsed_ms(drought_index_start)
    timings_ms["total_forecast"] = _elapsed_ms(forecast_start)

    debug = {
        "as_of_day": str(as_of),
        "timescale": timescale,
        "target_day": str(target_day),
        "horizon_days": horizon_days,
        "model_dir": model_dir,
        "forcing_dir": forcing_dir,
        "model_grid_path": model_grid_path,
        "processed_cache_dir": processed_cache_dir,
        "forcing_source": forcing_source,
        "forcing_start_day": str(start_day),
        "forcing_end_day": str(as_of),
        "forcing_meta": forcing_meta,
        "seq_len_sm": seq_len_sm,
        "seq_len_et": seq_len_et,
        "C_sm": C_sm,
        "C_et": C_et,
        "model_version": model_manifest["version"],
        "et_model_id": model_manifest["targets"]["ET"]["model_id"],
        "sm_model_id": model_manifest["targets"]["SM"]["model_id"],
        "selection_period": model_manifest["selection"]["period"],
        "prediction_semantics": model_manifest["prediction_semantics"],
        **history_debug,
        "tau": float(tau),
        "drought_sensitivity": float(drought_sensitivity),
        "configured_system_gpu": GPU_DEVICE_ID,
        "gpu_config_source": GPU_CONFIG_SOURCE,
        "tensorflow_device": tf_device,
        "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
        "model_warmup": model_warmup,
        "timings_ms": timings_ms,
        "cache_hits": cache_hits,
    }

    return ForecastResult(
        target_day=target_day,
        horizon_days=horizon_days,
        lat2d=lat2d,
        lon2d=lon2d,
        sm_pred=sm_pred_phys.astype(np.float32),
        et_pred=et_pred_phys.astype(np.float32),
        z_sm=z_sm.astype(np.float32),
        z_et=z_et.astype(np.float32),
        pdry=pdry.astype(np.float32),
        category=cat.astype(np.int8),
        debug=debug,
    )
