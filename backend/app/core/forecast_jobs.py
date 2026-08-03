from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, is_dataclass, replace
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

import numpy as np

from app.core.schemas import (
    BoundsResponse,
    ForecastJobAcceptedResponse,
    ForecastJobCreate,
    ForecastJobResponse,
    ForecastSummaryResponse,
    JobStatus,
)


ForecastCacheKey = tuple[Any, ...]


@dataclass
class ForecastJobRecord:
    job_id: str
    request: ForecastJobCreate
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: Any | None = None
    future: Future[Any] | None = None
    forecast_cache_key: ForecastCacheKey | None = None
    forecast_cache_key_hash: str | None = None
    coalesced_to_job_id: str | None = None


_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forecast-job")
_jobs: dict[str, ForecastJobRecord] = {}
_result_cache: OrderedDict[ForecastCacheKey, Any] = OrderedDict()
_inflight_by_cache_key: dict[ForecastCacheKey, str] = {}
_lock = Lock()
_RESULT_CACHE_MAX_ENTRIES = 12
_JOBS_MAX_ENTRIES = int(os.environ.get("FORECAST_JOBS_MAX_ENTRIES", "32"))
_FORECAST_CACHE_VERSION = "forecast_result_v1"
_MODEL_ARTIFACT_NAMES = (
    "keras_convlstm_sm_best.keras",
    "keras_convlstm_et_best.keras",
    "keras_convlstm_sm_norms.npz",
    "keras_convlstm_et_norms.npz",
    "model_manifest.json",
)


def list_timescales() -> list[dict[str, Any]]:
    config = _adapt_config()
    return [
        {
            "name": spec.name,
            "horizon_days": int(spec.horizon_days),
        }
        for spec in config.TIMESCALES.values()
    ]


def list_model_metadata() -> list[dict[str, Any]]:
    config = _adapt_config()
    records: list[dict[str, Any]] = []
    for spec in config.TIMESCALES.values():
        model_dir = _resolve_timescale_model_dir(config.BASE_DIR, spec.parent_dirs, spec.best_arch_folder)
        manifest_path = os.path.join(model_dir, "model_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        targets = manifest["targets"]

        def target_metadata(target: str) -> dict[str, Any]:
            item = targets[target]
            return {
                "model_id": item["model_id"],
                "family": item["family"],
                "architecture": item["architecture"],
                "trial": item["trial"],
                "input_channels": int(item["input_channels"]),
                "checkpoint_sha256": str(item["checkpoint"]["sha256"])[:16],
            }

        records.append(
            {
                "timescale": manifest["timescale"],
                "version": manifest["version"],
                "input_days": int(manifest["input_days"]),
                "horizon_days": int(manifest["horizon_days"]),
                "prediction_semantics": manifest["prediction_semantics"],
                "input_variables": list(manifest["input_variables"]),
                "selection_period": manifest["selection"]["period"],
                "independent_test_period": manifest["independent_test_period"],
                "et": target_metadata("ET"),
                "sm": target_metadata("SM"),
            }
        )
    return records


def submit_forecast_job(request: ForecastJobCreate) -> ForecastJobRecord:
    config = _adapt_config()
    if request.timescale not in config.TIMESCALES:
        allowed = ", ".join(config.TIMESCALES)
        raise ValueError(f"Unknown timescale {request.timescale!r}. Allowed values: {allowed}")

    request = _with_config_defaults(request, config)
    forecast_cache_key = _forecast_cache_key(request, config)
    forecast_cache_key_hash = _short_cache_key_hash(forecast_cache_key) if forecast_cache_key is not None else None
    now = _utc_now()
    job_id = str(uuid4())
    record = ForecastJobRecord(
        job_id=job_id,
        request=request,
        status="queued",
        created_at=now,
        updated_at=now,
        forecast_cache_key=forecast_cache_key,
        forecast_cache_key_hash=forecast_cache_key_hash,
    )

    with _lock:
        if forecast_cache_key is not None:
            cached_result = _get_cached_result_locked(forecast_cache_key)
            if cached_result is not None:
                record.status = "complete"
                record.started_at = now
                record.completed_at = now
                record.result = _result_with_cache_debug(
                    cached_result,
                    cache_key_hash=forecast_cache_key_hash,
                    cache_hit=True,
                    coalesced=False,
                )
                _jobs[job_id] = record
                _prune_jobs_locked(keep_job_id=job_id)
                return _copy_record(record)

            source_job_id = _inflight_by_cache_key.get(forecast_cache_key)
            source_record = _jobs.get(source_job_id) if source_job_id is not None else None
            if source_record is not None and source_record.status in ("queued", "running"):
                record.coalesced_to_job_id = source_job_id
                _jobs[job_id] = record
                return _copy_record(record)
            if source_job_id is not None:
                _inflight_by_cache_key.pop(forecast_cache_key, None)

        _jobs[job_id] = record
        if forecast_cache_key is not None:
            _inflight_by_cache_key[forecast_cache_key] = job_id
        record.future = _executor.submit(_run_forecast_job, job_id)

    return get_forecast_job(job_id)


def get_forecast_job(job_id: str) -> ForecastJobRecord | None:
    with _lock:
        record = _jobs.get(job_id)
        if record is None:
            return None
        return _copy_record(record)


def get_completed_result(job_id: str) -> Any | None:
    with _lock:
        record = _jobs.get(job_id)
        if record is None or record.status != "complete":
            return None
        return record.result


def require_known_job(job_id: str) -> ForecastJobRecord | None:
    return get_forecast_job(job_id)


def to_accepted_response(record: ForecastJobRecord) -> ForecastJobAcceptedResponse:
    return ForecastJobAcceptedResponse(job_id=record.job_id, status="queued")


def to_response(record: ForecastJobRecord) -> ForecastJobResponse:
    forecast = _forecast_summary(record.result) if record.result is not None else None
    return ForecastJobResponse(
        job_id=record.job_id,
        status=record.status,
        error=record.error,
        forecast=forecast,
    )


def _run_forecast_job(job_id: str) -> None:
    _mark_started(job_id)

    try:
        with _lock:
            record = _jobs[job_id]
            request = record.request
            cache_key_hash = record.forecast_cache_key_hash

        result = _execute_forecast(request)
        result = _result_with_cache_debug(
            result,
            cache_key_hash=cache_key_hash,
            cache_hit=False,
            coalesced=False,
        )
    except Exception as exc:
        _mark_error(job_id, f"{type(exc).__name__}: {exc}")
        return

    _mark_complete(job_id, result)


def _execute_forecast(request: ForecastJobCreate) -> Any:
    config = _adapt_config()
    from app.adapt.inference import run_forecast

    spec = config.TIMESCALES[request.timescale]
    model_dir = _resolve_timescale_model_dir(config.BASE_DIR, spec.parent_dirs, spec.best_arch_folder)
    bundle = {
        **config.BUNDLES[config.DEFAULT_BUNDLE_KEY],
        "model_dir": model_dir,
        "horizon_days": int(spec.horizon_days),
    }

    return run_forecast(
        bundle=bundle,
        timescale=request.timescale,
        as_of_day=request.as_of_day,
        history_days=None if request.history_days is None else int(request.history_days),
        tau=float(request.tau),
        drought_sensitivity=request.drought_sensitivity,
    )


def _with_config_defaults(request: ForecastJobCreate, config: Any) -> ForecastJobCreate:
    updates: dict[str, Any] = {}
    if request.tau is None:
        updates["tau"] = config.DEFAULT_COPULA_TAU
    if not updates:
        return request
    if hasattr(request, "model_copy"):
        return request.model_copy(update=updates)
    return request.copy(update=updates)


def _adapt_config() -> Any:
    from app.adapt import config

    return config


def _mark_started(job_id: str) -> None:
    now = _utc_now()
    with _lock:
        record = _jobs[job_id]
        record.status = "running"
        record.started_at = now
        record.updated_at = now


def _mark_error(job_id: str, error: str) -> None:
    now = _utc_now()
    with _lock:
        record = _jobs[job_id]
        record.status = "error"
        record.error = error
        record.completed_at = now
        record.updated_at = now
        _clear_inflight_locked(record)
        _mark_coalesced_error_locked(job_id, error, now)
        _prune_jobs_locked(keep_job_id=job_id)


def _mark_complete(job_id: str, result: Any) -> None:
    now = _utc_now()
    with _lock:
        record = _jobs[job_id]
        record.status = "complete"
        record.result = result
        record.completed_at = now
        record.updated_at = now
        _clear_inflight_locked(record)
        if record.forecast_cache_key is not None:
            _store_cached_result_locked(record.forecast_cache_key, result)
        _mark_coalesced_complete_locked(job_id, result, now)
        _prune_jobs_locked(keep_job_id=job_id)


def _forecast_summary(result: Any) -> ForecastSummaryResponse:
    lat2d = np.asarray(result.lat2d, dtype=np.float32)
    lon2d = np.asarray(result.lon2d, dtype=np.float32)
    lon_180 = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)

    return ForecastSummaryResponse(
        target_day=str(result.target_day),
        horizon_days=int(result.horizon_days),
        reliability_pct=None,
        bounds=BoundsResponse(
            lat_min=float(np.nanmin(lat2d)),
            lat_max=float(np.nanmax(lat2d)),
            lon_min=float(np.nanmin(lon_180)),
            lon_max=float(np.nanmax(lon_180)),
        ),
        debug=dict(getattr(result, "debug", {}) or {}),
    )


def _copy_record(record: ForecastJobRecord) -> ForecastJobRecord:
    return ForecastJobRecord(
        job_id=record.job_id,
        request=record.request,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        error=record.error,
        result=record.result,
        future=record.future,
        forecast_cache_key=record.forecast_cache_key,
        forecast_cache_key_hash=record.forecast_cache_key_hash,
        coalesced_to_job_id=record.coalesced_to_job_id,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _forecast_cache_key(request: ForecastJobCreate, config: Any) -> ForecastCacheKey | None:
    try:
        spec = config.TIMESCALES[request.timescale]
        model_dir = _resolve_timescale_model_dir(config.BASE_DIR, spec.parent_dirs, spec.best_arch_folder)
        bundle = config.BUNDLES[config.DEFAULT_BUNDLE_KEY]
        history_days = (
            int(request.history_days)
            if request.history_days is not None
            else int(getattr(spec, "default_history_days", config.DEFAULT_HISTORY_DAYS))
        )
        return (
            ("forecast_cache_version", _FORECAST_CACHE_VERSION),
            ("timescale", str(request.timescale)),
            ("as_of_day", str(request.as_of_day)),
            ("history_days", history_days),
            ("tau", float(request.tau)),
            ("drought_sensitivity", float(request.drought_sensitivity)),
            ("model_dir", os.path.realpath(model_dir)),
            ("horizon_days", int(spec.horizon_days)),
            ("model_artifacts", _model_artifact_fingerprint(model_dir)),
            (
                "bundle",
                (
                    ("forcing_dir", _normalized_optional_path(bundle.get("forcing_dir"))),
                    ("et_path", _normalized_optional_path(bundle.get("et_path"))),
                    ("sm_path", _normalized_optional_path(bundle.get("sm_path"))),
                    ("model_grid_path", _normalized_optional_path(bundle.get("model_grid_path"))),
                    ("cache_dir", _normalized_optional_path(bundle.get("cache_dir"))),
                    ("processed_cache_dir", _normalized_optional_path(bundle.get("processed_cache_dir"))),
                ),
            ),
        )
    except Exception:
        return None


def _resolve_timescale_model_dir(base_dir: str, parent_dirs: list[str], best_arch_folder: str) -> str:
    for parent_dir in parent_dirs:
        candidate = os.path.join(base_dir, parent_dir, best_arch_folder)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find model folder. Tried: "
        + ", ".join(os.path.join(base_dir, parent_dir, best_arch_folder) for parent_dir in parent_dirs)
    )


def _model_artifact_fingerprint(model_dir: str) -> tuple[tuple[str, int | None, int | None], ...]:
    fingerprint: list[tuple[str, int | None, int | None]] = []
    for artifact_name in _MODEL_ARTIFACT_NAMES:
        artifact_path = os.path.join(model_dir, artifact_name)
        try:
            stat = os.stat(artifact_path)
        except OSError:
            fingerprint.append((artifact_name, None, None))
            continue
        fingerprint.append((artifact_name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(fingerprint)


def _normalized_optional_path(value: Any) -> str | None:
    if value is None:
        return None
    return os.path.realpath(str(value))


def _short_cache_key_hash(cache_key: ForecastCacheKey) -> str:
    payload = json.dumps(cache_key, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _get_cached_result_locked(cache_key: ForecastCacheKey) -> Any | None:
    cached_result = _result_cache.get(cache_key)
    if cached_result is None:
        return None
    _result_cache.move_to_end(cache_key)
    return cached_result


def _store_cached_result_locked(cache_key: ForecastCacheKey, result: Any) -> None:
    _result_cache[cache_key] = result
    _result_cache.move_to_end(cache_key)
    while len(_result_cache) > _RESULT_CACHE_MAX_ENTRIES:
        _result_cache.popitem(last=False)


def _prune_jobs_locked(keep_job_id: str | None = None) -> None:
    if _JOBS_MAX_ENTRIES <= 0 or len(_jobs) <= _JOBS_MAX_ENTRIES:
        return

    candidates = [
        record
        for record in _jobs.values()
        if record.job_id != keep_job_id and record.status in ("complete", "error")
    ]
    candidates.sort(key=lambda record: record.updated_at)

    while len(_jobs) > _JOBS_MAX_ENTRIES and candidates:
        record = candidates.pop(0)
        _jobs.pop(record.job_id, None)


def _clear_inflight_locked(record: ForecastJobRecord) -> None:
    cache_key = record.forecast_cache_key
    if cache_key is not None and _inflight_by_cache_key.get(cache_key) == record.job_id:
        del _inflight_by_cache_key[cache_key]


def _mark_coalesced_complete_locked(source_job_id: str, result: Any, now: datetime) -> None:
    for record in _jobs.values():
        if record.coalesced_to_job_id != source_job_id or record.status != "queued":
            continue
        record.status = "complete"
        record.started_at = now
        record.completed_at = now
        record.updated_at = now
        record.result = _result_with_cache_debug(
            result,
            cache_key_hash=record.forecast_cache_key_hash,
            cache_hit=False,
            coalesced=True,
        )


def _mark_coalesced_error_locked(source_job_id: str, error: str, now: datetime) -> None:
    for record in _jobs.values():
        if record.coalesced_to_job_id != source_job_id or record.status != "queued":
            continue
        record.status = "error"
        record.error = error
        record.completed_at = now
        record.updated_at = now


def _result_with_cache_debug(
    result: Any,
    *,
    cache_key_hash: str | None,
    cache_hit: bool,
    coalesced: bool,
) -> Any:
    debug = dict(_result_debug(result))
    debug["forecast_result_cache_hit"] = bool(cache_hit)
    if cache_key_hash is not None:
        debug["forecast_result_cache_key"] = cache_key_hash
    if coalesced:
        debug["forecast_result_cache_coalesced"] = True

    if isinstance(result, dict):
        cloned = dict(result)
        cloned["debug"] = debug
        return cloned

    if is_dataclass(result) and not isinstance(result, type):
        try:
            return replace(result, debug=debug)
        except TypeError:
            pass

    try:
        cloned = copy.copy(result)
        setattr(cloned, "debug", debug)
        return cloned
    except Exception:
        try:
            setattr(result, "debug", debug)
        except Exception:
            return result
        return result


def _result_debug(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        raw_debug = result.get("debug", {})
    else:
        raw_debug = getattr(result, "debug", {})
    if isinstance(raw_debug, dict):
        return raw_debug
    return dict(raw_debug or {})
