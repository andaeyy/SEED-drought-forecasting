from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import Lock
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse, Response

from app.core.forecast_jobs import (
    get_completed_result,
    get_forecast_job,
    list_model_metadata,
    list_timescales,
    submit_forecast_job,
    to_accepted_response,
    to_response,
)
from app.core.map_features import forecast_result_scalar_to_geojson, forecast_result_to_geojson
from app.core.point_risk import drought_risk_at_coordinate
from app.core.schemas import (
    ForecastJobCreate,
    ForecastJobAcceptedResponse,
    ForecastJobResponse,
    HealthResponse,
    ModelBundleMetadata,
    ModelMetadataResponse,
    NldasLatestDayResponse,
    PointRiskResponse,
    TimescaleListResponse,
    TimescaleResponse,
)


router = APIRouter(prefix="/api")
_GEOJSON_CACHE_MAX_ENTRIES = 12
_geojson_cache: OrderedDict[tuple[Any, ...], bytes] = OrderedDict()
_geojson_cache_lock = Lock()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/timescales", response_model=TimescaleListResponse)
def timescales() -> TimescaleListResponse:
    return TimescaleListResponse(timescales=[TimescaleResponse(**item) for item in list_timescales()])


@router.get("/model-metadata", response_model=ModelMetadataResponse)
def model_metadata() -> ModelMetadataResponse:
    try:
        records = list_model_metadata()
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to read active model metadata: {type(exc).__name__}: {exc}",
        ) from exc
    return ModelMetadataResponse(models=[ModelBundleMetadata(**item) for item in records])


@router.get("/nldas/latest-day", response_model=NldasLatestDayResponse)
def latest_nldas_day(
    max_lookback_days: int = Query(21, ge=1, le=60),
    force_refresh: bool = Query(False),
) -> NldasLatestDayResponse:
    from app.core.nldas_availability import get_latest_nldas_availability

    try:
        latest = get_latest_nldas_availability(
            max_lookback_days=max_lookback_days,
            force_refresh=force_refresh,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to determine latest NLDAS day: {type(exc).__name__}: {exc}",
        ) from exc
    return NldasLatestDayResponse(**latest)


@router.post(
    "/forecast-jobs",
    response_model=ForecastJobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_forecast_job(request: ForecastJobCreate) -> ForecastJobAcceptedResponse:
    try:
        record = submit_forecast_job(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return to_accepted_response(record)


@router.get("/forecast-jobs/{job_id}", response_model=ForecastJobResponse)
def read_forecast_job(job_id: str) -> ForecastJobResponse:
    record = get_forecast_job(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast job not found")
    return to_response(record)


@router.get("/forecast-jobs/{job_id}/drought-risk.geojson")
def read_forecast_geojson(job_id: str) -> Response:
    record = _require_completed_record(job_id)
    result = _completed_result_for_record(record)
    cache_key = _geojson_cache_key(record, result, "drought")
    cache_key_hash = _geojson_cache_key_hash(cache_key)
    started_at = perf_counter()

    cached_body = _get_cached_geojson(cache_key)
    if cached_body is not None:
        elapsed_ms = _elapsed_ms(started_at)
        return Response(
            content=cached_body,
            media_type="application/geo+json",
            headers=_geojson_cache_headers(
                cache_hit=True,
                cache_key_hash=cache_key_hash,
                elapsed_ms=elapsed_ms,
                build_ms=None,
            ),
        )

    build_started_at = perf_counter()
    collection = forecast_result_to_geojson(result)
    build_ms = _elapsed_ms(build_started_at)
    response = JSONResponse(content=collection, media_type="application/geo+json")
    _store_cached_geojson(cache_key, bytes(response.body))
    response.headers.update(
        _geojson_cache_headers(
            cache_hit=False,
            cache_key_hash=cache_key_hash,
            elapsed_ms=_elapsed_ms(started_at),
            build_ms=build_ms,
        )
    )
    return response


@router.get("/forecast-jobs/{job_id}/layers/{layer}.geojson")
def read_forecast_layer_geojson(job_id: str, layer: str) -> Response:
    layer_key = layer.strip().lower()
    if layer_key not in {"et", "sm", "drought"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Layer must be one of: et, sm, drought",
        )

    record = _require_completed_record(job_id)
    result = _completed_result_for_record(record)
    cache_key = _geojson_cache_key(record, result, layer_key)
    cache_key_hash = _geojson_cache_key_hash(cache_key)
    started_at = perf_counter()

    cached_body = _get_cached_geojson(cache_key)
    if cached_body is not None:
        return Response(
            content=cached_body,
            media_type="application/geo+json",
            headers=_geojson_cache_headers(
                cache_hit=True,
                cache_key_hash=cache_key_hash,
                elapsed_ms=_elapsed_ms(started_at),
                build_ms=None,
            ),
        )

    build_started_at = perf_counter()
    collection = (
        forecast_result_to_geojson(result)
        if layer_key == "drought"
        else forecast_result_scalar_to_geojson(result, layer=layer_key)
    )
    build_ms = _elapsed_ms(build_started_at)
    response = JSONResponse(content=collection, media_type="application/geo+json")
    _store_cached_geojson(cache_key, bytes(response.body))
    response.headers.update(
        _geojson_cache_headers(
            cache_hit=False,
            cache_key_hash=cache_key_hash,
            elapsed_ms=_elapsed_ms(started_at),
            build_ms=build_ms,
        )
    )
    return response


@router.get("/forecast-jobs/{job_id}/point-risk", response_model=PointRiskResponse)
def read_point_risk(
    job_id: str,
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-360.0, le=360.0),
) -> PointRiskResponse:
    result = _require_completed_result(job_id)
    try:
        return drought_risk_at_coordinate(
            lat=lat,
            lon=lon,
            lat2d=result.lat2d,
            lon2d=result.lon2d,
            pdry=result.pdry,
            category=result.category,
            et_pred=result.et_pred,
            sm_pred=result.sm_pred,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


def _require_completed_result(job_id: str) -> Any:
    record = _require_completed_record(job_id)
    return _completed_result_for_record(record)


def _require_completed_record(job_id: str) -> Any:
    record = get_forecast_job(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast job not found")
    if record.status != "complete":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Forecast job is {record.status}",
        )
    return record


def _completed_result_for_record(record: Any) -> Any:
    result = get_completed_result(record.job_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Forecast result is not available",
        )
    return result


def _geojson_cache_key(record: Any, result: Any, layer: str) -> tuple[Any, ...]:
    forecast_key = getattr(record, "forecast_cache_key_hash", None)
    if forecast_key:
        return ("forecast", forecast_key, layer)
    return ("job", record.job_id, id(result), layer)


def _geojson_cache_key_hash(cache_key: tuple[Any, ...]) -> str:
    payload = ":".join(str(part) for part in cache_key)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _get_cached_geojson(cache_key: tuple[Any, ...]) -> bytes | None:
    with _geojson_cache_lock:
        cached_body = _geojson_cache.get(cache_key)
        if cached_body is None:
            return None
        _geojson_cache.move_to_end(cache_key)
        return cached_body


def _store_cached_geojson(cache_key: tuple[Any, ...], body: bytes) -> None:
    with _geojson_cache_lock:
        _geojson_cache[cache_key] = body
        _geojson_cache.move_to_end(cache_key)
        while len(_geojson_cache) > _GEOJSON_CACHE_MAX_ENTRIES:
            _geojson_cache.popitem(last=False)


def _geojson_cache_headers(
    *,
    cache_hit: bool,
    cache_key_hash: str,
    elapsed_ms: float,
    build_ms: float | None,
) -> dict[str, str]:
    headers = {
        "X-Forecast-Geojson-Cache-Hit": "1" if cache_hit else "0",
        "X-Forecast-Geojson-Cache-Key": cache_key_hash,
        "X-Forecast-Geojson-Elapsed-Ms": f"{elapsed_ms:.3f}",
    }
    if build_ms is not None:
        headers["X-Forecast-Geojson-Build-Ms"] = f"{build_ms:.3f}"
    return headers


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000.0
