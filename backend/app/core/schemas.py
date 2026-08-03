from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field


def default_as_of_day() -> date:
    return date.today() - timedelta(days=3)


JobStatus = Literal["queued", "running", "complete", "error"]


class HealthResponse(BaseModel):
    status: str = "ok"


class TimescaleResponse(BaseModel):
    name: str
    horizon_days: int


class TimescaleListResponse(BaseModel):
    timescales: list[TimescaleResponse]


class ModelTargetMetadata(BaseModel):
    model_id: str
    family: str
    architecture: str
    trial: str
    input_channels: int
    checkpoint_sha256: str


class ModelBundleMetadata(BaseModel):
    timescale: str
    version: str
    input_days: int
    horizon_days: int
    prediction_semantics: str
    input_variables: list[str]
    selection_period: str
    independent_test_period: str
    et: ModelTargetMetadata
    sm: ModelTargetMetadata


class ModelMetadataResponse(BaseModel):
    models: list[ModelBundleMetadata]


class NldasLatestDayResponse(BaseModel):
    short_name: str
    latest_available_day: date
    latest_granule_time: datetime | None = None
    complete_hour_count: int
    checked_days: int
    checked_recent_days: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "earthaccess"


class ForecastJobCreate(BaseModel):
    timescale: str = "Weekly"
    as_of_day: date = Field(default_factory=default_as_of_day)
    history_days: int | None = Field(default=None, ge=1)
    tau: float | None = Field(default=None, ge=0.000001, lt=1.0)
    drought_sensitivity: float = Field(default=1.0, ge=0.5, le=2.0)


class BoundsResponse(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class ForecastSummaryResponse(BaseModel):
    target_day: str
    horizon_days: int
    reliability_pct: float | None = None
    bounds: BoundsResponse
    debug: dict[str, Any]


class ForecastJobAcceptedResponse(BaseModel):
    job_id: str
    status: Literal["queued"]


class ForecastJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    error: str | None = None
    forecast: ForecastSummaryResponse | None = None


class PointRiskResponse(BaseModel):
    requested_lat: float
    requested_lon: float
    grid_lat: float
    grid_lon: float
    grid_row: int
    grid_col: int
    pdry: float
    pdry_pct: float
    category: int
    category_label: str
    risk_label: str
    et_mm_per_day: float | None = None
    sm_m3_per_m3: float | None = None
