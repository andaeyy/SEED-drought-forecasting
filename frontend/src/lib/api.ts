import {
  DEFAULT_TIMESCALES,
  type CoordinateCheckRequest,
  type CoordinateCheckResult,
  type ForecastJob,
  type ForecastLayer,
  type ForecastJobStatus,
  type ForecastRequest,
  type ForecastTimescale,
  type GeoJsonFeatureCollection,
  type ModelBundleMetadata,
  type ModelTargetMetadata,
  type NldasLatestDay,
  normalizeDroughtCategory
} from "./types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

type JsonRecord = Record<string, unknown>;

export class ApiError extends Error {
  status?: number;
  details?: unknown;

  constructor(message: string, status?: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

function normalizeModelTarget(value: unknown): ModelTargetMetadata {
  const source = isRecord(value) ? value : {};
  const modelId = stringFrom(source.model_id) ?? stringFrom(source.modelId);
  const family = stringFrom(source.family);
  const architecture = stringFrom(source.architecture);
  const trial = stringFrom(source.trial);
  const inputChannels = numberFrom(source.input_channels) ?? numberFrom(source.inputChannels);
  const checkpointSha256 = stringFrom(source.checkpoint_sha256) ?? stringFrom(source.checkpointSha256);

  if (!modelId || !family || !architecture || !trial || inputChannels === undefined || !checkpointSha256) {
    throw new ApiError("Model metadata response is incomplete");
  }

  return { modelId, family, architecture, trial, inputChannels, checkpointSha256 };
}

function normalizeModelMetadata(payload: unknown): ModelBundleMetadata[] {
  const envelope = isRecord(payload) ? payload : {};
  const source = Array.isArray(envelope.models) ? envelope.models : [];

  return source.map((value) => {
    const item = isRecord(value) ? value : {};
    const timescale = stringFrom(item.timescale);
    const version = stringFrom(item.version);
    const inputDays = numberFrom(item.input_days) ?? numberFrom(item.inputDays);
    const horizonDays = numberFrom(item.horizon_days) ?? numberFrom(item.horizonDays);
    const predictionSemantics =
      stringFrom(item.prediction_semantics) ?? stringFrom(item.predictionSemantics);
    const selectionPeriod = stringFrom(item.selection_period) ?? stringFrom(item.selectionPeriod);
    const independentTestPeriod =
      stringFrom(item.independent_test_period) ?? stringFrom(item.independentTestPeriod);
    const inputVariables = Array.isArray(item.input_variables)
      ? item.input_variables.filter((entry): entry is string => typeof entry === "string")
      : [];

    if (
      !timescale ||
      !version ||
      inputDays === undefined ||
      horizonDays === undefined ||
      !predictionSemantics ||
      !selectionPeriod ||
      !independentTestPeriod ||
      inputVariables.length !== 7
    ) {
      throw new ApiError("Model bundle metadata response is incomplete");
    }

    return {
      timescale,
      version,
      inputDays,
      horizonDays,
      predictionSemantics,
      inputVariables,
      selectionPeriod,
      independentTestPeriod,
      et: normalizeModelTarget(item.et),
      sm: normalizeModelTarget(item.sm)
    };
  });
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberFrom(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function stringFrom(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() !== "" ? value : undefined;
}

function joinApiUrl(pathOrUrl: string): string {
  if (/^https?:\/\//i.test(pathOrUrl)) {
    return pathOrUrl;
  }
  const path = pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`;
  return `${API_BASE}${path}`;
}

async function parseResponse(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function requestJson<T>(pathOrUrl: string, init?: RequestInit): Promise<T> {
  const response = await fetch(joinApiUrl(pathOrUrl), {
    ...init,
    headers: {
      Accept: "application/json, application/geo+json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers
    }
  });

  const payload = await parseResponse(response);
  if (!response.ok) {
    const detail = isRecord(payload)
      ? stringFrom(payload.detail) ?? stringFrom(payload.message) ?? stringFrom(payload.error)
      : undefined;
    throw new ApiError(detail ?? `Request failed with HTTP ${response.status}`, response.status, payload);
  }

  return payload as T;
}

function firstRecord(...values: unknown[]): JsonRecord | undefined {
  for (const value of values) {
    if (isRecord(value)) {
      return value;
    }
  }
  return undefined;
}

function normalizeTimescaleItem(value: unknown, fallbackKey?: string): ForecastTimescale | undefined {
  if (typeof value === "string") {
    return {
      id: value,
      name: value,
      label: value,
      horizonDays: DEFAULT_TIMESCALES.find((item) => item.id === value)?.horizonDays
    };
  }

  if (!isRecord(value)) {
    return undefined;
  }

  const name =
    stringFrom(value.name) ??
    stringFrom(value.id) ??
    stringFrom(value.key) ??
    stringFrom(value.label) ??
    fallbackKey;

  if (!name) {
    return undefined;
  }

  return {
    id: stringFrom(value.id) ?? name,
    name,
    label: stringFrom(value.label) ?? name,
    horizonDays:
      numberFrom(value.horizonDays) ??
      numberFrom(value.horizon_days) ??
      numberFrom(value.horizon) ??
      DEFAULT_TIMESCALES.find((item) => item.id === name)?.horizonDays,
    description: stringFrom(value.description)
  };
}

function normalizeTimescales(payload: unknown): ForecastTimescale[] {
  const envelope = isRecord(payload) ? payload : undefined;
  const source = envelope?.timescales ?? envelope?.data ?? envelope?.items ?? payload;

  if (Array.isArray(source)) {
    const normalized = source
      .map((item) => normalizeTimescaleItem(item))
      .filter((item): item is ForecastTimescale => Boolean(item));
    return normalized.length > 0 ? normalized : DEFAULT_TIMESCALES;
  }

  if (isRecord(source)) {
    const normalized = Object.entries(source)
      .map(([key, item]) => normalizeTimescaleItem(item, key))
      .filter((item): item is ForecastTimescale => Boolean(item));
    return normalized.length > 0 ? normalized : DEFAULT_TIMESCALES;
  }

  return DEFAULT_TIMESCALES;
}

function normalizeStatus(value: unknown): ForecastJobStatus {
  const status = stringFrom(value)?.toLowerCase();
  if (!status) {
    return "unknown";
  }
  if (["queued", "pending", "submitted", "created"].includes(status)) {
    return "queued";
  }
  if (["running", "processing", "started", "in_progress", "active"].includes(status)) {
    return "running";
  }
  if (["complete", "completed", "success", "succeeded", "done", "finished"].includes(status)) {
    return "complete";
  }
  if (["failed", "error", "errored", "cancelled", "canceled"].includes(status)) {
    return "error";
  }
  return "unknown";
}

function normalizeBounds(value: unknown): ForecastJob["bounds"] | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const latMin = numberFrom(value.latMin) ?? numberFrom(value.lat_min);
  const latMax = numberFrom(value.latMax) ?? numberFrom(value.lat_max);
  const lonMin = numberFrom(value.lonMin) ?? numberFrom(value.lon_min);
  const lonMax = numberFrom(value.lonMax) ?? numberFrom(value.lon_max);

  if (
    latMin === undefined ||
    latMax === undefined ||
    lonMin === undefined ||
    lonMax === undefined
  ) {
    return undefined;
  }

  return { latMin, latMax, lonMin, lonMax };
}

function normalizeJob(payload: unknown, fallback?: Partial<ForecastJob>): ForecastJob {
  const envelope: JsonRecord = isRecord(payload) ? payload : {};
  const source: JsonRecord = firstRecord(envelope.job, envelope.status, envelope.forecast, envelope) ?? {};
  const result: JsonRecord = firstRecord(source.result, envelope.result, source.summary, envelope.summary) ?? {};

  const jobId =
    stringFrom(source.job_id) ??
    stringFrom(source.jobId) ??
    stringFrom(source.id) ??
    stringFrom(envelope.job_id) ??
    stringFrom(envelope.jobId) ??
    stringFrom(envelope.id) ??
    fallback?.jobId ??
    "";

  const status = normalizeStatus(source.state ?? source.status ?? envelope.state ?? envelope.status);
  const forecast = firstRecord(envelope.forecast, source.forecast, result.forecast);
  const progress =
    numberFrom(source.progressPct) ??
    numberFrom(source.progress_pct) ??
    numberFrom(source.progress) ??
    numberFrom(envelope.progress);
  const reliability =
    numberFrom(result.reliabilityPct) ??
    numberFrom(result.reliability_pct) ??
    numberFrom(forecast?.reliabilityPct) ??
    numberFrom(forecast?.reliability_pct) ??
    numberFrom(result.forecast_reliability) ??
    numberFrom(source.reliabilityPct) ??
    numberFrom(source.reliability_pct);

  return {
    jobId,
    status,
    timescale:
      stringFrom(source.timescale) ??
      stringFrom(result.timescale) ??
      fallback?.timescale,
    asOfDay:
      stringFrom(source.asOfDay) ??
      stringFrom(source.as_of_day) ??
      stringFrom(result.asOfDay) ??
      stringFrom(result.as_of_day) ??
      fallback?.asOfDay,
    targetDay:
      stringFrom(source.targetDay) ??
      stringFrom(source.target_day) ??
      stringFrom(forecast?.targetDay) ??
      stringFrom(forecast?.target_day) ??
      stringFrom(result.targetDay) ??
      stringFrom(result.target_day),
    horizonDays:
      numberFrom(source.horizonDays) ??
      numberFrom(source.horizon_days) ??
      numberFrom(forecast?.horizonDays) ??
      numberFrom(forecast?.horizon_days) ??
      numberFrom(result.horizonDays) ??
      numberFrom(result.horizon_days) ??
      fallback?.horizonDays,
    reliabilityPct: reliability,
    bounds: normalizeBounds(source.bounds) ?? normalizeBounds(forecast?.bounds) ?? normalizeBounds(result.bounds),
    progressPct: progress !== undefined && progress <= 1 ? progress * 100 : progress,
    message:
      stringFrom(source.message) ??
      stringFrom(envelope.message) ??
      (status === "complete" ? "Forecast complete" : undefined),
    error:
      stringFrom(source.error) ??
      stringFrom(source.detail) ??
      stringFrom(envelope.error) ??
      stringFrom(envelope.detail),
    raw: payload
  };
}

function asFeatureCollection(payload: unknown): GeoJsonFeatureCollection | undefined {
  if (isRecord(payload) && payload.type === "FeatureCollection" && Array.isArray(payload.features)) {
    return payload as unknown as GeoJsonFeatureCollection;
  }
  return undefined;
}

async function normalizeGeoJson(payload: unknown): Promise<GeoJsonFeatureCollection> {
  const direct = asFeatureCollection(payload);
  if (direct) {
    return direct;
  }

  if (isRecord(payload)) {
    const nested = asFeatureCollection(payload.geojson) ?? asFeatureCollection(payload.data);
    if (nested) {
      return nested;
    }

    const url = stringFrom(payload.url) ?? stringFrom(payload.href) ?? stringFrom(payload.geojson_url);
    if (url) {
      const linked = await requestJson<unknown>(url);
      const linkedGeoJson = asFeatureCollection(linked);
      if (linkedGeoJson) {
        return linkedGeoJson;
      }
    }
  }

  throw new ApiError("Drought risk response was not a GeoJSON FeatureCollection");
}

function normalizeNldasLatestDay(payload: unknown): NldasLatestDay {
  const envelope: JsonRecord = isRecord(payload) ? payload : {};
  const source: JsonRecord = firstRecord(envelope.data, envelope) ?? {};
  const latestAvailableDay =
    stringFrom(source.latestAvailableDay) ??
    stringFrom(source.latest_available_day) ??
    stringFrom(source.asOfDay) ??
    stringFrom(source.as_of_day);

  if (!latestAvailableDay) {
    throw new ApiError("NLDAS availability response did not include a latest available day");
  }

  return {
    shortName: stringFrom(source.shortName) ?? stringFrom(source.short_name),
    latestAvailableDay,
    latestGranuleTime: stringFrom(source.latestGranuleTime) ?? stringFrom(source.latest_granule_time),
    completeHourCount: numberFrom(source.completeHourCount) ?? numberFrom(source.complete_hour_count),
    checkedDays: numberFrom(source.checkedDays) ?? numberFrom(source.checked_days),
    source: stringFrom(source.source),
    raw: payload
  };
}

function normalizeCoordinateResult(payload: unknown): CoordinateCheckResult {
  const envelope: JsonRecord = isRecord(payload) ? payload : {};
  const source: JsonRecord = firstRecord(envelope.result, envelope.coordinate, envelope.point, envelope) ?? {};
  const categoryValue = numberFrom(source.category) ?? numberFrom(source.drought_category);
  const categoryLabel =
    stringFrom(source.category_label) ??
    stringFrom(source.categoryLabel) ??
    stringFrom(source.drought_category_label) ??
    (categoryValue !== undefined ? normalizeDroughtCategory(categoryValue) : undefined);

  return {
    inputLat:
      numberFrom(source.requested_lat) ??
      numberFrom(source.requestedLat) ??
      numberFrom(source.input_lat) ??
      numberFrom(source.inputLat) ??
      numberFrom(source.lat),
    inputLon:
      numberFrom(source.requested_lon) ??
      numberFrom(source.requestedLon) ??
      numberFrom(source.input_lon) ??
      numberFrom(source.inputLon) ??
      numberFrom(source.lon),
    gridLat: numberFrom(source.grid_lat) ?? numberFrom(source.gridLat) ?? numberFrom(source.nearest_lat),
    gridLon: numberFrom(source.grid_lon) ?? numberFrom(source.gridLon) ?? numberFrom(source.nearest_lon),
    pdry: numberFrom(source.pdry) ?? numberFrom(source.probability_dry),
    pdryPct: numberFrom(source.pdry_pct) ?? numberFrom(source.pdryPct),
    category: categoryValue,
    categoryLabel,
    riskLabel: stringFrom(source.risk_label) ?? stringFrom(source.riskLabel),
    etMmPerDay: numberFrom(source.et_mm_per_day) ?? numberFrom(source.etMmPerDay),
    smM3PerM3: numberFrom(source.sm_m3_per_m3) ?? numberFrom(source.smM3PerM3),
    raw: payload
  };
}

export async function getTimescales(): Promise<ForecastTimescale[]> {
  const payload = await requestJson<unknown>("/api/timescales");
  return normalizeTimescales(payload);
}

export async function getModelMetadata(): Promise<ModelBundleMetadata[]> {
  const payload = await requestJson<unknown>("/api/model-metadata");
  return normalizeModelMetadata(payload);
}

export async function getLatestNldasDay(): Promise<NldasLatestDay> {
  const payload = await requestJson<unknown>("/api/nldas/latest-day");
  return normalizeNldasLatestDay(payload);
}

export async function submitForecastJob(request: ForecastRequest): Promise<ForecastJob> {
  const payload = await requestJson<unknown>("/api/forecast-jobs", {
    method: "POST",
    body: JSON.stringify({
      timescale: request.timescale,
      as_of_day: request.asOfDay
    })
  });

  return normalizeJob(payload, {
    timescale: request.timescale,
    asOfDay: request.asOfDay
  });
}

export async function getForecastJob(jobId: string): Promise<ForecastJob> {
  const payload = await requestJson<unknown>(`/api/forecast-jobs/${encodeURIComponent(jobId)}`);
  return normalizeJob(payload, { jobId });
}

export async function getDroughtRiskGeoJson(jobId: string): Promise<GeoJsonFeatureCollection> {
  const payload = await requestJson<unknown>(
    `/api/forecast-jobs/${encodeURIComponent(jobId)}/drought-risk.geojson`
  );

  return normalizeGeoJson(payload);
}

export async function getForecastLayerGeoJson(
  jobId: string,
  layer: ForecastLayer
): Promise<GeoJsonFeatureCollection> {
  const payload = await requestJson<unknown>(
    `/api/forecast-jobs/${encodeURIComponent(jobId)}/layers/${encodeURIComponent(layer)}.geojson`
  );

  return normalizeGeoJson(payload);
}

export async function checkCoordinate(
  jobId: string,
  request: CoordinateCheckRequest
): Promise<CoordinateCheckResult> {
  const params = new URLSearchParams({
    lat: String(request.lat),
    lon: String(request.lon)
  });
  const payload = await requestJson<unknown>(
    `/api/forecast-jobs/${encodeURIComponent(jobId)}/point-risk?${params.toString()}`
  );

  return normalizeCoordinateResult(payload);
}
