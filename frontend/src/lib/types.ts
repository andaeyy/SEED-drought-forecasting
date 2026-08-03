export type DroughtCategoryCode = "None" | "D0" | "D1" | "D2" | "D3" | "D4";
export type ForecastLayer = "et" | "sm" | "drought";

export const DROUGHT_CATEGORIES: Array<{
  value: number;
  code: DroughtCategoryCode;
  label: string;
  color: string;
}> = [
  { value: 0, code: "None", label: "None / Normal", color: "#FFFFFF" },
  { value: 1, code: "D0", label: "Abnormally Dry", color: "#FFFF00" },
  { value: 2, code: "D1", label: "Moderate Drought", color: "#FCD37F" },
  { value: 3, code: "D2", label: "Severe Drought", color: "#FFAA00" },
  { value: 4, code: "D3", label: "Extreme Drought", color: "#E60000" },
  { value: 5, code: "D4", label: "Exceptional Drought", color: "#730000" }
];

export const DROUGHT_COLOR_BY_CODE: Record<DroughtCategoryCode, string> =
  DROUGHT_CATEGORIES.reduce(
    (acc, item) => {
      acc[item.code] = item.color;
      return acc;
    },
    {} as Record<DroughtCategoryCode, string>
  );

export const DEFAULT_TIMESCALES: ForecastTimescale[] = [
  { id: "Weekly", name: "Weekly", label: "Weekly", horizonDays: 7 },
  { id: "Monthly", name: "Monthly", label: "Monthly", horizonDays: 30 },
  { id: "Seasonal", name: "Seasonal", label: "Seasonal", horizonDays: 90 }
];

export interface ForecastTimescale {
  id: string;
  name: string;
  label: string;
  horizonDays?: number;
  description?: string;
}

export interface ModelTargetMetadata {
  modelId: string;
  family: string;
  architecture: string;
  trial: string;
  inputChannels: number;
  checkpointSha256: string;
}

export interface ModelBundleMetadata {
  timescale: string;
  version: string;
  inputDays: number;
  horizonDays: number;
  predictionSemantics: string;
  inputVariables: string[];
  selectionPeriod: string;
  independentTestPeriod: string;
  et: ModelTargetMetadata;
  sm: ModelTargetMetadata;
}

export interface ForecastRequest {
  timescale: string;
  asOfDay: string;
}

export interface NldasLatestDay {
  shortName?: string;
  latestAvailableDay: string;
  latestGranuleTime?: string;
  completeHourCount?: number;
  checkedDays?: number;
  source?: string;
  raw?: unknown;
}

export type ForecastJobStatus = "queued" | "running" | "complete" | "error" | "unknown";

export interface ForecastBounds {
  latMin: number;
  latMax: number;
  lonMin: number;
  lonMax: number;
}

export interface ForecastJob {
  jobId: string;
  status: ForecastJobStatus;
  timescale?: string;
  asOfDay?: string;
  targetDay?: string;
  horizonDays?: number;
  reliabilityPct?: number;
  bounds?: ForecastBounds;
  progressPct?: number;
  message?: string;
  error?: string;
  raw?: unknown;
}

export interface CoordinateCheckRequest {
  lat: number;
  lon: number;
}

export interface CoordinateCheckResult {
  inputLat?: number;
  inputLon?: number;
  gridLat?: number;
  gridLon?: number;
  pdry?: number;
  pdryPct?: number;
  category?: number;
  categoryLabel?: DroughtCategoryCode | string;
  riskLabel?: string;
  etMmPerDay?: number;
  smM3PerM3?: number;
  raw?: unknown;
}

export type GeoJsonPosition = [number, number] | [number, number, number];

export type GeoJsonGeometry =
  | { type: "Point"; coordinates: GeoJsonPosition }
  | { type: "MultiPoint"; coordinates: GeoJsonPosition[] }
  | { type: "LineString"; coordinates: GeoJsonPosition[] }
  | { type: "MultiLineString"; coordinates: GeoJsonPosition[][] }
  | { type: "Polygon"; coordinates: GeoJsonPosition[][] }
  | { type: "MultiPolygon"; coordinates: GeoJsonPosition[][][] }
  | { type: "GeometryCollection"; geometries: GeoJsonGeometry[] };

export interface GeoJsonFeature {
  type: "Feature";
  geometry: GeoJsonGeometry | null;
  properties: Record<string, unknown> | null;
  id?: string | number;
}

export interface GeoJsonFeatureCollection {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
  bbox?: number[];
  properties?: Record<string, unknown>;
}

export interface DroughtMapFeatureProperties {
  droughtCategory: DroughtCategoryCode;
  droughtCategoryValue: number;
  droughtColor: string;
  droughtOpacity: number;
  droughtLabel: string;
  [key: string]: unknown;
}

const LABEL_TO_CODE: Record<string, DroughtCategoryCode> = {
  none: "None",
  normal: "None",
  "none / normal": "None",
  "no drought": "None",
  "0": "None",
  d0: "D0",
  "1": "D0",
  "abnormally dry": "D0",
  d1: "D1",
  "2": "D1",
  "moderate drought": "D1",
  d2: "D2",
  "3": "D2",
  "severe drought": "D2",
  d3: "D3",
  "4": "D3",
  "extreme drought": "D3",
  d4: "D4",
  "5": "D4",
  "exceptional drought": "D4"
};

export function normalizeDroughtCategory(value: unknown): DroughtCategoryCode {
  if (typeof value === "number" && Number.isFinite(value)) {
    const rounded = Math.round(value);
    return DROUGHT_CATEGORIES.find((item) => item.value === rounded)?.code ?? "None";
  }

  if (typeof value === "string") {
    const key = value.trim().toLowerCase();
    return LABEL_TO_CODE[key] ?? "None";
  }

  return "None";
}

export function droughtCategoryValue(code: DroughtCategoryCode): number {
  return DROUGHT_CATEGORIES.find((item) => item.code === code)?.value ?? 0;
}

export function droughtLabelForCategory(code: DroughtCategoryCode): string {
  return DROUGHT_CATEGORIES.find((item) => item.code === code)?.label ?? "None / Normal";
}

export function droughtColorForCategory(value: unknown): string {
  return DROUGHT_COLOR_BY_CODE[normalizeDroughtCategory(value)];
}
