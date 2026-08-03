"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Download, Info, Layers3 } from "lucide-react";
import CoordinateCheck from "@/components/CoordinateCheck";
import DroughtLegend from "@/components/DroughtLegend";
import DroughtMap from "@/components/DroughtMap";
import ForecastControls from "@/components/ForecastControls";
import ForecastSummary from "@/components/ForecastSummary";
import ModelDetails from "@/components/ModelDetails";
import SiteHeader from "@/components/SiteHeader";
import {
  getForecastLayerGeoJson,
  getForecastJob,
  getLatestNldasDay,
  getModelMetadata,
  getTimescales,
  submitForecastJob
} from "@/lib/api";
import {
  DEFAULT_TIMESCALES,
  type CoordinateCheckResult,
  type ForecastJob,
  type ForecastLayer,
  type ForecastTimescale,
  type GeoJsonFeatureCollection,
  type ModelBundleMetadata
} from "@/lib/types";

function defaultAsOfDay(): string {
  const date = new Date();
  date.setDate(date.getDate() - 3);
  return date.toISOString().slice(0, 10);
}

function errorMessage(caught: unknown, fallback: string): string {
  return caught instanceof Error ? caught.message : fallback;
}

function isActiveStatus(status: ForecastJob["status"] | undefined): boolean {
  return status === "queued" || status === "running" || status === "unknown";
}

export default function DashboardPage() {
  const [timescales, setTimescales] = useState<ForecastTimescale[]>(DEFAULT_TIMESCALES);
  const [selectedTimescale, setSelectedTimescale] = useState(DEFAULT_TIMESCALES[0].id);
  const [selectedLayer, setSelectedLayer] = useState<ForecastLayer>("drought");
  const [modelMetadata, setModelMetadata] = useState<ModelBundleMetadata[]>([]);
  const [asOfDay, setAsOfDay] = useState(defaultAsOfDay);
  const [latestNldasDay, setLatestNldasDay] = useState<string | null>(null);
  const [nldasDateError, setNldasDateError] = useState<string | null>(null);
  const [modelMetadataError, setModelMetadataError] = useState<string | null>(null);
  const [job, setJob] = useState<ForecastJob | null>(null);
  const [geojson, setGeojson] = useState<GeoJsonFeatureCollection | null>(null);
  const [geojsonKey, setGeojsonKey] = useState<string | null>(null);
  const [coordinateResult, setCoordinateResult] = useState<CoordinateCheckResult | null>(null);
  const [timescaleError, setTimescaleError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [isLoadingTimescales, setIsLoadingTimescales] = useState(true);
  const [isLoadingNldasDate, setIsLoadingNldasDate] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isFetchingGeoJson, setIsFetchingGeoJson] = useState(false);
  const userChangedAsOfDayRef = useRef(false);

  const selectedTimescaleMeta = useMemo(
    () => timescales.find((item) => item.id === selectedTimescale || item.name === selectedTimescale),
    [selectedTimescale, timescales]
  );
  const selectedModelMetadata = useMemo(
    () => modelMetadata.find((item) => item.timescale === selectedTimescale || item.timescale === selectedTimescaleMeta?.name),
    [modelMetadata, selectedTimescale, selectedTimescaleMeta?.name]
  );
  const displayedModelMetadata = useMemo(
    () => modelMetadata.find((item) => item.timescale === job?.timescale),
    [job?.timescale, modelMetadata]
  );

  useEffect(() => {
    let cancelled = false;
    async function loadInitialData() {
      setIsLoadingNldasDate(true);
      setIsLoadingTimescales(true);

      const [timescaleResult, dateResult, metadataResult] = await Promise.allSettled([
        getTimescales(),
        getLatestNldasDay(),
        getModelMetadata()
      ]);
      if (cancelled) return;

      if (timescaleResult.status === "fulfilled") {
        const loaded = timescaleResult.value;
        setTimescales(loaded);
        setSelectedTimescale((current) =>
          loaded.some((item) => item.id === current || item.name === current) ? current : loaded[0]?.id ?? current
        );
      } else {
        setTimescaleError(`Timescale API unavailable; using defaults. ${errorMessage(timescaleResult.reason, "")}`.trim());
      }

      if (dateResult.status === "fulfilled") {
        setLatestNldasDay(dateResult.value.latestAvailableDay);
        if (!userChangedAsOfDayRef.current) setAsOfDay(dateResult.value.latestAvailableDay);
      } else {
        setNldasDateError(`NLDAS availability unavailable; using fallback date. ${errorMessage(dateResult.reason, "")}`.trim());
      }

      if (metadataResult.status === "fulfilled") {
        setModelMetadata(metadataResult.value);
      } else {
        setModelMetadataError(errorMessage(metadataResult.reason, "Unable to load model provenance"));
      }

      setIsLoadingTimescales(false);
      setIsLoadingNldasDate(false);
    }

    void loadInitialData();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const jobId = job?.jobId;
    if (!jobId || !isActiveStatus(job.status)) return;
    const activeJobId: string = jobId;

    let cancelled = false;
    async function poll() {
      try {
        const latest = await getForecastJob(activeJobId);
        if (cancelled) return;
        setJob((current) => {
          if (!current || current.jobId !== activeJobId) return current;
          return {
            ...current,
            ...latest,
            jobId: latest.jobId || current.jobId,
            timescale: latest.timescale ?? current.timescale,
            asOfDay: latest.asOfDay ?? current.asOfDay,
            horizonDays: latest.horizonDays ?? current.horizonDays
          };
        });
        if (latest.status === "error") setRunError(latest.error ?? latest.message ?? "Forecast job failed");
      } catch (caught) {
        if (!cancelled) setRunError(errorMessage(caught, "Unable to poll forecast status"));
      }
    }

    void poll();
    const intervalId = window.setInterval(poll, 2000);
    return () => { cancelled = true; window.clearInterval(intervalId); };
  }, [job?.jobId, job?.status]);

  useEffect(() => {
    const jobId = job?.jobId;
    const nextGeojsonKey = jobId ? `${jobId}:${selectedLayer}` : null;
    if (!jobId || job.status !== "complete" || geojsonKey === nextGeojsonKey) return;
    const completedJobId: string = jobId;

    let cancelled = false;
    async function loadGeoJson() {
      setIsFetchingGeoJson(true);
      setMapError(null);
      setGeojson(null);
      try {
        const nextGeoJson = await getForecastLayerGeoJson(completedJobId, selectedLayer);
        if (!cancelled) {
          setGeojson(nextGeoJson);
          setGeojsonKey(nextGeojsonKey);
        }
      } catch (caught) {
        if (!cancelled) setMapError(errorMessage(caught, "Unable to load drought-risk GeoJSON"));
      } finally {
        if (!cancelled) setIsFetchingGeoJson(false);
      }
    }

    void loadGeoJson();
    return () => { cancelled = true; };
  }, [geojsonKey, job?.jobId, job?.status, selectedLayer]);

  function handleAsOfDayChange(value: string) {
    userChangedAsOfDayRef.current = true;
    setAsOfDay(value);
  }

  async function handleSubmitForecast() {
    setIsSubmitting(true);
    setRunError(null);
    setMapError(null);
    setGeojson(null);
    setGeojsonKey(null);
    setCoordinateResult(null);
    try {
      const submitted = await submitForecastJob({ timescale: selectedTimescale, asOfDay });
      if (!submitted.jobId) throw new Error("Forecast API did not return a job ID.");
      setJob({
        ...submitted,
        timescale: submitted.timescale ?? selectedTimescale,
        asOfDay: submitted.asOfDay ?? asOfDay,
        horizonDays: submitted.horizonDays ?? selectedTimescaleMeta?.horizonDays
      });
    } catch (caught) {
      setJob(null);
      setRunError(errorMessage(caught, "Forecast submission failed"));
    } finally {
      setIsSubmitting(false);
    }
  }

  function downloadGeoJson() {
    if (!geojson) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(geojson)], { type: "application/geo+json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `seed-${selectedLayer}-${selectedTimescale.toLowerCase()}-${job?.targetDay ?? asOfDay}.geojson`;
    link.click();
    URL.revokeObjectURL(url);
  }

  const status = job?.status ?? "idle";
  const isPolling = Boolean(job?.jobId && isActiveStatus(job.status));
  const isMapLoading = isFetchingGeoJson || isSubmitting || status === "queued" || status === "running";
  const resultIsDisplayed = Boolean(job?.status === "complete" && geojson);
  const resultIsStale = Boolean(
    resultIsDisplayed && (job?.timescale !== selectedTimescale || job?.asOfDay !== asOfDay)
  );
  const layerMeta: Record<ForecastLayer, { short: string; title: string }> = {
    et: { short: "ET", title: "Evapotranspiration endpoint" },
    sm: { short: "SM", title: "Soil-moisture endpoint" },
    drought: { short: "Dryness", title: "ET/SM dryness category" }
  };
  const mapContext = resultIsStale
    ? "Setup changed · rerun to update"
    : `${job?.timescale ?? selectedTimescaleMeta?.label ?? "Forecast"} · ${job?.targetDay ?? `lead day ${selectedTimescaleMeta?.horizonDays ?? "--"}`}`;

  return (
    <main className="min-h-screen bg-field text-ink">
      <div className="border-b border-line bg-white px-4 sm:px-6 lg:px-8">
        <SiteHeader />
      </div>

      <div className="border-b border-line bg-white px-4 py-3 sm:px-6 lg:px-8">
        <div className="mx-auto flex max-w-[1720px] flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-lg font-semibold text-forest-strong">Great Plains forecast workspace</h1>
            <p className="mt-0.5 text-xs text-muted">ET and soil-moisture endpoint outlooks</p>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted">
            <span className={`h-2 w-2 rounded-full ${modelMetadata.length ? "bg-accent" : "bg-caution"}`} aria-hidden="true" />
            {modelMetadata.length ? "6 production checkpoints loaded · 3 horizons × 2 targets" : "Checking model bundles"}
          </div>
        </div>
      </div>

      <div className="mx-auto grid max-w-[1720px] bg-white xl:h-[calc(100dvh-136px)] xl:grid-cols-[260px_minmax(0,1fr)_290px]">
        <aside className="border-b border-line bg-panel xl:overflow-y-auto xl:border-b-0 xl:border-r" aria-label="Forecast controls">
          <ForecastControls
            timescales={timescales}
            selectedTimescale={selectedTimescale}
            asOfDay={asOfDay}
            status={status}
            isLoadingTimescales={isLoadingTimescales}
            isSubmitting={isSubmitting}
            isLoadingAsOfDay={isLoadingNldasDate}
            timescaleError={timescaleError}
            asOfDayNote={nldasDateError ?? (latestNldasDay ? `Latest complete NLDAS day: ${latestNldasDay}` : null)}
            asOfDayNoteTone={nldasDateError ? "warning" : "info"}
            maxAsOfDay={latestNldasDay}
            onTimescaleChange={setSelectedTimescale}
            onAsOfDayChange={handleAsOfDayChange}
            onSubmit={handleSubmitForecast}
          />
          <div className="hidden xl:block">
            <CoordinateCheck
              jobId={job?.jobId}
              disabled={status !== "complete" || !job?.jobId}
              onResult={setCoordinateResult}
            />
          </div>
          <div className="hidden p-4 text-[11px] leading-5 text-muted xl:block">
            <div className="flex gap-2">
              <Info aria-hidden="true" className="mt-0.5 shrink-0" size={14} />
              <p>Outputs are endpoint maps at lead day {selectedTimescaleMeta?.horizonDays ?? "--"}, not full forecast sequences.</p>
            </div>
          </div>
        </aside>

        <div className="border-b border-line bg-panel xl:hidden">
          <ForecastSummary
            job={job}
            selectedTimescale={selectedTimescaleMeta}
            featureCount={geojson?.features.length}
            isPolling={isPolling}
            error={runError}
          />
        </div>

        <section className="min-w-0 border-b border-line xl:grid xl:min-h-0 xl:grid-rows-[auto_minmax(0,1fr)] xl:border-b-0" aria-label="Forecast map workspace">
          <div className="border-b border-line bg-white px-4 py-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-2 gap-y-0.5 text-xs">
                <Layers3 aria-hidden="true" className="shrink-0 text-accent" size={16} />
                <span className="shrink-0 font-semibold text-forest-strong">{layerMeta[selectedLayer].title}</span>
                <span className="basis-full pl-6 text-muted sm:basis-auto sm:pl-0">· {mapContext}</span>
              </div>
              <button
                type="button"
                disabled={!geojson}
                onClick={downloadGeoJson}
                className="inline-flex min-h-11 items-center gap-2 rounded-md border border-line-strong bg-white px-3 py-1.5 text-xs font-semibold text-forest hover:border-accent hover:bg-surface focus:outline-none focus:ring-2 focus:ring-accent/25 disabled:cursor-not-allowed disabled:text-muted disabled:opacity-55"
                title={geojson ? `Download ${layerMeta[selectedLayer].title} GeoJSON` : "Available after forecast completes"}
                aria-label={`Download ${layerMeta[selectedLayer].title} GeoJSON`}
              >
                <Download aria-hidden="true" size={14} />
                <span className="hidden sm:inline">Download {layerMeta[selectedLayer].short} GeoJSON</span>
              </button>
            </div>
            <div className="mt-2 flex gap-4 border-t border-line pt-2" role="tablist" aria-label="Forecast map layer">
              {(["et", "sm", "drought"] as ForecastLayer[]).map((layer) => (
                <button
                  key={layer}
                  type="button"
                  role="tab"
                  aria-selected={selectedLayer === layer}
                  onClick={() => setSelectedLayer(layer)}
                  className={`min-h-9 border-b-2 px-1 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-accent/25 ${selectedLayer === layer ? "border-accent text-accent-strong" : "border-transparent text-muted hover:text-ink"}`}
                >
                  {layerMeta[layer].short}
                </button>
              ))}
            </div>
          </div>
          <DroughtMap
            geojson={geojson}
            layer={selectedLayer}
            isLoading={isMapLoading}
            error={mapError}
            selectedPoint={coordinateResult}
            emptyMessage={`Run the selected ${selectedTimescaleMeta?.label?.toLowerCase() ?? ""} forecast to load ${layerMeta[selectedLayer].short}.`}
          />
          <div className="border-t border-line bg-panel xl:hidden">
            <DroughtLegend layer={selectedLayer} />
          </div>
        </section>

        <div className="border-b border-line bg-panel xl:hidden">
          <CoordinateCheck
            jobId={job?.jobId}
            disabled={status !== "complete" || !job?.jobId}
            onResult={setCoordinateResult}
          />
        </div>

        <aside className="bg-panel xl:overflow-y-auto xl:border-l xl:border-line" aria-label="Forecast details">
          <ModelDetails
            model={resultIsDisplayed && displayedModelMetadata ? displayedModelMetadata : selectedModelMetadata}
            error={modelMetadataError}
            resultIsDisplayed={resultIsDisplayed && !resultIsStale}
          />
          <div className="hidden xl:block">
            <ForecastSummary
              job={job}
              selectedTimescale={selectedTimescaleMeta}
              featureCount={geojson?.features.length}
              isPolling={isPolling}
              error={runError}
            />
          </div>
          <div className="hidden xl:block">
            <DroughtLegend layer={selectedLayer} />
          </div>
        </aside>
      </div>
    </main>
  );
}
