"use client";

import { useState } from "react";
import { Search } from "lucide-react";
import { checkCoordinate } from "@/lib/api";
import {
  DROUGHT_COLOR_BY_CODE,
  droughtLabelForCategory,
  normalizeDroughtCategory,
  type CoordinateCheckResult
} from "@/lib/types";

interface CoordinateCheckProps {
  jobId?: string;
  disabled?: boolean;
  onResult?: (result: CoordinateCheckResult | null) => void;
}

function formatNumber(value: number | undefined, digits = 3): string {
  return value === undefined || Number.isNaN(value) ? "--" : value.toFixed(digits);
}

export default function CoordinateCheck({ jobId, disabled = false, onResult }: CoordinateCheckProps) {
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [result, setResult] = useState<CoordinateCheckResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isChecking, setIsChecking] = useState(false);

  const category = normalizeDroughtCategory(result?.categoryLabel ?? result?.category);
  const color = DROUGHT_COLOR_BY_CODE[category];
  const errorId = error ? "coordinate-check-error" : undefined;

  async function handleSubmit() {
    if (!jobId) {
      return;
    }

    const latValue = Number(lat);
    const lonValue = Number(lon);
    if (!Number.isFinite(latValue) || !Number.isFinite(lonValue)) {
      setError("Enter numeric latitude and longitude values.");
      return;
    }

    setIsChecking(true);
    setError(null);
    try {
      const nextResult = await checkCoordinate(jobId, { lat: latValue, lon: lonValue });
      setResult(nextResult);
      onResult?.(nextResult);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Coordinate check failed";
      setError(message);
      setResult(null);
      onResult?.(null);
    } finally {
      setIsChecking(false);
    }
  }

  return (
    <section className="border-b border-line p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-forest-strong">Inspect coordinate</h2>
          <p className="mt-1 text-xs text-muted">Enter a Great Plains point; check after a forecast completes</p>
        </div>
      </div>
      <form
        className="mt-4 grid grid-cols-2 gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (!disabled && !isChecking) {
            handleSubmit();
          }
        }}
      >
        <label className="block">
          <span className="text-xs font-semibold text-muted">Latitude</span>
          <input
            className="mt-1.5 min-h-11 w-full rounded-md border border-line-strong bg-white px-2.5 py-2 text-sm text-ink outline-none placeholder:text-muted focus:border-accent focus:ring-2 focus:ring-accent/20 disabled:cursor-not-allowed disabled:bg-surface disabled:opacity-60"
            inputMode="decimal"
            placeholder="38.500"
            value={lat}
            disabled={isChecking}
            aria-describedby={errorId}
            aria-invalid={Boolean(error)}
            onChange={(event) => setLat(event.target.value)}
          />
        </label>
        <label className="block">
          <span className="text-xs font-semibold text-muted">Longitude</span>
          <input
            className="mt-1.5 min-h-11 w-full rounded-md border border-line-strong bg-white px-2.5 py-2 text-sm text-ink outline-none placeholder:text-muted focus:border-accent focus:ring-2 focus:ring-accent/20 disabled:cursor-not-allowed disabled:bg-surface disabled:opacity-60"
            inputMode="decimal"
            placeholder="-99.500"
            value={lon}
            disabled={isChecking}
            aria-describedby={errorId}
            aria-invalid={Boolean(error)}
            onChange={(event) => setLon(event.target.value)}
          />
        </label>
        <button
          className="col-span-2 inline-flex min-h-11 items-center justify-center gap-2 rounded-md border border-line-strong bg-white px-3 py-2 text-sm font-semibold text-forest transition hover:border-accent/60 hover:bg-surface focus:outline-none focus:ring-2 focus:ring-accent/30 disabled:cursor-not-allowed disabled:bg-surface disabled:text-muted"
          type="submit"
          disabled={disabled || isChecking || !jobId}
        >
          <Search aria-hidden="true" size={14} />
          {isChecking ? "Checking" : "Check coordinate"}
        </button>
      </form>

      {error ? (
        <p id={errorId} className="mt-3 text-sm font-medium text-danger" role="alert">
          {error}
        </p>
      ) : null}

      {result ? (
        <div className="mt-4 border-l-2 border-accent bg-field p-3" role="status" aria-live="polite">
          <div className="flex items-center gap-2">
            <span
              className="h-4 w-4 border border-line"
              style={{ backgroundColor: color }}
              aria-hidden="true"
            />
            <div>
              <p className="text-sm font-semibold text-ink">
                {category === "None" ? "None" : category} - {result.riskLabel ?? droughtLabelForCategory(category)}
              </p>
              <p className="text-xs text-muted">
                Grid: {formatNumber(result.gridLat)}, {formatNumber(result.gridLon)}
              </p>
            </div>
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-3 text-sm">
            <div>
              <dt className="text-xs font-medium text-muted">Target-day ET</dt>
              <dd className="font-semibold text-ink">{result.etMmPerDay === undefined ? "--" : `${result.etMmPerDay.toFixed(3)} mm/day`}</dd>
            </div>
            <div>
              <dt className="text-xs font-medium text-muted">SM at 6.2 cm model depth</dt>
              <dd className="font-semibold text-ink">{result.smM3PerM3 === undefined ? "--" : `${result.smM3PerM3.toFixed(4)} m3/m3`}</dd>
            </div>
            <div>
              <dt className="text-xs font-medium text-muted">ET-SM dryness percentile</dt>
              <dd className="font-semibold text-ink">
                {result.pdryPct !== undefined
                  ? `${result.pdryPct.toFixed(1)}%`
                  : result.pdry !== undefined
                    ? `${(result.pdry * 100).toFixed(1)}%`
                    : "--"}
              </dd>
            </div>
            <div>
              <dt className="text-xs font-medium text-muted">Category</dt>
              <dd className="font-semibold text-ink">{category}</dd>
            </div>
          </dl>
        </div>
      ) : null}
    </section>
  );
}
