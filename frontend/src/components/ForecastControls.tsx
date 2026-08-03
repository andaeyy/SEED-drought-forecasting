"use client";

import { LoaderCircle, Play } from "lucide-react";
import type { ForecastJobStatus, ForecastTimescale } from "@/lib/types";

interface ForecastControlsProps {
  timescales: ForecastTimescale[];
  selectedTimescale: string;
  asOfDay: string;
  status: ForecastJobStatus | "idle";
  isLoadingTimescales?: boolean;
  isSubmitting?: boolean;
  isLoadingAsOfDay?: boolean;
  timescaleError?: string | null;
  asOfDayNote?: string | null;
  asOfDayNoteTone?: "info" | "warning";
  maxAsOfDay?: string | null;
  onTimescaleChange: (value: string) => void;
  onAsOfDayChange: (value: string) => void;
  onSubmit: () => void;
}

export default function ForecastControls({
  timescales,
  selectedTimescale,
  asOfDay,
  status,
  isLoadingTimescales = false,
  isSubmitting = false,
  isLoadingAsOfDay = false,
  timescaleError,
  asOfDayNote,
  asOfDayNoteTone = "info",
  maxAsOfDay,
  onTimescaleChange,
  onAsOfDayChange,
  onSubmit
}: ForecastControlsProps) {
  const selected = timescales.find((item) => item.id === selectedTimescale || item.name === selectedTimescale);
  const isBusy = isSubmitting || status === "queued" || status === "running";
  const noteClass = asOfDayNoteTone === "warning" ? "text-caution" : "text-muted";
  const noteId = asOfDayNote ? "forecast-as-of-day-note" : undefined;
  const runLabel = selected ? `Run ${selected.label} forecast` : "Run forecast";

  return (
    <section className="border-b border-line p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-forest-strong">Forecast setup</h2>
          <p className="mt-1 text-xs text-muted">
            {selected?.horizonDays ? `Endpoint at lead day ${selected.horizonDays}` : "Select an endpoint horizon"}
          </p>
        </div>
      </div>

      <form
        className="mt-5 space-y-5"
        onSubmit={(event) => {
          event.preventDefault();
          if (!isBusy) onSubmit();
        }}
      >
        <fieldset disabled={isBusy || isLoadingTimescales}>
          <legend className="text-xs font-semibold text-muted">Horizon</legend>
          <div className="mt-2 grid grid-cols-3 overflow-hidden rounded-md border border-line-strong bg-field">
            {timescales.map((item) => {
              const active = item.id === selectedTimescale || item.name === selectedTimescale;
              return (
                <button
                  key={item.id}
                  type="button"
                  aria-pressed={active}
                    className={`min-h-11 border-r border-line-strong px-1.5 py-2 text-center last:border-r-0 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-accent/35 disabled:cursor-not-allowed disabled:opacity-60 ${
                    active ? "bg-forest font-semibold text-white" : "text-muted hover:bg-surface hover:text-ink"
                  }`}
                  onClick={() => onTimescaleChange(item.id)}
                >
                  <span className="block text-xs">{item.label}</span>
                  <span className={`block text-[10px] ${active ? "text-white/75" : "text-muted"}`}>
                    {item.horizonDays} d
                  </span>
                </button>
              );
            })}
          </div>
        </fieldset>

        <label className="block">
          <span className="flex items-center justify-between gap-2 text-xs font-semibold text-muted">
            Forecast origin
            {isLoadingAsOfDay ? <LoaderCircle aria-label="Checking NLDAS" className="animate-spin" size={14} /> : null}
          </span>
          <input
            className="mt-2 min-h-11 w-full rounded-md border border-line-strong bg-white px-3 py-2 text-base font-medium text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent/20 disabled:cursor-not-allowed disabled:bg-surface disabled:opacity-60"
            type="date"
            value={asOfDay}
            max={maxAsOfDay ?? undefined}
            disabled={isBusy}
            aria-describedby={noteId}
            onChange={(event) => onAsOfDayChange(event.target.value)}
          />
          {asOfDayNote ? <p id={noteId} className={`mt-1.5 text-[11px] leading-4 ${noteClass}`}>{asOfDayNote}</p> : null}
        </label>

        {timescaleError ? (
          <p className="border-l-2 border-caution bg-[#fff9ec] px-3 py-2 text-xs text-caution" role="alert">
            {timescaleError}
          </p>
        ) : null}

        <button
          type="submit"
          disabled={isBusy || !selectedTimescale || !asOfDay}
          className="inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-md bg-accent px-3 py-2.5 text-sm font-semibold text-white hover:bg-accent-strong focus:outline-none focus:ring-2 focus:ring-accent/30 focus:ring-offset-2 disabled:cursor-not-allowed disabled:bg-line disabled:text-muted"
        >
          {isBusy ? <LoaderCircle aria-hidden="true" className="animate-spin" size={16} /> : <Play aria-hidden="true" size={16} fill="currentColor" />}
          {isBusy ? "Forecast running" : runLabel}
        </button>
      </form>
    </section>
  );
}
