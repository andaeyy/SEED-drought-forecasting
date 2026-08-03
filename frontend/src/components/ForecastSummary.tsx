"use client";

import { CircleAlert, CircleCheck, Clock3 } from "lucide-react";
import type { ForecastJob, ForecastJobStatus, ForecastTimescale } from "@/lib/types";

interface ForecastSummaryProps {
  job: ForecastJob | null;
  selectedTimescale?: ForecastTimescale;
  featureCount?: number;
  isPolling?: boolean;
  error?: string | null;
}

const STATUS_LABEL: Record<ForecastJobStatus | "idle", string> = {
  idle: "Not run",
  queued: "Queued",
  running: "Running",
  complete: "Complete",
  error: "Failed",
  unknown: "Checking"
};

function statusClass(status: ForecastJobStatus | "idle"): string {
  if (status === "complete") return "border-accent/25 bg-[#edf7ef] text-accent-strong";
  if (status === "error") return "border-danger/25 bg-[#fff2f0] text-danger";
  if (status === "running" || status === "queued") return "border-caution/25 bg-[#fff9ec] text-caution";
  return "border-line bg-field text-muted";
}

export default function ForecastSummary({ job, selectedTimescale, featureCount, isPolling = false, error }: ForecastSummaryProps) {
  const status = job?.status ?? "idle";
  const StatusIcon = status === "complete" ? CircleCheck : status === "error" ? CircleAlert : Clock3;

  if (!job) {
    return (
      <section className="border-b border-line p-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-forest-strong">Run status</h2>
            <p className="mt-1 text-xs text-muted">{selectedTimescale?.label ?? "Selected"} setup · no result yet</p>
          </div>
          <span className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-md border px-2 py-1 text-xs font-semibold ${statusClass(status)}`} role="status" aria-live="polite">
            <StatusIcon aria-hidden="true" size={13} strokeWidth={2} />
            {STATUS_LABEL[status]}
          </span>
        </div>
        {error ? <p className="mt-2 text-xs font-medium text-danger" role="alert">{error}</p> : null}
      </section>
    );
  }

  return (
    <section className="border-b border-line p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-forest-strong">Run status</h2>
        <span className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-md border px-2 py-1 text-xs font-semibold ${statusClass(status)}`} role="status" aria-live="polite">
          <StatusIcon aria-hidden="true" size={13} strokeWidth={2} />
          {STATUS_LABEL[status]}
        </span>
      </div>

      <dl className="mt-4 divide-y divide-line border-y border-line text-xs">
        <div className="grid grid-cols-[0.9fr_1.1fr] gap-3 py-2.5">
          <dt className="text-muted">Forecast horizon</dt>
          <dd className="text-right font-semibold text-ink">
            {job?.timescale ?? selectedTimescale?.label ?? "--"} · lead day {job?.horizonDays ?? selectedTimescale?.horizonDays ?? "--"}
          </dd>
        </div>
        <div className="grid grid-cols-[0.9fr_1.1fr] gap-3 py-2.5">
          <dt className="text-muted">Target date</dt>
          <dd className="text-right font-semibold text-ink">{job?.targetDay ?? "No result yet"}</dd>
        </div>
        <div className="grid grid-cols-[0.9fr_1.1fr] gap-3 py-2.5">
          <dt className="text-muted">Grid cells</dt>
          <dd className="text-right font-semibold text-ink">{featureCount === undefined ? "No result yet" : featureCount.toLocaleString()}</dd>
        </div>
      </dl>

      {job?.jobId ? <p className="mt-3 break-all text-[11px] text-muted">Job {job.jobId}</p> : null}
      {isPolling ? <p className="mt-2 text-[11px] text-muted">Waiting for GPU inference.</p> : null}
      {job?.message ? <p className="mt-2 text-xs text-muted">{job.message}</p> : null}
      {job?.error || error ? <p className="mt-2 text-xs font-medium text-danger" role="alert">{job?.error ?? error}</p> : null}
    </section>
  );
}
