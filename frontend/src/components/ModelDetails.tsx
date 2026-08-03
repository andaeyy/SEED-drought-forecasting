import { Braces, Database } from "lucide-react";
import type { ModelBundleMetadata } from "@/lib/types";

interface ModelDetailsProps {
  model?: ModelBundleMetadata;
  error?: string | null;
  resultIsDisplayed?: boolean;
}

function modelLabel(architecture: string): string {
  const labels: Record<string, string> = {
    autoregressive: "Autoregressive ConvLSTM",
    encdec: "Encoder-decoder ConvLSTM",
    seq2seq: "Sequence-to-map ConvLSTM"
  };
  return labels[architecture.toLowerCase()] ?? architecture;
}

function registryLabel(version: string): string {
  const match = version.match(/v(\d{4})(\d{2})(\d{2})$/);
  return match ? `${match[1]}.${match[2]}.${match[3]}` : version;
}

export default function ModelDetails({ model, error, resultIsDisplayed = false }: ModelDetailsProps) {
  return (
    <section className="border-b border-line p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-forest-strong">
            {resultIsDisplayed ? "Models used for result" : "Models for selected setup"}
          </h2>
          <p className="mt-1 text-xs text-muted">Selected on 2019 validation · independently tested on 2020</p>
        </div>
      </div>

      {error ? <p className="mt-3 text-xs text-danger" role="alert">{error}</p> : null}
      {!model && !error ? <p className="mt-3 text-xs text-muted">Loading model provenance.</p> : null}

      {model ? (
        <div className="mt-4 space-y-4">
          {[{ label: "ET", item: model.et }, { label: "SM", item: model.sm }].map(({ label, item }) => (
            <div key={label} className="border-b border-line pb-3 last:border-b-0 last:pb-0">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-xs font-bold text-accent-strong">{label}</span>
                <span className="text-xs text-muted" title={`Checkpoint SHA-256 ${item.checkpointSha256}`}>
                  SHA {item.checkpointSha256.slice(0, 8)}
                </span>
              </div>
              <p className="mt-1 text-xs font-semibold text-ink">{modelLabel(item.architecture)}</p>
              <details className="mt-1 text-xs text-muted">
                <summary className="cursor-pointer font-medium text-forest hover:text-accent-strong">Model provenance</summary>
                <p className="mt-2 break-all leading-5">{item.modelId}</p>
              </details>
            </div>
          ))}

          <dl className="grid grid-cols-[18px_minmax(0,1fr)] gap-x-2 gap-y-2 border-t border-line pt-3 text-[11px] text-muted">
            <Database aria-hidden="true" size={14} />
            <div><span className="font-semibold text-ink">{model.inputDays} input days</span> · {model.inputVariables.length} NLDAS variables</div>
            <Braces aria-hidden="true" size={14} />
            <div title={model.version}><span className="font-semibold text-ink">Registry version</span> {registryLabel(model.version)}</div>
          </dl>
        </div>
      ) : null}
    </section>
  );
}
