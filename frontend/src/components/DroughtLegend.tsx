import { DROUGHT_CATEGORIES } from "@/lib/types";
import type { ForecastLayer } from "@/lib/types";

interface ForecastLegendProps {
  layer?: ForecastLayer;
}

const CONTINUOUS_LEGENDS = {
  et: {
    title: "Evapotranspiration",
    units: "mm/day",
    values: ["0", "1", "2", "3", "4+"],
    colors: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
  },
  sm: {
    title: "Soil moisture",
    units: "m3/m3",
    values: ["0.05", "0.15", "0.25", "0.35", "0.45+"],
    colors: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
  }
};

const DROUGHT_TEXTURES: Record<string, string | undefined> = {
  None: undefined,
  D0: "repeating-linear-gradient(0deg, transparent 0 3px, rgba(40,40,0,.72) 3px 4px)",
  D1: "repeating-linear-gradient(45deg, transparent 0 4px, rgba(90,55,0,.66) 4px 5px)",
  D2: "repeating-linear-gradient(45deg, transparent 0 4px, rgba(80,45,0,.65) 4px 5px), repeating-linear-gradient(-45deg, transparent 0 4px, rgba(80,45,0,.65) 4px 5px)",
  D3: "radial-gradient(circle at 2px 2px, rgba(255,255,255,.9) 0 1px, transparent 1.2px)",
  D4: "repeating-linear-gradient(-45deg, transparent 0 3px, rgba(255,255,255,.82) 3px 4px)"
};

export default function DroughtLegend({ layer = "drought" }: ForecastLegendProps) {
  if (layer !== "drought") {
    const legend = CONTINUOUS_LEGENDS[layer];
    return (
      <section className="p-4">
        <div className="flex items-baseline justify-between gap-3">
          <h2 className="text-sm font-semibold text-forest-strong">{legend.title}</h2>
          <span className="text-xs text-muted">{legend.units}</span>
        </div>
        <div className="mt-3 grid grid-cols-5 gap-1" aria-label={`${legend.title} color scale`}>
          {legend.values.map((value, index) => (
            <div key={value} className="text-center text-xs text-muted">
              <span
                className="mx-auto block h-3.5 w-full border border-black/20"
                style={{ backgroundColor: legend.colors[index] }}
                aria-hidden="true"
              />
              <span className="mt-1 block">{value}</span>
            </div>
          ))}
        </div>
      </section>
    );
  }

  return (
    <section className="p-4">
      <h2 className="text-sm font-semibold text-forest-strong">ET/SM dryness category</h2>
      <p className="mt-1 text-[11px] leading-4 text-muted">Experimental indicator; not a U.S. Drought Monitor forecast.</p>
      <div className="mt-3 grid gap-y-2">
        {DROUGHT_CATEGORIES.map((item) => (
          <div key={item.code} className="grid min-w-0 grid-cols-[18px_42px_minmax(0,1fr)] items-center gap-2 text-[11px]">
            <span
              className="h-3.5 w-4 border border-black/40"
              style={{
                backgroundColor: item.color,
                backgroundImage: DROUGHT_TEXTURES[item.code],
                backgroundSize: item.code === "D3" ? "4px 4px" : undefined
              }}
              aria-hidden="true"
            />
            <strong className="text-ink">{item.code === "None" ? "None" : item.code}</strong>
            <span className="text-muted">{item.code === "None" ? "No drought / normal" : item.label}</span>
          </div>
        ))}
        <div className="grid min-w-0 grid-cols-[18px_42px_minmax(0,1fr)] items-center gap-2 text-[11px]">
          <span
            className="h-3.5 w-4 border border-black/40 bg-[#d7ddd8]"
            style={{ backgroundImage: "repeating-linear-gradient(45deg, transparent 0 3px, rgba(70,80,72,.55) 3px 4px)" }}
            aria-hidden="true"
          />
          <strong className="text-ink">No data</strong>
          <span className="text-muted">Outside finite forecast support</span>
        </div>
      </div>
      <details className="mt-3 border-t border-line pt-2 text-[11px] text-muted">
        <summary className="cursor-pointer font-medium text-forest">Indicator method</summary>
        <p className="mt-2 leading-4">
          ET and SM are standardized with each checkpoint&apos;s training-target mean and spread, then combined with the fixed Kendall/Clayton transform (tau 0.4). D0-D4 thresholds are dryness percentiles 70, 80, 90, 95, and 98.
        </p>
      </details>
    </section>
  );
}
