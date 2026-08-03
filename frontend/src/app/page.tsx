import Image from "next/image";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import SiteHeader from "@/components/SiteHeader";

const HORIZONS = [
  { label: "Weekly", history: "10 days", endpoint: "Day 7" },
  { label: "Monthly", history: "45 days", endpoint: "Day 30" },
  { label: "Seasonal", history: "135 days", endpoint: "Day 90" }
];

export default function LandingPage() {
  return (
    <main className="bg-field text-ink">
      <section className="relative min-h-[84svh] overflow-hidden bg-forest-strong text-white">
        <Image
          src="/images/great-plains-fields-hero.png"
          alt="Aerial view of cultivated fields across the Great Plains"
          fill
          priority
          className="object-cover object-[58%_45%]"
          sizes="100vw"
        />
        <div className="hero-photo-overlay absolute inset-0" aria-hidden="true" />
        <div className="relative mx-auto flex min-h-[84svh] max-w-[1320px] flex-col px-5 sm:px-8 lg:px-12">
          <SiteHeader tone="hero" />
          <div className="flex flex-1 items-end pb-12 pt-24 sm:pb-16 lg:pb-20">
            <div className="max-w-3xl">
              <p className="mb-4 text-xs font-semibold text-[#dcebd0] sm:text-sm">
                NLDAS-based ET/SM drought indicators
              </p>
              <h1 className="max-w-3xl text-4xl font-semibold leading-[1.08] sm:text-5xl lg:text-6xl">
                Great Plains Drought Outlooks
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-7 text-white/85 sm:text-lg">
                Run day-7, day-30, and day-90 evapotranspiration and soil-moisture forecasts from seven NLDAS weather variables.
              </p>
              <div className="mt-8 flex flex-col items-start gap-3 sm:flex-row sm:flex-wrap sm:items-center">
                <Link
                  href="/dashboard"
                  className="inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-md bg-white px-5 py-3 text-sm font-semibold text-forest-strong transition hover:bg-[#eef5e9] focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-forest sm:w-auto"
                >
                  Open forecast dashboard
                  <ArrowRight aria-hidden="true" size={17} strokeWidth={2} />
                </Link>
                <a
                  href="#method"
                  className="inline-flex min-h-11 items-center py-3 text-sm font-semibold text-white underline decoration-white/45 underline-offset-4 hover:decoration-white focus:outline-none focus:ring-2 focus:ring-white sm:px-3"
                >
                  How forecasts are built
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="method" className="border-b border-line bg-white">
        <div className="mx-auto grid max-w-[1320px] gap-10 px-5 py-14 sm:px-8 lg:grid-cols-[0.8fr_1.2fr] lg:px-12 lg:py-20">
          <div>
            <p className="text-sm font-semibold text-accent">Forecast design</p>
            <h2 className="mt-2 text-3xl font-semibold leading-tight text-forest-strong">
              Forecast ET and soil moisture from NLDAS weather history.
            </h2>
            <p className="mt-4 max-w-lg text-base leading-7 text-muted">
              Each horizon pairs separately selected ET and soil-moisture models using the same seven NLDAS forcing variables.
            </p>
          </div>
          <div className="grid border-y border-line sm:grid-cols-3 sm:divide-x sm:divide-line">
            {HORIZONS.map((item) => (
              <div key={item.label} className="border-b border-line px-0 py-5 last:border-b-0 sm:border-b-0 sm:px-6">
                <p className="text-sm font-semibold text-forest">{item.label}</p>
                <p className="mt-3 text-2xl font-semibold text-ink">{item.endpoint}</p>
                <p className="mt-1 text-sm text-muted">from {item.history} of history</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="border-b border-line bg-field">
        <dl className="mx-auto grid max-w-[1320px] divide-y divide-line px-5 sm:grid-cols-3 sm:divide-x sm:divide-y-0 sm:px-8 lg:px-12">
          <div className="py-6 sm:pr-8">
            <dt className="text-sm font-semibold text-forest-strong">Inputs</dt>
            <dd className="mt-1 text-sm leading-6 text-muted">7 NLDAS forcing variables</dd>
          </div>
          <div className="py-6 sm:px-8">
            <dt className="text-sm font-semibold text-forest-strong">Outputs</dt>
            <dd className="mt-1 text-sm leading-6 text-muted">ET, soil moisture, and a derived dryness category</dd>
          </div>
          <div className="py-6 sm:pl-8">
            <dt className="text-sm font-semibold text-forest-strong">Provenance</dt>
            <dd className="mt-1 text-sm leading-6 text-muted">Model ID, architecture, registry version, and evaluation period</dd>
          </div>
        </dl>
      </section>

      <footer className="bg-forest-strong text-white/75">
        <div className="mx-auto flex max-w-[1320px] flex-col gap-2 px-5 py-8 text-sm sm:flex-row sm:items-center sm:justify-between sm:px-8 lg:px-12">
          <span className="font-semibold text-white">SEED Drought Outlook</span>
          <span>ET and soil-moisture forecasts at days 7, 30, and 90</span>
        </div>
      </footer>
    </main>
  );
}
