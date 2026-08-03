import Link from "next/link";
import { Leaf, Map } from "lucide-react";

interface SiteHeaderProps {
  tone?: "hero" | "light";
}

export default function SiteHeader({ tone = "light" }: SiteHeaderProps) {
  const hero = tone === "hero";
  return (
    <header className={`flex min-h-16 items-center justify-between border-b ${hero ? "border-white/25" : "border-line"}`}>
      <Link
        href="/"
        className={`inline-flex items-center gap-2 text-sm font-semibold ${hero ? "text-white" : "text-forest-strong"}`}
      >
        <span className={`grid h-8 w-8 place-items-center rounded-md ${hero ? "bg-white text-forest-strong" : "bg-forest text-white"}`}>
          <Leaf aria-hidden="true" size={17} strokeWidth={2} />
        </span>
        SEED
      </Link>
      <nav aria-label="Primary" className="flex items-center gap-2">
        {!hero ? (
          <Link href="/" className="px-3 py-2 text-sm font-medium text-muted hover:text-forest">
            Overview
          </Link>
        ) : null}
        <Link
          href="/dashboard"
          className={`inline-flex min-h-10 items-center gap-2 rounded-md px-4 py-2 text-sm font-semibold ${
            hero
              ? "border border-white/65 bg-forest/90 text-white hover:bg-forest"
              : "bg-forest text-white hover:bg-forest-strong"
          }`}
        >
          <Map aria-hidden="true" size={16} strokeWidth={2} />
          Dashboard
        </Link>
      </nav>
    </header>
  );
}
