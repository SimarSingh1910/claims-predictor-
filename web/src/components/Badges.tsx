// Provenance + slot-status badges. Provenance is a first-class honesty signal:
// a SYNTHETIC badge travels with every metric so a synthetic number is never
// mistaken for a real-world claim rate.
import type { Provenance } from "../types";

export function ProvenanceBadge({ provenance }: { provenance: Provenance }) {
  const synthetic = provenance === "synthetic";
  return (
    <span
      className={
        "inline-flex items-center gap-1 rounded px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide " +
        (synthetic
          ? "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30"
          : "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30")
      }
      title={
        synthetic
          ? "Target is synthetic — metrics validate the pipeline, not real claim rates."
          : "Backed by real AHC + claims data."
      }
    >
      {synthetic ? "Synthetic" : "Real"}
    </span>
  );
}

export function SlotStatusDot({ available }: { available: boolean }) {
  return (
    <span
      className={
        "inline-block h-2 w-2 rounded-full " +
        (available ? "bg-emerald-400" : "bg-slate-500")
      }
      aria-label={available ? "live" : "awaiting model"}
    />
  );
}
