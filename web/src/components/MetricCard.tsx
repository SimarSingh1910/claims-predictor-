// MetricCard — the ONE component that renders every metric, everywhere.
//
// This is the skeleton mechanism that makes future models zero-frontend-change:
// a slot is either available (render the real, unit-formatted number + provenance
// badge) or not (render an identical-sized "Awaiting model" skeleton). Dropping a
// trained joblib on disk flips `available` to true on the next /api/score — no
// React edit required.
//
// HARD RULE: in the skeleton state we NEVER render 0, "—", "N/A", or any derived
// value. The number's place is a shimmer bar. An empty slot stays visibly empty.
import { useState } from "react";
import type { SlotKey, Provenance } from "../types";
import { formatSlotValue } from "../lib/format";
import { ProvenanceBadge } from "./Badges";

interface MetricCardProps {
  slot: SlotKey;
  label: string;
  value: number | null;
  available: boolean;
  reason?: string;
  provenance?: Provenance;
  /** expected on-disk artifact path — surfaced in the tooltip for empty slots */
  artifactPath?: string;
  /** optional sub-line under the number, e.g. "mean across 200 members" */
  caption?: string;
}

export function MetricCard({
  slot,
  label,
  value,
  available,
  reason,
  provenance = "synthetic",
  artifactPath,
  caption,
}: MetricCardProps) {
  const live = available && value != null;

  return (
    <div
      className={
        "flex min-h-[104px] flex-col justify-between rounded-lg border p-4 " +
        (live
          ? "border-surface-border bg-surface-panel"
          : "border-dashed border-surface-border/70 bg-surface-panel/40")
      }
      data-slot={slot}
      data-state={live ? "live" : "awaiting"}
    >
      {/* header row: label + (live only) provenance badge */}
      <div className="flex items-start justify-between gap-2">
        <span className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
          {label}
        </span>
        {live && <ProvenanceBadge provenance={provenance} />}
      </div>

      {live ? (
        <div>
          <div className="text-2xl font-semibold tabular-nums text-slate-100">
            {formatSlotValue(slot, value)}
          </div>
          {caption && (
            <div className="mt-0.5 text-[11px] text-slate-500">{caption}</div>
          )}
        </div>
      ) : (
        // Two distinct empty states, same footprint / same shimmer, never a number:
        //  * !available          -> untrained slot: "Awaiting model"
        //  * available & value==null -> trained slot, empty/all-refused cohort:
        //                            "No scorable members"
        <EmptyState
          empty={available ? "no-members" : "awaiting"}
          reason={reason}
          artifactPath={artifactPath}
        />
      )}
    </div>
  );
}

function EmptyState({
  empty,
  reason,
  artifactPath,
}: {
  empty: "awaiting" | "no-members";
  reason?: string;
  artifactPath?: string;
}) {
  const [hover, setHover] = useState(false);
  const awaiting = empty === "awaiting";
  const line = awaiting ? "Awaiting model" : "No scorable members";
  // Untrained slot: point at the expected artifact. Empty cohort: the model is
  // present, so only the (backend) reason is worth surfacing.
  const tip = awaiting
    ? [reason, artifactPath && `expects: ${artifactPath}`].filter(Boolean).join("\n")
    : reason;

  return (
    <div
      className="relative"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {/* shimmer bar where the number would go — NOT a zero, NOT a dash */}
      <div className="shimmer h-6 w-24 rounded" aria-hidden="true" />
      <div className="mt-1.5 text-[11px] italic text-slate-500">{line}</div>

      {tip && hover && (
        <div className="absolute left-0 top-full z-10 mt-1 w-max max-w-[240px] whitespace-pre-line rounded-md border border-surface-border bg-surface-raised px-2.5 py-1.5 text-[11px] text-slate-300 shadow-lg">
          {tip}
        </div>
      )}
    </div>
  );
}
