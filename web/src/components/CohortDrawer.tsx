// Slide-in drawer for a clicked cohort: its four MetricCards (p12 live, the rest
// awaiting) plus confidence and spread. Reuses MetricCard so the untrained slots
// behave identically to everywhere else.
import { useEffect } from "react";
import type { Cohort, SlotDescriptor, Provenance, SlotKey } from "../types";
import { MetricCard } from "./MetricCard";
import { X } from "./icons";

const SLOT_ORDER: SlotKey[] = ["p12", "p24", "p36", "expected_cost"];

export function CohortDrawer({
  cohort,
  slots,
  provenance,
  onClose,
}: {
  cohort: Cohort | null;
  slots: SlotDescriptor[];
  provenance: Provenance;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!cohort) return null;
  const slotFor = (s: SlotKey) => slots.find((x) => x.slot === s);

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <aside className="relative z-50 flex h-full w-[420px] max-w-[90vw] flex-col overflow-y-auto border-l border-surface-border bg-surface-panel shadow-2xl">
        <div className="flex items-start justify-between border-b border-surface-border p-4">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-slate-500">
              Cohort
            </div>
            <h2 className="text-lg font-semibold text-slate-100">
              {cohort.age_group} · {cohort.gender === "M" ? "Male" : cohort.gender === "F" ? "Female" : cohort.gender}
            </h2>
            <div className="mt-0.5 text-xs text-slate-400">
              {cohort.n.toLocaleString()} members
              {cohort.low_n && (
                <span className="ml-1.5 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-300">
                  n&lt;10 — unreliable
                </span>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-slate-400 hover:bg-surface-raised hover:text-slate-200"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3 p-4">
          {SLOT_ORDER.map((slot) => {
            const sm = slotFor(slot);
            const available = sm?.available ?? false;
            const value = cohort[slot] as number | null;
            return (
              <MetricCard
                key={slot}
                slot={slot}
                label={sm?.label ?? slot}
                value={value}
                available={available}
                reason={sm?.reason ?? "model not trained yet"}
                provenance={provenance}
              />
            );
          })}
        </div>

        <div className="space-y-2 border-t border-surface-border p-4 text-sm">
          <Row label="Median p12"
               value={cohort.median_p12 != null ? `${(cohort.median_p12 * 100).toFixed(1)}%` : "—"} />
          <Row label="Spread (std)"
               value={cohort.p12_std != null ? `±${(cohort.p12_std * 100).toFixed(1)}%` : "—"} />
          <Row label="High-risk (p12>30%)"
               value={`${cohort.high_risk_count} · ${cohort.pct_high_risk?.toFixed(1) ?? "0"}%`} />
          <Row label="Expected claimants"
               value={cohort.expected_claimants != null ? String(cohort.expected_claimants) : "—"} />
          <Row label="Mean confidence"
               value={cohort.mean_confidence != null ? `${cohort.mean_confidence.toFixed(1)}% · ${cohort.confidence_band ?? ""}` : "—"} />
        </div>
      </aside>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-400">{label}</span>
      <span className="tabular-nums text-slate-200">{value}</span>
    </div>
  );
}
