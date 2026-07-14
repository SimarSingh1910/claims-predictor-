// MemberDrawer — per-member detail for a clicked table row: the four MetricCards,
// the two confidence meters, the missing-important list (grouped), and the full
// AHC values. Reuses MetricCard so untrained slots behave identically here.
import { useEffect } from "react";
import type {
  MemberRow,
  SlotDescriptor,
  Provenance,
  SlotKey,
} from "../types";
import { MetricCard } from "./MetricCard";
import { Meter } from "./Meter";
import { X } from "./icons";

const SLOT_ORDER: SlotKey[] = ["p12", "p24", "p36", "expected_cost"];

export function MemberDrawer({
  member,
  slots,
  provenance,
  onClose,
}: {
  member: MemberRow | null;
  slots: SlotDescriptor[];
  provenance: Provenance;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!member) return null;
  const slotFor = (s: SlotKey) => slots.find((x) => x.slot === s);

  // Group missing-important by clinical system for readability.
  const missingByGroup = new Map<string, string[]>();
  for (const m of member.missing_important) {
    const arr = missingByGroup.get(m.group) ?? [];
    arr.push(m.feature);
    missingByGroup.set(m.group, arr);
  }

  const entries = Object.entries(member.values);

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <aside className="relative z-50 flex h-full w-[480px] max-w-[92vw] flex-col overflow-y-auto border-l border-surface-border bg-surface-panel shadow-2xl">
        <div className="sticky top-0 flex items-start justify-between border-b border-surface-border bg-surface-panel p-4">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-slate-500">
              Member
            </div>
            <h2 className="font-mono text-lg font-semibold text-slate-100">
              {member.id}
            </h2>
            <div className="mt-0.5 text-xs text-slate-400">
              {member.cohort} · age {member.age ?? "—"}
              {member.high_risk && (
                <span className="ml-1.5 rounded bg-red-500/15 px-1.5 py-0.5 text-[10px] text-red-300">
                  high-risk
                </span>
              )}
              {!member.scored && (
                <span className="ml-1.5 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-300">
                  refused
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

        {!member.scored && (
          <div className="border-b border-surface-border bg-amber-500/[0.06] px-4 py-2 text-xs text-amber-200/90">
            {member.reason} — excluded from cohort statistics.
          </div>
        )}

        {/* four slots */}
        <div className="grid grid-cols-2 gap-3 p-4">
          {SLOT_ORDER.map((slot) => {
            const sm = slotFor(slot);
            return (
              <MetricCard
                key={slot}
                slot={slot}
                label={sm?.label ?? slot}
                value={member[slot] as number | null}
                available={sm?.available ?? false}
                reason={sm?.reason ?? "model not trained yet"}
                provenance={provenance}
              />
            );
          })}
        </div>

        {/* confidence meters */}
        <div className="space-y-3 border-t border-surface-border p-4">
          <Meter
            label="Panel completeness"
            pct={member.panel_completeness_pct}
            hint={`${member.n_important_present ?? 0}/${member.n_important_total ?? 0} important clinical params present`}
          />
          <Meter
            label="Model confidence"
            pct={member.model_confidence_pct}
            hint="coefficient-weighted coverage of present features"
          />
        </div>

        {/* missing important */}
        {member.missing_important.length > 0 && (
          <div className="border-t border-surface-border p-4">
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Missing important params ({member.missing_important.length})
            </h3>
            <div className="space-y-2">
              {[...missingByGroup.entries()].map(([group, feats]) => (
                <div key={group}>
                  <div className="text-[11px] text-slate-500">{group}</div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {feats.map((f) => (
                      <span
                        key={f}
                        className="rounded bg-surface-raised px-1.5 py-0.5 font-mono text-[10px] text-slate-400"
                      >
                        {f}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* full AHC values */}
        <div className="border-t border-surface-border p-4">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
            AHC values ({entries.length})
          </h3>
          <div className="overflow-hidden rounded-md border border-surface-border">
            <table className="w-full text-left text-[11px]">
              <tbody className="divide-y divide-surface-border">
                {entries.map(([k, v]) => (
                  <tr key={k}>
                    <td className="w-1/2 px-2.5 py-1 font-mono text-slate-500">
                      {k}
                    </td>
                    <td className="px-2.5 py-1 tabular-nums text-slate-300">
                      {v == null || v === "" ? (
                        <span className="text-slate-600">—</span>
                      ) : (
                        String(v)
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </aside>
    </div>
  );
}
