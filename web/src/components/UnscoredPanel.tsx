// Unscored panel — collapsible list of members the engine REFUSED to score, with
// reasons. This is framed as a feature: the engine refuses rather than guessing a
// mandatory value, so a refused member is excluded from cohort statistics instead
// of silently corrupting them.
import { useState } from "react";
import type { UnscoredRow } from "../types";
import { AlertTriangle } from "./icons";

export function UnscoredPanel({ unscored }: { unscored: UnscoredRow[] }) {
  const [open, setOpen] = useState(false);
  const n = unscored.length;

  return (
    <div className="rounded-lg border border-surface-border bg-surface-panel">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-3 text-left"
      >
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-400" />
          <span className="text-sm font-semibold text-slate-200">
            {n} refused {n === 1 ? "member" : "members"}
          </span>
          <span className="text-xs text-slate-500">
            excluded from cohort statistics — the engine refuses rather than guesses
          </span>
        </div>
        <span className="text-xs text-slate-400">{open ? "Hide" : "Show"}</span>
      </button>

      {open && (
        <div className="max-h-72 overflow-y-auto border-t border-surface-border">
          {n === 0 ? (
            <div className="px-4 py-6 text-center text-sm text-slate-500">
              Every member had the mandatory fields — none were refused.
            </div>
          ) : (
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-surface-raised text-slate-400">
                <tr>
                  <th className="px-4 py-2 font-medium">Row</th>
                  <th className="px-4 py-2 font-medium">Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border">
                {unscored.map((u) => (
                  <tr key={u.row}>
                    <td className="px-4 py-1.5 font-mono tabular-nums text-slate-400">
                      {u.row}
                    </td>
                    <td className="px-4 py-1.5 text-slate-300">{u.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
