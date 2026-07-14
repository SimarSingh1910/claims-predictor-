// Persistent honesty banner. The caveat text comes straight from /api/meta so the
// synthetic-target disclaimer is single-sourced from the backend, never hardcoded
// in the UI.
import { AlertTriangle } from "./icons";

export function HonestyBanner({ caveat }: { caveat: string | null }) {
  if (!caveat) return null;
  return (
    <div className="flex items-start gap-2 border-b border-amber-500/20 bg-amber-500/[0.07] px-5 py-2 text-[12px] text-amber-200/90">
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" />
      <span>{caveat}</span>
    </div>
  );
}
