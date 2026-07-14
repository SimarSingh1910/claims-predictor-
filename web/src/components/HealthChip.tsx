// A compact chip in the top bar showing backend liveness + which registry slots
// are live vs awaiting a model. This is the "health status chip showing which
// slots are live" required by the scaffold step.
import { CheckCircle2, XCircle, Loader2 } from "./icons";
import type { HealthResponse, ApiError } from "../types";
import { SlotStatusDot } from "./Badges";

interface Props {
  health: HealthResponse | null;
  error: ApiError | null;
  loading: boolean;
}

export function HealthChip({ health, error, loading }: Props) {
  if (loading) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md bg-surface-raised px-2.5 py-1 text-xs text-slate-400 ring-1 ring-surface-border">
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> connecting…
      </span>
    );
  }
  if (error || !health) {
    return (
      <span
        className="inline-flex items-center gap-1.5 rounded-md bg-red-500/10 px-2.5 py-1 text-xs text-red-300 ring-1 ring-red-500/30"
        title={error?.message ?? "backend unreachable"}
      >
        <XCircle className="h-3.5 w-3.5" /> backend offline
      </span>
    );
  }

  const live = health.registry.filter((r) => r.available).length;
  const total = health.registry.length;
  return (
    <span className="inline-flex items-center gap-2 rounded-md bg-surface-raised px-2.5 py-1 text-xs text-slate-300 ring-1 ring-surface-border">
      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
      <span className="font-medium">
        {live}/{total} slots live
      </span>
      <span className="flex items-center gap-1.5 border-l border-surface-border pl-2">
        {health.registry.map((r) => (
          <span
            key={r.slot}
            className="flex items-center gap-1"
            title={r.available ? `${r.slot}: live` : `${r.slot}: ${r.reason}`}
          >
            <SlotStatusDot available={r.available} />
            <span className="uppercase tracking-wide text-[10px] text-slate-400">
              {r.slot}
            </span>
          </span>
        ))}
      </span>
    </span>
  );
}
