// Meter — a labelled horizontal bar for the two confidence numbers a member
// carries: panel completeness (clinical coverage) and model confidence
// (importance-weighted coverage, per the active model's own basis). Colour
// bands: >=80 green, >=50 amber, else red.
export function Meter({
  label,
  pct,
  hint,
}: {
  label: string;
  pct: number | null;
  hint?: string;
}) {
  const v = pct ?? 0;
  const color =
    v >= 80 ? "#10b981" : v >= 50 ? "#f59e0b" : "#ef4444";
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-xs font-medium tabular-nums text-slate-200">
          {pct == null ? "—" : `${pct.toFixed(1)}%`}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-surface-raised">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.max(0, Math.min(100, v))}%`, background: color }}
        />
      </div>
      {hint && <div className="mt-1 text-[10px] text-slate-500">{hint}</div>}
    </div>
  );
}
