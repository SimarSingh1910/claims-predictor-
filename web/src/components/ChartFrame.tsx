// ChartFrame — the charting counterpart to MetricCard's skeleton contract.
//
// A chart whose series is unavailable must render an EMPTY plot frame with an
// "Awaiting model" overlay — never a flat zero line, never invented points. Wrap
// every chart in this: when `available` is false it shows the dashed frame +
// overlay; when true it renders its children (the real Recharts plot).
import type { ReactNode } from "react";

interface ChartFrameProps {
  title: string;
  available: boolean;
  reason?: string;
  /** approximate height so the empty frame matches the real chart's footprint */
  height?: number;
  children?: ReactNode;
}

export function ChartFrame({
  title,
  available,
  reason,
  height = 220,
  children,
}: ChartFrameProps) {
  return (
    <div className="rounded-lg border border-surface-border bg-surface-panel p-4">
      <div className="mb-3 text-[11px] font-medium uppercase tracking-wide text-slate-400">
        {title}
      </div>
      {available ? (
        <div style={{ height }}>{children}</div>
      ) : (
        <div
          style={{ height }}
          className="relative flex items-center justify-center rounded-md border border-dashed border-surface-border/70 bg-surface-base/40"
        >
          {/* faint gridlines to read as an empty plot, not a broken card */}
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 opacity-[0.15]"
            style={{
              backgroundImage:
                "repeating-linear-gradient(to top, transparent 0, transparent 34px, rgba(148,163,184,0.4) 34px, rgba(148,163,184,0.4) 35px)",
            }}
          />
          <div className="z-10 text-center">
            <div className="text-sm italic text-slate-500">Awaiting model</div>
            {reason && (
              <div className="mt-0.5 text-[11px] text-slate-600">{reason}</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
