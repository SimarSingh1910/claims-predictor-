// Horizon chart: p12 / p24 / p36 per age group, stacked as three strips.
// Today only p12 has a trained model, so its strip draws a real line; the p24 and
// p36 strips render an "Awaiting model" overlay. We NEVER connect a line through
// nulls or fabricate the untrained series — each strip is independent.
import type { Cohort, SlotKey } from "../../types";

const AGE_ORDER = ["<30", "30-39", "40-49", "50-59", "60+"];

interface SeriesDef {
  slot: SlotKey;
  label: string;
  available: boolean;
}

const STRIP_H = 64;
const W = 640;
const PAD = { left: 64, right: 16 };

// Weighted mean p12 per age group (weight by scored n across genders).
function ageGroupP12(cohorts: Cohort[]): Map<string, number | null> {
  const acc = new Map<string, { sum: number; n: number }>();
  for (const c of cohorts) {
    if (c.mean_p12 == null) continue;
    const a = acc.get(c.age_group) ?? { sum: 0, n: 0 };
    a.sum += c.mean_p12 * c.n;
    a.n += c.n;
    acc.set(c.age_group, a);
  }
  const out = new Map<string, number | null>();
  for (const ag of AGE_ORDER) {
    const a = acc.get(ag);
    out.set(ag, a && a.n > 0 ? a.sum / a.n : null);
  }
  return out;
}

export function HorizonChart({
  cohorts,
  series,
}: {
  cohorts: Cohort[];
  series: SeriesDef[];
}) {
  const p12ByAge = ageGroupP12(cohorts);
  const vals = AGE_ORDER.map((a) => p12ByAge.get(a)).filter((v): v is number => v != null);
  const maxV = Math.max(0.1, ...vals);
  const plotW = W - PAD.left - PAD.right;
  const stepX = plotW / (AGE_ORDER.length - 1);
  const xAt = (i: number) => PAD.left + i * stepX;

  return (
    <div className="space-y-2">
      {series.map((s) => (
        <div key={s.slot} className="flex items-center gap-3">
          <div className="w-14 shrink-0 text-right">
            <div className="font-mono text-[11px] text-slate-300">{s.slot}</div>
          </div>
          <div className="min-w-0 flex-1">
            {s.available ? (
              <svg viewBox={`0 0 ${W} ${STRIP_H}`} className="w-full"
                   role="img" aria-label={`${s.label} by age group`}>
                <P12Strip xAt={xAt} p12ByAge={p12ByAge} maxV={maxV} />
              </svg>
            ) : (
              <div
                style={{ height: STRIP_H }}
                className="relative flex items-center justify-center overflow-hidden rounded-md border border-dashed border-surface-border/70 bg-surface-base/40"
              >
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-0 opacity-[0.12]"
                  style={{
                    backgroundImage:
                      "repeating-linear-gradient(90deg, transparent 0, transparent 63px, rgba(148,163,184,0.5) 63px, rgba(148,163,184,0.5) 64px)",
                  }}
                />
                <span className="z-10 text-[11px] italic text-slate-500">
                  Awaiting model
                </span>
              </div>
            )}
          </div>
        </div>
      ))}

      {/* shared x-axis labels */}
      <div className="flex items-center gap-3">
        <div className="w-14 shrink-0" />
        <div className="relative min-w-0 flex-1">
          <svg viewBox={`0 0 ${W} 16`} className="w-full">
            {AGE_ORDER.map((ag, i) => (
              <text key={ag} x={xAt(i)} y={11} textAnchor="middle"
                    fontSize={10} fill="#94a3b8">{ag}</text>
            ))}
          </svg>
        </div>
      </div>
    </div>
  );
}

function P12Strip({
  xAt,
  p12ByAge,
  maxV,
}: {
  xAt: (i: number) => number;
  p12ByAge: Map<string, number | null>;
  maxV: number;
}) {
  const yAt = (v: number) => STRIP_H - 8 - (STRIP_H - 16) * (v / maxV);
  // Build path only across CONSECUTIVE non-null points; break on nulls.
  const pts = AGE_ORDER.map((ag, i) => {
    const v = p12ByAge.get(ag);
    return v == null ? null : { x: xAt(i), y: yAt(v), v };
  });

  const segments: { x: number; y: number }[][] = [];
  let cur: { x: number; y: number }[] = [];
  for (const p of pts) {
    if (p == null) {
      if (cur.length) segments.push(cur);
      cur = [];
    } else cur.push(p);
  }
  if (cur.length) segments.push(cur);

  return (
    <>
      {segments.map((seg, si) => (
        <polyline
          key={si}
          fill="none"
          stroke="#3b82f6"
          strokeWidth={2}
          points={seg.map((p) => `${p.x},${p.y}`).join(" ")}
        />
      ))}
      {pts.map((p, i) =>
        p == null ? null : (
          <circle key={i} cx={p.x} cy={p.y} r={3} fill="#3b82f6">
            <title>{AGE_ORDER[i]}: {(p.v * 100).toFixed(1)}%</title>
          </circle>
        )
      )}
    </>
  );
}
