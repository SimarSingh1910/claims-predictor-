// Cohort matrix: rows = age groups, columns = M/F. Each cell shows mean p12
// (colour-scaled) and n. low_n cells (n<10) are hatched, labelled "n<10", and
// EXCLUDED from the colour scale domain so a tiny cohort can't stretch the ramp.
// Clicking a populated cell opens the drawer for that cohort.
import type { Cohort } from "../types";
import { riskColor, normalize, textOn } from "../lib/scale";

const AGE_ORDER = ["<30", "30-39", "40-49", "50-59", "60+"];
const COLS: { key: string; label: string }[] = [
  { key: "M", label: "Male" },
  { key: "F", label: "Female" },
];

export function CohortMatrix({
  cohorts,
  onSelect,
}: {
  cohorts: Cohort[];
  onSelect: (c: Cohort) => void;
}) {
  const byKey = new Map<string, Cohort>();
  for (const c of cohorts) byKey.set(`${c.age_group}|${c.gender}`, c);

  // Colour domain from reliable cohorts only (n>=10, mean_p12 present).
  const reliable = cohorts.filter((c) => !c.low_n && c.mean_p12 != null);
  const domainVals = reliable.map((c) => c.mean_p12 as number);
  const dMin = domainVals.length ? Math.min(...domainVals) : 0;
  const dMax = domainVals.length ? Math.max(...domainVals) : 1;

  return (
    <div className="overflow-x-auto">
      <table className="border-separate" style={{ borderSpacing: 6 }}>
        <thead>
          <tr>
            <th />
            {COLS.map((c) => (
              <th key={c.key} className="px-2 pb-1 text-xs font-medium text-slate-400">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {AGE_ORDER.map((ag) => (
            <tr key={ag}>
              <td className="pr-2 text-right text-xs font-medium text-slate-400">
                {ag}
              </td>
              {COLS.map((col) => {
                const c = byKey.get(`${ag}|${col.key}`);
                return (
                  <td key={col.key}>
                    <CohortCell cohort={c} dMin={dMin} dMax={dMax} onSelect={onSelect} />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CohortCell({
  cohort,
  dMin,
  dMax,
  onSelect,
}: {
  cohort: Cohort | undefined;
  dMin: number;
  dMax: number;
  onSelect: (c: Cohort) => void;
}) {
  // No members in this age×gender bucket.
  if (!cohort || cohort.mean_p12 == null) {
    return (
      <div className="flex h-[76px] w-[132px] items-center justify-center rounded-md border border-dashed border-surface-border/60 bg-surface-panel/30 text-[11px] text-slate-600">
        no members
      </div>
    );
  }

  const p12 = cohort.mean_p12;

  if (cohort.low_n) {
    return (
      <button
        onClick={() => onSelect(cohort)}
        className="relative h-[76px] w-[132px] overflow-hidden rounded-md border border-surface-border text-left"
        title={`${cohort.age_group} ${cohort.gender}: n=${cohort.n} — n<10, unreliable`}
      >
        <span
          aria-hidden="true"
          className="absolute inset-0 opacity-70"
          style={{
            backgroundImage:
              "repeating-linear-gradient(45deg, #1e2637 0 6px, #131a29 6px 12px)",
          }}
        />
        <span className="relative flex h-full flex-col justify-between p-2">
          <span className="text-sm font-semibold tabular-nums text-slate-300">
            {(p12 * 100).toFixed(1)}%
          </span>
          <span className="text-[10px] leading-tight text-amber-300/80">
            n={cohort.n} · n&lt;10 — unreliable
          </span>
        </span>
      </button>
    );
  }

  const t = normalize(p12, dMin, dMax);
  const bg = riskColor(t);
  const fg = textOn(t);
  return (
    <button
      onClick={() => onSelect(cohort)}
      className="flex h-[76px] w-[132px] flex-col justify-between rounded-md p-2 text-left ring-1 ring-black/20 transition hover:ring-2 hover:ring-white/30"
      style={{ background: bg, color: fg }}
      title={`${cohort.age_group} ${cohort.gender}: ${(p12 * 100).toFixed(1)}% mean p12, n=${cohort.n}`}
    >
      <span className="text-lg font-semibold tabular-nums">
        {(p12 * 100).toFixed(1)}%
      </span>
      <span className="text-[10px] opacity-80">
        n={cohort.n} · {cohort.high_risk_count} high-risk
      </span>
    </button>
  );
}
