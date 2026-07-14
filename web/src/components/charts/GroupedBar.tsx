// Grouped bar: mean p12 by age group, split by gender. Dependency-free SVG.
// low_n cohorts render hatched + dimmed and are labelled, so a 3-person cohort
// never reads as a confident bar.
import type { Cohort } from "../../types";

const AGE_ORDER = ["<30", "30-39", "40-49", "50-59", "60+"];
const GENDERS: { key: string; label: string; color: string }[] = [
  { key: "M", label: "Male", color: "#3b82f6" },
  { key: "F", label: "Female", color: "#f472b6" },
];

const W = 640;
const H = 260;
const PAD = { top: 16, right: 16, bottom: 40, left: 44 };

export function GroupedBar({ cohorts }: { cohorts: Cohort[] }) {
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const byKey = new Map<string, Cohort>();
  for (const c of cohorts) byKey.set(`${c.age_group}|${c.gender}`, c);

  const maxP12 = Math.max(
    0.1,
    ...cohorts.filter((c) => c.mean_p12 != null).map((c) => c.mean_p12 as number)
  );
  const yMax = Math.ceil(maxP12 * 10) / 10; // round to next 10%

  const groupW = plotW / AGE_ORDER.length;
  const barW = Math.min(28, (groupW - 16) / GENDERS.length);

  const y = (v: number) => PAD.top + plotH * (1 - v / yMax);
  const ticks = Array.from({ length: 5 }, (_, i) => (yMax / 4) * i);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
         aria-label="Mean 12-month claim probability by age group and gender">
      {/* y gridlines + labels */}
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)}
                stroke="#232c40" strokeWidth={1} />
          <text x={PAD.left - 6} y={y(t) + 3} textAnchor="end"
                fontSize={10} fill="#64748b">
            {(t * 100).toFixed(0)}%
          </text>
        </g>
      ))}

      {AGE_ORDER.map((ag, gi) => {
        const gx = PAD.left + gi * groupW;
        const clusterW = barW * GENDERS.length + 6;
        const startX = gx + (groupW - clusterW) / 2;
        return (
          <g key={ag}>
            {GENDERS.map((gd, bi) => {
              const c = byKey.get(`${ag}|${gd.key}`);
              const v = c?.mean_p12 ?? null;
              const x = startX + bi * (barW + 6);
              if (v == null)
                return (
                  <text key={gd.key} x={x + barW / 2} y={PAD.top + plotH - 4}
                        textAnchor="middle" fontSize={9} fill="#475569">–</text>
                );
              const barH = plotH - (y(v) - PAD.top);
              const lowN = c?.low_n;
              return (
                <g key={gd.key}>
                  <rect
                    x={x} y={y(v)} width={barW} height={Math.max(0, barH)}
                    rx={2}
                    fill={lowN ? "url(#hatch)" : gd.color}
                    opacity={lowN ? 0.55 : 1}
                    stroke={lowN ? gd.color : "none"}
                    strokeWidth={lowN ? 1 : 0}
                  >
                    <title>
                      {ag} {gd.label}: {(v * 100).toFixed(1)}% (n={c?.n})
                      {lowN ? " — n<10, unreliable" : ""}
                    </title>
                  </rect>
                </g>
              );
            })}
            <text x={gx + groupW / 2} y={H - PAD.bottom + 16} textAnchor="middle"
                  fontSize={10} fill="#94a3b8">{ag}</text>
          </g>
        );
      })}

      {/* hatch pattern for low_n bars */}
      <defs>
        <pattern id="hatch" width="6" height="6" patternTransform="rotate(45)"
                 patternUnits="userSpaceOnUse">
          <rect width="6" height="6" fill="#334155" />
          <line x1="0" y1="0" x2="0" y2="6" stroke="#64748b" strokeWidth="2" />
        </pattern>
      </defs>

      {/* legend */}
      <g>
        {GENDERS.map((gd, i) => (
          <g key={gd.key} transform={`translate(${PAD.left + i * 90}, ${H - 6})`}>
            <rect width={10} height={10} rx={2} y={-9} fill={gd.color} />
            <text x={14} y={0} fontSize={10} fill="#94a3b8">{gd.label}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}
