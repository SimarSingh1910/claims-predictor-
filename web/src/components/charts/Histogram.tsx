// Histogram of p12 across scored members. Reads the backend's risk_distribution
// (10 fixed buckets 0-10% .. 90-100%). Dependency-free SVG.
import type { RiskBucket } from "../../types";
import { riskColor } from "../../lib/scale";

const W = 640;
const H = 240;
const PAD = { top: 16, right: 16, bottom: 44, left: 40 };

export function Histogram({ buckets }: { buckets: RiskBucket[] }) {
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const maxCount = Math.max(1, ...buckets.map((b) => b.count));
  const barW = plotW / buckets.length;

  const y = (v: number) => PAD.top + plotH * (1 - v / maxCount);
  const ticks = 4;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
         aria-label="Distribution of 12-month claim probability across scored members">
      {Array.from({ length: ticks + 1 }, (_, i) => {
        const v = (maxCount / ticks) * i;
        return (
          <g key={i}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(v)} y2={y(v)}
                  stroke="#232c40" strokeWidth={1} />
            <text x={PAD.left - 6} y={y(v) + 3} textAnchor="end"
                  fontSize={10} fill="#64748b">{Math.round(v)}</text>
          </g>
        );
      })}

      {buckets.map((b, i) => {
        const x = PAD.left + i * barW;
        const barH = plotH - (y(b.count) - PAD.top);
        // Colour each bucket by its risk position along the ramp.
        const t = (i + 0.5) / buckets.length;
        return (
          <g key={b.bucket}>
            <rect x={x + 3} y={y(b.count)} width={barW - 6}
                  height={Math.max(0, barH)} rx={2} fill={riskColor(t)}>
              <title>{b.bucket}: {b.count} members</title>
            </rect>
            {i % 2 === 0 && (
              <text x={x + barW / 2} y={H - PAD.bottom + 16} textAnchor="middle"
                    fontSize={9} fill="#94a3b8">{b.bucket}</text>
            )}
          </g>
        );
      })}

      <text x={PAD.left} y={H - 6} fontSize={10} fill="#64748b">
        p12 bucket · bar height = member count
      </text>
    </svg>
  );
}
