// Cohorts page — the demo page. Assembles: summary strip, the four group-level
// MetricCards (p12 live, the rest awaiting), the age×gender matrix with a click
// drawer, the charts, and the unscored panel.
import { useState } from "react";
import type { Cohort, ScoreResponse, SlotKey } from "../types";
import { MetricCard } from "../components/MetricCard";
import { CohortMatrix } from "../components/CohortMatrix";
import { CohortDrawer } from "../components/CohortDrawer";
import { UnscoredPanel } from "../components/UnscoredPanel";
import { ChartFrame } from "../components/ChartFrame";
import { GroupedBar } from "../components/charts/GroupedBar";
import { Histogram } from "../components/charts/Histogram";
import { HorizonChart } from "../components/charts/HorizonChart";

const SLOT_ORDER: SlotKey[] = ["p12", "p24", "p36", "expected_cost"];

export function Cohorts({ result }: { result: ScoreResponse }) {
  const [selected, setSelected] = useState<Cohort | null>(null);
  const s = result.summary;
  const slotFor = (slot: SlotKey) => result.slots.find((x) => x.slot === slot);

  const ratio =
    s.p12 != null && s.base_claim_rate > 0 ? s.p12 / s.base_claim_rate : null;

  const costSlot = slotFor("expected_cost");

  return (
    <div className="space-y-6 p-6">
      {/* header */}
      <div>
        <h1 className="text-lg font-semibold text-slate-100">Cohort risk view</h1>
        <p className="mt-1 text-sm text-slate-400">
          Per-member scores aggregated into age × gender cohorts · model{" "}
          <span className="font-mono text-slate-300">{result.model_used}</span>.
        </p>
      </div>

      {/* summary strip */}
      <div className="grid grid-cols-3 gap-3 sm:max-w-md">
        <Stat label="Total members" value={s.total_rows.toLocaleString()} />
        <Stat label="Scored" value={s.scored.toLocaleString()} tone="good" />
        <Stat label="Refused" value={s.unscored.toLocaleString()}
              tone={s.unscored > 0 ? "warn" : "muted"} />
      </div>

      {/* four group-level slots — equal citizens */}
      <div>
        <div className="mb-2 flex items-center gap-3">
          <h2 className="text-sm font-semibold text-slate-200">Group roll-up</h2>
          {ratio != null && (
            <span className="rounded bg-surface-raised px-2 py-0.5 text-xs text-slate-300 ring-1 ring-surface-border">
              group p12 {(s.p12! * 100).toFixed(1)}% ·{" "}
              <span className="font-semibold text-amber-300">
                {ratio.toFixed(1)}× base rate
              </span>{" "}
              <span className="text-slate-500">
                (base {(s.base_claim_rate * 100).toFixed(2)}%)
              </span>
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {SLOT_ORDER.map((slot) => {
            const sm = slotFor(slot);
            const available = sm?.available ?? false;
            return (
              <MetricCard
                key={slot}
                slot={slot}
                label={sm?.label ?? slot}
                value={s[slot] as number | null}
                available={available}
                reason={sm?.reason ?? "model not trained yet"}
                provenance={result.provenance}
                caption={
                  slot === "p12" && available
                    ? `mean across ${s.scored} members`
                    : undefined
                }
              />
            );
          })}
        </div>
      </div>

      {/* cohort matrix */}
      <div>
        <h2 className="mb-2 text-sm font-semibold text-slate-200">
          Cohort matrix
          <span className="ml-2 font-normal text-slate-500">
            mean p12 · click a cell for detail
          </span>
        </h2>
        <CohortMatrix cohorts={result.cohorts} onSelect={setSelected} />
      </div>

      {/* charts */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ChartFrame title="Mean p12 by age group, split by gender" available height={260}>
          <GroupedBar cohorts={result.cohorts} />
        </ChartFrame>

        <ChartFrame title="p12 distribution across scored members" available height={240}>
          <Histogram buckets={result.risk_distribution} />
        </ChartFrame>

        <div className="rounded-lg border border-surface-border bg-surface-panel p-4">
          <div className="mb-3 text-[11px] font-medium uppercase tracking-wide text-slate-400">
            Probability horizon · p12 / p24 / p36 by age group
          </div>
          <HorizonChart
            cohorts={result.cohorts}
            series={[
              { slot: "p12", label: "12-month", available: slotFor("p12")?.available ?? false },
              { slot: "p24", label: "24-month", available: slotFor("p24")?.available ?? false },
              { slot: "p36", label: "36-month", available: slotFor("p36")?.available ?? false },
            ]}
          />
        </div>

        <ChartFrame
          title="Expected cost by cohort"
          available={costSlot?.available ?? false}
          reason="Awaiting severity model (Phase 4–5)"
          height={240}
        />
      </div>

      {/* unscored */}
      <UnscoredPanel unscored={result.unscored} />

      <CohortDrawer
        cohort={selected}
        slots={result.slots}
        provenance={result.provenance}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}

function Stat({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "good" | "warn" | "muted";
}) {
  const color =
    tone === "good"
      ? "text-emerald-300"
      : tone === "warn"
      ? "text-amber-300"
      : tone === "muted"
      ? "text-slate-400"
      : "text-slate-100";
  return (
    <div className="rounded-lg border border-surface-border bg-surface-panel p-3">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div className={"mt-1 text-2xl font-semibold tabular-nums " + color}>
        {value}
      </div>
    </div>
  );
}
