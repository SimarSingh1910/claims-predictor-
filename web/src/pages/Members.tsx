// Members page — a sortable/filterable table of every scored member, a per-member
// detail drawer, and the single-member manual scorer (what-if). The table reads
// the members[] the score response now carries; the manual scorer works even with
// no upload.
import { useMemo, useState } from "react";
import type {
  ScoreResponse,
  MetaResponse,
  MemberRow,
  P12Model,
  ConfidenceBand,
} from "../types";
import { MemberDrawer } from "../components/MemberDrawer";
import { ManualScorer } from "../components/ManualScorer";
import { PagePlaceholder } from "./Placeholder";

type Tab = "table" | "manual";
type SortKey = "id" | "age" | "gender" | "cohort" | "p12" | "confidence_band";

export function Members({
  result,
  meta,
  model,
}: {
  result: ScoreResponse | null;
  meta: MetaResponse | null;
  model: P12Model;
}) {
  const [tab, setTab] = useState<Tab>("table");

  return (
    <div className="space-y-4 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">Members</h1>
          <p className="mt-1 text-sm text-slate-400">
            Per-member scores, refusals, and a single-member what-if scorer.
          </p>
        </div>
        <div className="flex rounded-md border border-surface-border bg-surface-panel p-0.5 text-xs">
          <TabBtn active={tab === "table"} onClick={() => setTab("table")}>
            Uploaded members
          </TabBtn>
          <TabBtn active={tab === "manual"} onClick={() => setTab("manual")}>
            Manual entry
          </TabBtn>
        </div>
      </div>

      {tab === "table" ? (
        result ? (
          <MembersTable result={result} />
        ) : (
          <PagePlaceholder
            title="No dataset yet"
            subtitle="Upload a CSV to see per-member scores — or use Manual entry to score one member."
          />
        )
      ) : (
        <ManualScorer meta={meta} model={model} />
      )}
    </div>
  );
}

function MembersTable({ result }: { result: ScoreResponse }) {
  const members = result.members;
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({
    key: "p12",
    dir: -1,
  });
  const [cohort, setCohort] = useState("all");
  const [band, setBand] = useState<"all" | ConfidenceBand>("all");
  const [risk, setRisk] = useState<"all" | "high" | "normal" | "refused">("all");
  const [q, setQ] = useState("");
  const [selected, setSelected] = useState<MemberRow | null>(null);

  const cohorts = useMemo(
    () => Array.from(new Set(members.map((m) => m.cohort))).sort(),
    [members]
  );

  const filtered = useMemo(() => {
    let rows = members;
    if (cohort !== "all") rows = rows.filter((m) => m.cohort === cohort);
    if (band !== "all") rows = rows.filter((m) => m.confidence_band === band);
    if (risk === "high") rows = rows.filter((m) => m.high_risk);
    else if (risk === "normal") rows = rows.filter((m) => m.scored && !m.high_risk);
    else if (risk === "refused") rows = rows.filter((m) => !m.scored);
    if (q.trim()) {
      const s = q.trim().toLowerCase();
      rows = rows.filter(
        (m) =>
          m.id.toLowerCase().includes(s) ||
          m.cohort.toLowerCase().includes(s)
      );
    }
    const { key, dir } = sort;
    return [...rows].sort((a, b) => {
      const av = a[key];
      const bv = b[key];
      if (av == null && bv == null) return 0;
      if (av == null) return 1; // nulls last
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
      return String(av).localeCompare(String(bv)) * dir;
    });
  }, [members, cohort, band, risk, q, sort]);

  const toggleSort = (key: SortKey) =>
    setSort((s) => (s.key === key ? { key, dir: (s.dir * -1) as 1 | -1 } : { key, dir: 1 }));

  const exportCsv = () => {
    const cols = ["id", "age", "gender", "cohort", "p12", "confidence_band", "high_risk", "scored"];
    const lines = [cols.join(",")];
    for (const m of filtered) {
      lines.push(
        [
          m.id,
          m.age ?? "",
          m.gender,
          m.cohort,
          m.p12 == null ? "" : m.p12.toFixed(6),
          m.confidence_band ?? "",
          m.high_risk,
          m.scored,
        ].join(",")
      );
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "members_export.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-3">
      {/* controls */}
      <div className="flex flex-wrap items-center gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search id or cohort…"
          className="w-48 rounded-md border border-surface-border bg-surface-raised px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-brand"
        />
        <Select value={cohort} onChange={setCohort} label="Cohort">
          <option value="all">All cohorts</option>
          {cohorts.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </Select>
        <Select value={risk} onChange={(v) => setRisk(v as typeof risk)} label="Risk">
          <option value="all">All risk</option>
          <option value="high">High-risk (p12&gt;30%)</option>
          <option value="normal">Normal</option>
          <option value="refused">Refused</option>
        </Select>
        <Select value={band} onChange={(v) => setBand(v as typeof band)} label="Confidence">
          <option value="all">All confidence</option>
          <option value="HIGH">HIGH</option>
          <option value="MEDIUM">MEDIUM</option>
          <option value="LOW">LOW</option>
        </Select>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-xs text-slate-500">
            {filtered.length} of {members.length}
          </span>
          <button
            onClick={exportCsv}
            className="rounded-md bg-surface-raised px-3 py-1.5 text-xs text-slate-300 ring-1 ring-surface-border hover:text-slate-100"
          >
            Export CSV
          </button>
        </div>
      </div>

      {/* table */}
      <div className="overflow-x-auto rounded-lg border border-surface-border">
        <table className="w-full text-left text-xs">
          <thead className="bg-surface-raised text-slate-400">
            <tr>
              <Th onClick={() => toggleSort("id")} active={sort.key === "id"} dir={sort.dir}>id</Th>
              <Th onClick={() => toggleSort("age")} active={sort.key === "age"} dir={sort.dir}>age</Th>
              <Th onClick={() => toggleSort("gender")} active={sort.key === "gender"} dir={sort.dir}>gender</Th>
              <Th onClick={() => toggleSort("cohort")} active={sort.key === "cohort"} dir={sort.dir}>cohort</Th>
              <Th onClick={() => toggleSort("p12")} active={sort.key === "p12"} dir={sort.dir}>p12</Th>
              <Th onClick={() => toggleSort("confidence_band")} active={sort.key === "confidence_band"} dir={sort.dir}>confidence</Th>
              <th className="px-3 py-2 font-medium">flag</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-surface-border">
            {filtered.map((m) => (
              <tr
                key={m.row}
                onClick={() => setSelected(m)}
                className="cursor-pointer hover:bg-surface-raised/50"
              >
                <td className="px-3 py-1.5 font-mono text-slate-300">{m.id}</td>
                <td className="px-3 py-1.5 tabular-nums text-slate-300">{m.age ?? "—"}</td>
                <td className="px-3 py-1.5 text-slate-300">{m.gender}</td>
                <td className="px-3 py-1.5 text-slate-400">{m.cohort}</td>
                <td className="px-3 py-1.5 tabular-nums">
                  {m.p12 == null ? (
                    <span className="text-slate-600">refused</span>
                  ) : (
                    <span className="text-slate-200">{(m.p12 * 100).toFixed(1)}%</span>
                  )}
                </td>
                <td className="px-3 py-1.5">
                  <BandPill band={m.confidence_band} />
                </td>
                <td className="px-3 py-1.5">
                  {m.high_risk && (
                    <span className="rounded bg-red-500/15 px-1.5 py-0.5 text-[10px] text-red-300">
                      high-risk
                    </span>
                  )}
                  {!m.scored && (
                    <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-300">
                      refused
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <MemberDrawer
        member={selected}
        slots={result.slots}
        provenance={result.provenance}
        modelUsed={result.model_used}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}

function Th({
  children,
  onClick,
  active,
  dir,
}: {
  children: React.ReactNode;
  onClick: () => void;
  active: boolean;
  dir: 1 | -1;
}) {
  return (
    <th
      onClick={onClick}
      className="cursor-pointer select-none px-3 py-2 font-medium hover:text-slate-200"
    >
      {children}
      {active && <span className="ml-1 text-slate-500">{dir === 1 ? "▲" : "▼"}</span>}
    </th>
  );
}

function BandPill({ band }: { band: ConfidenceBand | null }) {
  if (!band) return <span className="text-slate-600">—</span>;
  const color =
    band === "HIGH"
      ? "text-emerald-300"
      : band === "MEDIUM"
      ? "text-amber-300"
      : "text-red-300";
  return <span className={"text-[11px] font-medium " + color}>{band}</span>;
}

function Select({
  value,
  onChange,
  label,
  children,
}: {
  value: string;
  onChange: (v: string) => void;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <select
      aria-label={label}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-md border border-surface-border bg-surface-raised px-2 py-1.5 text-xs text-slate-200 outline-none focus:border-brand"
    >
      {children}
    </select>
  );
}

function TabBtn({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={
        "rounded px-3 py-1.5 transition " +
        (active ? "bg-brand text-white" : "text-slate-400 hover:text-slate-200")
      }
    >
      {children}
    </button>
  );
}
