// ManualScorer — single-member what-if. Fields grouped by clinical system,
// mandatory ones marked. Edits trigger a debounced live re-score via
// /api/score-one, so "what if HbA1c were 9?" updates p12 as you type. A blank
// mandatory field puts the card into the refusal state — the engine refuses
// rather than guessing.
import { useEffect, useMemo, useRef, useState } from "react";
import { postScoreOne, sampleUrl } from "../lib/api";
import type {
  MetaResponse,
  P12Model,
  ScoreOneResponse,
  MemberValue,
  ApiError,
  SlotKey,
} from "../types";
import { MetricCard } from "./MetricCard";
import { Meter } from "./Meter";
import { MODEL_LABEL } from "../lib/models";

const SLOT_ORDER: SlotKey[] = ["p12", "p24", "p36", "expected_cost"];
const DEMOGRAPHICS = ["age", "sex", "height_cm", "weight_kg"];

interface FieldGroup {
  name: string;
  fields: string[];
}

export function ManualScorer({
  meta,
  model,
}: {
  meta: MetaResponse | null;
  model: P12Model;
}) {
  const mandatory = useMemo(
    () => new Set(meta?.mandatory_fields ?? []),
    [meta]
  );

  const groups: FieldGroup[] = useMemo(() => {
    if (!meta) return [];
    const g: FieldGroup[] = [{ name: "Demographics & vitals", fields: DEMOGRAPHICS }];
    for (const [name, feats] of Object.entries(meta.core_panel.important_groups)) {
      g.push({ name, fields: feats });
    }
    return g;
  }, [meta]);

  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<ScoreOneResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [scoring, setScoring] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const setField = (k: string, v: string) =>
    setValues((prev) => ({ ...prev, [k]: v }));

  const loadSample = async () => {
    try {
      const text = await fetch(sampleUrl()).then((r) => r.text());
      const lines = text.replace(/\r\n?/g, "\n").split("\n").filter(Boolean);
      const cols = lines[0].split(",");
      const row = lines[1].split(",");
      const next: Record<string, string> = {};
      cols.forEach((c, i) => (next[c.trim()] = (row[i] ?? "").trim()));
      setValues(next);
    } catch {
      /* ignore — user can still type */
    }
  };

  // Which mandatory fields are currently blank → the form will refuse.
  const blankMandatory = [...mandatory].filter(
    (m) => !values[m] || values[m].trim() === ""
  );

  // Debounced live re-score whenever inputs change.
  useEffect(() => {
    if (Object.keys(values).length === 0) return;
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      const payload: Record<string, MemberValue> = {};
      for (const [k, v] of Object.entries(values)) {
        if (v == null || v.trim() === "") continue; // omit blanks -> "missing"
        if (k === "sex") payload[k] = v;
        else {
          const n = Number(v);
          payload[k] = Number.isFinite(n) ? n : v;
        }
      }
      setScoring(true);
      setError(null);
      try {
        const res = await postScoreOne(payload, model);
        setResult(res);
      } catch (e) {
        setError(e as ApiError);
      } finally {
        setScoring(false);
      }
    }, 400);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [values, model]);

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_360px]">
      {/* form */}
      <div className="space-y-5">
        <div className="flex items-center justify-between">
          <p className="text-sm text-slate-400">
            Enter values to score a single member. Edits re-score live.
          </p>
          <button
            onClick={loadSample}
            className="rounded-md bg-surface-raised px-3 py-1.5 text-xs text-slate-300 ring-1 ring-surface-border hover:text-slate-100"
          >
            Load a sample member
          </button>
        </div>

        {groups.map((grp) => (
          <div key={grp.name}>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
              {grp.name}
            </h3>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {grp.fields.map((f) => (
                <Field
                  key={f}
                  name={f}
                  value={values[f] ?? ""}
                  mandatory={mandatory.has(f)}
                  onChange={(v) => setField(f, v)}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* live result panel */}
      <div className="lg:sticky lg:top-4 lg:self-start">
        <div className="space-y-3 rounded-lg border border-surface-border bg-surface-panel p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200">Live score</h3>
            {scoring && <span className="text-xs text-slate-500">scoring…</span>}
          </div>

          {blankMandatory.length > 0 ? (
            <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200/90">
              Refused — missing mandatory:{" "}
              <span className="font-mono">{blankMandatory.join(", ")}</span>. The
              engine refuses rather than guessing these.
            </div>
          ) : error ? (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">
              {error.kind === "network"
                ? "Backend unreachable."
                : error.message}
            </div>
          ) : result ? (
            <>
              <div className="text-xs text-slate-400">
                Cohort: <span className="text-slate-200">{result.cohort}</span>
                {result.high_risk && (
                  <span className="ml-1.5 rounded bg-red-500/15 px-1.5 py-0.5 text-[10px] text-red-300">
                    high-risk
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2">
                {SLOT_ORDER.map((slot) => {
                  const sm = result.slots.find((x) => x.slot === slot);
                  return (
                    <MetricCard
                      key={slot}
                      slot={slot}
                      label={sm?.label ?? slot}
                      value={result[slot] as number | null}
                      available={sm?.available ?? false}
                      reason={sm?.reason ?? "model not trained yet"}
                      provenance={result.provenance}
                    />
                  );
                })}
              </div>
              <div className="space-y-2 pt-1">
                <Meter
                  label="Panel completeness"
                  pct={result.panel_completeness_pct}
                  hint={`${result.n_important_present ?? 0}/${result.n_important_total ?? 0} important params`}
                />
                <Meter
                  label="Model confidence"
                  pct={result.model_confidence_pct}
                  hint={`importance-weighted coverage of present features (${
                    MODEL_LABEL[result.model_used as P12Model] ?? result.model_used
                  })`}
                />
              </div>
            </>
          ) : (
            <div className="text-xs text-slate-500">
              Fill the mandatory fields to see a score.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Field({
  name,
  value,
  mandatory,
  onChange,
}: {
  name: string;
  value: string;
  mandatory: boolean;
  onChange: (v: string) => void;
}) {
  const isSex = name === "sex";
  return (
    <label className="flex flex-col gap-1">
      <span className="truncate font-mono text-[10px] text-slate-500" title={name}>
        {name}
        {mandatory && <span className="ml-0.5 text-red-400">*</span>}
      </span>
      {isSex ? (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={
            "rounded border bg-surface-raised px-2 py-1 text-xs text-slate-200 outline-none focus:border-brand " +
            (mandatory && !value ? "border-red-500/40" : "border-surface-border")
          }
        >
          <option value="">—</option>
          <option value="M">M</option>
          <option value="F">F</option>
        </select>
      ) : (
        <input
          type="number"
          inputMode="decimal"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={
            "rounded border bg-surface-raised px-2 py-1 text-xs tabular-nums text-slate-200 outline-none focus:border-brand " +
            (mandatory && !value ? "border-red-500/40" : "border-surface-border")
          }
        />
      )}
    </label>
  );
}
