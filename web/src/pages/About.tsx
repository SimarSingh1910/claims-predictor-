// About page — interview-proof. Explains what the engine does, the pricing
// formula (with which terms are live vs awaiting a model), why XGBoost with an
// honest synthetic-data caveat, the model registry / roadmap, the synthetic-target
// disclaimer, and a per-model metrics table (all read from real files via
// /api/metrics; the decile-lift figure is cited from the v1 model card).
import { useEffect, useState } from "react";
import type { MetaResponse, MetricsResponse, P12Model } from "../types";
import { getMetrics } from "../lib/api";
import { ProvenanceBadge, SlotStatusDot } from "../components/Badges";
import { AlertTriangle } from "../components/icons";

const MODELS: P12Model[] = ["logreg", "xgboost", "lightgbm"];
const MODEL_LABEL: Record<P12Model, string> = {
  logreg: "LogReg (v1)",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
};

// What each untrained slot is waiting on — the roadmap.
const SLOT_ROADMAP: Record<string, string> = {
  p12: "Trained — calibrated XGBoost on AHC features.",
  p24: "Needs 24-month longitudinal claims to define the label.",
  p36: "Needs 36-month longitudinal claims to define the label.",
  expected_cost: "Phase 4–5 severity model: E(claim amount | claim) in ₹.",
};

export function About({ meta }: { meta: MetaResponse | null }) {
  const [metrics, setMetrics] = useState<Record<string, MetricsResponse>>({});

  useEffect(() => {
    let alive = true;
    (async () => {
      const out: Record<string, MetricsResponse> = {};
      for (const m of MODELS) {
        try {
          out[m] = await getMetrics(m);
        } catch {
          /* skip a model whose metrics file is unreadable */
        }
      }
      if (alive) setMetrics(out);
    })();
    return () => {
      alive = false;
    };
  }, []);

  if (!meta) return <div className="p-6 text-sm text-slate-400">Loading…</div>;

  return (
    <div className="mx-auto max-w-4xl space-y-8 p-6">
      <header>
        <h1 className="text-xl font-semibold text-slate-100">
          HealthBridge Claim Engine
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
          A claim-propensity &amp; group-pricing engine for HCL Healthcare's
          HealthBridge platform. It scores each employee's Annual Health Checkup
          (AHC) record for claim risk, aggregates individuals into age × gender
          cohorts, and rolls those up into a group-level view that informs how an
          employer's health plan is priced. Every score is real inference against a
          pre-trained model — nothing is trained at request time.
        </p>
      </header>

      {/* synthetic caveat, prominent */}
      <div className="flex items-start gap-3 rounded-lg border border-amber-500/30 bg-amber-500/[0.08] p-4">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
        <div>
          <div className="text-sm font-semibold text-amber-200">
            Metrics are pipeline validation, not real-world claim rates
          </div>
          <p className="mt-1 text-xs leading-relaxed text-amber-100/80">
            {meta.caveat} Because the target was generated from the AHC features by
            a formula plus noise, model accuracy reflects that the pipeline works —
            not how well it predicts real claims. Every metric below carries a{" "}
            <span className="font-semibold">Synthetic</span> badge for that reason.
          </p>
        </div>
      </div>

      {/* the formula */}
      <section>
        <h2 className="mb-3 text-sm font-semibold text-slate-200">
          The pricing formula
        </h2>
        <div className="rounded-lg border border-surface-border bg-surface-panel p-5">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-3 text-sm">
            <Term label="P(claim)" live sub="p12 · trained" />
            <Op>×</Op>
            <Term label="E(cost | claim)" sub="severity · awaiting" />
            <Op>=</Op>
            <Term label="Expected cost" sub="per member" awaiting />
            <Op>→</Op>
            <Term label="Group premium" sub="cohort roll-up" awaiting />
          </div>
          <p className="mt-4 text-xs leading-relaxed text-slate-500">
            The frequency term <span className="font-mono text-slate-400">P(claim)</span>{" "}
            is live (12-month; 24- and 36-month await longitudinal claims). The
            severity term <span className="font-mono text-slate-400">E(cost)</span>{" "}
            is the Phase 4–5 model — until it ships, expected cost and premium stay
            empty rather than being guessed.
          </p>
        </div>
      </section>

      {/* why XGBoost */}
      <section>
        <h2 className="mb-3 text-sm font-semibold text-slate-200">Why XGBoost</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-lg border border-surface-border bg-surface-panel p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              The case for a tree
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">
              Real diseases are comorbid and correlated — diabetes × hypertension ×
              CKD compound super-additively. Tree models capture these interactions
              natively; a linear model cannot without hand-built interaction terms.
              The product targets real claims data, so the tree is the intended
              production model.
            </p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-panel p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Honest note on current data
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">
              On the current <span className="font-semibold">synthetic</span> data,
              LogReg slightly edges XGBoost ({fmtPr(metrics.logreg)} vs{" "}
              {fmtPr(metrics.xgboost)} PR-AUC) — because the synthetic risk is{" "}
              <span className="font-semibold">additive by construction</span>, which
              is exactly what a linear model captures best. A tree tying or slightly
              losing here is the correct, expected result; it's built to win once
              real comorbidity interactions appear.
            </p>
          </div>
        </div>
      </section>

      {/* model registry / roadmap */}
      <section>
        <h2 className="mb-3 text-sm font-semibold text-slate-200">
          Model registry &amp; roadmap
        </h2>
        <div className="overflow-hidden rounded-lg border border-surface-border">
          <table className="w-full text-left text-sm">
            <thead className="bg-surface-raised text-[11px] uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-3 py-2">Slot</th>
                <th className="px-3 py-2">Prediction</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">What it needs</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border">
              {meta.registry.map((r) => (
                <tr key={r.slot} className="bg-surface-panel align-top">
                  <td className="px-3 py-2 font-mono text-xs text-slate-300">{r.slot}</td>
                  <td className="px-3 py-2 text-slate-300">{r.label}</td>
                  <td className="whitespace-nowrap px-3 py-2">
                    <span className="inline-flex items-center gap-1.5">
                      <SlotStatusDot available={r.available} />
                      <span className={r.available ? "text-emerald-300" : "text-slate-400"}>
                        {r.available ? "trained" : "coming"}
                      </span>
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs text-slate-500">
                    {SLOT_ROADMAP[r.slot] ?? r.reason}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          When a future joblib is dropped at its registry path, that slot lights up
          across the whole UI with no code change — the acceptance criterion for the
          design.
        </p>
      </section>

      {/* metrics table */}
      <section>
        <div className="mb-3 flex items-center gap-2">
          <h2 className="text-sm font-semibold text-slate-200">
            Frequency-model metrics
          </h2>
          <ProvenanceBadge provenance="synthetic" />
        </div>
        <div className="overflow-x-auto rounded-lg border border-surface-border">
          <table className="w-full text-left text-sm">
            <thead className="bg-surface-raised text-[11px] uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-3 py-2">Model</th>
                <th className="px-3 py-2">PR-AUC</th>
                <th className="px-3 py-2">Brier (uncal.)</th>
                <th className="px-3 py-2">Brier (calibrated)</th>
                <th className="px-3 py-2">Calibration slope</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border">
              {MODELS.map((m) => {
                const md = metrics[m];
                const u = md?.headline?.uncalibrated;
                const c = md?.headline?.calibrated;
                const after = md?.calibration?.summary.after;
                const best =
                  m === "logreg"; // LogReg is the synthetic PR-AUC winner
                return (
                  <tr key={m} className="bg-surface-panel">
                    <td className="px-3 py-2 text-slate-200">
                      {MODEL_LABEL[m]}
                      {best && (
                        <span className="ml-1.5 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-300">
                          synthetic winner
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2 tabular-nums text-slate-300">{num(u?.pr_auc)}</td>
                    <td className="px-3 py-2 tabular-nums text-slate-300">{num(u?.brier)}</td>
                    <td className="px-3 py-2 tabular-nums text-slate-300">{num(c?.brier)}</td>
                    <td className="px-3 py-2 tabular-nums text-slate-300">{num(after?.slope)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* decile lift callout — cited from the v1 model card (validation) */}
        <div className="mt-3 flex items-center justify-between rounded-lg border border-surface-border bg-surface-panel p-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wide text-slate-400">
                Decile lift (validation)
              </span>
              <ProvenanceBadge provenance="synthetic" />
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Top decile 65.2% actual claim rate vs bottom decile 0.6% — clean and
              monotonic. Source:{" "}
              <span className="font-mono">models/v1_additive/MODEL_CARD.md</span>.
            </p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-semibold tabular-nums text-slate-100">
              109×
            </div>
            <div className="text-[11px] text-slate-500">top-to-bottom</div>
          </div>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Brier improves after isotonic calibration (e.g. XGBoost{" "}
          {num(metrics.xgboost?.headline?.uncalibrated?.brier)} →{" "}
          {num(metrics.xgboost?.headline?.calibrated?.brier)}); calibrated
          probabilities are what pricing must use.
        </p>
      </section>

      {/* base facts */}
      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Fact label="Base claim rate" value={`${(meta.base_claim_rate * 100).toFixed(2)}%`} />
        <Fact label="Feature panel" value={`${meta.core_panel.n_important}/${meta.core_panel.n_features}`} />
        <Fact label="Mandatory fields" value={String(meta.mandatory_fields.length)} />
        <Fact label="Provenance" value="Synthetic" />
      </section>
    </div>
  );
}

// --- small helpers ---
function num(v?: number | null): string {
  return v == null ? "—" : v.toFixed(3);
}
function fmtPr(md?: MetricsResponse): string {
  const v = md?.headline?.uncalibrated?.pr_auc;
  return v == null ? "—" : v.toFixed(3);
}

function Term({
  label,
  sub,
  live,
  awaiting,
}: {
  label: string;
  sub: string;
  live?: boolean;
  awaiting?: boolean;
}) {
  return (
    <div
      className={
        "rounded-md border px-3 py-2 " +
        (live
          ? "border-emerald-500/30 bg-emerald-500/[0.06]"
          : awaiting
          ? "border-dashed border-surface-border bg-surface-panel/40"
          : "border-surface-border bg-surface-panel")
      }
    >
      <div className={"font-mono text-sm " + (live ? "text-emerald-200" : "text-slate-300")}>
        {label}
      </div>
      <div className="mt-0.5 text-[10px] text-slate-500">{sub}</div>
    </div>
  );
}
function Op({ children }: { children: React.ReactNode }) {
  return <span className="text-lg text-slate-500">{children}</span>;
}
function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-surface-border bg-surface-panel p-3">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-lg font-semibold text-slate-100">{value}</div>
    </div>
  );
}
