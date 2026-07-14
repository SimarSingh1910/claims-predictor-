// Top bar: title, model selector (drives p12 only), provenance badge, health chip.
import { Activity } from "./icons";
import type { HealthResponse, ApiError, P12Model, Provenance } from "../types";
import { ProvenanceBadge } from "./Badges";
import { HealthChip } from "./HealthChip";

const MODEL_LABELS: Record<P12Model, string> = {
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  logreg: "LogReg",
};

interface Props {
  health: HealthResponse | null;
  healthError: ApiError | null;
  healthLoading: boolean;
  provenance: Provenance;
  models: P12Model[];
  model: P12Model;
  onModelChange: (m: P12Model) => void;
}

export function TopBar({
  health,
  healthError,
  healthLoading,
  provenance,
  models,
  model,
  onModelChange,
}: Props) {
  return (
    <header className="flex items-center justify-between gap-4 border-b border-surface-border bg-surface-panel px-5 py-3">
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-brand/15 ring-1 ring-brand/30">
          <Activity className="h-4.5 w-4.5 text-brand" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold text-slate-100">
            HealthBridge · Claim Engine
          </div>
          <div className="text-[11px] text-slate-400">
            Group claim-propensity &amp; pricing
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <ProvenanceBadge provenance={provenance} />

        <label className="flex items-center gap-2 text-xs text-slate-400">
          <span className="hidden sm:inline">p12 model</span>
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value as P12Model)}
            className="rounded-md border border-surface-border bg-surface-raised px-2 py-1 text-xs text-slate-200 outline-none focus:border-brand"
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {MODEL_LABELS[m] ?? m}
                {m === "xgboost" ? " (default)" : ""}
              </option>
            ))}
          </select>
        </label>

        <HealthChip
          health={health}
          error={healthError}
          loading={healthLoading}
        />
      </div>
    </header>
  );
}
