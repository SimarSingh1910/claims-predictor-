// Mirrors the FastAPI response shapes. Metric values are nullable BY DESIGN:
// an untrained slot (p24 / p36 / expected_cost today) is null everywhere, never 0
// and never derived from p12. Keep these optional/nullable so the UI is forced to
// handle "awaiting model" rather than silently rendering a fake number.

export type P12Model = "xgboost" | "lightgbm" | "logreg";
export type Provenance = "synthetic" | "real";
export type ConfidenceBand = "HIGH" | "MEDIUM" | "LOW";
export type SlotKey = "p12" | "p24" | "p36" | "expected_cost";

// ---- /api/health ----
export interface RegistrySlotStatus {
  slot: SlotKey;
  available: boolean;
  reason: string;
}
export interface HealthResponse {
  status: string;
  base_claim_rate: number;
  p12_models: P12Model[];
  registry: RegistrySlotStatus[];
}

// ---- /api/meta ----
export interface RegistryStatusFull {
  slot: SlotKey;
  label: string;
  kind: "classifier" | "regressor";
  unit: string;
  path: string;
  available: boolean;
  reason: string;
}
export interface MetaResponse {
  provenance: Provenance;
  caveat: string;
  base_claim_rate: number;
  mandatory_fields: string[];
  cohorts: {
    age_groups: { key: string; min: number; max: number }[];
    genders: string[];
    low_n_threshold: number;
    high_risk_p12: number;
  };
  core_panel: {
    n_features: number;
    n_important: number;
    important_groups: Record<string, string[]>;
    special_handling: Record<string, string>;
  };
  feature_list: string[];
  p12_models: P12Model[];
  default_p12_model: P12Model;
  registry: RegistryStatusFull[];
}

// ---- /api/score ----
export interface SlotDescriptor {
  slot: SlotKey;
  available: boolean;
  label: string;
  reason?: string; // present only when unavailable
}
export interface ScoreSummary {
  total_rows: number;
  scored: number;
  unscored: number;
  p12: number | null;
  p24: number | null;
  p36: number | null;
  expected_cost: number | null;
  expected_claimants: number | null;
  high_risk_pct: number | null;
  mean_confidence: number | null;
  base_claim_rate: number;
}
export interface Cohort {
  age_group: string;
  gender: string;
  n: number;
  mean_p12: number | null;
  median_p12: number | null;
  p12_std: number | null;
  p24: number | null;
  p36: number | null;
  expected_cost: number | null;
  expected_claimants: number | null;
  high_risk_count: number;
  pct_high_risk: number | null;
  mean_confidence: number | null;
  confidence_band: ConfidenceBand | null;
  low_n: boolean;
}
export interface RiskBucket {
  bucket: string;
  count: number;
}
export interface MissingImportant {
  feature: string;
  group: string;
}
export type MemberValue = string | number | null;
export interface MemberRow {
  row: number;
  id: string;
  age: number | null;
  sex: string | null;
  age_group: string;
  gender: string;
  cohort: string;
  p12: number | null;
  p24: number | null;
  p36: number | null;
  expected_cost: number | null;
  confidence_band: ConfidenceBand | null;
  model_confidence_pct: number | null;
  panel_completeness_pct: number | null;
  n_important_present: number | null;
  n_important_total: number | null;
  missing_important: MissingImportant[];
  high_risk: boolean;
  scored: boolean;
  reason: string;
  values: Record<string, MemberValue>;
}
export interface UnscoredRow {
  row: number;
  reason: string;
}
export interface ScoreResponse {
  model_used: P12Model;
  provenance: Provenance;
  slots: SlotDescriptor[];
  summary: ScoreSummary;
  cohorts: Cohort[];
  risk_distribution: RiskBucket[];
  members: MemberRow[];
  unscored: UnscoredRow[];
  caveat: string;
}

// ---- /api/score-one ----
export interface ScoreOneResponse {
  model_used: P12Model;
  provenance: Provenance;
  slots: SlotDescriptor[];
  age_group: string;
  gender: string;
  cohort: string;
  p12: number | null;
  p24: number | null;
  p36: number | null;
  expected_cost: number | null;
  panel_completeness_pct: number | null;
  model_confidence_pct: number | null;
  confidence_band: ConfidenceBand | null;
  n_important_present: number | null;
  n_important_total: number | null;
  missing_important: MissingImportant[];
  high_risk: boolean;
  scored: boolean;
  reason: string;
  caveat: string;
}

// ---- /api/metrics ----
export interface HeadlineMetrics {
  auc_roc?: number;
  pr_auc?: number;
  brier?: number;
  logloss?: number;
}
export interface MetricsResponse {
  model: string;
  available: boolean;
  reason?: string;
  source_file?: string;
  headline?: {
    available: boolean;
    uncalibrated?: HeadlineMetrics | null;
    calibrated?: HeadlineMetrics | null;
  };
  calibration?: {
    summary: {
      before?: { slope?: number; intercept?: number } | null;
      after?: { slope?: number; intercept?: number } | null;
    };
    points: { available: boolean; reason?: string };
  };
  decile?: { available: boolean; reason?: string };
}

// ---- error envelope ----
// The client normalises every failure to this so the UI can branch on `kind`.
export type ApiErrorKind = "network" | "validation" | "server";
export interface ApiError {
  kind: ApiErrorKind;
  status?: number;
  message: string;
  detail?: unknown; // e.g. { missing_columns, unknown_columns } from a 400
}
