// One-line rationale per frequency model, shown next to the selector so the choice
// is explained, not just labelled.
import type { P12Model } from "../types";

export const MODEL_LABEL: Record<P12Model, string> = {
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  logreg: "LogReg",
};

export const MODEL_RATIONALE: Record<P12Model, string> = {
  xgboost:
    "Tree model — captures comorbidity interactions. Recommended for real data.",
  lightgbm:
    "Gradient-boosted trees — production seed, similar interaction capture.",
  logreg:
    "Linear baseline — wins on additive synthetic risk; the internship deliverable.",
};
