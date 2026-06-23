# Model Card — v1 `additive` (FROZEN baseline)

**Status: FROZEN.** This is the recoverable working baseline for the HealthBridge
Claim Propensity engine. Restore with `git checkout v1-additive`.
Snapshot date: 2026-06-23. Do not modify files in this folder.

---

## What it is

A **frequency** pipeline only — it predicts **P(claim in next 12 months)** per member.
Three stacked, frozen components:

1. `preprocessor.joblib` — Phase-1 `ColumnTransformer`: clinical feature engineering
   (binary/ordinal encodes, chronic-disease → `has_*`, 10 abnormal-lab flags, PSA
   structural-missingness handling, `log1p` on skewed labs) + `StandardScaler`,
   **fit on train only**. 115 raw inputs → 133 engineered features.
2. `frequency_model.joblib` — `LogisticRegression(class_weight='balanced')`.
3. `calibrated_model.joblib` — the LogReg wrapped in isotonic calibration
   (`CalibratedClassifierCV`, prefit), so the probabilities are usable as true
   probabilities. **Pricing/scoring must use this one, never the raw LogReg.**
4. `graceful_scorer.py` (`score_member`) — partial-input wrapper: refuses on a
   missing mandatory minimum, imputes optional fields from train medians, derives
   computable values (eGFR/BMI/eAG/ratios), and returns two confidence numbers
   (clinical panel completeness + coefficient-weighted model confidence). Driven by
   `core_panel.json` (31 clinically-authoritative important labs; mandatory minimum:
   age, sex, hba1c, fbs, creatinine, total_cholesterol).

Companion files included so the joblibs unpickle and the scorer runs standalone:
`features.py` (defines `ClinicalFeatureEngineer`), `feature_names.json`.
The scorer also reads train medians from `splits/train.csv` at the project root.

## Target

**Additive synthetic claim target.** Latent claim risk = a weighted sum of
individual markers + chronic conditions + noise, with **NO interaction terms** —
each condition contributes independently. Realised as `claim_next_12m` (~16.04%
positive). Real TPA claims will replace this synthetic target later; the pipeline
is built to survive that swap with no structural change.

## Validation done (all on the held-out validation split; test stays sealed)

**Phase 2 — frequency model.** Primary metric PR-AUC (honest at 16% positives).
A tuned LightGBM (Randomized/GridSearch, 5-fold CV on train) could **not** beat the
linear baseline, so LogReg was kept:

| model | PR-AUC | AUC-ROC | Brier | log-loss |
|---|---|---|---|---|
| **LogReg (chosen)** | **0.5861** | 0.8410 | 0.1558 | 0.4799 |
| LightGBM (tuned) | 0.5759 | 0.8366 | 0.1545 | 0.4748 |

`scale_pos_weight`/`class_weight` = 5.236 (neg/pos). Decile lift on validation was
clean and monotonic: **top decile 65.2% actual claim rate vs bottom decile 0.6%
(~109× top-to-bottom; top decile 4.07× the average).**

**Phase 3 — calibration.** Isotonic, prefit, fit on one half of val and measured
on the held-out other half:

| | slope | intercept | Brier | log-loss |
|---|---|---|---|---|
| before | 0.942 | −1.633 | 0.1562 | 0.4804 |
| **after** | **0.966** | **−0.046** | **0.0970** | 0.3226 |

The `class_weight='balanced'` LogReg was systematically over-confident; calibration
nearly halved Brier and pulled the intercept to ~0.

**Real-data sanity check — CIBYL.** The v1 pipeline was run on a real
CIBYL/Thyrocare AHC export (2460 records, **1469 with a CIBYL score**) via the
graceful scorer. Result — a strong, expected agreement:

- **Spearman ρ(CIBYL score, predicted claim probability) = −0.543, p ≈ 3×10⁻⁵³,
  n = 677 HIGH-confidence members.** Higher CIBYL (healthier) → lower predicted
  claim probability.
- **0 refused** (all 1469 had the mandatory minimum); mean predicted claim
  probability 4.65%; mean claim rate falls monotonically across CIBYL labels
  (Watchful 11.1% → Fair 7.6% → Excellent 3.1%).
- **Blood pressure was present in ~52% of records** (not absent as initially
  assumed); diagnosed-condition labels (`chronic_disease`) were absent everywhere,
  so the `has_*` features are a known blind spot that honestly lowers model
  confidence. Artifacts: `real_predictions.csv`, `cibyl_claim_dashboard.html`.
- This is a **consistency check** (CIBYL shares input parameters with the model),
  **not** independent validation against real claims.

## NOT built in v1 (deliberately out of scope)

- **Phase 4 — severity** model (E[claim amount | claim]).
- **Phase 5 — pricing** engine (group premium = Σ P × E).
- **Phase 6 — explainability** (SHAP).
- **Phase 7 — sealed-test evaluation** (test split has never been scored).
- **Phase 8 — swap-readiness** runbook for real TPA claims.

## Known limitations

- Trained on **synthetic** claims; absolute probabilities are not real-world rates.
- Over-reliant on `has_*` condition flags (the model's heaviest features), which are
  absent from real exports → real-data confidence is dampened.
- Additive target has no comorbidity interactions — the motivation for the v2
  `compounded` experiment.
