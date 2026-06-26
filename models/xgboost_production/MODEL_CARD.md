# Model Card — XGBoost Frequency Model (3rd Frequency Option)

**Status:** experimental / production-track. **Not** the internship headline model.
**Family:** gradient-boosted trees (XGBoost). Sibling to the LogReg deliverable and
the LightGBM production seed.
**Artifacts:** `xgb_frequency_model.joblib`, `xgb_calibrated_model.joblib`,
`phase2_3_metrics_xgb.json`.

## What it is
A claim-propensity (frequency) model — `P(claim in next 12m)` — trained on the SAME
synthetic AHC data, through the SAME `preprocessor.joblib`, and calibrated the SAME
way (isotonic, `cv='prefit'`) as the Phase 2/3 LogReg model and the LightGBM seed.
It is a drop-in frequency-model object for the shared, model-agnostic pricing
pipeline: it produces a calibrated `P` that feeds `expected_cost = P x E(amount)`
exactly like the other two do.

## Why it exists
The internship deliverable is **LogReg** — interpretable, calibrated, validated on
real CIBYL data. But the synthetic data's risk is mostly **additive**, which a linear
model captures fully. **Real TPA claims** (arriving after the internship) will contain
**correlated-disease interactions** — comorbidities like diabetes x hypertension x CKD
that compound risk non-linearly. Trees exploit those interactions; linear models
structurally cannot. XGBoost is offered as a **second tree option alongside LightGBM**
so the team can compare both gradient-boosting families on real data and ship the
strongest. It is seeded now, on synthetic data, so the swap is a one-object change.

## Synthetic result (val)
Trained fair: 36-combo grid x 5-fold CV (180 fits),
`scale_pos_weight=5.236`, early stopping (50 rounds) on a train-derived holdout,
`random_state=42`. **No tuning to beat LogReg or LightGBM.**
Best params: `{'colsample_bytree': 0.8, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 600, 'subsample': 0.8}` (early-stopped at 597 trees).

| metric (val)                         | LogReg (deliverable) | LightGBM (seed) | XGBoost (seed) | winner |
|--------------------------------------|---------------------:|----------------:|---------------:|:------:|
| PR-AUC (primary, uncalibrated)       | 0.5861 | 0.5759 | 0.5778 | LogReg |
| AUC-ROC (uncalibrated)               | 0.8410 | 0.8366 | 0.8374 | LogReg |
| calibration slope (after, val_eval)  | 0.966 | 0.950 | 0.935 | — |
| Brier (after, val_eval)              | 0.0970 | 0.0981 | 0.0979 | LogReg |
| log-loss (after, val_eval)           | 0.3226 | 0.3382 | 0.3412 | LogReg |

**Framing: LogReg wins on PR-AUC, and a tree tying/slightly losing is the EXPECTED,
CORRECT result.** Additive synthetic risk gives the trees no interactions to exploit,
so they tie or slightly lose to LogReg. XGBoost (gap vs LogReg: -0.0083) is
expected to **win on real data**, where comorbidity interactions are present. Losing
here is not a failure — it is evidence the synthetic data is additive, as designed.

## How the team uses it (when real claims arrive)
1. Re-run `preprocessor.joblib` on the real, labelled data (do not refit features).
2. Retrain this model on real `claim_next_12m` (same grid is a fine starting point).
3. **Recalibrate** (isotonic, `cv='prefit'`) — honest probabilities are mandatory for
   pricing.
4. Compare LogReg vs LightGBM vs XGBoost on the **SEALED test set** (Phase 7), PR-AUC
   primary.
5. **Ship the winner.** Expectation: a tree overtakes LogReg once interactions exist;
   XGBoost vs LightGBM is then a head-to-head on the real data.

## What this model does NOT change
Severity (Phase 4), pricing (Phase 5), and evaluation (Phase 7) are **shared and
model-agnostic**. There is no separate pricing pipeline for XGBoost — it plugs into
the same calibrated-P slot. The LogReg, v2, and LightGBM artifacts are untouched.
