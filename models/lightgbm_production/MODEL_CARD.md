# Model Card — LightGBM Frequency Model (Production Seed)

**Status:** experimental / production-track. **Not** the internship headline model.
**Family:** gradient-boosted trees (LightGBM). Sibling to the LogReg deliverable.
**Artifacts:** `tree_frequency_model.joblib`, `tree_calibrated_model.joblib`,
`tree_metrics.json`.

## What it is
A claim-propensity (frequency) model — `P(claim in next 12m)` — trained on the SAME
synthetic AHC data, through the SAME `preprocessor.joblib`, and calibrated the SAME
way (isotonic, `cv='prefit'`) as the Phase 2/3 LogReg model. It is a drop-in
frequency-model object for the shared, model-agnostic pricing pipeline: it produces a
calibrated `P` that feeds `expected_cost = P x E(amount)` exactly like LogReg does.

## Why it exists
The internship deliverable is **LogReg** — interpretable, calibrated, and validated on
real CIBYL data. But the synthetic data's risk is mostly **additive**, which a linear
model captures fully. **Real TPA claims** (arriving after the internship) will contain
**correlated-disease interactions** — comorbidities like diabetes x hypertension x CKD
that compound risk non-linearly. Trees exploit those interactions; linear models
structurally cannot. So this tree is the model the **team graduates to when real data
arrives**. It is seeded now, on synthetic data, so the swap is a one-object change.

## Synthetic result (val)
Trained fair: same 36-combo grid x 5-fold CV (180 fits),
`scale_pos_weight=5.236`, `random_state=42`. **No tuning to beat LogReg.**

| metric (val)                         | LogReg (deliverable) | LightGBM (seed) | winner |
|--------------------------------------|---------------------:|----------------:|:------:|
| PR-AUC (primary, uncalibrated)       | 0.5861 | 0.5759 | LogReg |
| AUC-ROC (uncalibrated)               | 0.8410 | 0.8366 | LogReg |
| calibration slope (after, val_eval)  | 0.966 | 0.950 | — |
| Brier (after, val_eval)              | 0.0970 | 0.0981 | LogReg |
| log-loss (after, val_eval)           | 0.3226 | 0.3382 | LogReg |

**Framing: LogReg wins on PR-AUC by +0.0102, and that is the EXPECTED, CORRECT
result.** Additive synthetic risk gives the tree no interactions to exploit, so it ties
or slightly loses. The tree is expected to **win on real data**, where comorbidity
interactions are present. Losing here is not a failure of the tree — it is evidence the
synthetic data is additive, exactly as designed.

## How the team uses it (when real claims arrive)
1. Re-run `preprocessor.joblib` on the real, labelled data (do not refit features).
2. Retrain this tree on real `claim_next_12m` (same grid is a fine starting point).
3. **Recalibrate** (isotonic, `cv='prefit'`) — honest probabilities are mandatory for
   pricing.
4. Compare LogReg vs tree on the **SEALED test set** (Phase 7), PR-AUC primary.
5. **Ship the winner.** Expectation: the tree overtakes LogReg once interactions exist.

## What this model does NOT change
Severity (Phase 4), pricing (Phase 5), and evaluation (Phase 7) are **shared and
model-agnostic**. There is no separate pricing pipeline for the tree — it plugs into
the same calibrated-P slot. The LogReg artifacts are untouched.
