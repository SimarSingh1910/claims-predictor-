#!/usr/bin/env python3
"""
xgboost_frequency.py  —  the THIRD Phase-3 frequency option (XGBoost).

This is a SIBLING to the LogReg deliverable and the LightGBM production seed, NOT a
replacement for either. It trains an XGBoost frequency model on the SAME synthetic
data, through the SAME preprocessor, and calibrates it the SAME way — so Phase 3 now
offers three interchangeable, apples-to-apples options:

    LogReg (deliverable)  |  LightGBM (production seed)  |  XGBoost (this file)

--------------------------------------------------------------------------
WHY A THIRD FREQUENCY MODEL LIVES IN THE SAME PIPELINE
--------------------------------------------------------------------------
LogReg is the internship DELIVERABLE: interpretable, calibrated, validated on real
CIBYL data, and it scores BEST here because the synthetic risk is mostly ADDITIVE —
a linear model captures additive risk fully.

LightGBM and XGBoost are PRODUCTION SEEDS for the team: real TPA claims will carry
correlated-disease interactions (diabetes x hypertension x CKD compounding) that a
tree exploits and a linear model structurally cannot. So a tree is the EXPECTED
long-term winner — but only ON REAL DATA. On this additive synthetic data a tree
should TIE or slightly LOSE, and that is the correct, expected result. We do NOT
tune XGBoost to beat LogReg (or LightGBM); we report whatever a fair search gives.

Everything downstream — severity (Phase 4), pricing (Phase 5), evaluation (Phase 7)
— is SHARED and model-agnostic. XGBoost is just another frequency-model object that
plugs into the same calibrated-P -> P x E pricing slot.

--------------------------------------------------------------------------
WHAT THIS SCRIPT DOES (mirrors Phase 2 + Phase 3, XGBoost edition)
--------------------------------------------------------------------------
1. Load the EXISTING preprocessor.joblib + train/val splits (test stays sealed).
2. Train XGBoost with scale_pos_weight (the imbalance fix, NOT oversampling),
   5-fold CV on TRAIN ONLY, a fair grid comparable to the LightGBM one, with EARLY
   STOPPING on a small train-derived holdout (val is never used for training).
3. Report PR-AUC / AUC-ROC / Brier / log-loss on val.
4. Calibrate isotonic, cv='prefit', on the SAME val_calib/val_eval split Phase 3
   used -> XGBoost vs LogReg vs LightGBM calibration is measured on identical rows.
5. Print a THREE-WAY LogReg-vs-LightGBM-vs-XGBoost table and say plainly which wins.
6. Write models/xgboost_production/MODEL_CARD.md with the real numbers.
"""
import os, sys, json, warnings
from math import prod
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss)
from xgboost import XGBClassifier

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                 # so the preprocessor's custom transformer resolves
import features  # noqa: F401            # registers ClinicalFeatureEngineer for unpickling

SPLITS  = os.path.join(ROOT, "splits")
OUTDIR  = os.path.join(ROOT, "models", "xgboost_production")
os.makedirs(OUTDIR, exist_ok=True)
TARGET  = "claim_next_12m"

# ---- 1. load preprocessor + raw splits (test stays sealed) ------------------
print("Loading preprocessor + raw splits (test stays sealed)...")
preprocessor = joblib.load(os.path.join(ROOT, "preprocessor.joblib"))
train = pd.read_csv(os.path.join(SPLITS, "train.csv"))
val   = pd.read_csv(os.path.join(SPLITS, "val.csv"))

y_train = train[TARGET].astype(int).values
y_val   = val[TARGET].astype(int).values

# preprocessor was FIT in Phase 1 -> here we ONLY transform. No refit, no re-split.
X_train = preprocessor.transform(train)
X_val   = preprocessor.transform(val)
print(f"  X_train {X_train.shape}   X_val {X_val.shape}")
print(f"  claim rate -> train {y_train.mean():.4f}   val {y_val.mean():.4f}")

pos = y_train.sum(); neg = len(y_train) - pos
spw = neg / pos
print(f"  scale_pos_weight = neg/pos = {neg}/{pos} = {spw:.3f}")

# Early stopping needs a watch set the model does NOT train on. We carve it from
# TRAIN ONLY (stratified, seed 42) so that:
#   - val stays SEALED for calibration/eval, exactly as LogReg & LightGBM had it;
#   - test is never touched.
# Grid search then runs its 5-fold CV on the remaining train_fit rows; each fold
# early-stops against this fixed train-derived holdout. No leakage into selection.
X_fit, X_es, y_fit, y_es = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)
print(f"  train_fit {X_fit.shape[0]} rows (grid-search CV)   "
      f"train_es {X_es.shape[0]} rows (early-stopping watch set)")


def metrics(name, y_true, p):
    m = dict(
        auc_roc = roc_auc_score(y_true, p),
        pr_auc  = average_precision_score(y_true, p),
        brier   = brier_score_loss(y_true, p),
        logloss = log_loss(y_true, p),
    )
    print(f"\n  [{name}]  on val")
    print(f"    AUC-ROC : {m['auc_roc']:.4f}")
    print(f"    PR-AUC  : {m['pr_auc']:.4f}   (no-skill baseline = {y_true.mean():.4f})")
    print(f"    Brier   : {m['brier']:.4f}   (lower better)")
    print(f"    log-loss: {m['logloss']:.4f}   (lower better)")
    return m


# ---- 2. train XGBoost, 5-fold CV on TRAIN, fair grid + early stopping --------
print("\n" + "="*64)
print("XGBoost (3rd frequency option), scale_pos_weight, 5-fold CV on TRAIN only")
print("="*64)
base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",         # match the primary metric (PR-AUC) for early stopping
    scale_pos_weight=spw,        # reweight the 16% positives; do NOT oversample
    early_stopping_rounds=50,    # stop when the watch-set aucpr stalls
    tree_method="hist",
    random_state=SEED,
    n_jobs=2,                    # cap parallelism — RAM is tight
)
# A fair grid comparable to the LightGBM one (36 combos x 5 folds = 180 fits).
# n_estimators is the CAP; early stopping decides the actual tree count per fit.
# subsample/colsample sit around 0.8 per the brief. We do NOT expand this to chase
# a win — the goal is an honest read on XGBoost, not engineered superiority.
grid = {
    "max_depth":        [3, 5, 7],
    "n_estimators":     [300, 600],
    "learning_rate":    [0.03, 0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8],
}
N_SPLITS = 5
n_combos = prod(len(v) for v in grid.values())
total_fits = n_combos * N_SPLITS

print("\n  PARAMETER GRID:")
for k, v in grid.items():
    print(f"    {k:<18} {v}")
print(f"\n  candidates (combos) = {' x '.join(str(len(v)) for v in grid.values())}"
      f" = {n_combos}")
print(f"  CV folds            = {N_SPLITS}")
print(f"  TOTAL FITS          = {n_combos} x {N_SPLITS} = {total_fits}")
print(f"  early stopping      = 50 rounds on a {X_es.shape[0]}-row train holdout")
print(f"  n_jobs              = 2  (RAM-friendly; at most 2 fits at once)\n")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
search = GridSearchCV(
    base, grid,
    scoring="average_precision",   # PR-AUC, the primary metric, averaged over folds
    cv=cv,
    n_jobs=2,                      # cap parallelism — RAM is tight
    refit=True,
    verbose=2,
    return_train_score=False,
)
# eval_set is passed through to every fold's fit -> each fold early-stops on the
# fixed train-derived watch set (never part of any CV fold, never val/test).
search.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)

bi = search.best_index_
res = search.cv_results_
fold_scores = [res[f"split{i}_test_score"][bi] for i in range(N_SPLITS)]
print(f"\n  best CV PR-AUC : {search.best_score_:.4f}"
      f"  (+/- {res['std_test_score'][bi]:.4f} across folds)")
print(f"  per-fold PR-AUC: " + ", ".join(f"{s:.4f}" for s in fold_scores))
print(f"  best params    : {search.best_params_}")

xgb = search.best_estimator_
best_iter = getattr(xgb, "best_iteration", None)
if best_iter is not None:
    print(f"  best_iteration : {best_iter} (trees kept after early stopping)")
p_val_xgb = xgb.predict_proba(X_val)[:, 1]
m_xgb = metrics("XGBoost", y_val, p_val_xgb)

# ---- 3. calibrate, on the SAME val split Phase 3 used ----------------------
# Phase 3 split val 50/50 (stratified, seed 42): half to fit the calibrator, half
# held out to judge it. We reproduce that EXACT split so XGBoost's calibrated
# numbers are measured on the identical rows LogReg & LightGBM were, making the
# three-way side-by-side fair.
print("\n" + "="*64)
print("Calibration — isotonic, cv='prefit', same val_calib/val_eval split as Phase 3")
print("="*64)
X_cal, X_eval, y_cal, y_eval = train_test_split(
    X_val, y_val, test_size=0.50, stratify=y_val, random_state=SEED)
print(f"  val_calib {X_cal.shape[0]} rows (fit calibrator)   "
      f"val_eval {X_eval.shape[0]} rows (measure on this)")


def calibration_slope(y_true, p):
    """Slope of logit(observed) ~ logit(predicted): 1.0 = perfect; <1 over-confident."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logit, y_true)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def cal_report(tag, y_true, p):
    slope, intercept = calibration_slope(y_true, p)
    brier = brier_score_loss(y_true, p)
    ll = log_loss(y_true, p)
    print(f"  [{tag}]  slope={slope:.3f}  intercept={intercept:+.3f}  "
          f"Brier={brier:.4f}  log-loss={ll:.4f}")
    return dict(slope=slope, intercept=intercept, brier=brier, logloss=ll)


print("\nBEFORE calibration (XGBoost, on val_eval):")
xgb_before = cal_report("uncalibrated", y_eval, xgb.predict_proba(X_eval)[:, 1])

print("\nFitting CalibratedClassifierCV(method='isotonic', cv='prefit') on val_calib...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")      # sklearn 1.6+ nudges toward FrozenEstimator
    xgb_calibrated = CalibratedClassifierCV(xgb, method="isotonic", cv="prefit")
    xgb_calibrated.fit(X_cal, y_cal)

print("\nAFTER calibration (XGBoost, on val_eval):")
xgb_after = cal_report("calibrated", y_eval, xgb_calibrated.predict_proba(X_eval)[:, 1])

# ---- 4. three-way: re-measure saved LogReg & LightGBM on the SAME rows ------
# We don't trust copied numbers — we load every artifact and score it on the
# identical val / val_eval rows, so every cell of the table is computed here, now.
print("\n" + "="*64)
print("THREE-WAY  —  LogReg (deliverable)  vs  LightGBM (seed)  vs  XGBoost (seed)")
print("="*64)
logreg_raw = joblib.load(os.path.join(ROOT, "frequency_model.joblib"))      # Phase 2 winner
logreg_cal = joblib.load(os.path.join(ROOT, "calibrated_model.joblib"))     # Phase 3 output
lgbm_raw   = joblib.load(os.path.join(ROOT, "models", "lightgbm_production",
                                      "tree_frequency_model.joblib"))
lgbm_cal   = joblib.load(os.path.join(ROOT, "models", "lightgbm_production",
                                      "tree_calibrated_model.joblib"))


def uncal(model):
    p = model.predict_proba(X_val)[:, 1]
    return dict(
        auc_roc = roc_auc_score(y_val, p),
        pr_auc  = average_precision_score(y_val, p),
        brier   = brier_score_loss(y_val, p),
        logloss = log_loss(y_val, p),
    )


m_lr   = uncal(logreg_raw)
m_lgbm = uncal(lgbm_raw)
# m_xgb already computed on full val above

lr_after   = cal_report("LogReg   calibrated", y_eval, logreg_cal.predict_proba(X_eval)[:, 1])
lgbm_after = cal_report("LightGBM calibrated", y_eval, lgbm_cal.predict_proba(X_eval)[:, 1])
# xgb_after already computed above


def best_of(vals, better="high"):
    """Return the name of the winning model among a {name: value} dict."""
    if better == "high":
        return max(vals, key=vals.get)
    return min(vals, key=vals.get)


def row3(label, vals, fmt="{:.4f}", better="high"):
    win = best_of(vals, better)
    cells = "   ".join(f"{fmt.format(v):>10}" for v in vals.values())
    print(f"  {label:<28} {cells}    {win}")


names = ["LogReg", "LightGBM", "XGBoost"]
print(f"\n  {'metric':<28} {'LogReg':>10}   {'LightGBM':>10}   {'XGBoost':>10}    winner")
print("  " + "-"*78)
print("  -- discrimination on full val (uncalibrated raw scores) --")
row3("PR-AUC  (primary)",  dict(zip(names, [m_lr['pr_auc'],  m_lgbm['pr_auc'],  m_xgb['pr_auc']])),  better="high")
row3("AUC-ROC",            dict(zip(names, [m_lr['auc_roc'], m_lgbm['auc_roc'], m_xgb['auc_roc']])), better="high")
print("  -- calibration on held-out val_eval (after isotonic) --")
# slope: closer to 1.0 is better. We rank by distance-to-1 separately for clarity.
slopes = dict(zip(names, [lr_after['slope'], lgbm_after['slope'], xgb_after['slope']]))
slope_win = min(slopes, key=lambda k: abs(slopes[k] - 1.0))
slope_cells = "   ".join(f"{v:>10.4f}" for v in slopes.values())
print(f"  {'calibration slope':<28} {slope_cells}    {slope_win}")
row3("Brier",    dict(zip(names, [lr_after['brier'],   lgbm_after['brier'],   xgb_after['brier']])),   better="low")
row3("log-loss", dict(zip(names, [lr_after['logloss'], lgbm_after['logloss'], xgb_after['logloss']])), better="low")

pr_vals = dict(zip(names, [m_lr['pr_auc'], m_lgbm['pr_auc'], m_xgb['pr_auc']]))
pr_winner = best_of(pr_vals, "high")
gap_xgb_lr = m_xgb['pr_auc'] - m_lr['pr_auc']
print("\n  VERDICT:")
print(f"    PR-AUC on synthetic val:  LogReg {m_lr['pr_auc']:.4f}  |  "
      f"LightGBM {m_lgbm['pr_auc']:.4f}  |  XGBoost {m_xgb['pr_auc']:.4f}")
print(f"    Winner: {pr_winner}.  XGBoost - LogReg = {gap_xgb_lr:+.4f}.")
print("    This is the EXPECTED result. The synthetic risk is additive, which a linear")
print("    model captures fully; the trees have no comorbidity interactions to exploit")
print("    yet. On REAL claims (correlated-disease interactions) a tree is expected to")
print("    win. XGBoost tying/losing here is success, not a bug — we do NOT tune to win.")

# ---- 5. save artifacts ------------------------------------------------------
joblib.dump(xgb,            os.path.join(OUTDIR, "xgb_frequency_model.joblib"))
joblib.dump(xgb_calibrated, os.path.join(OUTDIR, "xgb_calibrated_model.joblib"))
metrics_json = {
    "model": "XGBoost (3rd frequency option)",
    "scale_pos_weight": float(spw),
    "best_params": search.best_params_,
    "best_iteration": (int(best_iter) if best_iter is not None else None),
    "best_cv_pr_auc": float(search.best_score_),
    "xgb_val_uncalibrated": m_xgb,
    "xgb_calibration_val_eval": {"before": xgb_before, "after": xgb_after},
    "logreg_val_uncalibrated": m_lr,
    "logreg_calibration_val_eval_after": lr_after,
    "lightgbm_val_uncalibrated": m_lgbm,
    "lightgbm_calibration_val_eval_after": lgbm_after,
    "pr_auc_winner_synthetic": pr_winner,
    "pr_auc_xgb_minus_logreg": float(gap_xgb_lr),
}
with open(os.path.join(OUTDIR, "phase2_3_metrics_xgb.json"), "w") as f:
    json.dump(metrics_json, f, indent=2)
print(f"\nSaved xgb_frequency_model.joblib + xgb_calibrated_model.joblib -> {OUTDIR}")
print("Saved phase2_3_metrics_xgb.json")

# ---- 6. write the model card with the real numbers --------------------------
def win_cell(vals, better):
    return best_of(vals, better)

pr_row   = dict(zip(names, [m_lr['pr_auc'],  m_lgbm['pr_auc'],  m_xgb['pr_auc']]))
roc_row  = dict(zip(names, [m_lr['auc_roc'], m_lgbm['auc_roc'], m_xgb['auc_roc']]))
brier_row= dict(zip(names, [lr_after['brier'],   lgbm_after['brier'],   xgb_after['brier']]))
ll_row   = dict(zip(names, [lr_after['logloss'], lgbm_after['logloss'], xgb_after['logloss']]))

card = f"""# Model Card — XGBoost Frequency Model (3rd Frequency Option)

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
Trained fair: {n_combos}-combo grid x {N_SPLITS}-fold CV ({total_fits} fits),
`scale_pos_weight={spw:.3f}`, early stopping (50 rounds) on a train-derived holdout,
`random_state=42`. **No tuning to beat LogReg or LightGBM.**
Best params: `{search.best_params_}`{f" (early-stopped at {best_iter} trees)" if best_iter is not None else ""}.

| metric (val)                         | LogReg (deliverable) | LightGBM (seed) | XGBoost (seed) | winner |
|--------------------------------------|---------------------:|----------------:|---------------:|:------:|
| PR-AUC (primary, uncalibrated)       | {m_lr['pr_auc']:.4f} | {m_lgbm['pr_auc']:.4f} | {m_xgb['pr_auc']:.4f} | {win_cell(pr_row,'high')} |
| AUC-ROC (uncalibrated)               | {m_lr['auc_roc']:.4f} | {m_lgbm['auc_roc']:.4f} | {m_xgb['auc_roc']:.4f} | {win_cell(roc_row,'high')} |
| calibration slope (after, val_eval)  | {lr_after['slope']:.3f} | {lgbm_after['slope']:.3f} | {xgb_after['slope']:.3f} | — |
| Brier (after, val_eval)              | {lr_after['brier']:.4f} | {lgbm_after['brier']:.4f} | {xgb_after['brier']:.4f} | {win_cell(brier_row,'low')} |
| log-loss (after, val_eval)           | {lr_after['logloss']:.4f} | {lgbm_after['logloss']:.4f} | {xgb_after['logloss']:.4f} | {win_cell(ll_row,'low')} |

**Framing: {pr_winner} wins on PR-AUC, and a tree tying/slightly losing is the EXPECTED,
CORRECT result.** Additive synthetic risk gives the trees no interactions to exploit,
so they tie or slightly lose to LogReg. XGBoost (gap vs LogReg: {gap_xgb_lr:+.4f}) is
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
"""
with open(os.path.join(OUTDIR, "MODEL_CARD.md"), "w", encoding="utf-8") as f:
    f.write(card)
print("Saved MODEL_CARD.md")
print("\nThird frequency option (XGBoost) complete. Phase 3 now has three calibrated")
print("options (LogReg, LightGBM, XGBoost). Next shared work: Phase 4 (severity).")
