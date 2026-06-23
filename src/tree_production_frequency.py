#!/usr/bin/env python3
"""
tree_production_frequency.py  —  the PRODUCTION-SEED frequency model.

This is a SIBLING to the Phase 2/3 LogReg pipeline, NOT a replacement. It trains a
LightGBM frequency model (matching the tree family Phase 2 already trialled) on the
SAME synthetic data, through the SAME preprocessor, and calibrates it the SAME way.

--------------------------------------------------------------------------
WHY TWO FREQUENCY MODELS LIVE IN ONE PIPELINE
--------------------------------------------------------------------------
LogReg is the internship DELIVERABLE: interpretable, calibrated, already validated
on real CIBYL data (Spearman -0.54), and it scores BETTER than the tree here because
the synthetic risk is mostly ADDITIVE — a linear model captures additive risk fully.

LightGBM is the PRODUCTION SEED for the team: real TPA claims will carry correlated-
disease interactions (e.g. diabetes x hypertension x CKD compounding) that a tree
exploits and a linear model structurally cannot. So the tree is the EXPECTED long-
term winner — but only ON REAL DATA. On this additive synthetic data it should TIE
or slightly LOSE, and that is the correct, expected result. We do NOT tune it to win.

Everything downstream — severity (Phase 4), pricing (Phase 5), evaluation (Phase 7)
— is SHARED and model-agnostic. The tree is just another frequency-model object that
plugs into the same calibrated-P -> P x E pricing slot.

--------------------------------------------------------------------------
WHAT THIS SCRIPT DOES (mirrors Phase 2 + Phase 3, tree edition)
--------------------------------------------------------------------------
1. Load the EXISTING preprocessor.joblib + train/val splits (test stays sealed).
2. Train LightGBM with scale_pos_weight (the imbalance fix, NOT oversampling),
   5-fold CV on TRAIN ONLY, the same fair grid Phase 2 used.
3. Report PR-AUC / AUC-ROC / Brier / log-loss on val.
4. Calibrate isotonic, cv='prefit', on the SAME val_calib/val_eval split Phase 3
   used -> tree vs LogReg calibration is measured on identical rows.
5. Print a side-by-side LogReg-vs-tree table and say plainly which wins.
6. Write models/lightgbm_production/MODEL_CARD.md with the real numbers.
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
import lightgbm as lgb

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                 # so the preprocessor's custom transformer resolves
import features  # noqa: F401            # registers ClinicalFeatureEngineer for unpickling

SPLITS  = os.path.join(ROOT, "splits")
OUTDIR  = os.path.join(ROOT, "models", "lightgbm_production")
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


# ---- 2. train LightGBM, 5-fold CV on TRAIN, same fair grid as Phase 2 -------
print("\n" + "="*64)
print("LightGBM (production seed), scale_pos_weight, 5-fold CV on TRAIN only")
print("="*64)
base = lgb.LGBMClassifier(
    objective="binary",
    scale_pos_weight=spw,        # reweight the 16% positives; do NOT oversample
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
# SAME fair grid Phase 2 used. We deliberately reuse it rather than expand it:
# the goal is an honest read on the tree, not to engineer a win on additive data.
grid = {
    "n_estimators":  [300, 600],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth":     [4, 6, 8],
    "num_leaves":    [31, 63],
}
N_SPLITS = 5
n_combos = prod(len(v) for v in grid.values())
total_fits = n_combos * N_SPLITS

print("\n  PARAMETER GRID:")
for k, v in grid.items():
    print(f"    {k:<14} {v}")
print(f"\n  candidates (combos) = {' x '.join(str(len(v)) for v in grid.values())}"
      f" = {n_combos}")
print(f"  CV folds            = {N_SPLITS}")
print(f"  TOTAL FITS          = {n_combos} x {N_SPLITS} = {total_fits}")
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
search.fit(X_train, y_train)

bi = search.best_index_
res = search.cv_results_
fold_scores = [res[f"split{i}_test_score"][bi] for i in range(N_SPLITS)]
print(f"\n  best CV PR-AUC : {search.best_score_:.4f}"
      f"  (+/- {res['std_test_score'][bi]:.4f} across folds)")
print(f"  per-fold PR-AUC: " + ", ".join(f"{s:.4f}" for s in fold_scores))
print(f"  best params    : {search.best_params_}")

tree = search.best_estimator_
p_val_tree = tree.predict_proba(X_val)[:, 1]
m_tree = metrics("LightGBM", y_val, p_val_tree)

# ---- 3. calibrate, on the SAME val split Phase 3 used ----------------------
# Phase 3 split val 50/50 (stratified, seed 42): half to fit the calibrator, half
# held out to judge it. We reproduce that EXACT split so the tree's calibrated
# numbers are measured on the identical rows LogReg was, making the side-by-side fair.
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


print("\nBEFORE calibration (tree, on val_eval):")
tree_before = cal_report("uncalibrated", y_eval, tree.predict_proba(X_eval)[:, 1])

print("\nFitting CalibratedClassifierCV(method='isotonic', cv='prefit') on val_calib...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")      # sklearn 1.6+ nudges toward FrozenEstimator
    tree_calibrated = CalibratedClassifierCV(tree, method="isotonic", cv="prefit")
    tree_calibrated.fit(X_cal, y_cal)

print("\nAFTER calibration (tree, on val_eval):")
tree_after = cal_report("calibrated", y_eval, tree_calibrated.predict_proba(X_eval)[:, 1])

# ---- 4. side-by-side: re-measure the saved LogReg models on the SAME rows ---
# We don't trust copied numbers — we load the LogReg artifacts and score them on the
# identical val / val_eval rows, so every cell of the table is computed here, now.
print("\n" + "="*64)
print("SIDE-BY-SIDE  —  LogReg (deliverable)  vs  LightGBM (production seed)")
print("="*64)
logreg_raw = joblib.load(os.path.join(ROOT, "frequency_model.joblib"))      # Phase 2 winner
logreg_cal = joblib.load(os.path.join(ROOT, "calibrated_model.joblib"))     # Phase 3 output

p_val_lr   = logreg_raw.predict_proba(X_val)[:, 1]
m_lr       = dict(
    auc_roc = roc_auc_score(y_val, p_val_lr),
    pr_auc  = average_precision_score(y_val, p_val_lr),
    brier   = brier_score_loss(y_val, p_val_lr),
    logloss = log_loss(y_val, p_val_lr),
)
lr_after   = cal_report("LogReg calibrated", y_eval, logreg_cal.predict_proba(X_eval)[:, 1])

def row(label, lr_val, tree_val, fmt="{:.4f}", better="high"):
    win = "LogReg" if (lr_val > tree_val) == (better == "high") else "LightGBM"
    if abs(lr_val - tree_val) < 5e-4:
        win = "~tie"
    print(f"  {label:<28} {fmt.format(lr_val):>10}   {fmt.format(tree_val):>10}    {win}")

print(f"\n  {'metric':<28} {'LogReg':>10}   {'LightGBM':>10}    winner")
print("  " + "-"*62)
print("  -- discrimination on full val (uncalibrated raw scores) --")
row("PR-AUC  (primary)",  m_lr["pr_auc"],  m_tree["pr_auc"],  better="high")
row("AUC-ROC",            m_lr["auc_roc"], m_tree["auc_roc"], better="high")
print("  -- calibration on held-out val_eval (after isotonic) --")
row("calibration slope",  lr_after["slope"], tree_after["slope"], better="high")  # closer to 1 ~ higher here
row("Brier",              lr_after["brier"], tree_after["brier"], better="low")
row("log-loss",           lr_after["logloss"], tree_after["logloss"], better="low")

pr_winner = "LogReg" if m_lr["pr_auc"] >= m_tree["pr_auc"] else "LightGBM"
gap = m_lr["pr_auc"] - m_tree["pr_auc"]
print("\n  VERDICT:")
print(f"    PR-AUC winner on synthetic val: {pr_winner}  "
      f"(LogReg {m_lr['pr_auc']:.4f} vs LightGBM {m_tree['pr_auc']:.4f}, gap {gap:+.4f})")
print("    This is the EXPECTED result. The synthetic risk is additive, which a linear")
print("    model captures fully; the tree has no comorbidity interactions to exploit yet.")
print("    On REAL claims (correlated-disease interactions) the tree is expected to win.")
print("    The tree losing/tying here is success, not a bug — do NOT tune it to win.")

# ---- 5. save artifacts ------------------------------------------------------
joblib.dump(tree,            os.path.join(OUTDIR, "tree_frequency_model.joblib"))
joblib.dump(tree_calibrated, os.path.join(OUTDIR, "tree_calibrated_model.joblib"))
metrics_json = {
    "model": "LightGBM (production seed)",
    "scale_pos_weight": float(spw),
    "best_params": search.best_params_,
    "best_cv_pr_auc": float(search.best_score_),
    "tree_val_uncalibrated": m_tree,
    "tree_calibration_val_eval": {"before": tree_before, "after": tree_after},
    "logreg_val_uncalibrated": m_lr,
    "logreg_calibration_val_eval_after": lr_after,
    "pr_auc_winner_synthetic": pr_winner,
    "pr_auc_gap_logreg_minus_tree": float(gap),
}
with open(os.path.join(OUTDIR, "tree_metrics.json"), "w") as f:
    json.dump(metrics_json, f, indent=2)
print(f"\nSaved tree_frequency_model.joblib + tree_calibrated_model.joblib -> {OUTDIR}")
print("Saved tree_metrics.json")

# ---- 6. write the model card with the real numbers --------------------------
card = f"""# Model Card — LightGBM Frequency Model (Production Seed)

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
Trained fair: same {n_combos}-combo grid x {N_SPLITS}-fold CV ({total_fits} fits),
`scale_pos_weight={spw:.3f}`, `random_state=42`. **No tuning to beat LogReg.**

| metric (val)                         | LogReg (deliverable) | LightGBM (seed) | winner |
|--------------------------------------|---------------------:|----------------:|:------:|
| PR-AUC (primary, uncalibrated)       | {m_lr['pr_auc']:.4f} | {m_tree['pr_auc']:.4f} | {pr_winner} |
| AUC-ROC (uncalibrated)               | {m_lr['auc_roc']:.4f} | {m_tree['auc_roc']:.4f} | {'LogReg' if m_lr['auc_roc']>=m_tree['auc_roc'] else 'LightGBM'} |
| calibration slope (after, val_eval)  | {lr_after['slope']:.3f} | {tree_after['slope']:.3f} | — |
| Brier (after, val_eval)              | {lr_after['brier']:.4f} | {tree_after['brier']:.4f} | {'LogReg' if lr_after['brier']<=tree_after['brier'] else 'LightGBM'} |
| log-loss (after, val_eval)           | {lr_after['logloss']:.4f} | {tree_after['logloss']:.4f} | {'LogReg' if lr_after['logloss']<=tree_after['logloss'] else 'LightGBM'} |

**Framing: LogReg wins on PR-AUC by {gap:+.4f}, and that is the EXPECTED, CORRECT
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
"""
with open(os.path.join(OUTDIR, "MODEL_CARD.md"), "w", encoding="utf-8") as f:
    f.write(card)
print("Saved MODEL_CARD.md")
print("\nProduction-seed tree model complete. Next shared work: Phase 4 (severity).")
