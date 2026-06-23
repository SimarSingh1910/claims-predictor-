#!/usr/bin/env python3
"""
phase2_frequency.py  —  Phase 2: the frequency model, P(claim in next 12m).

Pipeline:
    raw splits --(preprocessor.joblib)--> X  -->  classifier  -->  P(claim)

We RE-RUN the saved preprocessor on the raw splits instead of caching a
transformed CSV. The preprocessor is the single source of truth: re-running it
guarantees train/val are treated identically to how Phase 5/7 will treat new
data, and exercises the exact cold-start path the final product uses.

Two models:
    1. LogisticRegression(class_weight='balanced')   -- transparent baseline
    2. LightGBM(scale_pos_weight=neg/pos)             -- the real contender,
       tuned with 5-fold CV on TRAIN ONLY.
We keep whichever wins on PR-AUC.

--------------------------------------------------------------------------
WHY PR-AUC, NOT PLAIN AUC-ROC, AT 16% POSITIVES
--------------------------------------------------------------------------
AUC-ROC asks: "rank a random claimer above a random non-claimer?" Its x-axis
is the false-positive RATE, normalised by the 84% negatives — a huge, easy
pool. So a model can look great (0.8+) while still being wrong most times it
actually flags someone. AUC-ROC flatters you when classes are imbalanced.

PR-AUC (average precision) lives in the world we care about: of the members we
FLAG as likely to claim, what fraction truly claim (precision), across every
recall level. It never gets credit for the easy 84% of true negatives. The
no-skill baseline for PR-AUC is just the positive rate (~0.16), so any lift
above 0.16 is real signal. That is the honest yardstick for a 16%-positive
problem, which is why it's our PRIMARY metric.

--------------------------------------------------------------------------
WHY scale_pos_weight, NOT OVERSAMPLING
--------------------------------------------------------------------------
The imbalance fix is to make each of the rare positive rows COUNT MORE in the
loss, by neg/pos (~5.2x). scale_pos_weight does exactly that — it reweights the
gradient. Oversampling (e.g. SMOTE / duplicating positives) instead invents or
copies rows: it inflates the dataset, can leak duplicates across CV folds, and
distorts the base rate so the raw probabilities come out miscalibrated — which
is poison for Phase 3 (calibration) and Phase 5 (pricing, where the probability
must be a true probability). Reweighting keeps every real row exactly once and
leaves the data honest.
"""
import os, sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from math import prod
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss)
import lightgbm as lgb

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)              # so preprocessor's ClinicalFeatureEngineer resolves
import features  # noqa: F401         # registers the custom transformer class

SPLITS = os.path.join(ROOT, "splits")
TARGET = "claim_next_12m"

# ---- 1. load raw splits, run the Phase-1 preprocessor -----------------------
print("Loading preprocessor + raw splits (test stays sealed)...")
preprocessor = joblib.load(os.path.join(ROOT, "preprocessor.joblib"))
train = pd.read_csv(os.path.join(SPLITS, "train.csv"))
val   = pd.read_csv(os.path.join(SPLITS, "val.csv"))

y_train = train[TARGET].astype(int).values
y_val   = val[TARGET].astype(int).values

# preprocessor was already FIT in Phase 1 -> here we only TRANSFORM
X_train = preprocessor.transform(train)
X_val   = preprocessor.transform(val)
print(f"  X_train {X_train.shape}   X_val {X_val.shape}")
print(f"  claim rate -> train {y_train.mean():.4f}   val {y_val.mean():.4f}")

# imbalance ratio, computed on TRAIN only
pos = y_train.sum(); neg = len(y_train) - pos
spw = neg / pos
print(f"  scale_pos_weight = neg/pos = {neg}/{pos} = {spw:.3f}")


def metrics(name, y_true, p):
    """The four numbers the spec asks for, on validation."""
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


# ---- 2. STEP 1 — Baseline: Logistic Regression ------------------------------
print("\n" + "="*64)
print("STEP 1 — Baseline: LogisticRegression(class_weight='balanced')")
print("="*64)
logreg = LogisticRegression(
    class_weight="balanced", max_iter=2000, random_state=SEED, n_jobs=-1,
)
logreg.fit(X_train, y_train)
p_val_lr = logreg.predict_proba(X_val)[:, 1]
m_lr = metrics("LogReg", y_val, p_val_lr)

# ---- 3. STEP 2 — Main model: LightGBM, tuned 5-fold CV on TRAIN -------------
print("\n" + "="*64)
print("STEP 2 — LightGBM, scale_pos_weight, 5-fold CV tuned on TRAIN only")
print("="*64)
base = lgb.LGBMClassifier(
    objective="binary",
    scale_pos_weight=spw,        # the imbalance fix — reweight, don't resample
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
# Explicit discrete grid — wider than the first 12-combo attempt. Every combo
# is enumerated, so the fit count is exact and you can watch each [CV] line.
grid = {
    "n_estimators":  [300, 600],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth":     [4, 6, 8],
    "num_leaves":    [31, 63],
}
N_SPLITS = 5
n_combos = prod(len(v) for v in grid.values())
total_fits = n_combos * N_SPLITS

# --- show the size of the job BEFORE we start ---
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
    scoring="average_precision",   # PR-AUC, our primary metric, averaged over folds
    cv=cv,
    n_jobs=2,                      # cap parallelism — don't clone across every core
    refit=True,
    verbose=2,                     # print each fit as [CV n/total] ... completes
    return_train_score=False,
)
search.fit(X_train, y_train)

# --- best result + fold-to-fold spread for the winning combo ---
bi = search.best_index_
res = search.cv_results_
fold_scores = [res[f"split{i}_test_score"][bi] for i in range(N_SPLITS)]
print(f"\n  best CV PR-AUC : {search.best_score_:.4f}"
      f"  (+/- {res['std_test_score'][bi]:.4f} across folds)")
print(f"  per-fold PR-AUC: " + ", ".join(f"{s:.4f}" for s in fold_scores))
print(f"  fold spread    : min {min(fold_scores):.4f}  max {max(fold_scores):.4f}"
      f"  range {max(fold_scores)-min(fold_scores):.4f}")
print(f"  best params    : {search.best_params_}")

lgbm = search.best_estimator_
p_val_lgb = lgbm.predict_proba(X_val)[:, 1]
m_lgb = metrics("LightGBM", y_val, p_val_lgb)

# ---- 4. STEP 3 — Decile lift table on val (winning model) ------------------
# pick the winner by PR-AUC (primary metric)
if m_lgb["pr_auc"] >= m_lr["pr_auc"]:
    best_name, best_model, p_best = "LightGBM", lgbm, p_val_lgb
else:
    best_name, best_model, p_best = "LogReg", logreg, p_val_lr

print("\n" + "="*64)
print(f"STEP 3 — Decile lift table on val  (model: {best_name})")
print("="*64)
lift = pd.DataFrame({"p": p_best, "y": y_val})
# decile 10 = highest predicted risk. qcut on rank to handle ties cleanly.
lift["decile"] = pd.qcut(lift["p"].rank(method="first"), 10,
                         labels=range(1, 11)).astype(int)
tab = (lift.groupby("decile")
            .agg(members=("y", "size"),
                 mean_pred=("p", "mean"),
                 actual_claim_rate=("y", "mean"),
                 claims=("y", "sum"))
            .sort_index(ascending=False))
overall = y_val.mean()
tab["lift_vs_avg"] = tab["actual_claim_rate"] / overall
print(f"  overall val claim rate = {overall:.4f}\n")
print("  decile  members  mean_pred  actual_rate  claims  lift_vs_avg")
for d, r in tab.iterrows():
    print(f"    {d:>3}   {int(r.members):>6}    {r.mean_pred:6.3f}     "
          f"{r.actual_claim_rate:6.3f}     {int(r.claims):>5}    {r.lift_vs_avg:5.2f}x")
top, bot = tab.loc[10, "actual_claim_rate"], tab.loc[1, "actual_claim_rate"]
print(f"\n  top decile claims {top/bot:.1f}x as often as bottom decile "
      f"({top:.3f} vs {bot:.3f})")

# ---- 5. save the winning model ---------------------------------------------
out = os.path.join(ROOT, "frequency_model.joblib")
joblib.dump(best_model, out)
meta = {
    "winner": best_name,
    "scale_pos_weight": float(spw),
    "best_params": search.best_params_ if best_name == "LightGBM" else None,
    "val_metrics": {"LogReg": m_lr, "LightGBM": m_lgb},
}
with open(os.path.join(ROOT, "phase2_metrics.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nSaved {out}  (winner: {best_name})")
print("Saved phase2_metrics.json")
print("\nNOTE: raw probabilities here are reweighted, so they are NOT yet true")
print("probabilities. Phase 3 (calibration) fixes that before pricing.")
print("\nPhase 2 complete.")
