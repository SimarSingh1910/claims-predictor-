#!/usr/bin/env python3
"""
retrain_v2.py  —  rebuild the frequency pipeline on the COMPOUNDED target.

Phase 1 (preprocessor, same feature logic) + Phase 2 (LogReg vs tuned LightGBM)
+ interaction-augmented LogReg + Phase 3 (calibration). Saves into this folder.

The whole point: on the v1 ADDITIVE target, LogReg beat LightGBM (no interactions
to find). If the planted comorbidity interactions are now learnable, a tuned
LightGBM should BEAT LogReg on PR-AUC for claim_next_12m_compounded, and the
curated interaction terms should earn real coefficients in an augmented LogReg.
"""
import os, sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from scipy.stats import randint, uniform, loguniform
import lightgbm as lgb
import warnings

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import (ClinicalFeatureEngineer, BINARY_MAPS, ORDINAL_MAPS, CHRONIC_MAP)

V2_CSV = os.path.join(HERE, "healthbridge_ahc_v2_compounded_100k.csv")
TARGET = "claim_next_12m_compounded"
STRAT = "claim_next_12m"          # stratify on the additive target to reproduce v1's partition
# columns that are targets / leakage / audit -> never features
EXCLUDE = {"claim_next_12m", "claim_count_12m", "claim_amount_inr",
           "true_claim_propensity", "data_quality_flag",
           "claim_next_12m_compounded", "claim_amount_inr_compounded",
           "true_claim_propensity_compounded", "interaction_boost_logit",
           "n_comorbid_conditions", "severity_multiplier"}

# curated interaction combinations (for the augmented LogReg) — match the manifest
COMBOS = {
    "ix_hypothyroid_dys":      ["has_hypothyroidism", "has_dyslipidaemia"],
    "ix_obesity_htn_dys":      ["has_obesity", "has_hypertension", "has_dyslipidaemia"],
    "ix_diab_htn_dys":         ["has_diabetes", "has_hypertension", "has_dyslipidaemia"],
    "ix_diab_htn":             ["has_diabetes", "has_hypertension"],
    "ix_nafld_diab_obesity":   ["has_nafld", "has_diabetes", "has_obesity"],
    "ix_diab_dys":             ["has_diabetes", "has_dyslipidaemia"],
    "ix_diab_htn_ckd":         ["has_diabetes", "has_hypertension", "has_ckd"],
    "ix_diab_htn_dys_obesity": ["has_diabetes", "has_hypertension", "has_dyslipidaemia", "has_obesity"],
    "ix_hyperuric_htn_ckd":    ["has_hyperuricaemia", "has_hypertension", "has_ckd"],
    "ix_diab_ckd":             ["has_diabetes", "has_ckd"],
    "ix_htn_ckd":              ["has_hypertension", "has_ckd"],
    "ix_ckd_anaemia":          ["has_ckd", "has_anaemia"],
}

def metrics(y, p):
    return dict(pr_auc=average_precision_score(y, p), auc_roc=roc_auc_score(y, p),
                brier=brier_score_loss(y, p), logloss=log_loss(y, p))

# --------------------------------------------------------------------------
# split: reproduce v1 partition (stratify on additive target, seed 42)
# --------------------------------------------------------------------------
print("Loading v2 compounded dataset (features untouched)...")
df = pd.read_csv(V2_CSV)
feature_cols = [c for c in df.columns if c not in EXCLUDE]
assert len(feature_cols) == 115, f"expected 115 features, got {len(feature_cols)}"

train, temp = train_test_split(df, test_size=0.40, stratify=df[STRAT], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp[STRAT], random_state=SEED)
splitsdir = os.path.join(HERE, "splits"); os.makedirs(splitsdir, exist_ok=True)
keep = feature_cols + ["claim_next_12m", TARGET]
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[keep].to_csv(os.path.join(splitsdir, f"{name}.csv"), index=False)
print(f"  train {len(train)}  val {len(val)}  test {len(test)} (SEALED)")
print(f"  compounded claim rate: train {train[TARGET].mean():.4f}  val {val[TARGET].mean():.4f}")

X_train, y_train = train[feature_cols], train[TARGET].astype(int).values
X_val, y_val = val[feature_cols], val[TARGET].astype(int).values

# --------------------------------------------------------------------------
# Phase 1: preprocessor (same logic), fit on v2 train
# --------------------------------------------------------------------------
probe = ClinicalFeatureEngineer().fit(X_train)
allcols = probe.feature_names_
passthrough = set(BINARY_MAPS) | set(ORDINAL_MAPS) | set(CHRONIC_MAP.values())
passthrough |= {c for c in allcols if c.startswith("crystal_") or c.startswith("has_")}
passthrough |= {"psa_applicable", "hba1c_high", "hba1c_prediabetic", "bp_high",
                "bmi_obese", "bmi_overweight", "egfr_low", "ldl_high", "crp_elevated",
                "hb_low", "vitd_deficient"}
continuous = [c for c in allcols if c not in passthrough]
preprocessor = Pipeline([
    ("fe", ClinicalFeatureEngineer()),
    ("scale", ColumnTransformer([("num", StandardScaler(), continuous)],
                                remainder="passthrough", verbose_feature_names_out=False)),
])
preprocessor.set_output(transform="pandas")
Xt_train = preprocessor.fit_transform(X_train)
Xt_val = preprocessor.transform(X_val)
joblib.dump(preprocessor, os.path.join(HERE, "preprocessor.joblib"))

pos = y_train.sum(); spw = (len(y_train) - pos) / pos
print(f"\n  preprocessed -> {Xt_train.shape[1]} features  | scale_pos_weight {spw:.3f}")

# --------------------------------------------------------------------------
# Phase 2a: LogReg baseline
# --------------------------------------------------------------------------
logreg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED, n_jobs=-1)
logreg.fit(Xt_train, y_train)
m_lr = metrics(y_val, logreg.predict_proba(Xt_val)[:, 1])

# --------------------------------------------------------------------------
# Phase 2b: tuned LightGBM
# --------------------------------------------------------------------------
base = lgb.LGBMClassifier(objective="binary", scale_pos_weight=spw,
                          random_state=SEED, n_jobs=2, verbose=-1, subsample_freq=1)
dist = {"n_estimators": randint(300, 1100), "learning_rate": loguniform(0.01, 0.15),
        "max_depth": randint(3, 11), "num_leaves": randint(15, 120),
        "min_child_samples": randint(20, 200), "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4), "reg_alpha": loguniform(1e-3, 5),
        "reg_lambda": loguniform(1e-3, 5)}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
print("\n  tuning LightGBM (25 x 5-fold = 125 fits)...")
search = RandomizedSearchCV(base, dist, n_iter=25, scoring="average_precision",
                            cv=cv, n_jobs=2, random_state=SEED, refit=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    search.fit(Xt_train, y_train)
lgbm = search.best_estimator_
m_lgb = metrics(y_val, lgbm.predict_proba(Xt_val)[:, 1])

# --------------------------------------------------------------------------
# Phase 2c: interaction-augmented LogReg (do the curated terms earn coefficients?)
# --------------------------------------------------------------------------
def add_interactions(Xt):
    X = Xt.copy()
    for name, conds in COMBOS.items():
        if all(c in X.columns for c in conds):
            prod = np.ones(len(X))
            for c in conds:
                prod = prod * X[c].values
            X[name] = prod
    return X
Xi_train, Xi_val = add_interactions(Xt_train), add_interactions(Xt_val)
logreg_ix = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED, n_jobs=-1)
logreg_ix.fit(Xi_train, y_train)
m_lri = metrics(y_val, logreg_ix.predict_proba(Xi_val)[:, 1])
ix_coef = {name: float(logreg_ix.coef_[0][list(Xi_train.columns).index(name)])
           for name in COMBOS if name in Xi_train.columns}

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
print("\n" + "=" * 66)
print("PHASE 2 — frequency model on the COMPOUNDED target (val PR-AUC primary)")
print("=" * 66)
print(f"  {'model':28s}{'PR-AUC':>9}{'AUC-ROC':>9}{'Brier':>9}{'logloss':>9}")
for name, m in [("LogReg (baseline)", m_lr), ("LightGBM (tuned)", m_lgb),
                ("LogReg + interaction terms", m_lri)]:
    print(f"  {name:28s}{m['pr_auc']:>9.4f}{m['auc_roc']:>9.4f}{m['brier']:>9.4f}{m['logloss']:>9.4f}")
winner_name, winner, mw = max(
    [("LightGBM", lgbm, m_lgb), ("LogReg", logreg, m_lr)], key=lambda t: t[2]["pr_auc"])
delta = m_lgb["pr_auc"] - m_lr["pr_auc"]
print(f"\n  LightGBM - LogReg PR-AUC = {delta:+.4f}  -> "
      + ("LightGBM WINS: interactions became learnable (vs v1 where LogReg won)"
         if delta > 0 else "LightGBM did NOT overtake LogReg (report honestly)"))

print("\n  curated interaction coefficients in augmented LogReg (standardized):")
demo = {"ix_hypothyroid_dys","ix_obesity_htn_dys","ix_diab_htn_dys","ix_diab_htn",
        "ix_nafld_diab_obesity","ix_diab_dys"}
for name, c in sorted(ix_coef.items(), key=lambda kv: -kv[1]):
    tag = "demonstrator" if name in demo else "saturated"
    print(f"    {name:26s} coef {c:+.3f}   [{tag}]")
print(f"  augmented-LogReg PR-AUC {m_lri['pr_auc']:.4f} vs base LogReg {m_lr['pr_auc']:.4f} "
      f"({m_lri['pr_auc']-m_lr['pr_auc']:+.4f})")

# --------------------------------------------------------------------------
# Phase 3: calibrate the winner (isotonic, prefit; fit on half of val, judge on other)
# --------------------------------------------------------------------------
Xc, Xe, yc, ye = train_test_split(Xt_val, y_val, test_size=0.5, stratify=y_val, random_state=SEED)
before = metrics(ye, winner.predict_proba(Xe)[:, 1])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    calibrated = CalibratedClassifierCV(winner, method="isotonic", cv="prefit").fit(Xc, yc)
after = metrics(ye, calibrated.predict_proba(Xe)[:, 1])
print("\n" + "=" * 66)
print(f"PHASE 3 — calibration of winner ({winner_name}), on held-out val_eval")
print("=" * 66)
print(f"  Brier {before['brier']:.4f} -> {after['brier']:.4f}   "
      f"logloss {before['logloss']:.4f} -> {after['logloss']:.4f}")

# --------------------------------------------------------------------------
# Save artifacts
# --------------------------------------------------------------------------
joblib.dump(winner, os.path.join(HERE, "frequency_model.joblib"))
joblib.dump(calibrated, os.path.join(HERE, "calibrated_model.joblib"))
json.dump({"winner": winner_name, "scale_pos_weight": float(spw),
           "val_metrics": {"LogReg": m_lr, "LightGBM": m_lgb, "LogReg_interactions": m_lri},
           "lgbm_minus_logreg_pr_auc": float(delta),
           "best_lgbm_params": search.best_params_,
           "interaction_coefs": ix_coef,
           "calibration": {"before": before, "after": after}},
          open(os.path.join(HERE, "phase2_3_metrics_v2.json"), "w"), indent=2, default=float)
print(f"\nSaved v2: preprocessor.joblib, frequency_model.joblib ({winner_name}), "
      f"calibrated_model.joblib, phase2_3_metrics_v2.json")
print("v2 test split SEALED (not scored).")
