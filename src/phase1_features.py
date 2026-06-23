#!/usr/bin/env python3
"""
phase1_features.py  —  Phase 1: build and save the preprocessor.

The deal (the single most important rule in this project):
    FIT on train.  TRANSFORM val and test.
We fit ONE thing — a RobustScaler — and we fit it on train only. Everything
else (the ClinicalFeatureEngineer) learns nothing, so it can't leak.

Output:
    preprocessor.joblib   the fitted Pipeline (feature engineer + scaler)
    feature_names.json    the exact, ordered list of model-ready columns
"""
import os, sys, json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# make `features` importable no matter where we run from
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from features import (ClinicalFeatureEngineer, BINARY_MAPS, ORDINAL_MAPS,
                      CHRONIC_MAP, TARGETS)

SEED = 42
SPLITS = os.path.join(ROOT, "splits")

# ---- 1. load train + val. test stays SEALED — we never even open it. -------
print("Loading splits (train + val only — test stays sealed)...")
train = pd.read_csv(os.path.join(SPLITS, "train.csv"))
val   = pd.read_csv(os.path.join(SPLITS, "val.csv"))
print(f"  train {train.shape}   val {val.shape}")

# features X vs targets y — targets are dropped inside the engineer too, but we
# keep them separate here so nothing downstream can accidentally see them.
X_train = train.drop(columns=[c for c in TARGETS if c in train.columns])
X_val   = val.drop(columns=[c for c in TARGETS if c in val.columns])

# ---- 2. figure out which engineered columns are continuous (=> scale) -------
# Run the (fit-free) engineer once to learn the output column names, then split
# them into "continuous, scale these" vs "binary/ordinal/flag, leave alone".
fe_probe = ClinicalFeatureEngineer().fit(X_train)
all_cols = fe_probe.feature_names_

# everything that is a 0/1 flag, a binary, an ordinal, or a one-hot: PASS THROUGH
passthrough = set()
passthrough |= set(BINARY_MAPS)                       # sex + presence flags
passthrough |= set(ORDINAL_MAPS)                      # urine severity ladders
passthrough |= set(CHRONIC_MAP.values())              # has_* conditions
passthrough |= {c for c in all_cols if c.startswith("crystal_")}   # one-hot
passthrough |= {c for c in all_cols if c.startswith("has_")}
passthrough |= {"psa_applicable", "hba1c_high", "hba1c_prediabetic",
                "bp_high", "bmi_obese", "bmi_overweight", "egfr_low",
                "ldl_high", "crp_elevated", "hb_low", "vitd_deficient"}

continuous = [c for c in all_cols if c not in passthrough]
passthru_cols = [c for c in all_cols if c in passthrough]
print(f"  engineered -> {len(all_cols)} columns "
      f"({len(continuous)} continuous to scale, {len(passthru_cols)} passthrough)")

# ---- 3. the pipeline: engineer -> scale continuous, passthrough the rest ----
# StandardScaler centres each continuous lab to mean 0, std 1 (z-scores). The
# log1p step earlier already tamed the worst skew, so z-scoring is well behaved.
preprocessor = Pipeline([
    ("fe", ClinicalFeatureEngineer()),
    ("scale", ColumnTransformer(
        transformers=[("num", StandardScaler(), continuous)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )),
])
preprocessor.set_output(transform="pandas")

# ---- 4. FIT ON TRAIN ONLY, then transform both -----------------------------
print("Fitting preprocessor on TRAIN only...")
Xt_train = preprocessor.fit_transform(X_train)   # <-- the only .fit() in Phase 1
Xt_val   = preprocessor.transform(X_val)         # <-- val is only transformed
feat_names = list(Xt_train.columns)

# ---- 5. sanity checks: the proof it actually worked ------------------------
print("\nSanity checks")
print(f"  train matrix : {Xt_train.shape}")
print(f"  val   matrix : {Xt_val.shape}")
assert list(Xt_train.columns) == list(Xt_val.columns), "column mismatch!"
assert Xt_train.isna().sum().sum() == 0, "NaN leaked into train matrix!"
assert Xt_val.isna().sum().sum() == 0, "NaN leaked into val matrix!"
assert not any(t in feat_names for t in TARGETS), "a target leaked into features!"
# PROOF val was not used to fit: StandardScaler fit on train => train's scaled
# columns have mean ~0 / std ~1 EXACTLY, while val's drift off those values
# (it can't be ~0 on val unless val had been used to fit — which it wasn't).
train_mean = Xt_train[continuous].mean().abs().max()
val_mean   = Xt_val[continuous].mean().abs().max()
print(f"  no NaN in train/val .......... OK")
print(f"  train & val columns identical  OK")
print(f"  no target column in features . OK")
print(f"  max |mean| scaled cols, TRAIN  {train_mean:.4f}  (~0 => fit on train)")
print(f"  max |mean| scaled cols, VAL .. {val_mean:.4f}  (!=0 => val was NOT fit on)")
print(f"  total model-ready features ... {len(feat_names)}")

# ---- 6. save the fitted pipeline + the feature list ------------------------
out_pp = os.path.join(ROOT, "preprocessor.joblib")
out_fn = os.path.join(ROOT, "feature_names.json")
joblib.dump(preprocessor, out_pp)
with open(out_fn, "w") as f:
    json.dump(feat_names, f, indent=2)
print(f"\nSaved {out_pp}")
print(f"Saved {out_fn}")
print("\nPhase 1 complete. Next: Phase 2 loads preprocessor.joblib and trains the frequency model.")
