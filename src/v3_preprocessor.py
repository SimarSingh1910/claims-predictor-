#!/usr/bin/env python3
"""
v3_preprocessor.py  —  Step 3: fit the NEW preprocessor for the v3 real-data
schema and serialise it as a fresh artifact.

WHY A NEW PREPROCESSOR (and not a reuse of preprocessor.joblib):
The v3 files carry a 69-column schema of which only 26 columns overlap the
115-feature core panel the old pipeline was built for. The old artifact would
fail on this data. Worse, its ClinicalFeatureEngineer would RE-DERIVE features
(flags, ratios, eGFR) that are already baked into these CSVs — computing them
twice, from already-derived inputs. So: bypass the engineer entirely and do the
one thing that genuinely needs fitting — scaling raw clinical units, fit on
TRAIN ONLY.

WHAT THIS FITS
  * median imputation values  (continuous columns)   — fit on train
  * mean / std for scaling    (continuous columns)   — fit on train
Binary flags pass through unscaled so a logistic coefficient stays readable as
a log-odds contribution of the flag itself.

WHAT IT DELIBERATELY DOES NOT DO
  * no row dropping inside the transform — a fitted transformer must return one
    row out per row in. Invalid-row removal is a TRAINING-TIME step (clean_rows),
    applied before fitting; at inference the API refuses such members instead.
  * no feature selection — the preprocessor emits all 38 modelling columns. The
    reduction to 6-10 clinically-chosen features is a MODELLING decision and
    belongs to step 4, not to the schema layer.

THE 20 INJECTED COLUMNS
14 labs + 6 derived flags were bootstrap-sampled from empirical distributions and
are TARGET-INDEPENDENT NOISE. They exist for schema/pipeline validation only and
are excluded by INCLUDE_SYNTHETIC_LABS = False. A tree would happily split on them
and manufacture importance; that output is not interpretable and must never reach
a metric, a SHAP plot, a model card, or a cohort percentage.

Usage:  .venv\\Scripts\\python.exe src\\v3_preprocessor.py
Output: models/real_v3/preprocessor_v3.joblib   (+ schema JSON alongside)
"""
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "models", "real_v3")

DATA = {
    "train_expanded_v3": r"C:\Users\PC\Downloads\train_expanded_v3.csv",
    "train_real_v3":     r"C:\Users\PC\Downloads\train_real_v3.csv",
    "test_real_v3":      r"C:\Users\PC\Downloads\test_real_v3.csv",
}

TARGET = "had_hospitalisation"
SEED = 42

# ---- the gate ------------------------------------------------------------
INCLUDE_SYNTHETIC_LABS = False

SYNTHETIC_LABS = [
    "calcium_mg_dl", "sodium_mmol_l", "potassium_mmol_l", "chloride_mmol_l",
    "ggtp_u_l", "total_protein_g_dl", "albumin_g_dl", "alk_phosphatase_u_l",
    "vitamin_d_ng_ml", "vitamin_b12_pg_ml", "ft3_pg_ml", "ft4_ng_dl",
    "hs_crp_mg_l", "psa_ng_ml",
]
SYNTHETIC_FLAGS = [
    "flag_vitd_deficient", "flag_ggtp_high", "flag_hscrp_high",
    "flag_b12_low", "flag_alp_high", "flag_albumin_low",
]
SYNTHETIC_COLS = SYNTHETIC_LABS + SYNTHETIC_FLAGS

# Dependent one-hots: constant after the rel_self filter, so they carry no
# information and would be a rank-deficient column in a linear model.
REL_COLS = ["rel_self", "rel_spouse", "rel_parent"]
# No variance to learn from across all 11,113 rows.
DEAD_COLS = ["ix_dysglyc_renal", "ix_bp_renal", "ix_metabolic_triad", "flag_egfr_low"]
# Bookkeeping, not features.
NON_FEATURES = ["data_provenance", "is_synthetic", "policy_year", TARGET]

# Binary 0/1 columns — imputed with 0 and passed through UNSCALED.
BINARY_COLS = [
    "sex_male", "flag_hba1c_high", "flag_prediabetic", "flag_fbs_high",
    "flag_bp_high", "flag_bmi_obese", "flag_ldl_high", "flag_hb_low",
    "flag_urate_high", "flag_tg_high", "ix_obese_dysglyc",
]
BINARY_SYNTH = SYNTHETIC_FLAGS + ["psa_applicable"]

MIN_AGE = 18


# --------------------------------------------------------------------------
# Training-time row cleaning. NOT part of the fitted transform.
# --------------------------------------------------------------------------
def clean_rows(df, label=""):
    """Drop rows that cannot be a valid adult employee record. Reports what it
    removed rather than dropping silently."""
    n0 = len(df)
    bad_rel = int((df["rel_self"] != 1).sum())
    df = df[df["rel_self"] == 1]
    bad_age0 = int((df["age"] == 0).sum())
    bad_minor = int(((df["age"] > 0) & (df["age"] < MIN_AGE)).sum())
    df = df[df["age"] >= MIN_AGE]
    print(f"   {label:<20} {n0:>6} -> {len(df):>6}   "
          f"(dependents {bad_rel}, age==0 {bad_age0}, under-{MIN_AGE} {bad_minor})")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# psa_applicable routing (active only when INCLUDE_SYNTHETIC_LABS is True)
# --------------------------------------------------------------------------
def add_psa_applicable(df):
    """psa_ng_ml is missing on exactly the female rows — structurally, not by
    accident. Median-imputing it across sexes would invent a prostate reading for
    women. Instead we mark applicability with the existing Phase-1 `psa_applicable`
    convention and neutralise the value where it does not apply, so the model can
    learn "PSA matters only when applicable" rather than reading an imputed
    constant as signal."""
    out = df.copy()
    out["psa_applicable"] = (out["sex_male"] == 1).astype(int)
    out.loc[out["psa_applicable"] == 0, "psa_ng_ml"] = 0.0
    return out


def build_feature_lists(all_cols):
    """Resolve the modelling schema from the raw column list + the gate."""
    excluded = set(REL_COLS) | set(DEAD_COLS) | set(NON_FEATURES)
    if not INCLUDE_SYNTHETIC_LABS:
        excluded |= set(SYNTHETIC_COLS)
    feats = [c for c in all_cols if c not in excluded]
    # add_psa_applicable() may already have put the column on the frame, in which
    # case it is in all_cols — appending again would emit a duplicate column name
    # and ColumnTransformer would reject the frame.
    if INCLUDE_SYNTHETIC_LABS and "psa_applicable" not in feats:
        feats = feats + ["psa_applicable"]
    binary = [c for c in feats if c in set(BINARY_COLS) | set(BINARY_SYNTH)]
    continuous = [c for c in feats if c not in binary]
    return feats, continuous, binary


def build_preprocessor(continuous, binary):
    """median-impute + standardise the continuous columns; zero-impute and pass
    the binary flags through. Fit on TRAIN ONLY."""
    return Pipeline([
        ("ct", ColumnTransformer(
            transformers=[
                ("cont", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), continuous),
                ("bin", SimpleImputer(strategy="constant", fill_value=0), binary),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )),
    ])


def rule(t, ch="="):
    print("\n" + ch * 78); print(t); print(ch * 78)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    raw = {n: pd.read_csv(p) for n, p in DATA.items()}
    cols = list(raw["train_expanded_v3"].columns)
    raw = {n: df[cols] for n, df in raw.items()}     # canonical column order

    rule("1. ROW CLEANING (training-time only — not part of the transform)")
    frames = {n: clean_rows(df, n) for n, df in raw.items()}

    if INCLUDE_SYNTHETIC_LABS:
        frames = {n: add_psa_applicable(df) for n, df in frames.items()}
        cols = list(frames["train_expanded_v3"].columns)

    feats, continuous, binary = build_feature_lists(cols)

    rule("2. OUTPUT SCHEMA")
    print(f"INCLUDE_SYNTHETIC_LABS = {INCLUDE_SYNTHETIC_LABS}")
    print(f"raw columns in file        : {len(cols)}")
    print(f"  - dependent one-hots     : {len(REL_COLS)}  {REL_COLS}")
    print(f"  - dead features          : {len(DEAD_COLS)}  {DEAD_COLS}")
    print(f"  - non-features           : {len(NON_FEATURES)}  {NON_FEATURES}")
    if not INCLUDE_SYNTHETIC_LABS:
        print(f"  - injected synthetic     : {len(SYNTHETIC_COLS)}  (14 labs + 6 flags) GATED OFF")
    else:
        print(f"  + psa_applicable         : routed (psa zeroed where not applicable)")
    print(f"\nMODELLING FEATURES OUT     : {len(feats)}")
    print(f"  continuous (imputed+scaled): {len(continuous)}")
    for c in continuous:
        print(f"      {c}")
    print(f"  binary (imputed, unscaled) : {len(binary)}")
    for c in binary:
        print(f"      {c}")

    # ---- FIT ON TRAIN ONLY -------------------------------------------------
    rule("3. FIT — train_real_v3 only (test stays sealed)")
    tr = frames["train_real_v3"]
    pre = build_preprocessor(continuous, binary)
    pre.fit(tr[feats])
    out_names = list(pre.named_steps["ct"].get_feature_names_out())
    print(f"fitted on {len(tr)} rows x {len(feats)} features")
    print(f"transform output width: {len(out_names)}")
    assert out_names == continuous + binary, "feature-name order drifted"
    print("output column order == continuous + binary  ✓")

    Xtr = pre.transform(tr[feats])
    print(f"train transform: shape={Xtr.shape}  NaNs={int(np.isnan(Xtr).sum())}  "
          f"finite={bool(np.isfinite(Xtr).all())}")

    # ---- TRANSFORM DIFF ----------------------------------------------------
    rule("4. TRAIN vs TEST TRANSFORM DIFF")
    print("Only FEATURE columns of test_real_v3 are read here — the target is never")
    print("touched, so the sealed single-evaluation rule is intact.\n")
    te = frames["test_real_v3"]
    Xte = pre.transform(te[feats])
    print(f"test transform : shape={Xte.shape}  NaNs={int(np.isnan(Xte).sum())}  "
          f"finite={bool(np.isfinite(Xte).all())}")
    print(f"\nTrain is centred by construction (scaler fit on it). Any drift in the")
    print(f"test column stats IS the out-of-time covariate shift.\n")
    print(f"{'feature':<26}{'tr mean':>9}{'tr std':>8}{'te mean':>9}{'te std':>8}{'shift':>8}")
    print("-" * 78)
    rows = []
    for i, c in enumerate(out_names):
        a, b = Xtr[:, i], Xte[:, i]
        shift = abs(b.mean() - a.mean())
        rows.append((shift, c, a.mean(), a.std(), b.mean(), b.std()))
    for shift, c, am, asd, bm, bsd in sorted(rows, reverse=True):
        mark = "  <-- large" if shift > 0.25 else ""
        print(f"{c:<26}{am:>9.3f}{asd:>8.3f}{bm:>9.3f}{bsd:>8.3f}{shift:>8.3f}{mark}")
    big = [c for s, c, *_ in rows if s > 0.25]
    print(f"\n{len(big)} feature(s) shifted > 0.25 SD between train and test: {big}")

    # out-of-range check: values the scaler never saw
    rule("5. RANGE CHECK — test values outside the train range", "-")
    n_out = 0
    for i, c in enumerate(out_names):
        lo, hi = Xtr[:, i].min(), Xtr[:, i].max()
        k = int(((Xte[:, i] < lo) | (Xte[:, i] > hi)).sum())
        if k:
            n_out += 1
            print(f"   {c:<26} {k:>4}/{len(Xte)} test rows outside train range "
                  f"[{lo:.2f}, {hi:.2f}]")
    if not n_out:
        print("   none — every test value falls inside the train range")

    # ---- SERIALISE ---------------------------------------------------------
    rule("6. SERIALISE")
    meta = {
        "artifact": "preprocessor_v3",
        "schema_version": "v3",
        "target": TARGET,
        "fitted_on": "train_real_v3.csv (rel_self==1, age>=18)",
        "n_fit_rows": int(len(tr)),
        "n_fit_positives": int(tr[TARGET].sum()),
        "base_rate_fit": float(tr[TARGET].mean()),
        "include_synthetic_labs": INCLUDE_SYNTHETIC_LABS,
        "synthetic_cols_gated": SYNTHETIC_COLS,
        "input_features": feats,
        "output_features": out_names,
        "continuous": continuous,
        "binary_passthrough": binary,
        "dropped_rel": REL_COLS,
        "dropped_dead": DEAD_COLS,
        "min_age": MIN_AGE,
        "random_state": SEED,
        # Pinned so the InconsistentVersionWarning situation does not repeat.
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
        "python_version": platform.python_version(),
    }
    bundle = {"preprocessor": pre, "metadata": meta}
    path = os.path.join(OUTDIR, "preprocessor_v3.joblib")
    joblib.dump(bundle, path)
    with open(os.path.join(OUTDIR, "preprocessor_v3_schema.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {path}")
    print(f"wrote {os.path.join(OUTDIR, 'preprocessor_v3_schema.json')}")
    print(f"legacy preprocessor.joblib: UNTOUCHED  ✓")
    print(f"pinned: sklearn {meta['sklearn_version']}, numpy {meta['numpy_version']}, "
          f"pandas {meta['pandas_version']}, python {meta['python_version']}")

    # ---- ROUND-TRIP SMOKE TEST on the fixture ------------------------------
    rule("7. SMOKE TEST — reload + transform the expanded fixture")
    again = joblib.load(path)
    pre2, meta2 = again["preprocessor"], again["metadata"]
    fx = frames["train_expanded_v3"]
    Xfx = pre2.transform(fx[meta2["input_features"]])
    print(f"reloaded artifact; fixture transform shape={Xfx.shape}  "
          f"NaNs={int(np.isnan(Xfx).sum())}  finite={bool(np.isfinite(Xfx).all())}")
    same = np.allclose(pre2.transform(tr[feats]), Xtr, equal_nan=True)
    print(f"reloaded transform reproduces the in-memory one: {same}  "
          f"{'✓' if same else 'FAIL'}")
    print("\n(fixture used for shape/round-trip validation ONLY — never for metrics)")


if __name__ == "__main__":
    main()
