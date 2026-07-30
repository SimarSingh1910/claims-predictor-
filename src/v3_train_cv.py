#!/usr/bin/env python3
"""
v3_train_cv.py  —  Step 4.2: cross-validation on train_real_v3 ONLY.

test_real_v3 IS NOT LOADED BY THIS FILE. Grep it: the path never appears.

Protocol is fixed by src/v3_prereg.md and is not re-derived here:
  * L2 LogisticRegression, class_weight='balanced', C over a pre-specified grid,
    selected on mean CV PR-AUC, exact ties broken toward SMALLER C.
  * RepeatedStratifiedKFold(5 folds x 10 repeats), random_state=42.
  * Imputation + scaling refit INSIDE each fold. The frozen preprocessor artifact
    is deliberately NOT used here: its scaler saw all 576 training rows, so using
    it would leak distributional information across folds and flatter every CV
    number. (The earlier 0.5699 / 0.6170 figures were computed that way and are
    therefore optimistic; expect these to come in lower.)
  * HistGradientBoosting as an explicitly EXPLORATORY secondary.
  * LogReg vs HistGB compared by PAIRED bootstrap on identical resamples — not
    by eyeballing two independent intervals.

Reported CV figures are SELECTION-OPTIMISTIC (C is tuned on the same folds that
report performance). Nested CV is not viable at 42 positives. Declared, not
removed — see prereg §1.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, log_loss)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "models", "real_v3")

TRAIN_CSV = r"C:\Users\PC\Downloads\train_real_v3.csv"
TARGET = "had_hospitalisation"
SEED = 42
N_SPLITS, N_REPEATS = 5, 10
N_BOOT = 2000
MIN_AGE = 18

# prereg §2 — fixed a priori on clinical grounds, never selected on the data.
FEATURES = ["age", "sex_male", "hba1c_percent", "systolic_bp_mmhg",
            "bmi", "egfr", "comorbidity_count"]
CONTINUOUS = ["age", "hba1c_percent", "systolic_bp_mmhg", "bmi", "egfr",
              "comorbidity_count"]
BINARY = ["sex_male"]

C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

# Clinically expected sign. A contradiction here is a prompt to check the data,
# NOT a finding to report.
EXPECTED_SIGN = {
    "age": +1, "hba1c_percent": +1, "systolic_bp_mmhg": +1, "bmi": +1,
    "egfr": -1,                      # higher eGFR = better renal function
    "comorbidity_count": +1,
    "sex_male": 0,                   # ambiguous a priori
}
# Clinically meaningful increment for the odds-ratio column.
INCREMENT = {
    "age": (10, "per +10 years"), "hba1c_percent": (1, "per +1 %"),
    "systolic_bp_mmhg": (10, "per +10 mmHg"), "bmi": (5, "per +5 kg/m²"),
    "egfr": (10, "per +10 mL/min/1.73m²"),
    "comorbidity_count": (1, "per +1 condition"),
    "sex_male": (1, "male vs female"),
}


def rule(t, ch="="):
    print("\n" + ch * 78); print(t); print(ch * 78)


def make_pipeline(model):
    """Preprocessing lives INSIDE the pipeline so it refits per fold."""
    return Pipeline([
        ("ct", ColumnTransformer([
            ("cont", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), CONTINUOUS),
            ("bin", SimpleImputer(strategy="constant", fill_value=0), BINARY),
        ], remainder="drop", verbose_feature_names_out=False)),
        ("model", model),
    ])


def logreg(C):
    # penalty='l2' is sklearn's DEFAULT and the explicit kwarg is deprecated in
    # 1.8+, so it is omitted rather than passed. The model is unchanged: still
    # L2-penalised, exactly as pre-registered.
    return LogisticRegression(C=C, class_weight="balanced",
                              solver="lbfgs", max_iter=2000, random_state=SEED)


def histgb():
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=100, learning_rate=0.05, min_samples_leaf=20,
        l2_regularization=1.0, random_state=SEED)


def metrics(y, p):
    return {
        "pr_auc": average_precision_score(y, p),
        "auc_roc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p, labels=[0, 1]),
    }


def oof_by_repeat(make_model, X, y):
    """Out-of-fold probabilities, one full 576-length vector PER REPEAT, plus the
    per-fold metrics. Returns (oof[n_repeats, n], fold_rows)."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=SEED)
    oof = np.full((N_REPEATS, len(y)), np.nan)
    folds = []
    for k, (tr, te) in enumerate(cv.split(X, y)):
        rep, fold = divmod(k, N_SPLITS)
        pipe = make_pipeline(make_model())
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        oof[rep, te] = p
        m = metrics(y.iloc[te].to_numpy(), p)
        m.update(repeat=rep, fold=fold, n=len(te), pos=int(y.iloc[te].sum()))
        folds.append(m)
    assert not np.isnan(oof).any(), "every row must be held out once per repeat"
    return oof, pd.DataFrame(folds)


def boot_ci(y, p, idx_sets, fn):
    """Percentile CI over pre-drawn stratified resamples (shared across models so
    comparisons are paired)."""
    vals = []
    for idx in idx_sets:
        try:
            vals.append(fn(y[idx], p[idx]))
        except ValueError:
            continue
    v = np.asarray(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), v


def make_resamples(y, n_boot, seed):
    """Stratified bootstrap indices, drawn ONCE and reused for every model."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return [np.concatenate([rng.choice(pos, len(pos), replace=True),
                            rng.choice(neg, len(neg), replace=True)])
            for _ in range(n_boot)]


def fmt(v, lo, hi, nd=4):
    return f"{v:.{nd}f}  [{lo:.{nd}f}, {hi:.{nd}f}]"


def main():
    df = pd.read_csv(TRAIN_CSV)
    n0 = len(df)
    df = df[(df["rel_self"] == 1) & (df["age"] >= MIN_AGE)].reset_index(drop=True)
    X, y = df[FEATURES], df[TARGET]
    base = y.mean()
    print(f"train_real_v3: {n0} -> {len(df)} rows after rel_self + age>={MIN_AGE}")
    print(f"positives: {int(y.sum())}   base rate: {base:.4%}")
    print(f"features ({len(FEATURES)}): {FEATURES}")
    print(f"events per feature: {y.sum()/len(FEATURES):.1f}")
    print(f"\ntest_real_v3 is NOT read by this script.")

    # ---- 1. C surface ------------------------------------------------------
    rule("1. C SURFACE — mean CV PR-AUC across the pre-specified grid")
    print(f"RepeatedStratifiedKFold({N_SPLITS} x {N_REPEATS}) = "
          f"{N_SPLITS*N_REPEATS} fits per C, seed {SEED}")
    print(f"preprocessing refit inside every fold\n")
    print(f"{'C':>8}{'mean PR-AUC':>14}{'SD across repeats':>20}{'mean AUC':>11}")
    print("-" * 78)
    surface = []
    for C in C_GRID:
        oof, _ = oof_by_repeat(lambda C=C: logreg(C), X, y)
        per_rep = [metrics(y.to_numpy(), oof[r])["pr_auc"] for r in range(N_REPEATS)]
        per_rep_auc = [metrics(y.to_numpy(), oof[r])["auc_roc"] for r in range(N_REPEATS)]
        surface.append({"C": C, "pr_auc": float(np.mean(per_rep)),
                        "sd": float(np.std(per_rep)), "auc": float(np.mean(per_rep_auc))})
        print(f"{C:>8}{np.mean(per_rep):>14.4f}{np.std(per_rep):>20.4f}"
              f"{np.mean(per_rep_auc):>11.4f}")

    surf = pd.DataFrame(surface)
    best_pr = surf["pr_auc"].max()
    tied = surf[surf["pr_auc"] == best_pr]
    best_C = float(tied["C"].min())          # prereg tie-break: smaller C wins
    spread = best_pr - surf["pr_auc"].min()
    print(f"\nbest mean CV PR-AUC = {best_pr:.4f} at C = {best_C}")
    print(f"exact ties at the maximum: {len(tied)}"
          + (f" -> tie-break to smallest C = {best_C}  ✓" if len(tied) > 1 else ""))
    print(f"surface spread (max - min) = {spread:.4f} "
          f"({'FLAT' if spread < 0.02 else 'not flat'})")
    within = surf[surf["pr_auc"] >= best_pr - surf["sd"].loc[surf['pr_auc'].idxmax()]]
    print(f"C values within 1 SD of the best: {sorted(within['C'].tolist())}")
    print(f"  (a 1-SE rule would have chosen C = {within['C'].min()}; NOT applied —")
    print(f"   the prereg specifies plain argmax with exact ties to smaller C)")
    print(f"baseline PR-AUC (prevalence) = {base:.4f}")

    # ---- 2. CV metrics at the selected C -----------------------------------
    rule(f"2. CV METRICS at C = {best_C}  (selection-optimistic, see prereg §1)")
    oof_lr, folds_lr = oof_by_repeat(lambda: logreg(best_C), X, y)
    oof_gb, folds_gb = oof_by_repeat(histgb, X, y)

    yv = y.to_numpy()
    resamples = make_resamples(yv, N_BOOT, SEED)
    FNS = {"pr_auc": average_precision_score, "auc_roc": roc_auc_score,
           "brier": brier_score_loss,
           "logloss": lambda a, b: log_loss(a, b, labels=[0, 1])}

    results = {}
    for label, oof in (("LogReg (primary)", oof_lr), ("HistGB (exploratory)", oof_gb)):
        p_mean = oof.mean(axis=0)          # average OOF prob across the 10 repeats
        print(f"\n--- {label} ---")
        print(f"{'metric':<12}{'point estimate [95% bootstrap CI]':<40}{'per-repeat range'}")
        res = {}
        for k, fn in FNS.items():
            pt = fn(yv, p_mean)
            lo, hi, _ = boot_ci(yv, p_mean, resamples, fn)
            per_rep = [fn(yv, oof[r]) for r in range(N_REPEATS)]
            print(f"{k:<12}{fmt(pt, lo, hi):<40}"
                  f"[{min(per_rep):.4f}, {max(per_rep):.4f}]")
            res[k] = {"point": pt, "ci": [lo, hi],
                      "per_repeat_min": min(per_rep), "per_repeat_max": max(per_rep)}
        results[label] = res
        if label.startswith("LogReg"):
            print(f"{'':12}baseline PR-AUC = {base:.4f} "
                  f"({'INSIDE the PR-AUC interval' if res['pr_auc']['ci'][0] <= base <= res['pr_auc']['ci'][1] else 'outside the interval'})")
            print(f"{'':12}AUC 0.50 "
                  f"({'INSIDE the AUC interval' if res['auc_roc']['ci'][0] <= 0.5 <= res['auc_roc']['ci'][1] else 'outside the interval'})")

    # ---- 3. fold-level spread ---------------------------------------------
    rule("3. FOLD-LEVEL SPREAD — 50 folds, ~8 positives each")
    for label, folds in (("LogReg", folds_lr), ("HistGB", folds_gb)):
        print(f"\n--- {label} ---")
        print(f"fold size: n={folds['n'].min()}-{folds['n'].max()}, "
              f"positives per fold: {folds['pos'].min()}-{folds['pos'].max()} "
              f"(median {int(folds['pos'].median())})")
        print(f"{'metric':<12}{'mean':>9}{'SD':>9}{'min':>9}{'p25':>9}"
              f"{'median':>9}{'p75':>9}{'max':>9}")
        for k in FNS:
            s = folds[k]
            print(f"{k:<12}{s.mean():>9.4f}{s.std():>9.4f}{s.min():>9.4f}"
                  f"{s.quantile(.25):>9.4f}{s.median():>9.4f}"
                  f"{s.quantile(.75):>9.4f}{s.max():>9.4f}")
    print("\nA single fold holds ~8 events; per-fold PR-AUC is correspondingly")
    print("unstable. The spread above is the honest picture of that instability.")

    # ---- 4. paired bootstrap ----------------------------------------------
    rule("4. PAIRED BOOTSTRAP — LogReg vs HistGB on identical resamples")
    p_lr, p_gb = oof_lr.mean(axis=0), oof_gb.mean(axis=0)
    print(f"{'metric':<12}{'difference (LogReg - HistGB) [95% CI]':<44}{'excludes 0?'}")
    paired = {}
    for k, fn in FNS.items():
        d = []
        for idx in resamples:
            try:
                d.append(fn(yv[idx], p_lr[idx]) - fn(yv[idx], p_gb[idx]))
            except ValueError:
                continue
        d = np.asarray(d)
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        excl = "YES" if (lo > 0 or hi < 0) else "no"
        print(f"{k:<12}{fmt(float(d.mean()), lo, hi):<44}{excl}")
        paired[k] = {"mean_diff": float(d.mean()), "ci": [float(lo), float(hi)],
                     "excludes_zero": excl == "YES"}
    print("\nPaired on the same resamples, so this is the difference's own interval —")
    print("not an eyeball comparison of two independent intervals.")

    # ---- 5. coefficients in clinical units ---------------------------------
    rule("5. COEFFICIENTS — final LogReg fit on all 576 training rows")
    pipe = make_pipeline(logreg(best_C))
    pipe.fit(X, y)
    ct = pipe.named_steps["ct"]
    lr = pipe.named_steps["model"]
    scaler = ct.named_transformers_["cont"].named_steps["scale"]
    names = list(ct.get_feature_names_out())
    coefs = lr.coef_.ravel()

    scale_map = {c: s for c, s in zip(CONTINUOUS, scaler.scale_)}
    print(f"{'feature':<20}{'β (std)':>10}{'β (clinical unit)':>19}"
          f"{'OR':>9}  {'increment':<26}{'sign'}")
    print("-" * 78)
    contradictions = []
    coef_out = {}
    for nm, b_std in zip(names, coefs):
        s = scale_map.get(nm, 1.0)
        b_unit = b_std / s
        inc, inc_label = INCREMENT[nm]
        odds = float(np.exp(b_unit * inc))
        exp_sign = EXPECTED_SIGN[nm]
        got = int(np.sign(b_unit))
        if exp_sign == 0:
            flag = "n/a"
        elif got == exp_sign:
            flag = "as expected"
        else:
            flag = "*** CONTRADICTS ***"
            contradictions.append(nm)
        print(f"{nm:<20}{b_std:>10.4f}{b_unit:>19.6f}{odds:>9.3f}  "
              f"{inc_label:<26}{flag}")
        coef_out[nm] = {"beta_standardised": float(b_std),
                        "beta_clinical_unit": float(b_unit),
                        "odds_ratio": odds, "increment": inc_label,
                        "expected_sign": exp_sign, "observed_sign": got,
                        "contradicts": flag.startswith("***")}
    print(f"{'intercept':<20}{lr.intercept_[0]:>10.4f}")

    if contradictions:
        print(f"\n*** {len(contradictions)} coefficient(s) contradict clinical "
              f"expectation: {contradictions}")
        print("Per prereg §2 this is a prompt to check for a data problem, NOT a")
        print("finding. At 42 events with correlated predictors, sign flips are")
        print("also an expected symptom of low power — do not interpret them")
        print("as protective effects.")
    else:
        print("\nAll coefficients with an a-priori expected sign match it.")

    # ---- 6. persist --------------------------------------------------------
    rule("6. PERSIST (CV artifacts only — no test data involved)")
    out = {
        "step": "4.2 cross-validation on train_real_v3 only",
        "n_rows": int(len(df)), "n_positives": int(y.sum()),
        "base_rate": float(base), "features": FEATURES,
        "events_per_feature": float(y.sum() / len(FEATURES)),
        "cv": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS, "seed": SEED,
               "preprocessing_refit_per_fold": True},
        "c_surface": surface, "selected_C": best_C,
        "selection_optimistic": True,
        "cv_metrics": results, "paired_bootstrap": paired,
        "coefficients": coef_out,
        "intercept": float(lr.intercept_[0]),
        "contradicting_signs": contradictions,
    }
    path = os.path.join(OUTDIR, "v3_cv_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")
    print("no model artifact serialised yet — that happens at 4.3 with the")
    print("calibrated final model, so nothing scoreable exists before approval.")


if __name__ == "__main__":
    main()
