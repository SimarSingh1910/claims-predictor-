#!/usr/bin/env python3
"""
v3_sealed_eval.py  —  Step 4.3: THE single evaluation on test_real_v3.

    python src/v3_sealed_eval.py --dry-run    # fixture stand-in, no sealed read
    python src/v3_sealed_eval.py              # THE ONE REAL RUN

--dry-run swaps train_expanded_v3 in for the test file so every code path is
exercised before the sealed set is touched. That is what prereg §0 says the
fixture is for. Numbers from a dry run are meaningless and are labelled as such.

Protocol is fixed by src/v3_prereg.md. Nothing here re-derives it:
  * frozen models/real_v3/preprocessor_v3.joblib, fit on all of train_real,
    applied to test (§4.3);
  * primary LogReg C=10, sensitivity LogReg C=0.003 (§1.1), exploratory
    HistGradientBoosting (§3) — one pass, all three;
  * PR-AUC against the 9.27% TEST baseline (§4), not the 7.29% train baseline;
  * sigmoid calibration (§5);
  * predictions above age 51 suppressed, count reported (§6);
  * cohorts <30 / 30-39 / 40+, cells under n=30 suppressed (§7);
  * 2,000-iteration stratified bootstrap percentile CIs, resamples drawn once
    and SHARED so model comparisons are paired (§4).

NO RERUNS. If this breaks mid-evaluation, it gets logged in prereg §14 — it does
not get fixed and re-run.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, log_loss)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "models", "real_v3")
PRE_PATH = os.path.join(OUTDIR, "preprocessor_v3.joblib")

TRAIN_CSV = r"C:\Users\PC\Downloads\train_real_v3.csv"
TEST_CSV = r"C:\Users\PC\Downloads\test_real_v3.csv"
FIXTURE_CSV = r"C:\Users\PC\Downloads\train_expanded_v3.csv"

TARGET = "had_hospitalisation"
SEED = 42
N_BOOT = 2000
MIN_AGE = 18
MAX_VALIDATED_AGE = 51          # §6
MIN_CELL_N = 30                 # §7

FEATURES = ["age", "sex_male", "hba1c_percent", "systolic_bp_mmhg",
            "bmi", "egfr", "comorbidity_count"]
C_PRIMARY, C_SENSITIVITY = 10.0, 0.003

METRIC_FNS = {
    "pr_auc": average_precision_score,
    "auc_roc": roc_auc_score,
    "brier": brier_score_loss,
    "logloss": lambda y, p: log_loss(y, p, labels=[0, 1]),
}

BANDS = [(0, 30, "<30"), (30, 40, "30-39"), (40, 200, "40+")]


def rule(t, ch="="):
    print("\n" + ch * 78); print(t); print(ch * 78)


def band(a):
    for lo, hi, lab in BANDS:
        if lo <= a < hi:
            return lab
    return "unknown"


def clean(df):
    return df[(df["rel_self"] == 1) & (df["age"] >= MIN_AGE)].reset_index(drop=True)


def logreg(C):
    return LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                              max_iter=2000, random_state=SEED)


def histgb():
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=100, learning_rate=0.05, min_samples_leaf=20,
        l2_regularization=1.0, random_state=SEED)


def make_resamples(y, n_boot, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return [np.concatenate([rng.choice(pos, len(pos), replace=True),
                            rng.choice(neg, len(neg), replace=True)])
            for _ in range(n_boot)]


def ci(y, p, resamples, fn):
    vals = []
    for idx in resamples:
        try:
            vals.append(fn(y[idx], p[idx]))
        except ValueError:
            continue
    v = np.asarray(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def line(name, y, p, resamples, nd=4):
    out = {}
    print(f"{name}")
    for k, fn in METRIC_FNS.items():
        pt = fn(y, p)
        lo, hi = ci(y, p, resamples, fn)
        print(f"    {k:<10}{pt:.{nd}f}  [{lo:.{nd}f}, {hi:.{nd}f}]")
        out[k] = {"point": float(pt), "ci_low": lo, "ci_high": hi}
    return out


def main():
    dry = "--dry-run" in sys.argv
    eval_path = FIXTURE_CSV if dry else TEST_CSV
    mode = "DRY RUN (fixture stand-in — numbers meaningless)" if dry else \
           "SEALED EVALUATION — THE ONE PERMITTED RUN"

    rule(f"{mode}")
    print(f"evaluation file: {os.path.basename(eval_path)}")
    if dry:
        print("This does NOT consume the sealed read. Validating code paths only.")

    # ---- load ---------------------------------------------------------------
    bundle = joblib.load(PRE_PATH)
    pre, meta = bundle["preprocessor"], bundle["metadata"]
    print(f"\nfrozen preprocessor: {os.path.relpath(PRE_PATH, ROOT)}")
    print(f"  fitted on   : {meta['fitted_on']}  ({meta['n_fit_rows']} rows, "
          f"{meta['n_fit_positives']} positives)")
    print(f"  sklearn pin : {meta['sklearn_version']}")
    print(f"  gate        : INCLUDE_SYNTHETIC_LABS={meta['include_synthetic_labs']}")

    tr = clean(pd.read_csv(TRAIN_CSV))
    ev = clean(pd.read_csv(eval_path))
    in_feats, out_feats = meta["input_features"], meta["output_features"]
    sel = [out_feats.index(f) for f in FEATURES]

    Xtr = pre.transform(tr[in_feats])[:, sel]
    Xev = pre.transform(ev[in_feats])[:, sel]
    ytr, yev = tr[TARGET].to_numpy(), ev[TARGET].to_numpy()
    base = yev.mean()
    print(f"\ntrain: {Xtr.shape}  positives={int(ytr.sum())}  base={ytr.mean():.4%}")
    print(f"eval : {Xev.shape}  positives={int(yev.sum())}  base={base:.4%}")
    print(f"features ({len(FEATURES)}): {FEATURES}")

    # ---- §6 age suppression -------------------------------------------------
    rule("§6 AGE SUPPRESSION — model unvalidated above 51")
    over = ev["age"] > MAX_VALIDATED_AGE
    n_over = int(over.sum())
    print(f"train age range: {tr['age'].min():.0f}-{tr['age'].max():.0f}")
    print(f"eval  age range: {ev['age'].min():.0f}-{ev['age'].max():.0f}")
    print(f"eval rows above age {MAX_VALIDATED_AGE}: {n_over}  -> SUPPRESSED")
    if n_over:
        print(f"  ({n_over} prediction(s) withheld as extrapolation, not shown as a number)")
    else:
        print("  (none — the evaluation set contains no member beyond the")
        print("   validated range, so no prediction is withheld here. The")
        print("   restriction still binds at serving time.)")

    # ---- fit ---------------------------------------------------------------
    rule("FIT — on train only, then applied once to the evaluation set")
    models = {}
    m_pri = logreg(C_PRIMARY).fit(Xtr, ytr)
    m_sen = logreg(C_SENSITIVITY).fit(Xtr, ytr)
    m_gb = histgb().fit(Xtr, ytr)
    cal = CalibratedClassifierCV(logreg(C_PRIMARY), method="sigmoid", cv=5)
    cal.fit(Xtr, ytr)
    print(f"primary      LogReg C={C_PRIMARY}")
    print(f"sensitivity  LogReg C={C_SENSITIVITY}   (§1.1)")
    print(f"exploratory  HistGradientBoosting  (§3)")
    print(f"calibrated   sigmoid, cv=5 over the primary  (§5)")

    p = {
        "primary_uncal": m_pri.predict_proba(Xev)[:, 1],
        "primary_calibrated": cal.predict_proba(Xev)[:, 1],
        "sensitivity": m_sen.predict_proba(Xev)[:, 1],
        "histgb": m_gb.predict_proba(Xev)[:, 1],
    }
    resamples = make_resamples(yev, N_BOOT, SEED)

    # ---- §13.2 metrics ------------------------------------------------------
    rule(f"§13.2 METRICS — PR-AUC baseline = {base:.4%} (evaluation-set prevalence)")
    results = {}
    results["primary_calibrated"] = line(
        f"PRIMARY  LogReg C={C_PRIMARY}, sigmoid-calibrated  (§5)", yev,
        p["primary_calibrated"], resamples)
    results["primary_uncal"] = line(
        f"\nprimary, UNCALIBRATED (for transparency)", yev, p["primary_uncal"], resamples)
    results["sensitivity"] = line(
        f"\nSENSITIVITY  LogReg C={C_SENSITIVITY}, uncalibrated  (§1.1)", yev,
        p["sensitivity"], resamples)
    results["histgb"] = line(
        f"\nEXPLORATORY  HistGradientBoosting — NOT the deliverable  (§3)", yev,
        p["histgb"], resamples)

    print("\nNote: sigmoid is monotone, so PR-AUC and AUC-ROC are unchanged by")
    print("calibration EXCEPT for the cv=5 refit-and-average inside")
    print("CalibratedClassifierCV, which perturbs the ranking slightly. Brier and")
    print("log-loss are the metrics calibration is meant to move.")

    # ---- §9 triggers --------------------------------------------------------
    rule("§9 NULL-RESULT TRIGGERS")
    pri = results["primary_calibrated"]
    auc_null = pri["auc_roc"]["ci_low"] <= 0.50 <= pri["auc_roc"]["ci_high"]
    pr_null = pri["pr_auc"]["ci_low"] <= base <= pri["pr_auc"]["ci_high"]
    print(f"AUC-ROC interval [{pri['auc_roc']['ci_low']:.4f}, "
          f"{pri['auc_roc']['ci_high']:.4f}] includes 0.50 : {auc_null}")
    print(f"PR-AUC  interval [{pri['pr_auc']['ci_low']:.4f}, "
          f"{pri['pr_auc']['ci_high']:.4f}] includes {base:.4f} : {pr_null}")
    fired = auc_null or pr_null
    print(f"\n=> §9 framing applies: {fired}")

    # ---- primary vs sensitivity (§1.1) --------------------------------------
    rule("§1.1 PRIMARY vs SENSITIVITY — paired bootstrap on shared resamples")
    div = {}
    for k, fn in METRIC_FNS.items():
        d = []
        for idx in resamples:
            try:
                d.append(fn(yev[idx], p["primary_uncal"][idx])
                         - fn(yev[idx], p["sensitivity"][idx]))
            except ValueError:
                continue
        d = np.asarray(d)
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        excl = lo > 0 or hi < 0
        print(f"  {k:<10}{d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]   "
              f"excludes 0: {'YES' if excl else 'no'}")
        div[k] = {"mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi,
                  "excludes_zero": bool(excl)}
    material = any(v["excludes_zero"] for v in div.values())
    print(f"\nmaterial divergence: {material}")
    print("(If material, this is reported as a finding about coefficient")
    print(" instability at this sample size — §1.1 — not resolved by choosing")
    print(" whichever fit looks better.)")

    # ---- paired: primary vs HistGB (§4.1 caveat mandatory) ------------------
    rule("PRIMARY vs HistGB — paired bootstrap (§4.1 caveat is MANDATORY)")
    paired = {}
    for k, fn in METRIC_FNS.items():
        d = []
        for idx in resamples:
            try:
                d.append(fn(yev[idx], p["primary_uncal"][idx])
                         - fn(yev[idx], p["histgb"][idx]))
            except ValueError:
                continue
        d = np.asarray(d)
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        excl = lo > 0 or hi < 0
        print(f"  {k:<10}{d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]   "
              f"excludes 0: {'YES' if excl else 'no'}")
        paired[k] = {"mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi,
                     "excludes_zero": bool(excl)}
    print("\n§4.1 CAVEAT (not optional): the primary uses class_weight='balanced',")
    print("which pushes its probabilities toward 0.5; HistGB is unweighted so its")
    print("output sits near the base rate. Any Brier/log-loss gap compares a")
    print("reweighted model against an unweighted one — it is NOT a comparison of")
    print("probability estimators, and is what §5 calibration exists to remove.")

    # ---- §5 calibration assessment -----------------------------------------
    rule("§5 CALIBRATION — is a reliability curve estimable at this event count?")
    pc = p["primary_calibrated"]
    n_bins = 5
    edges = np.quantile(pc, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    print(f"{'bin':<6}{'n':>6}{'events':>8}{'mean pred':>12}{'observed':>11}")
    cal_rows = []
    estimable = True
    for i in range(n_bins):
        m = (pc > edges[i]) & (pc <= edges[i + 1])
        if not m.sum():
            continue
        ev_n = int(yev[m].sum())
        if ev_n < 10:
            estimable = False
        print(f"{i+1:<6}{int(m.sum()):>6}{ev_n:>8}{pc[m].mean():>12.4f}"
              f"{yev[m].mean():>11.4f}")
        cal_rows.append({"bin": i + 1, "n": int(m.sum()), "events": ev_n,
                         "mean_pred": float(pc[m].mean()),
                         "observed": float(yev[m].mean())})
    print(f"\nmean predicted overall: {pc.mean():.4f}   observed: {base:.4f}")
    if estimable:
        print("Every bin holds >= 10 events — a reliability curve is reportable.")
    else:
        print("At least one bin holds < 10 events. Per §5 the sigmoid-calibrated")
        print("probabilities are reported, but NO reliability curve is drawn: bins")
        print("this thin cannot support one and a curve would imply precision")
        print("the data does not contain.")

    # ---- §7 cohort table ----------------------------------------------------
    rule(f"§7 COHORT TABLE — bands <30 / 30-39 / 40+, cells under n={MIN_CELL_N} suppressed")
    ct = ev.copy()
    ct["_band"] = ct["age"].map(band)
    ct["_gender"] = np.where(ct["sex_male"] == 1, "M", "F")
    ct["_p"] = pc
    ct["_over"] = over.to_numpy()
    print(f"{'cohort':<16}{'n':>6}{'events':>8}{'observed':>11}"
          f"{'mean pred':>12}   status")
    print("-" * 78)
    cohorts = []
    for b_lab in [lab for _, _, lab in BANDS] + ["unknown"]:
        for g in ("F", "M"):
            sub = ct[(ct["_band"] == b_lab) & (ct["_gender"] == g)]
            if not len(sub):
                continue
            n = len(sub)
            # §6 binds INSIDE the cohort table too: a suppressed prediction must
            # not leak back in through a cohort average. Observed rate is a fact
            # about the data and keeps the full cell; mean predicted is a model
            # output and is computed over validated-age rows only.
            in_range = sub[~sub["_over"]]
            n_over_cell = int(sub["_over"].sum())
            supp = []
            if n < MIN_CELL_N:
                supp.append(f"SUPPRESSED n<{MIN_CELL_N}")
            if n_over_cell:
                supp.append(f"{n_over_cell} row(s) above age 51 excluded from mean pred")
            status = "; ".join(supp) if supp else "reportable"
            show = n >= MIN_CELL_N
            can_pred = show and len(in_range) > 0
            shown_rate = f"{sub[TARGET].mean():.2%}" if show else "—"
            shown_pred = f"{in_range['_p'].mean():.4f}" if can_pred else "—"
            print(f"{b_lab + ' · ' + g:<16}{n:>6}{int(sub[TARGET].sum()):>8}"
                  f"{shown_rate:>11}{shown_pred:>12}   {status}")
            cohorts.append({"band": b_lab, "gender": g, "n": n,
                            "events": int(sub[TARGET].sum()),
                            "observed": float(sub[TARGET].mean()) if show else None,
                            "mean_pred": float(in_range["_p"].mean()) if can_pred else None,
                            "n_used_for_mean_pred": int(len(in_range)) if can_pred else 0,
                            "suppressed": n < MIN_CELL_N,
                            "n_over_51": n_over_cell})
    n_rep = sum(1 for c in cohorts if not c["suppressed"])
    print(f"\n{n_rep} reportable cell(s), {len(cohorts)-n_rep} suppressed")

    # ---- coefficients (§2.1) ------------------------------------------------
    rule("COEFFICIENTS — primary, with §2.1 interpretation bars")
    for nm, b in zip(FEATURES, m_pri.coef_.ravel()):
        bar = ""
        if nm == "egfr":
            bar = "  <-- UNINTERPRETABLE (§2.1: R²=0.9507 on age+creatinine+sex)"
        elif nm == "hba1c_percent":
            bar = "  <-- UNINTERPRETABLE (§2.1 + §12.4 fill value, 29% at 5.4)"
        print(f"  {nm:<20}{b:+.4f}{bar}")
    print(f"  {'intercept':<20}{m_pri.intercept_[0]:+.4f}")

    # ---- persist ------------------------------------------------------------
    rule("PERSIST")
    out = {
        "step": "4.3 sealed evaluation",
        "dry_run": dry,
        "evaluation_file": os.path.basename(eval_path),
        "n_eval": int(len(ev)), "n_positives": int(yev.sum()),
        "base_rate": float(base),
        "features": FEATURES,
        "C_primary": C_PRIMARY, "C_sensitivity": C_SENSITIVITY,
        "age_suppression": {"max_validated_age": MAX_VALIDATED_AGE,
                            "n_suppressed": n_over},
        "metrics": results,
        "primary_vs_sensitivity": div, "material_divergence": bool(material),
        "primary_vs_histgb": paired,
        "null_triggers": {"auc_includes_0.5": bool(auc_null),
                          "pr_auc_includes_baseline": bool(pr_null),
                          "section9_applies": bool(fired)},
        "calibration": {"bins": cal_rows, "reliability_curve_estimable": bool(estimable),
                        "mean_predicted": float(pc.mean()), "observed": float(base)},
        "cohorts": cohorts,
        "coefficients": {nm: float(b) for nm, b in zip(FEATURES, m_pri.coef_.ravel())},
        "intercept": float(m_pri.intercept_[0]),
        "preprocessor_metadata": {k: meta[k] for k in
                                  ("sklearn_version", "numpy_version", "pandas_version",
                                   "include_synthetic_labs", "n_fit_rows")},
    }
    if dry:
        path = os.path.join(OUTDIR, "_dryrun_discard.json")
        print("DRY RUN — writing to a discardable path; no real artifact produced.")
    else:
        path = os.path.join(OUTDIR, "v3_sealed_eval_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")
    if not dry:
        joblib.dump({"model": cal, "uncalibrated": m_pri, "sensitivity": m_sen,
                     "features": FEATURES, "metadata": meta},
                    os.path.join(OUTDIR, "hospitalisation_p12_v3.joblib"))
        print(f"wrote {os.path.join(OUTDIR, 'hospitalisation_p12_v3.joblib')}")


if __name__ == "__main__":
    main()
