#!/usr/bin/env python3
"""
phase3_calibration.py  —  Phase 3: make the probabilities mean what they say.

--------------------------------------------------------------------------
WHY AN UNCALIBRATED PROBABILITY BREAKS THE PRICING ENGINE
--------------------------------------------------------------------------
Phase 5 prices a group as  expected_cost = P(claim) x E(amount | claim),  summed
over members. That formula is only valid if P is a TRUE probability: if the model
says 0.30 for a thousand members, almost exactly 300 of them must actually claim.
A model can rank members perfectly (great PR-AUC) yet still be miscalibrated —
e.g. it says "0.30" for a group that really claims 0.20. Ranking is unaffected,
but every premium built on that 0.30 is 50% too high. Garbage probability in →
garbage premium out. Calibration is the step that makes P bankable.

--------------------------------------------------------------------------
WHAT ISOTONIC CALIBRATION DOES, IN PLAIN TERMS
--------------------------------------------------------------------------
It learns a mapping  raw_score -> corrected_probability  with exactly one rule:
the mapping can only go up, never down (monotonic / "isotonic"). It bins members
by raw score, looks at the actual claim rate in each bin, and stretches/squeezes
the scores so each output matches the observed rate — while preserving the
original ranking. No shape assumed (unlike Platt/sigmoid), so it can fix odd
S-curves; the price is it needs a few hundred+ samples or it overfits.

--------------------------------------------------------------------------
WHAT cv='prefit' MEANS AND WHY WE USE IT
--------------------------------------------------------------------------
Normally CalibratedClassifierCV retrains the base model K times on CV folds. We
DON'T want that — our LogReg is already trained and chosen in Phase 2. cv='prefit'
tells sklearn: "the estimator is already fitted, do NOT touch it; just learn the
calibration map on the data I hand you now." We hand it val data the model never
trained on — exactly the held-out signal calibration needs. Refitting would throw
away our tuned model and risk leaking train data into the calibration step.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")                       # headless: render to file, no window
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import features  # noqa: F401  (registers ClinicalFeatureEngineer for unpickling)

SPLITS = os.path.join(ROOT, "splits")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTDIR, exist_ok=True)
TARGET = "claim_next_12m"

# --------------------------------------------------------------------------
# 1. load model + val, preprocess val (test stays sealed)
# --------------------------------------------------------------------------
print("Loading frequency_model.joblib + preprocessor + val split...")
preprocessor = joblib.load(os.path.join(ROOT, "preprocessor.joblib"))
base_model   = joblib.load(os.path.join(ROOT, "frequency_model.joblib"))
val = pd.read_csv(os.path.join(SPLITS, "val.csv"))
y_val = val[TARGET].astype(int).values
X_val = preprocessor.transform(val)
print(f"  X_val {X_val.shape}   claim rate {y_val.mean():.4f}")

# Split val: HALF to fit the calibrator, HALF held out to judge it honestly.
X_cal, X_eval, y_cal, y_eval = train_test_split(
    X_val, y_val, test_size=0.50, stratify=y_val, random_state=SEED)
print(f"  val_calib {X_cal.shape[0]} rows (fit calibrator)   "
      f"val_eval {X_eval.shape[0]} rows (measure on this)")


def calibration_slope(y_true, p):
    """Slope of logit(observed) ~ logit(predicted). 1.0 = perfectly calibrated;
    <1 = over-confident (scores too extreme); >1 = under-confident."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs")   # ~unregularised
    lr.fit(logit, y_true)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def report(tag, y_true, p):
    slope, intercept = calibration_slope(y_true, p)
    brier = brier_score_loss(y_true, p)
    ll = log_loss(y_true, p)
    print(f"  [{tag}]  slope={slope:.3f}  intercept={intercept:+.3f}  "
          f"Brier={brier:.4f}  log-loss={ll:.4f}")
    return dict(slope=slope, intercept=intercept, brier=brier, logloss=ll)

# --------------------------------------------------------------------------
# 2. BEFORE: is calibration needed? (measure on the held-out eval half)
# --------------------------------------------------------------------------
print("\nBEFORE calibration (on val_eval):")
p_before = base_model.predict_proba(X_eval)[:, 1]
m_before = report("uncalibrated", y_eval, p_before)
need = abs(m_before["slope"] - 1.0) > 0.05
print(f"  -> slope deviates from 1.0 by {abs(m_before['slope']-1):.3f}  =>  "
      f"calibration {'IS NEEDED' if need else 'not strictly needed, applying anyway'}")

# --------------------------------------------------------------------------
# 3. CALIBRATE: isotonic, prefit (base model is left untouched)
# --------------------------------------------------------------------------
print("\nFitting CalibratedClassifierCV(method='isotonic', cv='prefit') on val_calib...")
with warnings.catch_warnings():
    # sklearn 1.6+ nudges toward FrozenEstimator; cv='prefit' still works and is
    # exactly what the phase asks for.
    warnings.simplefilter("ignore")
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)

# --------------------------------------------------------------------------
# 4. AFTER: re-measure on the SAME held-out eval half
# --------------------------------------------------------------------------
print("\nAFTER calibration (on val_eval):")
p_after = calibrated.predict_proba(X_eval)[:, 1]
m_after = report("calibrated", y_eval, p_after)

print("\nCALIBRATION SLOPE  before -> after :  "
      f"{m_before['slope']:.3f}  ->  {m_after['slope']:.3f}   (target = 1.000)")
print("Brier              before -> after :  "
      f"{m_before['brier']:.4f} -> {m_after['brier']:.4f}")

# --------------------------------------------------------------------------
# 5. reliability curve: before vs after, on the diagonal
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 7.0))
ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.3, label="perfect calibration")
for p, lab, col in [(p_before, f"before (slope {m_before['slope']:.2f})", "#d6604d"),
                    (p_after,  f"after  (slope {m_after['slope']:.2f})",  "#2c7fb8")]:
    frac_pos, mean_pred = calibration_curve(y_eval, p, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color=col, lw=2, ms=6, label=lab)
ax.set_xlabel("mean predicted probability (per bin)")
ax.set_ylabel("actual claim rate (per bin)")
ax.set_title("Reliability curve — isotonic calibration (measured on held-out val_eval)")
ax.legend(loc="upper left", frameon=True)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.grid(alpha=0.25)
curve_path = os.path.join(OUTDIR, "calibration_curve.png")
fig.tight_layout(); fig.savefig(curve_path, dpi=130); plt.close(fig)
print(f"\nSaved reliability curve -> {curve_path}")

# --------------------------------------------------------------------------
# 6. save the calibrated model (the exact artifact we just measured)
# --------------------------------------------------------------------------
out = os.path.join(ROOT, "calibrated_model.joblib")
joblib.dump(calibrated, out)
with open(os.path.join(ROOT, "phase3_metrics.json"), "w") as f:
    json.dump({"before": m_before, "after": m_after,
               "calibration_needed": bool(need)}, f, indent=2)
print(f"Saved {out}")
print("Saved phase3_metrics.json")
print("\nNOTE: calibrated_model wraps the prefit LogReg and consumes the SAME")
print("preprocessor output. Phase 5 pricing will use preprocessor -> calibrated_model.")
print("\nPhase 3 complete.")
