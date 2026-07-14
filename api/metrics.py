"""
metrics.py — read-only plumbing for per-model metrics shown in the UI.

The metrics MUST correspond to the ACTIVE p12 model. We do NOT recompute or
retrain anything — we read the already-existing metrics JSON each model wrote at
training time, and normalise the (differently-shaped) files into one response.

Per-BLOCK availability, honest to the honesty rule:
  * headline  — auc_roc / pr_auc / brier / logloss, calibrated + uncalibrated.
  * calibration — slope/intercept before & after isotonic (a curve summary; the
    files do not store per-point curve coordinates, so calibration.points is
    available:false rather than fabricated).
  * decile    — no model wrote a decile/lift table to its metrics file, so this
    block is available:false. We NEVER fall back to another model's numbers and
    NEVER invent a table.

If a model's metrics file is missing entirely -> the whole response is
available:false for that model.
"""
import json
import os

from .registry import ROOT

# Each model's on-disk metrics file + the keys that hold ITS OWN numbers.
METRICS_SOURCES = {
    "xgboost": {
        "path": "models/xgboost_production/phase2_3_metrics_xgb.json",
        "uncal_key": "xgb_val_uncalibrated",
        "cal_key": "xgb_calibration_val_eval",     # {before, after}
    },
    "lightgbm": {
        "path": "models/lightgbm_production/tree_metrics.json",
        "uncal_key": "tree_val_uncalibrated",
        "cal_key": "tree_calibration_val_eval",     # {before, after}
    },
    "logreg": {
        # v1 split its numbers across phase2 (uncalibrated) + phase3 (calibrated).
        "path": "models/v1_additive/phase2_metrics.json",
        "phase3_path": "models/v1_additive/phase3_metrics.json",
        "uncal_key": "val_metrics.LogReg",
        "cal_key": None,                            # taken from phase3 file
    },
}

_HEADLINE_FIELDS = ["auc_roc", "pr_auc", "brier", "logloss"]


def _dig(d, dotted):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _unavailable(model, reason):
    return {
        "model": model, "available": False, "reason": reason,
        "headline": {"available": False, "reason": reason},
        "calibration": {"available": False, "reason": reason},
        "decile": {"available": False, "reason": reason},
    }


def load_metrics(model):
    """Return the normalised metrics response for one model. available:false if
    its file is missing — never substitute another model's numbers."""
    src = METRICS_SOURCES.get(model)
    if src is None:
        return _unavailable(model, f"unknown model '{model}'")

    path = os.path.join(ROOT, src["path"])
    if not os.path.exists(path):
        return _unavailable(model, "metrics file not found")
    data = json.load(open(path))

    # -- headline: uncalibrated (its own) + calibrated (after isotonic) ----
    uncal = _dig(data, src["uncal_key"]) or {}
    if src.get("phase3_path"):                       # logreg: calibrated lives in phase3
        p3_path = os.path.join(ROOT, src["phase3_path"])
        cal_after = (json.load(open(p3_path)).get("after") if os.path.exists(p3_path) else None) or {}
        cal_before = None
    else:
        cal_block = _dig(data, src["cal_key"]) or {}
        cal_after = cal_block.get("after") or {}
        cal_before = cal_block.get("before")

    headline = {
        "available": bool(uncal or cal_after),
        "uncalibrated": {k: uncal.get(k) for k in _HEADLINE_FIELDS} if uncal else None,
        "calibrated": {k: cal_after.get(k) for k in _HEADLINE_FIELDS if k in cal_after} if cal_after else None,
    }

    # -- calibration: slope/intercept summary; per-point curve not stored -----
    calibration = {
        "summary": {
            "before": ({"slope": cal_before.get("slope"), "intercept": cal_before.get("intercept")}
                       if cal_before else None),
            "after": ({"slope": cal_after.get("slope"), "intercept": cal_after.get("intercept")}
                      if cal_after else None),
        },
        "points": {"available": False,
                   "reason": "per-point calibration curve not stored in metrics file"},
    }

    # -- decile / lift table: not written to any metrics file -----------------
    decile = {"available": False,
              "reason": "decile/lift table not stored in metrics file"}

    return {
        "model": model,
        "available": True,
        "source_file": src["path"],
        "headline": headline,
        "calibration": calibration,
        "decile": decile,
    }
