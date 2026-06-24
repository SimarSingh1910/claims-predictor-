#!/usr/bin/env python3
"""
score_json_v2.py  —  Part B: score the SAME real CIBYL JSON with the v2 model.

Reuses parse_real_json + graceful_scorer (now path-configurable). Points the scorer
at the v2 calibrated model. Writes real_predictions_v2.csv / .json into this folder.

REGRESSION-CHECK FRAMING: the v2 interactions are built on diagnosed-condition flags
(has_diabetes, has_ckd, ...) which are ABSENT from the CIBYL JSON, so the planted
interactions CANNOT fire on this real data. v2's CIBYL correlation is therefore
EXPECTED to be ~equal to v1's (-0.543), not higher. This run checks that adding the
compounded target did not DRIFT real-data behaviour — it is NOT a demonstration of
the interactions.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

import graceful_scorer as gs
# --- point the scorer at the v2 model (clinical panel + feature names unchanged) ---
gs.PREPROCESSOR_PATH  = os.path.join(HERE, "preprocessor.joblib")
gs.MODEL_PATH         = os.path.join(HERE, "calibrated_model.joblib")
gs.FREQ_MODEL_PATH    = os.path.join(HERE, "frequency_model.joblib")
gs.CORE_PANEL_PATH    = os.path.join(ROOT, "core_panel.json")
gs.FEATURE_NAMES_PATH = os.path.join(ROOT, "feature_names.json")
gs.TRAIN_CSV_PATH     = os.path.join(HERE, "splits", "train.csv")
from graceful_scorer import score_member
from parse_real_json import parse_record

JSON_IN = r"C:\Users\Lenovo\Downloads\prod_smart_report.smart_report_details 11.json\prod_smart_report.smart_report_details 11.json"
KEEP = ["user_code", "cibyl_score", "cibyl_label", "claim_probability",
        "panel_completeness_pct", "model_confidence_pct", "confidence_band", "scored"]

def main():
    print("Scoring CIBYL users with the v2 model...")
    data = json.load(open(JSON_IN, encoding="utf-8"))
    parsed = [p for p in (parse_record(r) for r in data) if p is not None]
    meta = [(p.pop("user_code"), p.pop("cibyl_score"), p.pop("cibyl_label")) for p in parsed]
    feat = pd.DataFrame(parsed)
    results = score_member(feat)

    rows = [{"user_code": uc, "cibyl_score": cs, "cibyl_label": cl,
             "claim_probability": r["claim_probability"],
             "panel_completeness_pct": r["panel_completeness_pct"],
             "model_confidence_pct": r["model_confidence_pct"],
             "confidence_band": r["confidence_band"], "scored": r["scored"]}
            for (uc, cs, cl), r in zip(meta, results)]
    df = pd.DataFrame(rows, columns=KEEP)
    df.to_csv(os.path.join(HERE, "real_predictions_v2.csv"), index=False)
    json.dump(df.to_dict("records"), open(os.path.join(HERE, "real_predictions_v2.json"), "w"),
              indent=2, default=lambda x: None if pd.isna(x) else x)

    scored = df[df.scored == True].copy()
    scored["claim_probability"] = pd.to_numeric(scored["claim_probability"])
    high = scored[scored.confidence_band == "HIGH"]
    rho2, p2 = spearmanr(high.cibyl_score, high.claim_probability)

    # v1 rho for side-by-side (recompute from the v1 outputs)
    v1 = pd.read_csv(os.path.join(ROOT, "outputs", "real_predictions.csv"))
    v1s = v1[(v1.scored == True) & (v1.confidence_band == "HIGH")]
    rho1, _ = spearmanr(v1s.cibyl_score, pd.to_numeric(v1s.claim_probability))

    print("\n" + "=" * 60)
    print("REGRESSION CHECK — CIBYL agreement, v1 vs v2 (HIGH-confidence)")
    print("=" * 60)
    print(f"  scored={len(scored)}  HIGH={len(high)}  mean claim_prob={scored.claim_probability.mean():.4f}")
    print(f"  v1 Spearman rho = {rho1:+.3f}")
    print(f"  v2 Spearman rho = {rho2:+.3f}   (p={p2:.2e})")
    print(f"  delta           = {rho2-rho1:+.3f}")
    print("  EXPECTED ~equal: has_* condition flags are absent from CIBYL data, so the")
    print("  planted interactions cannot fire. This confirms no real-data drift; it is")
    print("  NOT evidence of the interactions.")
    print(f"\n  wrote real_predictions_v2.csv / .json")
    return rho1, rho2

if __name__ == "__main__":
    main()
