#!/usr/bin/env python3
"""
score_json.py  —  Steps 2+3: score every real AHC user with the graceful scorer
and write real_predictions.json / .csv.

Reuses (no retraining): parse_real_json.parse_record + graceful_scorer.score_member
+ the existing joblibs + core_panel.json.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from parse_real_json import parse_record
from graceful_scorer import score_member

# ---- set the input path here ----
JSON_IN = r"C:\Users\Lenovo\Downloads\prod_smart_report.smart_report_details 11.json\prod_smart_report.smart_report_details 11.json"
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTDIR, exist_ok=True)

KEEP = ["user_code", "cibyl_score", "cibyl_label", "claim_probability",
        "panel_completeness_pct", "model_confidence_pct", "confidence_band", "scored", "reason"]

def main():
    print(f"Loading {JSON_IN} ...")
    data = json.load(open(JSON_IN, encoding="utf-8"))
    print(f"  {len(data)} raw records")

    parsed = [p for p in (parse_record(r) for r in data) if p is not None]
    print(f"  {len(parsed)} have a CIBYL score (rest skipped)")

    # split identity/CIBYL off; the remainder is the model-feature dict
    meta = [(p.pop("user_code"), p.pop("cibyl_score"), p.pop("cibyl_label")) for p in parsed]
    feat_df = pd.DataFrame(parsed)            # missing fields -> NaN (preserved)

    print("Scoring through graceful_scorer (batch)...")
    results = score_member(feat_df)           # list[dict], one per row

    rows = []
    for (uc, cs, cl), res in zip(meta, results):
        rows.append({
            "user_code": uc, "cibyl_score": cs, "cibyl_label": cl,
            "claim_probability": res["claim_probability"],
            "panel_completeness_pct": res["panel_completeness_pct"],
            "model_confidence_pct": res["model_confidence_pct"],
            "confidence_band": res["confidence_band"],
            "scored": res["scored"], "reason": res["reason"],
        })
    df = pd.DataFrame(rows, columns=KEEP)

    # ---- write outputs ----
    json_path = os.path.join(OUTDIR, "real_predictions.json")
    csv_path = os.path.join(OUTDIR, "real_predictions.csv")
    json.dump(df[[c for c in KEEP if c != "reason"]].to_dict(orient="records"),
              open(json_path, "w"), indent=2, default=lambda x: None if pd.isna(x) else x)
    df.to_csv(csv_path, index=False)

    # ---- summary ----
    scored = df[df.scored]
    refused = df[~df.scored]
    bands = df["confidence_band"].value_counts().to_dict()
    high = scored[scored.confidence_band == "HIGH"]
    rho, p = (spearmanr(high.cibyl_score, high.claim_probability)
              if len(high) > 2 else (float("nan"), float("nan")))

    print("\n" + "=" * 56)
    print("SUMMARY")
    print("=" * 56)
    print(f"  total users (with CIBYL) : {len(df)}")
    print(f"  scored                   : {len(scored)}")
    print(f"  refused (missing mand.)  : {len(refused)}")
    print(f"  confidence bands         : " +
          ", ".join(f"{k}={bands.get(k,0)}" for k in ["HIGH", "MEDIUM", "LOW"]))
    print(f"  mean claim_probability   : {scored.claim_probability.mean():.4f}")
    print(f"  Spearman(CIBYL, claim_p) : {rho:+.3f}  (HIGH-confidence n={len(high)}, p={p:.2e})")
    print(f"     interpretation        : negative => higher CIBYL (healthier) -> lower claim prob (expected)")
    print(f"\n  wrote {json_path}")
    print(f"  wrote {csv_path}")

if __name__ == "__main__":
    main()
