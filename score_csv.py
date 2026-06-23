#!/usr/bin/env python3
"""
score_csv.py  —  Score members from a CSV against the Phase-3 calibrated model.

Usage:  py score_csv.py
Reads:  member_input.csv  (same columns as your training data, minus the 3 targets)
Prints: claim probability (next 12 months) + risk band, one line per row.

Edit member_input.csv freely — change any value, add rows, delete the examples.
Just keep the column names exactly as they are.
"""
import sys, os
import joblib, pandas as pd

# the preprocessor pickle references ClinicalFeatureEngineer in src/features.py,
# so src/ must be importable before we joblib.load() it
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

PREPROCESSOR = "preprocessor.joblib"
MODEL        = "calibrated_model.joblib"
INFILE       = "member_input.csv"
TARGETS      = ["claim_next_12m", "claim_count_12m", "claim_amount_inr"]

def band(p):
    if p > 0.60: return "VERY HIGH"
    if p > 0.30: return "HIGH"
    if p > 0.10: return "MODERATE"
    return "LOW"

def main():
    pre   = joblib.load(PREPROCESSOR)
    model = joblib.load(MODEL)
    df    = pd.read_csv(INFILE)

    # drop target columns if they happen to be present
    df = df.drop(columns=[c for c in TARGETS if c in df.columns])

    # keep a label column for readable output, if present
    labels = df["name"] if "name" in df.columns else pd.Series([f"row {i}" for i in range(len(df))])

    try:
        X = pre.transform(df)
    except Exception as e:
        # most common failure = column mismatch. Help the user fix it.
        print("\n[!] The preprocessor could not transform your CSV.")
        print("    This almost always means a column is missing, renamed, or extra.")
        print("    Error:", e)
        print("\n    Compare your columns against splits/val.csv — they must match")
        print("    (minus the 3 target columns). Don't rename or drop feature columns.")
        sys.exit(1)

    probs = model.predict_proba(X)[:, 1]

    print("\n" + "="*56)
    print(f"{'member':22s} {'claim prob (12m)':>16s}   risk")
    print("-"*56)
    for label, p in zip(labels, probs):
        print(f"{str(label):22s} {p:>15.1%}   {band(p)}")
    print("="*56)
    print("Note: probabilities are vs SYNTHETIC claims — directions are")
    print("meaningful, exact percentages are not real-world rates.\n")

if __name__ == "__main__":
    main()
