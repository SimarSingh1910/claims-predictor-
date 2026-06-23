#!/usr/bin/env python3
"""
derive_core_panel.py  —  Step 1+2 of the partial-input wrapper.

The IMPORTANT PANEL is CLINICALLY DEFINED by HCL — it is authoritative and is
NOT derived from model coefficients. The synthetic-trained model under-weights
things like electrolytes / vitamins / calcium / protein; that is an artifact of
the synthetic data, not of real clinical importance. So we take the clinical
list as ground truth, and we additionally record each feature's model |coef| so
core_panel.json shows BOTH facets:
    - important: does the clinical panel care about it?
    - model_coef: how much does the (synthetic) model currently weight it?
The gap between them (clinically important, model ~0) is EXPECTED and closes
when real claims data arrives. We surface it, we do not "fix" it.

Output: core_panel.json — one entry per of the 115 model features.
"""
import os, sys, json
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import features  # noqa: F401  (registers ClinicalFeatureEngineer for unpickling)

TARGETS = ["claim_next_12m", "claim_count_12m", "claim_amount_inr"]

# --------------------------------------------------------------------------
# AUTHORITATIVE clinical panel: model_column -> clinical group.
# NOT derived from coefficients. Provided by HCL clinical team.
# --------------------------------------------------------------------------
IMPORTANT_PANEL = {
    "haemoglobin_g_dl":          "Hemogram",
    "platelet_count_lakhs_cumm": "Hemogram",
    "total_wbc_cells_cumm":      "Hemogram",
    "rbc_million_cmm":           "Hemogram",
    "ldl_mg_dl":                 "Lipid Profile",
    "hdl_mg_dl":                 "Lipid Profile",
    "triglycerides_mg_dl":       "Lipid Profile",
    "total_cholesterol_mg_dl":   "Lipid Profile",
    "creatinine_mg_dl":          "Kidney Profile",
    "bun_mg_dl":                 "Kidney Profile",
    "uric_acid_mg_dl":           "Kidney Profile",
    "calcium_mg_dl":             "Kidney Profile",
    "sodium_meq_l":              "Electrolytes",
    "potassium_meq_l":           "Electrolytes",
    "chloride_meq_l":            "Electrolytes",
    "fbs_mg_dl":                 "Glucose",
    "hba1c_percent":             "Glucose",
    "alt_sgpt_u_l":              "Liver Function",
    "ast_sgot_u_l":              "Liver Function",
    "ggt_u_l":                   "Liver Function",
    "total_protein_g_dl":        "Liver Function",
    "albumin_g_dl":              "Liver Function",
    "alp_u_l":                   "Liver Function",
    "bilirubin_total_mg_dl":     "Liver Function",
    "vitamin_d_ng_ml":           "Vitamins",
    "vitamin_b12_pg_ml":         "Vitamins",
    "tsh_uiu_ml":                "Thyroid Profile",
    # CLINICAL CAVEAT: the clinical panel names FT3 / FT4 (FREE thyroid hormones),
    # but the model only has TOTAL T3 / T4 columns. We map FT3->total_t3_ng_dl and
    # FT4->total_t4_ug_dl: they fill the same diagnostic slot. Free vs total differ
    # clinically (free = unbound, active fraction) — FLAG FOR CORRECTION when real
    # data supplies free values.
    "total_t3_ng_dl":            "Thyroid Profile",
    "total_t4_ug_dl":            "Thyroid Profile",
    # CLINICAL CAVEAT: clinical list says hs-CRP (high-sensitivity); model column
    # is crp_mg_l. Same marker, mapped directly.
    "crp_mg_l":                  "Cardiac Risk Marker",
    # PSA is MALE-ONLY. For females it is NOT missing, it is not-applicable: the
    # Phase-1 psa_applicable flag handles this. Do NOT count PSA absence against a
    # female member's confidence (honoured in the confidence step, noted here).
    "psa_ng_ml":                 "Cancer Screening",
}

# --------------------------------------------------------------------------
# Step 2: MANDATORY MINIMUM — refuse to score if any is absent.
# Backbone glucose / renal / lipid markers + demographics. Without these the
# prediction is meaningless.
# --------------------------------------------------------------------------
MANDATORY_MINIMUM = ["age", "sex", "hba1c_percent", "fbs_mg_dl",
                     "creatinine_mg_dl", "total_cholesterol_mg_dl"]

# Demographics / anthropometry / vitals are handled SEPARATELY from the lab panel
# (the clinical list is lab-only):
#   - age + sex : mandatory, always present in an AHC record
#   - bmi       : derived from height_cm + weight_kg
#   - systolic/diastolic BP : important but often missing -> FLAG, do not fail
DEMO_VITALS = {
    "mandatory": ["age", "sex"],
    "derived": {"bmi": ["height_cm", "weight_kg"]},
    "important_but_often_missing": ["systolic_bp_mmhg", "diastolic_bp_mmhg"],
}

NEAR_ZERO = 0.05   # |coef| below this == "model effectively ignores it"

# --------------------------------------------------------------------------
# Load the 115 model features + the model's standardized |coef| per feature
# --------------------------------------------------------------------------
cols = pd.read_csv(os.path.join(ROOT, "splits", "train.csv"), nrows=1).columns.tolist()
model_features = [c for c in cols if c not in TARGETS]      # 115
assert len(model_features) == 115, f"expected 115, got {len(model_features)}"

model = joblib.load(os.path.join(ROOT, "frequency_model.joblib"))
feat_names = json.load(open(os.path.join(ROOT, "feature_names.json")))
coefs = np.ravel(model.coef_)
abscoef = {f: abs(float(c)) for f, c in zip(feat_names, coefs)}
# raw columns that explode into several engineered cols (no single direct coef):
#   chronic_disease -> has_*   |   urine_crystals -> crystal_*
# these are not in the clinical panel, so model_coef = null for them.

# sanity: every clinical-panel feature must exist as a direct model feature
for f in IMPORTANT_PANEL:
    assert f in abscoef, f"panel feature {f} has no direct model coef"

# --------------------------------------------------------------------------
# Build one entry per of the 115 features
# --------------------------------------------------------------------------
entries = []
for f in model_features:
    entries.append({
        "feature": f,
        "important": f in IMPORTANT_PANEL,
        "group": IMPORTANT_PANEL.get(f),                       # None if not in panel
        "model_coef": round(abscoef[f], 6) if f in abscoef else None,
    })

# --------------------------------------------------------------------------
# The expected gap: clinically important but model weight ~ 0
# --------------------------------------------------------------------------
ignored = [e for e in entries
           if e["important"] and (e["model_coef"] is not None) and e["model_coef"] < NEAR_ZERO]
ignored.sort(key=lambda e: e["model_coef"])

# --------------------------------------------------------------------------
# Console audit
# --------------------------------------------------------------------------
n_imp = sum(e["important"] for e in entries)
print("="*70)
print(f"IMPORTANT PANEL (clinically defined, authoritative) — {n_imp} lab features")
print("="*70)
by_group = {}
for e in entries:
    if e["important"]:
        by_group.setdefault(e["group"], []).append(e)
for g, items in by_group.items():
    print(f"\n  {g}")
    for e in sorted(items, key=lambda x: -x["model_coef"]):
        print(f"    {e['feature']:<28} |coef| {e['model_coef']:.3f}")

print("\n" + "="*70)
print("CLINICALLY IMPORTANT  but  MODEL WEIGHT ~ 0   (expected synthetic-data gap)")
print("="*70)
print("These matter clinically; the synthetic model barely uses them. Do NOT")
print("'fix' this — it closes once real claims data is trained on.")
for e in ignored:
    print(f"    {e['feature']:<28} {e['group']:<20} |coef| {e['model_coef']:.3f}")

print("\nMANDATORY MINIMUM (refuse to score if any absent):")
print("   ", MANDATORY_MINIMUM)
print("Demographics/vitals: age+sex mandatory | bmi <- height+weight | "
      "BP important-but-often-missing (flag, don't fail)")

# --------------------------------------------------------------------------
# Save core_panel.json
# --------------------------------------------------------------------------
out = {
    "_meta": {
        "description": "Per-feature importance for the partial-input wrapper. "
                       "IMPORTANT panel is CLINICALLY DEFINED (authoritative), not "
                       "coefficient-derived. model_coef shows the synthetic model's "
                       "current weight, recorded only to expose the clinical-vs-model gap.",
        "n_features": len(entries),
        "n_important": n_imp,
        "near_zero_threshold": NEAR_ZERO,
        "do_not_retrain": True,
    },
    "features": entries,                       # 115 entries: {feature, important, group, model_coef}
    "important_groups": {g: [e["feature"] for e in items] for g, items in by_group.items()},
    "mandatory_minimum": MANDATORY_MINIMUM,
    "demographics_and_vitals": DEMO_VITALS,
    "clinically_important_but_model_near_zero": [
        {"feature": e["feature"], "group": e["group"], "model_coef": e["model_coef"]}
        for e in ignored
    ],
    "special_handling": {
        "thyroid_free_vs_total": "Clinical panel names FT3/FT4 (free); model has total_t3_ng_dl/"
            "total_t4_ug_dl. Mapped to the same slot. Free vs total differ clinically — "
            "flag for correction when real data supplies free values.",
        "hs_crp": "Clinical 'hs-CRP' mapped directly to model column crp_mg_l.",
        "psa_male_only": "PSA is male-only structural missingness. For females it is "
            "not-applicable (Phase-1 psa_applicable flag), NOT missing. Do not count PSA "
            "absence against a female member's confidence.",
        "chronic_disease": "Raw chronic_disease explodes into has_* flags (the model's "
            "strongest features) but is absent from real AHC exports; model_coef=null here.",
    },
}
outpath = os.path.join(ROOT, "core_panel.json")
with open(outpath, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print(f"\nSaved {outpath}  ({len(entries)} feature entries, {n_imp} important)")
