#!/usr/bin/env python3
"""
graceful_scorer.py  —  Step 3: score a member from WHATEVER fields are available.

Wraps the EXISTING preprocessor.joblib + calibrated_model.joblib (no retraining).
Real AHC exports arrive partial (the CIBYL file lacked ~43 fields incl. BP), so
this layer:
  1. refuses if the MANDATORY_MINIMUM is absent (prediction would be meaningless),
  2. DERIVES computable values (eGFR, BMI, eAG, lipid ratios) from real inputs,
  3. IMPUTES everything else from train medians/modes (low-weight -> score barely moves),
  4. reports TWO confidence numbers:
       panel_completeness_pct : % of the clinical IMPORTANT_PANEL actually present,
       model_confidence_pct   : coef-weighted share of present features (model trust).

A value is "present" only if it was PROVIDED or DERIVED from real inputs — an
imputed median does NOT count as present.
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
IDS = ["CUG", "employee_id", "name", "ahc_date"]

# Map each engineered (coef-bearing) feature -> the RAW input column(s) it needs.
# Kept in sync with features.py transform(). Anything not listed here is a raw
# column that passes through under its own name (source = {itself}).
_ABNORMAL_FLAG_SOURCES = {
    "hba1c_high": {"hba1c_percent"}, "hba1c_prediabetic": {"hba1c_percent"},
    "bp_high": {"systolic_bp_mmhg", "diastolic_bp_mmhg"},
    "bmi_obese": {"bmi"}, "bmi_overweight": {"bmi"},
    "egfr_low": {"egfr_ml_min_173m2"}, "ldl_high": {"ldl_mg_dl"},
    "crp_elevated": {"crp_mg_l"}, "hb_low": {"haemoglobin_g_dl"},
    "vitd_deficient": {"vitamin_d_ng_ml"},
}

# --------------------------------------------------------------------------
# Resource loading (cached once per process)
# --------------------------------------------------------------------------
_R = {}
def _resources():
    if _R:
        return _R
    panel = json.load(open(os.path.join(ROOT, "core_panel.json")))
    important = {e["feature"]: e["group"] for e in panel["features"] if e["important"]}
    model_features = [e["feature"] for e in panel["features"]]   # the 115, in order

    pre = joblib.load(os.path.join(ROOT, "preprocessor.joblib"))
    model = joblib.load(os.path.join(ROOT, "calibrated_model.joblib"))
    feat_names = json.load(open(os.path.join(ROOT, "feature_names.json")))
    base = joblib.load(os.path.join(ROOT, "frequency_model.joblib"))
    abscoef = {f: abs(float(c)) for f, c in zip(feat_names, np.ravel(base.coef_))}

    # engineered feature -> source raw columns
    eng_sources = []
    for f in feat_names:
        if f.startswith("has_"):
            src = {"chronic_disease"}
        elif f.startswith("crystal_"):
            src = {"urine_crystals"}
        elif f in _ABNORMAL_FLAG_SOURCES:
            src = _ABNORMAL_FLAG_SOURCES[f]
        elif f == "psa_applicable":
            src = {"sex"}
        else:
            src = {f}                       # passthrough / scaled / ordinal / binary
        eng_sources.append((f, abscoef.get(f, 0.0), src))
    total_abscoef = sum(abscoef.values())

    # train medians (numeric) and modes (object) for imputation
    train = pd.read_csv(os.path.join(ROOT, "splits", "train.csv"))
    train = train.drop(columns=[c for c in TARGETS if c in train.columns])
    num_cols = train.select_dtypes(include=["number"]).columns.tolist()
    obj_cols = train.select_dtypes(exclude=["number"]).columns.tolist()
    medians = train[num_cols].median().to_dict()
    modes = {c: train[c].mode(dropna=True).iloc[0] for c in obj_cols}

    _R.update(dict(panel=panel, important=important, model_features=model_features,
                   mandatory=panel["mandatory_minimum"], pre=pre, model=model,
                   eng_sources=eng_sources, total_abscoef=total_abscoef,
                   num_cols=set(num_cols), obj_cols=set(obj_cols),
                   medians=medians, modes=modes))
    return _R


# --------------------------------------------------------------------------
# Deterministic clinical derivations — fill a target only if it's missing AND
# its inputs are present. Derived values count as REAL (not imputed).
# --------------------------------------------------------------------------
def _derive(frame):
    f = frame
    def have(*cols):  # boolean mask: all cols present (non-null) this row
        m = pd.Series(True, index=f.index)
        for c in cols:
            m &= f[c].notna()
        return m
    def fill(col, mask, values):
        need = f[col].isna() & mask
        f.loc[need, col] = values[need]

    if {"height_cm", "weight_kg"} <= set(f.columns):
        fill("bmi", have("height_cm", "weight_kg"),
             f["weight_kg"] / (f["height_cm"] / 100.0) ** 2)
    # CKD-EPI 2021 (race-free) eGFR from creatinine, age, sex
    if {"creatinine_mg_dl", "age", "sex"} <= set(f.columns):
        scr = f["creatinine_mg_dl"]; age = f["age"]; female = f["sex"].astype(str).str.upper().eq("F")
        kappa = np.where(female, 0.7, 0.9); alpha = np.where(female, -0.241, -0.302)
        ratio = scr / kappa
        egfr = (142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** (-1.200)
                * 0.9938 ** age * np.where(female, 1.012, 1.0))
        fill("egfr_ml_min_173m2", have("creatinine_mg_dl", "age", "sex"), pd.Series(egfr, index=f.index))
    if "hba1c_percent" in f.columns:                       # ADAG: eAG = 28.7*A1c - 46.7
        fill("estimated_avg_glucose_mg_dl", have("hba1c_percent"), 28.7 * f["hba1c_percent"] - 46.7)
    if {"total_cholesterol_mg_dl", "hdl_mg_dl"} <= set(f.columns):
        fill("non_hdl_cholesterol_mg_dl", have("total_cholesterol_mg_dl", "hdl_mg_dl"),
             f["total_cholesterol_mg_dl"] - f["hdl_mg_dl"])
        fill("cho_hdl_ratio", have("total_cholesterol_mg_dl", "hdl_mg_dl"),
             f["total_cholesterol_mg_dl"] / f["hdl_mg_dl"])
    if "triglycerides_mg_dl" in f.columns:
        fill("vldl_mg_dl", have("triglycerides_mg_dl"), f["triglycerides_mg_dl"] / 5.0)
    if {"ldl_mg_dl", "hdl_mg_dl"} <= set(f.columns):
        fill("ldl_hdl_ratio", have("ldl_mg_dl", "hdl_mg_dl"), f["ldl_mg_dl"] / f["hdl_mg_dl"])
    if {"triglycerides_mg_dl", "hdl_mg_dl"} <= set(f.columns):
        fill("tgl_hdl_ratio", have("triglycerides_mg_dl", "hdl_mg_dl"),
             f["triglycerides_mg_dl"] / f["hdl_mg_dl"])
    if {"total_protein_g_dl", "albumin_g_dl"} <= set(f.columns):
        fill("globulin_g_dl", have("total_protein_g_dl", "albumin_g_dl"),
             f["total_protein_g_dl"] - f["albumin_g_dl"])
    if {"albumin_g_dl", "globulin_g_dl"} <= set(f.columns):
        fill("ag_ratio", have("albumin_g_dl", "globulin_g_dl"), f["albumin_g_dl"] / f["globulin_g_dl"])
    if {"ast_sgot_u_l", "alt_sgpt_u_l"} <= set(f.columns):
        fill("ast_alt_ratio", have("ast_sgot_u_l", "alt_sgpt_u_l"), f["ast_sgot_u_l"] / f["alt_sgpt_u_l"])
    if {"bilirubin_total_mg_dl", "bilirubin_direct_mg_dl"} <= set(f.columns):
        fill("bilirubin_indirect_mg_dl", have("bilirubin_total_mg_dl", "bilirubin_direct_mg_dl"),
             f["bilirubin_total_mg_dl"] - f["bilirubin_direct_mg_dl"])
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f


def _impute(frame, R):
    f = frame
    for c in f.columns:
        if c == "chronic_disease":
            f[c] = f[c].fillna("")                         # missing -> no conditions
        elif c in IDS:
            f[c] = f[c].fillna("NA")
        elif c in R["num_cols"]:
            f[c] = f[c].fillna(R["medians"][c])
        elif c in R["obj_cols"]:
            f[c] = f[c].fillna(R["modes"][c])
    return f


def _band(pct):
    return "HIGH" if pct >= 80 else "MEDIUM" if pct >= 50 else "LOW"


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def score_member(raw):
    """Score a member (dict) or a batch (DataFrame). dict -> dict; DataFrame -> list[dict]."""
    R = _resources()
    is_single = isinstance(raw, dict)
    # which columns the caller actually SUPPLIED (column-level). Matters for
    # chronic_disease: a supplied-but-blank value means "no conditions" (a real
    # observation), whereas an omitted column means "unknown / not in export".
    provided_cols = set(raw.keys()) if is_single else set(raw.columns)
    df = pd.DataFrame([raw]) if is_single else raw.copy()

    # build the canonical 115-col frame: keep provided, add missing as NaN, drop extras
    frame = df.reindex(columns=R["model_features"]).reset_index(drop=True)
    for c in (R["num_cols"] & set(frame.columns)):         # coerce numerics (dicts may pass strings)
        frame[c] = pd.to_numeric(frame[c], errors="coerce")

    frame = _derive(frame)
    present = frame.notna()                                # PRESENT = provided or derived (pre-impute)
    # chronic_disease: blank-but-supplied = "no conditions" = a real, present value
    if "chronic_disease" in provided_cols:
        present["chronic_disease"] = True
    frame = _impute(frame, R)

    # score everyone in one batch (cheap); mandatory failures get prob nulled out
    X = R["pre"].transform(frame)
    probs = R["model"].predict_proba(X)[:, 1]

    important_items = list(R["important"].items())         # [(feature, group), ...]
    results = []
    for i in range(len(frame)):
        prow = present.iloc[i]
        present_raw = {c for c in frame.columns if prow[c]}
        female = str(frame.iloc[i]["sex"]).upper() == "F"

        # --- mandatory check (a mandatory col is "present" if provided/derived) ---
        missing_mand = [m for m in R["mandatory"] if m not in present_raw]
        scored = len(missing_mand) == 0

        # --- panel completeness (clinical). female: PSA excluded from denominator ---
        panel = [(f, g) for f, g in important_items
                 if not (female and f == "psa_ng_ml")]
        present_imp = [(f, g) for f, g in panel if f in present_raw]
        missing_imp = [{"feature": f, "group": g} for f, g in panel if f not in present_raw]
        n_total = len(panel)
        completeness = 100.0 * len(present_imp) / n_total

        # --- model confidence: coef-weighted coverage of present features ---
        covered = sum(w for _, w, src in R["eng_sources"] if src <= present_raw)
        model_conf = 100.0 * covered / R["total_abscoef"] if R["total_abscoef"] else 0.0

        results.append({
            "claim_probability": float(probs[i]) if scored else None,
            "panel_completeness_pct": round(completeness, 1),
            "model_confidence_pct": round(model_conf, 1),
            "confidence_band": _band(completeness),
            "scored": scored,
            "reason": "ok" if scored else
                      "missing mandatory: " + ", ".join(missing_mand),
            "missing_important": missing_imp,
            "n_important_present": len(present_imp),
            "n_important_total": n_total,
        })

    return results[0] if is_single else results


# --------------------------------------------------------------------------
# self-test / demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    src = pd.read_csv(os.path.join(ROOT, "member_input.csv"))

    def show(tag, res):
        p = res["claim_probability"]
        ps = f"{p:.1%}" if p is not None else "—"
        print(f"\n[{tag}]")
        print(f"  scored={res['scored']}  reason={res['reason']}")
        print(f"  claim_probability = {ps}")
        print(f"  panel_completeness = {res['panel_completeness_pct']}%  "
              f"({res['n_important_present']}/{res['n_important_total']})  "
              f"band={res['confidence_band']}")
        print(f"  model_confidence   = {res['model_confidence_pct']}%")
        if res["missing_important"]:
            miss = ", ".join(m["feature"] for m in res["missing_important"][:8])
            extra = "" if len(res["missing_important"]) <= 8 else f" (+{len(res['missing_important'])-8} more)"
            print(f"  missing_important  = {miss}{extra}")

    print("="*64); print("A) FULL records (all 115 fields) — batch of 2"); print("="*64)
    for tag, res in zip(src["name"], score_member(src)):
        show(tag, res)

    print("\n" + "="*64); print("B) PARTIAL record: drop BP + ~40 optional fields"); print("="*64)
    keep = ["age","sex","height_cm","weight_kg","bmi","hba1c_percent","fbs_mg_dl",
            "total_cholesterol_mg_dl","hdl_mg_dl","ldl_mg_dl","triglycerides_mg_dl",
            "creatinine_mg_dl","bun_mg_dl","uric_acid_mg_dl","haemoglobin_g_dl",
            "alt_sgpt_u_l","ast_sgot_u_l","chronic_disease"]
    partial = {k: src.iloc[0][k] for k in keep}
    show("HIGH_RISK, partial panel", score_member(partial))

    print("\n" + "="*64); print("C) REFUSAL: a mandatory field (creatinine) absent"); print("="*64)
    bad = {k: v for k, v in partial.items() if k != "creatinine_mg_dl"}
    show("missing mandatory", score_member(bad))

    print("\n" + "="*64); print("D) FEMALE: PSA is not-applicable, excluded from denominator"); print("="*64)
    fem = dict(src.iloc[0]); fem["sex"] = "F"; fem["psa_ng_ml"] = np.nan
    res = score_member(fem)
    show("female, no PSA", res)
    print(f"  -> n_important_total = {res['n_important_total']} (30 = PSA excluded), "
          f"PSA in missing list? {'psa_ng_ml' in [m['feature'] for m in res['missing_important']]}")
