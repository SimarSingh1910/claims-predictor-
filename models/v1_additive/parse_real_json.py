#!/usr/bin/env python3
"""
parse_real_json.py  —  turn a real CIBYL/Thyrocare AHC record into a model-feature
dict, PRESERVING missing values (no imputation here — the graceful scorer needs to
see what is genuinely absent to compute confidence honestly).

NOTE: the brief said to reuse map_real_to_model.py, but no such file exists in the
repo. This module IS that mapping logic, authored from the real JSON's structure.
The real data uses MANY display_name spellings per lab (Thyrocare vs other labs,
Title-case vs UPPER-case), so the map is alias-rich and case-normalized.

parse_record(record) -> dict | None   (None if the record has no CIBYL score)
"""
import re
import numpy as np
from features import ORDINAL_MAPS, BINARY_MAPS, CRYSTAL_CATEGORIES


def norm(name):
    """Normalize a display_name for lookup: collapse spaces + upper. Keep '.'/'+'."""
    return re.sub(r"\s+", " ", str(name).strip()).upper()


# --------------------------------------------------------------------------
# alias (raw display_name)  ->  model column.  Many aliases per column.
# --------------------------------------------------------------------------
_ALIASES = {
    # ---- glucose ----
    "fbs_mg_dl": ["Glucose Fasting", "FASTING BLOOD SUGAR(GLUCOSE)",
                  "FBS-FASTING BLOOD SUGAR(GLUCOSE)", "Blood Sugar Fasting"],
    "hba1c_percent": ["Glycosylated Hemoglobin (HbA1c)", "HbA1c", "HBA1C"],
    "estimated_avg_glucose_mg_dl": ["Estimated Average Glucose",
                                    "ESTIMATED AVERAGE GLUCOSE(EAG)"],
    # ---- liver ----
    "bilirubin_total_mg_dl": ["Bilirubin Total", "BILIRUBIN - TOTAL", "BILIRUBIN, TOTAL"],
    "bilirubin_direct_mg_dl": ["Bilirubin Direct", "BILIRUBIN -DIRECT", "BILIRUBIN, DIRECT"],
    "bilirubin_indirect_mg_dl": ["Bilirubin Indirect"],
    "ast_sgot_u_l": ["SGOT/AST", "ASPARTATE AMINOTRANSFERASE (SGOT )",
                     "ASPARTATE AMINOTRANSFERASE (AST/SGOT)"],
    "alt_sgpt_u_l": ["SGPT/ALT", "ALANINE TRANSAMINASE (SGPT)",
                     "ALANINE AMINOTRANSFERASE (ALT/SGPT)"],
    "ast_alt_ratio": ["AST/ALT Ratio"],
    "alp_u_l": ["ALKALINE PHOSPHATASE", "Alkaline Phosphatase"],
    "ggt_u_l": ["Gamma Glutamyl Transferase (GGT)", "GAMMA GLUTAMYL TRANSFERASE (GGT)"],
    "total_protein_g_dl": ["PROTEIN - TOTAL", "Total Protein", "TOTAL PROTEIN"],
    "albumin_g_dl": ["ALBUMIN - SERUM", "Albumin", "ALBUMIN"],
    "globulin_g_dl": ["Globulin", "GLOBULIN"],
    "ag_ratio": ["Albumin :Globulin Ratio", "ALBUMIN/GLOBULIN RATIO"],
    # ---- kidney ----
    "uric_acid_mg_dl": ["Uric Acid", "URIC ACID"],
    "bun_mg_dl": ["Bun", "BLOOD UREA NITROGEN (BUN)", "BLOOD UREA NITROGEN"],
    "creatinine_mg_dl": ["Creatinine", "CREATININE - SERUM", "CREATININE"],
    "egfr_ml_min_173m2": ["eGFR (CKD-EPI)", "eGFR"],
    "calcium_mg_dl": ["CALCIUM", "Calcium Serum"],
    # ---- electrolytes ----
    "sodium_meq_l": ["SODIUM"], "potassium_meq_l": ["POTASSIUM"], "chloride_meq_l": ["CHLORIDE"],
    # ---- CBC ----
    "haemoglobin_g_dl": ["Hemoglobin", "HEMOGLOBIN", "HEMOGLOBIN (HB)"],
    "pcv_percent": ["PCV", "HEMATOCRIT(PCV)", "HEMATOCRIT (PCV)"],
    "rbc_million_cmm": ["RBC Count", "TOTAL RBC", "RED BLOOD CELL (RBC) COUNT"],
    "mcv_fl": ["MCV", "MEAN CORPUSCULAR VOLUME(MCV)", "MEAN CORPUSCULAR VOLUME (MCV)"],
    "mch_pg": ["MCH", "MEAN CORPUSCULAR HEMOGLOBIN(MCH)", "MEAN CORPUSCULAR HEMOGLOBIN (MCH)"],
    "mchc_percent": ["MCHC", "MEAN CORP.HEMO.CONC(MCHC)",
                     "MEAN CORPUSCULAR HEMOGLOBIN CONCENTRATION(MCHC)"],
    "rdw_cv_percent": ["RDW (CV)", "RED CELL DISTRIBUTION WIDTH (RDW-CV)",
                       "RED CELL DISTRIBUTION WIDTH (RDW)"],
    "rdw_sd_fl": ["RDW-SD"],
    "total_wbc_cells_cumm": ["TLC", "TOTAL LEUCOCYTES COUNT (WBC)", "WHITE BLOOD CELL (WBC) COUNT"],
    "neutrophils_percent": ["Neutrophils", "NEUTROPHILS"],
    "lymphocytes_percent": ["Lymphocytes", "LYMPHOCYTE", "LYMPHOCYTES"],
    "monocytes_percent": ["Monocytes", "MONOCYTES"],
    "eosinophils_percent": ["Eosinophils", "EOSINOPHILS"],
    "basophils_percent": ["Basophils", "BASOPHILS"],
    "neutrophils_abs_cells_cumm": ["Neutrophils.", "NEUTROPHILS - ABSOLUTE COUNT",
                                   "ABSOLUTE NEUTROPHIL COUNT"],
    "lymphocytes_abs_cells_cumm": ["Lymphocytes.", "LYMPHOCYTES - ABSOLUTE COUNT",
                                   "ABSOLUTE LYMPHOCYTE COUNT"],
    "monocytes_abs_cells_cumm": ["Monocytes.", "MONOCYTES - ABSOLUTE COUNT",
                                 "ABSOLUTE MONOCYTE COUNT"],
    "eosinophils_abs_cells_cumm": ["Eosinophils.", "EOSINOPHILS - ABSOLUTE COUNT",
                                   "ABSOLUTE EOSINOPHIL COUNT"],
    "basophils_abs_cells_cumm": ["Basophils.", "BASOPHILS - ABSOLUTE COUNT",
                                 "ABSOLUTE BASOPHIL COUNT"],
    "platelet_count_lakhs_cumm": ["Platelet Count", "PLATELET COUNT"],
    "mpv_fl": ["Mean Platelet Volume (MPV)", "MEAN PLATELET VOLUME (MPV)"],
    "pdw_fl": ["PDW"], "pct_percent": ["PCT"], "plcr_percent": ["P-LCC", "P-LCR"],
    "ig_percent": ["IMMATURE GRANULOCYTE PERCENTAGE(IG%)"],
    "ig_abs_cells_cumm": ["IMMATURE GRANULOCYTES(IG)"],
    "nrbc_per_100_wbc": ["NUCLEATED RED BLOOD CELLS"],
    "nrbc_percent": ["NUCLEATED RED BLOOD CELLS %"],
    # ---- lipids ----
    "total_cholesterol_mg_dl": ["Total Cholesterol", "TOTAL CHOLESTEROL", "CHOLESTEROL, TOTAL"],
    "hdl_mg_dl": ["HDL Cholesterol", "HDL CHOLESTEROL - DIRECT", "HDL CHOLESTEROL"],
    "ldl_mg_dl": ["LDL Cholesterol", "LDL CHOLESTEROL - DIRECT", "CHOLESTEROL LDL"],
    "triglycerides_mg_dl": ["Triglycerides", "TRIGLYCERIDES"],
    "vldl_mg_dl": ["V.L.D.L Cholesterol", "VERY LOW DENSITY LIPOPROTEIN"],
    "non_hdl_cholesterol_mg_dl": ["Non HDL Cholesterol", "NON HDL CHOLESTEROL"],
    "cho_hdl_ratio": ["Chol/HDL Ratio", "CHOL/HDL RATIO"],
    "ldl_hdl_ratio": ["LDL/HDL Ratio", "LDL/HDL RATIO"],
    # ---- thyroid (FT3/FT4 mapped to total T3/T4 — same slot; free vs total differ
    #      clinically + in units; near-zero model weight; FLAG for correction) ----
    "tsh_uiu_ml": ["Thyroid Stimulating Hormone (Ultrasensitive)", "TSH - ULTRASENSITIVE",
                   "TSH (ULTRASENSITIVE)"],
    "total_t3_ng_dl": ["FREE TRIIODOTHYRONINE (FT3)"],
    "total_t4_ug_dl": ["FREE THYROXINE (FT4)"],
    # ---- vitamins / inflammatory / cancer ----
    "vitamin_d_ng_ml": ["Vitamin D 25 - Hydroxy", "25 - HYDROXYVITAMIN D"],
    "vitamin_b12_pg_ml": ["Vitamin - B12", "VITAMIN B-12", "VITAMIN B12"],
    "esr_mm_hr": ["ERYTHROCYTE SEDIMENTATION RATE (ESR)"],
    "crp_mg_l": ["HIGH SENSITIVITY C-REACTIVE PROTEIN (HS-CRP)"],
    "psa_ng_ml": ["PROSTATE SPECIFIC ANTIGEN (PSA)"],
    # ---- vitals (present in a minority of records) ----
    "systolic_bp_mmhg": ["BP Systolic"], "diastolic_bp_mmhg": ["BP Diastolic"],
    # ---- urine numeric (range/absent -> number; 'Absent'->0) ----
    "urine_ph": ["Reaction (pH)", "PH"],
    "urine_specific_gravity": ["Specific Gravity", "SPECIFIC GRAVITY"],
    "urine_volume_ml": ["Volume", "VOLUME"],
    "urine_pus_cells_hpf": ["Pus Cells (WBCs)", "URINARY LEUCOCYTES (PUS CELLS)", "PUS CELL (WBCS)"],
    "urine_rbc_hpf": ["Red blood Cells", "RED BLOOD CELLS"],
    "urine_epithelial_cells_hpf": ["Epithelial Cells", "EPITHELIAL CELLS"],
    # ---- urine categorical ----
    "urine_colour": ["Colour", "COLOUR", "COLOR"],
    "urine_appearance": ["Transparency", "APPEARANCE"],
    "urine_protein": ["Urine Protein (Albumin)", "URINARY PROTEIN", "PROTEIN"],
    "urine_glucose": ["Urine Glucose (sugar)", "URINARY GLUCOSE"],
    "urine_ketones": ["Urine Ketones (Acetone)", "URINE KETONE", "KETONES"],
    "urine_bilirubin": ["Bilirubin Urine", "URINARY BILIRUBIN"],
    "urine_urobilinogen": ["Urobilinogen", "UROBILINOGEN"],
    "urine_nitrite": ["Nitrite", "NITRITE"],
    "urine_leucocyte_esterase": ["Leucocyte esterase", "LEUCOCYTE ESTERASE", "LEUKOCYTE ESTERASE"],
    "urine_casts": ["Cast", "CASTS"],
    "urine_crystals": ["Crystals", "CRYSTALS"],
    "urine_bacteria": ["Bacteria", "BACTERIA"],
    "urine_yeast": ["Yeast Cells", "YEAST"],
    "urine_amorphous_deposits": ["Amorphous deposits"],
    "urine_bile_pigment": ["BILE PIGMENT"],
    "urine_bile_salt": ["BILE SALT"],
    "urine_parasite": ["PARASITE"],
    "urine_mucus": ["MUCUS"],
}
# build normalized lookup:  NAME -> model_col
NAME2COL = {}
for col, aliases in _ALIASES.items():
    for a in aliases:
        NAME2COL[norm(a)] = col

X1000_COLS = {"total_wbc_cells_cumm", "neutrophils_abs_cells_cumm", "lymphocytes_abs_cells_cumm",
              "monocytes_abs_cells_cumm", "eosinophils_abs_cells_cumm", "basophils_abs_cells_cumm",
              "ig_abs_cells_cumm"}
PLATELET_COL = "platelet_count_lakhs_cumm"            # value /100 -> lakhs
URINE_NUM_ABSENT0 = {"urine_pus_cells_hpf", "urine_rbc_hpf", "urine_epithelial_cells_hpf"}
URINE_NUM_PLAIN = {"urine_ph", "urine_specific_gravity", "urine_volume_ml"}
URINE_CAT = {c for c in NAME2COL.values() if c.startswith("urine_") and
             c not in URINE_NUM_ABSENT0 and c not in URINE_NUM_PLAIN}

# allowed categories + the "zero"/"one" canonical value per categorical urine col
_CATMAPS = {**{c: ORDINAL_MAPS[c] for c in ORDINAL_MAPS}, **{c: BINARY_MAPS[c] for c in BINARY_MAPS},
            "urine_crystals": {c: i for i, c in enumerate(CRYSTAL_CATEGORIES)}}
ALLOWED = {c: {k.upper(): k for k in m} for c, m in _CATMAPS.items()}
ZERO_CAT = {c: min(m, key=m.get) for c, m in _CATMAPS.items()}      # value 0 category
ONE_CAT = {c: max(m, key=m.get) for c, m in BINARY_MAPS.items()}    # value 1 category
_NEG = {"NEGATIVE", "ABSENT", "NIL", "NOT DETECTED", "NOT SEEN", "NONE", "NAD", "-VE", "NEG"}
_POS = {"POSITIVE", "PRESENT", "DETECTED", "SEEN", "+VE"}


def _num(s):
    """Extract a float from a messy display_result, or None."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip()
    if t == "" or "CANCEL" in t.upper():
        return None
    m = re.search(r"-?\d+\.?\d*", t.replace(",", ""))
    return float(m.group()) if m else None


def _urine_count(s):
    """'Absent'->0 ; '1-2'->1.5 ; '3'->3 ; junk->None."""
    if s is None:
        return None
    u = str(s).strip().upper()
    if u == "":
        return None
    if u in _NEG:
        return 0.0
    nums = re.findall(r"\d+\.?\d*", u)
    if not nums:
        return None
    vals = [float(x) for x in nums]
    return sum(vals) / len(vals)


def _cat(col, raw):
    """Map a messy urine value to one of the model's allowed categories, or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    u = s.upper()
    allowed = ALLOWED.get(col, {})
    if u in allowed:                         # exact (case-insensitive) match
        return allowed[u]
    if u in _NEG and col in ZERO_CAT:
        return ZERO_CAT[col]
    if (u in _POS or "+" in u) and col in ONE_CAT:    # binary present/positive
        return ONE_CAT[col]
    plus = u.count("+")                      # dipstick '+', '++', '+++'
    if plus and col in ALLOWED:
        cand = f"{plus}+".upper()
        if cand in allowed:
            return allowed[cand]
        if "POSITIVE" in allowed:
            return allowed["POSITIVE"]
    return None


def _age_years(s):
    """'25 Y,9 M,7 D' -> 25.77 ; '40' -> 40.0"""
    if s is None:
        return None
    t = str(s)
    y = re.search(r"(\d+)\s*Y", t, re.I)
    mo = re.search(r"(\d+)\s*M", t, re.I)
    d = re.search(r"(\d+)\s*D", t, re.I)
    if y or mo or d:
        return (int(y.group(1)) if y else 0) + (int(mo.group(1))/12 if mo else 0) + (int(d.group(1))/365 if d else 0)
    return _num(t)


def parse_record(record):
    meta = record.get("meta", {})
    data = record.get("data", {}) or {}
    cibyl = (((data.get("health_summary") or {}).get("health_cibyl")) or {})
    score = cibyl.get("score")
    if score is None:                        # no CIBYL -> skip
        return None

    out = {}

    # demographics
    age = _age_years(meta.get("patient_age"))
    if age is not None:
        out["age"] = age
    g = str(meta.get("gender", "")).strip().lower()
    if g in ("male", "m"):
        out["sex"] = "M"
    elif g in ("female", "f"):
        out["sex"] = "F"

    # anthropometry from physicals (preferred) -> height/weight/bmi
    height = weight = bmi = None
    for ph in record.get("physicals", []) or []:
        q = str(ph.get("question", "")).strip().lower()
        ans = ph.get("answer", {}) or {}
        val = _num(ans.get("result"))
        if q == "height" and val:
            height = val
        elif q == "weight" and val:
            weight = val
        elif q == "bmi" and val:
            bmi = val

    # labs
    for grp in data.get("lab_parameters", []) or []:
        for bucket, plist in (grp.get("parameter_details") or {}).items():
            for prm in plist or []:
                dn = prm.get("display_name")
                if not dn:
                    continue
                key = norm(dn)
                # vitals/anthro that sometimes live in lab_parameters
                if key == "HEIGHT" and height is None:
                    height = _num(prm.get("display_result")); continue
                if key == "WEIGHT" and weight is None:
                    weight = _num(prm.get("display_result")); continue
                if key == "BMI" and bmi is None:
                    bmi = _num(prm.get("display_result")); continue
                col = NAME2COL.get(key)
                if col is None or col in out:           # unknown or already filled
                    continue
                raw = prm.get("display_result")
                if col in URINE_CAT:
                    v = _cat(col, raw)
                elif col in URINE_NUM_ABSENT0:
                    v = _urine_count(raw)
                else:
                    v = _num(raw)
                    if v is not None:
                        if col in X1000_COLS:
                            v *= 1000.0
                        elif col == PLATELET_COL:
                            v /= 100.0
                if v is not None:
                    out[col] = v

    if height is not None:
        out["height_cm"] = height
    if weight is not None:
        out["weight_kg"] = weight
    if bmi is not None:
        out["bmi"] = bmi
    elif height and weight:
        out["bmi"] = weight / (height / 100.0) ** 2

    # chronic_disease: intentionally LEFT ABSENT (no labels in real data) -> known
    # has_* blind spot; lowers model_confidence honestly. Do not invent it.

    # attach CIBYL + identity
    out["user_code"] = meta.get("USER_CODE")
    out["cibyl_score"] = float(score)
    out["cibyl_label"] = cibyl.get("label")
    return out
