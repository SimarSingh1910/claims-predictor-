"""
features.py  —  Phase 1 feature engineering, as a reusable sklearn transformer.

WHY THIS IS A SEPARATE FILE (and not buried in the phase-1 script):
    joblib.dump() does NOT save the *code* of a custom class — it saves a
    reference ("look for ClinicalFeatureEngineer in module features"). So when
    Phase 5/7/8 loads preprocessor.joblib months from now, Python must be able
    to `import features`. Keeping the class in its own importable module is what
    makes the saved pipeline survive a cold start.

WHAT THIS TRANSFORMER DOES (all deterministic — NO fitting, so it cannot leak):
    Given the same row, it produces the same output no matter which split it
    sees. That is the whole point: train / val / test get identical treatment.
    The only thing in the whole pipeline that *learns* anything from the data
    is the scaler downstream, and that is fit on train only.

Output: an all-numeric, no-NaN DataFrame with a stable column order.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---- columns we throw away: pure identifiers, zero predictive value ---------
DROP_IDS = ["CUG", "employee_id", "name", "ahc_date"]

# ---- three raw targets: NEVER features. Removed here if present. ------------
TARGETS = ["claim_next_12m", "claim_count_12m", "claim_amount_inr"]

# ---- ordinal maps: order is clinically meaningful (worse = higher number) ---
ORDINAL_MAPS = {
    "urine_colour":             {"Straw": 0, "Pale Yellow": 1, "Yellow": 2, "Dark Yellow": 3},
    "urine_appearance":         {"Clear": 0, "Slightly Turbid": 1, "Turbid": 2},
    "urine_protein":            {"Negative": 0, "Trace": 1, "1+": 2, "2+": 3, "3+": 4},
    "urine_glucose":            {"Negative": 0, "Trace": 1, "Positive": 2, "1+": 3, "2+": 4, "3+": 5, "4+": 6},
    "urine_ketones":            {"Negative": 0, "Trace": 1, "1+": 2, "2+": 3},
    "urine_leucocyte_esterase": {"Negative": 0, "Trace": 1, "1+": 2, "2+": 3},
    "urine_bacteria":           {"Absent": 0, "Few": 1, "Moderate": 2, "Many": 3},
    "urine_casts":              {"Absent": 0, "Hyaline": 1, "Granular": 2},
    # discovered by inspecting the data — not in the original plan:
    "urine_bilirubin":          {"Negative": 0, "1+": 1, "2+": 2},
}

# ---- binary maps: two-state columns -> 0/1 ---------------------------------
BINARY_MAPS = {
    "sex":                {"M": 1, "F": 0},
    "urine_nitrite":      {"Positive": 1, "Negative": 0},
    "urine_urobilinogen": {"Increased": 1, "Normal": 0},
    # Absent/Present presence flags discovered in the data:
    "urine_yeast":             {"Present": 1, "Absent": 0},
    "urine_mucus":             {"Present": 1, "Absent": 0},
    "urine_parasite":          {"Present": 1, "Absent": 0},
    "urine_amorphous_deposits":{"Present": 1, "Absent": 0},
    "urine_bile_pigment":      {"Present": 1, "Absent": 0},
    "urine_bile_salt":         {"Present": 1, "Absent": 0},
}

# ---- nominal (unordered) -> one-hot. Type matters, order does not. ----------
# Fixed category list so the column set is identical across splits even if a
# rare value is missing from one split.
CRYSTAL_CATEGORIES = ["Absent", "Calcium Oxalate", "Uric Acid", "Triple Phosphate"]

# ---- chronic_disease: comma-separated multi-label -> 10 binary columns ------
# token-in-the-string  ->  output column name
CHRONIC_MAP = {
    "Type 2 Diabetes":        "has_diabetes",
    "Hypertension":           "has_hypertension",
    "Dyslipidaemia":          "has_dyslipidaemia",
    "Anaemia":                "has_anaemia",
    "Hypothyroidism":         "has_hypothyroidism",
    "Hyperthyroidism":        "has_hyperthyroidism",
    "Obesity":                "has_obesity",
    "Hyperuricaemia":         "has_hyperuricaemia",
    "Fatty Liver (NAFLD)":    "has_nafld",
    "Chronic Kidney Disease": "has_ckd",
}

# ---- skewed labs we compress with log1p (computed AFTER raw-value flags) ----
LOG1P_COLS = ["tsh_uiu_ml", "crp_mg_l", "ferritin_ng_ml",
              "creatinine_mg_dl", "alt_sgpt_u_l", "ggt_u_l"]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic AHC feature engineering. fit() learns nothing."""

    def fit(self, X, y=None):
        # Nothing to learn — but we record the output column order the first
        # time we see data, so transform() is stable and we can expose names.
        self.feature_names_ = list(self._engineer(X).columns)
        return self

    def transform(self, X):
        out = self._engineer(X)
        # guarantee identical column order to what fit() saw
        if hasattr(self, "feature_names_"):
            out = out.reindex(columns=self.feature_names_, fill_value=0)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)

    # -- the actual work -----------------------------------------------------
    def _engineer(self, X):
        df = X.copy()

        # 1. drop identifiers and any target columns that rode along
        df = df.drop(columns=[c for c in DROP_IDS + TARGETS if c in df.columns])

        # 2. ABNORMAL FLAGS — compute from RAW values BEFORE any log transform.
        #    These catch 'undiagnosed' members: healthy label but bad labs.
        df["hba1c_high"]        = (df["hba1c_percent"] >= 6.5).astype(int)
        df["hba1c_prediabetic"] = ((df["hba1c_percent"] >= 5.7) & (df["hba1c_percent"] < 6.5)).astype(int)
        df["bp_high"]           = ((df["systolic_bp_mmhg"] >= 140) | (df["diastolic_bp_mmhg"] >= 90)).astype(int)
        df["bmi_obese"]         = (df["bmi"] >= 30).astype(int)
        df["bmi_overweight"]    = ((df["bmi"] >= 25) & (df["bmi"] < 30)).astype(int)
        df["egfr_low"]          = (df["egfr_ml_min_173m2"] < 60).astype(int)
        df["ldl_high"]          = (df["ldl_mg_dl"] >= 160).astype(int)
        df["crp_elevated"]      = (df["crp_mg_l"] > 5).astype(int)
        df["hb_low"]            = (df["haemoglobin_g_dl"] < 12).astype(int)
        df["vitd_deficient"]    = (df["vitamin_d_ng_ml"] < 20).astype(int)

        # 3. PSA — structural missingness (NaN for every female). Not random.
        #    Flag applicability, then fill the NaN with 0 (a real, meaningful 0).
        df["psa_applicable"] = (df["sex"] == "M").astype(int)
        df["psa_ng_ml"] = df["psa_ng_ml"].fillna(0)

        # 4. sex + the simple binary columns -> 0/1
        for col, mapping in BINARY_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).astype("int8")

        # 5. ordinal columns -> integer severity
        for col, mapping in ORDINAL_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).astype("int8")

        # 6. urine_crystals -> one-hot (fixed categories, drop 'Absent' as base)
        if "urine_crystals" in df.columns:
            for cat in CRYSTAL_CATEGORIES:
                if cat == "Absent":
                    continue  # baseline / reference level
                colname = "crystal_" + cat.lower().replace(" ", "_")
                df[colname] = (df["urine_crystals"] == cat).astype("int8")
            df = df.drop(columns=["urine_crystals"])

        # 7. chronic_disease multi-label -> 10 binary has_* columns
        cd = df["chronic_disease"].fillna("")  # NaN = no condition
        for token, outcol in CHRONIC_MAP.items():
            df[outcol] = cd.str.contains(token, regex=False).astype("int8")
        df = df.drop(columns=["chronic_disease"])

        # 8. log1p the heavily right-skewed labs (flags already taken above)
        for col in LOG1P_COLS:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

        return df
