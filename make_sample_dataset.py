#!/usr/bin/env python3
"""
make_sample_dataset.py — build the demo sample the Upload page serves.

WHY THIS EXISTS
    The old /api/sample streamed 200 complete rows straight from val.csv. Every
    field was populated, so every member scored 100% panel completeness / 100%
    confidence, all in the HIGH band, with zero refusals. The whole point of the
    graceful-degradation engine — confidence bands, partial panels, the refusal
    path — was INVISIBLE when demoing with that file.

    Real AHC exports are never complete: labs get skipped, panels vary by package,
    some records are partial. This script bakes CLINICALLY PLAUSIBLE missingness
    into the sample so the engine's behaviour actually shows.

WHAT IT DOES
    Reads the first 200 rows of splits/val.csv (val ONLY — never test.csv),
    strips ID + ALL target columns (same column set/order the sample has always
    had), then blanks whole clinical sub-panels per row to land each member in a
    target confidence band:

        ~55% HIGH     complete / near-complete (0-2 optional labs missing)
        ~30% MEDIUM   a chunk of the optional panel missing
        ~8%  LOW      most of the optional panel missing, mandatory intact
        ~7%  REFUSED  >=1 MANDATORY field blank -> engine refuses

    Missingness is dropped in coherent sub-panels (vitamins, thyroid,
    electrolytes, iron studies, urine, cardiac fitness, hemogram differential,
    liver, ...) — not one random cell at a time — because that is how real panels
    drop. The core metabolic / renal / lipid backbone stays mostly present, which
    is why most rows remain scorable.

    Blank cells are written EMPTY (pandas writes NaN as an empty field) — never
    0 / "NA" / "null" / "-".

DETERMINISM
    Seeded with 42, so the file is byte-reproducible. Re-run to regenerate.

The confidence band comes from panel_completeness_pct = % of the 31-feature
clinical IMPORTANT panel present (30 for females — PSA excluded). Bands:
HIGH >= 80%, MEDIUM >= 50%, LOW < 50%. None of the important-panel features are
derivation targets, so blanking one genuinely removes it from the count.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
VAL_CSV = os.path.join(ROOT, "splits", "val.csv")
OUT_CSV = os.path.join(ROOT, "sample_dataset.csv")

SEED = 42
N_ROWS = 200

# Columns stripped from the sample (mirror api/main.py ID_COLS + TARGET_COLS) so
# the sample is upload-shaped and never leaks the label.
ID_COLS = ["CUG", "employee_id", "name", "ahc_date"]
TARGET_COLS = ["claim_next_12m", "claim_count_12m", "claim_amount_inr"]

# The mandatory minimum (from core_panel.json). Blanking any of these forces the
# engine to REFUSE the row.
MANDATORY = ["age", "sex", "hba1c_percent", "fbs_mg_dl",
             "creatinine_mg_dl", "total_cholesterol_mg_dl"]

# --------------------------------------------------------------------------
# Clinically coherent sub-panels. Each is a group a real lab either runs or
# skips as a unit. The trailing comment gives how many IMPORTANT-panel features
# the group contains — that count is what actually moves the confidence band.
# --------------------------------------------------------------------------
PANELS = {
    # --- important-bearing groups (moving these changes the band) -----------
    "vitamins":     ["vitamin_d_ng_ml", "vitamin_b12_pg_ml"],                 # 2 important
    "thyroid":      ["total_t3_ng_dl", "total_t4_ug_dl", "tsh_uiu_ml"],       # 3 important
    "electrolytes": ["sodium_meq_l", "potassium_meq_l", "chloride_meq_l"],    # 3 important
    "inflammatory": ["crp_mg_l", "esr_mm_hr", "ra_factor_iu_ml"],             # 1 important (crp)
    "liver":        ["bilirubin_total_mg_dl", "bilirubin_direct_mg_dl",
                     "bilirubin_indirect_mg_dl", "ast_sgot_u_l", "alt_sgpt_u_l",
                     "alp_u_l", "ggt_u_l", "total_protein_g_dl", "albumin_g_dl",
                     "globulin_g_dl", "ag_ratio", "ast_alt_ratio"],           # 7 important
    "kidney_extra": ["bun_mg_dl", "calcium_mg_dl", "uric_acid_mg_dl"],        # 3 important (creatinine is mandatory, kept)
    # --- realism-only groups (0 important; add texture, don't move the band) -
    "iron_studies": ["ferritin_ng_ml", "iron_ug_dl", "tibc_ug_dl", "uibc_ug_dl"],
    "cardiac_fit":  ["resting_hr_bpm", "qtcb_ms", "vo2_max_ml_kg_min",
                     "max_mets", "duke_treadmill_score"],
    "hemogram_diff": [
        "pcv_percent", "mcv_fl", "mch_pg", "mchc_percent", "rdw_cv_percent",
        "rdw_sd_fl", "neutrophils_percent", "lymphocytes_percent",
        "monocytes_percent", "eosinophils_percent", "basophils_percent",
        "mpv_fl", "pdw_fl", "pct_percent", "plcr_percent", "nrbc_per_100_wbc",
        "nrbc_percent", "ig_percent", "ig_abs_cells_cumm",
        "neutrophils_abs_cells_cumm", "lymphocytes_abs_cells_cumm",
        "monocytes_abs_cells_cumm", "eosinophils_abs_cells_cumm",
        "basophils_abs_cells_cumm",
    ],
    "urine": [
        "urine_ph", "urine_specific_gravity", "urine_pus_cells_hpf",
        "urine_rbc_hpf", "urine_colour", "urine_appearance", "urine_volume_ml",
        "urine_protein", "urine_glucose", "urine_ketones", "urine_bilirubin",
        "urine_urobilinogen", "urine_nitrite", "urine_leucocyte_esterase",
        "urine_epithelial_cells_hpf", "urine_casts", "urine_crystals",
        "urine_bacteria", "urine_yeast", "urine_mucus", "urine_parasite",
        "urine_amorphous_deposits", "urine_bile_pigment", "urine_bile_salt",
    ],
}

# Realism-only panels: dropped opportunistically on many rows regardless of the
# target band, because real exports skip these constantly and it never changes
# whether the row is scorable.
REALISM_PANELS = ["urine", "cardiac_fit", "iron_studies", "hemogram_diff"]


def blank(row, cols_present, *panel_names):
    """Blank every column of the named panels that exists in the sample frame."""
    for name in panel_names:
        for c in PANELS[name]:
            if c in cols_present:
                row[c] = np.nan


def main():
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(VAL_CSV, nrows=N_ROWS)
    df = df.drop(columns=[c for c in ID_COLS + TARGET_COLS if c in df.columns])
    df = df.reset_index(drop=True)
    cols = set(df.columns)
    n = len(df)

    # Assign a target band to each row. Counts sum to n; shuffle so bands are not
    # clustered by position in the file.
    n_high    = round(0.55 * n)
    n_medium  = round(0.30 * n)
    n_low     = round(0.08 * n)
    n_refused = n - n_high - n_medium - n_low   # remainder (~7%)
    labels = (["HIGH"] * n_high + ["MEDIUM"] * n_medium
              + ["LOW"] * n_low + ["REFUSED"] * n_refused)
    labels = list(rng.permutation(labels))

    # Cycle the mandatory field(s) we blank on refusals so the "reason" strings
    # differ across the unscored panel (not the same field every time).
    refusal_plan = []
    for k in range(n_refused):
        primary = MANDATORY[k % len(MANDATORY)]
        # every 4th refusal drops a SECOND mandatory field for reason variety
        if k % 4 == 3:
            secondary = MANDATORY[(k + 2) % len(MANDATORY)]
            refusal_plan.append([primary, secondary])
        else:
            refusal_plan.append([primary])
    refusal_i = 0

    out = df.copy()
    for i in range(n):
        band = labels[i]
        row = out.iloc[i].copy()

        # Every row skips a random subset of the realism-only panels (real exports
        # rarely carry all of these). These carry 0 important features, so they add
        # texture without disturbing the target band.
        for name in REALISM_PANELS:
            if rng.random() < 0.6:
                blank(row, cols, name)

        if band == "HIGH":
            # 0-2 optional labs missing. Sometimes fully complete, sometimes a
            # single small add-on (vitamins or CRP) skipped.
            choice = rng.choice(["none", "vitamins", "crp_only"], p=[0.45, 0.35, 0.20])
            if choice == "vitamins":
                blank(row, cols, "vitamins")                       # -2 important
            elif choice == "crp_only":
                row["crp_mg_l"] = np.nan if "crp_mg_l" in cols else row.get("crp_mg_l")  # -1 important

        elif band == "MEDIUM":
            # A solid chunk of the optional panel gone: the add-on packages a
            # mid-tier AHC bundle typically omits. 8-12 important dropped.
            blank(row, cols, "vitamins", "thyroid", "electrolytes")  # -8 important
            if rng.random() < 0.5:
                blank(row, cols, "inflammatory")                      # -1 (crp)
            if rng.random() < 0.4:
                blank(row, cols, "kidney_extra")                      # -3

        elif band == "LOW":
            # Bare-bones package: most optional labs skipped, only the mandatory
            # backbone (+ core lipid/hemogram) survives. >=16 important dropped.
            blank(row, cols, "vitamins", "thyroid", "electrolytes",
                  "inflammatory", "liver")                            # -16 important
            if rng.random() < 0.5:
                blank(row, cols, "kidney_extra")                      # -> -19

        else:  # REFUSED — blank mandatory field(s); still drop some optional panels
            blank(row, cols, "vitamins", "thyroid")
            for m in refusal_plan[refusal_i]:
                if m in cols:
                    row[m] = np.nan
            refusal_i += 1

        out.iloc[i] = row

    out.to_csv(OUT_CSV, index=False)

    # --- provenance summary (not consumed by the app) ----------------------
    from collections import Counter
    print(f"[make_sample] wrote {OUT_CSV}")
    print(f"[make_sample] {n} rows, {out.shape[1]} columns, seed={SEED}")
    print(f"[make_sample] target bands: {dict(Counter(labels))}")
    print(f"[make_sample] refusal fields: "
          f"{Counter(m for plan in refusal_plan for m in plan)}")


if __name__ == "__main__":
    main()
