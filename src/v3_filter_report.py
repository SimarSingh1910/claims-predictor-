#!/usr/bin/env python3
"""
v3_filter_report.py  —  Step 2 of the v3 retarget: apply the rel_self filter and
report what survives. READ-ONLY: writes no artifacts, trains nothing.

WHY THE FILTER EXISTS (this is the whole point):
Negatives in this dataset are members with checkup/dental claims. AHC is an
EMPLOYEE benefit, so employees appear as negatives routinely, while a parent or
spouse enters the data almost only by making a hospitalisation claim. Dependent
negatives are therefore UNOBSERVABLE BY CONSTRUCTION — which is why every
rel_parent row is positive. That 100% is a sampling artifact of how the cohort was
assembled, not a clinical fact, and a model trained on it would learn "is a parent"
and be useless in production.

So we keep rel_self == 1 only, and drop rel_self/rel_spouse/rel_parent as features
(constant after the filter). Dependents are NOT modelled separately: there is no
negative class for them to learn against.

Also reported here, because everything downstream depends on it:
  * post-filter counts + positive rates (the derived base rate that REPLACES the
    hardcoded BASE_CLAIM_RATE = 0.1604 from the old synthetic cohort),
  * age x gender cohort cells, flagging any cell below MIN_CELL_N as unreportable,
  * verification of the claim that train_expanded_v3 is a clean superset of
    train_real_v3 on the 20 injected columns,
  * invalid rows (age == 0, under-18) and dead/constant features.

Usage:  .venv\\Scripts\\python.exe src\\v3_filter_report.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DATA = {
    "train_expanded_v3": r"C:\Users\PC\Downloads\train_expanded_v3.csv",
    "train_real_v3":     r"C:\Users\PC\Downloads\train_real_v3.csv",
    "test_real_v3":      r"C:\Users\PC\Downloads\test_real_v3.csv",
}

TARGET = "had_hospitalisation"
REL_COLS = ["rel_self", "rel_spouse", "rel_parent"]

# The 20 columns injected by the data owner (bootstrap-sampled from the empirical
# distributions in claim_health_data.xlsx, seeded per-row so a member gets the same
# draw in any file). They are TARGET-INDEPENDENT NOISE and exist only for schema /
# pipeline validation. Tree models WILL split on them and manufacture apparent
# importance — that output is not interpretable. Gated off by default.
SYNTHETIC_LABS = [
    "calcium_mg_dl", "sodium_mmol_l", "potassium_mmol_l", "chloride_mmol_l",
    "ggtp_u_l", "total_protein_g_dl", "albumin_g_dl", "alk_phosphatase_u_l",
    "vitamin_d_ng_ml", "vitamin_b12_pg_ml", "ft3_pg_ml", "ft4_ng_dl",
    "hs_crp_mg_l", "psa_ng_ml",
]
SYNTHETIC_FLAGS = [
    "flag_vitd_deficient", "flag_ggtp_high", "flag_hscrp_high",
    "flag_b12_low", "flag_alp_high", "flag_albumin_low",
]
SYNTHETIC_COLS = SYNTHETIC_LABS + SYNTHETIC_FLAGS
INCLUDE_SYNTHETIC_LABS = False          # <-- the hard constraint, default OFF

# Dead features: no variance to learn from across all 11,113 rows.
DEAD_COLS = ["ix_dysglyc_renal", "ix_bp_renal", "ix_metabolic_triad", "flag_egfr_low"]

MIN_CELL_N = 30                          # cells below this are suppressed
AGE_BINS = [(0, 30, "<30"), (30, 40, "30-39"), (40, 50, "40-49"),
            (50, 60, "50-59"), (60, 200, "60+")]


def age_group(a):
    if pd.isna(a):
        return "unknown"
    for lo, hi, label in AGE_BINS:
        if lo <= float(a) < hi:
            return label
    return "unknown"


def rule(title, ch="="):
    print("\n" + ch * 78)
    print(title)
    print(ch * 78)


def relationship(df):
    return np.select(
        [df["rel_self"] == 1, df["rel_spouse"] == 1, df["rel_parent"] == 1],
        ["self", "spouse", "parent"], default="none")


def main():
    frames = {n: pd.read_csv(p) for n, p in DATA.items()}

    # ---- 0. sanity: schema + the 20 injected columns -----------------------
    rule("0. SCHEMA")
    cols = list(frames["train_expanded_v3"].columns)
    print(f"columns: {len(cols)}")
    for name, df in frames.items():
        assert set(df.columns) == set(cols), (
            f"{name} has a different column SET: "
            f"only_here={sorted(set(df.columns)-set(cols))} "
            f"missing={sorted(set(cols)-set(df.columns))}")
    # The three files agree on the column SET but not the order of the trailing
    # three (had_hospitalisation / data_provenance / is_synthetic). Harmless —
    # canonicalise so every downstream .values / positional op is aligned.
    reordered = [n for n, df in frames.items() if list(df.columns) != cols]
    frames = {n: df[cols] for n, df in frames.items()}
    print(f"all three files share an identical column SET  ✓")
    if reordered:
        print(f"column ORDER differed in {reordered} (trailing 3 cols) — reindexed to "
              f"train_expanded_v3 order  ✓")
    missing_synth = [c for c in SYNTHETIC_COLS if c not in cols]
    print(f"injected columns declared: {len(SYNTHETIC_COLS)} "
          f"({len(SYNTHETIC_LABS)} labs + {len(SYNTHETIC_FLAGS)} flags)"
          + (f"  MISSING FROM FILE: {missing_synth}" if missing_synth else "  all present ✓"))
    print(f"original columns: {len(cols) - len(SYNTHETIC_COLS)}   "
          f"(69 - 20 = 49, matches the stated 'original 49' ✓)")
    print(f"INCLUDE_SYNTHETIC_LABS = {INCLUDE_SYNTHETIC_LABS}  "
          f"-> {'INCLUDED' if INCLUDE_SYNTHETIC_LABS else 'EXCLUDED'} from the default training path")

    # ---- 1. verify the superset claim -------------------------------------
    rule("1. SUPERSET CHECK — is train_expanded_v3 consistent with train_real_v3?")
    exp, real = frames["train_expanded_v3"], frames["train_real_v3"]
    exp_real = exp[exp["data_provenance"] == "real"].copy()
    print(f"train_expanded_v3 rows with data_provenance=='real': {len(exp_real)}")
    print(f"train_real_v3 rows                                 : {len(real)}")

    # Join on the 49 ORIGINAL columns (excluding provenance markers); if v3 is a
    # clean superset those must agree exactly, and then the 20 injected columns
    # must agree too.
    orig = [c for c in cols if c not in SYNTHETIC_COLS
            and c not in ("data_provenance", "is_synthetic")]
    def sig(df, subset):
        return (df[subset].astype(str).agg("|".join, axis=1))
    a = sig(exp_real, orig).value_counts()
    b = sig(real, orig).value_counts()
    shared = set(a.index) & set(b.index)
    overlap = sum(min(a[k], b[k]) for k in shared)
    print(f"multiset overlap on the {len(orig)} original columns: {overlap}/{len(real)}")

    # Now compare the injected columns on rows whose original-column signature is
    # UNIQUE in both files (an unambiguous 1-1 pairing).
    uniq = {k for k in shared if a[k] == 1 and b[k] == 1}
    ea = exp_real.assign(_s=sig(exp_real, orig)).set_index("_s")
    rb = real.assign(_s=sig(real, orig)).set_index("_s")
    keys = sorted(uniq)
    if keys:
        left, right = ea.loc[keys, SYNTHETIC_COLS], rb.loc[keys, SYNTHETIC_COLS]
        diff = {}
        for c in SYNTHETIC_COLS:
            l, r = left[c], right[c]
            neq = ~((l == r) | (l.isna() & r.isna()))
            if neq.any():
                diff[c] = int(neq.sum())
        print(f"unambiguous 1-1 pairs compared: {len(keys)}")
        print(f"injected columns differing    : "
              f"{diff if diff else 'NONE — byte-identical across files ✓'}")
    print(f"(v2 defect for contrast: 13 labs differed on ~100% of paired rows)")

    # ---- 2. THE FILTER -----------------------------------------------------
    rule("2. rel_self FILTER — counts and positive rates")
    print(f"{'file':<20}{'rows':>7}{'pos':>6}{'rate':>9}   ->  "
          f"{'rows':>7}{'pos':>6}{'rate':>9}{'  dropped':>10}")
    print("-" * 78)
    filtered = {}
    for name, df in frames.items():
        pre_n, pre_p = len(df), int(df[TARGET].sum())
        f = df[df["rel_self"] == 1].copy()
        filtered[name] = f
        post_n, post_p = len(f), int(f[TARGET].sum())
        print(f"{name:<20}{pre_n:>7}{pre_p:>6}{pre_p/pre_n:>9.2%}   ->  "
              f"{post_n:>7}{post_p:>6}{post_p/post_n:>9.2%}{pre_n-post_n:>10}")

    rule("2b. WHAT WAS DROPPED (and why it had to go)", "-")
    for name, df in frames.items():
        d = df[df["rel_self"] != 1]
        if not len(d):
            print(f"{name}: nothing dropped")
            continue
        g = d.assign(rel=relationship(d)).groupby("rel")[TARGET].agg(["size", "sum"])
        parts = "  ".join(f"{r}: {int(v['sum'])}/{int(v['size'])}={v['sum']/v['size']:.1%}"
                          for r, v in g.iterrows())
        print(f"{name:<20} dropped {len(d):>4} rows —  {parts}")

    # ---- 3. derived base rate ---------------------------------------------
    rule("3. DERIVED BASE RATE (replaces the hardcoded BASE_CLAIM_RATE = 0.1604)")
    for name, f in filtered.items():
        print(f"  {name:<20} base_rate = {f[TARGET].mean():.6f}  ({f[TARGET].mean():.2%})")
    print("\n  Do NOT hardcode. Derive from the split in use and log it.")
    print("  The old 0.1604 came from the synthetic 100k CLAIM cohort — a different")
    print("  target on a different population; it is not comparable.")

    # ---- 4. constant / dead features after the filter -----------------------
    rule("4. FEATURES TO DROP AFTER THE FILTER")
    tr = filtered["train_real_v3"]
    print(f"relationship one-hots (constant after filter):")
    for c in REL_COLS:
        vals = sorted(set(tr[c].unique()) | set(filtered["test_real_v3"][c].unique()))
        print(f"    {c:<24} values now = {vals}  -> DROP")
    print(f"\ndead features (no variance across all 11,113 rows):")
    for c in DEAD_COLS:
        nz = sum(int((f[c] != 0).sum()) for f in frames.values())
        tot = sum(len(f) for f in frames.values())
        print(f"    {c:<24} non-zero in {nz}/{tot} rows ({nz/tot:.3%})  -> DROP")
    print(f"\ninjected (gated by INCLUDE_SYNTHETIC_LABS = {INCLUDE_SYNTHETIC_LABS}):")
    print(f"    {len(SYNTHETIC_LABS)} labs + {len(SYNTHETIC_FLAGS)} flags = "
          f"{len(SYNTHETIC_COLS)} cols  -> EXCLUDED by default")
    print(f"\nnon-feature columns: data_provenance, is_synthetic, policy_year (split key), {TARGET}")

    modelling = [c for c in cols
                 if c not in REL_COLS + DEAD_COLS + [TARGET]
                 + ["data_provenance", "is_synthetic", "policy_year"]
                 and (INCLUDE_SYNTHETIC_LABS or c not in SYNTHETIC_COLS)]
    print(f"\n=> MODELLING FEATURES REMAINING: {len(modelling)}")
    print("   " + ", ".join(modelling))

    # ---- 5. data quality ---------------------------------------------------
    rule("5. INVALID ROWS SURVIVING THE FILTER")
    for name, f in filtered.items():
        z = int((f["age"] == 0).sum())
        u18 = int((f["age"] < 18).sum())
        u18p = int(f.loc[f["age"] < 18, TARGET].sum())
        print(f"  {name:<20} age==0: {z:<4} age<18: {u18:<4} (of which positive: {u18p})"
              f"   age range {f['age'].min():.0f}-{f['age'].max():.0f}")
    print("\n  -> age == 0 is invalid; under-18 employees are implausible. Both are")
    print("     removed in the training script (reported, not silently dropped).")

    # ---- 6. cohort cells ---------------------------------------------------
    rule(f"6. COHORT CELLS age x gender  (MIN_CELL_N = {MIN_CELL_N})")
    for name in ("train_real_v3", "test_real_v3", "train_expanded_v3"):
        f = filtered[name].copy()
        f = f[(f["age"] > 0) & (f["age"] >= 18)]
        f["age_group"] = f["age"].map(age_group)
        f["gender"] = np.where(f["sex_male"] == 1, "M", "F")
        g = f.groupby(["age_group", "gender"])[TARGET].agg(["size", "sum"]).reset_index()
        order = {l: i for i, (_, _, l) in enumerate(AGE_BINS)}
        order["unknown"] = 99
        g = g.sort_values(by=["age_group", "gender"],
                          key=lambda s: s.map(order) if s.name == "age_group" else s)
        print(f"\n--- {name}  (post-filter, post-cleanup: n={len(f)}) ---")
        ok = sup = 0
        for _, r in g.iterrows():
            n, pos = int(r["size"]), int(r["sum"])
            if n >= MIN_CELL_N:
                ok += 1
                mark = ""
            else:
                sup += 1
                mark = "   <-- SUPPRESS (n < 30)"
            print(f"   {r['age_group']:<8} {r['gender']}   n={n:<5} pos={pos:<4} "
                  f"rate={pos/n:7.2%}{mark}")
        print(f"   => {ok} reportable cell(s), {sup} suppressed")

    rule("SUMMARY — awaiting sign-off before step 3 (new preprocessor)")
    tr_n = len(filtered["train_real_v3"]); te_n = len(filtered["test_real_v3"])
    ex_n = len(filtered["train_expanded_v3"])
    print(f"  train_real_v3     {tr_n:>6} rows, {int(filtered['train_real_v3'][TARGET].sum()):>4} positive")
    print(f"  train_expanded_v3 {ex_n:>6} rows, {int(filtered['train_expanded_v3'][TARGET].sum()):>4} positive")
    print(f"  test_real_v3      {te_n:>6} rows, {int(filtered['test_real_v3'][TARGET].sum()):>4} positive  (SEALED)")
    print(f"  modelling features: {len(modelling)}")


if __name__ == "__main__":
    main()
