#!/usr/bin/env python3
"""
fix_ahc_v2.py  —  v2 'compounded' target generator.

NOTE: the brief said to copy fix_ahc.py and edit it, but fix_ahc.py does not exist
in the repo (only its 100k OUTPUT csv survives). So this script operates on that
existing output instead — which is strictly SAFER for the requirement "leave every
member's AHC feature values untouched": we never regenerate a single feature, we
only ADD claim-target columns on the exact same 100,000 members.

WHAT IT DOES
  - Reuses the additive latent risk already in the data (`true_claim_propensity`,
    the additive synthetic propensity) as the v1 baseline.
  - Adds SUPER-ADDITIVE interaction bonuses (log-odds) for curated comorbidity
    combinations, clinically ordered (renal / cardiometabolic largest). Every bonus
    is asserted to exceed the sum of its components' individual reference weights,
    so the effect is genuinely super-additive (more than the parts).
  - Emits BOTH targets on identical data:
        claim_next_12m              (additive, UNCHANGED — copied straight through)
        claim_next_12m_compounded   (additive baseline + interaction bonuses)
  - Gives comorbid claimants a severity multiplier -> claim_amount_inr_compounded.

  *** SYNTHETIC, PLANTED interactions. The magnitudes below are injected by hand and
      are fully listed in INJECTION_MANIFEST.md. They must NEVER be reported as a
      real clinical finding. ***

This script does NOT train anything. It writes the v2 dataset + nothing else.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import CHRONIC_MAP          # token -> has_* column

SEED = 42
INFILE = os.path.join(ROOT, "healthbridge_ahc_modelready_100k.csv")
OUTFILE = os.path.join(HERE, "healthbridge_ahc_v2_compounded_100k.csv")

# --------------------------------------------------------------------------
# Individual condition reference weights (log-odds). These DOCUMENT the additive
# per-condition effect already baked into true_claim_propensity; we use them only
# to PROVE each interaction bonus is super-additive (bonus > sum of components).
# Clinically ordered: renal (ckd) and metabolic (diabetes) largest.
# --------------------------------------------------------------------------
INDIV = {
    "has_ckd": 0.35, "has_diabetes": 0.30, "has_hypertension": 0.25,
    "has_obesity": 0.18, "has_dyslipidaemia": 0.15, "has_anaemia": 0.15,
    "has_nafld": 0.15, "has_hyperuricaemia": 0.12, "has_hypothyroidism": 0.10,
}

# --------------------------------------------------------------------------
# Injected interaction bonuses (log-odds). (label, conditions, bonus, kind)
#
# v2 REBALANCE: the learnable signal must sit where the additive BASELINE rate is
# mid-range (~0.3-0.7), so a bonus visibly moves the outcome while staying < 0.85.
#   - DEMONSTRATORS: mid-range baseline -> bonus enlarged for a clear, sub-0.85 lift.
#     hypothyroid x dyslipidaemia (baseline 0.255) is the prime demonstrator: huge
#     headroom that v1 wasted.
#   - SATURATED: baseline already ~0.84-0.98 (renal / severe combos). A bonus only
#     crushes against 1.0 and is invisible to a model, so these are HELD MODEST and
#     clinically ordered, NOT inflated. (hypertension x ckd baseline=0.92 belongs
#     here despite being a 2-way — it is saturated, not mid-range.)
# Magnitudes are ordered by LEARNABILITY, not severity; severity ordering is kept
# WITHIN the saturated group. Every bonus still exceeds its components' sum.
# --------------------------------------------------------------------------
INTERACTIONS = [
    # ---- DEMONSTRATORS (mid-range baseline, sized for a visible sub-0.85 lift) ----
    ("hypothyroid x dyslipidaemia",            {"has_hypothyroidism", "has_dyslipidaemia"},               1.65, "demonstrator"),
    ("obesity x hypertension x dyslipidaemia", {"has_obesity", "has_hypertension", "has_dyslipidaemia"},   1.40, "demonstrator"),
    ("diabetes x hypertension x dyslipidaemia",{"has_diabetes", "has_hypertension", "has_dyslipidaemia"},  1.05, "demonstrator"),
    ("diabetes x hypertension",                {"has_diabetes", "has_hypertension"},                       1.00, "demonstrator"),
    ("nafld x diabetes x obesity",             {"has_nafld", "has_diabetes", "has_obesity"},               1.00, "demonstrator"),
    ("diabetes x dyslipidaemia",               {"has_diabetes", "has_dyslipidaemia"},                      0.95, "demonstrator"),
    # ---- SATURATED (baseline near ceiling: held modest, clinically ordered) ----
    ("diabetes x hypertension x ckd",          {"has_diabetes", "has_hypertension", "has_ckd"},            1.05, "saturated"),
    ("diabetes x hypertension x dyslipidaemia x obesity",
     {"has_diabetes", "has_hypertension", "has_dyslipidaemia", "has_obesity"},                            1.00, "saturated"),
    ("hyperuricaemia x hypertension x ckd",    {"has_hyperuricaemia", "has_hypertension", "has_ckd"},      0.90, "saturated"),
    ("diabetes x ckd",                         {"has_diabetes", "has_ckd"},                                0.80, "saturated"),
    ("hypertension x ckd",                     {"has_hypertension", "has_ckd"},                            0.75, "saturated"),
    ("ckd x anaemia",                          {"has_ckd", "has_anaemia"},                                 0.72, "saturated"),
]

# stacking rule: False = apply only the single STRONGEST matched interaction per
# member (avoids explosive logit sums for very comorbid members). Flip to True to
# add every matched term. Documented in the manifest; left as a reviewable lever.
STACKING = False

# severity: comorbid claimants cost more. multiplier on claim_amount_inr.
SEV_PER_EXTRA = 0.25   # +25% per comorbid condition beyond the first
SEV_CAP = 2.00

# --- guarantee super-additivity (programmatic audit) ---
for label, conds, bonus, kind in INTERACTIONS:
    comp_sum = sum(INDIV[c] for c in conds)
    assert bonus > comp_sum, f"{label}: bonus {bonus} !> component sum {comp_sum}"

CURATED = sorted(INDIV.keys())   # the 9 conditions that participate in interactions


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    print(f"Loading {INFILE} ...")
    df = pd.read_csv(INFILE)
    n = len(df)
    print(f"  {n:,} members, {df.shape[1]} columns (features UNTOUCHED)")

    # condition flags from chronic_disease (features stay untouched; this is read-only)
    cd = df["chronic_disease"].fillna("")
    flags = {col: cd.str.contains(tok, regex=False).to_numpy()
             for tok, col in CHRONIC_MAP.items()}

    # per-member interaction boost + bookkeeping
    boost = np.zeros(n)
    matched = [[] for _ in range(n)]
    per_term_fired = {}
    for label, conds, bonus, kind in INTERACTIONS:
        hit = np.logical_and.reduce([flags[c] for c in conds])
        per_term_fired[label] = int(hit.sum())
        idx = np.where(hit)[0]
        for i in idx:
            matched[i].append((label, bonus))
    for i in range(n):
        if matched[i]:
            bonuses = [b for _, b in matched[i]]
            boost[i] = sum(bonuses) if STACKING else max(bonuses)

    # compounded propensity = additive baseline (logit) + interaction boost
    p_add = df["true_claim_propensity"].to_numpy()
    z_comp = logit(p_add) + boost
    p_comp = sigmoid(z_comp)

    # realise compounded binary target (fresh seeded draw on the same members)
    rng = np.random.default_rng(SEED)
    u = rng.random(n)
    claim_comp = (u < p_comp).astype(int)

    # comorbidity count among the 9 curated conditions + severity multiplier
    n_comorbid = np.sum([flags[c] for c in CURATED], axis=0)
    sev_mult = np.where(n_comorbid >= 2,
                        np.minimum(SEV_CAP, 1 + SEV_PER_EXTRA * (n_comorbid - 1)), 1.0)
    # only boost cost for members who actually claim under the compounded target
    sev_mult_applied = np.where(claim_comp == 1, sev_mult, 1.0)
    amount_comp = df["claim_amount_inr"].to_numpy() * sev_mult_applied

    # --- attach new columns; claim_next_12m stays exactly as-is ---
    df["claim_next_12m_compounded"] = claim_comp
    df["claim_amount_inr_compounded"] = np.round(amount_comp, 2)
    # audit / leakage columns (drop before training, like true_claim_propensity):
    df["true_claim_propensity_compounded"] = np.round(p_comp, 6)
    df["interaction_boost_logit"] = np.round(boost, 4)
    df["n_comorbid_conditions"] = n_comorbid.astype(int)
    df["severity_multiplier"] = np.round(sev_mult_applied, 3)

    df.to_csv(OUTFILE, index=False)

    # --- summary for review ---
    print("\n" + "=" * 64)
    print("COMPOUNDED TARGET — before / after (identical members & features)")
    print("=" * 64)
    add_rate = df["claim_next_12m"].mean()
    comp_rate = claim_comp.mean()
    print(f"  additive   claim_next_12m            rate = {add_rate:.4f}")
    print(f"  compounded claim_next_12m_compounded rate = {comp_rate:.4f}  "
          f"(drift {100*(comp_rate-add_rate):+.2f} pp)")
    print(f"  members gaining interaction boost (>0)    = {(boost>0).sum():,}  "
          f"({(boost>0).mean()*100:.2f}%)")
    print(f"  stacking rule: {'SUM of matched' if STACKING else 'MAX matched only'}")

    def lift(conds):
        hit = np.logical_and.reduce([flags[c] for c in conds])
        return hit, df.loc[hit, "claim_next_12m"].mean(), claim_comp[hit].mean()

    for tag in ("demonstrator", "saturated"):
        print(f"\n  [{tag.upper()}]  combo  (fired)  +bonus   pre -> post   "
              + ("(post must be < 0.85)" if tag == "demonstrator" else "(at ceiling — not learnable)"))
        for label, conds, bonus, kind in INTERACTIONS:
            if kind != tag:
                continue
            hit, ra, rc = lift(conds)
            flag = ""
            if kind == "demonstrator":
                flag = "  <-- OK" if rc < 0.85 else "  <-- !! >=0.85"
            print(f"    {label:48s} ({per_term_fired[label]:5d})  +{bonus:.2f}  "
                  f"{ra:.3f} -> {rc:.3f}{flag}")

    # --- explicit verification the task asks for ---
    demo_bad = [label for label, conds, bonus, kind in INTERACTIONS
                if kind == "demonstrator" and lift(conds)[2] >= 0.85]
    print("\n  VERIFY demonstrators post-rate < 0.85 :",
          "ALL PASS" if not demo_bad else f"FAIL -> {demo_bad}")
    print(f"  VERIFY base rate ~16% (not drifting up): {comp_rate:.4f}",
          "OK" if comp_rate <= 0.17 else "<-- drifted high")
    print(f"\n  severity: comorbid claimants amount x up to {SEV_CAP} "
          f"(+{int(SEV_PER_EXTRA*100)}%/extra condition)")
    print(f"\nWrote {OUTFILE}")
    print("Both targets present:  claim_next_12m (additive)  +  claim_next_12m_compounded")
    print("NO models trained. Review INJECTION_MANIFEST.md before rebuilding on v2.")


if __name__ == "__main__":
    main()
