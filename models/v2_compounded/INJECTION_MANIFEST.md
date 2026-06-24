# INJECTION MANIFEST — v2 `compounded` target (REBALANCED)

> ⚠️ **THESE INTERACTIONS ARE SYNTHETIC AND PLANTED BY HAND.** Every magnitude
> below was injected by `fix_ahc_v2.py`. They exist to test whether a tree model
> can recover comorbidity interactions a linear model cannot. They are **NOT** real
> clinical findings and must **never** be reported as discoveries about real member
> risk. This file makes the planting fully auditable.

Generated: 2026-06-23 · seed 42 · 100,000 members · **no models trained by the generator.**

---

## What changed in the rebalance (v2.0 → v2.1)

The first cut put the biggest bonuses on the most *severe* combos (renal/diabetes),
but those already sit at **baseline ~0.91–0.98** from the additive model, so a bonus
just crushes against 1.0 and is **invisible to any model**. Rebalanced so the largest
bonuses land on **mid-range-baseline** combos that have headroom, where a bonus
**visibly moves the outcome**:

- **Bonuses are now ordered by LEARNABILITY, not severity.** `hypothyroid ×
  dyslipidaemia` (baseline 0.255 — huge headroom, wasted in v2.0) is now the prime
  demonstrator.
- **Saturated severe combos are HELD MODEST** (small bonuses, clinically ordered
  within the group). `hypertension × ckd` is classified **saturated** (baseline 0.92)
  even though it is a 2-way — it cannot be a demonstrator.
- Individual reference weights were lowered (below) so the de-inflated severe bonuses
  still clear the super-additivity bar.

## Method (unchanged)

- **Baseline:** the additive synthetic propensity `true_claim_propensity`.
  Features are **untouched**.
- **Injection:** `p_compounded = sigmoid( logit(true_claim_propensity) + boost )`,
  where `boost` is the **single strongest matched** interaction bonus (`STACKING =
  False`).
- **Realisation:** `claim_next_12m_compounded = 1` where a fresh seed-42 uniform draw
  `< p_compounded`. `claim_next_12m` (additive) is copied through unchanged.
- **Super-additivity is asserted** for every term: `bonus > Σ component reference weights`.

## Individual condition reference weights (log-odds)

Nominal per-condition increments — the additive baseline already carries the realised
per-condition risk; these only define the minimum bar each interaction must clear to
count as super-additive.

| condition | weight | | condition | weight |
|---|---|---|---|---|
| has_ckd | 0.35 | | has_anaemia | 0.15 |
| has_diabetes | 0.30 | | has_nafld | 0.15 |
| has_hypertension | 0.25 | | has_hyperuricaemia | 0.12 |
| has_obesity | 0.18 | | has_hypothyroidism | 0.10 |
| has_dyslipidaemia | 0.15 | | | |

## DEMONSTRATORS — mid-range baseline, sized for a visible sub-0.85 lift

These carry the **learnable** interaction signal. `margin` = bonus − Σcomponents.
`rate` = additive → compounded claim rate within members who have the combo.

| combination | bonus | Σcomp | margin | fired | rate add→comp | post<0.85 |
|---|---|---|---|---|---|---|
| hypothyroid × dyslipidaemia | **1.65** | 0.25 | +1.40 | 1982 | 0.255 → **0.409** | ✅ |
| obesity × hypertension × dyslipidaemia | **1.40** | 0.58 | +0.82 | 1553 | 0.605 → **0.733** | ✅ |
| diabetes × hypertension × dyslipidaemia | **1.05** | 0.70 | +0.35 | 1018 | 0.758 → **0.839** | ✅ |
| diabetes × hypertension | **1.00** | 0.55 | +0.45 | 2273 | 0.717 → **0.792** | ✅ |
| nafld × diabetes × obesity | **1.00** | 0.63 | +0.37 | 748 | 0.753 → **0.834** | ✅ |
| diabetes × dyslipidaemia | **0.95** | 0.45 | +0.50 | 2855 | 0.685 → **0.764** | ✅ |

## SATURATED — baseline near ceiling, held modest (NOT learnable demonstrators)

Bonuses kept small and clinically ordered within the group; their post-rates sit at
the ceiling, so a model cannot distinguish the interaction from the high baseline.

| combination | bonus | Σcomp | margin | fired | rate add→comp |
|---|---|---|---|---|---|
| diabetes × hypertension × ckd | **1.05** | 0.90 | +0.15 | 80 | 0.950 → 0.988 |
| diabetes × hypertension × dyslipidaemia × obesity (4-way) | **1.00** | 0.88 | +0.12 | 370 | 0.841 → 0.892 |
| hyperuricaemia × hypertension × ckd | **0.90** | 0.72 | +0.18 | 42 | 0.976 → 1.000 |
| diabetes × ckd | **0.80** | 0.65 | +0.15 | 343 | 0.915 → 0.959 |
| hypertension × ckd | **0.75** | 0.60 | +0.15 | 487 | 0.920 → 0.955 |
| ckd × anaemia | **0.72** | 0.50 | +0.22 | 440 | 0.923 → 0.952 |

## Severity multiplier (comorbid claims cost more)

For compounded claimants, `claim_amount_inr_compounded = claim_amount_inr ×
severity_multiplier`:

```
severity_multiplier = min(2.00, 1 + 0.25 × (n_comorbid_conditions − 1))   if n_comorbid ≥ 2
                    = 1.00                                                 otherwise
```
`n_comorbid_conditions` = count of the 9 curated conditions present in the member.

## Columns added to the dataset (126 total)

| column | role |
|---|---|
| `claim_next_12m` | additive target — **UNCHANGED** |
| `claim_next_12m_compounded` | **new** binary target (additive + interactions) |
| `claim_amount_inr_compounded` | severity-multiplied amount for comorbid claimants |
| `true_claim_propensity_compounded` | latent compounded probability (**leakage** — drop before training) |
| `interaction_boost_logit` | log-odds bonus applied (**audit/leakage** — drop) |
| `n_comorbid_conditions` | count of curated conditions present (**audit**) |
| `severity_multiplier` | applied cost multiplier (**audit**) |

## Net effect & verification

- Overall claim rate: additive **16.03%** → compounded **16.89%** (drift **+0.86 pp**;
  verifier requires ≤ 17% — **OK**, ≈ the prior compounded 16.81%).
- **All six demonstrators verified post-rate < 0.85.**
- 7,829 members (7.83%) received an interaction boost.

The hypothesis for the v2 rebuild: a tuned **LightGBM should now beat LogReg on PR-AUC**
for `claim_next_12m_compounded` (it can split on the demonstrator combinations), whereas
on the v1 additive target LogReg won. The saturated combos are deliberately not expected
to contribute learnable signal.
