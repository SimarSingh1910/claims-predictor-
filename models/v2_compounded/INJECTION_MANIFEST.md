# INJECTION MANIFEST — v2 `compounded` target

> ⚠️ **THESE INTERACTIONS ARE SYNTHETIC AND PLANTED BY HAND.** Every magnitude
> below was injected by `fix_ahc_v2.py`. They exist to test whether a tree model
> can recover comorbidity interactions that a linear model cannot. They are **NOT**
> real clinical findings and must **never** be reported as discoveries about real
> member risk. This file makes the planting fully auditable.

Generated: 2026-06-23 · seed 42 · 100,000 members · **no models trained.**

---

## Method

- **Baseline:** the additive synthetic propensity already in the data,
  `true_claim_propensity` (the v1 additive latent risk). Features are **untouched**.
- **Injection:** for each curated comorbidity combination present in a member, a
  super-additive **bonus is added in log-odds space** to that member's baseline
  logit. Compounded probability = `sigmoid(logit(true_claim_propensity) + boost)`.
- **Stacking rule:** `STACKING = False` → each member receives **only the single
  strongest matched bonus** (not the sum), to avoid explosive logits for highly
  comorbid members. Flip to `True` in `fix_ahc_v2.py` to sum all matched terms.
- **Realisation:** `claim_next_12m_compounded` = `1` where a fresh seed-42 uniform
  draw `< compounded probability`. `claim_next_12m` (additive) is **copied through
  unchanged**.
- **Super-additivity guarantee:** the script `assert`s every bonus **exceeds the sum
  of its components' individual reference weights** (below), so each interaction is
  provably more than the sum of its parts.

## Individual condition reference weights (log-odds)

Documented per-condition additive effects, used only to prove super-additivity.
Clinically ordered — renal/metabolic largest.

| condition | weight |
|---|---|
| has_ckd | 0.60 |
| has_diabetes | 0.50 |
| has_hypertension | 0.40 |
| has_obesity | 0.30 |
| has_dyslipidaemia | 0.25 |
| has_anaemia | 0.25 |
| has_nafld | 0.25 |
| has_hyperuricaemia | 0.20 |
| has_hypothyroidism | 0.18 |

## Injected interaction bonuses (exact magnitudes + realised effect)

`margin` = bonus − Σ(component weights) > 0 ⇒ super-additive.
`fired` = members with the combination. `rate` = additive → compounded claim rate
within those members (on this 100k).

### 2-way
| combination | bonus (logit) | Σcomp | margin | fired | claim rate add→comp |
|---|---|---|---|---|---|
| diabetes × ckd | **1.40** | 1.10 | +0.30 | 343 | 0.915 → 0.974 |
| hypertension × ckd | **1.30** | 1.00 | +0.30 | 487 | 0.920 → 0.967 |
| diabetes × hypertension | **1.10** | 0.90 | +0.20 | 2273 | 0.717 → 0.821 |
| ckd × anaemia | **1.05** | 0.85 | +0.20 | 440 | 0.923 → 0.964 |
| diabetes × dyslipidaemia | **0.95** | 0.75 | +0.20 | 2855 | 0.685 → 0.782 |
| hypothyroid × dyslipidaemia | **0.60** | 0.43 | +0.17 | 1982 | 0.255 → 0.320 |

### 3-way  (diabetes × hypertension × ckd is the largest 3-way)
| combination | bonus (logit) | Σcomp | margin | fired | claim rate add→comp |
|---|---|---|---|---|---|
| **diabetes × hypertension × ckd** | **2.40** | 1.50 | +0.90 | 80 | 0.950 → 1.000 |
| hyperuricaemia × hypertension × ckd | **1.90** | 1.20 | +0.70 | 42 | 0.976 → 1.000 |
| diabetes × hypertension × dyslipidaemia | **1.70** | 1.15 | +0.55 | 1018 | 0.758 → 0.884 |
| nafld × diabetes × obesity | **1.60** | 1.05 | +0.55 | 748 | 0.753 → 0.873 |
| obesity × hypertension × dyslipidaemia | **1.45** | 0.95 | +0.50 | 1553 | 0.605 → 0.748 |

### 4-way  (largest overall)
| combination | bonus (logit) | Σcomp | margin | fired | claim rate add→comp |
|---|---|---|---|---|---|
| diabetes × hypertension × dyslipidaemia × obesity | **2.50** | 1.45 | +1.05 | 370 | 0.841 → 0.941 |

## Severity multiplier (comorbid claims cost more)

For members who claim under the compounded target, `claim_amount_inr_compounded` =
`claim_amount_inr` × `severity_multiplier`, where

```
severity_multiplier = min(2.00, 1 + 0.25 × (n_comorbid_conditions − 1))   if n_comorbid ≥ 2
                    = 1.00                                                 otherwise
```

`n_comorbid_conditions` = count of the 9 curated conditions present in the member.

## Columns added to the dataset

| column | role |
|---|---|
| `claim_next_12m` | additive target — **UNCHANGED** (copied through) |
| `claim_next_12m_compounded` | **new** binary target (additive + interactions) |
| `claim_amount_inr_compounded` | severity-multiplied amount for comorbid claimants |
| `true_claim_propensity_compounded` | latent compounded probability (**leakage** — drop before training) |
| `interaction_boost_logit` | the log-odds bonus applied (**audit/leakage** — drop) |
| `n_comorbid_conditions` | count of curated conditions present (**audit**) |
| `severity_multiplier` | applied cost multiplier (**audit**) |

## Net effect

- Overall claim rate: additive **16.03%** → compounded **16.81%** (small, because the
  combinations are rare). The signal lives in the comorbid sub-populations, not the
  base rate.
- 7,829 members (7.83%) received an interaction boost.

## Reviewer note on ceiling effects

The additive baseline already drives the most severe combinations near `p≈0.9`, so
their compounded lift saturates (e.g. diabetes×hypertension×ckd 0.95→1.00). The most
**learnable** interaction signal sits in the mid-range combos
(diabetes×hypertension 0.72→0.82, obesity×hypertension×dyslipidaemia 0.61→0.75,
diabetes×hypertension×dyslipidaemia 0.76→0.88). If you want larger separability for
the v2 rebuild, raise the mid-range bonuses or enable `STACKING` — both are one-line
changes here. **Awaiting your review of these magnitudes before any v2 retraining.**
