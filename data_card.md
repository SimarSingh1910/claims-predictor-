# Data Card — Claim Propensity dataset (Phase 0)

- Generated: 2026-06-22
- Source file: `healthbridge_ahc_modelready_100k.csv`  (md5 `4f322e720043`)
- Produced by: `fix_ahc.py` (synthetic AHC + swappable claim target)
- Random seed: 42  (used for generation AND splitting)

## Splits (stratified on `claim_next_12m`)
| split | rows | claim rate |
|-------|------|-----------|
| train | 60,000 | 16.04% |
| val   | 20,000 | 16.04% |
| test  | 20,000 | 16.04% |

The matching claim rates above confirm the stratification worked.

## Leakage columns removed (NEVER use as features)
- `true_claim_propensity`
- `data_quality_flag`

## Targets (what you predict)
- `claim_next_12m`  — yes/no claim  (classification)
- `claim_count_12m` — how many claims (frequency)
- `claim_amount_inr`— claim cost in INR (severity / pricing)

## Rule
`test.csv` stays SEALED. Do not score it until Phase 7.
