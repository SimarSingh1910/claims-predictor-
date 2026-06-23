#!/usr/bin/env python3
"""
phase0_setup.py  —  Phase 0 of the Claim Propensity build plan.

What it does (the 4 guardrails), in order:
  1. Fixes ONE random seed so every run gives the same result.
  2. Drops the two LEAKAGE columns so the model can't cheat.
  3. STRATIFIED split into train / validation / test (60/20/20),
     keeping the ~16% claim rate identical in each.
  4. Writes a data_card.md so the result is reproducible & traceable.

Run it once. It creates a /splits folder and a data card. That's Phase 0 done.
"""
import numpy as np, pandas as pd, hashlib, datetime, os
from sklearn.model_selection import train_test_split

# ---- 1. ONE seed, used everywhere ---------------------------------------
SEED = 42
np.random.seed(SEED)

INFILE  = "healthbridge_ahc_modelready_100k.csv"   # the file you downloaded
OUTDIR  = "splits"
os.makedirs(OUTDIR, exist_ok=True)

print("Loading", INFILE, "...")
df = pd.read_csv(INFILE)
print(f"  {len(df):,} rows, {df.shape[1]} columns")

# ---- 2. Quarantine leakage columns (the model must NOT see these) -------
LEAKAGE = ["true_claim_propensity", "data_quality_flag"]
present = [c for c in LEAKAGE if c in df.columns]
df = df.drop(columns=present)
print("  dropped leakage columns:", present)

# ---- 3. Stratified split 60 / 20 / 20 on claim_next_12m -----------------
# first peel off 40% (-> temp), then split temp in half -> val & test
train, temp = train_test_split(df, test_size=0.40,
                               stratify=df["claim_next_12m"], random_state=SEED)
val, test   = train_test_split(temp, test_size=0.50,
                               stratify=temp["claim_next_12m"], random_state=SEED)

for name, part in [("train", train), ("val", val), ("test", test)]:
    path = os.path.join(OUTDIR, f"{name}.csv")
    part.to_csv(path, index=False)
    print(f"  wrote {path:18s} {len(part):>7,} rows   claim rate {part.claim_next_12m.mean()*100:5.2f}%")

# ---- 4. Data card (so any number you report is reproducible) ------------
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

card = f"""# Data Card — Claim Propensity dataset (Phase 0)

- Generated: {datetime.date.today()}
- Source file: `{INFILE}`  (md5 `{md5(INFILE)}`)
- Produced by: `fix_ahc.py` (synthetic AHC + swappable claim target)
- Random seed: {SEED}  (used for generation AND splitting)

## Splits (stratified on `claim_next_12m`)
| split | rows | claim rate |
|-------|------|-----------|
| train | {len(train):,} | {train.claim_next_12m.mean()*100:.2f}% |
| val   | {len(val):,} | {val.claim_next_12m.mean()*100:.2f}% |
| test  | {len(test):,} | {test.claim_next_12m.mean()*100:.2f}% |

The matching claim rates above confirm the stratification worked.

## Leakage columns removed (NEVER use as features)
{chr(10).join('- `'+c+'`' for c in present)}

## Targets (what you predict)
- `claim_next_12m`  — yes/no claim  (classification)
- `claim_count_12m` — how many claims (frequency)
- `claim_amount_inr`— claim cost in INR (severity / pricing)

## Rule
`test.csv` stays SEALED. Do not score it until Phase 7.
"""
with open("data_card.md", "w") as f:
    f.write(card)
print("  wrote data_card.md")
print("\nPhase 0 complete. Use splits/train.csv next. Leave splits/test.csv alone.")
