# Model Card — Hospitalisation Propensity (v3, real data)

**Status:** validated pipeline + honest power assessment. **NOT a production
model. No pricing output ships from it.**
**Target:** `had_hospitalisation` — **not** "claim". Different target, different
population, not comparable to the legacy `claim_next_12m` work or to
`BASE_CLAIM_RATE = 0.1604`.
**Artifacts:** `preprocessor_v3.joblib`, `hospitalisation_p12_v3.joblib`,
`v3_cv_results.json`, `v3_sealed_eval_results.json`.
**Protocol:** pre-registered in [`src/v3_prereg.md`](../../src/v3_prereg.md),
committed before the sealed set was read. Zero deviations.

---

## Headline result

**The pre-registered null triggers both fired.**

| metric (sealed test, n=410, 38 events) | primary LogReg C=10, sigmoid-calibrated |
|---|---|
| PR-AUC | **0.1101  [0.0877, 0.1784]** — baseline 0.0927, **interval includes it** |
| AUC-ROC | **0.5383  [0.4420, 0.6348]** — **interval includes 0.50** |
| Brier | 0.0848  [0.0829, 0.0864] |
| log-loss | 0.3132  [0.3026, 0.3228] |

> "A confidence interval on AUC-ROC that includes 0.50, or a PR-AUC interval
>  that includes the 9.27% baseline, is a pre-specified possible outcome. It
>  means 42 training events and 38 test events are insufficient to detect
>  signal at this sample size. It does NOT mean AHC parameters lack predictive
>  value for hospitalisation, and it does NOT mean the modelling approach is
>  wrong. Those questions are not answerable with this dataset and remain
>  open pending real TPA claims data at scale.
>  Reported either way, unchanged."

This was the *expected* outcome, recorded in advance: training CV gave AUC 0.5869
with a lower bound of 0.4993.

## Sample size — the binding constraint

| | rows | events | rate |
|---|---:|---:|---:|
| train (2021-22, 2022-23) | 576 | **42** | 7.29% |
| test (2023-24, sealed) | 410 | **38** | 9.27% |

**6.0 events per feature** across 7 features — below the conventional 10–20
minimum, and plausibly **7.0 across 6 effectively-informative features** given
the HbA1c defect below.

Out-of-time split: train ≤ 2022-23, test 2023-24. Covariate shift is mild — no
feature moved more than 0.25 SD between the two.

## Why the cohort was filtered to `rel_self == 1`

Negatives here are members with checkup/dental claims. AHC is an **employee**
benefit, so employees appear as negatives routinely while a parent or spouse
enters the data almost only by making a hospitalisation claim. Dependent
negatives are unobservable by construction: **every `rel_parent` row is positive
in all three files** (14/14, 18/18, 126/126).

Dependents were dropped, not modelled separately — there is no negative class for
them to learn against. This removed ~46% of the positive class and is the single
largest reason the event count is so low. It was still the right call: keeping
them would have produced a model that predicts "is a parent" at near-perfect
accuracy and is worthless in production.

## Why trees were dropped

LightGBM and XGBoost are not used here. **This is a sample-size decision at 42
events, not a rejection of the preference for trees on the full cohort — where
that preference is correct.** Real TPA claims will carry comorbidity
interactions that trees exploit and linear models structurally cannot.

HistGradientBoosting was run as a depth-constrained exploratory secondary. On the
sealed set it is **indistinguishable from the primary on ranking**: PR-AUC
+0.0124 [−0.0248, +0.0697], AUC-ROC −0.0243 [−0.1044, +0.0567], both paired
intervals spanning zero.

> **Brier/log-loss caveat, mandatory.** The primary uses
> `class_weight='balanced'`, which pushes its probabilities toward 0.5; HistGB is
> unweighted, so its output sits near the base rate. Comparing them on a proper
> scoring rule compares a reweighted model against an unweighted one — it is
> **not** a comparison of probability estimators, and that gap is exactly what
> sigmoid calibration exists to remove. Reading it as "HistGB is better
> calibrated than LogReg can be" is wrong.

## Coefficient instability — a finding in its own right

The pre-registered sensitivity fit (`C = 0.003`, the 1-SE choice) reached
**AUC 0.4585 — below chance** — against the primary's 0.5383, a paired difference
of +0.0544 [+0.0023, +0.1040] that excludes zero.

Two fits differing only in regularisation strength, on identical features and
identical rows, land on opposite sides of chance. That is the clearest single
statement of what 42 events buys you.

## Two coefficients are UNINTERPRETABLE

Reported with their diagnoses attached, never as protective effects.

| feature | β (std) | status |
|---|---:|---|
| `age` | +0.3235 | interpretable, sign as expected |
| `sex_male` | −0.5720 | no a-priori sign |
| `hba1c_percent` | −0.5726 | **UNINTERPRETABLE** |
| `systolic_bp_mmhg` | +0.2155 | interpretable, sign as expected |
| `bmi` | +0.2006 | interpretable, sign as expected |
| `egfr` | +0.3665 | **UNINTERPRETABLE** |
| `comorbidity_count` | +0.2213 | interpretable, sign as expected |

**`egfr` — not independently identifiable.** eGFR is the CKD-EPI function of
creatinine, age and sex; regressed on those three it gives **R² = 0.9507**, with
`corr(egfr, age) = −0.62`. Because `age` and `sex_male` are already in the model,
its coefficient flips from +0.11 alone to +0.32 once `age` enters. This is a
**specification flaw in the pre-registered feature set** — a lesson for the next
feature set, deliberately not patched mid-protocol. Do not include both `age` and
`egfr` next time.

**`hba1c_percent` — noise on top of a data defect.** Univariate r = −0.048, with
a non-monotone band gradient: 7.22% (<5.5) → 4.41% (5.5–5.7) → 11.58% (5.7–6.5) →
**0.00% (≥6.5, n=25, zero events)**. The zero-event diabetic band drives the
negative sign; ~1.8 events were expected there.

## Age coverage gap — the commercially important limitation

Training data spans ages **18–71**; the test set spans **21–51**. **The model is
unvalidated above age 51 — precisely the high-risk, high-premium segment.**

Predictions above 51 are **suppressed, never extrapolated**. Zero test rows fell
above 51, so nothing was withheld at evaluation, but the restriction binds at
serving time and means the `40+` band is validated only over ages 40–51.

## Calibration

Sigmoid (Platt), `cv=5`, fit on train. Isotonic was excluded in advance —
it overfits at 42 positives.

**No reliability curve is published.** Quintile bins on the test set hold 5–12
events each; below 10 events a bin cannot support a curve without implying
precision the data does not contain.

Aggregate: mean predicted **0.0765** vs observed **0.0927** — the model
under-predicts test prevalence, consistent with the 7.29% → 9.27% shift between
training and test periods.

## Cohort reporting

Bands `<30` / `30-39` / `40+`, cells under n=30 suppressed.

| cohort | n | events | observed | mean predicted |
|---|---:|---:|---:|---:|
| `<30 · F` | 74 | 7 | 9.46% | 0.0905 |
| `<30 · M` | 71 | 4 | 5.63% | 0.0585 |
| `30-39 · F` | 114 | 16 | 14.04% | 0.0846 |
| `30-39 · M` | 103 | 8 | 7.77% | 0.0626 |
| `40+ · F` | 28 | 2 | suppressed | suppressed |
| `40+ · M` | 20 | 1 | suppressed | suppressed |

The model orders sex correctly within both reportable age bands but inverts the
age ordering for women. At 7 and 16 events those rates carry very wide intervals;
this is not evidence of an age effect in either direction.

## Data defects — logged, not fixed

1. **`hba1c_percent` fill value (REAL data).** 168 of 576 training rows
   (**29.2%**) sit at exactly 5.4, against 38 distinct values. Consistent with a
   fill/default rather than a measurement distribution. One of seven
   pre-registered features is plausibly non-informative by construction.
2. **Float-precision fingerprint (fixture).** Synthetic rows carry 13–16 decimal
   places against 1–2 in real clinical data — `systolic_bp_mmhg` has 8,795 unique
   values across 8,815 synthetic rows versus 23 across 576 real rows. A
   synthetic-vs-real discriminator hits AUC 0.999 on this artifact alone, despite
   marginals matching closely.
3. **Injected columns (fixture).** 14 labs + 6 flags, target-independent noise,
   gated off by `INCLUDE_SYNTHETIC_LABS = False`. A tree will split on them and
   manufacture importance; that output must never reach a metric, SHAP plot,
   importance ranking or cohort percentage.
4. **Fingerprint scheme limits.** Duplicate clinical rows receive identical
   injected values (461 unique fingerprints across 627 rows), and `policy_year`
   is excluded from the fingerprint, so one member across two years gets
   identical labs. Both resolve given a `person_id` upstream.

## `train_expanded_v3` is a FIXTURE, not a training set

Its 9,373 synthetic rows carry formula-generated labels: HistGradientBoosting
recovers them at CV **AUC 0.9920 / PR-AUC 0.8947**, against 0.6170 / 0.1321 on
real rows. Fitting on it recovers the generator, not clinical signal.

Used only for shape checks, preprocessor round-trip, training-loop smoke tests
and serialisation. **No metric from it is model performance.**

An exploratory synthetic→real comparison gave AUC 0.8741 — a number that must be
read alongside the fact that the generator was likely built from those same real
rows, making the evaluation leaky, and that the float-precision discriminator
above separates the populations at AUC 0.999 on formatting alone.

## What ships

**No pricing output, regardless of the result.** The deliverable is a validated
pipeline plus an honest power assessment. A positive result would not have
upgraded it to production; this null result does not invalidate it.

The `p24`, `p36` and `expected_cost` registry slots stay empty — there are no
longitudinal labels and no amount column in this dataset, and nothing is
extrapolated from `p12`.

## For the next iteration

1. **Get more events.** 42 is the binding constraint on everything above.
2. **Drop `egfr` or `age`, not both.** Never include a derived quantity alongside
   its own inputs.
3. **Fix the HbA1c fill value at source**, or exclude the feature.
4. **Obtain dependent negatives**, or keep the model employee-only and say so.
5. **Extend age coverage above 51** before any pricing use is contemplated.
6. **Revisit trees** once the event count supports them — the preference is
   right, this dataset just cannot test it.
