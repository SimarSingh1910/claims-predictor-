# Pre-registration — v3 real-data hospitalisation model

**Status:** COMMITTED BEFORE EVALUATION.
**Written:** 2026-07-30

### Exactly what has been read from the sealed test set

Stating this precisely, because "sealed" is a claim that has to be auditable
rather than asserted.

**Has been read** — during the approved step-2 filter report, for descriptive
baseline statistics only:

* the positive count and rate (38 / 410 = 9.27%),
* age × gender cohort cell sizes and their observed rates,
* feature columns (no target) for the train/test transform-shift diff in step 3.

**Has not happened:** no model has been fit on it, scored against it, or
evaluated on it; no feature, hyperparameter, model family, threshold or
reporting choice in this document was informed by it. The 9.27% baseline appears
below because a PR-AUC is meaningless without the prevalence it is measured
against — it is the denominator of the primary metric, not a result.

The single evaluation in §8 remains untouched and unrun.
**Target:** `had_hospitalisation` (binary). This is **NOT** "claim" — the legacy
`claim_next_12m` target, the `BASE_CLAIM_RATE = 0.1604` constant, and all
"claim probability" copy in the app belong to a different target on a different
population and are not comparable.

## The rule this document exists to enforce

Everything below is fixed **before** the sealed test set is opened. After
evaluation, results are written into the RESULTS section and **nothing else in
this file changes**. Any departure from this protocol is written into the
DEVIATIONS section as a deviation — it is never edited into the plan to make the
plan look prescient.

---

## 0. Data and provenance

| split | file | rows (post-clean) | positives | rate |
|---|---|---:|---:|---:|
| train | `train_real_v3.csv` | 576 | 42 | 7.29% |
| test (SEALED) | `test_real_v3.csv` | 410 | 38 | **9.27%** |
| fixture | `train_expanded_v3.csv` | 9,391 | — | — |

Cleaning applied to every split: `rel_self == 1`, `age >= 18`.

**Why the `rel_self` filter.** Negatives in this dataset are members with
checkup/dental claims. AHC is an employee benefit, so employees appear as
negatives routinely, while a parent or spouse enters the data almost only by
making a hospitalisation claim. Dependent negatives are unobservable by
construction — every `rel_parent` row is positive in all three files (14/14,
18/18, 126/126). That 100% is a sampling artifact, not a clinical fact.
Dependents are dropped, not modelled separately: there is no negative class for
them to learn against. `rel_self`/`rel_spouse`/`rel_parent` are then constant
and are dropped as features.

**`train_expanded_v3` is a FIXTURE, not a training set.** Its 9,373 synthetic
rows carry formula-generated labels: HistGradientBoosting recovers them at
CV AUC 0.9920 / PR-AUC 0.8947, versus 0.6170 / 0.1321 on the real rows. A model
fit on it recovers the generator, not clinical signal. It is used **only** for
shape checks, preprocessor round-trip, training-loop smoke tests, serialisation
and API-contract validation. **No metric derived from it may be reported as
model performance.**

**Preprocessor.** `models/real_v3/preprocessor_v3.joblib`, already fit on train
only and frozen. 38 output features (27 continuous median-imputed + standardised,
11 binary zero-imputed and passed through unscaled). The 20 injected columns
(14 labs + 6 flags) are gated off by `INCLUDE_SYNTHETIC_LABS = False`.

### 0.1 Limitation — the sealing is blind for selection, not strictly blind

The consequence of the target read documented in the header, stated plainly:

**The test set is blind for model selection but not strictly blind.** The 9.27%
prevalence was already known when the primary metric was chosen. Choosing PR-AUC
as primary is a defensible choice for an imbalanced problem and would have been
made from the training prevalence alone (7.29%) — but it was not made in
ignorance of the test prevalence, and this document does not claim otherwise.

No feature, hyperparameter, model family, threshold or fold assignment was
informed by test data. The residual exposure is confined to the choice of which
metric to headline, and to the cohort-band decision in §7, which used test cell
counts to establish that finer bands would be unreportable.

---

## 1. Primary model

L2-penalised Logistic Regression, `class_weight='balanced'`, `random_state=42`,
`max_iter=2000`, solver `lbfgs`.

`C` tuned by **repeated stratified 5-fold CV, 10 repeats** (50 fits) on
`train_real_v3` only, selecting on mean CV PR-AUC. Pre-specified grid:

```
C ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}
```

Ties broken toward **stronger** regularisation (smaller `C`).

**Leakage control in CV.** Imputation and scaling are refit **inside each CV
fold** — the frozen artifact is not used for cross-validation, because its
scaler saw all 576 training rows and would leak distributional information
across folds. The frozen artifact is used for the final single test evaluation,
where fitting on the full training set and applying to held-out test is correct.

**CV metrics reported here are selection-optimistic.** `C` is tuned on the same
folds that report the cross-validated performance, so the reported CV figures
are biased upward by the selection itself. Nested cross-validation — the correct
remedy — is not viable at 42 positives: the inner folds would hold roughly 1–2
events each, and the resulting estimate would be noise. The bias is therefore
accepted and declared rather than removed.

**The sealed-test result in §13.2 carries no such bias**, and is the number that
should be believed over any CV figure in §13.1.

## 2. Primary feature set — fixed now, on clinical grounds

Seven features, chosen a priori. **Not selected by searching the training data.**
42 positives ÷ 7 features = **6 events per feature** (below the conventional
10–20 guideline; see §9).

| feature | clinical rationale |
|---|---|
| `age` | Strongest demographic driver of admission risk; monotone in essentially every morbidity model. |
| `sex_male` | Sex differences in admission patterns and in interaction with age. |
| `hba1c_percent` | Glycaemic control. Diabetes is a leading driver of admissions. |
| `systolic_bp_mmhg` | Hypertension — the cardiovascular and renal admission pathway. |
| `bmi` | Obesity; upstream of metabolic and surgical risk. |
| `egfr` | Renal function. CKD is among the strongest single predictors of admission. |
| `comorbidity_count` | Multimorbidity burden. Verified uncontaminated: reconstructs to 97.5% from six *real* flags, every injected flag correlates at \|r\| < 0.021, and it shows a monotone gradient with the target (2.89% → 16.67% across counts 0→5). |

The preprocessor emits 38 columns; the model consumes these 7. The remaining 31
are available but **not used by the primary model**, and no post-hoc swap is
permitted after seeing test results.

## 3. Secondary model — exploratory only

`HistGradientBoostingClassifier`, same 7 features, depth-constrained and
pre-specified: `max_depth=3`, `max_iter=100`, `learning_rate=0.05`,
`min_samples_leaf=20`, `l2_regularization=1.0`, `random_state=42`.

Reported with confidence intervals, labelled exploratory. **Never the
deliverable.** LightGBM and XGBoost are dropped for this dataset — 42 events
cannot support them. This is a sample-size decision, **not** a rejection of the
preference for trees on the full cohort, where that preference is correct.

## 4. Metrics

Primary: **PR-AUC**, always stated against the **9.27%** test baseline.
Secondary: AUC-ROC, Brier, log-loss.

All four reported with **2,000-iteration bootstrap percentile confidence
intervals** (stratified resampling of the evaluation set, `random_state=42`).

**A point estimate never appears without its interval on the same line.**

## 5. Calibration

Platt/sigmoid via `CalibratedClassifierCV(method='sigmoid', cv=5)` fit on train.
Isotonic is excluded: it overfits at 42 positives.

If the calibration curve cannot be meaningfully estimated on 38 test positives,
**uncalibrated probabilities are reported and labelled as such** — no curve is
drawn from bins that contain single-digit counts.

## 6. Age restriction

Training data spans ages 18–71; the test set spans **21–51**. The model is
**unvalidated above age 51** — which is precisely the high-risk, high-premium
segment.

Predictions above age 51 are **suppressed in all outputs, never extrapolated**.
This is surfaced in the UI and the model card, not buried in a footnote.

## 7. Cohort reporting

Real data: **`<30` / `30-39` / `40+`**. Cells with n < 30 are suppressed.
The finer five-band split is retained for the fixture only, clearly labelled
synthetic.

## 8. Evaluation discipline

**ONE** evaluation on `test_real_v3`. No reruns. No reselection of features,
hyperparameters, threshold or model family after seeing test results. If the
result is disappointing, it is reported disappointing.

---

## 9. Pre-registered framing of a null result

> "A confidence interval on AUC-ROC that includes 0.50, or a PR-AUC interval
>  that includes the 9.27% baseline, is a pre-specified possible outcome. It
>  means 42 training events and 38 test events are insufficient to detect
>  signal at this sample size. It does NOT mean AHC parameters lack predictive
>  value for hospitalisation, and it does NOT mean the modelling approach is
>  wrong. Those questions are not answerable with this dataset and remain
>  open pending real TPA claims data at scale.
>  Reported either way, unchanged."

**Prior expectation, recorded before evaluation:** cross-validated performance
on the training data is AUC 0.5699 (LogReg) and 0.6170 (HistGB) across all 38
features. A null or near-null test result is therefore the *expected* outcome,
not a surprise to be explained away afterwards.

## 10. Decision rule

**No pricing output ships from this model regardless of the result.**

The Phase 4–7 deliverable is a validated pipeline plus an honest power
assessment. A positive result does not upgrade it to production. A null result
does not invalidate it.

## 11. Exploratory secondary comparison — pre-registered as exploratory

Fitting on the synthetic fixture and evaluating on the real training rows gave
AUC 0.8741, higher than real-on-real CV at 0.6170.

**This number may only ever be reported with both confounds named in the same
sentence:** the generator was likely built from those same real rows, so the
evaluation is leaky; and a synthetic-vs-real discriminator reaches AUC 0.999 on
a float-precision fingerprint alone, so the two populations are trivially
distinguishable to a model. It is exploratory. It is not evidence that training
on the fixture is sound.

## 12. Known fixture defects — logged, not fixed now

1. **Float-precision fingerprint.** Synthetic rows carry 13–16 decimal places;
   real clinical values carry 1–2. `systolic_bp_mmhg` has **8,795 unique values
   across 8,815 synthetic rows** versus **23 unique values across 576 real
   rows**. A discriminator separates synthetic from real at AUC 0.999 on this
   artifact alone, despite marginal distributions matching closely (largest
   standardised mean difference 0.26).
2. **Target-independent injected columns.** 14 labs + 6 derived flags, bootstrap
   -sampled from empirical distributions, are noise with respect to the target.
   Gated off by `INCLUDE_SYNTHETIC_LABS = False`. A tree will split on them and
   manufacture apparent importance; that output is not interpretable and must
   not reach any metric, SHAP plot, importance ranking, model card or cohort
   percentage.
3. **Fingerprint scheme limits** (per the data owner): duplicate clinical rows
   receive identical injected values (461 unique fingerprints across 627 rows),
   and `policy_year` is excluded from the fingerprint, so the same member across
   two years receives identical lab values. Both resolve if a `person_id`
   becomes available upstream.

---

## 13. RESULTS — to be filled after evaluation, blanks only

### 13.1 Cross-validation on train (no test data involved)

| item | value |
|---|---|
| selected `C` | _____ |
| CV PR-AUC (mean [95% CI]) | _____ |
| CV AUC-ROC (mean [95% CI]) | _____ |
| CV Brier (mean [95% CI]) | _____ |
| CV log-loss (mean [95% CI]) | _____ |

Coefficients (standardised scale, primary model):

| feature | coefficient | direction as expected? |
|---|---|---|
| `age` | _____ | _____ |
| `sex_male` | _____ | _____ |
| `hba1c_percent` | _____ | _____ |
| `systolic_bp_mmhg` | _____ | _____ |
| `bmi` | _____ | _____ |
| `egfr` | _____ | _____ |
| `comorbidity_count` | _____ | _____ |
| intercept | _____ | — |

### 13.2 Sealed test evaluation — ONE run

Test baseline positive rate: **9.27%** (38/410).

| metric | primary LogReg | exploratory HistGB |
|---|---|---|
| PR-AUC [95% CI] | _____ | _____ |
| AUC-ROC [95% CI] | _____ | _____ |
| Brier [95% CI] | _____ | _____ |
| log-loss [95% CI] | _____ | _____ |

Calibration: _____ (sigmoid applied / uncalibrated + reason)

### 13.3 Cohort table (real bands, n ≥ 30 only, age ≤ 51)

| cohort | n | observed rate | mean predicted | suppressed? |
|---|---|---|---|---|
| _____ | _____ | _____ | _____ | _____ |

### 13.4 Verdict against §9

Does the AUC-ROC interval include 0.50? _____
Does the PR-AUC interval include 0.0927? _____
Conclusion: _____

### 13.5 Exploratory synthetic→real comparison (§11)

Value: _____ — reported only with both confounds named alongside.

---

## 14. DEVIATIONS FROM THIS PROTOCOL

Any deviation is recorded here with date, what changed, and why. Nothing above
is edited to accommodate it.

_(none at time of commit)_
