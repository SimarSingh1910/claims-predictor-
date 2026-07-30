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

### 1.1 Amendment — pre-specified `C` sensitivity analysis

*Added 2026-07-30, BEFORE the sealed read. Not a deviation.*

The tie-break rule above was under-specified and never fired: only one exact tie
existed at the maximum, so plain argmax selected **C = 10** — the *weakest*
regularisation in the grid, at 6 events per feature. That is the opposite of the
rule's intent. The CV surface is flat (mean CV PR-AUC 0.1060 → 0.1151 across four
orders of magnitude, spread 0.0091), and eight of nine grid values sit within
1 SD of the best; a 1-SE rule would have selected **C = 0.003**.

**The registered primary remains C = 10.** It is not changed after the fact.

**C = 0.003 is added as a pre-specified sensitivity analysis**, computed at the
same single evaluation and reported alongside the primary. It does **not**
replace the primary and does **not** license a second evaluation — both numbers
come out of the one permitted pass over the sealed set.

If the two diverge materially on the sealed set, that divergence is itself
reported as a finding about **coefficient instability at this sample size**, not
resolved by choosing whichever looks better.

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

### 2.1 Two coefficients are declared UNINTERPRETABLE in advance

*Added 2026-07-30, BEFORE the sealed read. The feature set is UNCHANGED — these
features stay in the model; only their coefficients are barred from
interpretation.*

**`egfr` — not independently identifiable.** eGFR is the CKD-EPI function of
creatinine, age and sex. Regressed on those three it gives **R² = 0.9507**, and
`corr(egfr, age) = −0.62`. Because `age` and `sex_male` are already in the model,
eGFR is ~95% determined by variables already present; its coefficient flips from
+0.11 alone to +0.32 once `age` is added. This is a **specification flaw in the
pre-registered feature set** — jointly owned by whoever chose and whoever
approved it — and it is a lesson for the next feature set, not a defect patched
mid-protocol.

**`hba1c_percent` — noise, compounded by a data-source defect.** Univariate
r = −0.048, with a non-monotone band gradient: 7.22% (<5.5) → 4.41% (5.5–5.7) →
11.58% (5.7–6.5) → **0.00% (≥6.5, n=25, zero events)**. The zero-event diabetic
band drives the negative sign; ~1.8 events were expected there. See §12.4 for the
fill-value defect that plausibly explains the whole pattern.

Both coefficients are reported **with these diagnoses attached** and are never
presented as protective effects.

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

### 4.1 Mandatory caveat on any LogReg-vs-HistGB proper-scoring comparison

*Added 2026-07-30, BEFORE the sealed read.*

Whenever Brier or log-loss are compared between the primary and the exploratory
model, **this paragraph travels with the table**:

> The primary LogReg uses `class_weight='balanced'`, which pushes its predicted
> probabilities toward 0.5; HistGB is unweighted, so its output sits near the
> base rate. Comparing the two on a proper scoring rule compares a reweighted
> model against an unweighted one — it is **not** a comparison of probability
> estimators, and the gap is precisely what the §5 sigmoid calibration exists to
> remove. The interval may be real; reading it as "HistGB is better calibrated
> than LogReg can be" is wrong.

Without this paragraph the table reads as HistGB winning. It is not optional.

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

### 12.4 `hba1c_percent` fill value — a defect in the REAL data, not the fixture

*Added 2026-07-30, BEFORE the sealed read.*

**168 of 576 training rows (29.2%) carry `hba1c_percent` exactly 5.4**, against
38 distinct values overall. A single value holding nearly a third of the mass is
consistent with a **fill/default**, not with a measurement distribution.

**Consequence, stated plainly:** one of the seven pre-registered features is
substantially non-informative *by construction*. This plausibly explains both the
near-zero univariate correlation (r = −0.048) and the non-monotone band gradient
recorded in §2.1 — the coefficient may be estimating nothing more than the
difference between "measured" and "defaulted".

**The feature set is unchanged.** This is logged as a data-source defect for the
next iteration, and it means the effective number of informative features is
plausibly six, not seven — worsening an events-per-feature ratio that was already
below convention.

---

## 13. RESULTS — to be filled after evaluation, blanks only

### 13.1 Cross-validation on train (no test data involved)

| item | value |
|---|---|
| selected `C` | **10** (flat surface: 0.1060 → 0.1151 across the grid, spread 0.0091) |
| CV PR-AUC (mean [95% CI]) | **0.1199  [0.0832, 0.2036]** — train baseline 0.0729 |
| CV AUC-ROC (mean [95% CI]) | **0.5869  [0.4993, 0.6709]** — includes 0.50 |
| CV Brier (mean [95% CI]) | 0.2312  [0.2201, 0.2429] |
| CV log-loss (mean [95% CI]) | 0.6563  [0.6303, 0.6838] |

Fold-level spread (50 folds, 8–9 events each): PR-AUC 0.0673–0.4073,
AUC-ROC 0.4209–0.7868. Any single-fold figure is meaningless at this event count.

Coefficients (standardised scale, primary model):

| feature | coefficient | direction as expected? |
|---|---|---|
| `age` | +0.3235 | yes |
| `sex_male` | −0.5720 | n/a — no a-priori sign |
| `hba1c_percent` | −0.5726 | **no — UNINTERPRETABLE, see §2.1 + §12.4** |
| `systolic_bp_mmhg` | +0.2155 | yes |
| `bmi` | +0.2006 | yes |
| `egfr` | +0.3665 | **no — UNINTERPRETABLE, see §2.1** |
| `comorbidity_count` | +0.2213 | yes |
| intercept | +0.0857 | — |

Signs are directionally stable across the whole `C` grid; regularisation shrinks
magnitude without correcting either contradiction.

### 13.2 Sealed test evaluation — ONE run

Test baseline positive rate: **9.27%** (38/410). PR-AUC is stated against this,
**not** against the 7.29% training baseline.

Primary is sigmoid-calibrated per §5. Run once, 2026-07-30.

| metric | primary LogReg (C=10) | sensitivity LogReg (C=0.003) | exploratory HistGB |
|---|---|---|---|
| PR-AUC [95% CI] | **0.1101  [0.0877, 0.1784]** | 0.0957  [0.0754, 0.1587] | 0.0998  [0.0844, 0.1332] |
| AUC-ROC [95% CI] | **0.5383  [0.4420, 0.6348]** | 0.4585  [0.3547, 0.5642] | 0.5371  [0.4511, 0.6264] |
| Brier [95% CI] | 0.0848  [0.0829, 0.0864] | 0.2443  [0.2406, 0.2480] | 0.0870  [0.0853, 0.0887] |
| log-loss [95% CI] | 0.3132  [0.3026, 0.3228] | 0.6817  [0.6743, 0.6893] | 0.3301  [0.3150, 0.3459] |

Primary uncalibrated, for transparency: PR-AUC 0.1060 [0.0826, 0.1773],
AUC-ROC 0.5123 [0.4181, 0.6100], Brier 0.2384 [0.2246, 0.2522],
log-loss 0.6714 [0.6399, 0.7033].

**Primary vs sensitivity — material divergence? YES**, on AUC-ROC:
+0.0544 [+0.0023, +0.1040], interval excludes zero. PR-AUC, Brier and log-loss
do not separate them. The sensitivity fit lands at **AUC 0.4585 — below chance**.
Per §1.1 this is reported as a finding about **coefficient instability at this
sample size**: two fits differing only in regularisation strength, on the same
seven features and the same 576 rows, land on opposite sides of chance. It is
**not** resolved by preferring the primary because it looks better.

**Paired bootstrap, primary vs HistGB** — no distinguishable ranking difference:
PR-AUC +0.0124 [−0.0248, +0.0697], AUC-ROC −0.0243 [−0.1044, +0.0567], both
intervals include zero. Brier +0.1511 [+0.1380, +0.1649] and log-loss
+0.3407 [+0.3101, +0.3724] exclude zero — **§4.1 caveat applies and is not
optional**: that gap compares a `class_weight='balanced'` model against an
unweighted one, not two probability estimators.

**Calibration:** sigmoid applied per §5. **No reliability curve is drawn.**
Quintile bins hold 5–12 events each; at least one bin is below 10, which cannot
support a curve without implying precision the data does not contain. Aggregate
calibration: mean predicted **0.0765** against observed **0.0927** — the model
under-predicts the test prevalence.

**Age suppression (§6): 0 rows** above age 51 in the test set (range 21–51), so
no prediction was withheld here. The restriction still binds at serving time, and
it means the 40+ band is validated only over ages 40–51.

### 13.3 Cohort table (real bands, n ≥ 30 only, age ≤ 51)

| cohort | n | events | observed rate | mean predicted | suppressed? |
|---|---:|---:|---:|---:|---|
| `<30 · F` | 74 | 7 | 9.46% | 0.0905 | no |
| `<30 · M` | 71 | 4 | 5.63% | 0.0585 | no |
| `30-39 · F` | 114 | 16 | 14.04% | 0.0846 | no |
| `30-39 · M` | 103 | 8 | 7.77% | 0.0626 | no |
| `40+ · F` | 28 | 2 | — | — | **YES, n < 30** |
| `40+ · M` | 20 | 1 | — | — | **YES, n < 30** |

4 reportable cells, 2 suppressed. The model orders the four reportable cells
correctly by sex (F above M in both age bands) but does not reproduce the
observed ordering across age bands: it predicts `<30 · F` (0.0905) above
`30-39 · F` (0.0846) while the observed rates run the other way (9.46% vs
14.04%). At 7 and 16 events those cell rates carry very wide intervals, so this
is not evidence of a systematic age effect in either direction.

### 13.4 Verdict against §9

Does the AUC-ROC interval include 0.50? **YES** — [0.4420, 0.6348].
Does the PR-AUC interval include 0.0927? **YES** — [0.0877, 0.1784].

**Conclusion: both pre-registered null triggers fired. §9 applies as written.**

> "A confidence interval on AUC-ROC that includes 0.50, or a PR-AUC interval
>  that includes the 9.27% baseline, is a pre-specified possible outcome. It
>  means 42 training events and 38 test events are insufficient to detect
>  signal at this sample size. It does NOT mean AHC parameters lack predictive
>  value for hospitalisation, and it does NOT mean the modelling approach is
>  wrong. Those questions are not answerable with this dataset and remain
>  open pending real TPA claims data at scale.
>  Reported either way, unchanged."

This is the outcome the recorded prior expectation in §9 anticipated: train CV
gave AUC 0.5869 with a lower bound of 0.4993, so a test interval spanning 0.50
is the expected result, not a surprise.

### 13.5 Exploratory synthetic→real comparison (§11)

Value: **AUC 0.8741**, from fitting on the synthetic fixture and evaluating on
the real *training* rows — a number that must be read alongside the fact that the
generator was likely built from those same real rows, making the evaluation
leaky, and that a synthetic-vs-real discriminator reaches AUC 0.999 on a
float-precision fingerprint alone, so the two populations are trivially
separable to a model.

No synthetic→*test* variant was computed: it was not pre-registered, and adding
an unregistered analysis to the single sealed pass is precisely what this
protocol exists to prevent.

---

## 14. DEVIATIONS FROM THIS PROTOCOL

Any deviation is recorded here with date, what changed, and why. Nothing above
is edited to accommodate it.

**2026-07-30, after the sealed evaluation: NONE. No deviations occurred.**

The single permitted evaluation ran once and completed without error. Every
model, metric, band, threshold and suppression rule was as registered. Nothing
was re-run, re-selected or re-specified after the sealed read.

For the record, two things that are explicitly *not* deviations:

* **A dry run preceded the sealed evaluation.** `src/v3_sealed_eval.py --dry-run`
  substitutes `train_expanded_v3` for the test file so every code path executes
  before the sealed set is opened. That is the fixture's registered purpose
  (§0), it read no test data, and its numbers were discarded. It caught one bug:
  cohort mean-predicted was averaging over rows the §6 age rule had suppressed.
  Fixed before the sealed read — the test set contains no member above 51, so
  the path never triggered in the real run regardless.
* **The §1.1 sensitivity fit and the primary came out of the same single pass**,
  as registered. No second evaluation occurred.
