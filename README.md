# HealthBridge Claim Engine — local web app

A locally-run claim-propensity & group-pricing tool for HCL Healthcare's
HealthBridge platform. Upload a CSV of employee Annual Health Checkup (AHC)
records; every member is scored individually by a **pre-trained** model, then
aggregated into age × gender cohorts and a group-level roll-up. Nothing is trained
at request time.

> **Synthetic-data caveat.** The current claim target is synthetic (generated from
> the AHC features by a formula + noise). All metrics reflect *pipeline validation*,
> not real-world claim rates. Real AHC + claims data arrives later; the pipeline is
> built so real claims drop in with no structural change.

---

## Architecture

```
  React + Vite + TypeScript          FastAPI (uvicorn)              on disk
  ─────────────────────────          ────────────────              ───────
  Upload · Cohorts · Members  ──▶   /api/score      ──▶   preprocessor.joblib
  About            (fetch/JSON)      /api/score-one         Model Registry:
                                     /api/metrics             p12  → xgb_calibrated_model.joblib
  MetricCard (skeleton mechanism)    /api/meta                p24  → (awaiting)
  drives every metric render         /api/sample              p36  → (awaiting)
                                     /api/health              expected_cost → (awaiting)
```

- **Backend** loads the joblib artifacts **once at startup** and calls them; no DB,
  uploads are processed **in memory** and never persisted.
- **Model Registry** (`api/registry.py`) resolves four prediction slots against
  disk. A present artifact makes its slot live; a missing one is reported
  `available: false` and renders an "Awaiting model" skeleton in the UI. Untrained
  slots return `null` — never `0`, never a value derived from another slot.
- **Frequency model** for `p12` is swappable (`xgboost` default · `lightgbm` ·
  `logreg`) via the top-bar selector; preprocessing and calibration are shared.

---

## How to run

**Prerequisites**

- Python via **Anaconda** (has pandas / scikit-learn / xgboost / lightgbm); the
  backend also needs `fastapi`, `uvicorn`, `python-multipart`.
- Node 18+ (for the Vite dev server).

**Start both servers**

```bash
./run.sh        # macOS / Linux / Git Bash
run.bat         # Windows
```

- Backend → http://localhost:8000  (health: `/api/health`)
- Frontend → http://localhost:5173

Then open the frontend, click **Download sample dataset** (or drop your own CSV),
and **Score dataset**.

> Tailwind is loaded via its Play CDN and charts are dependency-free inline SVG, so
> the frontend needs no build-time CSS/chart packages.

## Sample data

`GET /api/sample` returns 200 anonymised rows drawn from `splits/val.csv` (ID and
target columns stripped). The sealed `splits/test.csv` is **never** read by any
code path. The Upload page's "Download sample dataset" button uses this endpoint.

---

## Adding a future model (zero frontend/backend code change)

The whole design exists so a new model lights up its slot by **existing on disk**:

1. **Train** the model against its target (e.g. `claim_next_24m` for `p24`, or
   `claim_amount_inr` for the severity model), using the shared
   `preprocessor.joblib` feature pipeline.
2. **Save** it to the slot's registry path:

   | slot            | target             | path                                                        |
   | --------------- | ------------------ | ----------------------------------------------------------- |
   | `p12`           | `claim_next_12m`   | `models/xgboost_production/xgb_calibrated_model.joblib` ✅   |
   | `p24`           | `claim_next_24m`   | `models/xgboost_production/xgb_24m_calibrated.joblib`       |
   | `p36`           | `claim_next_36m`   | `models/xgboost_production/xgb_36m_calibrated.joblib`       |
   | `expected_cost` | `claim_amount_inr` | `models/severity/severity_model.joblib`                     |

   Classifier slots must expose `predict_proba`; the severity slot uses `predict`.
3. **Restart** the API. The registry picks up the new artifact, the slot goes live,
   and its MetricCard / cohort aggregates / charts fill in — with no edit to the
   backend or frontend.

Verify the mechanism end to end with `python test_step9.py` (it drops a dummy
joblib at `p24`, confirms it scores live, then removes it).
