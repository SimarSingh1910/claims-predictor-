"""
main.py — FastAPI app for the HealthBridge Claim Engine local web app.

Loads the inference singleton ONCE at startup, exposes the registry status, and
serves metadata + a sample dataset. /api/score is stubbed this step.

Honesty rules enforced here:
  * provenance is "synthetic" everywhere; a caveat travels with the metadata.
  * the sealed test set is never read — /api/sample draws from val.csv only.
  * untrained slots are reported unavailable; nothing is fabricated.
"""
import io
import os
import math
import traceback

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .registry import registry_status, print_startup_table
from .model_service import SERVICE, ROOT, BASE_CLAIM_RATE
from .metrics import load_metrics
from . import cohorts as cohort_engine

# ID / target columns to strip when anonymising a sample.
ID_COLS = ["CUG", "employee_id", "name", "ahc_date"]
TARGET_COLS = ["claim_next_12m", "claim_count_12m", "claim_amount_inr"]

# Upload guard: reject anything larger than this (in-memory processing only).
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

CAVEAT = (
    "The current claim target is SYNTHETIC - generated from AHC features plus "
    "noise. All metrics reflect PIPELINE VALIDATION, not real-world claim rates. "
    "Real AHC + claims data (~1,500 rows) arrives shortly."
)

# Cohort definitions (display buckets; scoring stays per-individual).
COHORT_DEFS = {
    "age_groups": [
        {"key": "<30", "min": 0, "max": 29},
        {"key": "30-39", "min": 30, "max": 39},
        {"key": "40-49", "min": 40, "max": 49},
        {"key": "50-59", "min": 50, "max": 59},
        {"key": "60+", "min": 60, "max": 200},
    ],
    "genders": ["M", "F"],
    "low_n_threshold": 10,      # cohorts smaller than this are flagged low_n
    "high_risk_p12": 0.30,      # p12 above this counts as high-risk
}

app = FastAPI(title="HealthBridge Claim Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """500 handler that never leaks a stack trace to the client. The full
    traceback is logged server-side; the response is a generic message."""
    print("[500] unhandled error on", request.url.path)
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "internal server error", "detail": "an unexpected error occurred"},
    )


@app.on_event("startup")
def _startup():
    SERVICE.load()
    print_startup_table(SERVICE.registry)


@app.get("/api/health")
def health():
    """Liveness + registry availability. p12 must be available; the rest not yet."""
    return {
        "status": "ok",
        "base_claim_rate": BASE_CLAIM_RATE,
        "p12_models": SERVICE.available_p12_models(),
        "registry": [
            {"slot": e["slot"], "available": e["available"], "reason": e["reason"]}
            for e in SERVICE.registry.values()
        ],
    }


@app.get("/api/meta")
def meta():
    """Everything the UI needs to render: panel, mandatory fields, cohorts,
    feature list, provenance + caveat, and full registry status."""
    panel = SERVICE.core_panel
    feature_list = [e["feature"] for e in panel["features"]]
    return {
        "provenance": "synthetic",
        "caveat": CAVEAT,
        "base_claim_rate": BASE_CLAIM_RATE,
        "mandatory_fields": SERVICE.mandatory,
        "cohorts": COHORT_DEFS,
        "core_panel": {
            "n_features": panel["_meta"]["n_features"],
            "n_important": panel["_meta"]["n_important"],
            "important_groups": panel["important_groups"],
            "special_handling": panel.get("special_handling", {}),
        },
        "feature_list": feature_list,
        "p12_models": SERVICE.available_p12_models(),
        "default_p12_model": "xgboost",
        "registry": registry_status(SERVICE.registry),
    }


def _slot_summary():
    """The four-slot descriptor block (available/label/reason) for responses."""
    out = []
    for e in SERVICE.registry.values():
        item = {"slot": e["slot"], "available": e["available"], "label": e["label"]}
        if not e["available"]:
            item["reason"] = e["reason"]
        out.append(item)
    return out


def _clean(v):
    """Make a raw cell value JSON-safe: NaN/inf -> None, numpy scalars -> native.
    (Plain `json` would emit invalid `NaN` tokens otherwise.)"""
    if v is None:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _build_members(df, scores):
    """Per-member rows for the Members table + drawer. Includes the raw AHC values
    (in-memory, local only) so the drawer needs no extra fetch. Aligned by
    position; df and scores share a reset index."""
    id_col = next((c for c in ID_COLS if c in df.columns), None)
    hi = cohort_engine.HIGH_RISK_P12
    members = []
    for i in range(len(df)):
        sc = scores.iloc[i]
        raw = df.iloc[i]
        p12 = _clean(sc["p12"])
        ag = cohort_engine.age_group(raw.get("age"))
        g = cohort_engine.norm_gender(raw.get("sex"))
        members.append({
            "row": i,
            "id": str(raw[id_col]) if id_col and pd.notna(raw[id_col]) else f"#{i}",
            "age": _clean(raw.get("age")),
            "sex": _clean(raw.get("sex")),
            "age_group": ag,
            "gender": g,
            "cohort": f"{ag} · {g}",
            "p12": p12,
            "p24": None, "p36": None, "expected_cost": None,
            "confidence_band": sc["confidence_band"],
            "model_confidence_pct": _clean(sc["model_confidence_pct"]),
            "panel_completeness_pct": _clean(sc["panel_completeness_pct"]),
            "n_important_present": _clean(sc["n_important_present"]),
            "n_important_total": _clean(sc["n_important_total"]),
            "missing_important": sc["missing_important"],
            "high_risk": bool(p12 is not None and p12 > hi),
            "scored": bool(sc["scored"]),
            "reason": sc["reason"],
            "values": {k: _clean(raw[k]) for k in df.columns},
        })
    return members


def _validate_schema(df):
    """Return (missing, unknown). missing = mandatory columns absent (can't score
    without them). unknown = columns outside the known AHC universe. Optional
    feature columns may be absent — those are handled per-row by graceful_scorer."""
    known = set(SERVICE.core_panel_features()) | set(ID_COLS) | set(TARGET_COLS)
    cols = set(df.columns)
    missing = [c for c in SERVICE.mandatory if c not in cols]
    unknown = sorted(c for c in cols if c not in known)
    return missing, unknown


@app.post("/api/score")
async def score(file: UploadFile = File(...),
                model: str = Query("xgboost")):
    """Upload a CSV of AHC records -> score every member individually with the
    selected p12 model -> bifurcate into age x gender cohorts -> group roll-up.

    Untrained slots (p24/p36/expected_cost) stay null everywhere — never 0, never
    derived from p12. Members missing a mandatory field are refused and excluded
    from every statistic."""
    if model not in SERVICE.available_p12_models():
        raise HTTPException(400, detail={
            "error": "unknown model",
            "requested": model,
            "available": SERVICE.available_p12_models(),
        })

    # CSV-only: require a .csv extension or a csv content-type. The upload UI only
    # offers .csv, so this just rejects a mistaken drop of another file kind.
    fname = (file.filename or "").lower()
    ctype = (file.content_type or "").lower()
    if not (fname.endswith(".csv") or "csv" in ctype):
        raise HTTPException(400, detail={
            "error": "unsupported file type — upload a .csv",
            "filename": file.filename, "content_type": file.content_type,
        })

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, detail={
            "error": "file too large",
            "max_bytes": MAX_UPLOAD_BYTES,
            "got_bytes": len(raw),
        })

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, detail={"error": "could not parse CSV",
                                         "detail": str(e)})
    if len(df) == 0:
        raise HTTPException(400, detail={"error": "empty CSV (no data rows)"})

    missing, unknown = _validate_schema(df)
    if missing or unknown:
        raise HTTPException(400, detail={
            "error": "schema validation failed",
            "missing_columns": missing,      # mandatory columns absent
            "unknown_columns": unknown,      # columns outside the AHC feature set
        })

    # --- per-member inference (average OUTPUTS, never INPUTS) ---------------
    scores = SERVICE.score_frame(df, model)
    scores = scores.reset_index(drop=True)
    scores["_age"] = df["age"].reset_index(drop=True) if "age" in df.columns else None
    scores["_sex"] = df["sex"].reset_index(drop=True) if "sex" in df.columns else None

    # --- refusals: excluded from every statistic ---------------------------
    unscored = [
        {"row": int(i), "reason": r["reason"]}
        for i, r in scores.iterrows() if not r["scored"]
    ]
    scored_df = scores[scores["scored"]].copy()

    # --- aggregation --------------------------------------------------------
    cohorts = cohort_engine.build_cohorts(scored_df)
    rollup = cohort_engine.group_rollup(scored_df, BASE_CLAIM_RATE)
    dist = cohort_engine.risk_distribution(scored_df)

    return {
        "model_used": model,
        "provenance": "synthetic",
        "slots": _slot_summary(),
        "summary": {
            "total_rows": int(len(df)),
            "scored": int(len(scored_df)),
            "unscored": int(len(unscored)),
            "p12": rollup["p12"],
            "p24": rollup["p24"],
            "p36": rollup["p36"],
            "expected_cost": rollup["expected_cost"],
            "expected_claimants": rollup["expected_claimants"],
            "high_risk_pct": rollup["high_risk_pct"],
            "mean_confidence": rollup["mean_confidence"],
            "base_claim_rate": BASE_CLAIM_RATE,
        },
        "cohorts": cohorts,
        "risk_distribution": dist,
        "members": _build_members(df, scores),
        "unscored": unscored,
        "caveat": CAVEAT,
    }


@app.post("/api/score-one")
def score_one(member: dict = Body(...), model: str = Query("xgboost")):
    """Score a single member from a raw AHC JSON dict — powers the manual-entry
    scorer ("what if HbA1c were 9?"). Same engine as the batch path. A blank
    mandatory field yields a refusal, not a guess."""
    if model not in SERVICE.available_p12_models():
        raise HTTPException(400, detail={
            "error": "unknown model",
            "requested": model,
            "available": SERVICE.available_p12_models(),
        })
    r = SERVICE.score_one(member, model)
    ag = cohort_engine.age_group(member.get("age"))
    g = cohort_engine.norm_gender(member.get("sex"))
    p12 = _clean(r["p12"])
    return {
        "model_used": model,
        "provenance": "synthetic",
        "slots": _slot_summary(),
        "age_group": ag,
        "gender": g,
        "cohort": f"{ag} · {g}",
        "p12": p12,
        "p24": None, "p36": None, "expected_cost": None,
        "panel_completeness_pct": _clean(r["panel_completeness_pct"]),
        "model_confidence_pct": _clean(r["model_confidence_pct"]),
        "confidence_band": r["confidence_band"],
        "n_important_present": _clean(r["n_important_present"]),
        "n_important_total": _clean(r["n_important_total"]),
        "missing_important": r["missing_important"],
        "high_risk": bool(p12 is not None and p12 > cohort_engine.HIGH_RISK_P12),
        "scored": r["scored"],
        "reason": r["reason"],
        "caveat": CAVEAT,
    }


@app.get("/api/metrics")
def metrics(model: str = Query("xgboost")):
    """Read-only per-model metrics (headline / calibration / decile) from the
    model's OWN metrics file. Never recomputed, never cross-substituted."""
    return load_metrics(model)


@app.get("/api/sample")
def sample():
    """200 anonymised rows from val.csv as a downloadable CSV. Never touches
    test.csv. ID and target columns are stripped so the sample is upload-shaped."""
    val_path = os.path.join(ROOT, "splits", "val.csv")
    df = pd.read_csv(val_path, nrows=200)
    df = df.drop(columns=[c for c in ID_COLS + TARGET_COLS if c in df.columns])

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=healthbridge_sample_200.csv"},
    )
