"""
cohorts.py — bifurcation + aggregation engine.

THE AGGREGATION RULE (non-negotiable): average the OUTPUTS, never the INPUTS.
Every member is scored individually upstream (model_service.score_frame). Here we
only bin already-computed per-member scores into cohorts and average those. We
never build a synthetic "average member" and score it — the model is non-linear, so
score(mean(x)) != mean(score(x)), and averaging inputs erases the high-risk tail
the engine exists to detect.

Null-safety: an unavailable slot (p24/p36/expected_cost today) aggregates to null,
never 0, never derived from p12.
"""
import math

import numpy as np
import pandas as pd

AGE_BINS = [(0, 30, "<30"), (30, 40, "30-39"), (40, 50, "40-49"),
            (50, 60, "50-59"), (60, 200, "60+")]
MIN_COHORT_N = 10
HIGH_RISK_P12 = 0.30
N_RISK_BUCKETS = 10


def age_group(age):
    """Map a numeric age to its bin label. None/NaN -> 'unknown'."""
    if age is None or (isinstance(age, float) and math.isnan(age)):
        return "unknown"
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "unknown"
    for lo, hi, label in AGE_BINS:
        if lo <= a < hi:
            return label
    return "unknown"


def norm_gender(sex):
    """Normalise a sex value to 'M' / 'F' / 'unknown'."""
    if sex is None:
        return "unknown"
    s = str(sex).strip().upper()
    if s in ("M", "MALE"):
        return "M"
    if s in ("F", "FEMALE"):
        return "F"
    return "unknown"


def _mean_or_none(series):
    """Mean of non-null values, or None if the whole column is null (unavailable
    slot). Never returns 0 to stand in for 'no model'."""
    s = series.dropna()
    return float(s.mean()) if len(s) else None


def _band_from_pct(pct):
    return "HIGH" if pct >= 80 else "MEDIUM" if pct >= 50 else "LOW"


def _dominant_band(bands):
    """Most common confidence band in a cohort (ties -> highest severity present)."""
    order = ["HIGH", "MEDIUM", "LOW"]
    counts = {b: 0 for b in order}
    for b in bands:
        if b in counts:
            counts[b] += 1
    best = max(counts.values()) if counts else 0
    if best == 0:
        return None
    for b in order:                       # deterministic tie-break by declared order
        if counts[b] == best:
            return b
    return None


def _cohort_stats(sub):
    """Aggregate one cohort's already-scored members (rows where scored==True)."""
    n = len(sub)
    p12 = sub["p12"].dropna()
    mean_p12 = float(p12.mean()) if len(p12) else None
    high_risk_count = int((sub["p12"] > HIGH_RISK_P12).sum())
    return {
        "n": n,
        "mean_p12": mean_p12,
        "median_p12": float(p12.median()) if len(p12) else None,
        "p12_std": float(p12.std(ddof=0)) if len(p12) else None,
        # Null-safe: unavailable slots average to None, not 0.
        "p24": _mean_or_none(sub["p24"]),
        "p36": _mean_or_none(sub["p36"]),
        "expected_cost": _mean_or_none(sub["expected_cost"]),
        # Actuarially meaningful headline figure.
        "expected_claimants": int(round(n * mean_p12)) if mean_p12 is not None else None,
        "high_risk_count": high_risk_count,
        "pct_high_risk": round(100.0 * high_risk_count / n, 1) if n else None,
        "mean_confidence": round(float(sub["model_confidence_pct"].mean()), 1) if n else None,
        "confidence_band": _dominant_band(sub["confidence_band"].tolist()),
        "low_n": n < MIN_COHORT_N,
    }


def build_cohorts(scored_df):
    """
    scored_df: score_frame output joined with the raw age/sex, restricted to
    scored==True members. Returns the list of cohort dicts (age_group x gender).
    """
    df = scored_df.copy()
    df["age_group"] = df["_age"].map(age_group)
    df["gender"] = df["_sex"].map(norm_gender)

    cohorts = []
    for (ag, g), sub in df.groupby(["age_group", "gender"], sort=True):
        stats = _cohort_stats(sub)
        cohorts.append({"age_group": ag, "gender": g, **stats})
    # Order by age-bin order then gender for stable display.
    ag_order = {label: i for i, (_, _, label) in enumerate(AGE_BINS)}
    ag_order["unknown"] = len(ag_order)
    cohorts.sort(key=lambda c: (ag_order.get(c["age_group"], 99), c["gender"]))
    return cohorts


def group_rollup(scored_df, base_claim_rate):
    """Whole-population roll-up over scored members. Same aggregation rule:
    mean of individual probabilities."""
    n = len(scored_df)
    p12 = scored_df["p12"].dropna()
    mean_p12 = float(p12.mean()) if len(p12) else None
    high_risk_count = int((scored_df["p12"] > HIGH_RISK_P12).sum())
    return {
        "scored": n,
        "p12": mean_p12,
        "p24": _mean_or_none(scored_df["p24"]),
        "p36": _mean_or_none(scored_df["p36"]),
        "expected_cost": _mean_or_none(scored_df["expected_cost"]),
        "expected_claimants": int(round(n * mean_p12)) if mean_p12 is not None else None,
        "high_risk_count": high_risk_count,
        "high_risk_pct": round(high_risk_count / n, 4) if n else None,
        "mean_confidence": round(float(scored_df["model_confidence_pct"].mean()), 1) if n else None,
        "base_claim_rate": base_claim_rate,
    }


def risk_distribution(scored_df):
    """p12 histogram, 10 fixed buckets 0-10% .. 90-100%. Refused members (p12
    None) are excluded."""
    edges = np.linspace(0.0, 1.0, N_RISK_BUCKETS + 1)
    p12 = scored_df["p12"].dropna().to_numpy()
    counts, _ = np.histogram(p12, bins=edges)
    out = []
    for i in range(N_RISK_BUCKETS):
        lo, hi = int(round(edges[i] * 100)), int(round(edges[i + 1] * 100))
        out.append({"bucket": f"{lo}-{hi}%", "count": int(counts[i])})
    return out
