#!/usr/bin/env python3
"""
test_step9.py — hardening/verification tests for the Claim Engine backend.

Run under the Anaconda python (has pandas/sklearn/xgboost + fastapi):
    python test_step9.py

Covers:
  1. No code path READS splits/test.csv (sealed set).
  2. An unavailable slot returns value=None everywhere — never 0, never derived.
  3. THE registry test: drop a dummy joblib at p24's path, reload, confirm p24
     scores LIVE with zero code change; remove it, confirm it goes empty again.
  4. Models load once at startup; scoring 1,000 rows is fast.
"""
import os
import re
import glob
import time
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((ok, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------
# 1. sealed test set: no source file READS splits/test.csv
# --------------------------------------------------------------------------
def test_no_test_csv_read():
    print("\n1. Sealed test set — no code path reads test.csv")
    read_re = re.compile(r"(read_csv|open)\s*\([^)]*test\.csv", re.I)
    offenders = []
    for base in ("api", "src"):
        for path in glob.glob(os.path.join(HERE, base, "*.py")):
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if read_re.search(line):
                        offenders.append(f"{os.path.relpath(path, HERE)}:{i}")
    check("no read of splits/test.csv in api/ or src/", not offenders,
          "clean" if not offenders else ", ".join(offenders))


# --------------------------------------------------------------------------
# 2 & 3 & 4 use the in-process service
# --------------------------------------------------------------------------
def load_service_fresh():
    """Import + (re)load SERVICE, picking up whatever joblibs are on disk now."""
    from api.model_service import SERVICE
    SERVICE.load()
    return SERVICE


def sample_frame(n=None):
    df = pd.read_csv(os.path.join(HERE, "splits", "val.csv"))
    df = df.drop(columns=[c for c in
                          ["claim_next_12m", "claim_count_12m", "claim_amount_inr",
                           "CUG", "employee_id", "name", "ahc_date"]
                          if c in df.columns])
    return df if n is None else df.head(n)


def test_unavailable_slots_null(SERVICE):
    print("\n2. Unavailable slots -> None (never 0, never derived)")
    df = sample_frame(200)
    scores = SERVICE.score_frame(df, "xgboost")
    ok = True
    for slot in ("p24", "p36", "expected_cost"):
        col = scores[slot]
        # every value must be exactly None — not 0, not equal to p12
        bad = [v for v in col if v is not None]
        if bad:
            ok = False
            check(f"{slot} all None", False, f"{len(bad)} non-null values")
        else:
            check(f"{slot} all None", True)
    # and p12 IS populated (so we know the frame really scored)
    check("p12 populated (sanity)", scores["p12"].notna().any())
    return ok


def test_registry_swap(SERVICE):
    print("\n3. Registry swap — drop a dummy joblib at p24's path")
    from api.registry import MODEL_REGISTRY
    p24_path = os.path.join(HERE, MODEL_REGISTRY["p24"]["path"])
    p12_path = os.path.join(HERE, MODEL_REGISTRY["p12"]["path"])

    # precondition: p24 empty
    SERVICE = load_service_fresh()
    check("p24 unavailable before drop", not SERVICE.slot_available("p24"))
    before = SERVICE.score_frame(sample_frame(20), "xgboost")
    check("p24 values None before drop", before["p24"].isna().all())

    created = False
    try:
        # DROP: copy the trained p12 calibrated model to p24's registry path.
        os.makedirs(os.path.dirname(p24_path), exist_ok=True)
        shutil.copyfile(p12_path, p24_path)
        created = True

        # RESTART: reload the service — this is the only "restart" needed; NO
        # frontend edit, NO code edit.
        SERVICE = load_service_fresh()
        check("p24 available after drop", SERVICE.slot_available("p24"))

        after = SERVICE.score_frame(sample_frame(20), "xgboost")
        live = after["p24"].notna().sum()
        check("p24 scores LIVE after drop", live == 20, f"{live}/20 rows have p24")
        # values must be genuine probabilities in [0,1]
        vals = [v for v in after["p24"] if v is not None]
        in_range = all(0.0 <= v <= 1.0 for v in vals)
        check("p24 values are valid probabilities", in_range)
    finally:
        # REMOVE the dummy and confirm the slot goes empty again.
        if created and os.path.exists(p24_path):
            os.remove(p24_path)
    SERVICE = load_service_fresh()
    check("p24 unavailable after removal", not SERVICE.slot_available("p24"))
    gone = SERVICE.score_frame(sample_frame(20), "xgboost")
    check("p24 values None after removal", gone["p24"].isna().all())


def test_scoring_speed(SERVICE):
    print("\n4. Scoring speed — 1,000 rows")
    df = sample_frame()
    # tile up to 1000 rows
    reps = (1000 // len(df)) + 1
    big = pd.concat([df] * reps, ignore_index=True).head(1000)
    t0 = time.perf_counter()
    SERVICE.score_frame(big, "xgboost")
    dt = time.perf_counter() - t0
    check("scored 1,000 rows < 5s", dt < 5.0, f"{dt*1000:.0f} ms")


if __name__ == "__main__":
    print("=" * 64)
    print("STEP 9 — hardening / verification tests")
    print("=" * 64)

    test_no_test_csv_read()

    t0 = time.perf_counter()
    SERVICE = load_service_fresh()
    print(f"\n(startup load took {(time.perf_counter()-t0):.2f}s — models loaded ONCE)")

    test_unavailable_slots_null(SERVICE)
    test_registry_swap(SERVICE)
    test_scoring_speed(SERVICE)

    n_fail = sum(1 for ok, *_ in results if not ok)
    print("\n" + "=" * 64)
    print(f"RESULT: {len(results)-n_fail}/{len(results)} passed, {n_fail} failed")
    print("=" * 64)
    raise SystemExit(1 if n_fail else 0)
