"""
registry.py — the Model Registry. The heart of the design.

Four prediction slots. Some are trained; some are NOT yet. At startup we check
which artifacts EXIST on disk. A present artifact makes its slot available and is
loaded once. A missing artifact marks the slot unavailable with the honest reason
"model not trained yet".

Non-negotiable: we NEVER synthesise, derive, or fall back for a missing slot. When
a future joblib is dropped into its registry path, that slot lights up on the next
startup with zero code changes anywhere else. That is the acceptance criterion.
"""
import os
import joblib

# Project root = parent of this api/ directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_REGISTRY = {
    "p12": {
        "path": "models/xgboost_production/xgb_calibrated_model.joblib",
        "label": "12-month claim probability",
        "kind": "classifier", "unit": "probability",
    },
    "p24": {
        "path": "models/xgboost_production/xgb_24m_calibrated.joblib",
        "label": "24-month claim probability",
        "kind": "classifier", "unit": "probability",
    },
    "p36": {
        "path": "models/xgboost_production/xgb_36m_calibrated.joblib",
        "label": "36-month claim probability",
        "kind": "classifier", "unit": "probability",
    },
    "expected_cost": {
        "path": "models/severity/severity_model.joblib",
        "label": "Expected claim cost",
        "kind": "regressor", "unit": "INR",
    },
}

NOT_TRAINED = "model not trained yet"


def load_registry():
    """
    Resolve every slot against disk. Returns an ordered dict:
        slot -> {label, kind, unit, path, available, reason, model}

    `model` is the loaded artifact when available, else None. No fallbacks, no
    derivations — a missing artifact stays empty.
    """
    resolved = {}
    for slot, spec in MODEL_REGISTRY.items():
        abspath = os.path.join(ROOT, spec["path"])
        entry = {
            "slot": slot,
            "label": spec["label"],
            "kind": spec["kind"],
            "unit": spec["unit"],
            "path": spec["path"],
            "available": False,
            "reason": NOT_TRAINED,
            "model": None,
        }
        if os.path.exists(abspath):
            try:
                entry["model"] = joblib.load(abspath)
                entry["available"] = True
                entry["reason"] = "ok"
            except Exception as e:  # artifact present but unreadable — honest failure
                entry["available"] = False
                entry["reason"] = f"failed to load: {type(e).__name__}: {e}"
        resolved[slot] = entry
    return resolved


def registry_status(resolved):
    """Public status view (no model objects) for API responses."""
    return [
        {"slot": e["slot"], "label": e["label"], "kind": e["kind"],
         "unit": e["unit"], "path": e["path"],
         "available": e["available"], "reason": e["reason"]}
        for e in resolved.values()
    ]


def print_startup_table(resolved):
    """Log the startup registry table: slot | available | path."""
    print("\n" + "=" * 68)
    print("MODEL REGISTRY")
    print("=" * 68)
    print(f"{'slot':<14} {'available':<11} {'path'}")
    print("-" * 68)
    for e in resolved.values():
        mark = "yes" if e["available"] else "no"
        print(f"{e['slot']:<14} {mark:<11} {e['path']}")
        if not e["available"]:
            print(f"{'':<14} {'':<11} -> {e['reason']}")
    print("=" * 68 + "\n")
