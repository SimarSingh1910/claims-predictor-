"""
model_service.py — the inference singleton. Loaded ONCE at startup, never per
request. Nothing is trained here; we only load pre-fitted artifacts and call them.

Responsibilities:
  * own the Model Registry (which slots exist on disk).
  * load the shared preprocessor + core_panel via the existing graceful_scorer.
  * hold the THREE alternate frequency models for the p12 slot
    ({"xgboost" (default), "lightgbm", "logreg"}) so the UI's model selector can
    swap which model scores p12 — preprocessor/calibration basis stay shared.
  * score_frame(df, p12_model_key) -> per-row DataFrame with all four slots
    (nulls where a slot's model is untrained), plus completeness/confidence.

We REUSE graceful_scorer.score_member (imported, not reimplemented). It refuses
members missing the mandatory minimum, derives computable clinical values, imputes
the rest from train medians/modes, and reports panel completeness + model
confidence. We drive it by swapping its cached resource bundle per p12 model.

Note on confidence weighting: model_confidence_pct is importance-weighted coverage
of present features, and each p12 model weights it by ITS OWN importances — LogReg
by abs(coef_), XGBoost/LightGBM by feature_importances_ (gain). We read those off
the already-trained models ONCE at startup (READ-ONLY; nothing is refit) and pass
the active model's weight vector into graceful_scorer per request, so the metric
describes the model that is actually scoring — not a fixed foreign basis.
"""
import os
import sys
import copy
import time

import pandas as pd

from .registry import load_registry, ROOT

# Make src/ importable so we can reuse graceful_scorer + features unchanged.
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import graceful_scorer as gs  # noqa: E402  (reused wholesale; not reimplemented)

# Shared artifacts (all selectors) — point graceful_scorer at the root pipeline.
PREPROCESSOR_PATH = os.path.join(ROOT, "preprocessor.joblib")
FREQ_COEF_BASIS_PATH = os.path.join(ROOT, "frequency_model.joblib")  # LogReg coefs
CORE_PANEL_PATH = os.path.join(ROOT, "core_panel.json")
FEATURE_NAMES_PATH = os.path.join(ROOT, "feature_names.json")
TRAIN_CSV_PATH = os.path.join(ROOT, "splits", "train.csv")

# The three p12 frequency models. Each entry's calibrated model swaps into the
# scoring slot; the preprocessor + coef basis stay shared.
P12_MODELS = {
    "xgboost": os.path.join(ROOT, "models/xgboost_production/xgb_calibrated_model.joblib"),
    "lightgbm": os.path.join(ROOT, "models/lightgbm_production/tree_calibrated_model.joblib"),
    "logreg": os.path.join(ROOT, "calibrated_model.joblib"),
}
DEFAULT_P12 = "xgboost"

BASE_CLAIM_RATE = 0.1604

# Output columns of score_frame, in order. The trailing three carry per-member
# detail used by the Members table / drawer; cohort aggregation ignores them.
SCORE_COLUMNS = [
    "p12", "p24", "p36", "expected_cost",
    "panel_completeness_pct", "model_confidence_pct", "confidence_band",
    "scored", "reason",
    "missing_important", "n_important_present", "n_important_total",
]


class ModelService:
    """Singleton holding the registry + per-selector resource caches."""

    def __init__(self):
        self.registry = None          # slot -> entry (with loaded model or None)
        self._caches = {}             # p12 key -> graceful_scorer resource bundle
        self.core_panel = None
        self.mandatory = None

    # -- startup ----------------------------------------------------------
    def load(self):
        """Load everything ONCE. Called at FastAPI startup."""
        t0 = time.perf_counter()
        self.registry = load_registry()

        # Pin graceful_scorer to the shared root pipeline + LogReg coef basis.
        gs.PREPROCESSOR_PATH = PREPROCESSOR_PATH
        gs.FREQ_MODEL_PATH = FREQ_COEF_BASIS_PATH
        gs.CORE_PANEL_PATH = CORE_PANEL_PATH
        gs.FEATURE_NAMES_PATH = FEATURE_NAMES_PATH
        gs.TRAIN_CSV_PATH = TRAIN_CSV_PATH

        # Build one resource bundle per available p12 model. Bundles share the
        # preprocessor / medians / feature mapping and differ in the scoring model
        # AND in the confidence weighting basis (each model's own importances).
        for key, model_path in P12_MODELS.items():
            if not os.path.exists(model_path):
                continue
            gs._R.clear()
            gs.MODEL_PATH = model_path
            gs._resources()                      # populates gs._R for this model
            bundle = dict(gs._R)                  # snapshot the bundle
            self._build_conf_basis(key, bundle)  # weight vector from THIS model
            self._caches[key] = bundle
        gs._R.clear()

        if DEFAULT_P12 not in self._caches:
            raise RuntimeError(
                f"default p12 model '{DEFAULT_P12}' missing at {P12_MODELS[DEFAULT_P12]}"
            )

        # Expose core panel / mandatory list (from the default bundle).
        bundle = self._caches[DEFAULT_P12]
        self.core_panel = bundle["panel"]
        self.mandatory = bundle["mandatory"]
        dt = time.perf_counter() - t0
        print(f"[model_service] p12 models loaded: {sorted(self._caches)} "
              f"(default={DEFAULT_P12}) in {dt:.2f}s — loaded ONCE at startup")

    # -- per-model confidence basis --------------------------------------
    def _build_conf_basis(self, key, bundle):
        """Read THIS model's own importances (READ-ONLY) into a weight vector for
        model_confidence, aligned positionally to feat_names. Stores a feature->weight
        dict on the bundle when usable, else marks the model's confidence unavailable
        (no silent fallback to another model's weights). Length mismatch FAILS LOUD
        inside importance_vector — a silent off-by-one would misattribute every weight."""
        feat_names = bundle["feat_names"]
        n = len(feat_names)
        w, reason = gs.importance_vector(bundle["model"], n)   # raises on length mismatch
        if w is None:
            bundle["conf_available"] = False
            bundle["conf_reason"] = reason
            bundle["conf_weights"] = None
            print(f"[model_service] confidence basis for '{key}': UNAVAILABLE — {reason}")
            return
        est = gs._base_estimator(bundle["model"])
        basis = "feature_importances_ (gain)" if hasattr(est, "feature_importances_") else "abs(coef_)"
        bundle["conf_available"] = True
        bundle["conf_reason"] = None
        bundle["conf_basis"] = basis
        bundle["conf_weights"] = {feat_names[i]: float(w[i]) for i in range(n)}
        n_nonzero = int(sum(1 for v in w if v != 0.0))
        print(f"[model_service] confidence basis for '{key}': {basis} — "
              f"len={len(w)} matches feature count={n} "
              f"({n_nonzero}/{n} non-zero, sum={float(w.sum()):.4f})")

    def confidence_basis(self):
        """Per-model confidence availability + basis, for API/meta transparency."""
        out = {}
        for key, b in self._caches.items():
            out[key] = {
                "available": bool(b.get("conf_available")),
                "basis": b.get("conf_basis"),
                "reason": b.get("conf_reason"),
            }
        return out

    # -- introspection ----------------------------------------------------
    def available_p12_models(self):
        return sorted(self._caches)

    def slot_available(self, slot):
        return bool(self.registry and self.registry[slot]["available"])

    def core_panel_features(self):
        """The 115 canonical AHC feature names, in panel order."""
        return [e["feature"] for e in self.core_panel["features"]]

    # -- scoring ----------------------------------------------------------
    # Registry-driven extra slots. p12 always comes from the selected frequency
    # bundle (with graceful refusal logic). p24 / p36 / expected_cost are produced
    # ONLY when their registry artifact exists — driven purely by which joblib is on
    # disk. Drop a trained model at a slot's registry path and it scores live here
    # with no other code change; remove it and the slot goes empty again.
    EXTRA_SLOTS = ("p24", "p36", "expected_cost")

    def _prepare_X(self, df: pd.DataFrame, bundle: dict):
        """Build the model matrix exactly as graceful_scorer does (reindex to the
        115 features, coerce, derive, impute, transform) so extra-slot models see
        the same inputs p12 did."""
        R = bundle
        frame = df.reindex(columns=R["model_features"]).reset_index(drop=True)
        for c in (R["num_cols"] & set(frame.columns)):
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
        frame = gs._derive(frame)
        frame = gs._impute(frame, R)
        return R["pre"].transform(frame)

    def _score_extra_slots(self, df, bundle, scored_flags):
        """Return {slot: [value|None per row]} for every AVAILABLE extra slot.
        Refused rows (scored_flags[i] False) get None — no slot is scored for a
        member we refused. Unavailable slots are absent from the dict entirely."""
        need = [s for s in self.EXTRA_SLOTS if self.slot_available(s)]
        if not need:
            return {}
        X = self._prepare_X(df, bundle)
        out = {}
        for slot in need:
            entry = self.registry[slot]
            model = entry["model"]
            if entry["kind"] == "regressor":
                preds = model.predict(X)
            else:  # classifier -> P(claim within window)
                preds = model.predict_proba(X)[:, 1]
            out[slot] = [
                float(preds[i]) if scored_flags[i] else None
                for i in range(len(scored_flags))
            ]
        return out

    def score_frame(self, df: pd.DataFrame, p12_model_key: str = DEFAULT_P12) -> pd.DataFrame:
        """
        Score every row individually with the selected p12 model, returning a
        DataFrame aligned to df.index with SCORE_COLUMNS.

        p12 comes from graceful_scorer. p24 / p36 / expected_cost are filled ONLY
        if their registry slot is trained; otherwise they stay None (never
        invented, never derived from p12).
        """
        key = p12_model_key if p12_model_key in self._caches else DEFAULT_P12
        bundle = self._caches[key]
        # Point graceful_scorer at the selected resource bundle, then score with
        # THIS model's own confidence weights (None => confidence unavailable).
        gs._R = bundle
        rows = gs.score_member(df.reset_index(drop=True), weights=bundle["conf_weights"])
        if isinstance(rows, dict):   # single-row guard (df always -> list here)
            rows = [rows]

        extra = self._score_extra_slots(df, bundle, [r["scored"] for r in rows])
        out = [self._row_to_slots(r, extra, i) for i, r in enumerate(rows)]
        return pd.DataFrame(out, columns=SCORE_COLUMNS, index=df.index)

    def _row_to_slots(self, r: dict, extra: dict = None, i: int = 0) -> dict:
        """Shape one graceful_scorer result into the four-slot + detail record.
        Extra slots are filled from `extra` when their model is available, else
        stay None — never invented, never derived from p12."""
        extra = extra or {}
        return {
            "p12": r["claim_probability"],              # None when refused
            "p24": extra.get("p24", [None])[i] if "p24" in extra else None,
            "p36": extra.get("p36", [None])[i] if "p36" in extra else None,
            "expected_cost": extra.get("expected_cost", [None])[i] if "expected_cost" in extra else None,
            "panel_completeness_pct": r["panel_completeness_pct"],
            "model_confidence_pct": r["model_confidence_pct"],
            "confidence_band": r["confidence_band"],
            "scored": r["scored"],
            "reason": r["reason"],
            "missing_important": r.get("missing_important", []),
            "n_important_present": r.get("n_important_present"),
            "n_important_total": r.get("n_important_total"),
        }

    def score_one(self, member: dict, p12_model_key: str = DEFAULT_P12) -> dict:
        """Score a single member (raw AHC dict) for the manual-entry scorer. Same
        engine as the batch path; returns the four slots + completeness /
        confidence / missing-important detail."""
        key = p12_model_key if p12_model_key in self._caches else DEFAULT_P12
        bundle = self._caches[key]
        gs._R = bundle
        r = gs.score_member(dict(member), weights=bundle["conf_weights"])   # dict in -> dict out
        one = pd.DataFrame([dict(member)])
        extra = self._score_extra_slots(one, bundle, [r["scored"]])
        return self._row_to_slots(r, extra, 0)


# Module-level singleton.
SERVICE = ModelService()
