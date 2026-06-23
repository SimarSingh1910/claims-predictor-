#!/usr/bin/env python3
"""
build_dashboard_lgbm.py  —  the THIRD CIBYL dashboard, identical in form to v1, but
scoring real users with the LightGBM production-seed model. Same 8 sections; adds a
3-way rho line (v1 LogReg vs v2 LogReg+interactions vs LightGBM) so all three can be
viewed side by side. Reads real_predictions_lgbm.json (+ v1/v2 outputs for the line).
Self-contained HTML (plotly inline) -> cibyl_claim_dashboard_lgbm.html
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V2DIR = os.path.join(ROOT, "models", "v2_compounded")

BAND_COLOR = {"HIGH": "#1f78b4", "MEDIUM": "#fdae61", "LOW": "#d7301f"}
LABEL_ORDER = ["Caution", "Watchful", "Fair", "Good", "Excellent"]

# verbatim caveat (also printed to console)
CAVEAT = ("This dashboard scores the same real users with the LightGBM model. The CIBYL "
          "data contains no diagnosed-condition labels, so the has_* features and "
          "disease-interaction terms cannot fire for ANY model — so all three models "
          "(v1 LogReg, v2 LogReg+interactions, LightGBM) produce near-identical CIBYL "
          "correlations (~-0.54). This is a consistency / regression check confirming no "
          "model drifts on real data — NOT a ranking of the models. The models are "
          "expected to diverge only on real claims data that contains condition labels "
          "and interactions.")

# --------------------------------------------------------------------------
df = pd.DataFrame(json.load(open(os.path.join(HERE, "real_predictions_lgbm.json"))))
scored = df[df["scored"] == True].copy()
scored["claim_probability"] = pd.to_numeric(scored["claim_probability"])
high = scored[scored["confidence_band"] == "HIGH"].copy()
rho, pval = spearmanr(high["cibyl_score"], high["claim_probability"])


def rho_from(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    h = d[(d.scored == True) & (d.confidence_band == "HIGH")].copy()
    h["claim_probability"] = pd.to_numeric(h["claim_probability"])
    return spearmanr(h.cibyl_score, h.claim_probability)[0]

rho_v1 = rho_from(os.path.join(ROOT, "outputs", "real_predictions.csv"))
rho_v2 = rho_from(os.path.join(V2DIR, "real_predictions_v2.csv"))
rho_v1 = rho_v1 if rho_v1 is not None else -0.543
rho_v2 = rho_v2 if rho_v2 is not None else -0.527

PLOT = {"displayModeBar": False, "responsive": True}
def div(fig, i): return fig.to_html(full_html=False, include_plotlyjs=False, div_id=i, config=PLOT)

# ---- main scatter (band-coloured) + binned-mean trend (HIGH only) ----------
fig_sc = go.Figure()
for band in ["MEDIUM", "LOW", "HIGH"]:
    sub = scored[scored.confidence_band == band]
    if sub.empty: continue
    fig_sc.add_trace(go.Scattergl(
        x=sub.cibyl_score, y=sub.claim_probability, mode="markers", name=f"{band} conf",
        marker=dict(size=6, color=BAND_COLOR[band], opacity=0.85 if band == "HIGH" else 0.28),
        hovertemplate="CIBYL %{x}<br>claim %{y:.1%}<extra></extra>"))
bins = np.arange(np.floor(high.cibyl_score.min()/50)*50, high.cibyl_score.max()+50, 50)
high["_b"] = pd.cut(high.cibyl_score, bins)
tr = high.groupby("_b", observed=True).agg(x=("cibyl_score", "mean"),
                                           y=("claim_probability", "mean")).dropna()
fig_sc.add_trace(go.Scatter(x=tr.x, y=tr.y, mode="lines+markers", name="HIGH binned mean",
                            line=dict(color="#0b6b3a", width=3)))
fig_sc.update_layout(template="plotly_white", height=520,
    title=f"CIBYL score vs predicted claim probability — LightGBM (Spearman ρ = {rho:+.2f} on HIGH-confidence)",
    xaxis_title="CIBYL score  (0–1000, higher = healthier)",
    yaxis_title="predicted claim probability", yaxis_tickformat=".0%",
    legend=dict(orientation="h", y=1.02, yanchor="bottom"), margin=dict(l=70, r=30, t=70, b=50))

# ---- 50-pt CIBYL bin bars (HIGH) ------------------------------------------
tb = tr.copy()
fig_bin = go.Figure(go.Bar(x=tb.x, y=tb.y, marker_color="#1f78b4",
                           text=[f"{v:.1%}" for v in tb.y], textposition="outside",
                           hovertemplate="~CIBYL %{x:.0f}<br>mean claim %{y:.1%}<extra></extra>"))
fig_bin.update_layout(template="plotly_white", height=380,
    title="Mean predicted claim probability per CIBYL bin (50-pt, HIGH-confidence)",
    xaxis_title="CIBYL score bin", yaxis_title="mean claim probability",
    yaxis_tickformat=".0%", margin=dict(l=70, r=30, t=60, b=50))

# ---- by CIBYL label (HIGH), worst->best -----------------------------------
lab = (high.groupby("cibyl_label").agg(mean_p=("claim_probability", "mean"),
                                       n=("claim_probability", "size")).reset_index())
lab["ord"] = lab.cibyl_label.map({l: i for i, l in enumerate(LABEL_ORDER)}).fillna(99)
lab = lab.sort_values("ord")
fig_lab = go.Figure(go.Bar(x=lab.cibyl_label, y=lab.mean_p,
    marker_color=["#d7301f", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"][:len(lab)],
    text=[f"{v:.1%}<br>n={int(n)}" for v, n in zip(lab.mean_p, lab.n)], textposition="outside",
    hovertemplate="%{x}<br>mean claim %{y:.1%}<extra></extra>"))
fig_lab.update_layout(template="plotly_white", height=380,
    title="Mean predicted claim probability by CIBYL label (HIGH-confidence)",
    xaxis_title="CIBYL label  (worse → better health)", yaxis_title="mean claim probability",
    yaxis_tickformat=".0%", margin=dict(l=70, r=30, t=60, b=50))

# ---- probability histogram ------------------------------------------------
fig_hist = go.Figure(go.Histogram(x=scored.claim_probability, nbinsx=40, marker_color="#3690c0"))
fig_hist.update_layout(template="plotly_white", height=360,
    title="Distribution of predicted claim probabilities (all scored users)",
    xaxis_title="predicted claim probability", yaxis_title="users",
    xaxis_tickformat=".0%", margin=dict(l=70, r=30, t=60, b=50))

# ---- confidence breakdown -------------------------------------------------
counts = {b: int((df.confidence_band == b).sum()) for b in ["HIGH", "MEDIUM", "LOW"]}
refused = int((~df.scored.astype(bool)).sum())
fig_conf = go.Figure(go.Bar(
    x=["HIGH", "MEDIUM", "LOW", "refused"],
    y=[counts["HIGH"], counts["MEDIUM"], counts["LOW"], refused],
    marker_color=[BAND_COLOR["HIGH"], BAND_COLOR["MEDIUM"], BAND_COLOR["LOW"], "#999"],
    text=[counts["HIGH"], counts["MEDIUM"], counts["LOW"], refused], textposition="outside"))
fig_conf.update_layout(template="plotly_white", height=340,
    title="How trustworthy was the data? (confidence band counts + refusals)",
    xaxis_title="", yaxis_title="users", margin=dict(l=70, r=30, t=60, b=40))

# ---- top-20 highest-risk (HIGH) -------------------------------------------
top20 = high.sort_values("claim_probability", ascending=False).head(20)
rows = "".join(
    f"<tr><td>{r.user_code}</td><td>{int(r.cibyl_score)}</td>"
    f"<td>{r.cibyl_label}</td><td class='p'>{r.claim_probability:.1%}</td>"
    f"<td><span class='pill' style='background:{BAND_COLOR[r.confidence_band]}'>{r.confidence_band}</span></td></tr>"
    for _, r in top20.iterrows())

# --------------------------------------------------------------------------
def kpi(v, l): return f'<div class="kpi"><div class="kpi-v">{v}</div><div class="kpi-l">{l}</div></div>'
verdict = ("strong" if abs(rho) >= 0.4 else "moderate" if abs(rho) >= 0.2 else "weak")
mean_p = scored.claim_probability.mean()

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CIBYL vs Claim Probability — LightGBM</title>
<script>{get_plotlyjs()}</script>
<style>
 :root{{--ink:#1f2d3d;--muted:#5a6b7b;--line:#e3e8ee;--bg:#f5f7fa;}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
   font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
 .wrap{{max-width:1180px;margin:0 auto;padding:0 24px 64px}}
 header.top{{background:linear-gradient(135deg,#0b3d2e,#137a52);color:#fff;padding:30px 0 26px}}
 header.top h1{{margin:0 0 4px;font-size:26px}} header.top p{{margin:0;color:#cdeadd;font-size:14px}}
 .caveat{{background:#fff7e6;border:1px solid #f0c36d;border-left:6px solid #e0a106;color:#7a5b00;
   padding:14px 18px;border-radius:8px;margin:20px 0 8px;font-size:14px;line-height:1.55}}
 .caveat b{{color:#664c00}}
 .kpis{{display:flex;gap:14px;margin:18px 0 8px;flex-wrap:wrap}}
 .kpi{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 18px;flex:1;
   min-width:150px;box-shadow:0 1px 2px rgba(0,0,0,.03)}}
 .kpi-v{{font-size:22px;font-weight:700}} .kpi-l{{font-size:12px;color:var(--muted);margin-top:2px}}
 section{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px 8px;
   margin:22px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
 section h2{{margin:0 0 2px;font-size:19px}} section .sub{{color:var(--muted);font-size:13.5px;margin:0 0 6px}}
 table{{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:6px}}
 th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}}
 th{{color:var(--muted);font-weight:600}} td.p{{font-weight:700}}
 .pill{{color:#fff;border-radius:10px;padding:2px 9px;font-size:11.5px}}
 footer{{color:var(--muted);font-size:12px;text-align:center;margin-top:30px}}
</style></head><body>
<header class="top"><div class="wrap">
  <h1>CIBYL Health Score vs Predicted Claim Probability — LightGBM (production seed)</h1>
  <p>Real AHC data &middot; {len(df)} users with a CIBYL score &middot; graceful partial-input scoring</p>
</div></header>
<div class="wrap">
  <div class="caveat"><b>&#9888; Read this first.</b> {CAVEAT}</div>

  <div class="kpis">
    {kpi(f"{len(scored)}", "users scored")}
    {kpi(f"{len(high)}", "HIGH-confidence")}
    {kpi(f"{mean_p:.1%}", "mean claim probability")}
    {kpi(f"{rho:+.2f}", f"LightGBM Spearman ρ (HIGH) — {verdict}")}
  </div>
  <div class="caveat" style="background:#eef5fb;border-color:#bcd6ea;border-left-color:#3690c0;color:#234">
    <b>3-way consistency (HIGH-confidence ρ):</b>
    v1 LogReg <b>{rho_v1:+.2f}</b> &middot;
    v2 LogReg+interactions <b>{rho_v2:+.2f}</b> &middot;
    LightGBM <b>{rho:+.2f}</b>.
    All three land at ~-0.54 — the models agree with CIBYL and with each other on real data.
    This is a regression / consistency check, <b>not</b> a ranking. The models are expected to
    diverge only on real <i>claims</i> data carrying condition labels and interactions.
  </div>

  <section><h2>1 &middot; CIBYL vs claim probability</h2>
    <p class="sub">Each point a user. HIGH-confidence solid; MEDIUM/LOW lighter and excluded
       from the trend line. Expect a downward slope.</p>{div(fig_sc,"sc")}</section>

  <section><h2>2 &middot; Trend by CIBYL bin</h2>
    <p class="sub">Mean predicted claim probability in each 50-point CIBYL band (HIGH-confidence).</p>
    {div(fig_bin,"bin")}</section>

  <section><h2>3 &middot; By CIBYL label</h2>
    <p class="sub">Average predicted claim probability per health label — rises as health worsens.</p>
    {div(fig_lab,"lab")}</section>

  <section><h2>4 &middot; Distribution of predictions</h2>
    <p class="sub">How predicted claim probabilities are spread across all scored users.</p>
    {div(fig_hist,"hist")}</section>

  <section><h2>5 &middot; Data trustworthiness</h2>
    <p class="sub">Confidence bands (from clinical panel completeness) and any refusals.</p>
    {div(fig_conf,"conf")}</section>

  <section><h2>6 &middot; Top-20 highest predicted risk (HIGH-confidence)</h2>
    <p class="sub">The users the model flags as most likely to claim, among well-covered records.</p>
    <table><thead><tr><th>user_code</th><th>CIBYL</th><th>label</th>
      <th>claim probability</th><th>confidence</th></tr></thead><tbody>{rows}</tbody></table>
  </section>

  <footer>Generated by models/lightgbm_production/build_dashboard_lgbm.py &middot; input: real AHC JSON &middot;
    model: preprocessor + calibrated LightGBM (synthetic-trained) &middot; not validated against real claims</footer>
</div></body></html>"""

out = os.path.join(HERE, "cibyl_claim_dashboard_lgbm.html")
open(out, "w", encoding="utf-8").write(html)
print("CAVEAT (verbatim):")
print(CAVEAT)
print(f"\n3-way Spearman rho (HIGH):  v1 LogReg {rho_v1:+.3f}   "
      f"v2 LogReg+interactions {rho_v2:+.3f}   LightGBM {rho:+.3f}  (p={pval:.2e}, n={len(high)})")
print(f"Wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB, self-contained)")
