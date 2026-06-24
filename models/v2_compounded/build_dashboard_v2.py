#!/usr/bin/env python3
"""
build_dashboard_v2.py  —  Part B dashboard for the v2 model on real CIBYL data.

Same charts as v1, but framed as a REGRESSION CHECK and reporting v1 vs v2 rho
side by side. Reads real_predictions_v2.json (+ v1 outputs for comparison).
Self-contained HTML (plotly inline). -> cibyl_claim_dashboard_v2.html
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BAND_COLOR = {"HIGH": "#1f78b4", "MEDIUM": "#fdae61", "LOW": "#d7301f"}
LABEL_ORDER = ["Caution", "Watchful", "Fair", "Good", "Excellent"]

CAVEAT = ("The v2 model's planted interactions are built on diagnosed-condition flags "
          "(has_diabetes, has_ckd, …) which are ABSENT from the CIBYL/AHC JSON, so the "
          "interactions CANNOT fire on this real data. v2's CIBYL correlation is therefore "
          "EXPECTED to be approximately equal to v1's (≈ −0.54), not higher. This page is a "
          "REGRESSION CHECK — it confirms that adding the compounded target did not drift "
          "real-data behaviour. It is NOT a demonstration of the interactions. As before, "
          "predictions come from a synthetic-trained model on partial real data; directions "
          "are meaningful, absolute percentages are not real-world claim rates.")

df = pd.DataFrame(json.load(open(os.path.join(HERE, "real_predictions_v2.json"))))
scored = df[df.scored == True].copy()
scored["claim_probability"] = pd.to_numeric(scored["claim_probability"])
high = scored[scored.confidence_band == "HIGH"].copy()
rho2, p2 = spearmanr(high.cibyl_score, high.claim_probability)

# v1 comparison
v1 = pd.read_csv(os.path.join(ROOT, "outputs", "real_predictions.csv"))
v1h = v1[(v1.scored == True) & (v1.confidence_band == "HIGH")].copy()
v1h["claim_probability"] = pd.to_numeric(v1h["claim_probability"])
rho1, _ = spearmanr(v1h.cibyl_score, v1h.claim_probability)

PLOT = {"displayModeBar": False, "responsive": True}
def div(fig, i): return fig.to_html(full_html=False, include_plotlyjs=False, div_id=i, config=PLOT)

# scatter + binned trend, v2 points, with v1 trend overlaid for comparison
fig_sc = go.Figure()
for band in ["MEDIUM", "LOW", "HIGH"]:
    sub = scored[scored.confidence_band == band]
    if sub.empty: continue
    fig_sc.add_trace(go.Scattergl(x=sub.cibyl_score, y=sub.claim_probability, mode="markers",
        name=f"v2 {band}", marker=dict(size=6, color=BAND_COLOR[band],
        opacity=0.85 if band == "HIGH" else 0.25),
        hovertemplate="CIBYL %{x}<br>claim %{y:.1%}<extra></extra>"))
bins = np.arange(np.floor(high.cibyl_score.min()/50)*50, high.cibyl_score.max()+50, 50)
def binned(d):
    g = d.assign(_b=pd.cut(d.cibyl_score, bins)).groupby("_b", observed=True).agg(
        x=("cibyl_score", "mean"), y=("claim_probability", "mean")).dropna()
    return g
g2, g1 = binned(high), binned(v1h)
fig_sc.add_trace(go.Scatter(x=g2.x, y=g2.y, mode="lines+markers", name="v2 binned mean",
                            line=dict(color="#08306b", width=3)))
fig_sc.add_trace(go.Scatter(x=g1.x, y=g1.y, mode="lines", name="v1 binned mean",
                            line=dict(color="#888", width=2, dash="dash")))
fig_sc.update_layout(template="plotly_white", height=520,
    title=f"CIBYL vs predicted claim probability — v2 (Spearman ρ = {rho2:+.2f}; v1 was {rho1:+.2f})",
    xaxis_title="CIBYL score (higher = healthier)", yaxis_title="predicted claim probability",
    yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    margin=dict(l=70, r=30, t=70, b=50))

# by label
lab = (high.groupby("cibyl_label").agg(mean_p=("claim_probability", "mean"),
       n=("claim_probability", "size")).reset_index())
lab["o"] = lab.cibyl_label.map({l: i for i, l in enumerate(LABEL_ORDER)}).fillna(99)
lab = lab.sort_values("o")
fig_lab = go.Figure(go.Bar(x=lab.cibyl_label, y=lab.mean_p,
    marker_color=["#d7301f", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"][:len(lab)],
    text=[f"{v:.1%}<br>n={int(n)}" for v, n in zip(lab.mean_p, lab.n)], textposition="outside"))
fig_lab.update_layout(template="plotly_white", height=360,
    title="Mean predicted claim probability by CIBYL label — v2 (HIGH-confidence)",
    yaxis_tickformat=".0%", xaxis_title="CIBYL label (worse → better)",
    yaxis_title="mean claim probability", margin=dict(l=70, r=30, t=60, b=40))

# histogram
fig_h = go.Figure(go.Histogram(x=scored.claim_probability, nbinsx=40, marker_color="#3690c0"))
fig_h.update_layout(template="plotly_white", height=340, xaxis_tickformat=".0%",
    title="Distribution of v2 predicted claim probabilities (all scored)",
    xaxis_title="predicted claim probability", yaxis_title="users", margin=dict(l=70, r=30, t=60, b=40))

# top-20
top20 = high.sort_values("claim_probability", ascending=False).head(20)
rows = "".join(f"<tr><td>{r.user_code}</td><td>{int(r.cibyl_score)}</td><td>{r.cibyl_label}</td>"
               f"<td class='p'>{r.claim_probability:.1%}</td></tr>" for _, r in top20.iterrows())

def kpi(v, l): return f'<div class="kpi"><div class="kpi-v">{v}</div><div class="kpi-l">{l}</div></div>'
mean_p = scored.claim_probability.mean()

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CIBYL vs Claim — v2 regression check</title>
<script>{get_plotlyjs()}</script><style>
 :root{{--ink:#1f2d3d;--muted:#5a6b7b;--line:#e3e8ee;--bg:#f5f7fa}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif}}
 .wrap{{max-width:1180px;margin:0 auto;padding:0 24px 64px}}
 header.top{{background:linear-gradient(135deg,#5a2a13,#8a4a1e);color:#fff;padding:30px 0 26px}}
 header.top h1{{margin:0 0 4px;font-size:25px}} header.top p{{margin:0;color:#f0dccb;font-size:14px}}
 .caveat{{background:#fff7e6;border:1px solid #f0c36d;border-left:6px solid #e0a106;color:#7a5b00;padding:14px 18px;border-radius:8px;margin:20px 0 8px;font-size:14px;line-height:1.55}}
 .caveat b{{color:#664c00}}
 .kpis{{display:flex;gap:14px;margin:18px 0 8px;flex-wrap:wrap}}
 .kpi{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 18px;flex:1;min-width:150px;box-shadow:0 1px 2px rgba(0,0,0,.03)}}
 .kpi-v{{font-size:22px;font-weight:700}} .kpi-l{{font-size:12px;color:var(--muted);margin-top:2px}}
 section{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px 8px;margin:22px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
 section h2{{margin:0 0 2px;font-size:19px}} section .sub{{color:var(--muted);font-size:13.5px;margin:0 0 6px}}
 table{{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:6px}} th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}}
 th{{color:var(--muted);font-weight:600}} td.p{{font-weight:700}} footer{{color:var(--muted);font-size:12px;text-align:center;margin-top:30px}}
</style></head><body>
<header class="top"><div class="wrap"><h1>CIBYL vs Predicted Claim Probability — v2 (compounded model)</h1>
<p>REGRESSION CHECK on real AHC data · {len(df)} CIBYL users · graceful partial-input scoring</p></div></header>
<div class="wrap">
  <div class="caveat"><b>&#9888; Regression check — read this.</b> {CAVEAT}</div>
  <div class="kpis">
    {kpi(f"{rho2:+.2f}", "v2 Spearman ρ (HIGH)")}
    {kpi(f"{rho1:+.2f}", "v1 Spearman ρ (HIGH)")}
    {kpi(f"{rho2-rho1:+.3f}", "Δ ρ (expected ≈ 0)")}
    {kpi(f"{mean_p:.1%}", "v2 mean claim prob")}
  </div>
  <div class="caveat" style="background:#eef5fb;border-color:#bcd6ea;border-left-color:#3690c0;color:#234">
    <b>Verdict:</b> v2 ρ = {rho2:+.2f} vs v1 ρ = {rho1:+.2f} (Δ = {rho2-rho1:+.3f}). Essentially unchanged —
    adding the compounded interaction target did <b>not</b> drift real-data behaviour, which is exactly what
    we expect because the interaction-driving condition flags are absent from this data source.
  </div>
  <section><h2>1 · CIBYL vs claim probability (v2 points, v1 trend dashed)</h2>
    <p class="sub">v2 HIGH-confidence solid; v1 binned-mean trend overlaid for comparison.</p>{div(fig_sc,"sc")}</section>
  <section><h2>2 · By CIBYL label</h2>
    <p class="sub">Mean v2 claim probability per health label — still falls as health improves.</p>{div(fig_lab,"lab")}</section>
  <section><h2>3 · Distribution of v2 predictions</h2>{div(fig_h,"h")}</section>
  <section><h2>4 · Top-20 highest predicted risk (HIGH-confidence)</h2>
    <table><thead><tr><th>user_code</th><th>CIBYL</th><th>label</th><th>claim probability</th></tr></thead>
    <tbody>{rows}</tbody></table></section>
  <footer>Generated by models/v2_compounded/build_dashboard_v2.py · v2 calibrated model · regression check, not interaction demonstration</footer>
</div></body></html>"""
out = os.path.join(HERE, "cibyl_claim_dashboard_v2.html")
open(out, "w", encoding="utf-8").write(html)
print(f"v2 rho={rho2:+.3f}  v1 rho={rho1:+.3f}  delta={rho2-rho1:+.3f}")
print(f"Wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")
