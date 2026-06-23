#!/usr/bin/env python3
"""
correlation_dashboard.py  —  manager-facing AHC <-> claim correlation dashboard.

Outputs a single self-contained HTML (outputs/correlation_dashboard.html) with
plotly.js embedded INLINE, so it opens in any browser offline, no server.

Two complementary views:
  1) UNIVARIATE correlations  — descriptive point-biserial / Pearson r of each
     ORIGINAL AHC parameter (+ has_X condition flags) with claim_next_12m.
  2) MODEL coefficients       — what the trained LogReg actually weighted. Because
     Phase 1 StandardScaled the features, coefficients are standardized and their
     magnitudes are directly comparable across features.

DATA: splits/train.csv ONLY.  test.csv is never touched.
"""
import os, sys, json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from features import CHRONIC_MAP            # token -> has_X column name

TARGET = "claim_next_12m"
TARGETS_ALL = ["claim_next_12m", "claim_count_12m", "claim_amount_inr"]
IDS = ["CUG", "employee_id", "name", "ahc_date"]
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# colours: one for "raises claim risk", one for "lowers it"
C_RAISE = "#d6604d"   # warm red  -> positive correlation, more likely to claim
C_LOWER = "#4393c3"   # cool blue -> negative correlation, less likely to claim

CAVEAT = ("These relationships are computed on synthetically generated claims "
          "data. They validate that the analysis pipeline correctly identifies "
          "clinically sensible risk drivers — they are NOT findings about "
          "real member behaviour. The same analysis will produce true "
          "correlations once real claims data is available.")

# --------------------------------------------------------------------------
# Clinical-system mapping. Every continuous feature + has_X flag is assigned
# to one of the 8 systems so the manager sees which body systems drive claims.
# --------------------------------------------------------------------------
SYSTEM_MAP = {}
def _assign(system, cols):
    for c in cols:
        SYSTEM_MAP[c] = system

_assign("Demographic", ["age", "height_cm", "weight_kg"])
_assign("Metabolic", [
    "bmi", "fbs_mg_dl", "ppbs_mg_dl", "hba1c_percent", "estimated_avg_glucose_mg_dl",
    "total_cholesterol_mg_dl", "hdl_mg_dl", "ldl_mg_dl", "triglycerides_mg_dl",
    "vldl_mg_dl", "non_hdl_cholesterol_mg_dl", "cho_hdl_ratio", "ldl_hdl_ratio",
    "tgl_hdl_ratio", "uric_acid_mg_dl", "vitamin_d_ng_ml",
    "has_diabetes", "has_dyslipidaemia", "has_obesity", "has_hyperuricaemia"])
_assign("Renal", [
    "bun_mg_dl", "creatinine_mg_dl", "egfr_ml_min_173m2", "calcium_mg_dl",
    "sodium_meq_l", "potassium_meq_l", "chloride_meq_l", "urine_ph",
    "urine_specific_gravity", "urine_pus_cells_hpf", "urine_rbc_hpf",
    "urine_volume_ml", "urine_epithelial_cells_hpf", "psa_ng_ml", "has_ckd"])
_assign("Hepatic", [
    "bilirubin_total_mg_dl", "bilirubin_direct_mg_dl", "bilirubin_indirect_mg_dl",
    "ast_sgot_u_l", "alt_sgpt_u_l", "alp_u_l", "ggt_u_l", "total_protein_g_dl",
    "albumin_g_dl", "globulin_g_dl", "ag_ratio", "ast_alt_ratio", "has_nafld"])
_assign("Haematologic", [
    "rbc_million_cmm", "haemoglobin_g_dl", "pcv_percent", "mcv_fl", "mch_pg",
    "mchc_percent", "rdw_cv_percent", "rdw_sd_fl", "total_wbc_cells_cumm",
    "neutrophils_percent", "lymphocytes_percent", "monocytes_percent",
    "eosinophils_percent", "basophils_percent", "platelet_count_lakhs_cumm",
    "mpv_fl", "pdw_fl", "pct_percent", "plcr_percent", "nrbc_per_100_wbc",
    "nrbc_percent", "ig_percent", "ig_abs_cells_cumm", "neutrophils_abs_cells_cumm",
    "lymphocytes_abs_cells_cumm", "monocytes_abs_cells_cumm",
    "eosinophils_abs_cells_cumm", "basophils_abs_cells_cumm",
    "ferritin_ng_ml", "iron_ug_dl", "tibc_ug_dl", "uibc_ug_dl",
    "vitamin_b12_pg_ml", "has_anaemia"])
_assign("Thyroid", ["total_t3_ng_dl", "total_t4_ug_dl", "tsh_uiu_ml",
                    "has_hypothyroidism", "has_hyperthyroidism"])
_assign("Inflammatory", ["esr_mm_hr", "crp_mg_l", "ra_factor_iu_ml"])
_assign("Cardiac-fitness", [
    "systolic_bp_mmhg", "diastolic_bp_mmhg", "resting_hr_bpm", "qtcb_ms",
    "vo2_max_ml_kg_min", "max_mets", "duke_treadmill_score", "has_hypertension"])

SYSTEM_ORDER = ["Metabolic", "Cardiac-fitness", "Renal", "Hepatic",
                "Haematologic", "Thyroid", "Inflammatory", "Demographic"]
SYSTEM_COLOR = {
    "Metabolic": "#e6550d", "Cardiac-fitness": "#d62728", "Renal": "#3182bd",
    "Hepatic": "#756bb1", "Haematologic": "#e377c2", "Thyroid": "#31a354",
    "Inflammatory": "#bcbd22", "Demographic": "#636363",
}

def pretty(col):
    return col.replace("_", " ")

# --------------------------------------------------------------------------
# 1. load TRAIN ONLY and engineer the has_X flags
# --------------------------------------------------------------------------
print("Loading splits/train.csv (train only — test never touched)...")
df = pd.read_csv(os.path.join(ROOT, "splits", "train.csv"))
print(f"  {len(df):,} rows")

cd = df["chronic_disease"].fillna("")
for token, outcol in CHRONIC_MAP.items():
    df[outcol] = cd.str.contains(token, regex=False).astype(int)

has_cols = list(CHRONIC_MAP.values())
numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
# continuous = numeric, minus targets AND minus the has_X flags we just added
continuous = [c for c in numeric if c not in TARGETS_ALL and c not in has_cols]
feature_cols = continuous + has_cols

# safety: flag anything we forgot to map
unmapped = [c for c in feature_cols if c not in SYSTEM_MAP]
if unmapped:
    print("  WARNING unmapped columns ->", unmapped)

# --------------------------------------------------------------------------
# 2. UNIVARIATE correlations with the claim target
# --------------------------------------------------------------------------
rows = []
y = df[TARGET]
for col in feature_cols:
    sub = df[[col, TARGET]].dropna()        # pairwise-complete (handles psa NaN)
    if sub[col].nunique() < 2:
        continue
    r, p = pearsonr(sub[col], sub[TARGET])
    rows.append(dict(feature=col, corr=r, abscorr=abs(r), pval=p,
                     system=SYSTEM_MAP.get(col, "Other"),
                     direction="raises" if r >= 0 else "lowers"))
uni = pd.DataFrame(rows).sort_values("abscorr", ascending=False).reset_index(drop=True)

# --------------------------------------------------------------------------
# 3. MODEL coefficients (standardized -> comparable magnitudes)
# --------------------------------------------------------------------------
model = joblib.load(os.path.join(ROOT, "frequency_model.joblib"))
feat_names = json.load(open(os.path.join(ROOT, "feature_names.json")))
coefs = np.ravel(model.coef_)
mdf = pd.DataFrame(dict(feature=feat_names, coef=coefs, abscoef=np.abs(coefs)))
mdf["system"] = mdf["feature"].map(lambda f: SYSTEM_MAP.get(f, "Other"))
mdf["direction"] = np.where(mdf["coef"] >= 0, "raises", "lowers")
mdf = mdf.sort_values("abscoef", ascending=False).reset_index(drop=True)

# --------------------------------------------------------------------------
# 4. console sanity check: top-10 ranked parameters
# --------------------------------------------------------------------------
print("\nTOP 10 PARAMETERS by |univariate correlation| with claim_next_12m")
print(f"{'rank':>4}  {'parameter':<28}{'system':<16}{'corr':>8}  direction")
for i, r in uni.head(10).iterrows():
    print(f"{i+1:>4}  {r['feature']:<28}{r['system']:<16}{r['corr']:>+8.3f}  "
          f"{'^ raises risk' if r['corr']>=0 else 'v lowers risk'}")

# --------------------------------------------------------------------------
# 5. FIGURES
# --------------------------------------------------------------------------
PLOT_CFG = {"displayModeBar": False, "responsive": True}

def fig_div(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       div_id=div_id, config=PLOT_CFG)

# ---- Fig 1: top 20 univariate drivers, horizontal bars by direction --------
top = uni.head(20).iloc[::-1]   # reverse so the strongest sits at the top
fig1 = go.Figure(go.Bar(
    x=top["corr"], y=[pretty(c) for c in top["feature"]], orientation="h",
    marker_color=[C_RAISE if d == "raises" else C_LOWER for d in top["direction"]],
    text=[f"{v:+.3f}" for v in top["corr"]], textposition="outside",
    hovertemplate="%{y}<br>corr = %{x:+.3f}<extra></extra>",
))
fig1.update_layout(
    template="plotly_white", height=640,
    title="Top 20 claim drivers — univariate correlation with claim_next_12m",
    xaxis_title="point-biserial / Pearson correlation  (← lowers risk | raises risk →)",
    margin=dict(l=200, r=80, t=60, b=40),
)
fig1.add_vline(x=0, line_width=1, line_color="#888")

# ---- Fig 2a: system summary (mean |corr| per system) -----------------------
sysg = (uni.groupby("system")
            .agg(mean_abs=("abscorr", "mean"), max_abs=("abscorr", "max"),
                 n=("feature", "size"))
            .reindex([s for s in SYSTEM_ORDER if s in uni.system.unique()])
            .reset_index())
sysg = sysg.sort_values("mean_abs", ascending=True)
fig2a = go.Figure(go.Bar(
    x=sysg["mean_abs"], y=sysg["system"], orientation="h",
    marker_color=[SYSTEM_COLOR.get(s, "#888") for s in sysg["system"]],
    text=[f"avg {m:.3f} | peak {x:.3f} | {int(n)} params"
          for m, x, n in zip(sysg["mean_abs"], sysg["max_abs"], sysg["n"])],
    textposition="outside",
    hovertemplate="%{y}<br>mean |corr| = %{x:.3f}<extra></extra>",
))
fig2a.update_layout(
    template="plotly_white", height=380,
    title="Which clinical systems drive claims (mean |correlation| per system)",
    xaxis_title="mean absolute correlation across the system's parameters",
    margin=dict(l=140, r=220, t=60, b=40),
)

# ---- Fig 2b: top 5 params per system, grouped & coloured by system ---------
parts = []
for s in SYSTEM_ORDER:
    sub = uni[uni.system == s].head(5)
    parts.append(sub)
detail = pd.concat(parts)
# order bars: system block, strongest at top within block
detail["sys_rank"] = detail["system"].map({s: i for i, s in enumerate(SYSTEM_ORDER)})
detail = detail.sort_values(["sys_rank", "abscorr"], ascending=[False, True])
fig2b = go.Figure()
for s in SYSTEM_ORDER:
    sub = detail[detail.system == s]
    if sub.empty:
        continue
    fig2b.add_trace(go.Bar(
        x=sub["corr"], y=[pretty(c) for c in sub["feature"]], orientation="h",
        name=s, marker_color=SYSTEM_COLOR.get(s, "#888"),
        text=[f"{v:+.3f}" for v in sub["corr"]], textposition="outside",
        hovertemplate="%{y}<br>"+s+"<br>corr = %{x:+.3f}<extra></extra>",
    ))
fig2b.update_layout(
    template="plotly_white", height=760, barmode="overlay",
    title="Top parameters within each clinical system (up to 5 per system)",
    xaxis_title="correlation with claim_next_12m",
    legend_title="clinical system",
    margin=dict(l=200, r=80, t=60, b=40),
)
fig2b.add_vline(x=0, line_width=1, line_color="#888")

# ---- Fig 3: model agreement -------------------------------------------------
# shared feature set = features present in BOTH the univariate set and the model
shared = uni.merge(mdf[["feature", "coef", "abscoef"]], on="feature", how="inner")
sign_agree = int((np.sign(shared["corr"]) == np.sign(shared["coef"])).sum())
rho, _ = spearmanr(shared["abscorr"], shared["abscoef"])

# 3a: scatter univariate corr vs model coef on shared features
fig3a = go.Figure(go.Scatter(
    x=shared["corr"], y=shared["coef"], mode="markers",
    marker=dict(size=9, color=np.where(
        np.sign(shared["corr"]) == np.sign(shared["coef"]), "#2ca02c", "#d62728"),
        line=dict(width=0.5, color="#444")),
    text=[pretty(c) for c in shared["feature"]],
    hovertemplate="%{text}<br>univariate r = %{x:+.3f}<br>model coef = %{y:+.3f}<extra></extra>",
))
fig3a.update_layout(
    template="plotly_white", height=460,
    title=(f"Agreement: univariate correlation vs model coefficient "
           f"(Spearman |rank| ρ = {rho:.2f}, signs agree {sign_agree}/{len(shared)})"),
    xaxis_title="univariate correlation with claim",
    yaxis_title="standardized model coefficient (LogReg)",
    margin=dict(l=70, r=40, t=60, b=50),
)
fig3a.add_hline(y=0, line_width=1, line_color="#bbb")
fig3a.add_vline(x=0, line_width=1, line_color="#bbb")

# 3b: side-by-side top-15 ranked bars (shared features), each by its own rank
uni_top = shared.sort_values("abscorr", ascending=False).head(15).iloc[::-1]
mod_top = shared.sort_values("abscoef", ascending=False).head(15).iloc[::-1]
fig3b = make_subplots(rows=1, cols=2, horizontal_spacing=0.18,
                      subplot_titles=("Ranked by UNIVARIATE correlation",
                                      "Ranked by MODEL coefficient"))
fig3b.add_trace(go.Bar(
    x=uni_top["corr"], y=[pretty(c) for c in uni_top["feature"]], orientation="h",
    marker_color=[C_RAISE if v >= 0 else C_LOWER for v in uni_top["corr"]],
    text=[f"{v:+.3f}" for v in uni_top["corr"]], textposition="outside",
    showlegend=False, hovertemplate="%{y}<br>r = %{x:+.3f}<extra></extra>"), row=1, col=1)
fig3b.add_trace(go.Bar(
    x=mod_top["coef"], y=[pretty(c) for c in mod_top["feature"]], orientation="h",
    marker_color=[C_RAISE if v >= 0 else C_LOWER for v in mod_top["coef"]],
    text=[f"{v:+.3f}" for v in mod_top["coef"]], textposition="outside",
    showlegend=False, hovertemplate="%{y}<br>coef = %{x:+.3f}<extra></extra>"), row=1, col=2)
fig3b.update_layout(template="plotly_white", height=560,
                    title="Same top parameters, two independent methods — they line up",
                    margin=dict(l=20, r=20, t=80, b=40))

# model-only top features (engineered flags the univariate view can't see)
model_only = mdf[~mdf.feature.isin(shared.feature)].head(6)

# --------------------------------------------------------------------------
# 6. assemble the HTML
# --------------------------------------------------------------------------
def kpi(label, value):
    return f'<div class="kpi"><div class="kpi-v">{value}</div><div class="kpi-l">{label}</div></div>'

top_param = uni.iloc[0]
mo_list = "".join(
    f"<li><b>{pretty(r.feature)}</b> "
    f"<span class='{ 'pos' if r.coef>=0 else 'neg'}'>{r.coef:+.3f}</span></li>"
    for _, r in model_only.iterrows())

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AHC ↔ Claim Correlation Dashboard</title>
<script>{get_plotlyjs()}</script>
<style>
  :root {{ --ink:#1f2d3d; --muted:#5a6b7b; --line:#e3e8ee; --bg:#f5f7fa; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
          color:var(--ink); background:var(--bg); }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:0 24px 64px; }}
  header.top {{ background:linear-gradient(135deg,#13344a,#1e5470); color:#fff;
               padding:30px 0 26px; }}
  header.top .wrap {{ padding-bottom:0; }}
  header.top h1 {{ margin:0 0 4px; font-size:26px; letter-spacing:.2px; }}
  header.top p {{ margin:0; color:#cfe0ec; font-size:14px; }}
  .caveat {{ background:#fff7e6; border:1px solid #f0c36d; border-left:6px solid #e0a106;
             color:#7a5b00; padding:14px 18px; border-radius:8px; margin:20px 0 8px;
             font-size:14px; line-height:1.55; }}
  .caveat b {{ color:#664c00; }}
  .kpis {{ display:flex; gap:14px; margin:18px 0 8px; flex-wrap:wrap; }}
  .kpi {{ background:#fff; border:1px solid var(--line); border-radius:10px;
          padding:14px 18px; flex:1; min-width:170px; box-shadow:0 1px 2px rgba(0,0,0,.03); }}
  .kpi-v {{ font-size:22px; font-weight:700; }}
  .kpi-l {{ font-size:12px; color:var(--muted); margin-top:2px; }}
  section {{ background:#fff; border:1px solid var(--line); border-radius:12px;
             padding:18px 20px 8px; margin:22px 0; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
  section h2 {{ margin:0 0 2px; font-size:19px; }}
  section .sub {{ color:var(--muted); font-size:13.5px; margin:0 0 6px; }}
  .legend {{ font-size:12.5px; color:var(--muted); margin:4px 0 10px; }}
  .sw {{ display:inline-block; width:11px; height:11px; border-radius:2px;
         vertical-align:middle; margin:0 4px 0 12px; }}
  .pos {{ color:{C_RAISE}; font-weight:600; }}
  .neg {{ color:{C_LOWER}; font-weight:600; }}
  .note {{ background:#f0f5fa; border:1px solid #d6e3ef; border-radius:8px;
           padding:12px 16px; font-size:13.5px; color:#33485c; margin:6px 0 14px; }}
  .note ul {{ margin:6px 0 0; padding-left:18px; }} .note li {{ margin:2px 0; }}
  footer {{ color:var(--muted); font-size:12px; text-align:center; margin-top:30px; }}
</style></head>
<body>
<header class="top"><div class="wrap">
  <h1>AHC &harr; Claim Correlation Dashboard</h1>
  <p>HealthBridge Claim Propensity &middot; Phase-1 feature signals vs claim_next_12m &middot; train split (60,000 members)</p>
</div></header>

<div class="wrap">
  <div class="caveat"><b>&#9888; Read this first.</b> {CAVEAT}</div>

  <div class="kpis">
    {kpi("Parameters analysed", f"{len(uni)}")}
    {kpi("Strongest driver", f"{pretty(top_param['feature'])} ({top_param['corr']:+.2f})")}
    {kpi("Sign agreement (univariate vs model)", f"{sign_agree}/{len(shared)}")}
    {kpi("Rank agreement (Spearman &rho;)", f"{rho:.2f}")}
  </div>

  <section>
    <h2>1 &middot; Top claim drivers</h2>
    <p class="sub">The 20 AHC parameters most strongly correlated with claiming in the next 12 months.</p>
    <div class="legend">
      <span class="sw" style="background:{C_RAISE}"></span>higher value &rarr; <b>raises</b> claim risk
      <span class="sw" style="background:{C_LOWER}"></span>higher value &rarr; <b>lowers</b> claim risk
    </div>
    {fig_div(fig1, "fig1")}
  </section>

  <section>
    <h2>2 &middot; By clinical system</h2>
    <p class="sub">Grouping the same correlations into body systems shows the manager where claim risk concentrates.</p>
    {fig_div(fig2a, "fig2a")}
    {fig_div(fig2b, "fig2b")}
  </section>

  <section>
    <h2>3 &middot; Model agreement</h2>
    <p class="sub">Two independent methods &mdash; a simple correlation and the trained model's standardized
       coefficients &mdash; should point at the same drivers. They do.</p>
    {fig_div(fig3a, "fig3a")}
    {fig_div(fig3b, "fig3b")}
    <div class="note">
      The trained model also leans on <b>engineered abnormal-lab flags</b> that the raw-parameter
      correlations can't see (these catch undiagnosed members &mdash; healthy label, abnormal labs).
      Its top model-only signals:
      <ul>{mo_list}</ul>
    </div>
  </section>

  <footer>
    Generated by src/correlation_dashboard.py &middot; data: splits/train.csv (test split sealed) &middot;
    univariate = point-biserial/Pearson r &middot; model = StandardScaled LogReg coefficients
  </footer>
</div>
</body></html>"""

outpath = os.path.join(OUTDIR, "correlation_dashboard.html")
with open(outpath, "w", encoding="utf-8") as f:
    f.write(html)

size_mb = os.path.getsize(outpath) / 1e6
print(f"\nWrote {outpath}  ({size_mb:.1f} MB, fully self-contained)")
print("Open it in any browser — no server needed.")
