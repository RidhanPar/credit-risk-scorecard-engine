"""Credit Risk Decisioning Engine — 5-tab Streamlit application.

Tabs
----
1  Dataset Overview    — IV ranking, WoE bin plots, target distribution
2  Model Performance   — ROC curves, score distribution, PSI panel
3  Credit Simulator    — Live scoring, risk tier, adverse action letter
4  Explainability      — SHAP waterfall, LIME local explanation
5  Model Monitoring    — PSI gauge, per-feature drift table
"""

import sys
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import joblib
import streamlit as st
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import (
    get_woe_contributions, load_binning_process, transform_woe,
)
from src.scorecard import get_score_tier, load_scorecard_artefact, score_applicant
from src.monitoring import calculate_psi, calculate_feature_psi, interpret_psi

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Decisioning Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1rem; max-width: 1200px; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    color: #64748b !important; font-size: 0.73rem !important;
    font-weight: 700 !important; letter-spacing: 0.07em; text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.7rem !important; font-weight: 700 !important; color: #0f172a !important;
}

/* Sidebar dark */
section[data-testid="stSidebar"] { background: #0f172a !important; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

/* Decision banners */
.dec-approve { background:#f0fdf4; border-left:5px solid #22c55e; border-radius:8px; padding:14px 20px; }
.dec-review  { background:#fffbeb; border-left:5px solid #f59e0b; border-radius:8px; padding:14px 20px; }
.dec-decline { background:#fef2f2; border-left:5px solid #ef4444; border-radius:8px; padding:14px 20px; }

/* Plain-english card */
.plain-eng { background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:16px 20px; }

/* Tab font */
button[data-baseweb="tab"] { font-weight: 600; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINNING_PATH = str(PROJECT_ROOT / "models" / "binning_process.pkl")
MODEL_PATH   = str(PROJECT_ROOT / "models" / "scorecard_model.pkl")
XGB_PATH     = str(PROJECT_ROOT / "models" / "xgb_model.pkl")
DB_PATH      = str(PROJECT_ROOT / "data"   / "credit_risk.db")
SQL_PATH     = str(PROJECT_ROOT / "sql"    / "feature_extraction.sql")

TIER_COLOR = {"Low Risk": "#22c55e", "Medium Risk": "#f59e0b",
              "High Risk": "#f97316", "Declined":   "#ef4444"}
TIER_BG    = {"Low Risk": "#f0fdf4", "Medium Risk": "#fffbeb",
              "High Risk": "#fff7ed", "Declined":   "#fef2f2"}
BASE_SCORE = 600.0
_FEAT_EXCLUDE = {"id", "default_flag", "created_at"}

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_STATE_DEFAULTS = {
    "app_submitted":  False,
    "raw_features":   {},
    "score":          None,
    "tier":           None,
    "xgb_prob":       None,
    "woe_series":     None,
    "contributions":  None,
}
for k, v in _STATE_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loaders
# ─────────────────────────────────────────────────────────────────────────────
def _models_exist() -> bool:
    return all(Path(p).exists() for p in [BINNING_PATH, MODEL_PATH, XGB_PATH])


@st.cache_resource
def _load_artefacts():
    bp      = load_binning_process(BINNING_PATH)
    art     = load_scorecard_artefact(MODEL_PATH)
    xgb_art = joblib.load(XGB_PATH)
    return bp, art, xgb_art


@st.cache_data(show_spinner=False)
def _raw_dataset() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    sql = Path(SQL_PATH).read_text()
    df  = pd.read_sql(sql, con)
    con.close()
    return df.drop(columns=["id"], errors="ignore")


@st.cache_data(show_spinner=False)
def _perf_data() -> dict:
    bp, art, xgb_art = _load_artefacts()
    df = _raw_dataset()

    feat_names = art["feature_names"]
    all_feats  = [c for c in df.columns if c not in _FEAT_EXCLUDE]

    X, y = df[all_feats], df["default_flag"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr_woe = transform_woe(X_tr, bp, feat_names)
    X_te_woe = transform_woe(X_te, bp, feat_names)

    lr     = art["model"]
    factor = art["factor"]

    proba_lr  = lr.predict_proba(X_te_woe)[:, 1]
    auc_lr    = roc_auc_score(y_te, proba_lr)
    gini_lr   = 2 * auc_lr - 1
    ks_lr     = ks_2samp(proba_lr[y_te == 0], proba_lr[y_te == 1]).statistic
    log_odds_te = lr.decision_function(X_te_woe)
    scores_te   = np.clip(np.round(BASE_SCORE - factor * log_odds_te).astype(int), 300, 850)

    # Full-dataset scores (for distribution plot)
    X_all_woe = transform_woe(X, bp, feat_names)
    lo_all    = lr.decision_function(X_all_woe)
    scores_all = np.clip(np.round(BASE_SCORE - factor * lo_all).astype(int), 300, 850)

    # XGBoost
    enc       = xgb_art["encoder"]
    cat_cols  = xgb_art["categorical_cols"]
    feat_xgb  = xgb_art["feature_names"]
    X_te_enc  = X_te.copy()
    X_te_enc[cat_cols] = enc.transform(X_te_enc[cat_cols].astype(str))
    proba_xgb = xgb_art["model"].predict_proba(X_te_enc[feat_xgb])[:, 1]
    auc_xgb   = roc_auc_score(y_te, proba_xgb)
    gini_xgb  = 2 * auc_xgb - 1
    ks_xgb    = ks_2samp(proba_xgb[y_te == 0], proba_xgb[y_te == 1]).statistic

    # ROC curve data
    from sklearn.metrics import roc_curve
    fpr_lr, tpr_lr, _ = roc_curve(y_te, proba_lr)
    fpr_xg, tpr_xg, _ = roc_curve(y_te, proba_xgb)

    return dict(
        X_tr_woe=X_tr_woe, X_te_woe=X_te_woe,
        X_tr=X_tr, X_te=X_te,
        y_tr=y_tr, y_te=y_te,
        proba_lr=proba_lr, proba_xgb=proba_xgb,
        auc_lr=auc_lr, gini_lr=gini_lr, ks_lr=ks_lr,
        auc_xgb=auc_xgb, gini_xgb=gini_xgb, ks_xgb=ks_xgb,
        fpr_lr=fpr_lr, tpr_lr=tpr_lr,
        fpr_xg=fpr_xg, tpr_xg=tpr_xg,
        scores_all=scores_all, y_all=y.values,
        scores_te=scores_te,
    )


@st.cache_data(show_spinner=False)
def _psi_data() -> dict:
    bp, art, _ = _load_artefacts()
    df = _raw_dataset()

    feat_names = art["feature_names"]
    all_feats  = [c for c in df.columns if c not in _FEAT_EXCLUDE]
    X, y = df[all_feats], df["default_flag"]
    X_tr, X_te, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr_woe = transform_woe(X_tr, bp, feat_names)
    X_te_woe = transform_woe(X_te, bp, feat_names)

    lr, factor = art["model"], art["factor"]
    lo_tr  = lr.decision_function(X_tr_woe)
    lo_te  = lr.decision_function(X_te_woe)
    sc_tr  = np.clip(np.round(BASE_SCORE - factor * lo_tr).astype(int), 300, 850)
    sc_te  = np.clip(np.round(BASE_SCORE - factor * lo_te).astype(int), 300, 850)

    rng = np.random.default_rng(42)
    sc_prod = np.clip(sc_te + rng.integers(-30, 30, size=len(sc_te)), 300, 850)
    overall_psi = calculate_psi(sc_tr, sc_prod)

    feat_psi_df = calculate_feature_psi(X_tr_woe, X_te_woe, feat_names)

    return dict(
        scores_train=sc_tr, scores_prod=sc_prod,
        overall_psi=overall_psi,
        feat_psi_df=feat_psi_df,
    )


@st.cache_resource
def _shap_explainer():
    """Build SHAP LinearExplainer for the logistic scorecard."""
    bp, art, _ = _load_artefacts()
    df = _raw_dataset()
    feat_names = art["feature_names"]
    all_feats  = [c for c in df.columns if c not in _FEAT_EXCLUDE]
    X, y = df[all_feats], df["default_flag"]
    X_tr, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr_woe = transform_woe(X_tr, bp, feat_names)
    lr = art["model"]
    explainer = shap.LinearExplainer(lr, X_tr_woe)
    return explainer, X_tr_woe


@st.cache_resource
def _lime_explainer():
    """Build LIME LimeTabularExplainer using WoE training data."""
    from lime import lime_tabular
    _, X_tr_woe = _shap_explainer()
    bp, art, _ = _load_artefacts()
    feat_names = art["feature_names"]
    return lime_tabular.LimeTabularExplainer(
        X_tr_woe.values,
        feature_names=list(feat_names),
        class_names=["Good", "Bad"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
def _iv_table(bp) -> pd.DataFrame:
    summary = bp.summary()
    iv_df = pd.DataFrame({
        "Feature": summary["name"],
        "IV":      summary["iv"].round(4),
    }).sort_values("IV", ascending=False).reset_index(drop=True)
    iv_df["Predictive Power"] = pd.cut(
        iv_df["IV"],
        bins=[-np.inf, 0.02, 0.10, 0.30, 0.50, np.inf],
        labels=["Unpredictive", "Weak", "Medium", "Strong", "Very Strong"],
    )
    return iv_df


def _woe_bin_data(bp, feature: str) -> pd.DataFrame | None:
    try:
        ob = bp.get_binning(feature)
        ob.binning_table.build()
        t = ob.binning_table.table.copy()
        # Remove aggregate rows
        exclude = {"Special", "Missing", "Totals", "nan"}
        if "Bin" in t.columns:
            t = t[~t["Bin"].astype(str).isin(exclude)]
        t = t[t["WoE"].notna()]
        return t[["Bin", "WoE"]].head(15) if "Bin" in t.columns else None
    except Exception:
        return None


def _decision_info(score: int) -> tuple[str, str, str, str]:
    """Returns (label, banner_css, rate_text, detail_text)."""
    if score >= 700:
        return ("APPROVED — Low Risk",      "dec-approve",
                "Eligible for base rate",
                "Standard terms available. No additional conditions required.")
    if score >= 600:
        return ("APPROVED — Medium Risk",   "dec-approve",
                "Base rate + 2%",
                "Approve subject to enhanced monitoring or reduced initial limit.")
    if score < 500:
        return ("DECLINED",                 "dec-decline",
                "—",
                "Score below minimum threshold. Application declined.")
    return     ("REFERRED — Manual Review", "dec-review",
                "Pending analyst decision",
                "Borderline score. Refer to senior credit analyst within 2 business days.")


def _build_raw_features(form_data: dict) -> dict:
    """Merge form inputs with defaults to create a complete feature dict."""
    d = form_data.copy()
    defaults = dict(
        personal_status="male single", other_parties="none",
        residence_since=2.0, property_magnitude="real estate",
        other_payment_plans="none", housing="own",
        existing_credits=1.0, job="skilled",
        num_dependents=1.0, own_telephone="none",
        foreign_worker="no", installment_commitment=3.0,
    )
    for k, v in defaults.items():
        if k not in d:
            d[k] = v
    # Derived features
    d["debt_to_income_proxy"]  = round(d["credit_amount"] / (max(d["duration"], 1) * max(d.get("installment_commitment", 3), 1)), 2)
    d["ever_late_flag"]        = 1 if d["credit_history"] in ("delayed previously", "critical/other existing credit") else 0
    d["poor_checking_flag"]    = 1 if d["checking_status"] in ("<0", "no checking") else 0
    d["age_per_month_credit"]  = round(d["age"] / max(d["duration"], 1), 2)
    return d


def _score_and_explain(raw: dict, bp, art: dict, xgb_art: dict):
    """Score an applicant and compute WoE contributions. Returns a result dict."""
    lr           = art["model"]
    feat_names   = art["feature_names"]
    factor       = art["factor"]
    coefficients = dict(zip(feat_names, lr.coef_[0]))

    score  = score_applicant(raw, bp, MODEL_PATH)
    tier   = get_score_tier(score)

    # WoE series
    row_df = pd.DataFrame([raw])
    all_bp = bp.variable_names
    for col in all_bp:
        if col not in row_df.columns:
            row_df[col] = np.nan
    woe_arr    = np.asarray(bp.transform(row_df[all_bp].to_numpy(), metric="woe"))[0]
    woe_series = pd.Series(woe_arr, index=all_bp)
    woe_sel    = woe_series[feat_names]
    contribs   = get_woe_contributions(woe_sel, coefficients)

    # XGBoost probability
    try:
        enc      = xgb_art["encoder"]
        cat_cols = xgb_art["categorical_cols"]
        feat_xgb = xgb_art["feature_names"]
        xgb_row  = pd.DataFrame([raw])
        xgb_row[cat_cols] = enc.transform(xgb_row[cat_cols].astype(str))
        xgb_prob = float(xgb_art["model"].predict_proba(xgb_row[feat_xgb])[0, 1])
    except Exception:
        xgb_prob = None

    return dict(score=score, tier=tier, woe_series=woe_series,
                woe_sel=woe_sel, contributions=contribs, xgb_prob=xgb_prob)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────
def _tab_dataset(bp, art):
    df = _raw_dataset()
    iv_df = _iv_table(bp)

    st.markdown("#### Dataset at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applicants",  f"{len(df):,}")
    c2.metric("Good Credit",       f"{(1 - df['default_flag'].mean()) * 100:.1f}%")
    c3.metric("Bad / Default",     f"{df['default_flag'].mean() * 100:.1f}%")
    c4.metric("Features Used",     str(len(art["feature_names"])))

    st.divider()
    left, right = st.columns([1, 1.4], gap="large")

    # Pie chart
    with left:
        st.markdown("##### Target Distribution")
        good = int((df["default_flag"] == 0).sum())
        bad  = int((df["default_flag"] == 1).sum())
        fig_pie = go.Figure(go.Pie(
            labels=["Good (Non-default)", "Bad (Default)"],
            values=[good, bad],
            marker_colors=["#22c55e", "#ef4444"],
            hole=0.42,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value}<extra></extra>",
        ))
        fig_pie.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), height=300,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # IV table
    with right:
        st.markdown("##### Feature IV Rankings")
        st.caption("IV ≥ 0.30 = Strong  ·  0.10–0.30 = Medium  ·  0.02–0.10 = Weak")

        def _iv_style(row):
            iv = row["IV"]
            if iv >= 0.30:
                bg = "#dcfce7"
            elif iv >= 0.10:
                bg = "#fef9c3"
            else:
                bg = "#ffedd5"
            return [f"background-color: {bg}"] * len(row)

        styled = iv_df.style.apply(_iv_style, axis=1).format({"IV": "{:.4f}"})
        st.dataframe(styled, use_container_width=True, hide_index=True, height=295)

    st.divider()
    st.markdown("##### WoE Bin Plots — Top 5 Features by IV")
    st.caption("Weight of Evidence per bin. Positive WoE = more goods than bads (lower risk). Negative WoE = more bads (higher risk).")

    top5 = iv_df.head(5)["Feature"].tolist()
    cols = st.columns(5)
    for i, feat in enumerate(top5):
        t = _woe_bin_data(bp, feat)
        with cols[i]:
            if t is not None and len(t) > 0:
                bar_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in t["WoE"]]
                fig = go.Figure(go.Bar(
                    x=t["Bin"].astype(str),
                    y=t["WoE"].round(3),
                    marker_color=bar_colors,
                    hovertemplate="%{x}<br>WoE: %{y:.3f}<extra></extra>",
                ))
                fig.update_layout(
                    title=dict(text=feat, font_size=11, x=0.5),
                    xaxis=dict(tickangle=-40, tickfont_size=8),
                    yaxis_title="WoE",
                    margin=dict(t=35, b=50, l=30, r=10),
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#f8fafc",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Bin data unavailable for {feat}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
def _tab_performance():
    st.markdown("#### Python vs R — Parallel Implementation")
    st.caption("Same dataset, same train/test split (80/20 stratified, random_state=42). "
               "R GLM built with scorecard pkg; run `r_analysis/03_scorecard_model.R` to populate.")

    # Try to load R metrics
    r_metrics_path = PROJECT_ROOT / "r_analysis" / "output" / "r_metrics.json"
    r_auc = r_gini = r_ks = "TBD"
    if r_metrics_path.exists():
        import json
        with open(r_metrics_path) as f:
            rm = json.load(f)
        r_auc, r_gini, r_ks = f"{rm.get('auc', 'TBD'):.3f}", f"{rm.get('gini', 'TBD'):.3f}", f"{rm.get('ks', 'TBD'):.3f}"

    perf = _perf_data()

    # Metrics table
    cmp = pd.DataFrame({
        "Model":     ["Python Logistic Scorecard", "Python XGBoost", "R GLM Scorecard"],
        "Algorithm": ["sklearn LR + optbinning WoE", "XGBoost (ordinal encoded)",
                      "glm(binomial) + scorecard WoE"],
        "AUC":       [f"{perf['auc_lr']:.3f}", f"{perf['auc_xgb']:.3f}", r_auc],
        "Gini":      [f"{perf['gini_lr']:.3f}", f"{perf['gini_xgb']:.3f}", r_gini],
        "KS":        [f"{perf['ks_lr']:.3f}",  f"{perf['ks_xgb']:.3f}",  r_ks],
    })
    st.dataframe(cmp, use_container_width=True, hide_index=True)

    st.divider()
    # ROC curves
    col_roc1, col_roc2 = st.columns(2)

    def _roc_fig(fpr, tpr, auc_val, title, color):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#94a3b8", width=1),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"AUC = {auc_val:.3f}  ·  Gini = {2*auc_val-1:.3f}",
            line=dict(color=color, width=2.2),
            fill="tozeroy", fillcolor=f"rgba({color[1:3]},{color[3:5]},{color[5:7]},0.08)"
            if len(color) == 7 else "rgba(37,99,235,0.08)",
        ))
        fig.update_layout(
            title=dict(text=title, font_size=13, x=0.5),
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            legend=dict(x=0.3, y=0.05, bgcolor="rgba(255,255,255,0.8)"),
            margin=dict(t=40, b=40, l=40, r=20), height=360,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
        )
        return fig

    with col_roc1:
        st.plotly_chart(
            _roc_fig(perf["fpr_lr"], perf["tpr_lr"], perf["auc_lr"],
                     "ROC — Python Logistic Scorecard", "#2563eb"),
            use_container_width=True,
        )
    with col_roc2:
        st.plotly_chart(
            _roc_fig(perf["fpr_xg"], perf["tpr_xg"], perf["auc_xgb"],
                     "ROC — Python XGBoost", "#16a34a"),
            use_container_width=True,
        )

    st.divider()
    # Score distribution
    st.markdown("##### Score Distribution — Full Dataset (n=1,000)")
    st.caption("Vertical lines mark risk tier boundaries: Declined (<500) · High (500) · Medium (600) · Low (700)")

    scores = perf["scores_all"]
    y_all  = perf["y_all"]
    hist_df = pd.DataFrame({"score": scores, "Class": np.where(y_all == 0, "Good", "Bad")})

    fig_dist = go.Figure()
    for cls, color in [("Good", "#22c55e"), ("Bad", "#ef4444")]:
        mask = hist_df["Class"] == cls
        fig_dist.add_trace(go.Histogram(
            x=hist_df.loc[mask, "score"], name=cls,
            marker_color=color, opacity=0.65,
            xbins=dict(start=300, end=850, size=25),
            hovertemplate=f"{cls}: %{{y}} applicants<extra></extra>",
        ))
    for xval, label, color in [
        (500, "Declined|High", "#ef4444"),
        (600, "High|Medium",   "#f97316"),
        (700, "Medium|Low",    "#22c55e"),
    ]:
        fig_dist.add_vline(x=xval, line_dash="dash", line_color=color, line_width=1.5)
        fig_dist.add_annotation(x=xval, y=0, text=f"  {xval}", showarrow=False,
                                 textangle=-90, font=dict(size=10, color=color), yanchor="bottom")
    fig_dist.update_layout(
        barmode="overlay", xaxis_title="Credit Score", yaxis_title="Count",
        legend_title="Credit Risk", height=360,
        margin=dict(t=20, b=40, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()
    # PSI panel
    psi_d = _psi_data()
    st.markdown("##### PSI Monitoring Panel")
    psi_val = psi_d["overall_psi"]
    psi_status = interpret_psi(psi_val)
    psi_color = "#22c55e" if psi_val < 0.10 else ("#f59e0b" if psi_val < 0.25 else "#ef4444")

    pm1, pm2, pm3 = st.columns(3)
    pm1.metric("Overall Score PSI", f"{psi_val:.4f}",
               delta=None, help="PSI between training and ±30-point simulated production shift")
    pm2.metric("Status", psi_status)
    pm3.metric("Threshold (Slight Shift)", "0.10")

    feat_psi_df = psi_d["feat_psi_df"].head(10)
    feat_psi_df = feat_psi_df.rename(columns={"feature": "Feature", "psi": "PSI", "status": "Status"})
    fig_psi = go.Figure(go.Bar(
        x=feat_psi_df["PSI"], y=feat_psi_df["Feature"],
        orientation="h",
        marker_color=[
            "#ef4444" if s == "Major Shift" else
            "#f59e0b" if s.startswith("Slight") else "#22c55e"
            for s in feat_psi_df["Status"]
        ],
        hovertemplate="%{y}: PSI %{x:.4f}<extra></extra>",
    ))
    fig_psi.add_vline(x=0.10, line_dash="dash", line_color="#f59e0b", annotation_text=" 0.10")
    fig_psi.add_vline(x=0.25, line_dash="dash", line_color="#ef4444", annotation_text=" 0.25")
    fig_psi.update_layout(
        title="PSI per Feature (WoE distribution: train vs test)",
        xaxis_title="PSI", height=340,
        margin=dict(t=40, b=30, l=160, r=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
    )
    st.plotly_chart(fig_psi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Credit Decision Simulator
# ─────────────────────────────────────────────────────────────────────────────
def _tab_simulator(bp, art, xgb_art):
    st.markdown("#### Applicant Score Simulator")
    st.caption("Enter applicant details to generate a credit decision with full explainability. "
               "Top 8 features by IV shown — remaining features use modal defaults.")

    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        with st.form("application_form"):
            st.markdown("**Financial Details**")
            checking_status = st.selectbox(
                "Checking account status",
                ["no checking", "<0", "0<=X<200", ">=200"]
            )
            credit_amount = st.number_input(
                "Credit amount (DM)", min_value=250, max_value=18000, value=3000, step=100
            )
            duration = st.slider("Loan duration (months)", 4, 72, 24)

            st.markdown("**Credit History**")
            credit_history = st.selectbox("Credit history", [
                "existing paid", "all paid", "no credits/all paid",
                "delayed previously", "critical/other existing credit",
            ])

            st.markdown("**Purpose & Savings**")
            purpose = st.selectbox("Loan purpose", [
                "new car", "used car", "furniture/equipment", "radio/tv",
                "domestic appliance", "repairs", "education", "business",
                "retraining", "vacation", "other",
            ])
            savings_status = st.selectbox("Savings / bonds", [
                "no known savings", "<100", "100<=X<500", "500<=X<1000", ">=1000",
            ])

            st.markdown("**Employment & Personal**")
            employment = st.selectbox("Employment (years)", [
                "unemployed", "<1", "1<=X<4", "4<=X<7", ">=7",
            ])
            age = st.slider("Age", 19, 75, 35)

            submitted = st.form_submit_button(
                "⚡ Evaluate Application", use_container_width=True, type="primary"
            )

    if submitted:
        form_data = dict(
            checking_status=checking_status, duration=float(duration),
            credit_history=credit_history, purpose=purpose,
            credit_amount=float(credit_amount), savings_status=savings_status,
            employment=employment, age=float(age),
        )
        raw = _build_raw_features(form_data)
        result = _score_and_explain(raw, bp, art, xgb_art)

        # Persist to session state for Tab 4
        st.session_state.app_submitted  = True
        st.session_state.raw_features   = raw
        st.session_state.score          = result["score"]
        st.session_state.tier           = result["tier"]
        st.session_state.xgb_prob       = result["xgb_prob"]
        st.session_state.woe_series     = result["woe_series"]
        st.session_state.contributions  = result["contributions"]

    with right:
        if not st.session_state.app_submitted:
            st.info("Fill in the applicant details and click **Evaluate Application** to see the decision.")
            return

        score   = st.session_state.score
        tier    = st.session_state.tier
        xgb_p   = st.session_state.xgb_prob
        contribs = st.session_state.contributions

        t_color = TIER_COLOR[tier]

        # KPI row
        k1, k2, k3 = st.columns(3)
        k1.metric("Credit Score", str(score))
        k2.metric("Risk Tier", tier.replace(" Risk", ""))
        k3.metric("XGBoost P(Default)", f"{xgb_p:.1%}" if xgb_p else "—")

        st.markdown("")

        # Decision banner
        dec_label, dec_css, rate_text, detail = _decision_info(score)
        st.markdown(
            f"<div class='{dec_css}'>"
            f"<strong style='font-size:1.05rem'>{dec_label}</strong>"
            f"&nbsp;&nbsp;<span style='color:#64748b;font-size:0.85rem'>·  Interest rate tier: {rate_text}</span>"
            f"<p style='margin:6px 0 0;color:#475569;font-size:0.88rem'>{detail}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("")

        # Contribution chart
        top_n = contribs.head(6)[::-1]
        bar_colors = ["#ef4444" if v > 0 else "#22c55e" for v in top_n.values]
        fig_c = go.Figure(go.Bar(
            x=top_n.values.round(3), y=top_n.index,
            orientation="h", marker_color=bar_colors,
            text=[f"{v:+.3f}" for v in top_n.values.round(3)],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        ))
        fig_c.add_vline(x=0, line_color="#94a3b8", line_width=1, line_dash="dash")
        fig_c.update_layout(
            title="Feature Contributions (WoE × coefficient)",
            xaxis_title="Log-odds contribution",
            margin=dict(t=40, b=30, l=150, r=60), height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
        )
        st.plotly_chart(fig_c, use_container_width=True)

        # Adverse action letter (declined / referred)
        if score < 600:
            adverse_factors = contribs[contribs > 0].head(3)
            letter_lines = "\n".join(
                f"  {i+1}. {feat.replace('_', ' ').title()} (contribution: {val:+.3f})"
                for i, (feat, val) in enumerate(adverse_factors.items())
            )
            with st.expander("📄 Adverse Action Letter (GDPR Art. 22 compliant)", expanded=True):
                st.markdown(f"""
**NOTICE OF CREDIT DECISION**

Dear Applicant,

Thank you for your application. After careful review of your application using our
automated credit assessment system, we are unable to approve your request at this time.

The primary factors influencing this decision were:

{letter_lines}

You have the right to:
- Request a human review of this decision (GDPR Article 22)
- Request information on the specific data used in this assessment
- Request correction of any inaccurate personal data

To exercise these rights, please contact our credit team within 30 days of this notice.

*This assessment was generated by an automated system for demonstration purposes only.*
""")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Explainability
# ─────────────────────────────────────────────────────────────────────────────
def _tab_explainability(art):
    st.markdown("#### Prediction Explainability")

    if not st.session_state.app_submitted:
        st.info("💡 Submit an application in the **Credit Simulator** tab to see explainability outputs.")
        return

    bp, _art, _ = _load_artefacts()
    lr           = art["model"]
    feat_names   = art["feature_names"]
    woe_series   = st.session_state.woe_series
    woe_sel      = woe_series[feat_names]
    score        = st.session_state.score
    contributions = st.session_state.contributions

    woe_row = pd.DataFrame([woe_sel.values], columns=feat_names)

    left, right = st.columns(2, gap="large")

    # ── SHAP waterfall ────────────────────────────────────────────────────────
    with left:
        st.markdown("##### SHAP Waterfall")
        st.caption("Each bar shows how a feature pushed the prediction away from the population base rate. "
                   "Red = increases default probability. Blue = reduces it.")
        try:
            shap_exp, _ = _shap_explainer()
            sv = shap_exp(woe_row)
            plt.close("all")
            fig_shap, ax_shap = plt.subplots(figsize=(6, 4.5))
            shap.plots.waterfall(sv[0], show=False, max_display=8)
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close("all")
        except Exception as e:
            st.warning(f"SHAP plot unavailable: {e}")
            # Fallback: show contribution bar chart
            st.bar_chart(contributions.head(8))

    # ── LIME explanation ──────────────────────────────────────────────────────
    with right:
        st.markdown("##### LIME Local Explanation")
        st.caption("Top 3 features with strongest local influence on this specific prediction. "
                   "Values are WoE-encoded. Red = increases default risk. Green = reduces risk.")
        try:
            lime_exp = _lime_explainer()
            explanation = lime_exp.explain_instance(
                woe_row.values[0],
                lr.predict_proba,
                num_features=5,
                labels=[1],
            )
            lime_items = explanation.as_list(label=1)[:3]

            # Strip LIME bin conditions to get cleaner feature names
            lime_feats = [item[0].split(" ")[0].split("<=")[0].split(">")[0].strip()
                          for item in lime_items]
            lime_vals  = [item[1] for item in lime_items]
            lime_colors = ["#ef4444" if v > 0 else "#22c55e" for v in lime_vals]

            fig_lime = go.Figure(go.Bar(
                x=lime_vals, y=lime_feats, orientation="h",
                marker_color=lime_colors,
                text=[f"{v:+.4f}" for v in lime_vals],
                textposition="outside",
                hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
            ))
            fig_lime.add_vline(x=0, line_color="#94a3b8", line_width=1, line_dash="dash")
            fig_lime.update_layout(
                xaxis_title="Contribution to P(default)",
                margin=dict(t=20, b=40, l=180, r=80), height=280,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
            )
            st.plotly_chart(fig_lime, use_container_width=True)
        except Exception as e:
            st.warning(f"LIME explanation unavailable: {e}")

    # ── Plain-English card ────────────────────────────────────────────────────
    st.divider()
    top_pos = contributions[contributions < 0].index[0] if any(contributions < 0) else "N/A"
    top_neg = contributions[contributions > 0].index[0] if any(contributions > 0) else "N/A"
    pos_woe = f"{woe_sel.get(top_pos, 0.0):+.3f} WoE" if top_pos != "N/A" else ""
    neg_woe = f"{woe_sel.get(top_neg, 0.0):+.3f} WoE" if top_neg != "N/A" else ""

    st.markdown(
        f"<div class='plain-eng'>"
        f"<strong style='font-size:1rem;color:#1e40af'>Plain-English Explanation</strong><br>"
        f"<p style='margin:8px 0 0;color:#1e293b'>"
        f"This applicant received a credit score of <strong>{score}</strong>. "
        f"The strongest positive factor was <strong>{top_pos.replace('_',' ')}</strong> "
        f"({pos_woe}), indicating a favourable credit signal. "
        f"The main risk factor was <strong>{top_neg.replace('_',' ')}</strong> "
        f"({neg_woe}), which increased the estimated probability of default."
        f"</p></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Model Monitoring
# ─────────────────────────────────────────────────────────────────────────────
def _tab_monitoring():
    st.markdown("#### Population Stability Index Dashboard")
    st.caption("Detecting score and feature distribution drift between training and deployment populations.")

    psi_d = _psi_data()
    overall_psi = psi_d["overall_psi"]
    psi_status  = interpret_psi(overall_psi)

    st.info(
        "**What is PSI?**  The Population Stability Index measures how much a distribution has "
        "shifted between two time periods. A PSI < 0.10 means the model's input distribution is "
        "stable — predictions remain trustworthy. PSI 0.10–0.25 indicates drift worth investigating; "
        "PSI > 0.25 signals the model may need retraining on a more recent population.",
        icon="ℹ️",
    )

    st.divider()
    psi_col1, psi_col2 = st.columns([1, 2], gap="large")

    # Gauge
    with psi_col1:
        psi_color = "#22c55e" if overall_psi < 0.10 else ("#f59e0b" if overall_psi < 0.25 else "#ef4444")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall_psi,
            number=dict(suffix="", valueformat=".4f", font_size=28),
            gauge=dict(
                axis=dict(range=[0, 0.35], tickwidth=1),
                bar=dict(color=psi_color, thickness=0.3),
                steps=[
                    dict(range=[0, 0.10], color="#dcfce7"),
                    dict(range=[0.10, 0.25], color="#fef9c3"),
                    dict(range=[0.25, 0.35], color="#fee2e2"),
                ],
                threshold=dict(line=dict(color="red", width=3), value=0.25),
            ),
            delta=dict(reference=0.10, valueformat=".4f"),
            title=dict(text=f"Overall Score PSI<br><span style='font-size:14px;color:{psi_color}'>{psi_status}</span>"),
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=20, l=30, r=30),
                                 paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("""
| PSI | Status |
|-----|--------|
| < 0.10 | 🟢 Stable |
| 0.10 – 0.25 | 🟡 Monitor |
| > 0.25 | 🔴 Retrain |
""")

    # Feature PSI table
    with psi_col2:
        st.markdown("##### Per-Feature PSI")
        feat_psi_df = psi_d["feat_psi_df"].copy()
        feat_psi_df.columns = [c.capitalize() for c in feat_psi_df.columns]

        def _psi_row_color(row):
            p = row["Psi"]
            if p >= 0.25: return ["background-color: #fee2e2"] * len(row)
            if p >= 0.10: return ["background-color: #fef9c3"] * len(row)
            return ["background-color: #dcfce7"] * len(row)

        styled_psi = (feat_psi_df
            .style
            .apply(_psi_row_color, axis=1)
            .format({"Psi": "{:.4f}"}))
        st.dataframe(styled_psi, use_container_width=True, hide_index=True, height=280)

    st.divider()
    # Score drift histogram
    st.markdown("##### Score Distribution Drift — Training vs Simulated Production")
    scores_tr   = psi_d["scores_train"]
    scores_prod = psi_d["scores_prod"]

    fig_drift = make_subplots(rows=1, cols=2,
                               subplot_titles=["Training Distribution", "Simulated Production (±30 pts)"])
    for col, data, color, name in [
        (1, scores_tr,   "#2563eb", "Training"),
        (2, scores_prod, "#f97316", "Production"),
    ]:
        fig_drift.add_trace(go.Histogram(
            x=data, name=name, marker_color=color, opacity=0.75,
            xbins=dict(start=300, end=850, size=30),
            hovertemplate=f"{name}: %{{y}} applicants<extra></extra>",
        ), row=1, col=col)

    for xval in [500, 600, 700]:
        for col in [1, 2]:
            fig_drift.add_vline(x=xval, line_dash="dash", line_color="#94a3b8",
                                line_width=1, row=1, col=col)

    ks_stat = ks_2samp(scores_tr, scores_prod).statistic
    fig_drift.update_layout(
        height=320, showlegend=False,
        annotations=[dict(
            x=0.5, y=1.12, xref="paper", yref="paper",
            text=f"KS statistic between distributions: {ks_stat:.4f}  ·  Overall PSI: {overall_psi:.4f}",
            showarrow=False, font_size=11, font_color="#475569",
        )],
        margin=dict(t=60, b=30, l=40, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
    )
    fig_drift.update_xaxes(title_text="Credit Score")
    fig_drift.update_yaxes(title_text="Count", col=1)
    st.plotly_chart(fig_drift, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown("## 💳 CreditIQ")
        st.markdown("### Risk Decisioning Engine")
        st.divider()

        st.info(
            "Model trained on the **German Credit Dataset** (UCI, n=1,000). "
            "All predictions are for **demonstration purposes only** and do not "
            "constitute financial advice.",
            icon="⚠️",
        )

        st.markdown("**Links**")
        st.markdown("🔗 [GitHub Repository](https://github.com/RidhanPar/credit-risk-scorecard-engine)")
        st.markdown("🚀 [Live App](https://credit-risk-scorecard-engine-ridhanpar.streamlit.app)")

        st.divider()
        st.markdown("**Model Versions**")
        st.markdown("""
| Model | Version |
|-------|---------|
| Python Scorecard | v1.0 |
| XGBoost | v1.0 |
| R GLM | v1.0 |
""")

        st.divider()
        st.markdown("**Tabs**")
        st.markdown("""
1. 📊 Dataset Overview
2. 📈 Model Performance
3. ⚡ Credit Simulator
4. 🔍 Explainability
5. 🛡️ Monitoring
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _sidebar()

    if not _models_exist():
        st.error(
            "**Model artefacts not found.**  "
            "Run `python train_pipeline.py` to train and persist all models first."
        )
        st.code("python train_pipeline.py", language="bash")
        return

    bp, art, xgb_art = _load_artefacts()

    t1, t2, t3, t4, t5 = st.tabs([
        "📊  Dataset Overview",
        "📈  Model Performance",
        "⚡  Credit Simulator",
        "🔍  Explainability",
        "🛡️  Model Monitoring",
    ])

    with t1:
        _tab_dataset(bp, art)
    with t2:
        with st.spinner("Computing model metrics…"):
            _tab_performance()
    with t3:
        _tab_simulator(bp, art, xgb_art)
    with t4:
        _tab_explainability(art)
    with t5:
        with st.spinner("Computing PSI metrics…"):
            _tab_monitoring()


if __name__ == "__main__":
    main()
