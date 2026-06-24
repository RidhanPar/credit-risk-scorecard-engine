"""Credit Risk Scorecard Engine — Streamlit UI.

Real-world fintech application layout:
  Tab 1 — Applicant Scoring   : score gauge, risk tier, WoE contributions, decision
  Tab 2 — Model Performance   : IV table, ROC-AUC / Gini / KS, feature importance
  Tab 3 — Methodology         : PDO scaling, WoE/IV theory, risk policy
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import get_woe_contributions, load_binning_process
from src.scorecard import get_score_tier, load_scorecard_artefact, score_applicant
from src.xgb_model import predict_proba_xgb

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditIQ — Risk Scorecard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Layout ── */
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1140px; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

/* ── Section labels ── */
.label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
    margin-top: 1rem;
}

/* ── Decision banners ── */
.dec-approve  { background:#f0fdf4; border-left:5px solid #22c55e; border-radius:8px; padding:14px 18px; }
.dec-review   { background:#fffbeb; border-left:5px solid #f59e0b; border-radius:8px; padding:14px 18px; }
.dec-decline  { background:#fef2f2; border-left:5px solid #ef4444; border-radius:8px; padding:14px 18px; }

/* ── Sidebar dark theme ── */
section[data-testid="stSidebar"] { background: #0f172a !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-size: 1rem !important; }
section[data-testid="stSidebar"] .label { color: #475569 !important; }
section[data-testid="stSidebar"] hr { border-color: #1e293b !important; }

/* ── Tab style ── */
button[data-baseweb="tab"] { font-weight: 600; font-size: 0.88rem; padding: 8px 16px; }
button[data-baseweb="tab"][aria-selected="true"] { color: #2563eb !important; }

/* ── Info box ── */
div[data-testid="stInfo"] { background: #eff6ff; border-color: #93c5fd; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINNING_PATH = str(PROJECT_ROOT / "models" / "binning_process.pkl")
MODEL_PATH   = str(PROJECT_ROOT / "models" / "scorecard_model.pkl")
XGB_PATH     = str(PROJECT_ROOT / "models" / "xgb_model.pkl")
DB_PATH      = str(PROJECT_ROOT / "data"   / "credit_risk.db")
SQL_PATH     = str(PROJECT_ROOT / "sql"    / "feature_extraction.sql")

TIER_COLOR = {"Low Risk": "#22c55e", "Medium Risk": "#f59e0b",
              "High Risk": "#f97316", "Declined": "#ef4444"}
TIER_BG    = {"Low Risk": "#f0fdf4", "Medium Risk": "#fffbeb",
              "High Risk": "#fff7ed", "Declined": "#fef2f2"}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def models_exist() -> bool:
    return (Path(BINNING_PATH).exists()
            and Path(MODEL_PATH).exists()
            and Path(XGB_PATH).exists())


@st.cache_resource
def load_models():
    bp       = load_binning_process(BINNING_PATH)
    artefact = load_scorecard_artefact(MODEL_PATH)
    return bp, artefact


# ---------------------------------------------------------------------------
# Score gauge (fixed to not overflow)
# ---------------------------------------------------------------------------
def _gauge(score: int, color: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.6, 2.2), subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi)

    # Five coloured zone bands (faint)
    zone_clrs = ["#ef4444", "#f97316", "#f59e0b", "#84cc16", "#22c55e"]
    for i, c in enumerate(zone_clrs):
        t0 = i * np.pi / 5
        t1 = (i + 1) * np.pi / 5
        ax.plot(np.linspace(t0, t1, 60), [1] * 60,
                color=c, linewidth=13, solid_capstyle="butt", alpha=0.18)

    # Active arc
    norm    = (score - 300) / 550
    t_score = norm * np.pi
    ax.plot(np.linspace(0, t_score, 200), [1] * 200,
            color=color, linewidth=13, solid_capstyle="round")

    # Needle
    ax.annotate("", xy=(t_score, 0.80), xytext=(t_score, 0.05),
                arrowprops=dict(arrowstyle="-|>", color="#1e293b",
                                lw=1.8, mutation_scale=9))
    ax.scatter([t_score], [0.05], color="#1e293b", s=35, zorder=5)

    # Tick labels
    for val, lbl in [(300, "300"), (575, "575"), (850, "850")]:
        t = (val - 300) / 550 * np.pi
        ax.text(t, 1.32, lbl, ha="center", va="center",
                fontsize=8, color="#64748b", fontweight="600")

    ax.set_ylim(0, 1.45)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    fig.tight_layout(pad=0.2)
    return fig


# ---------------------------------------------------------------------------
# Contribution bar chart
# ---------------------------------------------------------------------------
def _contrib_chart(contributions: pd.Series, n: int = 7) -> plt.Figure:
    top    = contributions.head(n)[::-1]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in top.values]

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    fig.patch.set_color("#f8fafc")
    ax.set_facecolor("#f8fafc")

    bars = ax.barh(top.index, top.values, color=colors, height=0.55, edgecolor="none")
    for bar, val in zip(bars, top.values):
        pad = 0.012 if val >= 0 else -0.012
        ax.text(val + pad, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8, color="#374151", fontweight="600")

    ax.axvline(0, color="#94a3b8", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xlabel("Log-odds contribution  (WoE × coefficient)", fontsize=8.5, color="#64748b")
    ax.tick_params(axis="y", labelsize=9, colors="#374151")
    ax.tick_params(axis="x", labelsize=8, colors="#94a3b8")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.grid(axis="x", alpha=0.35, color="#e2e8f0", linestyle="--")
    ax.set_title("Feature contributions to credit decision",
                 fontsize=10, fontweight="700", color="#1e293b", pad=10, loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------
def _decision(score: int) -> tuple[str, str, str]:
    if score >= 700:
        return "APPROVE", "Low default probability — standard terms available.", "dec-approve"
    if score >= 620:
        return "APPROVE WITH CONDITIONS", "Approve with reduced credit limit and enhanced monitoring.", "dec-review"
    if score >= 500:
        return "MANUAL REVIEW", "Borderline case — refer to senior credit analyst.", "dec-review"
    return "DECLINE", "Score below minimum threshold. High default probability.", "dec-decline"


# ---------------------------------------------------------------------------
# Sidebar applicant form
# ---------------------------------------------------------------------------
def _sidebar() -> dict:
    with st.sidebar:
        st.markdown("### ⚙️  Applicant Profile")
        st.markdown("---")

        st.markdown('<p class="label">Financial</p>', unsafe_allow_html=True)
        checking_status  = st.selectbox("Checking account status",
            ["no checking", "<0", "0<=X<200", ">=200"])
        credit_amount    = st.number_input("Credit amount (DM)", 500, 20000, 3000, step=100)
        duration         = st.slider("Loan duration (months)", 4, 72, 24)
        savings_status   = st.selectbox("Savings / bonds",
            ["no known savings", "<100", "100<=X<500", "500<=X<1000", ">=1000"])
        installment_commitment = st.slider("Installment rate (% of income)", 1, 4, 3)

        st.markdown('<p class="label">Credit History</p>', unsafe_allow_html=True)
        credit_history   = st.selectbox("Credit history",
            ["existing paid", "all paid", "no credits/all paid",
             "delayed previously", "critical/other existing credit"])
        existing_credits = st.slider("Existing credits at this bank", 1, 4, 1)
        other_payment_plans = st.selectbox("Other instalment plans", ["none", "bank", "stores"])

        st.markdown('<p class="label">Employment</p>', unsafe_allow_html=True)
        employment       = st.selectbox("Employment (years)",
            ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"])
        job              = st.selectbox("Job category",
            ["skilled", "unskilled resident",
             "high qualif/self emp/mgmt", "unskilled non-resident"])

        st.markdown('<p class="label">Personal</p>', unsafe_allow_html=True)
        age              = st.slider("Age", 19, 75, 35)
        personal_status  = st.selectbox("Personal status",
            ["male single", "male mar/wid", "male div/sep",
             "female div/dep/mar", "female single"])
        num_dependents   = st.slider("Dependants", 1, 2, 1)
        own_telephone    = st.selectbox("Own telephone", ["none", "yes"])
        foreign_worker   = st.selectbox("Foreign worker", ["yes", "no"])

        st.markdown('<p class="label">Property & Housing</p>', unsafe_allow_html=True)
        property_magnitude = st.selectbox("Property / collateral",
            ["real estate", "life insurance", "car", "no known property"])
        housing          = st.selectbox("Housing", ["own", "rent", "for free"])
        residence_since  = st.slider("Residence since (years)", 1, 4, 2)

        st.markdown('<p class="label">Loan Purpose</p>', unsafe_allow_html=True)
        purpose          = st.selectbox("Purpose",
            ["new car", "used car", "furniture/equipment", "radio/tv",
             "domestic appliance", "repairs", "education", "business",
             "retraining", "vacation", "other"])

        st.markdown('<p class="label">Other Parties</p>', unsafe_allow_html=True)
        other_parties    = st.selectbox("Guarantor / co-applicant",
            ["none", "co applicant", "guarantor"])

    return dict(
        checking_status=checking_status,
        duration=float(duration),
        credit_history=credit_history,
        purpose=purpose,
        credit_amount=float(credit_amount),
        savings_status=savings_status,
        employment=employment,
        installment_commitment=float(installment_commitment),
        personal_status=personal_status,
        other_parties=other_parties,
        residence_since=float(residence_since),
        property_magnitude=property_magnitude,
        age=float(age),
        other_payment_plans=other_payment_plans,
        housing=housing,
        existing_credits=float(existing_credits),
        job=job,
        num_dependents=float(num_dependents),
        own_telephone=own_telephone,
        foreign_worker=foreign_worker,
        debt_to_income_proxy=round(credit_amount / (max(duration, 1) * max(installment_commitment, 1)), 2),
        ever_late_flag=1 if credit_history in ("delayed previously", "critical/other existing credit") else 0,
        poor_checking_flag=1 if checking_status in ("<0", "no checking") else 0,
        age_per_month_credit=round(age / max(duration, 1), 2),
    )


# ---------------------------------------------------------------------------
# Performance data (cached — runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def _perf_data(_bp, _lr, _feature_names: tuple):
    import sqlite3
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from scipy.stats import ks_2samp
    from src.feature_engineering import transform_woe

    con = sqlite3.connect(DB_PATH)
    sql = Path(SQL_PATH).read_text()
    df  = pd.read_sql(sql, con)
    con.close()

    feat_cols = [c for c in df.columns if c not in {"id", "default_flag", "created_at"}]
    X = df[feat_cols]
    y = df["default_flag"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    woe_test = transform_woe(X_test, _bp, list(_feature_names))
    proba    = _lr.predict_proba(woe_test)[:, 1]

    auc  = roc_auc_score(y_test, proba)
    gini = 2 * auc - 1
    ks   = ks_2samp(proba[y_test == 0], proba[y_test == 1]).statistic

    summary = _bp.summary()
    iv_df = (
        pd.DataFrame({"Feature": summary["name"], "IV": summary["iv"].round(4)})
        .sort_values("IV", ascending=False)
        .reset_index(drop=True)
    )
    iv_df.index += 1

    return dict(auc=auc, gini=gini, ks=ks, n_test=len(X_test), iv_df=iv_df)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ── App header ───────────────────────────────────────────────────────────
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown("## 📊 CreditIQ — Risk Scorecard Engine")
        st.caption(
            "German Credit Dataset · Logistic Regression + PDO Scaling · "
            "Base score 600 @ 1:1 odds · PDO = 20"
        )
    with hc2:
        st.markdown(
            "<div style='text-align:right;padding-top:14px;"
            "color:#94a3b8;font-size:0.78rem'>"
            "optbinning · WoE/IV · XGBoost</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    # ── Guard ────────────────────────────────────────────────────────────────
    if not models_exist():
        st.error(
            "Model artefacts not found in `models/`. "
            "Run `python train_pipeline.py` to generate them."
        )
        return

    bp, artefact  = load_models()
    lr_model      = artefact["model"]
    feature_names = artefact["feature_names"]
    coefficients  = dict(zip(feature_names, lr_model.coef_[0]))

    raw = _sidebar()

    # ── Compute score ────────────────────────────────────────────────────────
    score     = score_applicant(raw, bp, MODEL_PATH)
    tier      = get_score_tier(score)
    t_color   = TIER_COLOR[tier]
    dec, dec_text, dec_css = _decision(score)

    # WoE series for contributions
    row_df = pd.DataFrame([raw])
    all_bp  = bp.variable_names
    for col in all_bp:
        if col not in row_df.columns:
            row_df[col] = np.nan

    woe_arr    = np.asarray(bp.transform(row_df[all_bp].to_numpy(), metric="woe"))[0]
    woe_series = pd.Series(woe_arr, index=all_bp)
    woe_sel    = woe_series[feature_names]
    contribs   = get_woe_contributions(woe_sel, coefficients)

    # XGBoost probability
    try:
        xgb_prob = float(predict_proba_xgb(pd.DataFrame([raw]), XGB_PATH)[0])
    except Exception:
        xgb_prob = None

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📋  Applicant Scoring",
        "📈  Model Performance",
        "📚  Methodology",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Applicant Scoring
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Credit Score", str(score))
        k2.metric("Risk Tier", tier.replace(" Risk", ""))
        k3.metric("P(Default) XGBoost", f"{xgb_prob:.1%}" if xgb_prob is not None else "—")
        k4.metric("Recommendation", dec.split()[0])

        st.markdown("")

        # Decision banner
        st.markdown(
            f"<div class='{dec_css}'>"
            f"<span style='font-size:1rem;font-weight:700;color:#1e293b'>{dec}</span>"
            f"<p style='margin:4px 0 0;color:#475569;font-size:0.88rem'>{dec_text}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Score gauge (left) + Contributions (right) ────────────────────
        left, right = st.columns([1, 1.55], gap="large")

        with left:
            st.markdown('<p class="label">Credit Score Gauge</p>', unsafe_allow_html=True)
            gauge_fig = _gauge(score, t_color)
            st.pyplot(gauge_fig, use_container_width=True)

            # Score + tier badge — rendered inside this column, never overflow
            st.markdown(
                f"<div style='text-align:center;margin-top:-4px'>"
                f"<span style='font-size:2.6rem;font-weight:800;color:{t_color}'>{score}</span>"
                f"<br>"
                f"<span style='display:inline-block;background:{TIER_BG[tier]};"
                f"color:{t_color};font-weight:700;font-size:0.85rem;"
                f"padding:4px 14px;border-radius:20px;margin-top:4px'>"
                f"{tier}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("")
            st.markdown("**Score ranges**")
            st.markdown("""
| Score | Tier |
|-------|------|
| 700 – 850 | ✅ Low Risk |
| 600 – 699 | ⚠️ Medium Risk |
| 500 – 599 | 🟠 High Risk |
| 300 – 499 | ❌ Declined |
""")

        with right:
            st.markdown('<p class="label">Feature Contributions</p>', unsafe_allow_html=True)
            st.caption(
                "🔴 Red = increases default risk (lowers score)  "
                "🟢 Green = reduces risk (raises score)"
            )
            contrib_fig = _contrib_chart(contribs, n=7)
            st.pyplot(contrib_fig, use_container_width=True)

            with st.expander("Full WoE contribution table", expanded=False):
                out_df = pd.DataFrame({
                    "Feature": contribs.index,
                    "WoE": woe_sel.reindex(contribs.index).values.round(4),
                    "Coefficient": [coefficients.get(f, 0) for f in contribs.index],
                    "Log-odds contribution": contribs.values.round(4),
                })
                st.dataframe(out_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Model Performance
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### Model Validation Results")
        st.caption("Test split: 20%, stratified, random_state=42 — same split used during training")

        with st.spinner("Computing performance metrics from test set…"):
            try:
                perf = _perf_data(bp, lr_model, tuple(feature_names))

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ROC-AUC  (Logistic)", f"{perf['auc']:.3f}",
                          delta="vs 0.799 XGBoost", delta_color="normal")
                m2.metric("Gini Coefficient",     f"{perf['gini']:.3f}")
                m3.metric("KS Statistic",         f"{perf['ks']:.3f}")
                m4.metric("Test Applicants",      str(perf["n_test"]))

                st.markdown("")
                st.info(
                    "**Why logistic outperforms XGBoost here:** WoE pre-encoding "
                    "linearises the log-odds relationship for each feature, removing "
                    "non-monotonic patterns. A linear model is then optimal — "
                    "XGBoost wastes capacity re-learning what WoE already encodes. "
                    "This is standard knowledge in credit scoring (Siddiqi, 2006).",
                    icon="💡",
                )

                st.divider()
                st.markdown("#### Information Value (IV) by Feature")
                st.caption(
                    "IV < 0.02 = uninformative · 0.02–0.10 = weak · "
                    "0.10–0.30 = medium · 0.30–0.50 = strong · > 0.50 = suspicious"
                )

                def _iv_color(val: float) -> str:
                    if val >= 0.30:
                        return "color: #16a34a; font-weight: 700"
                    if val >= 0.10:
                        return "color: #d97706; font-weight: 600"
                    return "color: #94a3b8"

                styled = (
                    perf["iv_df"]
                    .style
                    .applymap(lambda v: _iv_color(v), subset=["IV"])
                    .format({"IV": "{:.4f}"})
                    .bar(subset=["IV"], color="#bfdbfe", vmin=0, vmax=0.6)
                )
                st.dataframe(styled, use_container_width=True)

                st.divider()
                st.markdown("#### Model Comparison Summary")
                comp = pd.DataFrame({
                    "Model":    ["Logistic Scorecard", "XGBoost Challenger"],
                    "ROC-AUC":  [f"{perf['auc']:.3f}", "0.799"],
                    "Gini":     [f"{perf['gini']:.3f}", "0.598"],
                    "KS":       [f"{perf['ks']:.3f}",  "0.413"],
                    "Champion": ["✅ Yes", "❌ Challenger"],
                })
                st.dataframe(comp, use_container_width=True, hide_index=True)

            except Exception as exc:
                st.error(f"Could not compute performance metrics: {exc}")
                st.info("Fallback — known results from training run: "
                        "Logistic AUC 0.818 · Gini 0.636 · KS 0.449")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Methodology
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("#### Scorecard Architecture")
            st.markdown("""
**1 · Feature Store (SQLite + CTE)**

Raw applicant records are stored in SQLite. A CTE-based extraction query
derives four engineered signals:

| Engineered Feature | Rationale |
|---|---|
| `debt_to_income_proxy` | Credit amount ÷ (duration × instalment rate) |
| `ever_late_flag` | 1 if any prior delinquency on record |
| `poor_checking_flag` | 1 if checking balance < 0 or no account |
| `age_per_month_credit` | Age ÷ duration — older short loans = lower risk |

---

**2 · WoE / IV Encoding (optbinning)**

Each feature is binned optimally (CP-SAT solver) and replaced with its
**Weight of Evidence** value:

```
WoE = ln( % non-defaults / % defaults )
```

Features with IV < 0.02 are dropped. WoE encodes the monotone
log-odds signal, making logistic regression near-optimal.

---

**3 · Logistic Regression**

Trained on WoE features. Coefficients are directly interpretable:
`coefficient × WoE = log-odds contribution` per feature.
""")

        with c2:
            st.markdown("#### PDO Score Scaling")
            st.markdown(r"""
Industry-standard **Points to Double the Odds** formula:

```
Factor = PDO / ln(2)          →  28.854  (PDO = 20)
Offset = Base − Factor×ln(θ)  →  600     (θ = 1:1 base odds)
Score  = Offset + Factor × (−log-odds)
```

Score range: **300 – 850** (mirrors US FICO convention).

---

**Risk Policy Tiers**

| Score | Decision | Action |
|-------|----------|--------|
| 700 – 850 | ✅ Approve | Auto-approve, standard rate |
| 620 – 699 | ⚠️ Conditional | Approve with limit cap |
| 500 – 619 | 🟠 Review | Refer to analyst |
| 300 – 499 | ❌ Decline | Auto-decline |

---

**Population Stability Index (PSI)**

Monitors feature / score distribution drift between
development and live populations:

| PSI | Interpretation |
|-----|----------------|
| < 0.10 | Stable — no action |
| 0.10 – 0.25 | Slight shift — monitor |
| > 0.25 | Major shift — revalidate |
""")

        st.divider()
        st.markdown("#### Tech Stack")
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.info("**Data**\nSQLite · pandas · sklearn fetch_openml")
        tc2.info("**Modelling**\noptbinning · scikit-learn · XGBoost")
        tc3.info("**Interpretability**\nSHAP TreeExplainer · WoE contributions")
        tc4.info("**Infrastructure**\nDocker · GitHub Actions · Streamlit Cloud")


if __name__ == "__main__":
    main()
