"""Streamlit credit scoring UI.

Provides an interactive form for scoring a single applicant using the trained
logistic regression scorecard.  Displays:
  - Credit score (300-850 gauge)
  - Risk tier with colour coding
  - WoE contribution breakdown for the top 5 features
  - XGBoost default probability for comparison
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
from src.monitoring import calculate_psi, interpret_psi
from src.scorecard import (
    get_score_tier,
    load_scorecard_artefact,
    score_applicant,
)
from src.xgb_model import predict_proba_xgb

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINNING_PATH = str(PROJECT_ROOT / "models" / "binning_process.pkl")
MODEL_PATH = str(PROJECT_ROOT / "models" / "scorecard_model.pkl")
XGB_PATH = str(PROJECT_ROOT / "models" / "xgb_model.pkl")


# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all model artefacts once and cache for the session."""
    bp = load_binning_process(BINNING_PATH)
    artefact = load_scorecard_artefact(MODEL_PATH)
    return bp, artefact


def models_exist() -> bool:
    return (
        Path(BINNING_PATH).exists()
        and Path(MODEL_PATH).exists()
        and Path(XGB_PATH).exists()
    )


# ---------------------------------------------------------------------------
# Tier styling
# ---------------------------------------------------------------------------
TIER_COLORS = {
    "Low Risk": "#4CAF50",
    "Medium Risk": "#FF9800",
    "High Risk": "#FF5722",
    "Declined": "#F44336",
}

TIER_ICONS = {
    "Low Risk": "✅",
    "Medium Risk": "⚠️",
    "High Risk": "🔶",
    "Declined": "❌",
}


# ---------------------------------------------------------------------------
# Score gauge (matplotlib figure)
# ---------------------------------------------------------------------------
def _draw_score_gauge(score: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi)

    theta_range = np.linspace(0, np.pi, 300)
    score_norm = (score - 300) / (850 - 300)
    theta_score = score_norm * np.pi

    # Background arc
    ax.plot(theta_range, [1] * 300, color="#e0e0e0", linewidth=12, solid_capstyle="round")

    # Coloured arc up to current score
    colors = ["#F44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
    segment = int(score_norm * 5)
    seg_color = colors[min(segment, 4)]
    ax.plot(np.linspace(0, theta_score, 100), [1] * 100,
            color=seg_color, linewidth=12, solid_capstyle="round")

    # Needle
    ax.plot([theta_score, theta_score], [0, 0.85], color="#212121", linewidth=2)
    ax.scatter([theta_score], [0], color="#212121", s=30, zorder=5)

    ax.set_ylim(0, 1.1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    for val, label in [(300, "300"), (575, "575"), (850, "850")]:
        t = (val - 300) / (850 - 300) * np.pi
        ax.text(t, 1.25, label, ha="center", va="center", fontsize=8, color="#555")

    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    fig.tight_layout(pad=0)
    return fig


# ---------------------------------------------------------------------------
# Contribution bar chart
# ---------------------------------------------------------------------------
def _contribution_chart(contributions: pd.Series, n: int = 5) -> plt.Figure:
    top = contributions.head(n)
    colors = ["#F44336" if v > 0 else "#4CAF50" for v in top.values]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("Log-odds contribution (WoE × coefficient)", fontsize=9)
    ax.set_title("Top feature contributions", fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("📊 Credit Risk Scorecard Engine")
    st.caption(
        "German Credit Dataset · Logistic Regression + PDO Scaling · "
        "Base score 600 @ 1:1 odds · PDO = 20"
    )

    if not models_exist():
        st.error(
            "Models not found.  Please run `python train_pipeline.py` first to train "
            "and save the model artefacts, then restart this app."
        )
        st.code("python train_pipeline.py", language="bash")
        return

    bp, artefact = load_models()
    lr_model = artefact["model"]
    feature_names: list[str] = artefact["feature_names"]
    factor: float = artefact["factor"]
    coefficients: dict = dict(zip(feature_names, lr_model.coef_[0]))

    # ------------------------------------------------------------------
    # Sidebar — applicant input form
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Applicant Details")

        checking_status = st.selectbox(
            "Checking account status",
            ["no checking", "<0", "0<=X<200", ">=200"],
        )
        duration = st.slider("Loan duration (months)", 4, 72, 24)
        credit_amount = st.number_input("Credit amount (DM)", 500, 20000, 3000, step=100)
        credit_history = st.selectbox(
            "Credit history",
            [
                "existing paid",
                "all paid",
                "no credits/all paid",
                "delayed previously",
                "critical/other existing credit",
            ],
        )
        purpose = st.selectbox(
            "Purpose",
            ["new car", "used car", "furniture/equipment", "radio/tv",
             "domestic appliance", "repairs", "education", "business",
             "retraining", "vacation", "other"],
        )
        savings_status = st.selectbox(
            "Savings account / bonds",
            ["no known savings", "<100", "100<=X<500", "500<=X<1000", ">=1000"],
        )
        employment = st.selectbox(
            "Employment (years)",
            ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
        )
        installment_commitment = st.slider("Installment rate (% of disposable income)", 1, 4, 3)
        age = st.slider("Age (years)", 19, 75, 35)
        personal_status = st.selectbox(
            "Personal status",
            ["male single", "male mar/wid", "male div/sep",
             "female div/dep/mar", "female single"],
        )
        property_magnitude = st.selectbox(
            "Property / collateral",
            ["real estate", "life insurance", "car", "no known property"],
        )
        housing = st.selectbox("Housing", ["own", "rent", "for free"])
        other_payment_plans = st.selectbox(
            "Other instalment plans", ["none", "bank", "stores"]
        )
        other_parties = st.selectbox(
            "Other parties (guarantor/co-applicant)", ["none", "co applicant", "guarantor"]
        )
        existing_credits = st.slider("Number of existing credits at this bank", 1, 4, 1)
        residence_since = st.slider("Residence since (years)", 1, 4, 2)
        num_dependents = st.slider("Number of dependants", 1, 2, 1)
        job = st.selectbox(
            "Job category",
            ["skilled", "unskilled resident", "high qualif/self emp/mgmt",
             "unskilled non-resident"],
        )
        own_telephone = st.selectbox("Own telephone", ["none", "yes"])
        foreign_worker = st.selectbox("Foreign worker", ["yes", "no"])

    # Assemble raw feature dict
    raw_features = dict(
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
        # Derived features expected by the SQL / binning process
        debt_to_income_proxy=round(credit_amount / (max(duration, 1) * max(installment_commitment, 1)), 2),
        ever_late_flag=1 if credit_history in ("delayed previously", "critical/other existing credit") else 0,
        poor_checking_flag=1 if checking_status in ("<0", "no checking") else 0,
        age_per_month_credit=round(age / max(duration, 1), 2),
    )

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------
    score = score_applicant(raw_features, bp, MODEL_PATH)
    tier = get_score_tier(score)
    tier_color = TIER_COLORS[tier]
    tier_icon = TIER_ICONS[tier]

    # WoE contributions for chart
    row_df = pd.DataFrame([raw_features])
    all_bp_features = bp.variable_names
    for col in all_bp_features:
        if col not in row_df.columns:
            row_df[col] = np.nan
    woe_series = pd.Series(
        np.asarray(bp.transform(row_df[all_bp_features], metric="woe"))[0],
        index=all_bp_features,
    )
    woe_selected = woe_series[feature_names]
    contributions = get_woe_contributions(woe_selected, coefficients)

    # XGBoost probability (requires XGB artefact)
    try:
        xgb_row = pd.DataFrame([{k: v for k, v in raw_features.items()
                                  if k in ["checking_status","duration","credit_history",
                                           "purpose","credit_amount","savings_status",
                                           "employment","installment_commitment",
                                           "personal_status","other_parties",
                                           "residence_since","property_magnitude","age",
                                           "other_payment_plans","housing","existing_credits",
                                           "job","num_dependents","own_telephone","foreign_worker",
                                           "debt_to_income_proxy","ever_late_flag",
                                           "poor_checking_flag","age_per_month_credit"]}])
        xgb_prob = float(predict_proba_xgb(xgb_row, XGB_PATH)[0])
    except Exception:
        xgb_prob = None

    # ------------------------------------------------------------------
    # Score display
    # ------------------------------------------------------------------
    col1, col2, col3 = st.columns([1.2, 1.3, 1.5])

    with col1:
        st.subheader("Credit Score")
        gauge_fig = _draw_score_gauge(score)
        st.pyplot(gauge_fig, use_container_width=False)
        st.markdown(
            f"<h1 style='text-align:center; color:{tier_color}; margin-top:-8px'>"
            f"{score}</h1>",
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Risk Assessment")
        st.markdown(
            f"<div style='background:{tier_color}22; border-left:5px solid {tier_color};"
            f" padding:16px; border-radius:8px; margin-top:8px'>"
            f"<span style='font-size:2rem'>{tier_icon}</span>"
            f"<h3 style='color:{tier_color}; margin:4px 0'>{tier}</h3>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**Score ranges**")
        st.markdown("""
| Score | Tier |
|-------|------|
| 700 – 850 | ✅ Low Risk |
| 600 – 699 | ⚠️ Medium Risk |
| 500 – 599 | 🔶 High Risk |
| 300 – 499 | ❌ Declined |
""")

    with col3:
        st.subheader("Feature Contributions")
        contrib_fig = _contribution_chart(contributions, n=5)
        st.pyplot(contrib_fig, use_container_width=True)
        if xgb_prob is not None:
            st.markdown("---")
            xgb_color = "#4CAF50" if xgb_prob < 0.3 else ("#FF9800" if xgb_prob < 0.5 else "#F44336")
            st.markdown(
                f"**XGBoost P(default):** "
                f"<span style='color:{xgb_color}; font-size:1.3rem'>"
                f"**{xgb_prob:.1%}**</span>",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # Detailed WoE breakdown table
    # ------------------------------------------------------------------
    with st.expander("Full WoE contribution table"):
        contrib_df = pd.DataFrame({
            "Feature": contributions.index,
            "WoE": woe_selected.reindex(contributions.index).values.round(4),
            "Coefficient": [coefficients.get(f, 0) for f in contributions.index],
            "Contribution (log-odds)": contributions.values.round(4),
        })
        st.dataframe(contrib_df, use_container_width=True)


if __name__ == "__main__":
    main()
