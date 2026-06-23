"""Population Stability Index (PSI) monitoring for deployed scorecards.

PSI measures the shift between the score distribution seen at training time
(expected) and the distribution seen in production (actual).  It is the
standard monitoring metric for consumer credit scorecards.

PSI interpretation (industry convention):
    PSI < 0.10   — Stable: no meaningful distribution shift, model is safe to use.
    0.10–0.25    — Slight shift: investigate feature drift; plan model review.
    PSI > 0.25   — Major shift: model likely degraded; trigger retraining.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "assets" / "screenshots"

PSI_STABLE = 0.10
PSI_WARNING = 0.25


# ---------------------------------------------------------------------------
# Core PSI calculation
# ---------------------------------------------------------------------------

def calculate_psi(
    expected_scores: np.ndarray,
    actual_scores: np.ndarray,
    n_bins: int = 10,
    score_min: int = 300,
    score_max: int = 850,
) -> float:
    """Compute the Population Stability Index between two score distributions.

    PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%)) across bins.
    Epsilon smoothing is applied to avoid division by zero in empty bins.

    Args:
        expected_scores: Scores from the training / reference population.
        actual_scores:   Scores from the current / monitoring population.
        n_bins:          Number of equal-width bins (default 10).
        score_min:       Lower bound for binning range.
        score_max:       Upper bound for binning range.

    Returns:
        float: PSI value (non-negative; higher = more drift).
    """
    bins = np.linspace(score_min, score_max, n_bins + 1)
    epsilon = 1e-6  # prevents log(0)

    exp_counts, _ = np.histogram(expected_scores, bins=bins)
    act_counts, _ = np.histogram(actual_scores, bins=bins)

    exp_pct = exp_counts / len(expected_scores) + epsilon
    act_pct = act_counts / len(actual_scores) + epsilon

    psi_per_bin = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    psi = float(np.sum(psi_per_bin))

    logger.info("PSI = %.4f  (%s)", psi, interpret_psi(psi))
    return psi


def interpret_psi(psi: float) -> str:
    """Return a human-readable interpretation of a PSI value.

    Args:
        psi: PSI value from ``calculate_psi()``.

    Returns:
        str: Status string — 'Stable', 'Slight Shift', or 'Major Shift'.
    """
    if psi < PSI_STABLE:
        return "Stable"
    elif psi < PSI_WARNING:
        return "Slight Shift — investigate feature drift"
    else:
        return "Major Shift — consider model retraining"


# ---------------------------------------------------------------------------
# PSI by feature (feature-level drift detection)
# ---------------------------------------------------------------------------

def calculate_feature_psi(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    numeric_cols: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute PSI for each numeric feature to pinpoint which features drifted.

    Args:
        expected_df:  Reference feature DataFrame (training time).
        actual_df:    Current feature DataFrame (production).
        numeric_cols: List of numeric column names to evaluate.
        n_bins:       Number of quantile-based bins per feature.

    Returns:
        pd.DataFrame: Columns ``feature``, ``psi``, ``status``, sorted by PSI desc.
    """
    rows = []
    for col in numeric_cols:
        if col not in expected_df.columns or col not in actual_df.columns:
            continue
        exp = expected_df[col].dropna().values
        act = actual_df[col].dropna().values
        if len(exp) == 0 or len(act) == 0:
            continue

        col_min = min(exp.min(), act.min())
        col_max = max(exp.max(), act.max())
        bins = np.linspace(col_min, col_max, n_bins + 1)
        epsilon = 1e-6

        exp_counts, _ = np.histogram(exp, bins=bins)
        act_counts, _ = np.histogram(act, bins=bins)
        exp_pct = exp_counts / len(exp) + epsilon
        act_pct = act_counts / len(act) + epsilon
        feature_psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

        rows.append({
            "feature": col,
            "psi": round(feature_psi, 4),
            "status": interpret_psi(feature_psi),
        })

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_psi_summary(
    expected_scores: np.ndarray,
    actual_scores: np.ndarray,
    n_bins: int = 10,
    score_min: int = 300,
    score_max: int = 850,
    save: bool = True,
) -> plt.Figure:
    """Plot expected vs actual score distributions with PSI annotation.

    Args:
        expected_scores: Reference score array.
        actual_scores:   Current population score array.
        n_bins:          Number of bins for histogram.
        score_min:       Minimum score axis value.
        score_max:       Maximum score axis value.
        save:            If True, write PNG to assets/screenshots/.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    psi = calculate_psi(expected_scores, actual_scores, n_bins, score_min, score_max)
    bins = np.linspace(score_min, score_max, n_bins + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(expected_scores, bins=bins, alpha=0.6, color="#2196F3",
            label="Expected (training)", density=True)
    ax.hist(actual_scores, bins=bins, alpha=0.6, color="#FF9800",
            label="Actual (monitoring)", density=True)

    status = interpret_psi(psi)
    color = "#4CAF50" if psi < PSI_STABLE else ("#FF9800" if psi < PSI_WARNING else "#F44336")
    ax.set_title(f"Score Distribution Stability  |  PSI={psi:.4f}  ({status})",
                 fontsize=12, fontweight="bold", color=color)
    ax.set_xlabel("Credit Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = SCREENSHOTS_DIR / "psi_summary.png"
        fig.savefig(path, dpi=150)
        logger.info("PSI plot saved to %s", path)

    return fig
