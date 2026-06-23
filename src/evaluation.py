"""Model evaluation functions for credit risk scorecards.

Implements the domain-standard metrics used in consumer lending:
  - ROC-AUC: overall discriminatory power
  - Gini coefficient: 2 * AUC - 1  (industry shorthand for scorecard quality)
  - KS statistic: maximum separation between good and bad score distributions

Gini > 0.5 is considered acceptable; > 0.6 is strong for consumer credit.
KS > 0.3 is the typical minimum for a scorecard to be used in production.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "assets" / "screenshots"


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the ROC-AUC score.

    Args:
        y_true: Ground-truth binary labels (1 = default).
        y_prob: Predicted default probabilities.

    Returns:
        float: ROC-AUC in [0, 1].
    """
    return float(roc_auc_score(y_true, y_prob))


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Gini coefficient (normalised AUC).

    Gini = 2 * AUC - 1.  A random model gives Gini = 0; perfect = 1.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted default probabilities.

    Returns:
        float: Gini coefficient in [0, 1].
    """
    auc = roc_auc(y_true, y_prob)
    return float(2 * auc - 1)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov (KS) statistic.

    KS = max(TPR - FPR) across all thresholds.  Measures the maximum separation
    between the cumulative distributions of good and bad applicants.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted default probabilities.

    Returns:
        float: KS statistic in [0, 1].
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """Compute and log AUC, Gini, and KS for a model.

    Args:
        y_true:     Ground-truth binary labels.
        y_prob:     Predicted default probabilities.
        model_name: Label used in logging output.

    Returns:
        dict: Keys ``roc_auc``, ``gini``, ``ks``.
    """
    metrics = {
        "roc_auc": roc_auc(y_true, y_prob),
        "gini": gini_coefficient(y_true, y_prob),
        "ks": ks_statistic(y_true, y_prob),
    }
    logger.info(
        "[%s] ROC-AUC=%.4f  Gini=%.4f  KS=%.4f",
        model_name, metrics["roc_auc"], metrics["gini"], metrics["ks"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob_scorecard: np.ndarray,
    y_prob_xgb: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Plot overlaid ROC curves for scorecard vs XGBoost.

    Args:
        y_true:            Ground-truth binary labels.
        y_prob_scorecard:  Scorecard default probabilities.
        y_prob_xgb:        XGBoost default probabilities.
        save:              If True, save PNG to assets/screenshots/.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for probs, label, color in [
        (y_prob_scorecard, "Logistic Scorecard", "#2196F3"),
        (y_prob_xgb, "XGBoost", "#FF5722"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Scorecard vs XGBoost", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = SCREENSHOTS_DIR / "roc_curve.png"
        fig.savefig(path, dpi=150)
        logger.info("ROC curve saved to %s", path)

    return fig


def plot_score_distribution(
    scores: np.ndarray,
    y_true: np.ndarray,
    model_name: str = "Logistic Scorecard",
    save: bool = True,
) -> plt.Figure:
    """Plot overlapping score distributions for good vs bad applicants.

    Args:
        scores:     Credit scores (or probabilities) for each applicant.
        y_true:     Binary target (1 = bad/default).
        model_name: Used in the plot title.
        save:       If True, save PNG to assets/screenshots/.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    good_scores = scores[y_true == 0]
    bad_scores = scores[y_true == 1]

    ax.hist(good_scores, bins=30, alpha=0.65, color="#4CAF50", label="Good (non-default)", density=True)
    ax.hist(bad_scores, bins=30, alpha=0.65, color="#F44336", label="Bad (default)", density=True)
    ax.axvline(np.mean(good_scores), color="#4CAF50", linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(bad_scores), color="#F44336", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Score Distribution — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        fname = model_name.lower().replace(" ", "_").replace("/", "_")
        path = SCREENSHOTS_DIR / f"score_distribution_{fname}.png"
        fig.savefig(path, dpi=150)
        logger.info("Score distribution saved to %s", path)

    return fig


def plot_ks_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """Plot the KS separation curve (cumulative good vs bad rate).

    Args:
        y_true:     Ground-truth binary labels.
        y_prob:     Predicted default probabilities.
        model_name: Label used in the title.
        save:       If True, save PNG to assets/screenshots/.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    sorted_idx = np.argsort(y_prob)[::-1]
    y_sorted = y_true[sorted_idx]
    n = len(y_sorted)
    cum_bad = np.cumsum(y_sorted) / y_sorted.sum()
    cum_good = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    pct_pop = np.arange(1, n + 1) / n

    ks = float(np.max(np.abs(cum_bad - cum_good)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pct_pop, cum_bad, color="#F44336", lw=2, label="Cumulative Bad Rate")
    ax.plot(pct_pop, cum_good, color="#4CAF50", lw=2, label="Cumulative Good Rate")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Population %", fontsize=12)
    ax.set_ylabel("Cumulative Rate", fontsize=12)
    ax.set_title(f"KS Curve — {model_name}  (KS={ks:.3f})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        fname = model_name.lower().replace(" ", "_")
        path = SCREENSHOTS_DIR / f"ks_curve_{fname}.png"
        fig.savefig(path, dpi=150)
        logger.info("KS curve saved to %s", path)

    return fig
