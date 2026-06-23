"""Logistic regression credit scorecard with classic points scaling.

Implements the industry-standard scorecard methodology:
  - Train LogisticRegression on WoE-encoded features
  - Scale to a credit score using Points-to-Double-Odds (PDO) methodology
  - Base score = 600 at base odds 1:1, PDO = 20

Scaling formulae (Siddiqi, 2006):
    Factor = PDO / ln(2)
    Offset = BaseScore - Factor * ln(BaseOdds)
    Score  = Offset + Factor * ln(P_good / P_bad)
           = Offset - Factor * log_odds   (where log_odds = ln(P_bad/P_good))

Each feature's partial points contribution:
    points_i = -Factor * coef_i * WoE_i
"""

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = os.getenv(
    "SCORECARD_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "scorecard_model.pkl"),
)

# Scorecard scaling constants
PDO: int = int(os.getenv("PDO", "20"))
BASE_SCORE: int = int(os.getenv("BASE_SCORE", "600"))
BASE_ODDS: float = float(os.getenv("BASE_ODDS", "1.0"))

SCORE_MIN = 300
SCORE_MAX = 850


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------

def _compute_scaling_factors(
    pdo: int = PDO,
    base_score: int = BASE_SCORE,
    base_odds: float = BASE_ODDS,
) -> tuple[float, float]:
    """Compute Factor (B) and Offset (A) for credit score scaling.

    Args:
        pdo:        Points to Double the Odds.
        base_score: Credit score at the base odds.
        base_odds:  Reference odds at which ``base_score`` is assigned.

    Returns:
        tuple[float, float]: ``(factor, offset)``.
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def log_odds_to_score(
    log_odds: float,
    factor: float,
    offset: float,
) -> int:
    """Convert a log-odds value to a clamped credit score (300–850).

    Args:
        log_odds: ln(P_bad / P_good) from the logistic regression.
        factor:   Scaling factor B.
        offset:   Scaling offset A.

    Returns:
        int: Credit score clipped to [SCORE_MIN, SCORE_MAX].
    """
    score = offset - factor * log_odds
    return int(np.clip(round(score), SCORE_MIN, SCORE_MAX))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_scorecard(
    X_woe: pd.DataFrame,
    y: pd.Series,
    model_path: str = DEFAULT_MODEL_PATH,
    C: float = 1.0,
) -> tuple[LogisticRegression, float, float]:
    """Train a logistic regression scorecard on WoE features.

    Args:
        X_woe:      WoE-encoded feature matrix (training set).
        y:          Binary target (1 = default, 0 = non-default).
        model_path: Path to persist the fitted model artefact.
        C:          Inverse regularisation strength.

    Returns:
        tuple:
            - Fitted ``LogisticRegression``.
            - Scaling factor (B).
            - Scaling offset (A).
    """
    logger.info("Training LogisticRegression on %d samples, %d features ...",
                len(X_woe), X_woe.shape[1])

    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_woe, y)

    factor, offset = _compute_scaling_factors()
    logger.info(
        "Scorecard scaling: factor=%.4f, offset=%.4f (PDO=%d, base=%d, base_odds=%.1f)",
        factor, offset, PDO, BASE_SCORE, BASE_ODDS,
    )

    artefact = {
        "model": model,
        "feature_names": list(X_woe.columns),
        "factor": factor,
        "offset": offset,
        "pdo": PDO,
        "base_score": BASE_SCORE,
        "base_odds": BASE_ODDS,
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, model_path)
    logger.info("Scorecard artefact saved to %s", model_path)

    return model, factor, offset


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_scores(
    X_woe: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
) -> np.ndarray:
    """Compute credit scores for a batch of WoE-encoded applicants.

    Args:
        X_woe:      WoE-encoded feature matrix.
        model_path: Path to the saved scorecard artefact.

    Returns:
        np.ndarray: Integer credit scores in [300, 850], shape (n_samples,).
    """
    artefact = joblib.load(model_path)
    model: LogisticRegression = artefact["model"]
    factor: float = artefact["factor"]
    offset: float = artefact["offset"]
    feature_names: list[str] = artefact["feature_names"]

    X = X_woe[feature_names]
    # log_odds = ln(P_bad/P_good) = -(model log-odds for the positive class)
    log_odds = model.decision_function(X)  # positive = more likely bad
    scores = np.array([log_odds_to_score(lo, factor, offset) for lo in log_odds])
    return scores


def score_applicant(
    features: dict,
    bp,
    model_path: str = DEFAULT_MODEL_PATH,
) -> int:
    """Score a single applicant and return their credit score.

    Args:
        features:   Dict of raw feature name → value (pre-WoE, as collected
                    from the application form or API request).
        bp:         Fitted ``BinningProcess`` for WoE transformation.
        model_path: Path to the scorecard artefact.

    Returns:
        int: Credit score in range [300, 850].
    """
    artefact = joblib.load(model_path)
    model: LogisticRegression = artefact["model"]
    factor: float = artefact["factor"]
    offset: float = artefact["offset"]
    feature_names: list[str] = artefact["feature_names"]

    row_df = pd.DataFrame([features])

    # Ensure all feature columns are present for the binning process
    all_bp_features = bp.variable_names
    for col in all_bp_features:
        if col not in row_df.columns:
            row_df[col] = np.nan

    woe_matrix = bp.transform(row_df[all_bp_features], metric="woe")
    woe_df = pd.DataFrame(woe_matrix, columns=all_bp_features)
    X = woe_df[feature_names]

    log_odds = float(model.decision_function(X)[0])
    return log_odds_to_score(log_odds, factor, offset)


def get_score_tier(score: int) -> str:
    """Map a numeric credit score to a human-readable risk tier.

    Args:
        score: Credit score in [300, 850].

    Returns:
        str: One of 'Low Risk', 'Medium Risk', 'High Risk', 'Declined'.
    """
    if score >= 700:
        return "Low Risk"
    elif score >= 600:
        return "Medium Risk"
    elif score >= 500:
        return "High Risk"
    else:
        return "Declined"


def load_scorecard_artefact(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Load the full scorecard artefact dictionary from disk.

    Args:
        model_path: Path to the pickled artefact.

    Returns:
        dict: Keys include ``model``, ``feature_names``, ``factor``, ``offset``.
    """
    return joblib.load(model_path)
