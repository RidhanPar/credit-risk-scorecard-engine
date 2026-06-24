"""Weight of Evidence (WoE) feature engineering using optbinning.

Uses ``optbinning.BinningProcess`` — the industry-standard library for credit
scorecard feature engineering — to compute WoE transformations and Information
Value (IV) for all candidate features.

Features with IV < IV_THRESHOLD are dropped before modelling.
IV interpretation (Siddiqi, 2006):
    < 0.02  — useless
    0.02–0.1 — weak
    0.1–0.3  — medium
    0.3–0.5  — strong
    > 0.5   — suspicious (possible target leakage)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from optbinning import BinningProcess

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINNING_PATH = os.getenv(
    "BINNING_PROCESS_PATH",
    str(PROJECT_ROOT / "models" / "binning_process.pkl"),
)

IV_THRESHOLD = 0.02

# Columns that are identifiers or the target — never used as features
_EXCLUDE_COLS = {"id", "default_flag", "created_at"}

# Categorical columns in the German Credit dataset
_CATEGORICAL_COLS = {
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker",
    # Derived binary flags are treated as categorical for binning
    "ever_late_flag", "poor_checking_flag",
}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_binning_process(
    df: pd.DataFrame,
    binning_path: str = DEFAULT_BINNING_PATH,
    iv_threshold: float = IV_THRESHOLD,
) -> tuple[BinningProcess, list[str], pd.DataFrame]:
    """Fit a BinningProcess on training data and select features by IV.

    Args:
        df:            Training DataFrame including ``default_flag`` target.
        binning_path:  Path to persist the fitted BinningProcess.
        iv_threshold:  Minimum IV to retain a feature (default 0.02).

    Returns:
        tuple:
            - Fitted ``BinningProcess`` object.
            - List of selected feature names (IV >= threshold).
            - IV summary DataFrame sorted by IV descending.
    """
    target = df["default_flag"].values

    feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
    categorical_vars = [
        col for col in feature_cols
        if col in _CATEGORICAL_COLS or df[col].dtype == object
    ]

    logger.info("Fitting BinningProcess on %d features ...", len(feature_cols))
    bp = BinningProcess(
        variable_names=feature_cols,
        categorical_variables=categorical_vars,
        max_n_bins=10,
        min_bin_size=0.05,
    )
    bp.fit(df[feature_cols], target)

    # Build IV summary table
    iv_df = _build_iv_table(bp, feature_cols)

    # Select features above threshold
    selected = iv_df.loc[iv_df["iv"] >= iv_threshold, "feature"].tolist()
    logger.info(
        "Features selected (IV >= %.2f): %d / %d",
        iv_threshold, len(selected), len(feature_cols),
    )

    # Persist binning process
    Path(binning_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bp, binning_path)
    logger.info("BinningProcess saved to %s", binning_path)

    return bp, selected, iv_df


def _build_iv_table(bp: BinningProcess, feature_cols: list[str]) -> pd.DataFrame:
    """Extract IV values from a fitted BinningProcess.

    Args:
        bp:           Fitted ``BinningProcess``.
        feature_cols: List of variable names passed to the process.

    Returns:
        pd.DataFrame: Columns ``feature`` and ``iv``, sorted descending by IV.
    """
    rows = []
    summary = bp.summary()
    for _, row in summary.iterrows():
        rows.append({"feature": row["name"], "iv": round(float(row["iv"]), 4)})
    iv_df = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return iv_df


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------

def transform_woe(
    df: pd.DataFrame,
    bp: BinningProcess,
    selected_features: list[str],
) -> pd.DataFrame:
    """Apply WoE transformation to selected features.

    Args:
        df:                Input DataFrame (may be train or test split).
        bp:                Fitted ``BinningProcess``.
        selected_features: Feature columns to transform and return.

    Returns:
        pd.DataFrame: WoE-encoded features (same row count as ``df``).
    """
    all_features = [c for c in df.columns if c not in _EXCLUDE_COLS]
    woe_df = pd.DataFrame(
        np.asarray(bp.transform(df[all_features], metric="woe")),
        columns=all_features,
        index=df.index,
    )
    return woe_df[selected_features]


# ---------------------------------------------------------------------------
# Load persisted process
# ---------------------------------------------------------------------------

def load_binning_process(binning_path: str = DEFAULT_BINNING_PATH) -> BinningProcess:
    """Load a previously fitted BinningProcess from disk.

    Args:
        binning_path: Path to the pickled BinningProcess.

    Returns:
        BinningProcess: Fitted binning process ready for inference.
    """
    return joblib.load(binning_path)


# ---------------------------------------------------------------------------
# Convenience: WoE contribution for a single applicant (used by Streamlit)
# ---------------------------------------------------------------------------

def get_woe_contributions(
    applicant_woe: pd.Series,
    coefficients: dict[str, float],
) -> pd.Series:
    """Compute each feature's contribution to the log-odds (WoE * coefficient).

    Args:
        applicant_woe: WoE-encoded feature values for one applicant.
        coefficients:  Mapping of feature name → logistic regression coefficient.

    Returns:
        pd.Series: Contribution scores, sorted by absolute value descending.
    """
    contributions = {
        feat: applicant_woe[feat] * coef
        for feat, coef in coefficients.items()
        if feat in applicant_woe.index
    }
    series = pd.Series(contributions)
    return series.reindex(series.abs().sort_values(ascending=False).index)
