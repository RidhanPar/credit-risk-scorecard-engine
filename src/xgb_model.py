"""XGBoost comparison model for credit risk.

Trains on raw (non-WoE) features to provide a benchmark against the interpretable
logistic scorecard. Categorical features are ordinally encoded so XGBoost can
consume them without manual WoE transformation.

When to choose each model:
  - Scorecard: regulatory submissions, credit bureau integrations, adverse action
    letters — any context where a points breakdown must be explained to the applicant.
  - XGBoost: internal risk monitoring, early-warning systems, situations where
    predictive lift is worth the interpretability trade-off.
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = os.getenv(
    "XGB_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "xgb_model.pkl"),
)

_EXCLUDE_COLS = {"id", "default_flag", "created_at"}


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _split_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return lists of numeric and categorical feature columns.

    Args:
        df: Raw feature DataFrame.

    Returns:
        tuple[list[str], list[str]]: ``(numeric_cols, categorical_cols)``.
    """
    feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def encode_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """Ordinal-encode categorical columns; handle unseen values as -1.

    Args:
        X_train:          Training feature DataFrame.
        X_test:           Test feature DataFrame.
        categorical_cols: Columns to encode.

    Returns:
        tuple:
            - Encoded training DataFrame.
            - Encoded test DataFrame.
            - Fitted ``OrdinalEncoder``.
    """
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[categorical_cols] = enc.fit_transform(X_train[categorical_cols].astype(str))
    X_test[categorical_cols] = enc.transform(X_test[categorical_cols].astype(str))

    return X_train, X_test, enc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgb_model(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: pd.Series,
    model_path: str = DEFAULT_MODEL_PATH,
) -> tuple[XGBClassifier, pd.DataFrame, pd.DataFrame]:
    """Train XGBoost on raw (non-WoE) features and persist the artefact.

    Args:
        X_train_raw: Raw training features (categorical columns as strings).
        X_test_raw:  Raw test features.
        y_train:     Binary training target (1 = default).
        model_path:  Path to save the pickled artefact.

    Returns:
        tuple:
            - Fitted ``XGBClassifier``.
            - Encoded training DataFrame.
            - Encoded test DataFrame.
    """
    numeric_cols, categorical_cols = _split_features(X_train_raw)
    logger.info(
        "Training XGBoost: %d numeric, %d categorical features",
        len(numeric_cols), len(categorical_cols),
    )

    X_train_enc, X_test_enc, encoder = encode_categoricals(
        X_train_raw, X_test_raw, categorical_cols
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train_enc, y_train,
        eval_set=[(X_test_enc, None)] if False else None,  # no eval logging
        verbose=False,
    )

    artefact = {
        "model": model,
        "encoder": encoder,
        "feature_names": list(X_train_enc.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, model_path)
    logger.info("XGBoost artefact saved to %s", model_path)

    return model, X_train_enc, X_test_enc


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_proba_xgb(
    X_raw: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
) -> np.ndarray:
    """Return default probabilities for raw feature input.

    Args:
        X_raw:      Raw feature DataFrame (same schema as training input).
        model_path: Path to the pickled XGBoost artefact.

    Returns:
        np.ndarray: Probability of default (class 1), shape (n_samples,).
    """
    artefact = joblib.load(model_path)
    model: XGBClassifier = artefact["model"]
    encoder: OrdinalEncoder = artefact["encoder"]
    categorical_cols: list[str] = artefact["categorical_cols"]
    feature_names: list[str] = artefact["feature_names"]

    X = X_raw.copy()
    X[categorical_cols] = encoder.transform(X[categorical_cols].astype(str))
    return model.predict_proba(X[feature_names])[:, 1]


def get_feature_importance(model_path: str = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    """Return XGBoost feature importances ranked by gain.

    Args:
        model_path: Path to the XGBoost artefact.

    Returns:
        pd.DataFrame: Columns ``feature`` and ``importance`` sorted descending.
    """
    artefact = joblib.load(model_path)
    model: XGBClassifier = artefact["model"]
    feature_names: list[str] = artefact["feature_names"]

    importance = model.get_booster().get_score(importance_type="gain")
    rows = [
        {"feature": feature_names[int(k.lstrip("f"))], "importance": v}
        if k.startswith("f") and k[1:].isdigit()
        else {"feature": k, "importance": v}
        for k, v in importance.items()
    ]
    df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    return df
