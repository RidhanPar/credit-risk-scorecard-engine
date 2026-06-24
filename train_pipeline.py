"""End-to-end credit risk model training pipeline.

Run this script once to:
  1. Download and store the German Credit dataset in SQLite
  2. Fit WoE binning process and select features by IV
  3. Train logistic regression scorecard (with PDO points scaling)
  4. Train XGBoost comparison model
  5. Evaluate both models (Gini, KS, ROC-AUC) and save comparison plots

Usage:
    python train_pipeline.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure src is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import run_data_pipeline
from src.evaluation import (
    compute_all_metrics,
    plot_roc_curve,
    plot_score_distribution,
    plot_ks_curve,
)
from src.feature_engineering import fit_binning_process, transform_woe
from src.monitoring import calculate_psi, plot_psi_summary
from src.scorecard import predict_scores, train_scorecard
from src.xgb_model import compute_shap_analysis, get_feature_importance, predict_proba_xgb, train_xgb_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_EXCLUDE_COLS = {"id", "default_flag", "created_at"}


def main() -> None:
    logger.info("=" * 60)
    logger.info("Credit Risk Scorecard Engine — Training Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data ingestion
    # ------------------------------------------------------------------
    logger.info("Step 1/6: Data ingestion + SQLite feature store")
    df = run_data_pipeline()
    logger.info("Feature DataFrame shape: %s", df.shape)
    logger.info("Default rate: %.1f%%", df["default_flag"].mean() * 100)

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    logger.info("Step 2/6: Train/test split (80/20, stratified)")
    feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
    X = df[feature_cols]
    y = df["default_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # ------------------------------------------------------------------
    # 3. WoE feature engineering
    # ------------------------------------------------------------------
    logger.info("Step 3/6: WoE feature engineering (optbinning)")
    train_with_target = X_train.copy()
    train_with_target["default_flag"] = y_train.values

    bp, selected_features, iv_df = fit_binning_process(train_with_target)

    logger.info("\n  Top 15 features by Information Value (IV):")
    logger.info("\n%s", iv_df.head(15).to_string(index=False))

    X_train_woe = transform_woe(X_train, bp, selected_features)
    X_test_woe = transform_woe(X_test, bp, selected_features)

    # ------------------------------------------------------------------
    # 4. Train logistic regression scorecard
    # ------------------------------------------------------------------
    logger.info("Step 4/6: Training logistic regression scorecard")
    lr_model, factor, offset = train_scorecard(X_train_woe, y_train)

    # Predictions on test set
    test_scores = predict_scores(X_test_woe)
    scorecard_proba = lr_model.predict_proba(X_test_woe[selected_features])[:, 1]

    sc_metrics = compute_all_metrics(y_test.values, scorecard_proba, "Logistic Scorecard")

    # ------------------------------------------------------------------
    # 5. Train XGBoost comparison model
    # ------------------------------------------------------------------
    logger.info("Step 5/6: Training XGBoost comparison model")
    xgb_model, X_train_enc, X_test_enc = train_xgb_model(X_train, X_test, y_train)
    xgb_proba = predict_proba_xgb(X_test)
    xgb_metrics = compute_all_metrics(y_test.values, xgb_proba, "XGBoost")

    # ------------------------------------------------------------------
    # 5b. SHAP interpretability analysis for XGBoost
    # ------------------------------------------------------------------
    logger.info("Step 5b: SHAP analysis (XGBoost)")
    shap_df = compute_shap_analysis(xgb_model, X_test_enc, output_dir="output")
    logger.info("\nTop 5 features by mean |SHAP|:\n%s", shap_df.head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Evaluation plots + PSI demo
    # ------------------------------------------------------------------
    logger.info("Step 6/6: Generating evaluation plots")

    plot_roc_curve(y_test.values, scorecard_proba, xgb_proba)
    plot_score_distribution(test_scores, y_test.values, "Logistic Scorecard")
    plot_ks_curve(y_test.values, scorecard_proba, "Logistic Scorecard")

    # PSI demo: simulate a slight distribution shift on 20% of test scores
    rng = np.random.default_rng(42)
    simulated_production_scores = test_scores + rng.integers(-30, 30, size=len(test_scores))
    simulated_production_scores = np.clip(simulated_production_scores, 300, 850)
    psi_val = calculate_psi(test_scores, simulated_production_scores)
    plot_psi_summary(test_scores, simulated_production_scores)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 60)
    results = pd.DataFrame({
        "Model": ["Logistic Scorecard", "XGBoost"],
        "ROC-AUC": [sc_metrics["roc_auc"], xgb_metrics["roc_auc"]],
        "Gini":    [sc_metrics["gini"],    xgb_metrics["gini"]],
        "KS":      [sc_metrics["ks"],      xgb_metrics["ks"]],
    })
    logger.info("\n%s", results.to_string(index=False))

    fi = get_feature_importance()
    logger.info("\nTop 10 XGBoost features by gain:\n%s", fi.head(10).to_string(index=False))

    logger.info("\nIV Table (selected features):\n%s",
                iv_df[iv_df["feature"].isin(selected_features)].to_string(index=False))

    logger.info("\nPSI (simulated production shift): %.4f", psi_val)
    logger.info("\nPlots saved to assets/screenshots/ and output/")
    logger.info("Models saved to models/")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
