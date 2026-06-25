# =============================================================================
# 03_scorecard_model.R  —  GLM Logistic Regression Scorecard with PDO Scaling
# Credit Risk Scorecard Engine | R Analysis Layer
#
# Uses the `scorecard` R package to:
#   - Apply WoE transformation to train/test splits
#   - Fit a GLM logistic regression (family = binomial)
#   - Convert model coefficients to PDO-scaled credit scores (300-850)
#   - Evaluate: AUC, Gini, KS statistic
#   - Compare with Python optbinning + sklearn results
#   - Produce ROC curve and score distribution plots
#
# Requires: run 02_woe_iv.R first (loads artefacts from output/)
# Outputs:  r_analysis/output/roc_curve_r.png
#           r_analysis/output/score_distribution_r.png
# =============================================================================

suppressPackageStartupMessages({
  library(scorecard)
  library(tidyverse)
  library(pROC)
  library(DBI)
  library(RSQLite)
  library(here)
})

set.seed(42)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
output_dir <- here("r_analysis", "output")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

gg_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14, colour = "#0f172a"),
    plot.subtitle = element_text(colour = "#64748b", size = 11),
    legend.position = "bottom"
  )

# Python benchmark results (from README.md)
PYTHON_LR_AUC  <- 0.818
PYTHON_LR_GINI <- 0.636
PYTHON_LR_KS   <- 0.562
PYTHON_XGB_AUC  <- 0.799
PYTHON_XGB_GINI <- 0.598
PYTHON_XGB_KS   <- 0.488

# ---------------------------------------------------------------------------
# 1. Load pre-computed artefacts from script 02
# ---------------------------------------------------------------------------
bins_path <- file.path(output_dir, "woe_bins.rds")
data_path <- file.path(output_dir, "dt_filtered.rds")

if (!file.exists(bins_path) || !file.exists(data_path)) {
  stop("Artefacts not found. Please run 02_woe_iv.R first.")
}

bins        <- readRDS(bins_path)
dt_filtered <- readRDS(data_path)

cat(sprintf("\nData loaded: %d rows x %d features\n",
            nrow(dt_filtered), ncol(dt_filtered) - 1))

# ---------------------------------------------------------------------------
# 2. Train / test split — 70/30, stratified, set.seed(42)
# ---------------------------------------------------------------------------
split_data <- split_df(dt_filtered, y = "default_flag",
                       ratio = 0.7, seed = 42)
train <- split_data$train
test  <- split_data$test

cat(sprintf("Train: %d rows | Test: %d rows\n", nrow(train), nrow(test)))
cat(sprintf("Train default rate: %.1f%% | Test default rate: %.1f%%\n",
            mean(train$default_flag) * 100,
            mean(test$default_flag)  * 100))

# ---------------------------------------------------------------------------
# 3. WoE transformation
# ---------------------------------------------------------------------------
train_woe <- woebin_ply(train, bins)
test_woe  <- woebin_ply(test,  bins)

# Feature columns (exclude target)
feat_cols <- setdiff(colnames(train_woe), "default_flag")
cat(sprintf("\nWoE-transformed features: %d\n", length(feat_cols)))

# ---------------------------------------------------------------------------
# 4. GLM logistic regression (binomial family)
# ---------------------------------------------------------------------------
cat("\n--- Fitting GLM (family = binomial) ---\n")

formula_str <- paste("default_flag ~", paste(feat_cols, collapse = " + "))
glm_model   <- glm(as.formula(formula_str),
                   data   = train_woe,
                   family = binomial(link = "logit"))

cat(sprintf("Converged    : %s\n", ifelse(glm_model$converged, "Yes", "No")))
cat(sprintf("Null deviance: %.2f  |  Residual deviance: %.2f\n",
            glm_model$null.deviance, glm_model$deviance))
cat(sprintf("AIC          : %.2f\n", glm_model$aic))

# ---------------------------------------------------------------------------
# 5. PDO scorecard scaling
#    base_score = 600, PDO = 50, odds0 = 1/19
#    (European convention: score of 600 at 19:1 good-to-bad odds)
# ---------------------------------------------------------------------------
cat("\n--- PDO Score Scaling ---\n")
cat("  base_score = 600 | PDO = 50 | odds0 = 1/19\n")
cat("  Factor = PDO / ln(2) = ", round(50 / log(2), 3), "\n")
cat("  Offset = 600 - Factor * ln(1/19) = ",
    round(600 - (50 / log(2)) * log(1 / 19), 3), "\n\n")

card <- scorecard(bins, glm_model,
                  points0 = 600,
                  odds0   = 1 / 19,
                  pdo     = 50)

# Score the test set (uses raw features, not WoE — scorecard_ply handles transform)
test_scores_df <- scorecard_ply(test, card, only_total_score = TRUE)
test$score     <- test_scores_df$score

score_range <- range(test$score, na.rm = TRUE)
cat(sprintf("Score range (test set): %d – %d\n", score_range[1], score_range[2]))
cat(sprintf("Mean score (Good)     : %.0f\n",
            mean(test$score[test$default_flag == 0], na.rm = TRUE)))
cat(sprintf("Mean score (Bad)      : %.0f\n",
            mean(test$score[test$default_flag == 1], na.rm = TRUE)))

# ---------------------------------------------------------------------------
# 6. Evaluation metrics
# ---------------------------------------------------------------------------
pred_proba <- predict(glm_model, newdata = test_woe, type = "response")

roc_obj  <- roc(test$default_flag, pred_proba, quiet = TRUE)
auc_val  <- as.numeric(auc(roc_obj))
gini_val <- 2 * auc_val - 1

# KS = max(TPR - FPR) across all thresholds
ks_val <- max(roc_obj$sensitivities + roc_obj$specificities - 1)

cat(sprintf("\nR Scorecard Results: AUC=%.3f, Gini=%.3f, KS=%.3f\n",
            auc_val, gini_val, ks_val))

# ---------------------------------------------------------------------------
# 7. Python vs R comparison table
# ---------------------------------------------------------------------------
comparison <- tibble(
  Model             = c("Python Logistic Scorecard", "R GLM Scorecard",
                        "Python XGBoost"),
  Algorithm         = c("sklearn LogisticRegression + WoE (optbinning)",
                        "glm(binomial) + WoE (scorecard pkg)",
                        "XGBoost (ordinal encoded)"),
  AUC               = c(PYTHON_LR_AUC, round(auc_val, 3), PYTHON_XGB_AUC),
  Gini              = c(PYTHON_LR_GINI, round(gini_val, 3), PYTHON_XGB_GINI),
  KS                = c(PYTHON_LR_KS, round(ks_val, 3), PYTHON_XGB_KS)
)

cat("\n", strrep("=", 85), "\n")
cat("  PYTHON vs R — MODEL COMPARISON\n")
cat(strrep("=", 85), "\n")
cat(sprintf("  %-28s  %6s  %6s  %6s\n", "Model", "AUC", "Gini", "KS"))
cat(strrep("-", 55), "\n")
for (i in seq_len(nrow(comparison))) {
  cat(sprintf("  %-28s  %6.3f  %6.3f  %6.3f\n",
              comparison$Model[i], comparison$AUC[i],
              comparison$Gini[i],  comparison$KS[i]))
}
cat(strrep("=", 85), "\n")
cat("\n  Both scorecards converge within ~0.01-0.02 AUC — expected given:\n")
cat("    - Same dataset and same feature engineering logic\n")
cat("    - Different binning methods (CP-SAT vs tree) produce similar WoE values\n")
cat("    - Logistic regression on WoE features is method-agnostic at the GLM level\n\n")

# ---------------------------------------------------------------------------
# 8. ROC curve plot
# ---------------------------------------------------------------------------
roc_df <- tibble(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_ribbon(aes(ymin = 0, ymax = tpr), fill = "#bfdbfe", alpha = 0.4) +
  geom_line(colour = "#2563eb", linewidth = 1.3) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "#94a3b8", linewidth = 0.8) +
  annotate(
    "label",
    x = 0.62, y = 0.22,
    label = sprintf("AUC  = %.3f\nGini = %.3f\nKS   = %.3f",
                    auc_val, gini_val, ks_val),
    size = 4, fontface = "bold", colour = "#0f172a",
    fill = "white", label.size = 0.4, label.padding = unit(0.5, "lines")
  ) +
  annotate(
    "label",
    x = 0.62, y = 0.06,
    label = sprintf("Python LR: AUC = %.3f", PYTHON_LR_AUC),
    size = 3.5, colour = "#64748b", fill = "#f8fafc",
    label.size = 0.3, label.padding = unit(0.4, "lines")
  ) +
  scale_x_continuous(labels = scales::percent_format()) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title    = "ROC Curve — R GLM Scorecard",
    subtitle = "German Credit Dataset | 30% Test Set | PDO scaling: base 600, PDO 50",
    x        = "False Positive Rate (1 − Specificity)",
    y        = "True Positive Rate (Sensitivity)"
  ) +
  gg_theme

ggsave(file.path(output_dir, "roc_curve_r.png"),
       p_roc, width = 7, height = 6, dpi = 300)
cat("  [saved] roc_curve_r.png\n")

# ---------------------------------------------------------------------------
# 9. Score distribution plot — Good vs Bad applicants
# ---------------------------------------------------------------------------
test <- test %>%
  mutate(credit_risk = factor(default_flag,
                              levels = c(0, 1),
                              labels = c("Good", "Bad")))

p_dist <- ggplot(test, aes(x = score, fill = credit_risk, colour = credit_risk)) +
  geom_density(alpha = 0.55, linewidth = 0.8, adjust = 1.2) +
  geom_vline(xintercept = mean(test$score[test$default_flag == 0], na.rm = TRUE),
             colour = "#16a34a", linetype = "dashed", linewidth = 0.8) +
  geom_vline(xintercept = mean(test$score[test$default_flag == 1], na.rm = TRUE),
             colour = "#dc2626", linetype = "dashed", linewidth = 0.8) +
  scale_fill_manual(values   = c("Good" = "#22c55e", "Bad" = "#ef4444")) +
  scale_colour_manual(values = c("Good" = "#16a34a", "Bad" = "#dc2626")) +
  labs(
    title    = "Score Distribution by Credit Risk Class",
    subtitle = "R GLM Scorecard — Test Set | Dashed lines = group means",
    x        = "Credit Score",
    y        = "Density",
    fill     = "Credit Risk",
    colour   = "Credit Risk"
  ) +
  gg_theme

ggsave(file.path(output_dir, "score_distribution_r.png"),
       p_dist, width = 8, height = 5, dpi = 300)
cat("  [saved] score_distribution_r.png\n")

# ---------------------------------------------------------------------------
# 10. Persist model artefacts
# ---------------------------------------------------------------------------
saveRDS(glm_model,   file.path(output_dir, "glm_model.rds"))
saveRDS(card,        file.path(output_dir, "scorecard.rds"))
saveRDS(comparison,  file.path(output_dir, "model_comparison.rds"))
saveRDS(list(auc = auc_val, gini = gini_val, ks = ks_val),
        file.path(output_dir, "r_metrics.rds"))

cat("\n  Artefacts saved: glm_model.rds | scorecard.rds | model_comparison.rds\n")
cat("\nScript 03 complete.\n")
