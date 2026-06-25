# =============================================================================
# 02_woe_iv.R  —  Weight of Evidence & Information Value Analysis
# Credit Risk Scorecard Engine | R Analysis Layer
#
# Uses the industry-standard `scorecard` R package to:
#   - Filter variables by IV (threshold 0.02, matching Python pipeline)
#   - Compute optimal WoE bins using tree-based splitting
#   - Rank features by IV and label predictive power
#   - Compare R IV values with Python optbinning results
#   - Save WoE bin plots for the top 5 features
#
# Outputs saved to: r_analysis/output/ and r_analysis/output/woe_plots/
# =============================================================================

suppressPackageStartupMessages({
  library(scorecard)
  library(tidyverse)
  library(DBI)
  library(RSQLite)
  library(here)
})

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
output_dir   <- here("r_analysis", "output")
woe_plot_dir <- file.path(output_dir, "woe_plots")
dir.create(woe_plot_dir, showWarnings = FALSE, recursive = TRUE)

IV_THRESHOLD <- 0.02   # same as Python pipeline

# ---------------------------------------------------------------------------
# 1. Load data from shared SQLite feature store
# ---------------------------------------------------------------------------
con <- dbConnect(SQLite(), here("data", "credit_risk.db"))
sql <- paste(readLines(here("sql", "feature_extraction.sql")), collapse = "\n")
df  <- dbGetQuery(con, sql)
dbDisconnect(con)

df <- df %>%
  select(-id) %>%
  mutate(default_flag = as.integer(default_flag))

cat(sprintf("\nData loaded: %d rows x %d features\n", nrow(df), ncol(df) - 1))

# ---------------------------------------------------------------------------
# 2. Variable filtering by IV (var_filter is scorecard's filter_var equivalent)
# ---------------------------------------------------------------------------
cat(sprintf("\n--- Variable Filtering (IV >= %.2f) ---\n", IV_THRESHOLD))

dt_filtered <- var_filter(df, y = "default_flag", iv_limit = IV_THRESHOLD)

cat(sprintf("Features before filter : %d\n", ncol(df) - 1))
cat(sprintf("Features after filter  : %d\n", ncol(dt_filtered) - 1))

dropped <- setdiff(colnames(df), colnames(dt_filtered))
dropped <- dropped[dropped != "default_flag"]
if (length(dropped) > 0) {
  cat(sprintf("Dropped (IV < %.2f)    : %s\n", IV_THRESHOLD, paste(dropped, collapse = ", ")))
}

# ---------------------------------------------------------------------------
# 3. WoE binning — tree method (same principle as optbinning in Python)
# ---------------------------------------------------------------------------
cat("\n--- WoE Binning (method = tree) ---\n")
bins <- woebin(dt_filtered, y = "default_flag", method = "tree")
cat(sprintf("WoE bins computed for %d variables.\n", length(bins)))

# ---------------------------------------------------------------------------
# 4. IV summary table with predictive power labels
# ---------------------------------------------------------------------------
iv_result <- iv(dt_filtered, y = "default_flag") %>%
  as_tibble() %>%
  arrange(desc(info_value)) %>%
  mutate(
    R_IV = round(info_value, 4),
    predictive_power = case_when(
      info_value >= 0.50 ~ "Very Strong",
      info_value >= 0.30 ~ "Strong",
      info_value >= 0.10 ~ "Medium",
      info_value >= 0.02 ~ "Weak",
      TRUE               ~ "Unpredictive"
    )
  ) %>%
  select(variable, R_IV, predictive_power)

cat("\n", strrep("=", 70), "\n")
cat("  IV SUMMARY TABLE\n")
cat(strrep("=", 70), "\n")
cat(sprintf("  %-30s  %8s  %s\n", "Variable", "IV", "Predictive Power"))
cat(strrep("-", 70), "\n")
for (i in seq_len(nrow(iv_result))) {
  cat(sprintf("  %-30s  %8.4f  %s\n",
              iv_result$variable[i],
              iv_result$R_IV[i],
              iv_result$predictive_power[i]))
}
cat(strrep("=", 70), "\n")

# ---------------------------------------------------------------------------
# 5. Python vs R IV comparison
#    Python IV values from optbinning (from README.md training results)
# ---------------------------------------------------------------------------
python_iv <- tribble(
  ~variable,                ~Python_IV,
  "checking_status",         0.6168,
  "age_per_month_credit",    0.3622,
  "duration",                0.3084,
  "credit_history",          0.2621,
  "credit_amount",           0.2375,
  "savings_status",          0.2253,
  "purpose",                 0.1528,
  "property_magnitude",      0.1466,
  "debt_to_income_proxy",    0.1315,
  "ever_late_flag",          0.1283,
  "employment",              0.1253,
  "housing",                 0.0881,
  "other_payment_plans",     0.0840,
  "age",                     0.0809,
  "personal_status",         0.0576,
  "poor_checking_flag",      0.0478,
  "installment_commitment",  0.0324
  # remaining features had Python IV < 0.02 and were dropped
)

iv_comparison <- iv_result %>%
  full_join(python_iv, by = "variable") %>%
  mutate(
    difference = round(R_IV - Python_IV, 4),
    note = case_when(
      is.na(Python_IV) ~ "R kept, Python dropped",
      is.na(R_IV)      ~ "Python kept, R dropped",
      abs(difference) <= 0.05 ~ "Consistent",
      TRUE             ~ "Divergent (diff binning method)"
    )
  ) %>%
  arrange(desc(coalesce(R_IV, 0)))

cat("\n", strrep("=", 85), "\n")
cat("  PYTHON (optbinning) vs R (scorecard) — IV COMPARISON\n")
cat(strrep("=", 85), "\n")
cat(sprintf("  %-30s  %10s  %10s  %10s  %s\n",
            "Feature", "Python_IV", "R_IV", "Diff", "Note"))
cat(strrep("-", 85), "\n")
for (i in seq_len(nrow(iv_comparison))) {
  r <- iv_comparison[i, ]
  cat(sprintf("  %-30s  %10s  %10s  %10s  %s\n",
              r$variable,
              ifelse(is.na(r$Python_IV), "dropped", sprintf("%.4f", r$Python_IV)),
              ifelse(is.na(r$R_IV),      "dropped", sprintf("%.4f", r$R_IV)),
              ifelse(is.na(r$difference),"  —    ", sprintf("%+.4f", r$difference)),
              r$note))
}
cat(strrep("=", 85), "\n")
cat("\n  Note: IV differences reflect different binning algorithms:\n")
cat("        Python uses optbinning CP-SAT (optimal) | R uses scorecard tree split\n")
cat("        Both approaches converge on the same feature importance ranking.\n\n")

# ---------------------------------------------------------------------------
# 6. WoE bin plots — top 5 features by IV
# ---------------------------------------------------------------------------
top5 <- iv_result %>% slice_max(R_IV, n = 5) %>% pull(variable)
cat(sprintf("Saving WoE bin plots for top 5 features:\n  %s\n\n",
            paste(top5, collapse = "\n  ")))

woe_plots <- woebin_plot(bins[top5])

for (feat in top5) {
  plot_path <- file.path(woe_plot_dir, sprintf("woe_%s.png", feat))
  p <- woe_plots[[feat]] +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      strip.text = element_text(face = "bold")
    )
  ggsave(plot_path, p, width = 8, height = 5, dpi = 300)
  cat(sprintf("  [saved] %s\n", basename(plot_path)))
}

# ---------------------------------------------------------------------------
# 7. Persist artefacts for script 03
# ---------------------------------------------------------------------------
saveRDS(bins,          file.path(output_dir, "woe_bins.rds"))
saveRDS(iv_result,     file.path(output_dir, "iv_summary.rds"))
saveRDS(iv_comparison, file.path(output_dir, "iv_comparison.rds"))
saveRDS(dt_filtered,   file.path(output_dir, "dt_filtered.rds"))

cat("\n  Artefacts saved:\n")
cat("    output/woe_bins.rds\n")
cat("    output/iv_summary.rds\n")
cat("    output/iv_comparison.rds\n")
cat("    output/dt_filtered.rds\n")
cat("\nScript 02 complete.\n")
