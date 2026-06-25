# =============================================================================
# 01_eda.R  —  Exploratory Data Analysis
# Credit Risk Scorecard Engine | R Analysis Layer
#
# Loads the German Credit dataset from the shared SQLite feature store,
# produces publication-quality EDA plots using ggplot2, and prints a
# structured summary of the dataset.
#
# Outputs saved to: r_analysis/output/
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(corrplot)
  library(scales)
  library(skimr)
  library(DBI)
  library(RSQLite)
  library(here)
})

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
output_dir <- here("r_analysis", "output")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

gg_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14, color = "#0f172a"),
    plot.subtitle = element_text(color = "#64748b", size = 11),
    plot.caption  = element_text(color = "#94a3b8", size = 9),
    strip.text    = element_text(face = "bold", size = 10),
    legend.position = "bottom"
  )

# ---------------------------------------------------------------------------
# 1. Load data from shared SQLite feature store
# ---------------------------------------------------------------------------
con <- dbConnect(SQLite(), here("data", "credit_risk.db"))
sql <- paste(readLines(here("sql", "feature_extraction.sql")), collapse = "\n")
df  <- dbGetQuery(con, sql)
dbDisconnect(con)

# Remove row-identifier; ensure target is integer
df <- df %>%
  select(-id) %>%
  mutate(default_flag = as.integer(default_flag))

# Readable target factor for plots
df <- df %>%
  mutate(credit_risk = factor(default_flag,
                              levels = c(0, 1),
                              labels = c("Good", "Bad")))

# ---------------------------------------------------------------------------
# 2. Dataset summary
# ---------------------------------------------------------------------------
cat("\n", strrep("=", 60), "\n")
cat("  GERMAN CREDIT DATASET — SUMMARY\n")
cat(strrep("=", 60), "\n\n")

cat(sprintf("  Rows            : %d\n", nrow(df)))
cat(sprintf("  Features        : %d\n", ncol(df) - 2))   # excl. default_flag + credit_risk
cat(sprintf("  Bad (default)   : %d (%.1f%%)\n",
            sum(df$default_flag), mean(df$default_flag) * 100))
cat(sprintf("  Good            : %d (%.1f%%)\n",
            sum(df$default_flag == 0), mean(df$default_flag == 0) * 100))

cat("\n  Missing values per column:\n")
na_counts <- df %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") %>%
  filter(n_missing > 0)

if (nrow(na_counts) == 0) {
  cat("  -> No missing values detected.\n")
} else {
  print(na_counts, n = Inf)
}

cat("\n  Variable types:\n")
df %>%
  select(-credit_risk) %>%
  summarise(across(everything(), class)) %>%
  pivot_longer(everything(), names_to = "column", values_to = "type") %>%
  count(type) %>%
  { cat(sprintf("    %s: %d columns\n", .$type, .$n)) }

# ---------------------------------------------------------------------------
# 3. Target variable distribution
# ---------------------------------------------------------------------------
target_counts <- df %>%
  count(credit_risk) %>%
  mutate(pct = n / sum(n),
         label = sprintf("%d\n(%.1f%%)", n, pct * 100))

p_target <- ggplot(target_counts, aes(x = credit_risk, y = n, fill = credit_risk)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = label), vjust = -0.3, fontface = "bold", size = 5) +
  scale_fill_manual(values = c("Good" = "#22c55e", "Bad" = "#ef4444")) +
  scale_y_continuous(limits = c(0, 800), expand = expansion(mult = c(0, 0.05))) +
  labs(
    title    = "Target Variable Distribution",
    subtitle = "German Credit Dataset — 1,000 applicants",
    x        = "Credit Risk",
    y        = "Count",
    caption  = "Source: UCI German Credit (sklearn fetch_openml)"
  ) +
  gg_theme +
  theme(panel.grid.major.x = element_blank())

ggsave(file.path(output_dir, "01_target_distribution.png"),
       p_target, width = 6, height = 5, dpi = 300)
cat("\n  [saved] 01_target_distribution.png\n")

# ---------------------------------------------------------------------------
# 4. Numeric feature distributions faceted by credit risk
# ---------------------------------------------------------------------------
numeric_cols <- df %>%
  select(where(is.numeric), -default_flag) %>%
  colnames()

p_hist <- df %>%
  select(all_of(numeric_cols), credit_risk) %>%
  pivot_longer(-credit_risk, names_to = "feature", values_to = "value") %>%
  ggplot(aes(x = value, fill = credit_risk)) +
  geom_histogram(bins = 25, alpha = 0.75, position = "identity", colour = NA) +
  facet_wrap(~feature, scales = "free", ncol = 4) +
  scale_fill_manual(values = c("Good" = "#22c55e", "Bad" = "#ef4444")) +
  labs(
    title    = "Numeric Feature Distributions by Credit Risk",
    subtitle = "Overlaid histograms — Good (green) vs Bad (red) applicants",
    x        = NULL,
    y        = "Count",
    fill     = "Credit Risk"
  ) +
  gg_theme +
  theme(axis.text.x = element_text(size = 7),
        axis.text.y = element_text(size = 7))

ggsave(file.path(output_dir, "02_feature_distributions.png"),
       p_hist, width = 14, height = 11, dpi = 300)
cat("  [saved] 02_feature_distributions.png\n")

# ---------------------------------------------------------------------------
# 5. Correlation matrix (numeric features + target)
# ---------------------------------------------------------------------------
cor_df  <- df %>% select(all_of(numeric_cols), default_flag)
cor_mat <- cor(cor_df, use = "complete.obs")

png(file.path(output_dir, "03_correlation_matrix.png"),
    width = 2600, height = 2200, res = 300)
corrplot(
  cor_mat,
  method      = "color",
  order       = "hclust",
  type        = "upper",
  tl.cex      = 0.85,
  tl.col      = "#1e293b",
  addCoef.col = "#374151",
  number.cex  = 0.6,
  col         = colorRampPalette(c("#ef4444", "white", "#22c55e"))(200),
  title       = "Feature Correlation Matrix (hclust order)",
  mar         = c(0, 0, 2, 0),
  cl.cex      = 0.75
)
dev.off()
cat("  [saved] 03_correlation_matrix.png\n")

# ---------------------------------------------------------------------------
# 6. Missing value completeness heatmap
# ---------------------------------------------------------------------------
sample_rows <- min(150, nrow(df))

na_long <- df %>%
  slice_head(n = sample_rows) %>%
  mutate(row_id = row_number()) %>%
  pivot_longer(-row_id, names_to = "column", values_to = "value") %>%
  mutate(is_missing = is.na(value))

p_na <- ggplot(na_long, aes(x = column, y = row_id, fill = is_missing)) +
  geom_tile(colour = "white", linewidth = 0.08) +
  scale_fill_manual(values = c("FALSE" = "#f0fdf4", "TRUE" = "#ef4444"),
                    labels = c("Present", "Missing")) +
  scale_y_reverse() +
  labs(
    title    = sprintf("Data Completeness Map (first %d rows)", sample_rows),
    subtitle = "Red = missing value | Green = present",
    x        = NULL,
    y        = "Row index",
    fill     = NULL,
    caption  = "No missing values expected in this dataset"
  ) +
  gg_theme +
  theme(axis.text.x = element_text(angle = 50, hjust = 1, size = 7.5),
        legend.position = "top")

ggsave(file.path(output_dir, "04_missing_value_heatmap.png"),
       p_na, width = 16, height = 6, dpi = 300)
cat("  [saved] 04_missing_value_heatmap.png\n")

# ---------------------------------------------------------------------------
# 7. Default rate by key categorical features
# ---------------------------------------------------------------------------
cat_cols <- c("checking_status", "credit_history", "savings_status",
              "employment", "purpose")

default_by_cat <- df %>%
  select(all_of(cat_cols), default_flag) %>%
  pivot_longer(-default_flag, names_to = "feature", values_to = "category") %>%
  group_by(feature, category) %>%
  summarise(default_rate = mean(default_flag), n = n(), .groups = "drop")

p_cat <- ggplot(default_by_cat,
                aes(x = reorder(category, default_rate), y = default_rate,
                    fill = default_rate)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = percent(default_rate, accuracy = 1)),
            hjust = -0.15, size = 2.8, fontface = "bold", colour = "#374151") +
  coord_flip() +
  facet_wrap(~feature, scales = "free_y", ncol = 2) +
  scale_fill_gradient(low = "#bbf7d0", high = "#dc2626") +
  scale_y_continuous(labels = percent_format(), limits = c(0, 0.85)) +
  labs(
    title    = "Default Rate by Key Categorical Features",
    subtitle = "Features ranked by default rate within each category",
    x        = NULL,
    y        = "Default Rate"
  ) +
  gg_theme

ggsave(file.path(output_dir, "05_default_rate_by_category.png"),
       p_cat, width = 13, height = 10, dpi = 300)
cat("  [saved] 05_default_rate_by_category.png\n")

# ---------------------------------------------------------------------------
# 8. Skimr summary
# ---------------------------------------------------------------------------
cat("\n", strrep("=", 60), "\n")
cat("  SKIMR VARIABLE SUMMARY\n")
cat(strrep("=", 60), "\n\n")
print(skim(df %>% select(-credit_risk, -default_flag)))

cat("\n", strrep("=", 60), "\n")
cat("  EDA COMPLETE — plots saved to:", output_dir, "\n")
cat(strrep("=", 60), "\n\n")
