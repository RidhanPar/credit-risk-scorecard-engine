# One-time package installation for the R credit risk analysis layer.
# Run this script once before executing 01_eda.R, 02_woe_iv.R, 03_scorecard_model.R
# or rendering credit_risk_report.Rmd.

packages <- c(
  # Credit scorecard: WoE binning, IV, PDO scaling
  "scorecard",
  # Data manipulation and visualisation
  "tidyverse",
  # ROC / AUC analysis
  "pROC",
  # Correlation matrix visualisation
  "corrplot",
  # Axis / label formatting helpers
  "scales",
  # Rich summary statistics
  "skimr",
  # R Markdown rendering
  "rmarkdown",
  # Table formatting in reports
  "knitr",
  # SQLite access (reads data/credit_risk.db shared with Python pipeline)
  "DBI",
  "RSQLite",
  # Project-relative file paths (resolves from .git root)
  "here"
)

# Only install packages that are not already present
to_install <- setdiff(packages, installed.packages()[, "Package"])

if (length(to_install) == 0) {
  message("All packages already installed.")
} else {
  message("Installing: ", paste(to_install, collapse = ", "))
  install.packages(to_install, repos = "https://cran.rstudio.com/")
  message("Installation complete.")
}
