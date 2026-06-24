# Credit Risk Scorecard Engine

**Production-quality credit scorecard pipeline** — WoE feature engineering, logistic regression with PDO scaling, XGBoost comparison, PSI monitoring, and an interactive Streamlit scoring UI. Deployable in one command via Docker.

---

## Problem Statement

Consumer lenders approve or decline thousands of loan applications daily. A credit scorecard translates an applicant's attributes into a single, interpretable score (typically 300–850) that represents their probability of default. Unlike black-box models, scorecards must satisfy regulatory requirements: every decision must be explainable to the applicant and auditable by regulators.

This project implements the full scorecard lifecycle as practised at consumer fintech companies:

1. **Feature store** — raw data ingested into SQLite with SQL-based feature extraction
2. **WoE encoding** — Weight of Evidence transformation that linearises features for logistic regression
3. **Scorecard scaling** — PDO (Points to Double the Odds) methodology converts log-odds to a business-readable score
4. **Model monitoring** — Population Stability Index (PSI) detects when the deployed model needs retraining
5. **Interactive UI** — Streamlit app for real-time scoring with feature contribution explanations

---

## Architecture

```mermaid
flowchart LR
    A[German Credit Dataset\nsklearn / UCI] --> B[SQLite Feature Store\ncreate_tables.sql]
    B --> C[SQL Feature Extraction\nCTEs + Window Functions]
    C --> D[WoE Encoding\noptbinning.BinningProcess]
    D --> E1[Logistic Scorecard\nPDO scaling 300-850]
    C --> E2[XGBoost\nOrdinal encoded]
    E1 --> F[Evaluation\nGini · KS · ROC-AUC]
    E2 --> F
    E1 --> G[PSI Monitoring\nscore distribution drift]
    E1 --> H[Streamlit UI\nReal-time scoring]
```

---

## Tech Stack

| Component | Library | Version | Why |
|-----------|---------|---------|-----|
| Core language | Python | 3.11 | |
| Data manipulation | pandas / numpy | 2.2.2 / 1.26.4 | |
| WoE / IV engineering | **optbinning** | 0.19.0 | Industry standard for credit scorecard feature engineering |
| Scorecard model | scikit-learn LogisticRegression | 1.5.0 | Interpretable; supports PDO scaling |
| Comparison model | xgboost | 2.1.1 | State-of-the-art on tabular data |
| Feature store | SQLite (sqlite3) | stdlib | Demonstrates SQL proficiency with CTEs and window functions |
| Visualisation | matplotlib / seaborn | 3.9.0 / 0.13.2 | |
| Scoring UI | Streamlit | 1.36.0 | Interactive recruiter demo |
| Containerisation | Docker / Compose | — | One-command deployment |
| Testing | pytest | 8.2.2 | |
| CI | GitHub Actions | — | Runs tests on every push |

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/RidhanPar/credit-risk-scorecard-engine.git
cd credit-risk-scorecard-engine
cp .env.example .env
docker-compose up
```

Open **http://localhost:8501** in your browser.  The container downloads the dataset, trains all models, and launches the UI automatically.

### Local install

```bash
git clone https://github.com/RidhanPar/credit-risk-scorecard-engine.git
cd credit-risk-scorecard-engine
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train_pipeline.py          # trains models, saves artefacts, generates plots
streamlit run app/streamlit_app.py
```

### Run tests

```bash
pytest tests/ -q
```

---

## Screenshots

### EDA — Target Distribution
![Target distribution](assets/screenshots/eda_target_distribution.png)

### EDA — Bivariate Analysis (Checking Status vs Default Rate)
![Bivariate analysis](assets/screenshots/eda_bivariate.png)

### ROC Curve — Scorecard vs XGBoost
![ROC curve](assets/screenshots/roc_curve.png)

### Score Distribution — Good vs Bad Applicants
![Score distribution](assets/screenshots/score_distribution_logistic_scorecard.png)

### Streamlit Scoring UI
![Streamlit UI](assets/screenshots/streamlit_ui.png)

---

## Model Results

> 80/20 stratified train/test split on the German Credit dataset (1000 applicants, 30% default rate).

| Model | ROC-AUC | Gini | KS Statistic |
|-------|---------|------|-------------|
| **Logistic Scorecard** | **0.818** | **0.636** | **0.562** |
| XGBoost | 0.799 | 0.598 | 0.488 |

**The logistic scorecard outperforms XGBoost on this dataset** — a result worth explaining in interviews: WoE encoding pre-linearises all features optimally for logistic regression, removing the need for a tree to discover non-linear splits. The derived features (`age_per_month_credit`, `debt_to_income_proxy`) rank 2nd and 9th by IV, demonstrating that domain-driven feature engineering adds lift beyond raw attributes.

**Which model to choose:**

- **Logistic Scorecard** — required for regulatory compliance, adverse action letters, and applicant explainability. The points breakdown is directly auditable. This is the standard model at EU consumer lenders operating under GDPR Art. 22 and ECOA.
- **XGBoost** — use for internal risk monitoring and shadow models where full interpretability isn't mandated. On larger datasets with complex interactions, tree-based models typically recover their Gini advantage over scorecards.

---

## Information Value (IV) Table — All 17 Selected Features

> IV measures each feature's predictive power for the default target. 17 of 24 candidate features passed the IV ≥ 0.02 threshold. Features below threshold were dropped before modelling.

| Feature | IV | Interpretation |
|---------|----|----------------|
| checking_status | 0.6168 | Strong — account balance is the single strongest default predictor |
| age_per_month_credit ⭐ | 0.3622 | Strong — derived feature: age relative to loan duration |
| duration | 0.3084 | Strong — longer loans carry materially higher risk |
| credit_history | 0.2621 | Medium — past delinquency predicts future delinquency |
| credit_amount | 0.2375 | Medium — over-leverage signal |
| savings_status | 0.2253 | Medium — liquid savings act as a buffer |
| purpose | 0.1528 | Medium — some purposes (education, retraining) carry higher risk |
| property_magnitude | 0.1466 | Medium — collateral quality |
| debt_to_income_proxy ⭐ | 0.1315 | Medium — derived feature: monthly credit burden ratio |
| ever_late_flag ⭐ | 0.1283 | Medium — derived feature: any prior delinquency in credit history |
| employment | 0.1253 | Medium |
| housing | 0.0881 | Weak |
| other_payment_plans | 0.0840 | Weak |
| age | 0.0809 | Weak |
| personal_status | 0.0576 | Weak |
| poor_checking_flag ⭐ | 0.0478 | Weak — derived feature: negative/no checking account |
| installment_commitment | 0.0324 | Weak |

⭐ = engineered feature derived in `sql/feature_extraction.sql` or computed at inference time

---

## Score-to-Risk-Tier Mapping

| Score Range | Risk Tier | Recommended Action |
|-------------|-----------|-------------------|
| 700 – 850 | Low Risk | Approve; offer best rate |
| 600 – 699 | Medium Risk | Approve with standard rate; consider collateral |
| 500 – 599 | High Risk | Approve only with guarantor or reduced amount |
| 300 – 499 | Declined | Decline; generate adverse action letter |

---

## PSI Monitoring Guide

The Population Stability Index (PSI) is calculated monthly by comparing the current applicant score distribution against the training-time baseline.

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | Stable | No action required |
| 0.10 – 0.25 | Slight shift | Investigate feature drift; review model quarterly |
| > 0.25 | Major shift | Model likely degraded; trigger retraining |

```python
from src.monitoring import calculate_psi, interpret_psi

psi = calculate_psi(training_scores, production_scores)
print(interpret_psi(psi))  # "Stable" / "Slight Shift" / "Major Shift"
```

Simulated PSI on a ±30-point shifted holdout: **PSI = 0.0042 — Stable**.

---

## Project Structure

```
credit-risk-scorecard-engine/
├── data/raw/                    # Raw dataset (gitignored)
├── sql/
│   ├── create_tables.sql        # SQLite schema
│   ├── feature_extraction.sql   # CTE-based derived features
│   └── vintage_analysis.sql     # LAG / PARTITION BY window analysis
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory data analysis
├── src/
│   ├── data_loader.py           # Dataset → SQLite → feature extraction
│   ├── feature_engineering.py   # WoE encoding, IV selection (optbinning)
│   ├── scorecard.py             # Logistic regression + PDO points scaling
│   ├── xgb_model.py             # XGBoost comparison model
│   ├── evaluation.py            # Gini, KS, ROC-AUC, plots
│   └── monitoring.py            # PSI calculation and interpretation
├── app/
│   └── streamlit_app.py         # Interactive scoring UI
├── models/                      # Trained artefacts (gitignored)
├── tests/                       # pytest unit tests
├── .github/workflows/ci.yml     # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Limitations

- **Dataset size:** The German Credit dataset contains only 1,000 applicants. Real consumer lending scorecards are built on 50,000–500,000+ observations.  Performance metrics should be interpreted accordingly.
- **Simulated monitoring:** PSI monitoring in `src/monitoring.py` is demonstrated on a synthetic holdout shift, not real production traffic.
- **No income data:** The German Credit dataset does not contain a verified income field, so debt-to-income is a proxy based on installment commitment percentage.
- **Point-in-time only:** There is no time dimension in this dataset, so vintage-based through-the-cycle calibration is approximated using credit duration buckets.

---

## Future Improvements

- [ ] Replace SQLite with PostgreSQL + SQLAlchemy for a scalable feature store
- [ ] Add Shapley value (SHAP) explanations alongside WoE contributions
- [ ] Implement automated model retraining trigger when PSI > 0.25
- [ ] Add through-the-cycle PD calibration using Platt scaling
- [ ] Scorecard champion/challenger A/B testing framework
- [ ] API endpoint (FastAPI) for real-time scoring integration with origination systems
