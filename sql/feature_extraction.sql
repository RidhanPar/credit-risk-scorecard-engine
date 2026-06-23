-- Feature Extraction Query
-- Computes engineered features from raw applicants table using CTEs.
-- Debt-to-income proxy and ever-late flag are domain-specific derived variables
-- that signal credit risk beyond the raw attributes.

WITH base_features AS (
    SELECT
        id,
        checking_status,
        duration,
        credit_history,
        purpose,
        credit_amount,
        savings_status,
        employment,
        installment_commitment,
        personal_status,
        other_parties,
        residence_since,
        property_magnitude,
        age,
        other_payment_plans,
        housing,
        existing_credits,
        job,
        num_dependents,
        own_telephone,
        foreign_worker,
        default_flag
    FROM applicants
),

engineered_features AS (
    SELECT
        *,
        -- Debt-to-income proxy: monthly credit burden relative to credit duration
        -- Higher values signal over-leverage
        ROUND(
            CAST(credit_amount AS REAL) / (NULLIF(duration, 0) * NULLIF(installment_commitment, 0)),
            2
        ) AS debt_to_income_proxy,

        -- Ever-late flag: 1 if credit history shows any past delinquency
        CASE
            WHEN credit_history IN ('delayed previously', 'critical/other existing credit')
            THEN 1
            ELSE 0
        END AS ever_late_flag,

        -- Checking account risk flag: no or negative balance is a strong default predictor
        CASE
            WHEN checking_status IN ('<0', 'no checking') THEN 1
            ELSE 0
        END AS poor_checking_flag,

        -- Age-duration interaction: older applicants with shorter loans are lower risk
        ROUND(CAST(age AS REAL) / NULLIF(duration, 0), 2) AS age_per_month_credit

    FROM base_features
)

SELECT * FROM engineered_features;
