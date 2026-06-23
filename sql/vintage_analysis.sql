-- Vintage Analysis: Rolling Default Rate by Credit Duration Bucket
-- Uses LAG() window function to compare each bucket's default rate
-- against the prior bucket, revealing how credit term length affects risk.

WITH duration_buckets AS (
    SELECT
        id,
        duration,
        default_flag,
        CASE
            WHEN duration <= 12  THEN '01_0-12m'
            WHEN duration <= 24  THEN '02_13-24m'
            WHEN duration <= 36  THEN '03_25-36m'
            WHEN duration <= 48  THEN '04_37-48m'
            ELSE                      '05_49m+'
        END AS duration_bucket
    FROM applicants
),

bucket_stats AS (
    SELECT
        duration_bucket,
        COUNT(*)                                                              AS total_applications,
        SUM(default_flag)                                                     AS total_defaults,
        ROUND(100.0 * SUM(default_flag) / COUNT(*), 2)                       AS default_rate_pct,
        ROUND(AVG(CAST(duration AS REAL)), 1)                                AS avg_duration_months
    FROM duration_buckets
    GROUP BY duration_bucket
),

vintage_with_lag AS (
    SELECT
        duration_bucket,
        total_applications,
        total_defaults,
        default_rate_pct,
        avg_duration_months,
        LAG(default_rate_pct) OVER (ORDER BY duration_bucket)                AS prev_bucket_default_rate,
        ROUND(
            default_rate_pct
            - LAG(default_rate_pct) OVER (ORDER BY duration_bucket),
            2
        )                                                                     AS default_rate_change_pct,
        SUM(total_applications) OVER ()                                       AS grand_total_applications,
        ROUND(100.0 * total_applications / SUM(total_applications) OVER (), 1) AS pct_of_portfolio
    FROM bucket_stats
)

SELECT
    duration_bucket,
    total_applications,
    total_defaults,
    default_rate_pct,
    prev_bucket_default_rate,
    default_rate_change_pct,
    avg_duration_months,
    pct_of_portfolio
FROM vintage_with_lag
ORDER BY duration_bucket;
