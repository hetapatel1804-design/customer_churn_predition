-- =====================================================================
-- churn_analysis.sql
-- SQL analysis layer for the Customer Churn Prediction System.
--
-- Assumes two tables in a relational database:
--   customers            -- cleaned + feature-engineered customer records
--                            (built by src/data_preprocessing.py +
--                            src/feature_engineering.py, one row/customer)
--   churn_predictions    -- model output (built by src/predict.py), joined
--                            to customers on customerID
--
-- Written and tested against SQLite (data/processed/churn.db) using the
-- ACTUAL processed dataset — every query below has been run and returns
-- real rows/values, not illustrative placeholders.
-- =====================================================================


-- 1. Total customers
SELECT COUNT(*) AS total_customers
FROM customers;


-- 2. Total churned customers
SELECT COUNT(*) AS total_churned_customers
FROM customers
WHERE ChurnFlag = 1;


-- 3. Overall churn rate (%)
SELECT
    ROUND(100.0 * SUM(CASE WHEN ChurnFlag = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers;


-- 4. Churn by contract type
SELECT
    Contract,
    COUNT(*)                                              AS total_customers,
    SUM(CASE WHEN ChurnFlag = 1 THEN 1 ELSE 0 END)        AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN ChurnFlag = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- 5. Churn by payment method
SELECT
    PaymentMethod,
    COUNT(*)                                              AS total_customers,
    SUM(CASE WHEN ChurnFlag = 1 THEN 1 ELSE 0 END)        AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN ChurnFlag = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- 6. Churn by internet service
SELECT
    InternetService,
    COUNT(*)                                              AS total_customers,
    SUM(CASE WHEN ChurnFlag = 1 THEN 1 ELSE 0 END)        AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN ChurnFlag = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- 7. High-value customers currently at risk
--    (top tercile by TotalCharges, i.e. CustomerValueSegment = 'High',
--     AND model-flagged High Risk)
SELECT
    c.customerID,
    c.TotalCharges,
    c.MonthlyCharges,
    c.tenure,
    p.ChurnProbability,
    p.RiskSegment
FROM customers c
JOIN churn_predictions p ON c.customerID = p.customerID
WHERE c.CustomerValueSegment = 'High'
  AND p.RiskSegment = 'High Risk'
ORDER BY p.ChurnProbability DESC;


-- 8. High-risk customers (model-flagged)
SELECT
    p.customerID,
    p.ChurnProbability,
    p.RiskSegment,
    p.TopRiskFactors,
    p.SuggestedAction
FROM churn_predictions p
WHERE p.RiskSegment = 'High Risk'
ORDER BY p.ChurnProbability DESC;


-- 9. Average monthly charges by churn status
SELECT
    ChurnLabel,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(AVG(tenure), 1)         AS avg_tenure_months
FROM customers
GROUP BY ChurnLabel;


-- 10. Customers with high churn probability (threshold-parameterized view)
--     Uses a CTE + window function to rank each high-probability customer
--     by expected revenue at risk within their risk segment, then keeps
--     only the top 20 per segment (a window-function alias cannot be
--     filtered directly in the same SELECT, so the RANK() result is
--     wrapped and filtered in an outer query).
WITH high_prob_customers AS (
    SELECT
        p.customerID,
        p.ChurnProbability,
        p.RiskSegment,
        p.MonthlyRevenueAtRisk,
        RANK() OVER (
            PARTITION BY p.RiskSegment
            ORDER BY p.MonthlyRevenueAtRisk DESC
        ) AS revenue_risk_rank
    FROM churn_predictions p
    WHERE p.ChurnProbability >= 0.70
)
SELECT *
FROM high_prob_customers
WHERE revenue_risk_rank <= 20
ORDER BY RiskSegment, revenue_risk_rank;


-- ---------------------------------------------------------------------
-- BONUS: aggregate Revenue at Risk across the whole customer base
-- (used by the Streamlit "Executive Overview" page)
-- ---------------------------------------------------------------------
SELECT
    RiskSegment,
    COUNT(*)                              AS n_customers,
    ROUND(SUM(MonthlyRevenueAtRisk), 2)   AS total_monthly_revenue_at_risk,
    ROUND(AVG(ChurnProbability), 4)       AS avg_churn_probability
FROM churn_predictions
GROUP BY RiskSegment
ORDER BY total_monthly_revenue_at_risk DESC;
