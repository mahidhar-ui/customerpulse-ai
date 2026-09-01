SELECT COUNT(*) AS total_customers
FROM customers;


SELECT
    ROUND(
        100.0 * SUM(churn) / COUNT(*),
        2
    ) AS churn_rate
FROM customers;


SELECT
    subscription_type,
    COUNT(*) AS customers,
    SUM(churn) AS churned_customers,
    ROUND(
        100.0 * SUM(churn) / COUNT(*),
        2
    ) AS churn_rate
FROM customers
GROUP BY subscription_type
ORDER BY churn_rate DESC;


SELECT
    customer_id,
    monthly_charges,
    support_tickets,
    payment_failures
FROM customers
WHERE support_tickets >= 5
AND payment_failures >= 2
ORDER BY monthly_charges DESC;