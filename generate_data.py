import numpy as np
import pandas as pd

np.random.seed(42)

N = 10000

data = pd.DataFrame({
    "customer_id": range(10001, 10001 + N),

    "age": np.random.randint(18, 70, N),

    "tenure_months": np.random.randint(1, 72, N),

    "monthly_charges": np.round(
        np.random.uniform(20, 150, N), 2
    ),

    "login_frequency": np.random.randint(1, 31, N),

    "support_tickets": np.random.poisson(2, N),

    "payment_failures": np.random.poisson(0.5, N),

    "usage_hours": np.round(
        np.random.uniform(1, 100, N), 2
    ),

    "contract_type": np.random.choice(
        ["Monthly", "Yearly", "Two Year"],
        N,
        p=[0.5, 0.3, 0.2]
    ),

    "subscription_type": np.random.choice(
        ["Basic", "Standard", "Premium"],
        N
    )
})


# Create churn probability
risk_score = (
    0.03 * data["support_tickets"]
    + 0.08 * data["payment_failures"]
    - 0.02 * data["tenure_months"]
    - 0.03 * data["login_frequency"]
    - 0.01 * data["usage_hours"]
)

risk_probability = 1 / (1 + np.exp(-risk_score))

data["churn"] = (
    np.random.random(N) < risk_probability
).astype(int)


# Save
data.to_csv(
    "data/customers.csv",
    index=False
)

print("Dataset created successfully!")
print(data.head())
print("\nShape:", data.shape)
print("\nChurn distribution:")
print(data["churn"].value_counts())