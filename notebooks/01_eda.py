import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("data/customers.csv")


# -----------------------------
# 1. Basic Information
# -----------------------------

print("=" * 50)
print("CUSTOMERPULSE AI - EDA")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# -----------------------------
# 2. First 5 Rows
# -----------------------------

print("\nFirst 5 Rows:")
print(df.head())


# -----------------------------
# 3. Data Types
# -----------------------------

print("\nData Types:")
print(df.dtypes)


# -----------------------------
# 4. Missing Values
# -----------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# 5. Duplicate Rows
# -----------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# -----------------------------
# 6. Statistical Summary
# -----------------------------

print("\nStatistical Summary:")
print(df.describe())


# -----------------------------
# 7. Churn Distribution
# -----------------------------

print("\nChurn Distribution:")
print(df["churn"].value_counts())

print("\nChurn Percentage:")
print(
    df["churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# -----------------------------
# 8. Churn Visualization
# -----------------------------

df["churn"].value_counts().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.show()


# -----------------------------
# 9. Churn by Subscription
# -----------------------------

print("\nChurn by Subscription Type:")

churn_by_subscription = (
    df.groupby("subscription_type")["churn"]
    .agg(["count", "sum", "mean"])
)

churn_by_subscription.columns = [
    "total_customers",
    "churned_customers",
    "churn_rate"
]

churn_by_subscription["churn_rate"] *= 100

print(
    churn_by_subscription
    .sort_values("churn_rate", ascending=False)
)


# -----------------------------
# 10. Churn by Contract
# -----------------------------

print("\nChurn by Contract Type:")

churn_by_contract = (
    df.groupby("contract_type")["churn"]
    .agg(["count", "sum", "mean"])
)

churn_by_contract.columns = [
    "total_customers",
    "churned_customers",
    "churn_rate"
]

churn_by_contract["churn_rate"] *= 100

print(
    churn_by_contract
    .sort_values("churn_rate", ascending=False)
)


print("\nEDA completed successfully!")