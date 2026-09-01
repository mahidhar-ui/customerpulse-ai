import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

from features import create_features


# Load data
df = pd.read_csv("data/customers.csv")

# Feature engineering
df = create_features(df)


# Separate features and target
X = df.drop(
    columns=["customer_id", "churn"]
)

y = df["churn"]


# Identify columns
categorical_columns = [
    "contract_type",
    "subscription_type"
]

numerical_columns = [
    column for column in X.columns
    if column not in categorical_columns
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)


# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train
pipeline.fit(X_train, y_train)


# Predictions
predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]


# Evaluation
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        probabilities
    )
)


# Save model
joblib.dump(
    pipeline,
    "models/churn_model.pkl"
)

print("\nModel saved successfully!")