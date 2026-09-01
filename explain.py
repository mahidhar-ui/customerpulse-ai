import pandas as pd
import joblib
import shap

from features import create_features


df = pd.read_csv(
    "data/customers.csv"
)

df = create_features(df)

X = df.drop(
    columns=["customer_id", "churn"]
)

model_pipeline = joblib.load(
    "models/churn_model.pkl"
)

preprocessor = (
    model_pipeline
    .named_steps["preprocessor"]
)

model = (
    model_pipeline
    .named_steps["model"]
)

X_transformed = preprocessor.transform(X)

feature_names = (
    preprocessor
    .get_feature_names_out()
)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_transformed
)

print("SHAP analysis completed.")

print(
    "Feature count:",
    len(feature_names)
)