import pandas as pd
import joblib

from features import create_features


model = joblib.load(
    "models/churn_model.pkl"
)


def predict_customer(customer_data):

    df = pd.DataFrame(
        [customer_data]
    )

    df = create_features(df)

    probability = model.predict_proba(df)[0][1]

    prediction = int(
        probability >= 0.5
    )

    if probability >= 0.8:
        risk = "CRITICAL"

    elif probability >= 0.6:
        risk = "HIGH"

    elif probability >= 0.4:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    return {
        "churn_probability": round(
            float(probability),
            4
        ),
        "prediction": prediction,
        "risk": risk
    }


customer = {
    "age": 35,
    "tenure_months": 5,
    "monthly_charges": 90,
    "login_frequency": 3,
    "support_tickets": 6,
    "payment_failures": 2,
    "usage_hours": 10,
    "contract_type": "Monthly",
    "subscription_type": "Basic"
}


result = predict_customer(customer)

print(result)