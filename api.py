from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CustomerPulse AI API",
    description="Customer churn prediction API",
    version="1.0.0"
)


# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL LOCATIONS
# ============================================================

MODEL_FILES = [
    BASE_DIR / "model.pkl",
    BASE_DIR / "churn_model.pkl",
    BASE_DIR / "customer_churn_model.pkl",
    BASE_DIR / "models" / "model.pkl",
    BASE_DIR / "models" / "churn_model.pkl"
]


# ============================================================
# LOAD MODEL
# ============================================================

model = None
model_path = None


for path in MODEL_FILES:

    if path.exists():

        try:

            model = joblib.load(path)
            model_path = path

            print(
                f"Model loaded successfully: {path}"
            )

            break

        except Exception as e:

            print(
                f"Could not load model {path}: {e}"
            )


if model is None:

    print(
        "WARNING: No ML model was found."
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class CustomerData(BaseModel):

    age: int

    tenure_months: int

    monthly_charges: float

    login_frequency: int

    support_tickets: int

    payment_failures: int

    usage_hours: float

    contract_type: str

    subscription_type: str


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_engineered_features(df):

    df = df.copy()


    # --------------------------------------------------------
    # Usage per login
    # --------------------------------------------------------

    df["usage_per_login"] = (
        df["usage_hours"]
        / df["login_frequency"].replace(0, 1)
    )


    # --------------------------------------------------------
    # Payment risk
    # --------------------------------------------------------

    df["payment_risk"] = (
        df["payment_failures"]
    )


    # --------------------------------------------------------
    # Support intensity
    # --------------------------------------------------------

    df["support_intensity"] = (
        df["support_tickets"]
        / df["tenure_months"].replace(0, 1)
    )


    # --------------------------------------------------------
    # Customer value
    # --------------------------------------------------------

    df["customer_value"] = (
        df["monthly_charges"]
        * df["tenure_months"]
    )


    return df


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def get_risk(probability):

    if probability >= 0.80:

        return "CRITICAL"

    elif probability >= 0.60:

        return "HIGH"

    elif probability >= 0.30:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# RECOMMENDATIONS
# ============================================================

def get_recommendations(
    probability,
    payment_failures,
    support_tickets,
    login_frequency,
    contract_type
):

    recommendations = []


    if probability >= 0.80:

        recommendations.append(
            "Immediately contact the customer with a personalized retention offer."
        )

    elif probability >= 0.60:

        recommendations.append(
            "Prioritize this customer for proactive retention outreach."
        )

    elif probability >= 0.30:

        recommendations.append(
            "Monitor customer engagement and consider proactive communication."
        )

    else:

        recommendations.append(
            "Customer appears relatively stable. Continue normal engagement."
        )


    if payment_failures > 0:

        recommendations.append(
            "Investigate payment failures and offer payment assistance."
        )


    if support_tickets >= 5:

        recommendations.append(
            "Review unresolved support issues and improve customer support."
        )


    if login_frequency <= 3:

        recommendations.append(
            "Consider a re-engagement campaign because of low login frequency."
        )


    if contract_type == "Monthly":

        recommendations.append(
            "Consider offering a longer-term plan or loyalty incentive."
        )


    return recommendations


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "service": "CustomerPulse AI",
        "model_loaded": model is not None,
        "model": str(model_path) if model_path else None
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(model_path)
        if model_path
        else None
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(customer: CustomerData):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "ML model was not found. "
                "Place model.pkl in the project directory."
            )
        )


    try:

        # ----------------------------------------------------
        # Convert request to dictionary
        # ----------------------------------------------------

        customer_dict = customer.model_dump()


        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [customer_dict]
        )


        # ----------------------------------------------------
        # CREATE ENGINEERED FEATURES
        # ----------------------------------------------------

        input_df = create_engineered_features(
            input_df
        )


        # ----------------------------------------------------
        # SHOW DATA FOR DEBUGGING
        # ----------------------------------------------------

        print(
            "\nPrediction input:"
        )

        print(
            input_df.to_string(
                index=False
            )
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )[0]


        # ----------------------------------------------------
        # CHURN PROBABILITY
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_df
            )[0]


            # -----------------------------------------------
            # Find class 1
            # -----------------------------------------------

            if hasattr(
                model,
                "classes_"
            ):

                classes = list(
                    model.classes_
                )

                if 1 in classes:

                    churn_index = classes.index(
                        1
                    )

                    churn_probability = float(
                        probabilities[churn_index]
                    )

                else:

                    churn_probability = float(
                        probabilities[-1]
                    )

            else:

                if len(probabilities) > 1:

                    churn_probability = float(
                        probabilities[1]
                    )

                else:

                    churn_probability = float(
                        probabilities[0]
                    )

        else:

            churn_probability = float(
                prediction
            )


        # ----------------------------------------------------
        # Keep probability between 0 and 1
        # ----------------------------------------------------

        churn_probability = max(
            0.0,
            min(
                1.0,
                churn_probability
            )
        )


        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        risk = get_risk(
            churn_probability
        )


        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        recommendations = get_recommendations(
            churn_probability,
            customer.payment_failures,
            customer.support_tickets,
            customer.login_frequency,
            customer.contract_type
        )


        # ----------------------------------------------------
        # Engineered features
        # ----------------------------------------------------

        engineered_features = {

            "customer_value": float(
                input_df[
                    "customer_value"
                ].iloc[0]
            ),

            "usage_per_login": float(
                input_df[
                    "usage_per_login"
                ].iloc[0]
            ),

            "payment_risk": float(
                input_df[
                    "payment_risk"
                ].iloc[0]
            ),

            "support_intensity": float(
                input_df[
                    "support_intensity"
                ].iloc[0]
            )
        }


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {

            "prediction": int(
                prediction
            ),

            "churn_probability": churn_probability,

            "risk": risk,

            "recommendations": recommendations,

            "engineered_features": engineered_features
        }


    except Exception as e:

        print(
            "\nPrediction error:"
        )

        print(
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
